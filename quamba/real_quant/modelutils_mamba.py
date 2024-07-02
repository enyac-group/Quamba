import os
import gc
import copy
import logging
from tqdm import tqdm
from functools import partial

import torch
from datasets import load_dataset

from mamba_ssm.modules.block import Block

from .qMambaLayer import QMamba, MambaSimple
from .qJamba import QJambaMambaMixer, QJambaRMSNorm
from .qHadamard import Hadamard
from .fusedNorm import FusedRMSNorm
from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver

logger = logging.getLogger(__name__)

def run_calibration(_model, model_type, tokenizer,
                    num_samples=512, seq_len=512,
                    use_had_transform=True,
                    calibration_dataset=None,
                    preprocess_fn=None):

    device = next(_model.parameters()).device

    if model_type == "jamba":
        model = _model # use reference. do not copy Jamba. it is too large
        layers = model.model.layers
        # ~/.cache/huggingface/modules/transformers_modules/Jamba-v0.1/modeling_jamba.py
        # isinstace does not work for transformers_modules
        is_traget_block = lambda block: "JambaMambaDecoderLayer" in str(type(block))
        get_mamba = lambda block: block.mamba
        is_calib_ops = lambda op: "JambaRMSNorm" in str(type(op)) or "Linear" in str(type(op))
        percentile_alpha=0.999
        if use_had_transform:
            # insert Hadamard transform
            for i in tqdm(range(len(layers))):
                if "JambaMambaDecoderLayer" in str(type(layers[i])):
                    layers[i].mamba.out_proj = torch.nn.Sequential(
                        Hadamard(layers[i].mamba.out_proj.in_features),
                        layers[i].mamba.out_proj
                    )
    elif model_type == "mamba":
        model = copy.deepcopy(_model) # we will use MabmaSimple for the replica
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, torch.nn.Linear)
        percentile_alpha=0.99999
        # use simplied mamba block to get the scaling factors
        # from linear layers without pain
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                m = MambaSimple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'jamba'")
    model.config.use_cache = False
    model.eval()

    # register min/max observers
    observers = [{} for _ in range(len(layers))]
    
    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        
        for name, m in get_mamba(layers[i]).named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0] # out_proj.1 Jamba with Hadamard
                if op == "x_proj":
                    observers[i][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )

    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
        calibration_dataset.shuffle(seed=42)

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    # collect in/output scaling factors for layers
    act_scales = [{} for _ in range(len(layers))]
    for i in range(len(layers)):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)


    if model_type == "jamba" and use_had_transform:
        layers = model.model.layers
        # ~/.cache/huggingface/modules/transformers_modules/Jamba-v0.1/modeling_jamba.py
        # isinstace does not work for transformers_modules
        for i in tqdm(range(len(layers))):
            # remove Hadamard transform
            if "JambaMambaDecoderLayer" in str(type(layers[i])):
                layers[i].mamba.out_proj = layers[i].mamba.out_proj[1]
    elif model_type == "mamba":
        del model  # delete the replica with MambaSimple blocks
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'jamba'")

    gc.collect() # garbage collection
    torch.cuda.empty_cache()  # clear cache and gpu memory

    return act_scales
    

def quantize_blocks(model, model_type, act_scales, device):
    model.config.use_cache = False
    if model_type == "jamba":
        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            if "JambaMambaDecoderLayer" in str(type(layers[i])):
                layers[i].input_layernorm = QJambaRMSNorm(
                    originalLayer=layers[i].input_layernorm,
                    output_scale=act_scales[i]["in_proj:input"].item())
                layers[i].mamba = QJambaMambaMixer(
                    originalLayer=layers[i].mamba,
                    act_scales=act_scales[i])
            # JambaAttentionDecoderLayer should already be quantized with BitsandBytes
            # garbage collection and clean cache
            gc.collect()
            torch.cuda.empty_cache()
    elif model_type == "mamba":
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                # replace with fused RMSNorm
                layers[i].fused_add_norm = True
                layers[i].norm = FusedRMSNorm(
                    originalLayer=layers[i].norm,
                    output_scale=act_scales[i]["in_proj:input"].item()
                ).to(device)

                # replace with QMamba, hardcode for 8-bit, see QMamba
                layers[i].mixer = QMamba(
                    originalLayer=layers[i].mixer,
                    act_scales=act_scales[i],
                ).to(device)

                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'jamba'")
    return model

def quantize_model_mamba(model, model_type, tokenizer, device, args):

    act_scales = {}
    # Run calibration to get scale
    if args.act_scales_cache and os.path.isfile(args.act_scales_cache):
        logging.info("Found activation scales cache, starting loading")
        act_scales = torch.load(args.act_scales_cache)
    else:
        # use default dataset (pile) to calibrate scaling factors
        act_scales = run_calibration(model, model_type, tokenizer)
        
        if args.act_scales_cache:
            logging.info(f"Store activation scales at {args.act_scales_cache}")
            torch.save(act_scales, args.act_scales_cache)
    logging.info(f"Start Quantizing Model")
    # Replace module with quantized version
    model = quantize_blocks(model, model_type, act_scales, device)
    return model




