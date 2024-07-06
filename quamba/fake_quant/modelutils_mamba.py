import os
import logging
from functools import partial
from tqdm import tqdm

import torch
from datasets import load_dataset

from mamba_ssm.modules.block import Block

from quamba.fake_quant.qMambaLayer import QMamba
from quamba.fake_quant.qJamba import QJambaMambaMixer
from quamba.fake_quant.qActLayer import QAct
from quamba.fake_quant.qConvLayer import QConv1D
from quamba.fake_quant.qLinearLayer import QLinearLayer
from quamba.fake_quant.observer import build_observer
from quamba.fake_quant.qSelectiveScan import QSScan
from quamba.fake_quant.hadamard_utils import apply_exact_had_to_linear
from quamba.fake_quant.rotation_utils import HadamardTransform
from quamba.fake_quant.smooth_quant_utils import smooth_mamba

def run_calibration(
    model, tokenizer, percentile_u=True, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    observers = {}
    
    def stat_act_hook(m, inputs, outputs, name):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        # register the new information to observer
        observers[name].update(inputs.clone().detach())

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, QAct):
            tensor_name = m.tensor_name
            hooks.append(
                m.register_forward_hook(partial(stat_act_hook, name=name))
            )
            if percentile_u and tensor_name == "u_quant":
                a_observer_type = "PerTensorPercentileObserver"
            else:
                a_observer_type = "PerTensorMinmaxObserver"
            logging.debug(f"Create observer for tensor {name} with {a_observer_type} observer")
            observers[name] = build_observer(
                observer_type=a_observer_type, 
                n_bits=8,
                clip_ratio=1.0,
                sym=True,
                percentile_alpha=0.99999
            )

    logging.info("Prepare calibration input")
    calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    calibration_dataset.shuffle(seed=42)
    logging.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(calibration_dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids) 
    
    for h in hooks:
        h.remove()
        
    act_scales = {}
    for name, observer in observers.items():
        act_scales[name] = observer.get_quantization_parameters()
    return act_scales
        
def rotate_out_proj(model):
    for name, m in model.named_modules():
        if isinstance(m, QLinearLayer):
            if "out_proj" in name:
                logging.debug(f"Apply Hadamard Transform onto {name}")
                apply_exact_had_to_linear(m, had_dim=-1, output=False)
                
def activate_rotate_module(model):
    for name, m in model.named_modules():
        if isinstance(m, (HadamardTransform)):
            logging.debug(f"Activate on-line hadamard transform module, {name}")
            m.configure(do_rotate=True)
            
def activate_quant_module(model):
    for name, m in model.named_modules():
        if isinstance(m, (QAct, QLinearLayer, QConv1D, QSScan)):
            m.is_quant_mode = True

def configure_weight_quant(model):
    for name, m in model.named_modules():
        if isinstance(m, (QLinearLayer, QConv1D, QSScan)):
            logging.debug(f"Set {name} to 8 bit quant.")
            m.configure(
                n_bits=8,
            )
            
def configure_act_quant(model, act_scales):
    for name, m in model.named_modules():
        if isinstance(m, QAct):
            (scale, base) = act_scales.get(name)
            if scale is None:
                raise ValueError(f"Static quantization requires scale for {name}, but got None. Please check the calibration process.")
            
            logging.debug(f"Configure QAct module: {name} with 8 bits quantization.")
            m.configure(
                n_bits = 8,
                sym = True,
                o_scales=scale, 
                o_base=base,
            )

def prepare_quantize_model_mamba(model, device, model_type="mamba"):
    logging.info(f"Inserting/Creating Quantized module")
    model.config.use_cache = False
    if model_type == "jamba":
        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            block_str=str(type(layers[i]))
            if "JambaMambaDecoderLayer" in block_str:
                logging.info("=== Quant " + block_str)
                layers[i].mamba = QJambaMambaMixer(layers[i].mamba)
            else:
                logging.info("=== Layer "+str(i) + " Not Quantized.  " + block_str)
    elif model_type == "mamba":
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            m = None
            if isinstance(layers[i], Block):
                m = QMamba(originalLayer=layers[i].mixer)
            if m is None:
                continue

            m = m.to(device)
            layers[i].mixer = m
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'jamba'")
    return model

def prepare_act_scales(model, device, tokenizer, args):
    act_scales = {}
    if args.act_scales_cache and os.path.isfile(args.act_scales_cache):
        logging.info("Found actiavtion scales cache, starting loading")
        act_scales = torch.load(args.act_scales_cache)
    else:
        logging.info(f"Start calibration for activation quantization with {args.calib_data_num}")
        act_scales = run_calibration(model, tokenizer, percentile_u=args.do_percentile_u, num_samples=args.calib_data_num)
        if args.act_scales_cache:
            torch.save(act_scales, args.act_scales_cache)
    return act_scales

def quantize_model_mamba(model, model_type, tokenizer, device, args):
    logging.info(f"Start Quantizing Model")
    
    model = prepare_quantize_model_mamba(model, device, model_type)
    if args.do_smoothing:
        logging.info(f"Start doing smoothing")
        smooth_mamba(model, tokenizer, num_samples=5 if args.testing else 512)
    if args.do_hadamard:
        rotate_out_proj(model)
        activate_rotate_module(model)
    configure_weight_quant(model)
    act_scales = prepare_act_scales(model, device, tokenizer, args)
    configure_act_quant(model, act_scales)
    activate_quant_module(model)
    return model




