import os
import logging
from functools import partial
from tqdm import tqdm

import torch
from datasets import load_dataset

from mamba_ssm.modules.block import Block

from fake_quant.qMambaLayer import QMamba
from fake_quant.qJamba import QJambaMambaMixer, QJambaSdpaAttention, QJambaSparseMoeBlock
from fake_quant.qActLayer import QAct
from fake_quant.qConvLayer import QConv1D
from fake_quant.qLinearLayer import QLinearLayer
from fake_quant.observer import build_observer
from fake_quant.qSelectiveScan import QSScan
from fake_quant.hadamard_utils import apply_exact_had_to_linear
from fake_quant.rotation_utils import HadamardTransform
from fake_quant.smooth_quant_utils import smooth_mamba

def run_calibration(
    model, tokenizer, act_quant_configs, num_samples=512, seq_len=512):
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
            act_quant_config = act_quant_configs.get(tensor_name, act_quant_configs["default"])
            a_bits = act_quant_config["bits"]  
            a_clip_ratio = act_quant_config["clip_ratio"]
            a_sym = act_quant_config.get("sym", True)
            a_percentile_alpha = act_quant_config.get("percentile_alpha", 0.99999)
            a_observer_type = act_quant_config.get("observer_type", "PerTensorMinmaxObserver")
            logging.debug(f"Create observer for {name} with {a_observer_type} observer")
            observers[name] = build_observer(
                observer_type=a_observer_type, 
                n_bits=a_bits, 
                clip_ratio=a_clip_ratio, 
                sym=a_sym,
                percentile_alpha=a_percentile_alpha
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
                logging.debug(f"Apply Hadamard Weights of {name}")
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

def configure_weight_quant(model, weight_quant_configs):
    for name, m in model.named_modules():
        layer_type = name.split(".")[-1]
        weight_quant_config = weight_quant_configs.get(layer_type, weight_quant_configs["default"])
        w_bits = weight_quant_config["bits"]
        w_clip_ratio = weight_quant_config["clip_ratio"]
        logging.debug(f"Set {name} to {w_bits} bit quant.")
        #NOTE(brian1009): I still keep the following cord, 
        # as we might need some different specification for different type of layer
        if isinstance(m, (QLinearLayer)):
            m.configure(
                n_bits=w_bits,
                clip_ratio=w_clip_ratio,
            )
        elif isinstance(m, QConv1D):
            m.configure(
                n_bits=w_bits,
                clip_ratio=w_clip_ratio,
            )
        elif isinstance(m, QSScan):
            m.configure(
                n_bits=w_bits,
                clip_ratio=w_clip_ratio,
            )
            
def configure_act_quant(model, tokenizer, act_quant_configs, do_calibrate=False, num_calib_samples=512, act_scales_cache=None):
    act_scales = {}
    if do_calibrate:
        # Run calibration to get scale
        if act_scales_cache and os.path.isfile(act_scales_cache):
            logging.info("Found actiavtion scales cache, starting loading")
            act_scales = torch.load(act_scales_cache)
        else:
            logging.info(f"Start calibration for activation quantization with {num_calib_samples}")
            act_scales = run_calibration(model, tokenizer, act_quant_configs, num_samples=num_calib_samples)
            
        if act_scales_cache:
            torch.save(act_scales, act_scales_cache)
    for name, m in model.named_modules():
        if isinstance(m, QAct):
            tensor_name = m.tensor_name
            if tensor_name not in act_quant_configs:
                logging.warning(f"Activation {tensor_name} is not in the config, use default config")
            act_quant_config = act_quant_configs.get(tensor_name, act_quant_configs["default"])
            a_bits = act_quant_config["bits"]
            a_quant_type = act_quant_config["quant_type"]
            is_static = act_quant_config["is_static"]    
            a_clip_ratio = act_quant_config["clip_ratio"]
            a_sym = act_quant_config.get("sym", True)
            if is_static:
                (scale, base) = act_scales.get(name)
                if scale is None:
                    raise ValueError(f"Static quantization requires scale for {name}, please run calibration first using --do_calibrate")
            else:
                scale = None
                base = None
            
            logging.debug(f"Set {name} to {a_quant_type} quant. #bits {a_bits}. {'dynamic' if not is_static else 'static'}, sym: {a_sym}")
            m.configure(
                n_bits = a_bits, 
                sym = a_sym,
                quantization_type=a_quant_type, 
                o_scales=scale, 
                o_base = base,
                clip_ratio=a_clip_ratio, 
                static_quant=is_static,
            )

def prepare_quantize_model_mamba(model, device, model_type="mamba", quant_attn=False, quant_mamba=True, quant_moe=False, act_quant=True):
    logging.info(f"Inserting/Creating Quantized module")
    model.config.use_cache = False
    if model_type == "jamba":
        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            block_str=str(type(layers[i]))
            if quant_mamba and "JambaMambaDecoderLayer" in block_str:
                logging.info("=== Quant " + block_str)
                layers[i].mamba = QJambaMambaMixer(layers[i].mamba, act_quant=act_quant)
            elif quant_attn and "JambaAttentionDecoderLayer" in block_str:
                logging.info("=== Quant " + block_str)
                layers[i].self_attn = QJambaSdpaAttention(layers[i].self_attn)
            elif quant_moe:
                logging.info("=== Quant " + block_str)
                layers[i].moe = QJambaSparseMoeBlock(layers[i].moe, act_quant=act_quant)
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

def quantize_model_mamba(model, model_type, tokenizer, device, args, quantization_config):
    logging.info(f"Start Quantizing Model")

    act_quant_configs = quantization_config.get("act_quantization_config", False)
    weight_quant_configs = quantization_config["weight_quantization_config"]
    do_rotation = quantization_config.get("do_rotation", False)
    quant_attn = quantization_config.get("quant_attn", False)
    quant_mamba = quantization_config.get("quant_mamba", True)
    quant_moe = quantization_config.get("quant_moe", False)
    
    model = prepare_quantize_model_mamba(model, device, model_type, act_quant=bool(act_quant_configs), quant_attn=quant_attn, quant_mamba=quant_mamba, quant_moe=quant_moe)
    if args.do_smoothing:
        logging.info(f"Start doing smoothing")
        smooth_mamba(model, tokenizer, num_samples=5 if args.testing else 512)
    if do_rotation:
        logging.info(f"Start doing rotation")
        rotate_out_proj(model)
        activate_rotate_module(model)
    configure_weight_quant(model, weight_quant_configs)
    configure_act_quant(model, tokenizer,
                        do_calibrate=args.do_calibrate,
                        act_quant_configs=act_quant_configs,
                        act_scales_cache=args.act_scales_cache,
                        num_calib_samples=args.calib_data_num
                    )
    activate_quant_module(model)
    return model




