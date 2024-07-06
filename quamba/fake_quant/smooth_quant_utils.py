import torch
import torch.nn as nn
from functools import partial
import logging
from datasets import load_dataset
from mamba_ssm.utils.generation import InferenceParams
from tqdm import tqdm
from quamba.fake_quant.qLinearLayer import QLinearLayer

class SmoothModule(nn.Module):
    def __init__(self, weight_to_smooth, tensor_name=None):
        super(SmoothModule, self).__init__()
        self.tensor_name = tensor_name
        self.weight_to_smooth=weight_to_smooth
        self.register_buffer("scales", None)
        self.activated = False
    @torch.no_grad()
    def forward(self, x, reverse=False):
        assert not torch.isnan(x).any(), "Input tensor x contains NaNs."
        if not self.activated:
            return x
        else:
            if reverse:
                return x.mul(self.scales)
            else:
                return x.div(self.scales)
        
    def configure(self, scales):
        self.scales = scales
        assert not torch.isnan(self.scales).any(), "Scales contains NaNs."
        self.activated = True
        

def get_act_scalers_mamba(model, tokenizer, 
                num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    
    act_scales = {}
    
    def stat_act_hook(m, inputs, outputs, name):
        x = inputs[0].clone().detach() if isinstance(inputs, tuple) else inputs.clone().detach()
        assert x.dim() == 3, "Assuming x is of input shape (B, L, D)"
        comming_max = x.abs().amax(dim=(0, 1))
        
        if name not in act_scales:
            act_scales[name] = comming_max
        else:
            act_scales[name] = torch.max(act_scales[name], comming_max)
    
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            hooks.append(
                m.register_forward_hook(partial(stat_act_hook, name=name))
            )
            
    logging.info("Prepare for smoothing..")
    calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    calibration_dataset.shuffle(seed=42)
    logging.info("Run smoothing calibration")
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(calibration_dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for h in hooks:
        h.remove()


    return act_scales


@torch.no_grad()
def smooth_fc(weight, act_scale, alpha=0.5):
    device = weight.device
    dtype = weight.dtype
    act_scale = act_scale.to(device).to(dtype)
    # linear fc weight shape [out_dim, in_dim]
    weight_scale = weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5) # [out_dim, in_dim] -> [1, in_dim]

    if act_scale.dim() == 0:
        sm_scale = (act_scale[None].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
            min=1e-5).to(device).to(dtype)
    else:
        sm_scale = (act_scale[None, :].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
            min=1e-5).to(device).to(dtype)

    return weight.mul_(sm_scale), sm_scale

def smooth_mamba(model, tokenizer, 
                num_samples=512, seq_len=512):
    
    act_scales = get_act_scalers_mamba(model, tokenizer, num_samples, seq_len)
    #TODO: Calculate the real act scales with linear layers.
    smooth_scales = {}
    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            name_prefix = ".".join(name.split(".")[:-1])
            weight_name = name_prefix + "." + m.weight_to_smooth

            # smoothmodule contains weight name for the 
            # corresponding QLinearLayer
            weight_module = model.get_submodule(weight_name)
            original_weight = weight_module.weight.clone()
            scale = act_scales[name]
            smooth_weight, sm_scale = smooth_fc(weight_module.weight, scale, alpha=0.5)
            smooth_scales[name] = sm_scale

            m.weight = smooth_weight

            # logging.info(f"Configure smooth module {name}")
            # m.configure(smooth_scales[name])

    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            logging.info(f"Configure smooth module {name}")
            m.configure(smooth_scales[name])
            
    