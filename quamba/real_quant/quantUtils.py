import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange, repeat

def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max

def _get_out_type(n_bits, sym):
    assert n_bits <= 16

    if n_bits == 16:
        out_type = torch.int16 if sym else torch.uint16
    elif n_bits == 8:
        out_type = torch.int8 if sym else torch.uint8
    elif n_bits == 4:
        out_type = torch.int8 if sym else torch.uint8 # since torch.int4 has not been released yet, use int8 at the moment
    else:
        raise ValueError(f"Unsupported quantization bit-width: {n_bits},"
                         "only 16, 8, 4 are supported")
    return out_type

def uniform_symmetric_quantization(w:torch.tensor, n_bits, scales):
    q_min, q_max = _get_quant_range(n_bits=n_bits, sym=True) # currently support symmetric only
    w = torch.clamp(torch.round(w / scales), q_min, q_max) # TODO(brian1009): fix output type
    return w

def get_uniform_symmetric_quantization_params(w_max, n_bits, clip_ratio):
    _, q_max = _get_quant_range(n_bits=n_bits, sym=True)
    if clip_ratio < 1.0:
        w_max = w_max * clip_ratio
    scales = w_max / q_max
    return scales


@torch.no_grad()
def quantize_tensor_per_tensor_absmax(
        w: torch.tensor, n_bits, clip_ratio=1.0,
        scales = None, real_quant=True
    ):
    q_min, q_max = _get_quant_range(n_bits, sym=True)
    #Calculating the scale dynamically
    if scales is None: 
        w_max = w.abs().amax().clamp(min=1e-5)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max    
    # real quant
    w = torch.clamp(torch.round(w / scales), q_min, q_max)
    if not real_quant:
        return w * scales, 1.0
    else:
        return w, scales.float().item()



#This function assuming the tensor to be quantized are of shape (B, L, D)
@torch.no_grad()
def quantize_activation_per_tensor_absmax(
    x: torch.tensor, n_bits, clip_ratio=1.0,
    scales = None, real_quant=True):
    
    assert x.dim() == 3
    saved_shape = x.shape 
    q_min, q_max = _get_quant_range(n_bits, sym=True)
    #Calculating the scale dynamically
    if scales is None: 
        w_max = x.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        
    # real quant
    x = torch.clamp(torch.round(x / scales), q_min, q_max)
    #x = torch.clamp(x/scales, q_min, q_max)
    #x = torch.clamp((x / scales).round(), q_min, q_max)
    if not real_quant:
        x =  x * scales
        scales = torch.ones_like(scales)
    
    return x.reshape(saved_shape), scales
    
    
    
    
if __name__ == '__main__':
    num_runs = 100
    test_pass = True
    for _ in range(num_runs):
        rand_input = torch.randn(16, 197, 384) # random input of shape (B, L, D)
        batch_quant_result, batch_scales = quantize_activation_per_tensor_absmax(
            x = rand_input,
            n_bits = 8,
            real_quant=False
        ) # (B, L, D)
        
        golden_quant_result = []
        golden_scales = []
        for i in range(rand_input.shape[0]):
            single_data = rand_input[i].unsqueeze(0)
            single_quant_result, scale = quantize_tensor_per_tensor_absmax(
                w = single_data,
                n_bits=8,
                real_quant=False
            )
            golden_scales.append(scale.view(1, 1, 1))
            golden_quant_result.append(single_quant_result)
            
        golden_scales = torch.concat(golden_scales)
        golden_quant_result = torch.concat(golden_quant_result)
        if not torch.allclose(batch_quant_result, golden_quant_result):
            test_pass = False
    
    print("Batch Quant v.s. Single Quant Then Concat:", test_pass) #Expected to be True
    
    