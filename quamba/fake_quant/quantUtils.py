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


@torch.no_grad()
def uniform_affine_fake_quantization(
        w: torch.tensor, n_bits, sym,
        scales = None, base = None
    ):
    q_min, q_max = _get_quant_range(n_bits, sym=sym)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    return w

@torch.no_grad()
def dynamic_per_tensor_absmax_quantization(
        w: torch.tensor, n_bits, sym, clip_ratio=1.0,
    ):
    q_min, q_max = _get_quant_range(n_bits, sym=sym)
    #Calculating the scale dynamically
    if sym:
        w_max = w.abs().amax().clamp(min=1e-5)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(w)
    else:
        w_max = w.amax()
        w_min = w.amin()
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)  
        
    # fake quantization
    return uniform_affine_fake_quantization(w, n_bits, sym, scales, base)



@torch.no_grad()
def bached_dynamic_per_tensor_absmax_quantization(
    x: torch.tensor, n_bits, sym, clip_ratio=1.0):
    
    assert x.dim() == 3
    q_min, q_max = _get_quant_range(n_bits, sym=sym)
    #Calculating the scale dynamically
    if sym:
        w_max = x.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(x)
    else:
        w_max = x.amax(dim=(-2, -1), keepdim=True)
        w_min = x.amin(dim=(-2, -1), keepdim=True)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        
    return uniform_affine_fake_quantization(x, n_bits, sym, scales, base)
    
    
#This function assuming the tensor to be quantized are of shape (B, L, D)
@torch.no_grad()
def dynamic_per_token_absmax_quantization(
    x: torch.tensor, n_bits, sym, clip_ratio=1.0):
    
    assert x.dim() == 3
    q_min, q_max = _get_quant_range(n_bits, sym=sym)
    #Calculating the scale dynamically
    if sym:
        w_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        bases = torch.zeros_like(x)
    else:
        w_max = x.amax(dim=-1, keepdim=True)
        w_min = x.amin(dim=-1, keepdim=True)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        bases = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        
    return uniform_affine_fake_quantization(x, n_bits, sym, scales, bases)
    
    
if __name__ == '__main__':
    # Create a tensor with random inputs
    x = torch.randn(2, 3, 4)
    print(x)
    # Invoke dynamic_per_tensor_absmax_quantization
    quantized_tensor1 = dynamic_per_tensor_absmax_quantization(x, n_bits=8, sym=True, clip_ratio=0.9)
    
    # Invoke bached_dynamic_per_tensor_absmax_quantization
    quantized_tensor2 = bached_dynamic_per_tensor_absmax_quantization(x, n_bits=8, sym=True, clip_ratio=0.9)
    
    # Invoke dynamic_per_token_absmax_quantization
    quantized_tensor3 = dynamic_per_token_absmax_quantization(x, n_bits=8, sym=True, clip_ratio=0.9)
    
    # Print the quantized tensors
    print("Quantized Tensor 1:", quantized_tensor1)
    print("Quantized Tensor 2:", quantized_tensor2)
    print("Quantized Tensor 3:", quantized_tensor3)