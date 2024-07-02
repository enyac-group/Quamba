import torch
import torch.nn as nn

import quant_linear_cuda

from .hadamard_utils import get_had_fn
from .quantUtils import quantize_tensor_per_tensor_absmax


class W8A8B8O8Linear(torch.nn.Module):

    def __init__(self,
        originalLayer: nn.Linear,
        input_scale=1.0,
        output_scale=1.0):
        super().__init__()
        
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features

        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight.clone().detach().to(torch.float32),
            n_bits = 8,
            clip_ratio = 1.0,
            scales = None,
            real_quant = True
        )
        int8_weight = int8_weight.to(torch.int8).t()
        self.register_buffer('weight', int8_weight)
        self.register_buffer('a', torch.tensor(
            [[input_scale / output_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight.device)
        )
        self.register_buffer('b', torch.tensor(
            [[weight_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight.device)
        )
        
        # for gemv
        self.alpha = weight_scale * input_scale / output_scale

        assert originalLayer.bias is None
        self.bias = None

    def to(self, *args, **kwargs):
        super(W8A8B8O8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # input [(bsize*seqlen) x in_dim] --> dim last, use row major
        # weight [out_dim x in_dim] --> use column major
        # output [(bsize*seqlen) x out_dim] --> dim last row major
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)
        if x.shape[0] == 1:
            quant_linear_cuda.cutlass_scaled_mv_dq(y, self.weight.t(), x, self.alpha, 0.0)
        else:
            quant_linear_cuda.cutlass_scaled_mm_dq(y, x, self.weight, self.a, self.b)
        y = y.view(*x_shape[:-1], -1)
        return y

    @torch.no_grad()
    def to_seqlen_last(self, x):
        B, L, D = x.shape
        x = x.view(-1, D) # this may only works for B=1
        # weight.t() [out_dim x in_dim] --> use row major
        # input.t() [(bsize*seqlen) x in_dim] --> dim last, use col major
        # output out_dim x (bsize*seqlen) --> seqlen last --> row major
        pad = 0
        if x.shape[0] % 16 != 0: # cutlass alignment
            # (padding_left,padding_right, padding_top, padding_bottom)
            pad = 16 - x.shape[0] % 16
            x = nn.functional.pad(x, (0, 0, pad, 0), "constant", 0)
        y = torch.empty((self.out_features, x.shape[0]), dtype=x.dtype, device=x.device)
        quant_linear_cuda.cutlass_scaled_mm_dq(y, self.weight.t(), x.t(), self.b, self.a)
        if pad != 0:
            y = y[..., pad:].contiguous()
        y = y.view(B, -1, L)
        return y
    
    def __repr__(self):
        return f"W8A8B8O8Linear(in_features={self.in_features}, out_features={self.out_features})"
    

class W8A8BF16OF16Linear(torch.nn.Module):

    def __init__(self,
        originalLayer: nn.Linear,
        input_scale=1.0,
        out_dtype=torch.float16):
        super().__init__()
        
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
        self.out_dtype = out_dtype

        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight.clone().detach().to(torch.float32),
            n_bits = 8,
            clip_ratio = 1.0,
            scales = None,
            real_quant = True
        )
        int8_weight = int8_weight.to(torch.int8).t()
        self.register_buffer('weight', int8_weight)
        self.register_buffer('a', torch.tensor(
            [[input_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight.device)
        )
        self.register_buffer('b', torch.tensor(
            [[weight_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight.device))
        # for gemv
        self.alpha = weight_scale * input_scale

        assert originalLayer.bias is None
        self.bias = None

    def to(self, *args, **kwargs):
        super(W8A8BF16OF16Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.out_features), dtype=self.out_dtype, device=x.device)
        if x.shape[0] == 1:
            quant_linear_cuda.cutlass_scaled_mv_dq(y, self.weight.t(), x, self.alpha, 0.0)
        else:
            quant_linear_cuda.cutlass_scaled_mm_dq(y, x, self.weight, self.a, self.b)
        y = y.view(*x_shape[:-1], -1)
        return y

    def __repr__(self):
        return f"W8A8BF16OF16Linear(in_features={self.in_features}, out_features={self.out_features})"


class HadW8A8BF16OF16Linear(torch.nn.Module):

    def __init__(self,
        originalLayer: nn.Linear,
        input_scale=1.0,
        out_dtype=torch.float16):
        super().__init__()
        assert originalLayer.weight.is_cuda, "Hadamard transform must be on CUDA"
        
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
        self.out_dtype = out_dtype

        self.transform_fn, self.N, self.had_scale = get_had_fn(
            originalLayer.in_features)

        w_H = self.transform_fn(
            originalLayer.weight.clone().detach().contiguous(), self.had_scale)

        int8_weight_H, weight_H_scale = quantize_tensor_per_tensor_absmax(
            w_H,
            n_bits = 8,
            clip_ratio = 1.0,
            scales = None,
            real_quant = True
        )
        int8_weight_H = int8_weight_H.to(torch.int8).t()
        self.register_buffer('weight', int8_weight_H)
        self.register_buffer('a', torch.tensor(
            [[input_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight_H.device)
        )
        self.register_buffer('b', torch.tensor(
            [[weight_H_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight_H.device))
        # for gemv
        self.alpha = weight_H_scale * input_scale

        assert originalLayer.bias is None
        self.bias = None


    def to(self, *args, **kwargs):
        super(HadW8A8BF16OF16Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # view overhead is around 3 us on A5000
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.out_features), dtype=self.out_dtype, device=x.device)
        if x.shape[0] == 1:
            quant_linear_cuda.cutlass_scaled_mv_dq(y, self.weight.t(), x, self.alpha, 0.0)
        else:
            quant_linear_cuda.cutlass_scaled_mm_dq(y, x, self.weight, self.a, self.b)
        y = y.view(*x_shape[:-1], -1)
        return y

    def __repr__(self):
        return f"HadW8A8BF16OF16Linear(in_features={self.in_features}, out_features={self.out_features}, N={self.N})"
    


class HadLinear(torch.nn.Linear):

    def __init__(self,
        originalLayer: nn.Linear):
        assert originalLayer.weight.is_cuda, "Hadamard transform must be on CUDA"
        super().__init__(
            originalLayer.in_features,
            originalLayer.out_features,
            True if originalLayer.bias is not None else False,
            originalLayer.weight.device,
            originalLayer.weight.dtype,
        )
        
        self.transform_fn, self.N, self.had_scale = get_had_fn(
            originalLayer.in_features)

        w_H = self.transform_fn(
            originalLayer.weight.clone().detach().contiguous(), self.had_scale) 
        self.weight = torch.nn.Parameter(w_H)

    def __repr__(self):
        return f"HadLinear(in_features={self.in_features}, out_features={self.out_features}, N={self.N})"
    