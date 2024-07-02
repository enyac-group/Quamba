import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from .quantUtils import quantize_tensor_per_tensor_absmax

import quant_causal_conv1d_cuda

class QCausalConv1D(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Conv1d,
        input_scale=1.0,
        output_scale=1.0
    ):
        super().__init__()
        
        self.input_scale = input_scale
        self.output_scale = output_scale

        # Copy convolution-specific parameters
        self.in_channels = originalLayer.in_channels
        self.out_channels = originalLayer.out_channels
        self.kernel_size = originalLayer.kernel_size
        self.stride = originalLayer.stride
        self.padding = originalLayer.padding
        self.dilation = originalLayer.dilation
        self.groups = originalLayer.groups

        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight.clone().detach(),
            n_bits = 8,
            clip_ratio = 1.0,
            scales = None,
            real_quant = True
        )
        int8_weight = int8_weight.to(torch.int8).contiguous()
        self.register_buffer('weight', rearrange(int8_weight, "d 1 w -> d w"))
        self.weight_scale = weight_scale
        if originalLayer.bias is not None:
            int8_bias, bias_scale = quantize_tensor_per_tensor_absmax(
                originalLayer.bias.clone().detach(),
                n_bits = 8,
                clip_ratio = 1.0,
                scales = None,
                real_quant = True
            )
            int8_bias = int8_bias.to(torch.int8).contiguous()
            self.register_buffer('bias', int8_bias)
            self.bias_scale = bias_scale
        else:
            self.bias = None
            self.bias_scale = 1.0

    @torch.no_grad()
    def forward(self, x):
        y = quant_causal_conv1d_cuda.fwd(
                x, self.input_scale,
                self.weight, self.weight_scale,
                self.output_scale,
                self.bias_scale, self.bias,
                None, None, None, True
            )
        return y

    @torch.no_grad()
    def update(self, x, conv_state):
        # update conv_state in-place
        y = quant_causal_conv1d_cuda.update(
            x, conv_state, self.input_scale,
            self.weight, self.weight_scale,
            self.output_scale,
            self.bias_scale, self.bias, True
        ) 
        return y

    def to(self, *args, **kwargs):
        super(QCausalConv1D, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    def __repr__(self):
        return f"QCausalConv1D({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"

    