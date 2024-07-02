from typing import Dict
import torch
import torch.nn as nn
from functools import partial

# from mamba_ssm.ops.triton.layernorm import RMSNorm
import rms_norm_cuda

class FusedRMSNorm(torch.nn.Module):

    def __init__(self,
        # originalLayer: RMSNorm, # triton issue
        originalLayer,
        output_scale: float):
        super().__init__()
        
        self.eps = originalLayer.eps
        self.dim = tuple(originalLayer.weight.shape)
        self.register_buffer('weight', originalLayer.weight.clone())
        self.output_scale = output_scale


    def to(self, *args, **kwargs):
        super(FusedRMSNorm, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, residual=None, prenorm=False, **kwargs):
        ret = rms_norm_cuda.fwd(x, self.dim, self.weight, residual, self.eps, self.output_scale)
        # ret is a list
        if residual is not None:
            y, residual = ret
            return y if not prenorm else (y, residual)
        else:
            y = ret[0]
            return y if not prenorm else (y, x)

    def __repr__(self):
        return f"FusedRMSNorm(dim={self.dim}, eps={self.eps}, output_scale={self.output_scale})"
