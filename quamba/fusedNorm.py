from typing import Dict
import torch
import torch.nn as nn
from functools import partial

# from mamba_ssm.ops.triton.layernorm import RMSNorm
import rms_norm_cuda

class FusedRMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.drop = None # no dropout for quantized norm
        self.register_buffer("weight", torch.empty(hidden_size, **factory_kwargs))
        self.dim = tuple(self.weight.shape)
        self.output_scale = 0.0
        self.register_parameter("bias", None)
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    @classmethod
    def from_fp16(cls,
        # originalLayer: RMSNorm, # triton issue
        originalLayer,
        output_scale: float):
        qnorm = cls(originalLayer.weight.shape[0], eps=originalLayer.eps,
            dropout_p=0.0, device=originalLayer.weight.device,
            dtype=originalLayer.weight.dtype)
        # qnorm.weight is a buffer, not parameter
        qnorm.weight = originalLayer.weight.data.clone()
        qnorm.output_scale = output_scale
        return qnorm

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'output_scale'] = self.output_scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'output_scale']
    
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
