import torch
import torch.nn as nn
from functools import partial

from .hadamard_utils import get_had_fn, get_qhad_fn


class QHadamard(torch.nn.Module):

    def __init__(self, dim, x_H_scale=1.0):
        super().__init__()
        self.dim = dim
        self.x_H_scale = x_H_scale
        self.transform_fn, self.N, self.had_scale = get_qhad_fn(dim)
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'x_H_scale'] = self.x_H_scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.x_H_scale = state_dict[prefix + 'x_H_scale']
        del state_dict[prefix + 'x_H_scale']

    def to(self, *args, **kwargs):
        super(QHadamard, self).to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # TODO: fuse quantization into cuda
        # Hadamard will become an overhead when L is large (L>=512)
        qx = self.transform_fn(x, self.had_scale/self.x_H_scale) 
        return qx

    def __repr__(self):
        return f"QHadamard(dim={self.dim})"
    

class Hadamard(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform_fn, self.N, self.had_scale = get_had_fn(dim)

    def to(self, *args, **kwargs):
        super(Hadamard, self).to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        return self.transform_fn(x.contiguous(), self.had_scale) 

    def __repr__(self):
        return f"Hadamard(dim={self.dim})"
