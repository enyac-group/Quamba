from packaging import version

import torch

import triton
assert version.parse(triton.__version__) >= version.parse("3.0.0"), \
    f"Triton >= 3.0.0 is requried, but get {triton.__version__}"

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from .triton.quant_normgated import _qlayer_normgated_fwd
from .triton.quant_norm import _qlayer_norm_fwd

class QRMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.empty(hidden_size, device=device, dtype=torch.float16))
        self.bias = None
        self.output_scale = 0.0
        self.dim = hidden_size
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    @classmethod
    def from_fp16(cls, originalLayer: RMSNorm, output_scale=None):
        qnorm = cls(originalLayer.weight.shape[0], originalLayer.eps, originalLayer.weight.device)
        qnorm.weight = originalLayer.weight.clone().contiguous()
        
        qnorm.output_scale = None
        if output_scale is not None:
            assert type(output_scale) is float, "Only support per-tensor static quant for output"
            qnorm.output_scale = output_scale
        return qnorm

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'output_scale'] = self.output_scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'output_scale']

    def to(self, *args, **kwargs):
        super(QRMSNorm, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False, **kwargs):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, residual_out, per_token_scale = _qlayer_norm_fwd(
            x,
            self.weight,
            self.bias,
            self.eps,
            residual,
            static_out_scale=self.output_scale,
            residual_dtype=residual_dtype,
            is_rms_norm=True,
        )
        if self.output_scale is not None:
            y = y.reshape(x_shape_og)
            residual_out = residual_out.reshape(x_shape_og)
            return y if not prenorm else (y, residual_out)
        else:
            # output per_token scaling factor if output_scale is None
            y = y.reshape(x_shape_og)
            residual_out = residual_out.reshape(x_shape_og)
            per_token_scale = per_token_scale.reshape(x_shape_og[0:-1])
            return (y, per_token_scale) if not prenorm else (y, residual_out, per_token_scale)

    def __repr__(self):
        return f"QRMSNorm(dim={self.dim}, eps={self.eps}, output_scale={self.output_scale})"


class QRMSNormGated(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None,
                 norm_before_gate=True, use_float16_output=False, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.empty(hidden_size, device=device, dtype=torch.float16))
        self.bias = None
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.use_float16_output = use_float16_output
        self.z_scale = 0.0
        self.output_scale = 0.0
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    @classmethod
    def from_fp16(cls, originalLayer: RMSNormGated, z_scale=None, output_scale=None, use_float16_output=False):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        qnorm = cls(originalLayer.weight.shape[0], originalLayer.eps, originalLayer.group_size,
            originalLayer.norm_before_gate, use_float16_output, originalLayer.weight.device)

        qnorm.weight = originalLayer.weight.clone()
        if z_scale is not None:
            assert type(z_scale) is float, "Only support per-tensor static quant for input z"
            qnorm.z_scale = z_scale
        if output_scale is not None:
            # output "static per-tensor" scaling factor if output_scale is provided and use_float16_output is False
            assert type(output_scale) is float, "Only support per-tensor static quant for output"
            qnorm.output_scale = output_scale
        else:
            # output "dynamic per-token" scaling factor if output_scale is None and use_float16_output is False
            qnorm.output_scale = None 

        return qnorm

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'z_scale'] = self.z_scale
        state_dict[prefix + 'output_scale'] = self.output_scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.z_scale = state_dict[prefix + 'z_scale']
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'z_scale']
        del state_dict[prefix + 'output_scale']

    def forward(self, x, q_z=None):
        """If q_z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if q_z is not None:
            assert q_z.shape == x_shape_og
            q_z = q_z.reshape(-1, q_z.shape[-1])
            if q_z.stride(-1) != 1:
                q_z = q_z.contiguous()
        y, y_token_scale = _qlayer_normgated_fwd(
            x, self.weight, self.bias,
            self.eps, q_z=q_z,
            z_scale=self.z_scale,  
            static_out_scale=self.output_scale,
            use_float16_output=self.use_float16_output,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True)
        if self.use_float16_output:
            return y.reshape(x_shape_og)
        else:
            if self.output_scale is not None:
                return y.reshape(x_shape_og)
            else:
                # output per_token scaling factor if output_scale is None
                return y.reshape(x_shape_og), y_token_scale.reshape(x_shape_og[0:-1])

    def __repr__(self):
        return f"QRMSNormGated(dim={self.weight.shape[0]}, eps={self.eps}, z_scale={self.z_scale}, output_scale={self.output_scale}, use_float16_output={self.use_float16_output})"