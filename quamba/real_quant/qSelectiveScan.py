"""
The code is modfied from
https://github.com/state-spaces/mamba
"""

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

try:
    import quant_sscan_cuda
except ImportError:
    quant_sscan_cuda = None

try:
    from .triton.selective_state_update import qsscan_update_triton
except ImportError:
    qsscan_update_triton = None

from .quantUtils import quantize_tensor_per_tensor_absmax

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()
    return w, scales


class QSScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, u_scale, delta, delta_scale, A, A_scale, B, B_scale, C, C_scale, out_scale,
                D=None, D_scale=None, z=None, z_scale=None, delta_bias=None, delta_bias_scale=None,
                delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
        out, x = quant_sscan_cuda.fwd(
            u, u_scale, delta, delta_scale, A, A_scale, B, B_scale, C, C_scale, out_scale,
            D, D_scale, z, z_scale, delta_bias, delta_bias_scale, delta_softplus)
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        return out if not return_last_state else (out, last_state)
        # if z is None:
        #     return out if not return_last_state else (out, last_state)
        # else: # has z
        #     ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
        #     out_z = rest[0]
        #     return out_z if not return_last_state else (out_z, last_state)


def quant_selective_scan_fn(
        u, scale_u, delta, scale_delta,
        A, scale_A, B, scale_B, C, scale_C, out_scale,
        D=None, scale_D=None, z=None, scale_z=None,
        delta_bias=None, scale_delta_bias=None,
        delta_softplus=False, return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return QSScanFn.apply(
        u, scale_u, delta, scale_delta, A, scale_A, 
        B, scale_B, C, scale_C, out_scale,
        D, scale_D, z, scale_z, delta_bias, scale_delta_bias, 
        delta_softplus, return_last_state)


#W8A8B8O8 QSScan
class QSScan(nn.Module):

    def __init__(self, A_log, D=None, dt_bias=None, delta_softplus=True,
                 u_scale=1.0, dt_scale=1.0, B_scale=1.0, C_scale=1.0, z_scale=1.0, output_scale=1.0):
        super().__init__()

        A_log_quant, A_log_scale = quantize_weight_per_tensor_absmax(A_log, n_bits=8)
        A_log_quant = A_log_quant.to(torch.int8)
        self.register_buffer('A_log', A_log_quant)
        self.register_buffer('A_scale', A_log_scale.float())

        if D is not None:
            D_quant, D_scale = quantize_weight_per_tensor_absmax(D, n_bits=8)
            D_quant = D_quant.to(torch.int8)
            self.register_buffer('D', D_quant)
            self.register_buffer('D_scale', D_scale.float())
        else:
            self.D = None
        
        if dt_bias is not None:
            dt_bias_quant, dt_bias_scale = quantize_weight_per_tensor_absmax(dt_bias, n_bits=8)
            dt_bias_quant = dt_bias_quant.to(torch.int8)
            self.register_buffer('dt_bias', dt_bias_quant)
            self.register_buffer('dt_bias_scale', dt_bias_scale.float())
        else:
            self.dt_bias = None
            self.register_buffer('dt_bias_scale', torch.tensor(0.0))
        
        self.register_buffer('u_scale', u_scale.float())
        self.register_buffer('dt_scale', dt_scale.float())
        self.register_buffer('B_scale', B_scale.float())
        self.register_buffer('C_scale', C_scale.float())
        self.register_buffer('z_scale', z_scale.float())
        self.register_buffer('output_scale', output_scale.float())
        self.delta_softplus = delta_softplus

        if qsscan_update_triton is not None:
            self.qssm_update_fn = qsscan_update_triton
        else:
            self.qssm_update_fn = quant_sscan_cuda.update

    #NOTE(HY): Only activate q_sscan when real_quant is True,
    # since quantize_tensor_per_tensor_absmax only returns real scales when real_quant is True.
    @torch.no_grad()
    def forward(self, u, dt, B, C, z=None, return_last_state=False):

        # q_sscan output int8 y
        y = quant_selective_scan_fn(
            u, self.u_scale[None],
            dt, self.dt_scale[None],
            self.A_log, self.A_scale[None], 
            B, self.B_scale[None],
            C, self.C_scale[None],
            self.output_scale[None],
            D=self.D if self.D is not None else None,
            scale_D=self.D_scale[None] if self.D is not None else None,
            z=z, scale_z=self.z_scale[None],
            delta_bias=self.dt_bias if self.dt_bias is not None else None,
            scale_delta_bias=self.dt_bias_scale[None] if self.dt_bias is not None else None,
            delta_softplus=self.delta_softplus,
            return_last_state=return_last_state,
        )
        return y


    @torch.no_grad()
    def update(self, ssm_state, u, dt, B, C, z=None):

        y = self.qssm_update_fn(
            ssm_state, u, self.u_scale[None],
            dt, self.dt_scale[None],
            self.A_log, self.A_scale[None],
            B, self.B_scale[None], 
            C, self.C_scale[None],
            self.D if self.D is not None else None,
            self.D_scale[None] if self.D is not None else None,
            z if z is not None else None,
            self.z_scale[None] if z is not None else None,
            self.dt_bias if self.dt_bias is not None else None,
            self.dt_bias_scale[None] if self.dt_bias is not None else None,
            self.delta_softplus
        )
        return y, ssm_state
    
    def to(self, *args, **kwargs):
        super(QSScan, self).to(*args, **kwargs)
        self.A_log = self.A_log.to(*args, **kwargs)
        if self.D is not None:
            self.D = self.D.to(*args, **kwargs)
        if self.dt_bias is not None:
            self.dt_bias = self.dt_bias.to(*args, **kwargs)
        return self

    def __repr__(self):
        return f"QSScan()"