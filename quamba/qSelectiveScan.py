"""
The code is modfied from
https://github.com/state-spaces/mamba
"""
import torch
import torch.nn as nn
from einops import rearrange

import quant_sscan_cuda
from .triton.selective_state_update import quant_sscan_update_triton

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()
    return w, scales


class QSScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, u_scale, delta, delta_scale, A, A_scale, B, B_scale, C, C_scale, ssm_state_scale,
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
            u, u_scale, delta, delta_scale, A, A_scale, B, B_scale, C, C_scale, ssm_state_scale,
            D, D_scale, z, z_scale, delta_bias, delta_bias_scale, delta_softplus)
        # last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        last_state = x[:, :, -1, :]  # (batch, dim, dstate)
        return out if not return_last_state else (out, last_state)
        # if z is None:
        #     return out if not return_last_state else (out, last_state)
        # else: # has z
        #     ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
        #     out_z = rest[0]
        #     return out_z if not return_last_state else (out_z, last_state)


def quant_selective_scan_fn(
        u, scale_u, delta, scale_delta,
        A, scale_A, B, scale_B, C, scale_C, ssm_state_scale,
        D=None, scale_D=None, z=None, scale_z=None,
        delta_bias=None, scale_delta_bias=None,
        delta_softplus=False, return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return QSScanFn.apply(
        u, scale_u, delta, scale_delta, A, scale_A, 
        B, scale_B, C, scale_C, ssm_state_scale,
        D, scale_D, z, scale_z, delta_bias, scale_delta_bias, 
        delta_softplus, return_last_state)


#W8A8B8O8 QSScan
class QSScan(nn.Module):

    def __init__(self, d_state, d_inner, delta_softplus=True):
        device = torch.device("cuda") # triton must be on Cuda
        
        super().__init__()
        self.delta_softplus = delta_softplus
        # allocate dt_bias
        self.register_buffer("dt_bias", torch.empty((d_inner), device=device, dtype=torch.int8))
        self.register_buffer("dt_bias_scale", torch.tensor(1.0, device=device, dtype=torch.float32))

        # allocate A_log
        self.register_buffer("A_log", torch.empty((d_inner, d_state), device=device, dtype=torch.int8))
        self.register_buffer("A_scale", torch.tensor(1.0, device=device, dtype=torch.float32))

        # allocate D "skip" parameter
        self.register_buffer("D", torch.empty((d_inner), device=device, dtype=torch.int8)) 
        self.register_buffer("D_scale", torch.tensor(1.0, device=device, dtype=torch.float32))

        # allocate scaling factors
        self.register_buffer('u_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('dt_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('B_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('C_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('z_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('ssm_state_scale', torch.tensor(1.0, device=device, dtype=torch.float32))

    @classmethod
    def from_fp16(cls, d_state, d_inner, A_log, D=None, dt_bias=None, delta_softplus=True,
                 ssm_state_scale=1.0, u_scale=1.0, dt_scale=1.0, B_scale=1.0, C_scale=1.0, z_scale=1.0):
        qsscan = cls(d_state, d_inner)

        A_log_quant, A_log_scale = quantize_weight_per_tensor_absmax(A_log, n_bits=8)
        A_log_quant = A_log_quant.to(torch.int8)
        qsscan.A_log = A_log_quant
        qsscan.A_scale = A_log_scale.float()

        if D is not None:
            D_quant, D_scale = quantize_weight_per_tensor_absmax(D, n_bits=8)
            D_quant = D_quant.to(torch.int8)
            qsscan.D = D_quant
            qsscan.D_scale = D_scale.float()
        else:
            qsscan.D = None
        
        if dt_bias is not None:
            dt_bias_quant, dt_bias_scale = quantize_weight_per_tensor_absmax(dt_bias, n_bits=8)
            dt_bias_quant = dt_bias_quant.to(torch.int8)
            qsscan.dt_bias = dt_bias_quant
            qsscan.dt_bias_scale = dt_bias_scale.float()
        else:
            qsscan.dt_bias = None
            qsscan.dt_bias_scale = torch.tensor(0.0)
        
        qsscan.u_scale = u_scale.float()
        qsscan.dt_scale = dt_scale.float()
        qsscan.B_scale = B_scale.float()
        qsscan.C_scale = C_scale.float()
        qsscan.z_scale = z_scale.float()
        qsscan.ssm_state_scale = ssm_state_scale.float()
        qsscan.delta_softplus = delta_softplus
        return qsscan

    #NOTE(HY): Only activate q_sscan when quamba is True,
    # since quantize_tensor_per_tensor_absmax only returns real scales when quamba is True.
    @torch.no_grad()
    def forward(self, u, dt, B, C, z=None, return_last_state=False):

        # q_sscan output int8 y
        y = quant_selective_scan_fn(
            u, self.u_scale[None],
            dt, self.dt_scale[None],
            self.A_log, self.A_scale[None], 
            B, self.B_scale[None],
            C, self.C_scale[None],
            ssm_state_scale=self.ssm_state_scale[None],
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

        y = quant_sscan_update_triton(
            ssm_state, u, self.u_scale[None],
            dt, self.dt_scale[None],
            self.A_log, self.A_scale[None],
            self.ssm_state_scale[None],
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
        self.u_scale = self.u_scale.to(*args, **kwargs)
        self.A_log = self.A_log.to(*args, **kwargs)
        self.A_scale = self.A_scale.to(*args, **kwargs)
        if self.D is not None:
            self.D = self.D.to(*args, **kwargs)
        if self.D_scale is not None:
            self.D_scale = self.D_scale.to(*args, **kwargs)
        if self.dt_bias is not None:
            self.dt_bias = self.dt_bias.to(*args, **kwargs)
        if self.dt_bias_scale is not None:
            self.dt_bias_scale = self.dt_bias_scale.to(*args, **kwargs)
        if self.B_scale is not None:
            self.B_scale = self.B_scale.to(*args, **kwargs)
        if self.C_scale is not None:
            self.C_scale = self.C_scale.to(*args, **kwargs)
        if self.z_scale is not None:
            self.z_scale = self.z_scale.to(*args, **kwargs)
        return self

    def __repr__(self):
        return f"QSScan()"