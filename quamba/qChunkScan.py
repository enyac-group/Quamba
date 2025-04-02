"""
The code is modfied from
https://github.com/state-spaces/mamba
"""
import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from quamba.quant_utils import quantize_tensor_per_tensor_absmax
from quamba.triton.selective_state_update import quant_sscan_update_triton, quamba2_sscan_update_triton
from quamba.triton.quant_ssd_combined import _quant_mamba_chunk_scan_combined_fwd, _quamba2_mamba_chunk_scan_combined_fwd


class Quamba2ChunkScan(nn.Module):

    def __init__(self, d_ssm, headdim, d_state, ngroups, D_has_hdim, chunk_size,
                 nhead_groups=0, ndim_groups=0, delta_softplus=True,
                 quant_dt=True, quant_B=True, quant_C=True, quant_z=False,
                 dt_limit=(0.0, float("inf")), device=None, **kwargs):
        factory_kwargs = {"device": "cuda"} # we use partial function, so it is a hardcode for putting everything on cuda
        # factory_kwargs = {"device": device}
        super().__init__()

        self.d_ssm = d_ssm
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        self.D_has_hdim = D_has_hdim
        self.chunk_size = chunk_size
        self.nhead_groups = nhead_groups
        self.ndim_groups = ndim_groups
        self.delta_softplus = delta_softplus
        self.dt_limit = dt_limit

        nheads = d_ssm // headdim
        # create space for dt bias
        self.register_buffer('dt_bias', torch.empty(nheads, dtype=torch.float16, **factory_kwargs)) # we do not quant dt_bias
        # create space for A
        self.register_buffer('A_log', torch.empty(nheads, dtype=torch.int8, **factory_kwargs))
        self.register_buffer('A_log_scale', torch.empty([], dtype=torch.float32, **factory_kwargs)) # no-shape
        # create space for D
        self.register_buffer('D', torch.empty(
            self.d_ssm if self.D_has_hdim else nheads,
            dtype=torch.int8, **factory_kwargs))
        self.register_buffer('D_scale', torch.empty([], dtype=torch.float32, **factory_kwargs)) # no-shape

        # Create space for scales
        quant_params = {'dt': quant_dt, 'z': quant_z}
        for name, quant_flag in quant_params.items():
            if quant_flag:
                self.register_buffer(f'{name}_scale', torch.empty([], dtype=torch.float32, **factory_kwargs))
            else:
                setattr(self, f'{name}_scale', None)

        if nhead_groups > 0 and ndim_groups > 0:
            # create space for SSM state scale
            self.register_buffer('ssm_state_scale', torch.empty(
                [self.ngroups, self.nhead_groups, self.ndim_groups, self.d_state],
                dtype=torch.float32, **factory_kwargs))
            self.register_buffer('x_scales', torch.empty(
                [ngroups, nhead_groups, ndim_groups], dtype=torch.float32, **factory_kwargs))
            self.register_buffer('x_head_group_range', torch.empty(
                [ngroups, nhead_groups], dtype=torch.int32, **factory_kwargs))
            self.register_buffer('x_dim_group_range', torch.empty(
                [ngroups, nhead_groups, ndim_groups], dtype=torch.int32, **factory_kwargs))
            quant_params = {'B': quant_B, 'C': quant_C}
            for name, quant_flag in quant_params.items():
                if quant_flag:
                    self.register_buffer(f'{name}_scale', torch.empty([ngroups], dtype=torch.float32, **factory_kwargs))
                else:
                    setattr(self, f'{name}_scale', None)
        elif nhead_groups ==0 and ndim_groups == 0:
            # create space for SSM state scale
            self.register_buffer('ssm_state_scale', torch.empty([1], dtype=torch.float32, **factory_kwargs))
            # scales must be on cuda for triton kernels
            self.register_buffer('x_scales', torch.empty([1], dtype=torch.float32, **factory_kwargs))
            self.x_head_group_range = None
            self.x_dim_group_range = None
            quant_params = {'B': quant_B, 'C': quant_C}
            for name, quant_flag in quant_params.items():
                if quant_flag:
                    self.register_buffer(f'{name}_scale', torch.empty([1], dtype=torch.float32, **factory_kwargs))
                else:
                    setattr(self, f'{name}_scale', None)
        else:
            raise ValueError("nhead_groups and ndim_groups must be both 0 or both not 0")

        # setup chunk_scan fwd and update functions
        self.set_chunk_scan_fn()

    @classmethod
    def from_fp16(cls, d_ssm, headdim, d_state, ngroups,
                 x_scales, x_head_group_range, x_dim_group_range,
                 A_log, chunk_size, ssm_state_scale, D=None, D_has_hdim=False, dt_bias=None, delta_softplus=True,
                 dt_scale=None, B_scale=None, C_scale=None, z_scale=None, dt_limit=(0.0, float("inf"))):

        nhead_groups = x_head_group_range.shape[1] if x_head_group_range is not None else 0
        ndim_groups = x_dim_group_range.shape[1] if x_dim_group_range is not None else 0
        qchunkscan = cls(d_ssm, headdim, d_state, ngroups, D_has_hdim,
                         chunk_size, nhead_groups, ndim_groups, delta_softplus, dt_limit)

        A_log_quant, A_log_scale = quantize_tensor_per_tensor_absmax(A_log, n_bits=8)
        qchunkscan.A_log = A_log_quant.to(torch.int8)
        qchunkscan.A_log_scale = A_log_scale.float().to(A_log.device)

        if x_head_group_range is not None and x_dim_group_range is not None:
            qchunkscan.ssm_state_scale = cls.get_ssm_state_scales(
                ssm_state_scale, ngroups,
                x_head_group_range.shape[1], x_dim_group_range.shape[2],
                d_state, A_log.device)
        else:
            qchunkscan.ssm_state_scale = ssm_state_scale.float().to(A_log.device)

        if D is not None:
            # rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            D_quant, D_scale = quantize_tensor_per_tensor_absmax(D, n_bits=8)
            qchunkscan.D = D_quant.to(torch.int8)
            qchunkscan.D_scale = D_scale.float().to(D.device)
        else:
            qchunkscan.D = None
        
        if dt_bias is not None:
            qchunkscan.dt_bias = dt_bias.to(torch.float16) # we do not quant dt_bias
        else:
            qchunkscan.dt_bias = None  

        # use scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        scales = {'dt_scale': dt_scale, 'B_scale': B_scale, 'C_scale': C_scale, 'z_scale': z_scale}
        for name, scale in scales.items():
            setattr(qchunkscan, name, scale.float() if scale is not None else None)  # Move to CUDA and convert to float

        qchunkscan.x_scales = x_scales.to(A_log.device)
        qchunkscan.x_head_group_range = None if x_head_group_range is None else x_head_group_range.to(A_log.device)
        qchunkscan.x_dim_group_range = None if x_dim_group_range is None else x_dim_group_range.to(A_log.device)
        qchunkscan.set_chunk_scan_fn()

        return qchunkscan

    @staticmethod
    def get_ssm_state_scales(scales, ngroups, n_head_groups, n_dim_groups, dstates, device):
        # out scales shape: [ngroups, n_head_groups, n_dim_groups, dstates]
        if isinstance(scales, list):
            out_scales = torch.empty((ngroups, n_head_groups, n_dim_groups, dstates), device=device)
            for ssd_g in range(ngroups):
                for idx, (h_gsize, ch_gsize, ch_scales) in enumerate(scales[ssd_g]):
                    # h_gsize: int, ch_gsize: List[int], ch_scales: List[float]
                    # ch_scales is a tensor with shape: [4, 128]
                    out_scales[ssd_g, idx, :, :] = ch_scales.to(device)
            return out_scales
        else:
            return scales.to(device)

    def set_chunk_scan_fn(self):

        if self.x_head_group_range is not None and self.x_dim_group_range is not None:            
            # get chunk_scan_combined_fwd
            # scales must be on cuda before using partial for triton kernels
            self.chunk_scan_combined_fwd = partial(
                _quamba2_mamba_chunk_scan_combined_fwd,
                    x_scales=self.x_scales, x_head_group_range=self.x_head_group_range,
                    x_dim_group_range=self.x_dim_group_range, dt_scale=self.dt_scale,
                    q_A_log=self.A_log, A_log_scale=self.A_log_scale,
                    ssm_state_scale=self.ssm_state_scale,
                    B_scale=self.B_scale, C_scale=self.C_scale,
                    q_D=self.D, D_scale=self.D_scale,
                    chunk_size=self.chunk_size, dt_bias=self.dt_bias,
                    initial_states=None, seq_idx=None,
                    cu_seqlens=None, dt_softplus=self.delta_softplus,
                    dt_limit=self.dt_limit, mm_dtype=torch.float16)

            # get chunk_scan_combined_update
            # scales must be on cuda before using partial for triton kernels
            self.chunk_scan_combined_update = partial(
                quamba2_sscan_update_triton,
                    x_scales=self.x_scales, x_head_group_range=self.x_head_group_range,
                    x_dim_group_range=self.x_dim_group_range, dt_scale=self.dt_scale,
                    q_A_log=repeat(self.A_log, "h -> h p n", p=self.headdim, n=self.d_state),
                    A_log_scale=self.A_log_scale,
                    ssm_state_scale=self.ssm_state_scale,
                    B_scale=self.B_scale, C_scale=self.C_scale,
                    q_D=repeat(self.D, "h -> h p", p=self.headdim) if self.D is not None else None,
                    D_scale=self.D_scale if self.D is not None else None,
                    dt_bias=repeat(self.dt_bias, "h -> h p", p=self.headdim) if self.dt_bias is not None else None,
                    dt_softplus=self.delta_softplus)
        elif self.x_head_group_range is None and self.x_dim_group_range is None:
            # scales must be on cuda for triton kernels
            self.x_scales = self.x_scales.float()
            self.x_head_group_range = None
            self.x_dim_group_range = None

            # get chunk_scan_combined_fwd
            # scales must be on cuda before using partial for triton kernels
            self.chunk_scan_combined_fwd = partial(
                _quant_mamba_chunk_scan_combined_fwd,
                    x_scale=self.x_scales, dt_scale=self.dt_scale,
                    q_A_log=self.A_log, A_log_scale=self.A_log_scale,
                    ssm_state_scale=self.ssm_state_scale,
                    B_scale=self.B_scale, C_scale=self.C_scale,
                    q_D=self.D, D_scale=self.D_scale,
                    chunk_size=self.chunk_size, dt_bias=self.dt_bias,
                    initial_states=None, seq_idx=None,
                    cu_seqlens=None, dt_softplus=self.delta_softplus,
                    dt_limit=self.dt_limit, mm_dtype=torch.float16)
            
            # get chunk_scan_combined_update
            # scales must be on cuda before using partial for triton kernels
            self.chunk_scan_combined_update = partial(
                quant_sscan_update_triton,
                    x_scale=self.x_scales, dt_scale=self.dt_scale,
                    q_A_log=repeat(self.A_log, "h -> h p n", p=self.headdim, n=self.d_state),
                    A_log_scale=self.A_log_scale,
                    ssm_state_scale=self.ssm_state_scale,
                    B_scale=self.B_scale, C_scale=self.C_scale,
                    q_D=repeat(self.D, "h -> h p", p=self.headdim) if self.D is not None else None,
                    D_scale=self.D_scale if self.D is not None else None,
                    dt_bias=repeat(self.dt_bias, "h -> h p", p=self.headdim) if self.dt_bias is not None else None,
                    dt_softplus=self.delta_softplus)
        else:
            raise ValueError("x_head_group_range and x_dim_group_range must be both None or both not None")

    @torch.no_grad()
    def forward(self, x, dt, B, C, z=None, return_final_states=False):
        # output y is fp16
        y, final_states = self.chunk_scan_combined_fwd(
            q_x=x, q_dt=dt, q_B=B, q_C=C, q_z=z,
            z_scale=self.z_scale.cuda() if z is not None else None)
        return y if not return_final_states else (y, final_states)

    @torch.no_grad()
    def update(self, ssm_state, x, dt, B, C, z=None):
        dt = repeat(dt, "b h -> b h p", p=self.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        y = self.chunk_scan_combined_update(
            state=ssm_state, q_x=x_reshaped, q_dt=dt, q_B=B, q_C=C, 
            q_z=z if z is not None else None,
            z_scale=self.z_scale.cuda() if z is not None else None,
        )
        return y

    def __repr__(self):
        return f"QChunkScan()"