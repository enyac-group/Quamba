# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.ops.triton.softplus import softplus

@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_DT_BIAS_SCALE": lambda args: args["dt_bias_scale"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _quant_sscan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, x_scale, dt_ptr, dt_scale, dt_bias_ptr, dt_bias_scale, A_log_ptr, A_log_scale,
    state_scale, B_ptr, B_scale, C_ptr, C_scale, D_ptr, D_scale, z_ptr, z_scale, out_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_DT_BIAS_SCALE: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_log_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_log_ptrs = A_log_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    x = tl.load(x_scale) * tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    state_scale = tl.load(state_scale)
    state = state_scale * tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_scale) * tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            if HAS_DT_BIAS_SCALE:
                dt += tl.load(dt_bias_scale) * tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            else:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A_log = tl.load(A_log_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        A = -tl.exp(tl.load(A_log_scale) * A_log)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_scale) * tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = -tl.exp(tl.load(A_log_scale) * tl.load(A_log_ptr).to(tl.float32))
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_scale) * tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_scale) * tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_scale) * tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_scale) * tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        dB = B[None, :] * dt[:, None]
    else:
        dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    qstates = tl.clamp(tl.extra.cuda.libdevice.rint((state*1e2) / (state_scale*1e2)), -128, 127).to(tl.int8) # Triton 3.0.0 required
    tl.store(state_ptrs, qstates, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)
    

def quant_sscan_update_triton(state, q_x, x_scale, q_dt, dt_scale, q_A_log, A_log_scale, ssm_state_scale, q_B, B_scale, q_C, C_scale,
                              q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, dt_bias_scale=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate), torch.float16
        q_x: (batch, dim) or (batch, nheads, dim), torch.int8
        x_scales: (ngroups, nhead_groups, n_dim_groups), torch.float32
        x_head_group_range: (n_ssd_groups, n_head_groups), torch.int32
        x_dim_group_range: (n_ssd_groups, n_dim_groups), torch.int32
        q_dt: (batch, dim) or (batch, nheads, dim), torch.int8
        dt_scale: torch.float32
        q_A_log: (dim, dstate) or (nheads, dim, dstate), torch.int8
        A_log_scale: torch.float32
        q_B: (batch, dstate) or (batch, ngroups, dstate), torch.int8
        B_scale: torch.float32
        q_C: (batch, dstate) or (batch, ngroups, dstate), torch.int8
        C_scale: torch.float32
        q_D: (dim,) or (nheads, dim), torch.int8
        D_scale: torch.float32
        q_z: (batch, dim) or (batch, nheads, dim), torch.int8
        z_scale: torch.float32
        dt_bias: (dim,) or (nheads, dim), torch.float16
        dt_softplus: bool
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if q_x.dim() == 2:
        q_x = q_x.unsqueeze(1)
    if q_dt.dim() == 2:
        q_dt = q_dt.unsqueeze(1)
    if q_A_log.dim() == 2:
        q_A_log = q_A_log.unsqueeze(0)
    if q_B.dim() == 2:
        q_B = q_B.unsqueeze(1)
    if q_C.dim() == 2:
        q_C = q_C.unsqueeze(1)
    if q_D is not None and q_D.dim() == 1:
        q_D = q_D.unsqueeze(0)
    if q_z is not None and q_z.dim() == 2:
        q_z = q_z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    ngroups = q_B.shape[1]
    batch, nheads, dim, dstate = state.shape
    assert q_x.is_cuda
    assert q_x.shape == (batch, nheads, dim)
    assert q_dt.shape == q_x.shape
    assert q_A_log.shape == (nheads, dim, dstate)
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"

    assert B_scale.is_cuda
    assert B_scale.numel() == 1
    assert C_scale.is_cuda
    assert C_scale.numel() == 1
    assert q_B.shape == (batch, ngroups, dstate)
    assert q_C.shape == q_B.shape
    if q_D is not None:
        assert q_D.shape == (nheads, dim)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    out = torch.empty_like(q_x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = ((q_z.stride(0), q_z.stride(1), q_z.stride(2)) if q_z is not None else (0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((32, 4) if dstate <= 32 else
                                     ((32, 4) if dstate <= 64 else
                                      ((32, 4) if dstate <= 128 else
                                       ((16, 8))))))
    tie_hdim = q_A_log.stride(-1) == 0 and q_A_log.stride(-2) == 0 and q_dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(q_x.device.index):
        _quant_sscan_update_kernel[grid](
            state, q_x, x_scale, q_dt, dt_scale, dt_bias, dt_bias_scale, q_A_log, A_log_scale,
            ssm_state_scale, q_B, B_scale, q_C, C_scale, q_D, D_scale, q_z, z_scale, out,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            q_x.stride(0), q_x.stride(1), q_x.stride(2),
            q_dt.stride(0), q_dt.stride(1), q_dt.stride(2),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            q_A_log.stride(0), q_A_log.stride(1), q_A_log.stride(2),
            q_B.stride(0), q_B.stride(1), q_B.stride(2),
            q_C.stride(0), q_C.stride(1), q_C.stride(2),
            *(q_D.stride(0), q_D.stride(1)) if q_D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2],
            out.stride(0), out.stride(1), out.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out



@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _quamba2_sscan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, x_scales_ptr, x_head_group_range_ptr, x_dim_group_range_ptr,
    dt_ptr, dt_scale, dt_bias_ptr, A_log_ptr, A_log_scale, state_scale,
    B_ptr, B_scale, C_ptr, C_scale, D_ptr, D_scale, 
    z_ptr, z_scale, out_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_state_scale_group, stride_state_scale_head, stride_state_scale_dim, stride_state_scale_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # group quant paramters
    nhead_groups: tl.constexpr,
    ndim_groups: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_log_ptr += pid_h * stride_A_head
    B_scale_ptr = B_scale + (pid_h // nheads_ngroups_ratio)
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_scale_ptr = C_scale + (pid_h // nheads_ngroups_ratio)
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_log_ptrs = A_log_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # load x_head_group_range: [n_ssd_groups, n_head_groups]
    x_head_group_range_ptr = x_head_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups
    x_head_group_range = tl.load(x_head_group_range_ptr + tl.arange(0, nhead_groups))
    x_head_gidx = tl.sum(tl.where(pid_h % nheads_ngroups_ratio >= x_head_group_range, 1, 0))
    # load x_dim_group_range: [n_ssd_groups, n_head_groups, n_dim_groups]
    x_dim_group_range_ptr = x_dim_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_dim_group_range_ptr = x_dim_group_range_ptr + (x_head_gidx*ndim_groups)
    x_dim_group_range = tl.load(x_dim_group_range_ptr + tl.arange(0, ndim_groups))
    # load x_scales
    x_dim_gidx = tl.sum(tl.where(offs_m[:, None] >= x_dim_group_range[None, :], 1, 0), axis=-1)
    x_scales_ptr = x_scales_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_scales_ptrs = x_scales_ptr + (x_head_gidx*ndim_groups + x_dim_gidx)
    x_scales = tl.load(x_scales_ptrs)
    x = x_scales * tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    # load state_scale: [n_ssd_groups, n_head_groups, n_dim_groups, dstate]
    state_scale_ptr = state_scale + (pid_h // nheads_ngroups_ratio) * stride_state_scale_group
    state_scale_ptr = state_scale_ptr + x_head_gidx * stride_state_scale_head
    state_scale_ptrs = state_scale_ptr + (x_dim_gidx[:, None] * stride_state_scale_dim + offs_n[None, :] * stride_state_scale_dstate)
    state_scale_load = tl.load(state_scale_ptrs, mask=(x_dim_gidx[:, None] < ndim_groups) & (offs_n[None, :] < dstate), other=0.0)
    state = state_scale_load * tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_scale) * tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A_log = tl.load(A_log_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        A = -tl.exp(tl.load(A_log_scale) * A_log)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_scale) * tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = -tl.exp(tl.load(A_log_scale) * tl.load(A_log_ptr).to(tl.float32))
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_scale_ptr) * tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_scale_ptr) * tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_scale) * tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_scale) * tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        dB = B[None, :] * dt[:, None]
    else:
        dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    # !!!!!!!!!!!!!! This is important !!!!!!!!!!!!!! Triton division seems to be not numerical stable
    qstates = tl.clamp(tl.extra.cuda.libdevice.rint((state*1e6) / (state_scale_load*1e6)), -128, 127) # Triton 3.0.0 required
    tl.store(state_ptrs, qstates, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)
    

def quamba2_sscan_update_triton(state, q_x, x_scales, x_head_group_range, x_dim_group_range,
                                q_dt, dt_scale, q_A_log, A_log_scale, ssm_state_scale, q_B, B_scale, q_C, C_scale,
                                q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate), torch.float16
        q_x: (batch, dim) or (batch, nheads, dim), torch.int8
        x_scales: (ngroups, nhead_groups, n_dim_groups), torch.float32
        x_head_group_range: (n_ssd_groups, n_head_groups), torch.int32
        x_dim_group_range: (n_ssd_groups, n_dim_groups), torch.int32
        q_dt: (batch, dim) or (batch, nheads, dim), torch.int8
        dt_scale: torch.float32
        q_A_log: (dim, dstate) or (nheads, dim, dstate), torch.int8
        A_log_scale: torch.float32
        q_B: (batch, dstate) or (batch, ngroups, dstate), torch.int8
        B_scale: torch.float32
        q_C: (batch, dstate) or (batch, ngroups, dstate), torch.int8
        C_scale: torch.float32
        q_D: (dim,) or (nheads, dim), torch.int8
        D_scale: torch.float32
        q_z: (batch, dim) or (batch, nheads, dim), torch.int8
        z_scale: torch.float32
        dt_bias: (dim,) or (nheads, dim), torch.float16
        dt_softplus: bool
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if q_x.dim() == 2:
        q_x = q_x.unsqueeze(1)
    if q_dt.dim() == 2:
        q_dt = q_dt.unsqueeze(1)
    if q_A_log.dim() == 2:
        q_A_log = q_A_log.unsqueeze(0)
    if q_B.dim() == 2:
        q_B = q_B.unsqueeze(1)
    if q_C.dim() == 2:
        q_C = q_C.unsqueeze(1)
    if q_D is not None and q_D.dim() == 1:
        q_D = q_D.unsqueeze(0)
    if q_z is not None and q_z.dim() == 2:
        q_z = q_z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    ngroups = q_B.shape[1]
    batch, nheads, dim, dstate = state.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1]  # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2]    # [n_ssd_groups, n_head_groups, n_dim_groups]
    assert q_x.is_cuda
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups*ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32

    assert ssm_state_scale.is_cuda
    assert ssm_state_scale.dtype == torch.float32
    assert ssm_state_scale.shape == (ngroups, nhead_groups, ndim_groups, dstate)
    assert q_x.shape == (batch, nheads, dim)
    assert q_dt.shape == q_x.shape
    assert q_A_log.shape == (nheads, dim, dstate)
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert q_B.shape == (batch, ngroups, dstate)
    assert q_C.shape == q_B.shape
    if q_D is not None:
        assert q_D.shape == (nheads, dim)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    out = torch.empty_like(q_x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = ((q_z.stride(0), q_z.stride(1), q_z.stride(2)) if q_z is not None else (0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((32, 4) if dstate <= 32 else
                                     ((32, 4) if dstate <= 64 else
                                      ((32, 4) if dstate <= 128 else
                                       ((16, 8))))))
    tie_hdim = q_A_log.stride(-1) == 0 and q_A_log.stride(-2) == 0 and q_dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(q_x.device.index):
        _quamba2_sscan_update_kernel[grid](
            state, q_x, x_scales, x_head_group_range, x_dim_group_range,
            q_dt, dt_scale, dt_bias, q_A_log, A_log_scale, ssm_state_scale,
            q_B, B_scale, q_C, C_scale, q_D, D_scale, q_z, z_scale, out,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            ssm_state_scale.stride(0), ssm_state_scale.stride(1), ssm_state_scale.stride(2), ssm_state_scale.stride(3),
            q_x.stride(0), q_x.stride(1), q_x.stride(2),
            q_dt.stride(0), q_dt.stride(1), q_dt.stride(2),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            q_A_log.stride(0), q_A_log.stride(1), q_A_log.stride(2),
            q_B.stride(0), q_B.stride(1), q_B.stride(2),
            q_C.stride(0), q_C.stride(1), q_C.stride(2),
            *(q_D.stride(0), q_D.stride(1)) if q_D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2],
            out.stride(0), out.stride(1), out.stride(2),
            nhead_groups,
            ndim_groups,
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, "b h d -> b h d 1"))  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out
