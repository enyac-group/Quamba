
import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.ops.triton.softplus import softplus


@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _quant_quant_ssm_states_kernel(
    # Pointers to matrices
    state_out_ptr, state_ptr, state_scale,
    # Matrix dimensions
    batch, nheads, dim, dstate,
    # Strides
    stride_state_out_batch, stride_state_out_head, stride_state_out_dim, stride_state_out_dstate,
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_out_ptr += pid_b * stride_state_out_batch + pid_h * stride_state_out_head
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_out_ptrs = state_out_ptr + (offs_m[:, None] * stride_state_out_dim + offs_n[None, :] * stride_state_out_dstate)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)

    state_scale_load = tl.load(state_scale)
    # load fp16 states
    states = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    # quantize fp16 states to int8
    qstates = tl.clamp(tl.extra.cuda.libdevice.rint((states*1e2) / (state_scale_load*1e2)), -128, 127).to(tl.int8) # Triton 3.0.0 required
    # store int8 states
    tl.store(state_out_ptrs, qstates, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    

def _quant_quant_ssm_states(state_fp16, ssm_state_scale):
    batch, nheads, dim, dstate = state_fp16.shape
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    state_int8 = torch.empty_like(state_fp16, dtype=torch.int8)
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((32, 4) if dstate <= 32 else
                                     ((32, 4) if dstate <= 64 else
                                      ((32, 4) if dstate <= 128 else
                                       ((16, 8))))))
    with torch.cuda.device(state_fp16.device.index):
        _quant_quant_ssm_states_kernel[grid](
            state_int8, state_fp16, ssm_state_scale,
            batch, nheads, dim, dstate,
            state_int8.stride(0), state_int8.stride(1), state_int8.stride(2), state_int8.stride(3),
            state_fp16.stride(0), state_fp16.stride(1), state_fp16.stride(2), state_fp16.stride(3),
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    return state_int8



@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _quamba2_quant_ssm_states_kernel(
    # Pointers to matrices
    state_out_ptr, state_ptr, x_head_group_range_ptr, x_dim_group_range_ptr, state_scale,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_out_batch, stride_state_out_head, stride_state_out_dim, stride_state_out_dstate,
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_state_scale_group, stride_state_scale_head, stride_state_scale_dim, stride_state_scale_dstate,
    # group quant paramters
    nhead_groups: tl.constexpr,
    ndim_groups: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_out_ptr += pid_b * stride_state_out_batch + pid_h * stride_state_out_head
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_out_ptrs = state_out_ptr + (offs_m[:, None] * stride_state_out_dim + offs_n[None, :] * stride_state_out_dstate)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)

    # load x_head_group_range: [n_ssd_groups, n_head_groups]
    x_head_group_range_ptr = x_head_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups
    x_head_group_range = tl.load(x_head_group_range_ptr + tl.arange(0, nhead_groups))
    x_head_gidx = tl.sum(tl.where(pid_h % nheads_ngroups_ratio >= x_head_group_range, 1, 0))
    # load x_dim_group_range: [n_ssd_groups, n_head_groups, n_dim_groups]
    x_dim_group_range_ptr = x_dim_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_dim_group_range_ptr = x_dim_group_range_ptr + (x_head_gidx*ndim_groups)
    x_dim_group_range = tl.load(x_dim_group_range_ptr + tl.arange(0, ndim_groups))
    x_dim_gidx = tl.sum(tl.where(offs_m[:, None] >= x_dim_group_range[None, :], 1, 0), axis=-1)

    # load state_scale: [n_ssd_groups, n_head_groups, n_dim_groups, dstate]
    state_scale_ptr = state_scale + (pid_h // nheads_ngroups_ratio) * stride_state_scale_group
    state_scale_ptr = state_scale_ptr + x_head_gidx * stride_state_scale_head
    state_scale_ptrs = state_scale_ptr + (x_dim_gidx[:, None] * stride_state_scale_dim + offs_n[None, :] * stride_state_scale_dstate)
    state_scale_load = tl.load(state_scale_ptrs, mask=(x_dim_gidx[:, None] < ndim_groups) & (offs_n[None, :] < dstate), other=0.0)
    # load fp16 states
    states = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    # quantize fp16 states to int8
    qstates = tl.clamp(tl.extra.cuda.libdevice.rint((states*1e6) / (state_scale_load*1e6)), -128, 127).to(tl.int8) # Triton 3.0.0 required
    # store int8 states
    tl.store(state_out_ptrs, qstates, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    

def _quamba2_quant_ssm_states(state_fp16, x_head_group_range, x_dim_group_range, ssm_state_scale):

    batch, nheads, dim, dstate = state_fp16.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    ngroups = x_head_group_range.shape[0]  # [n_ssd_groups, n_head_groups]
    nhead_groups = x_head_group_range.shape[1]  # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2]    # [n_ssd_groups, n_head_groups, n_dim_groups]
    assert ssm_state_scale.shape[0] == ngroups, "ssm_state_scale must have shape [n_ssd_group, x_nhead_group, n_dim_group, dstate]"
    assert ssm_state_scale.shape[1] == nhead_groups, "ssm_state_scale must have shape [n_ssd_group, x_nhead_group, n_dim_group, dstate]"
    assert ssm_state_scale.shape[2] == ndim_groups, "ssm_state_scale must have shape [n_ssd_group, x_nhead_group, n_dim_group, dstate]"
    assert ssm_state_scale.shape[3] == dstate, "ssm_state_scale must have shape [n_ssd_group, x_nhead_group, n_dim_group, dstate]"
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    state_int8 = torch.empty_like(state_fp16, dtype=torch.int8)
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((32, 4) if dstate <= 32 else
                                     ((32, 4) if dstate <= 64 else
                                      ((32, 4) if dstate <= 128 else
                                       ((16, 8))))))
    with torch.cuda.device(state_fp16.device.index):
        _quamba2_quant_ssm_states_kernel[grid](
            state_int8, state_fp16, x_head_group_range, x_dim_group_range, ssm_state_scale,
            batch, nheads, dim, dstate, nheads // ngroups,
            state_int8.stride(0), state_int8.stride(1), state_int8.stride(2), state_int8.stride(3),
            state_fp16.stride(0), state_fp16.stride(1), state_fp16.stride(2), state_fp16.stride(3),
            ssm_state_scale.stride(0), ssm_state_scale.stride(1), ssm_state_scale.stride(2), ssm_state_scale.stride(3),
            nhead_groups,
            ndim_groups,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    return state_int8
