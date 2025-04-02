import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _quant_chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, x_scale, b_ptr, b_scale, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    USE_FLOA32_MM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    if USE_FLOA32_MM:
        mm_dtype = tl.float32
    else:
        mm_dtype = tl.float16
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # scaling factors are float32 
        x = tl.load(x_scale) * tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_scale) * tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(mm_dtype)
        x = x.to(mm_dtype)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _quant_chunk_state_fwd(
        q_B, B_scale, q_x, x_scale,
        dt, dA_cumsum, mm_dtype=torch.float16,
        seq_idx=None, states=None, states_in_fp32=True
    ):
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert q_x.is_cuda
    assert nheads % ngroups == 0
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert mm_dtype is torch.float16 or mm_dtype is torch.float32
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else torch.float16
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=q_x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    with torch.cuda.device(q_x.device.index):
        _quant_chunk_state_fwd_kernel[grid](
            q_x, x_scale, q_B, B_scale, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),
            q_B.stride(0), q_B.stride(1), q_B.stride(2), q_B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            USE_FLOA32_MM = mm_dtype is torch.float32,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _quamba2_chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, x_scales_ptr, x_head_group_range_ptr, x_dim_group_range_ptr,
    b_ptr, b_scale, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # group quant paramters
    nhead_groups: tl.constexpr,
    ndim_groups: tl.constexpr,
    # Meta-parameters
    USE_FLOA32_MM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_scale_ptr = b_scale + (pid_h // nheads_ngroups_ratio)
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # load x_head_group_range: [n_ssd_groups, n_head_groups]
    x_head_group_range_ptr = x_head_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups
    x_head_group_range = tl.load(x_head_group_range_ptr + tl.arange(0, nhead_groups))
    x_head_gidx = tl.sum(tl.where(pid_h % nheads_ngroups_ratio >= x_head_group_range, 1, 0))
    # load x_dim_group_range: [n_ssd_groups, n_head_groups, n_dim_groups]
    x_dim_group_range_ptr = x_dim_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_dim_group_range_ptr = x_dim_group_range_ptr + x_head_gidx*ndim_groups
    x_dim_group_range = tl.load(x_dim_group_range_ptr + tl.arange(0, ndim_groups))
    # load x_scales
    x_dim_gidx = tl.sum(tl.where(offs_m[:, None] >= x_dim_group_range[None, :], 1, 0), axis=-1)
    x_scales_ptr = x_scales_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_scales_ptrs = x_scales_ptr + (x_head_gidx*ndim_groups + x_dim_gidx)
    x_scales = tl.load(x_scales_ptrs)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    if USE_FLOA32_MM:
        mm_dtype = tl.float32
    else:
        mm_dtype = tl.float16
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # scaling factors are float32 
        x = x_scales[:, None] * tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_scale_ptr) * tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(mm_dtype)
        x = x.to(mm_dtype)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _quamba2_chunk_state_fwd(
        q_B, B_scale, q_x, x_scales, x_head_group_range, x_dim_group_range,
        dt, dA_cumsum, mm_dtype=torch.float16,
        seq_idx=None, states=None, states_in_fp32=True
    ):
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1] # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2] # [n_ssd_groups, n_dim_groups]
    assert q_x.is_cuda
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups * ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32
    assert nheads % ngroups == 0

    assert B_scale.is_cuda
    assert B_scale.numel() == ngroups
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert mm_dtype is torch.float16 or mm_dtype is torch.float32
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else torch.float16
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=q_x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    with torch.cuda.device(q_x.device.index):
        _quamba2_chunk_state_fwd_kernel[grid](
            q_x, x_scales, x_head_group_range, x_dim_group_range,
            q_B, B_scale, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),
            q_B.stride(0), q_B.stride(1), q_B.stride(2), q_B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            nhead_groups=nhead_groups,
            ndim_groups=ndim_groups,
            USE_FLOA32_MM = mm_dtype is torch.float32,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states