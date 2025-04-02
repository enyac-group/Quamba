from packaging import version

import torch

import triton
import triton.language as tl

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _quant_chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr, x_ptr, x_scale, z_ptr, z_scale, out_ptr, out_x_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, C_ptr, C_scale, prev_states_ptr, D_ptr, D_scale,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    USE_FLOA32_MM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_FLOA32_MM:
        mm_dtype = tl.float32
    else:
        mm_dtype = tl.float16
    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or pid_c > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_scale) * tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(mm_dtype)
            C = C.to(mm_dtype)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_scale) * tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(mm_dtype)
                C = C.to(mm_dtype)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    # move CB pointer
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    # move x pointer
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # move dt pointer
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    # move dA_cumsum pointer
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        x = tl.load(x_scale) * tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        cb = cb.to(mm_dtype)
        x = x.to(mm_dtype)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_scale) * tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_scale) * tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


def _quant_chunk_scan_fwd(
        cb, q_x, x_scale,
        dt, dA_cumsum, q_C, C_scale, states,
        q_D=None, D_scale=None, q_z=None, z_scale=None,
        seq_idx=None, mm_dtype=torch.float16
    ):
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_C.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert q_x.is_cuda
    assert nheads % ngroups == 0
    assert mm_dtype is torch.float16 or mm_dtype is torch.float32
    assert q_C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=q_x.device, dtype=torch.float16)
    if q_z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=q_x.device, dtype=torch.float16)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    z_strides = ((q_z.stride(0), q_z.stride(1), q_z.stride(2), q_z.stride(3))
                  if q_z is not None else (0, 0, 0, 0))
    _quant_chunk_scan_fwd_kernel[grid](
        cb, q_x, x_scale, q_z, z_scale,
        out, out_x, dt, dA_cumsum, seq_idx,
        q_C, C_scale, states, q_D, D_scale,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        q_C.stride(0), q_C.stride(1), q_C.stride(2), q_C.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        q_D.stride(0) if q_D is not None else 0,
        IS_CAUSAL=True,
        HAS_D=q_D is not None,
        D_HAS_HDIM=q_D.dim() == 2 if q_D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=q_z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        USE_FLOA32_MM = mm_dtype is torch.float32,
        IS_TRITON_22=TRITON_22,
    )
    return out, out_x



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _quamba2_chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr, x_ptr, x_scales_ptr, x_head_group_range_ptr, x_dim_group_range_ptr, z_ptr, z_scale, 
    out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, C_scale, prev_states_ptr, D_ptr, D_scale,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    # group quant paramters
    nhead_groups: tl.constexpr,
    ndim_groups: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    USE_FLOA32_MM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_scale_ptr = C_scale + (pid_h // nheads_ngroups_ratio)
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_FLOA32_MM:
        mm_dtype = tl.float32
    else:
        mm_dtype = tl.float16
    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or pid_c > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_scale_ptr) * tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(mm_dtype)
            C = C.to(mm_dtype)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_scale_ptr) * tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(mm_dtype)
                C = C.to(mm_dtype)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    # move CB pointer
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    # load x_head_group_range: [n_ssd_groups, n_head_groups]
    x_head_group_range_ptr = x_head_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups
    x_head_group_range = tl.load(x_head_group_range_ptr + tl.arange(0, nhead_groups))
    x_head_gidx = tl.sum(tl.where(pid_h % nheads_ngroups_ratio >= x_head_group_range, 1, 0))
    # load x_dim_group_range: [n_ssd_groups, n_head_groups, n_dim_groups]
    x_dim_group_range_ptr = x_dim_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_dim_group_range_ptr = x_dim_group_range_ptr + x_head_gidx*ndim_groups
    x_dim_group_range = tl.load(x_dim_group_range_ptr + tl.arange(0, ndim_groups))
    # load x_scales: [n_ssd_groups, n_head_groups, n_dim_groups]
    x_dim_gidx = tl.sum(tl.where(offs_n[:, None] >= x_dim_group_range[None, :], 1, 0), axis=-1)
    x_scales_ptr = x_scales_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_scales_ptrs = x_scales_ptr + (x_head_gidx*ndim_groups + x_dim_gidx)
    x_scales = tl.load(x_scales_ptrs) 
    # move x pointer
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # move dt pointer
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    # move dA_cumsum pointer
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        x = x_scales * tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        cb = cb.to(mm_dtype)
        x = x.to(mm_dtype)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = x_scales * tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_scale) * tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))




def _quamba2_chunk_scan_fwd(
        cb, q_x, x_scales, x_head_group_range, x_dim_group_range,
        dt, dA_cumsum, q_C, C_scale, states,
        q_D=None, D_scale=None, q_z=None, z_scale=None,
        seq_idx=None, mm_dtype=torch.float16
    ):
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_C.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1] # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2] # [n_ssd_groups, n_head_groups, n_dim_groups]

    assert C_scale.is_cuda
    assert C_scale.numel() == ngroups
    assert q_x.is_cuda
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups * ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32
    assert nheads % ngroups == 0
    assert mm_dtype is torch.float16 or mm_dtype is torch.float32
    assert q_C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=q_x.device, dtype=torch.float16)
    if q_z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=q_x.device, dtype=torch.float16)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    z_strides = ((q_z.stride(0), q_z.stride(1), q_z.stride(2), q_z.stride(3))
                  if q_z is not None else (0, 0, 0, 0))
    _quamba2_chunk_scan_fwd_kernel[grid](cb,
        q_x, x_scales, x_head_group_range, x_dim_group_range,
        q_z, z_scale, out, out_x, dt, dA_cumsum, seq_idx,
        q_C, C_scale, states, q_D, D_scale,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        q_C.stride(0), q_C.stride(1), q_C.stride(2), q_C.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        q_D.stride(0) if q_D is not None else 0,
        nhead_groups=nhead_groups,
        ndim_groups=ndim_groups,
        IS_CAUSAL=True,
        HAS_D=q_D is not None,
        D_HAS_HDIM=q_D.dim() == 2 if q_D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=q_z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        USE_FLOA32_MM = mm_dtype is torch.float32,
        IS_TRITON_22=TRITON_22,
    )
    return out, out_x