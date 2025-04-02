import torch
from einops import rearrange, repeat

from quamba.triton.quant_chunk_cumsum import _quant_chunk_cumsum_fwd
from quamba.triton.quant_state_passing import _quant_state_passing_fwd
from quamba.triton.quant_chunk_state import _quant_chunk_state_fwd, _quamba2_chunk_state_fwd
from quamba.triton.quant_chunk_scan import _quant_chunk_scan_fwd, _quamba2_chunk_scan_fwd
from quamba.triton.quant_bmm_chunk import _quant_bmm_chunk_fwd, _quamba2_bmm_chunk_fwd
from quamba.triton.quant_ssm_states import _quant_quant_ssm_states, _quamba2_quant_ssm_states

def _quant_mamba_chunk_scan_combined_fwd(
        q_x, x_scale, q_dt, dt_scale, q_A_log, A_log_scale,
        q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape

    assert x_scale.is_cuda
    assert x_scale.numel() == 1
    assert B_scale.is_cuda
    assert B_scale.numel() == 1
    assert C_scale.is_cuda
    assert C_scale.numel() == 1

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.dtype == torch.int8
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _quant_chunk_state_fwd(q_B, B_scale, q_x, x_scale, dt, dA_cumsum, mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = _quant_state_passing_fwd(
                                rearrange(states, "... p n -> ... (p n)"),
                                dA_cumsum[:, :, :, -1],
                                initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                    if initial_states is not None else None,
                                seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                            )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    CB = _quant_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _quant_chunk_scan_fwd(
        CB, q_x, x_scale, dt, dA_cumsum, q_C, C_scale, states,
        q_D=q_D, D_scale=D_scale, q_z=q_z, z_scale=z_scale,
        seq_idx=seq_idx, mm_dtype=torch.float16
    )
    final_states = _quant_quant_ssm_states(final_states, ssm_state_scale)
    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")


def _quamba2_mamba_chunk_scan_combined_fwd(
        q_x, x_scales, x_head_group_range, x_dim_group_range,
        q_dt, dt_scale, q_A_log, A_log_scale, q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1] # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2] # [n_ssd_groups, n_head_groups, n_dim_groups]
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups*ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32

    assert B_scale.is_cuda
    assert B_scale.numel() == ngroups
    assert C_scale.is_cuda
    assert C_scale.numel() == ngroups

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.dtype == torch.int8
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    assert ssm_state_scale.is_cuda
    assert ssm_state_scale.dtype == torch.float32
    assert ssm_state_scale.shape == (ngroups, nhead_groups, ndim_groups, dstate)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _quamba2_chunk_state_fwd(q_B, B_scale, q_x, x_scales, x_head_group_range, x_dim_group_range, dt, dA_cumsum, mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = _quant_state_passing_fwd(
                                rearrange(states, "... p n -> ... (p n)"),
                                dA_cumsum[:, :, :, -1],
                                initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                    if initial_states is not None else None,
                                seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                            )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    CB = _quamba2_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _quamba2_chunk_scan_fwd(
        CB, q_x, x_scales, x_head_group_range, x_dim_group_range, dt, dA_cumsum, q_C, C_scale, states,
        q_D=q_D, D_scale=D_scale, q_z=q_z, z_scale=z_scale,
        seq_idx=seq_idx, mm_dtype=torch.float16
    )
    final_states = _quamba2_quant_ssm_states(final_states, x_head_group_range, x_dim_group_range, ssm_state_scale)
    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")