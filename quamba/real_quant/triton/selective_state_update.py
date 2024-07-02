# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, x_scale, dt_ptr, dt_scale, dt_bias_ptr, dt_bias_scale,
    A_log_ptr, A_log_scale, B_ptr, B_scale, C_ptr, C_scale, D_ptr, D_scale, z_ptr, z_scale, out_ptr,
    # Matrix dimensions
    batch, dim, dstate,
    # Strides
    stride_state_batch, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_dim,
    stride_dt_batch, stride_dt_dim,
    stride_dt_bias_dim,
    stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_dstate,
    stride_C_batch, stride_C_dstate,
    stride_D_dim,
    stride_z_batch, stride_z_dim,
    stride_out_batch, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    state_ptr += pid_b * stride_state_batch
    x_ptr += pid_b * stride_x_batch
    dt_ptr += pid_b * stride_dt_batch
    B_ptr += pid_b * stride_B_batch
    C_ptr += pid_b * stride_C_batch
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch
    out_ptr += pid_b * stride_out_batch

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    A_log_ptrs = A_log_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32) * tl.load(x_scale)
    dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32) * tl.load(dt_scale)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32) * tl.load(dt_bias_scale)
    if DT_SOFTPLUS:
        dt = tl.log(1.0 + tl.exp(dt))
    A_log = tl.load(A_log_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32) * tl.load(A_log_scale)
    A = -tl.exp(A_log)  # (d_inner, d_state)
    dA = tl.exp(A * dt[:, None])
    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32) * tl.load(B_scale)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32) * tl.load(C_scale)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32) * tl.load(D_scale)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32) * tl.load(z_scale)

    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)


def qsscan_update_triton(state, x, x_scale, dt, dt_scale, A_log, A_log_scale, B, B_scale, C, C_scale,
                            D=None, D_scale=None, z=None, z_scale=None, dt_bias=None, dt_bias_scale=None,
                            dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A_log: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = state.shape
    assert x.shape == (batch, dim)
    assert dt.shape == x.shape
    assert A_log.shape == (dim, dstate)
    assert B.shape == (batch, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (dim,)
    out = torch.empty_like(x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch)
    z_strides = ((z.stride(0), z.stride(1)) if z is not None else (0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state, x, x_scale, dt, dt_scale, dt_bias, dt_bias_scale, 
            A_log, A_log_scale, B, B_scale, C, C_scale, D, D_scale, z, z_scale, out,
            batch, dim, dstate,
            state.stride(0), state.stride(1), state.stride(2),
            x.stride(0), x.stride(1),
            dt.stride(0), dt.stride(1),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A_log.stride(0), A_log.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            D.stride(0) if D is not None else 0,
            z_strides[0], z_strides[1],
            out.stride(0), out.stride(1),
            dt_softplus,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    return out


def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = state.shape
    assert x.shape == (batch, dim)
    assert dt.shape == x.shape
    assert A.shape == (dim, dstate)
    assert B.shape == (batch, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (dim,)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b d -> b d 1") * A)  # (batch, dim, dstate)
    dB = rearrange(dt, "b d -> b d 1") * rearrange(B, "b n -> b 1 n")  # (batch, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, "b d -> b d 1"))  # (batch, dim, dstate
    out = torch.einsum("bdn,bn->bd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    return (out if z is None else out * F.silu(z)).to(x.dtype)


if __name__ == "__main__":

    @torch.no_grad()
    def quantize_per_tensor_absmax(w, n_bits=8):
        scales = w.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w = w.div_(scales).round_()
        # print(scales.float().shape, scales.float().dtype)
        return w.to(torch.int8), scales.float()

    rtol=1e-02
    atol=1e-01

    # bsize = 1
    # expand = 2
    # d_model = 768
    # d_inner = expand * d_model
    # d_state = 16
    # seqlen = 2560
    bsize = 1
    d_inner = 64
    d_state = 16
    seqlen = 12
    device = torch.device('cuda:0')
    idtype = torch.float16 
    wdtype = torch.float16
    """
        Test QSScan Update
    """

    A = repeat(
            torch.arange(1, d_state + 1, dtype=wdtype, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
    A = A + torch.arange(0, d_inner, dtype=wdtype, device=device).reshape((d_inner, 1))
    A_log = torch.log(A)  # Keep A_log in fp32
    A = -torch.exp(A_log.float())  # (d_inner, d_state)
    D = torch.ones(d_inner, dtype=wdtype, device=device)  # Keep in fp32

    ssm_state = torch.rand((bsize, d_inner, d_state), dtype=torch.float32, device=device)  #  ssm_state is float !!!
    x = torch.rand((bsize, d_inner), dtype=idtype, device=device)
    z = torch.rand((bsize, d_inner), dtype=idtype, device=device)
    dt = torch.rand((bsize, d_inner), dtype=idtype, device=device)
    B = torch.rand((bsize, d_state), dtype=idtype, device=device)
    C = torch.rand((bsize, d_state), dtype=idtype, device=device)
    dt_proj_bias = torch.rand((d_inner,), dtype=wdtype, device=device)


    import torch.nn as nn
    ssm_state_gt = ssm_state.clone().to(torch.float16)
    act = nn.SiLU()
    dt_clone = F.softplus(dt.clone() + dt_proj_bias)
    dA = torch.exp(torch.einsum("bd,dn->bdn", dt_clone, A)) # [1, 1536] * [1536, 16] -> [1, 1536, 16]
    dB = torch.einsum("bd,bn->bdn", dt_clone, B) # [1, 1536] * [1, 16] -> [1, 1536, 16]
    ssm_state_gt.copy_(ssm_state_gt * dA + rearrange(x, "b d -> b d 1") * dB) # [1, 1536, 16] * [1, 1536, 16] + [1, 1536, 1] * [1, 1536, 16]
    y = torch.einsum("bdn,bn->bd", ssm_state_gt, C) # [1, 1536, 16] * [1, 16] ->  [1, 1536]
    y = y + D * x # [1, 1536] + [1536] * [1, 1536]
    y = y * act(z)  # (B D)

    A_log_quant, A_log_scale = quantize_per_tensor_absmax(A_log, n_bits=8)
    D_quant, D_scale = quantize_per_tensor_absmax(D, n_bits=8)
    dt_bias, dt_bias_scale = quantize_per_tensor_absmax(dt_proj_bias, n_bits=8)
    q_x, x_scale = quantize_per_tensor_absmax(x.clone(), n_bits=8)
    q_z, z_scale = quantize_per_tensor_absmax(z.clone(), n_bits=8)
    q_dt, dt_scale = quantize_per_tensor_absmax(dt.clone(), n_bits=8)
    q_B, B_scale = quantize_per_tensor_absmax(B.clone(), n_bits=8)
    q_C, C_scale = quantize_per_tensor_absmax(C.clone(), n_bits=8)
    q_y, y_scale = quantize_per_tensor_absmax(y.clone(), n_bits=8) 

    ssm_state_q = ssm_state.clone().to(torch.float32)
    y_q = qsscan_update_triton(ssm_state_q, q_x, x_scale,
        q_dt, dt_scale, A_log_quant, A_log_scale, q_B, B_scale, 
        q_C, C_scale, D_quant, D_scale, q_z, z_scale, dt_bias, dt_bias_scale,
        dt_softplus=True)
    print(y_q.shape, y_q.type())
    print(y.shape, y.type())
    y_q = y_q.float()
    y = y.float()
    print(y_q)
    print(y)
    assert torch.allclose(y, y_q, rtol=rtol, atol=atol)
    print(ssm_state_q.shape, ssm_state_q.type()) #  ssm_state is float !!!
    print(ssm_state_gt.shape, ssm_state_gt.type())
    ssm_state_q = ssm_state_q.float()
    ssm_state_gt = ssm_state_gt.float()
    assert torch.allclose(ssm_state_gt, ssm_state_q, rtol=rtol, atol=atol)