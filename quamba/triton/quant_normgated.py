import torch
import torch.utils.benchmark as benchmark

import triton
assert triton.__version__ >= '3.0.0', f"Triton >= 3.0.0 is requried, but get {triton.__version__}"
import triton.language as tl

@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["QZ"] is not None})
@triton.jit
def _qlayer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    QZ,  # pointer to the other branch
    Z_scale, # floating-point scalar, the QZ scaling factor
    STATIC_SCALE_IN, # floating-point scalar, INPUT PER-TENSOR STATIC scaling factor
    DYNAMIC_SCALE_OUT, # pointer to OUTPUT PER-TOKEN DYNAMIC scaling factor
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    USE_FLOAT16_OUTPUT: tl.constexpr,
    IS_STATIC_SCALE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        QZ += row * stride_z_row + group * N
    DYNAMIC_SCALE_OUT += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(QZ + cols, mask=cols < N).to(tl.float32) * Z_scale
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(cols < N, x - mean, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(QZ + cols, mask=mask).to(tl.float32) * Z_scale
        y *= z * tl.sigmoid(z)
    if USE_FLOAT16_OUTPUT:
        # Write output
        tl.store(Y + cols, y, mask=mask)
    else:
        if IS_STATIC_SCALE:
            per_tensor_scale = STATIC_SCALE_IN # load a floating-point scalar, static per-tensor scaling factor
            y = tl.clamp(tl.extra.cuda.libdevice.rint(y / per_tensor_scale), -128, 127).to(tl.int8) # Triton 3.0.0 required
        else:
            per_token_scale = tl.max(y) / 127. # compute dynamic per-token scaling factor
            tl.store(DYNAMIC_SCALE_OUT + row, per_token_scale) # write dynamic scaling factor
            y = tl.clamp(tl.extra.cuda.libdevice.rint(y / per_token_scale), -128, 127).to(tl.int8) # Triton 3.0.0 required
        # Write output
        tl.store(Y + cols, y, mask=mask)


def _qlayer_normgated_fwd(
        x, weight, bias, eps,
        q_z=None, z_scale=None,
        static_out_scale=None,
        use_float16_output=False,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False
    ):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if q_z is not None:
        assert q_z.stride(-1) == 1
        assert q_z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if use_float16_output:
        out = torch.empty_like(x, dtype=torch.float16)
    else:
        out = torch.empty_like(x, dtype=torch.int8)
    assert out.stride(-1) == 1
    per_token_scale = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _qlayer_norm_fwd_1pass_kernel[grid](x, out, weight, bias, q_z, z_scale, static_out_scale, per_token_scale,
                                           x.stride(0), out.stride(0), q_z.stride(0) if q_z is not None else 0,
                                           M, group_size, eps,
                                           BLOCK_N=BLOCK_N,
                                           NORM_BEFORE_GATE=norm_before_gate,
                                           IS_RMS_NORM=is_rms_norm,
                                           USE_FLOAT16_OUTPUT=use_float16_output,
                                           IS_STATIC_SCALE=static_out_scale is not None,
                                           num_warps=num_warps)
    return out, per_token_scale


