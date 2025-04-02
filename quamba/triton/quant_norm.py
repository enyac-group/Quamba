import math
import warnings

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

import triton
import triton.language as tl

def config_prune(configs):

    if torch.version.hip:
        try:
            # set warp size based on gcn architecure 
            gcn_arch_name = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx10" in gcn_arch_name or "gfx11" in gcn_arch_name:
                # radeon
                warp_size = 32
            else:
                # instinct
                warp_size = 64
        except AttributeError as e:
            # fall back to crude method to set warp size
            device_name = torch.cuda.get_device_properties(0).name
            if 'instinct' in device_name.lower():
                warp_size = 64
            else:
                warp_size = 32
            warnings.warn(f"{e}, warp size set to {warp_size} based on device name: {device_name}", UserWarning)

    else:
        # cuda 
        warp_size = 32    

    max_block_sz = 1024
    max_num_warps = max_block_sz // warp_size
    pruned_configs = [config for config in configs if config.num_warps <= max_num_warps]
    return pruned_configs

configs_autotune = [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
        ]

pruned_configs_autotune = config_prune(configs_autotune)

configs_autotune = [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
        ]

pruned_configs_autotune = config_prune(configs_autotune)

@triton.autotune(
    configs = pruned_configs_autotune,
    key=["N", "HAS_RESIDUAL", "IS_RMS_NORM", "HAS_BIAS", "IS_STATIC_SCALE"],
)
@triton.jit
def _qlayer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    STATIC_SCALE_IN, # floating-point scalar, INPUT PER-TENSOR STATIC scaling factor
    DYNAMIC_SCALE_OUT, # pointer to OUTPUT PER-TOKEN DYNAMIC scaling factor
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_STATIC_SCALE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if IS_STATIC_SCALE:
        # per_tensor_scale = tl.load(STATIC_SCALE_IN).to(tl.float32) # load static per-tensor scaling factor
        per_tensor_scale = STATIC_SCALE_IN # load a floating-point scalar, static per-tensor scaling factor
        y = tl.clamp(tl.extra.cuda.libdevice.rint(y / per_tensor_scale), -128, 127).to(tl.int8) # Triton 3.0.0 required
    else:
        per_token_scale = tl.max(y) / 127. # compute dynamic per-token scaling factor
        tl.store(DYNAMIC_SCALE_OUT + row, per_token_scale) # write dynamic scaling factor
        y = tl.clamp(tl.extra.cuda.libdevice.rint(y / per_token_scale), -128, 127).to(tl.int8) # Triton 3.0.0 required
    # Write quantized output
    tl.store(Y + cols, y, mask=mask)


def _qlayer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    residual=None,
    static_out_scale=None,
    residual_dtype=None,
    is_rms_norm=False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    y = torch.empty_like(x, dtype=torch.int8)
    # dynamically compute per_token scaling factor if static_out_scale is None
    per_token_scale = torch.empty((M, ), dtype=torch.float32, device=x.device)
    assert y.stride(-1) == 1
    if (
        residual is not None
        or (residual_dtype is not None and residual_dtype != x.dtype)
    ):
        residual_out = torch.empty(
            M, N, device=x.device, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _qlayer_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            residual,
            residual_out,
            static_out_scale,
            per_token_scale, # dynamically compute per_token scaling factor if static_out_scale is None
            x.stride(0),
            y.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            M,
            N,
            eps,
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            bias is not None,
            static_out_scale is not None,
        )
    if static_out_scale:
        return (
            y,
            residual_out if residual_out is not None else x, None
        )
    else:
        # output per_token scaling factor if static_out_scale is None
        return (
            y,
            residual_out if residual_out is not None else x, per_token_scale
        )