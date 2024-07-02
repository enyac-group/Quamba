import numpy as np

import torch
import torch.utils.benchmark as benchmark

from quamba import W8A8B8O8Linear, W8A8BF16OF16Linear # editable installation
# from q_mamba import W8A8B8O8Linear, W8A8BF16OF16Linear

import quant_linear_cuda

torch.manual_seed(1234)

# https://github.com/ken012git/q_mamba/blob/dev/asymmetric/fake_quant/observer.py
def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max


def _get_minmax_quantization_params(w_max, w_min, n_bits, clip_ratio, sym):
    q_min, q_max = _get_quant_range(n_bits=n_bits, sym=sym)
    if sym:
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        assert w_min is not None, "w_min should not be None for asymmetric quantization."
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        base = base - 128 # left shift to the range of [-128, 127]
    return scales, base

# B, L, M, N = 1, 1, 768, 768*4
B, L, M, N = 1, 1, 16, 32

x = torch.randn(B, L, M).to(torch.float16).cuda()
linear = torch.nn.Linear(M, N, bias=False, dtype=torch.float16).cuda()
y_gt = linear(x)
y_gt = y_gt.float()
t1 = benchmark.Timer(
    stmt='y_gt = linear(x)',
    setup='from __main__ import linear',
    globals={'x': x})
print(">> latency: ", t1.timeit(100))
print("============================")

x_scale = x.clone().abs().max() / 127
qx = (x.clone() / x_scale).round().to(torch.int8)
qx = qx.cuda()
y_scale = y_gt.abs().max() / 127


linear_int8_fp16 = W8A8BF16OF16Linear(linear, x_scale).cuda()
q_y = linear_int8_fp16(qx)
assert q_y.dtype == torch.float16, f"{q_y.shape, q_y.dtype}"
y_hat = q_y.float()
amax = (y_gt - y_hat).abs().max()
r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
print(">> diff: ", r2.item(), amax.item())
assert r2 < 1e-3 and amax < 0.1
t2 = benchmark.Timer(
    stmt='q_y = linear_int8_fp16(qx)',
    setup='from __main__ import linear_int8_fp16',
    globals={'qx': qx})
print(">> latency: ", t2.timeit(100))
print("============================")

linear_int8 = W8A8B8O8Linear(linear, x_scale, y_scale).cuda()
q_y = linear_int8(qx)
assert q_y.dtype == torch.int8, f"{q_y.shape, q_y.dtype}"
y_hat = q_y.float() * y_scale
amax = (y_gt - y_hat).abs().max()
r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
print(">> diff: ", r2.item(), amax.item())
assert r2 < 1e-3 and amax < 0.1
t3 = benchmark.Timer(
    stmt='q_y = linear_int8(qx)',
    setup='from __main__ import linear_int8',
    globals={'qx': qx})
print(">> latency: ", t3.timeit(100))
print("============================")


q_y = linear_int8.to_seqlen_last(qx)
print("B L D: ", qx.shape, ", qx.is_contiguous(): ", qx.is_contiguous())
print("B D L: ", q_y.shape, ", q_y.is_contiguous(): ", q_y.is_contiguous())
print(q_y.shape, q_y.stride())
y_hat = (q_y.float() * y_scale).transpose(1, 2) # B D L -> B L D
amax = (y_gt - y_hat).abs().max()
r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
print(">> diff: ", r2.item(), amax.item())
assert r2 < 1e-3 and amax < 0.1
t4 = benchmark.Timer(
    stmt='q_y = linear_int8.to_seqlen_last(qx)',
    setup='from __main__ import linear_int8',
    globals={'qx': qx})
print(">> latency: ", t4.timeit(100))
print("============================")
