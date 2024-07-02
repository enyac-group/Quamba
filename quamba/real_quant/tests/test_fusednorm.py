import torch
import torch.utils.benchmark as benchmark

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn, rms_norm_ref

import rms_norm_cuda
from model import FusedRMSNorm # editable installation
# from q_mamba import FusedRMSNorm

# dtype = torch.float16
dtype = torch.float32
rtol=1e-02
atol=1e-01
torch.manual_seed(1234)

dim = 768
seqlen = 1
x = torch.rand((1, seqlen, dim)).to(dtype).cuda()
residual = torch.rand((1, seqlen, dim)).to(dtype).cuda()
w = torch.rand(dim).to(dtype).cuda()

y = rms_norm_ref(x + residual, w, bias=None)

norm = RMSNorm(dim, eps=1e-6, dtype=dtype).cuda()
norm.weight = torch.nn.Parameter(w)
y, residual_out = norm(x, prenorm=True)
y_scale = y.abs().max() / 128.

fused_norm = FusedRMSNorm(norm, {"in_proj:input": y_scale}).cuda()
y_, residual_out_ = fused_norm(x, prenorm=True)
y_ = y_.to(dtype) * y_scale
yq = (y / y_scale).round().clamp(-128, 127) * y_scale
assert torch.allclose(residual_out, residual_out_)
assert torch.allclose(yq, y_, rtol=rtol, atol=atol)

y, residual_out = norm(x, residual, prenorm=True)
y_scale = y.abs().max() / 128.

y_, residual_out_ = rms_norm_cuda.fwd(x, (dim, ), w, residual, 1e-6, y_scale)
y_ = y_.to(dtype) * y_scale

yq = (y / y_scale).round().clamp(-128, 127) * y_scale
assert torch.allclose(residual_out, residual_out_)
print((residual_out-residual_out_).max())
print((yq-y_).max())
assert torch.allclose(yq, y_, rtol=rtol, atol=atol)

fused_norm = FusedRMSNorm(norm, {"in_proj:input": y_scale}).cuda()
y_, residual_out_ = fused_norm(x, residual, prenorm=True)
y_ = y_.to(dtype) * y_scale
print((residual_out-residual_out_).max())
assert torch.allclose(residual_out, residual_out_)
print((yq-y_).max())
assert torch.allclose(yq, y_, rtol=rtol, atol=atol)

t1 = benchmark.Timer(
    stmt='y = rms_norm_ref(x + residual, w, bias=None)',
    setup='from mamba_ssm.ops.triton.layernorm import rms_norm_ref',
    globals={'x': x, 'w': w, 'residual': residual})
print(t1.timeit(100))

t1 = benchmark.Timer(
    stmt='y = norm(x, residual, prenorm=True)',
    setup='from __main__ import norm',
    globals={'x': x, 'residual': residual})
print(t1.timeit(100))

t1 = benchmark.Timer(
    stmt='y = rms_norm_cuda.fwd(x, (dim, ), w, residual, 1e-6, y_scale)',
    setup='import rms_norm_cuda',
    globals={'dim': dim, 'x': x, 'w': w, 'residual': residual, 'y_scale': y_scale})
print(t1.timeit(100))


t1 = benchmark.Timer(
    stmt='y_ = fused_norm(x, residual, prenorm=True)',
    setup='from __main__ import fused_norm',
    globals={'x': x, 'residual': residual})
print(t1.timeit(100))
