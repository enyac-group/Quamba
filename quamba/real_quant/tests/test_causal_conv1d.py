import copy

import torch
import torch.utils.benchmark as benchmark
from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from quamba import QCausalConv1D # editable installation
# from q_mamba importQCausalConv1D

import quant_causal_conv1d_cuda

torch.manual_seed(1234)

# d_inner = 768*2
# # d_inner = 8 # debugging
# conv_bias=True
# d_conv=4
# seqlen = 128
batch = 1
d_inner = 2560*2
conv_bias=True
d_conv= 4
seqlen = 1024

x = torch.rand((1, d_inner, seqlen)).cuda() # B, D, L
conv1d = torch.nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        ).cuda()
act = torch.nn.SiLU()
print(conv1d.weight.shape, conv1d.bias.shape)
y = act(conv1d(x)[..., :seqlen])
y_scale = y.abs().max() / 127
qy = (y / y_scale).round() * y_scale

y_ = causal_conv1d_fn(
    x=x,
    weight=rearrange(conv1d.weight, "d 1 w -> d w"),
    bias=conv1d.bias,
    activation="silu",
)
assert torch.allclose(y, y_)
t1 = benchmark.Timer(
    stmt='y_gt = causal_conv1d_fn(x, weight, bias, None, activation)',
    setup='from causal_conv1d import causal_conv1d_fn',
    globals={
        'x': x,
        'weight': rearrange(conv1d.weight.clone(), "d 1 w -> d w"),
        'bias': conv1d.bias.clone(),
        'activation': "silu",
    })
print(">> causal_conv1d_fn seq last latency: ")
print(t1.timeit(100))
print("============================")

w_scale = conv1d.weight.abs().max() / 127
qw = (conv1d.weight / w_scale).round().to(torch.int8)
qw = rearrange(qw, "d 1 w -> d w")
b_scale = conv1d.bias.abs().max() / 127
qb = (conv1d.bias / b_scale).round().to(torch.int8)

x_scale = x.abs().max() / 127
qx = (x / x_scale).round().to(torch.int8)

y_ = quant_causal_conv1d_cuda.fwd(
        qx, x_scale,
        qw, w_scale,
        y_scale,
        b_scale, qb,
        None, None, None, True
    )
print(y_.shape, y_.dtype)
y_ = y_.float() * y_scale
# print(y_)
# print(qy)
rtol=1e-02
atol=1e-01
assert torch.allclose(qy, y_, rtol=rtol, atol=atol)

t1 = benchmark.Timer(
    stmt='y_ = quant_causal_conv1d_cuda.fwd(qx, x_scale, qw, w_scale, y_scale, b_scale, qb, None, None, None, True)',
    setup='import quant_causal_conv1d_cuda',
    globals={
        'qx': qx,
        'x_scale': x_scale,
        'qw': qw,
        'w_scale': w_scale,
        'y_scale': y_scale,
        'b_scale': b_scale,
        'qb': qb,
    })
print(">> qconv1d latency: ")
print(t1.timeit(100))

qconv1d = QCausalConv1D(
    originalLayer=copy.deepcopy(conv1d),
    input_scale=x_scale,
    output_scale=y_scale,            
)
y_ = qconv1d(qx)
print(y_.shape, y_.dtype)
y_ = y_.float() * y_scale
assert torch.allclose(qy, y_, rtol=rtol, atol=atol)
# print(x.shape)
# print(x.size())
# print(x.stride(0))
# print(x.stride(1))
# print(x.stride(2))
# print(conv1d.weight.shape)
# print(conv1d.weight.size(0))
# print(conv1d.weight.size(1))
# print(conv1d.weight.size(2))

x = torch.rand((1, 1, d_inner)).cuda() # B, 1, D
x = x.squeeze(1)
conv_state = torch.rand(
    1,
    d_inner,
    d_conv,
    device=conv1d.weight.device,
    dtype=conv1d.weight.dtype,
)

c_s = conv_state.clone()
# update conv_state in-place
y = causal_conv1d_update(
    x,
    c_s,
    rearrange(conv1d.weight, "d 1 w -> d w"),
    conv1d.bias,
    activation="silu",
)

qx = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
qconv_state = (conv_state.clone() / x_scale).round().clamp(-128, 127).to(torch.int8)
qc_s = qconv_state.clone()

y_ = quant_causal_conv1d_cuda.update(qx, qc_s, x_scale, qw, w_scale, y_scale, b_scale, qb, True) # update conv_state in-place
qy_ = y_.float() * y_scale
qy = (y / y_scale).round().clamp(-128, 127) * y_scale
assert torch.allclose(qy, qy_, rtol=rtol, atol=atol)


qc_s = qconv_state.clone()
y_ = qconv1d.update(qx, qc_s) # update conv_state in-place
qy_ = y_.float() * y_scale
qy = (y / y_scale).round().clamp(-128, 127) * y_scale
assert torch.allclose(qy, qy_, rtol=rtol, atol=atol)