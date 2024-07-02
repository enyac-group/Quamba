import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import torch.utils.benchmark as benchmark

from quamba import QSScan
# from q_mamba import QSScan

torch.manual_seed(0)
torch.set_printoptions(precision=4, threshold=None, edgeitems=6, linewidth=180, profile=None, sci_mode=False)


"""
u: r(B D L)
delta: r(B D L)
A: c(D N) or r(D N)
B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
D: r(D)
z: r(B D L)
delta_bias: r(D), fp32

out: r(B D L)
last_state (optional): r(B D dstate) or c(B D dstate)
"""

@torch.no_grad()
def quantize_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div_(scales).round_()
    return w.to(torch.int8), scales

rtol=1e-02
atol=1e-01

bsize = 1
expand = 2
d_model = 768
d_inner = expand * d_model
d_state = 16
seqlen = 2560
# bsize = 1
# d_inner = 64
# d_state = 16
# seqlen = 12
device = torch.device('cuda:0')


# idtype = torch.float32 
# wdtype = torch.float32
idtype = torch.float16 
wdtype = torch.float16

"""
    Test QSScan Forward
"""

A = repeat(
        torch.arange(1, d_state + 1, dtype=wdtype, device=device),
        "n -> d n",
        d=d_inner,
    ).contiguous()
A_log = torch.log(A)  # Keep A_log in fp32
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L143
# https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.cpp#L234
A = -torch.exp(A_log.float())  # (d_inner, d_state) ----->>>>> So the ssm_state is float !!!
D = torch.ones(d_inner, dtype=wdtype, device=device)  # Keep in fp32

ssm_state = torch.rand((bsize, d_inner, d_state), dtype=torch.float32, device=device) #  ssm_state is float !!!
x = torch.rand((bsize, d_inner, seqlen), dtype=idtype, device=device)
z = torch.rand((bsize, d_inner, seqlen), dtype=idtype, device=device)
dt = torch.rand((bsize, d_inner, seqlen), dtype=idtype, device=device)
B = torch.rand((bsize, d_state, seqlen), dtype=idtype, device=device)
C = torch.rand((bsize, d_state, seqlen), dtype=idtype, device=device)
dt_proj_bias = torch.rand((d_inner,), dtype=wdtype, device=device)

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
y, last_state = selective_scan_fn(x, dt, A, B, C, D.float(), z=z, delta_bias=None, delta_softplus=True, return_last_state=True)
# y, last_state = selective_scan_fn(x, dt, A, B, C, D.float(), z=z, delta_bias=dt_proj_bias.float(), delta_softplus=True, return_last_state=True)
# print(y.shape, y.type())
y = y.transpose(1, 2).contiguous()
# print(last_state.shape, last_state.type()) #  ssm_state is float !!!

q_x, x_scale = quantize_per_tensor_absmax(x.clone(), n_bits=8)
q_z, z_scale = quantize_per_tensor_absmax(z.clone(), n_bits=8)
q_dt, dt_scale = quantize_per_tensor_absmax(dt.clone(), n_bits=8)
q_B, B_scale = quantize_per_tensor_absmax(B.clone(), n_bits=8)
q_C, C_scale = quantize_per_tensor_absmax(C.clone(), n_bits=8)
q_y, y_scale = quantize_per_tensor_absmax(y.clone(), n_bits=8) 

q_sscan = QSScan(A_log, D, dt_bias=None, delta_softplus=True,
                    u_scale=x_scale, dt_scale=dt_scale, B_scale=B_scale,
                    C_scale=C_scale, z_scale=z_scale, output_scale=y_scale)

y_, ssm_state_ = q_sscan.forward(q_x.contiguous(), q_dt, q_B, q_C, z=q_z, return_last_state=True)
print(y_.shape, y_.type())
print(y.shape, y.type())
# y_ = y_.float() * y_scale
y_ = y_.float()
y = y.float()
assert torch.allclose(y, y_, rtol=rtol, atol=atol)



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
ssm_state_gt = ssm_state.clone().to(torch.float16)
act = nn.SiLU()
dt_clone = F.softplus(dt.clone() + dt_proj_bias)
dA = torch.exp(torch.einsum("bd,dn->bdn", dt_clone, A)) # [1, 1536] * [1536, 16] -> [1, 1536, 16]
dB = torch.einsum("bd,bn->bdn", dt_clone, B) # [1, 1536] * [1, 16] -> [1, 1536, 16]
ssm_state_gt.copy_(ssm_state_gt * dA + rearrange(x, "b d -> b d 1") * dB) # [1, 1536, 16] * [1, 1536, 16] + [1, 1536, 1] * [1, 1536, 16]
y = torch.einsum("bdn,bn->bd", ssm_state_gt, C) # [1, 1536, 16] * [1, 16] ->  [1, 1536]
y = y + D * x # [1, 1536] + [1536] * [1, 1536]
y = y * act(z)  # (B D)
# print(ssm_state_gt.shape, ssm_state_gt.type())
# print(y.shape, y.type())

q_x, x_scale = quantize_per_tensor_absmax(x.clone(), n_bits=8)
q_z, z_scale = quantize_per_tensor_absmax(z.clone(), n_bits=8)
q_dt, dt_scale = quantize_per_tensor_absmax(dt.clone(), n_bits=8)
q_B, B_scale = quantize_per_tensor_absmax(B.clone(), n_bits=8)
q_C, C_scale = quantize_per_tensor_absmax(C.clone(), n_bits=8)
q_y, y_scale = quantize_per_tensor_absmax(y.clone(), n_bits=8) 

q_sscan = QSScan(A_log, D, dt_bias=dt_proj_bias, delta_softplus=True,
                    u_scale=x_scale.float(), dt_scale=dt_scale.float(), B_scale=B_scale.float(),
                    C_scale=C_scale.float(), z_scale=z_scale.float(), output_scale=y_scale.float())
ssm_state_ = ssm_state.clone()
y_, ssm_state_ = q_sscan.update(ssm_state_, q_x.contiguous(), q_dt, q_B, q_C, z=q_z)
print(y_.shape, y_.type())
print(y.shape, y.type())
y_ = y_.float()
y = y.float()
assert torch.allclose(y, y_, rtol=rtol, atol=atol)
print(ssm_state_.shape, ssm_state_.type()) #  ssm_state is float !!!
print(ssm_state_gt.shape, ssm_state_gt.type())
ssm_state_ = ssm_state_.float()
ssm_state_gt = ssm_state_gt.float()
assert torch.allclose(ssm_state_gt, ssm_state_, rtol=rtol, atol=atol)
