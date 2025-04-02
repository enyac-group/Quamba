import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import torch.utils.benchmark as benchmark

from quamba import QSScan

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
    w = w.div_(scales).round_().clamp(-q_max, q_max)
    return w.to(torch.int8), scales


torch.manual_seed(1234)

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

class TestClass:
    # a map specifying multiple argument sets for a test method
    params = {
        "test_qsscan_forward": [
            dict(batch=1, seqlen=33, d_ssm=768*2,  d_state=16, dt_softplus=True, seq_idx=None, dtype=torch.float16),    # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, d_state=128, dt_softplus=True, seq_idx=None, dtype=torch.float16), # mamba-2.7B
            dict(batch=4, seqlen=1024, d_ssm=8192, d_state=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),
        ],
        "test_qsscan_update": [
            dict(batch=1, d_ssm=768*2, d_state=16, dt_softplus=True, seq_idx=None, dtype=torch.float16),    # mamba-130m
            dict(batch=4, d_ssm=2560*2, d_state=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),  # mamba-2.7B
            dict(batch=4, d_ssm=8192, d_state=128, dt_softplus=True, seq_idx=None, dtype=torch.float16), 
        ],
    }

    def test_qsscan_forward(self, batch, seqlen, d_ssm, d_state, dt_softplus, seq_idx, dtype):

        rtol=1e-02
        atol=1e-01
        device = torch.device('cuda:0')
        idtype = dtype
        wdtype = dtype
        """
            Test QSScan Forward
        """
        A = repeat(
                torch.arange(1, d_state + 1, dtype=wdtype, device=device),
                "n -> d n",
                d=d_ssm,
            ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L143
        # https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.cpp#L234
        A = -torch.exp(A_log.float())  # (d_inner, d_state) ----->>>>> So the ssm_state is float !!!
        D = torch.ones(d_ssm, dtype=wdtype, device=device)  # Keep in fp32

        ssm_state = torch.rand((batch, d_ssm, d_state), dtype=torch.float32, device=device) #  ssm_state is float !!!
        x = torch.rand((batch, d_ssm, seqlen), dtype=idtype, device=device)
        z = torch.rand((batch, d_ssm, seqlen), dtype=idtype, device=device)
        dt = torch.rand((batch, d_ssm, seqlen), dtype=idtype, device=device)
        B = torch.rand((batch, d_state, seqlen), dtype=idtype, device=device)
        C = torch.rand((batch, d_state, seqlen), dtype=idtype, device=device)
        dt_proj_bias = torch.rand((d_ssm,), dtype=wdtype, device=device)

        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        y, last_state = selective_scan_fn(x, dt, A, B, C, D.float(), z=z, delta_bias=None, delta_softplus=dt_softplus, return_last_state=True)
        # y, last_state = selective_scan_fn(x, dt, A, B, C, D.float(), z=z, delta_bias=dt_proj_bias.float(), delta_softplus=True, return_last_state=True)
        y = y.transpose(1, 2).contiguous()

        q_x, x_scale = quantize_per_tensor_absmax(x.clone(), n_bits=8)
        q_z, z_scale = quantize_per_tensor_absmax(z.clone(), n_bits=8)
        q_dt, dt_scale = quantize_per_tensor_absmax(dt.clone(), n_bits=8)
        q_B, B_scale = quantize_per_tensor_absmax(B.clone(), n_bits=8)
        q_C, C_scale = quantize_per_tensor_absmax(C.clone(), n_bits=8)
        q_ssm_state, ssm_state_scale = quantize_per_tensor_absmax(ssm_state.clone(), n_bits=8) 

        q_sscan = QSScan.from_fp16(d_state, d_ssm, A_log, D, dt_bias=None, delta_softplus=dt_softplus,
                            ssm_state_scale=ssm_state_scale, u_scale=x_scale, dt_scale=dt_scale,
                            B_scale=B_scale, C_scale=C_scale, z_scale=z_scale)

        y_, last_state_ = q_sscan.forward(q_x.contiguous(), q_dt, q_B, q_C, z=q_z, return_last_state=True)
        print(last_state_.shape, last_state_.dtype)
        y_ = y_.float()
        y = y.float()
        # assert torch.allclose(y, y_, rtol=rtol, atol=atol)
        r2 = (y_ - y).pow(2).mean() / y.pow(2).mean()
        assert r2 < 1e-3

        last_state_ = last_state_.float() * ssm_state_scale
        last_state = last_state.float()
        r2 = (last_state_ - last_state).pow(2).mean() / last_state.pow(2).mean()
        assert r2 < 1e-3


    def test_qsscan_update(self, batch, d_ssm, d_state, dt_softplus, seq_idx, dtype):

        rtol=1e-02
        atol=1e-01
        device = torch.device('cuda:0')
        idtype = dtype
        wdtype = dtype
        """
            Test QSScan Update
        """
        A = repeat(
                torch.arange(1, d_state + 1, dtype=wdtype, device=device),
                "n -> d n",
                d=d_ssm,
            ).contiguous()
        A = A + torch.arange(0, d_ssm, dtype=wdtype, device=device).reshape((d_ssm, 1))
        A_log = torch.log(A)  # Keep A_log in fp32
        q_A_log, A_log_scale = quantize_per_tensor_absmax(A_log.clone(), n_bits=8)
        A = -torch.exp(q_A_log.float() * A_log_scale)  # (d_inner, d_state)
        D = torch.ones(d_ssm, dtype=wdtype, device=device)  # Keep in fp32

        ssm_state = torch.rand((batch, d_ssm, d_state), dtype=torch.float32, device=device)  #  ssm_state is float !!!
        x = torch.rand((batch, d_ssm), dtype=idtype, device=device)
        z = torch.rand((batch, d_ssm), dtype=idtype, device=device)
        dt = torch.rand((batch, d_ssm), dtype=idtype, device=device)
        B = torch.rand((batch, d_state), dtype=idtype, device=device)
        C = torch.rand((batch, d_state), dtype=idtype, device=device)
        dt_proj_bias = torch.rand((d_ssm,), dtype=wdtype, device=device)
        # ssm_state_gt = ssm_state.clone().to(torch.float16)

        q_x, x_scale = quantize_per_tensor_absmax(x.clone(), n_bits=8)
        q_z, z_scale = quantize_per_tensor_absmax(z.clone(), n_bits=8)
        q_dt, dt_scale = quantize_per_tensor_absmax(dt.clone(), n_bits=8)
        q_B, B_scale = quantize_per_tensor_absmax(B.clone(), n_bits=8)
        q_C, C_scale = quantize_per_tensor_absmax(C.clone(), n_bits=8)
        q_ssm_state, ssm_state_scale = quantize_per_tensor_absmax(ssm_state.clone(), n_bits=8) 

        x_ = (q_x * x_scale).to(torch.float32)
        z_ = (q_z * z_scale).to(torch.float32)
        dt_ = (q_dt * dt_scale).to(torch.float32)
        B_ = (q_B * B_scale).to(torch.float32)
        C_ = (q_C * C_scale).to(torch.float32)
        ssm_state_gt = (q_ssm_state * ssm_state_scale)

        act = nn.SiLU()
        dt_clone = dt_.clone() + dt_proj_bias
        if dt_softplus:
            dt_clone = F.softplus(dt_clone)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt_clone, A)) # [1, 1536] * [1536, 16] -> [1, 1536, 16]
        dB = torch.einsum("bd,bn->bdn", dt_clone, B_) # [1, 1536] * [1, 16] -> [1, 1536, 16]
        ssm_state_gt.copy_(ssm_state_gt * dA + rearrange(x_, "b d -> b d 1") * dB) # [1, 1536, 16] * [1, 1536, 16] + [1, 1536, 1] * [1, 1536, 16]
        y = torch.einsum("bdn,bn->bd", ssm_state_gt, C_) # [1, 1536, 16] * [1, 16] ->  [1, 1536]
        y = y + D * x_ # [1, 1536] + [1536] * [1, 1536]
        y = y * act(z_)  # (B D)

        q_sscan = QSScan.from_fp16(d_state, d_ssm, A_log, D, dt_bias=dt_proj_bias, delta_softplus=dt_softplus,
                                   ssm_state_scale=ssm_state_scale.float(),
                            u_scale=x_scale.float(), dt_scale=dt_scale.float(), B_scale=B_scale.float(),
                            C_scale=C_scale.float(), z_scale=z_scale.float())
        y_, ssm_state_ = q_sscan.update(q_ssm_state, q_x.contiguous(), q_dt, q_B, q_C, z=q_z)
        y_ = y_.float()
        y = y.float()
        r2 = (y_ - y).pow(2).mean() / y.pow(2).mean()
        # assert torch.allclose(y, y_, rtol=rtol, atol=atol)
        assert r2 < 1e-3
        
        ssm_state_ = (ssm_state_ * ssm_state_scale).float()
        # ssm_state_gt = ssm_state_gt.float()
        ssm_state_gt = (ssm_state_gt / ssm_state_scale).round().clamp(-128, 127).float() * ssm_state_scale
        # assert torch.allclose(ssm_state_gt, ssm_state_, rtol=rtol, atol=atol)
        r2 = (ssm_state_ - ssm_state_gt).pow(2).mean() / ssm_state_gt.pow(2).mean()
        assert r2 < 1e-3
