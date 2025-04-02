
import math

import torch
import torch.nn as nn

from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
from quamba.triton.quant_chunk_cumsum import _quant_chunk_cumsum_fwd

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
        "test_quant_chunk_cumsum_fwd": [
            dict(batch=1, seqlen=33, nheads=24, dstate=16, chunk_size=128, dtype=torch.float16),    # mamba-130m
            dict(batch=4, seqlen=1024, nheads=24, dstate=16, chunk_size=256, dtype=torch.float16),  # mamba-130m
        ],
    }
    def test_quant_chunk_cumsum_fwd(self, batch, seqlen, nheads, dstate, chunk_size, dtype):
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt_limit=(0.0, float("inf"))
        dt_softplus=True

        # Initialize log dt bias
        dt_init = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        dt_bias = nn.Parameter(inv_dt)

        A_init_range=(1, dstate)
        A = torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        A_log = nn.Parameter(A_log)
        A = -torch.exp(A_log.float())

        dt = torch.rand((batch, seqlen, 24), dtype=torch.float16)
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt.cuda(), A.cuda(), chunk_size, dt_bias=dt_bias.cuda(), dt_softplus=dt_softplus, dt_limit=dt_limit)

        dt_scale = dt.float().abs().max() / 127
        q_dt = (dt.float() / dt_scale).round().to(torch.int8)

        A_log_scale = A_log.float().abs().max() / 127
        q_A_log = (A_log.float() / A_log_scale).round().to(torch.int8)

        # Everything on CUDA
        q_dA_cumsum, q_dt_out = _quant_chunk_cumsum_fwd(
            q_dt.cuda(), dt_scale.cuda(),
            q_A_log.cuda(), A_log_scale.cuda(),
            chunk_size, dt_bias=dt_bias.cuda(),
            dt_softplus=dt_softplus, dt_limit=dt_limit)

        torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

        rtol=1e-03
        atol=1e-02
        amax = (dA_cumsum - q_dA_cumsum).abs().max()
        r2 = (dA_cumsum - q_dA_cumsum).pow(2).mean() / dA_cumsum.pow(2).mean()
        assert r2 < 1e-3
        amax = (dt_out - q_dt_out).abs().max()
        r2 = (dt_out - q_dt_out).pow(2).mean() / dt_out.pow(2).mean()
        assert r2 < 1e-3
        # assert r2 < 1e-3 and amax < 0.1
        # assert torch.allclose(dA_cumsum.float(), q_dA_cumsum.float(), rtol=rtol, atol=atol)
        # assert torch.allclose(dt_out.float(), q_dt_out.float(), rtol=rtol, atol=atol)



