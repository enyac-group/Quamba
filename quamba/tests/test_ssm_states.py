import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from quamba.triton.quant_ssm_states import _quant_quant_ssm_states, _quamba2_quant_ssm_states
from quamba.quant_utils import quantize_tensor_head_channel_grouping
from quamba.quant_utils import quantize_states_head_channel_grouping, dequantize_states_head_channel_grouping

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
        "test_quant_quant_ssm_states": [
            dict(batch=1, seqlen=33, d_ssm=768*2, headdim=64, dstate=16, ngroups=1, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),       # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, headdim=64, dstate=128, ngroups=1, chunk_size=256, dt_softplus=True, seq_idx=None, dtype=torch.float16),   # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=8192, headdim=64, dstate=128, ngroups=8, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),     # mamba2-8B
        ],
        "test_quamba2_quant_ssm_states": [
            dict(batch=1, seqlen=33, d_ssm=768*2, headdim=64, dstate=16, ngroups=1, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),       # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, headdim=64, dstate=128, ngroups=1, chunk_size=256, dt_softplus=True, seq_idx=None, dtype=torch.float16),   # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=8192, headdim=64, dstate=128, ngroups=8, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),     # mamba2-8B
        ],
    }
    def test_quant_quant_ssm_states(self, batch, seqlen, d_ssm, headdim, dstate, ngroups, chunk_size, dt_softplus, seq_idx, dtype):
        nheads = d_ssm // headdim
        ssm_states = torch.rand((batch, nheads, headdim, dstate), dtype=dtype).cuda()
        ssm_state_scale = ssm_states.float().abs().amax() / 127.
        q_ssm_states = _quant_quant_ssm_states(ssm_states, ssm_state_scale)
        q_ssm_states = q_ssm_states * ssm_state_scale
        r2 = (ssm_states - q_ssm_states).pow(2).mean() / ssm_states.pow(2).mean()
        assert r2 < 1e-3
        assert torch.allclose(ssm_states.float(), q_ssm_states.float(), rtol=1e-2, atol=1e-2)


    def test_quamba2_quant_ssm_states(self, batch, seqlen, d_ssm, headdim, dstate, ngroups, chunk_size, dt_softplus, seq_idx, dtype):
        nheads = d_ssm // headdim
        ssm_states = torch.rand((batch, nheads, headdim, dstate), dtype=dtype).cuda()

        nhead_group = 4
        nhead_group_size = (nheads // ngroups) // nhead_group
        x_head_group_range = torch.arange(nhead_group_size, (nheads // ngroups) + nhead_group_size, nhead_group_size).to(torch.int32).cuda()
        x_head_group_range = x_head_group_range.repeat(ngroups, 1) # [n_ssd_groups, n_head_groups]
        ndim_group = 4
        ndim_group_size = headdim // ndim_group
        x_dim_group_range = torch.arange(ndim_group_size, headdim+ndim_group_size, ndim_group_size).to(torch.int32).cuda()
        x_dim_group_range = x_dim_group_range.repeat(ngroups, nhead_group, 1) # [n_ssd_groups, n_head_groups, n_dim_groups]

        _, ssm_state_scale = quantize_states_head_channel_grouping(ssm_states, x_head_group_range, x_dim_group_range, n_bits=8)
        q_ssm_states = _quamba2_quant_ssm_states(ssm_states, x_head_group_range, x_dim_group_range, ssm_state_scale)
        deq_ssm_states = dequantize_states_head_channel_grouping(q_ssm_states, x_head_group_range, x_dim_group_range, ssm_state_scale)
        r2 = (ssm_states - deq_ssm_states).pow(2).mean() / ssm_states.pow(2).mean()
        assert r2 < 1e-3
        assert torch.allclose(ssm_states.float(), deq_ssm_states.float(), rtol=1e-2, atol=1e-2)
