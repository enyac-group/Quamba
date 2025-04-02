import math
import torch
from einops import rearrange

from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd

from quamba.triton.quant_state_passing import _quant_state_passing_fwd


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
        "test_quant_state_passing_fwd": [
            dict(batch=1, seqlen=1024, d_ssm=768*2, headdim=64, dstate=16, chunk_size=128, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-130m
            dict(batch=4, seqlen=1024, d_ssm=768*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-130m
            dict(batch=1, seqlen=512, d_ssm=2560*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-2.7B
            dict(batch=4, seqlen=512, d_ssm=2560*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-2.7B
            dict(batch=4, seqlen=512, d_ssm=8192, headdim=64, dstate=128, chunk_size=128, ngroups=8, seq_idx=None, dtype=torch.float16),  # mamba2-8B
        ],
    }
    def test_quant_state_passing_fwd(self, batch, seqlen, d_ssm, headdim, dstate, chunk_size, ngroups, seq_idx, dtype):
        chunk_size = 256
        mm_dtype = torch.float16
        seq_idx = None
        initial_states = None
        nchunks = math.ceil(seqlen / chunk_size)
        nheads = d_ssm // headdim
        states = torch.rand([batch, nchunks, nheads, headdim, dstate], dtype=torch.float16).cuda() # [bsize, nchunks, nheads, dstate]
        dA_cumsum = torch.rand([batch, nheads, nchunks, chunk_size], dtype=torch.float16).cuda() # [bsize, nheads, nchunks, chunk_size]
        states_out, final_states = _state_passing_fwd(
                                    rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                    initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                        if initial_states is not None else None,
                                    seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                                )
        q_states_out, q_final_states = _quant_state_passing_fwd(
                                    rearrange(states, "... p n -> ... (p n)"),
                                    dA_cumsum[:, :, :, -1],
                                    initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                        if initial_states is not None else None,
                                    seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                                )
        r2 = (states_out - q_states_out).pow(2).mean() / states_out.pow(2).mean()
        # in-place modify ssm_state
        assert r2 < 1e-3, f"{r2}"
        assert torch.allclose(states_out, q_states_out, rtol=1e-2, atol=1e-2)
        r2 = (final_states - q_final_states).pow(2).mean() / final_states.pow(2).mean()
        assert r2 < 1e-3, f"{r2}"
        # assert torch.allclose(final_states, q_final_states, rtol=1e-2, atol=1e-2)
    


