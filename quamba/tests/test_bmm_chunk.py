import torch

from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
from quamba.triton.quant_bmm_chunk import _quant_bmm_chunk_fwd, _quamba2_bmm_chunk_fwd

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
        "test_quant_bmm_chunk_fwd": [
            dict(batch=1, seqlen=33, ngroups=1, dstate=16, chunk_size=128, seq_idx=None, dtype=torch.float16),      # mamba2-130m
            dict(batch=4, seqlen=1024, ngroups=1, dstate=16, chunk_size=256, seq_idx=None, dtype=torch.float16),    # mamba2-2.7B
            dict(batch=4, seqlen=1024, ngroups=8, dstate=128, chunk_size=128, seq_idx=None, dtype=torch.float16),   # mamba2-8B
        ],
        "test_quamba2_bmm_chunk_fwd": [
            dict(batch=1, seqlen=33, ngroups=1, dstate=16, chunk_size=128, seq_idx=None, dtype=torch.float16),      # mamba2-130m
            dict(batch=4, seqlen=1024, ngroups=1, dstate=16, chunk_size=256, seq_idx=None, dtype=torch.float16),    # mamba2-2.7B
            dict(batch=4, seqlen=1024, ngroups=8, dstate=128, chunk_size=128, seq_idx=None, dtype=torch.float16),   # mamba2-8B
        ],
    }
    def test_quant_bmm_chunk_fwd(self, batch, seqlen, ngroups, dstate, chunk_size, seq_idx, dtype):
        B = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        C = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()

        B_scale = B.abs().amax().to(torch.float32) / 127.
        q_B = (B / B_scale).clamp(-128, 127).round().to(torch.int8)
        C_scale = C.abs().amax().to(torch.float32) / 127.
        q_C = (C / C_scale).clamp(-128, 127).round().to(torch.int8)
        
        CB = _bmm_chunk_fwd(
            (q_C*C_scale).to(torch.float16),
            (q_B*B_scale).to(torch.float16),
            chunk_size, seq_idx=seq_idx, output_dtype=torch.float32
        )

        # use B_scale.item() and x_scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        q_CB = _quant_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

        rtol=1e-03
        atol=1e-02
        amax = (CB - q_CB).abs().max()
        r2 = (CB - q_CB).pow(2).mean() / CB.pow(2).mean()
        assert torch.allclose(CB, q_CB, rtol=rtol, atol=atol), f"amax = {amax}, r2 = {r2}"
        # assert amax < 0.1, f"amax = {amax}"
        # assert r2 < 0.001, f"r2 = {r2}"


    def test_quamba2_bmm_chunk_fwd(self, batch, seqlen, ngroups, dstate, chunk_size, seq_idx, dtype):
        B = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        C = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()

        B_scale = B.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_B = (B / B_scale[None, None, :, None]).clamp(-128, 127).round().to(torch.int8)
        C_scale = C.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_C = (C / C_scale[None, None, :, None]).clamp(-128, 127).round().to(torch.int8)
        
        CB = _bmm_chunk_fwd(
            (q_C*C_scale[None, None, :, None]).to(torch.float16),
            (q_B*B_scale[None, None, :, None]).to(torch.float16),
            chunk_size, seq_idx=seq_idx, output_dtype=torch.float32
        )

        # use B_scale.item() and x_scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        q_CB = _quamba2_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

        rtol=1e-03
        atol=1e-02
        amax = (CB - q_CB).abs().max()
        r2 = (CB - q_CB).pow(2).mean() / CB.pow(2).mean()
        assert torch.allclose(CB, q_CB, rtol=rtol, atol=atol), f"amax = {amax}, r2 = {r2}"
        # assert amax < 0.1, f"amax = {amax}"
        # assert r2 < 0.001, f"r2 = {r2}"




