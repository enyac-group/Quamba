import torch

from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
from quamba.triton.quant_chunk_scan import _quant_chunk_scan_fwd, _quamba2_chunk_scan_fwd
from quamba.quant_utils import quantize_tensor_head_channel_grouping

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
        "test_quant_chunk_scan_fwd": [
            dict(batch=1, seqlen=1024, d_ssm=768*2, headdim=64, dstate=16, chunk_size=128, ngroups=1, seq_idx=None, dtype=torch.float16),   # mamba2-130m
            dict(batch=4, seqlen=1024, d_ssm=768*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-130m
            dict(batch=1, seqlen=256, d_ssm=2560*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-2.7B
            dict(batch=4, seqlen=256, d_ssm=2560*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-2.7B
            dict(batch=4, seqlen=256, d_ssm=8192, headdim=64, dstate=128, chunk_size=128, ngroups=8, seq_idx=None, dtype=torch.float16),    # mamba2-8B
        ],
        "test_quamba2_chunk_scan_fwd": [
            dict(batch=1, seqlen=1024, d_ssm=768*2, headdim=64, dstate=16, chunk_size=128, ngroups=1, seq_idx=None, dtype=torch.float16),   # mamba2-130m
            dict(batch=4, seqlen=1024, d_ssm=768*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-130m
            dict(batch=1, seqlen=256, d_ssm=2560*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-2.7B
            dict(batch=4, seqlen=256, d_ssm=2560*2, headdim=64, dstate=128, chunk_size=256, ngroups=1, seq_idx=None, dtype=torch.float16),  # mamba2-2.7B
            dict(batch=4, seqlen=256, d_ssm=8192, headdim=64, dstate=128, chunk_size=128, ngroups=8, seq_idx=None, dtype=torch.float16),    # mamba2-8B
        ],
    }
    def test_quant_chunk_scan_fwd(self, batch, seqlen, d_ssm, headdim, dstate, chunk_size, ngroups, seq_idx, dtype):
        nheads = d_ssm // headdim
        dt = torch.rand((batch, nheads, seqlen//chunk_size, chunk_size), dtype=torch.float32).cuda()
        dA_cumsum = torch.rand((batch, nheads, seqlen//chunk_size, chunk_size), dtype=torch.float32).cuda()
        x = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        z = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        C = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        CB = torch.rand((batch, seqlen//chunk_size, ngroups, chunk_size, chunk_size), dtype=dtype).cuda()
        D = torch.rand((nheads), dtype=dtype).cuda()
        states = torch.rand((batch, seqlen//chunk_size, nheads, headdim, dstate), dtype=torch.float32).cuda()

        out, out_x = _chunk_scan_fwd(CB.clone(), x.clone(), dt.clone(), dA_cumsum.clone(), C.clone(), states.clone(), D=D.clone(), z=z.clone(), seq_idx=seq_idx)

        x_scale = x.float().abs().max() / 127. # float32 scaling factor
        q_x = (x.float() / x_scale).round().clamp(-128, 127).to(torch.int8)
        z_scale = z.float().abs().max() / 127. # float32 scaling factor
        q_z = (z.float() / z_scale).round().clamp(-128, 127).to(torch.int8)
        C_scale = C.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_C = (C / C_scale[None, None, :, None]).clamp(-128, 127).round().to(torch.int8)
        D_scale = D.float().abs().max() / 127. # float32 scaling factor
        q_D = (D.float() / D_scale).round().clamp(-128, 127).to(torch.int8)

        # # use B_scale.item() and x_scale.item() will cause Floating point exception (core dumped), since python use float64
        # # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        out, out_x = _chunk_scan_fwd(
            CB.clone(), x.clone(), dt.clone(), dA_cumsum.clone(),
            (q_C*C_scale[None, None, :, None]).to(dtype), states.clone(),
            D=(q_D*D_scale).to(dtype), z=(q_z*z_scale).to(dtype),
            seq_idx=seq_idx)
        q_out, q_out_x = _quant_chunk_scan_fwd(
            CB, q_x, x_scale,
            dt, dA_cumsum, q_C, C_scale, states, q_D=q_D, D_scale=D_scale,
            q_z=q_z, z_scale=z_scale, seq_idx=seq_idx, mm_dtype=torch.float16)

        rtol=1e-03
        atol=1e-02
        amax = (out - q_out).abs().max()
        r2 = (out - q_out).pow(2).mean() / out.pow(2).mean()
        # assert amax < 0.1, f"amax = {amax}"
        assert r2 < 0.001, f"r2 = {r2}"

        amax = (out_x - q_out_x).abs().max()
        r2 = (out_x - q_out_x).pow(2).mean() / out_x.pow(2).mean()
        # assert amax < 0.1, f"amax = {amax}"
        assert r2 < 0.001, f"r2 = {r2}"
    
    def test_quamba2_chunk_scan_fwd(self, batch, seqlen, d_ssm, headdim, dstate, chunk_size, ngroups, seq_idx, dtype):
        nheads = d_ssm // headdim
        dt = torch.rand((batch, nheads, seqlen//chunk_size, chunk_size), dtype=torch.float32).cuda()
        dA_cumsum = torch.rand((batch, nheads, seqlen//chunk_size, chunk_size), dtype=torch.float32).cuda()
        x = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        z = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        C = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        CB = torch.rand((batch, seqlen//chunk_size, ngroups, chunk_size, chunk_size), dtype=dtype).cuda()
        D = torch.rand((nheads), dtype=dtype).cuda()
        states = torch.rand((batch, seqlen//chunk_size, nheads, headdim, dstate), dtype=torch.float32).cuda()

        nhead_group = 4
        nhead_group_size = (nheads // ngroups) // nhead_group
        x_head_group_range = torch.arange(nhead_group_size, (nheads // ngroups) + nhead_group_size, nhead_group_size).to(torch.int32).cuda()
        x_head_group_range = x_head_group_range.repeat(ngroups, 1) # [n_ssd_groups, n_head_groups]
        ndim_group = 4
        ndim_group_size = headdim // ndim_group
        x_dim_group_range = torch.arange(ndim_group_size, headdim+ndim_group_size, ndim_group_size).to(torch.int32).cuda()
        x_dim_group_range = x_dim_group_range.repeat(ngroups, nhead_group, 1) # [n_ssd_groups, n_head_groups, n_dim_groups]

        out, out_x = _chunk_scan_fwd(CB.clone(), x.clone(), dt.clone(), dA_cumsum.clone(), C.clone(), states.clone(), D=D.clone(), z=z.clone(), seq_idx=seq_idx)

        q_x, x_scales = quantize_tensor_head_channel_grouping(x, x_head_group_range, x_dim_group_range, n_bits=8)
        z_scale = z.float().abs().max() / 127. # float32 scaling factor
        q_z = (z.float() / z_scale).round().clamp(-128, 127).to(torch.int8)
        # C_scale = C.float().abs().max() / 127. # float32 scaling factor
        # q_C = (C.float() / C_scale).round().clamp(-128, 127).to(torch.int8)
        C_scale = C.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_C = (C / C_scale[None, None, :, None]).clamp(-128, 127).round().to(torch.int8)
        D_scale = D.float().abs().max() / 127. # float32 scaling factor
        q_D = (D.float() / D_scale).round().clamp(-128, 127).to(torch.int8)

        # # use B_scale.item() and x_scale.item() will cause Floating point exception (core dumped), since python use float64
        # # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        out, out_x = _chunk_scan_fwd(
            CB.clone(), x.clone(), dt.clone(), dA_cumsum.clone(),
            (q_C*C_scale[None, None, :, None]).to(dtype), states.clone(),
            D=(q_D*D_scale).to(dtype), z=(q_z*z_scale).to(dtype),
            seq_idx=seq_idx)
        q_out, q_out_x = _quamba2_chunk_scan_fwd(
            CB, q_x, x_scales, x_head_group_range, x_dim_group_range,
            dt, dA_cumsum, q_C, C_scale, states, q_D=q_D, D_scale=D_scale,
            q_z=q_z, z_scale=z_scale, seq_idx=seq_idx, mm_dtype=torch.float16)

        rtol=1e-03
        atol=1e-02
        amax = (out - q_out).abs().max()
        r2 = (out - q_out).pow(2).mean() / out.pow(2).mean()
        # assert amax < 0.1, f"amax = {amax}"
        assert r2 < 0.001, f"r2 = {r2}"

        amax = (out_x - q_out_x).abs().max()
        r2 = (out_x - q_out_x).pow(2).mean() / out_x.pow(2).mean()
        # assert amax < 0.1, f"amax = {amax}"
        assert r2 < 0.001, f"r2 = {r2}"
