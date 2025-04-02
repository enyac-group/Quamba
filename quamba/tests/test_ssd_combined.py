import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

from quamba.triton.quant_ssd_combined import _quant_mamba_chunk_scan_combined_fwd
from quamba.quant_utils import quantize_tensor_head_channel_grouping
from quamba.quant_utils import quantize_states_head_channel_grouping, dequantize_states_head_channel_grouping
from quamba.qChunkScan import Quamba2ChunkScan

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
        "test_qchunkscan_forward": [
            dict(batch=1, seqlen=33, d_ssm=768*2, headdim=64, dstate=16, ngroups=1, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),       # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, headdim=64, dstate=128, ngroups=1, chunk_size=256, dt_softplus=True, seq_idx=None, dtype=torch.float16),   # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=8192, headdim=64, dstate=128, ngroups=8, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),     # mamba2-8B
        ],
        "test_qchunkscan_update": [
            dict(batch=1, seqlen=33, d_ssm=768*2, headdim=64, dstate=16, ngroups=1, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),       # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, headdim=64, dstate=128, ngroups=1, chunk_size=256, dt_softplus=True, seq_idx=None, dtype=torch.float16),   # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=8192, headdim=64, dstate=128, ngroups=8, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),     # mamba2-8B
        ],
        "test_qchunkscan_group_heads_forward": [
            dict(batch=1, seqlen=33, d_ssm=768*2, headdim=64, dstate=16, ngroups=1, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),       # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, headdim=64, dstate=128, ngroups=1, chunk_size=256, dt_softplus=True, seq_idx=None, dtype=torch.float16),   # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=8192, headdim=64, dstate=128, ngroups=8, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),     # mamba2-8B
        ],
        "test_qchunkscan_group_heads_update": [
            dict(batch=1, seqlen=33, d_ssm=768*2, headdim=64, dstate=16, ngroups=1, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),       # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=2560*2, headdim=64, dstate=128, ngroups=1, chunk_size=256, dt_softplus=True, seq_idx=None, dtype=torch.float16),   # mamba-130m
            dict(batch=4, seqlen=1024, d_ssm=8192, headdim=64, dstate=128, ngroups=8, chunk_size=128, dt_softplus=True, seq_idx=None, dtype=torch.float16),     # mamba2-8B
        ],
    }
    def test_qchunkscan_forward(self, batch, seqlen, d_ssm, headdim, dstate, ngroups, chunk_size, dt_softplus, seq_idx, dtype):
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt_limit=(0.0, float("inf"))
        nheads = d_ssm // headdim

        # Initialize log dt bias
        dt_init = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        dt_bias = nn.Parameter(inv_dt).cuda()

        A_init_range=(1, dstate)
        A = torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        A_log = nn.Parameter(A_log).cuda()
        A = -torch.exp(A_log.float())

        dt = torch.rand((batch, seqlen, nheads), dtype=torch.float16).cuda()
        dA_cumsum = torch.rand((batch, nheads, seqlen//chunk_size, chunk_size), dtype=torch.float32).cuda()
        x = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        z = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        B = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        C = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        D = torch.rand((nheads), dtype=dtype).cuda()

        x_head_group_range = None
        x_dim_group_range = None

        out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
                                                                    x, dt, A, B, C, chunk_size,
                                                                    D=D, z=None, dt_bias=dt_bias,
                                                                    dt_softplus=dt_softplus,
                                                                    dt_limit=dt_limit)
        out = rearrange(out, "b s h p -> b s (h p)", p=headdim)

        dt_scale = dt.float().abs().max() / 127. # float32 scaling factor
        q_dt = (dt.float() / dt_scale).round().clamp(-128, 127).to(torch.int8)
        x_scale = x.float().abs().max() / 127. # float32 scaling factor
        q_x = (x.float() / x_scale).round().clamp(-128, 127).to(torch.int8)
        B_scale = B.abs().max().to(torch.float32) / 127.
        q_B = (B / B_scale).clamp(-128, 127).round().to(torch.int8)
        C_scale = C.abs().max().to(torch.float32) / 127.
        q_C = (C / C_scale).clamp(-128, 127).round().to(torch.int8)
        ssm_state_scale = final_states.float().abs().amax() / 127.

        qchunk_scan = Quamba2ChunkScan.from_fp16(d_ssm, headdim, dstate, ngroups, x_scale, x_head_group_range, x_dim_group_range,
                    A_log, chunk_size, D=D, dt_bias=dt_bias, delta_softplus=True,
                    dt_scale=dt_scale, B_scale=B_scale, C_scale=C_scale,
                    ssm_state_scale=ssm_state_scale, dt_limit=(0.0, float("inf")))
        # use scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        q_out, q_final_states = qchunk_scan(q_x, q_dt, q_B, q_C, z=None, return_final_states=True)
        q_out = rearrange(q_out, "b s h p -> b s (h p)", p=headdim)
        # amax = (out - q_out).abs().max()
        r2 = (out - q_out).pow(2).mean() / out.pow(2).mean()
        assert r2 < 1e-3
        # assert torch.allclose(out, q_out, rtol=1e-2, atol=1e-2)
        q_final_states = q_final_states * ssm_state_scale
        r2 = (final_states - q_final_states).pow(2).mean() / final_states.pow(2).mean()
        assert r2 < 1e-3
        # assert torch.allclose(final_states, q_final_states, rtol=1e-2, atol=1e-2)


    def test_qchunkscan_update(self, batch, seqlen, d_ssm, headdim, dstate, ngroups, chunk_size, dt_softplus, seq_idx, dtype):
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt_limit=(0.0, float("inf"))
        nheads = d_ssm // headdim

        # Initialize log dt bias
        dt_init = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        dt_bias = nn.Parameter(inv_dt).cuda()

        A_init_range=(1, dstate)
        A = torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        A_log = nn.Parameter(A_log).cuda()
        A = -torch.exp(A_log.float())

        dt = torch.rand((batch, nheads), dtype=torch.float16).cuda()
        x = torch.rand((batch, nheads*headdim), dtype=dtype).cuda()
        z = torch.rand((batch, nheads*headdim), dtype=dtype).cuda()
        B = torch.rand((batch, ngroups*dstate), dtype=dtype).cuda()
        C = torch.rand((batch, ngroups*dstate), dtype=dtype).cuda()
        D = torch.rand((nheads), dtype=dtype).cuda()

        ssm_state = torch.rand((batch, nheads, headdim, dstate), dtype=torch.float16).cuda()

        x_head_group_range = None
        x_dim_group_range = None

        A_log_scale = A_log.float().abs().max() / 127. # float32 scaling factor
        q_A_log = (A_log.float() / A_log_scale).round().clamp(-128, 127).to(torch.int8)
        dt_scale = dt.float().abs().max() / 127. # float32 scaling factor
        q_dt = (dt.float() / dt_scale).round().clamp(-128, 127).to(torch.int8)
        x_scale = x.float().abs().max() / 127. # float32 scaling factor
        q_x = (x.float() / x_scale).round().clamp(-128, 127).to(torch.int8)
        B_scale = B.abs().max().to(torch.float32) / 127.
        q_B = (B.reshape(-1, ngroups, dstate) / B_scale).clamp(-128, 127).round().to(torch.int8).reshape(-1, ngroups*dstate)
        C_scale = C.abs().max().to(torch.float32) / 127.
        q_C = (C.reshape(-1, ngroups, dstate) / C_scale).clamp(-128, 127).round().to(torch.int8).reshape(-1, ngroups*dstate)
        D_scale = D.float().abs().max() / 127. # float32 scaling factor
        q_D = (D.float() / D_scale).round().clamp(-128, 127).to(torch.int8)

        A_ = -torch.exp((q_A_log*A_log_scale).float())
        A_ = repeat(A_, "h -> h p n", p=headdim, n=dstate).to(dtype=torch.float32)
        dt_ = repeat(q_dt*dt_scale, "b h -> b h p", p=headdim).to(dtype=dtype)
        dt_bias_ = repeat(dt_bias, "h -> h p", p=headdim).to(dtype=dtype)
        D_ = repeat(q_D*D_scale, "h -> h p", p=headdim).to(dtype=dtype)
        B_ = (rearrange(q_B, "b (g n) -> b g n", g=ngroups)*B_scale).to(dtype=dtype)
        C_ = (rearrange(q_C, "b (g n) -> b g n", g=ngroups)*C_scale).to(dtype=dtype)

        x_ = repeat(q_x*x_scale, "b (h p) -> b h p", p=headdim).to(dtype=dtype)
        # if not self.rmsnorm:
        #     z = rearrange(z, "b (h p) -> b h p", p=headdim)

        # in-place modify ssm_state
        ssm_state_gt = ssm_state.clone()
        y = selective_state_update(
            ssm_state_gt, x_, dt_, A_, B_, C_, D_, z=None,
            dt_bias=dt_bias_, dt_softplus=dt_softplus
        )
        y = rearrange(y, "b h p -> b (h p)")

        ssm_state_scale = ssm_state.float().abs().amax() / 127.
        qchunk_scan = Quamba2ChunkScan.from_fp16(d_ssm, headdim, dstate, ngroups, x_scale, x_head_group_range, x_dim_group_range,
                    A_log, chunk_size, D=D, dt_bias=dt_bias, delta_softplus=True,
                    dt_scale=dt_scale, B_scale=B_scale, C_scale=C_scale, 
                    ssm_state_scale=ssm_state_scale, dt_limit=(0.0, float("inf")))
        # use scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        # in-place modify ssm_state
        ssm_state_ = (ssm_state.clone().float() / ssm_state_scale).round().clamp(-128, 127).to(torch.int8)
        y_ = qchunk_scan.update(ssm_state_, q_x, q_dt, q_B, q_C, z=None)
        y_ = rearrange(y_, "b h p -> b (h p)")
        r2 = (y - y_).pow(2).mean() / y.pow(2).mean()
        # in-place modify ssm_state
        assert r2 < 1e-3
        assert torch.allclose(y, y_, rtol=1e-2, atol=1e-2)
        ssm_state_ = (ssm_state_ * ssm_state_scale).to(torch.float16)
        r2 = (ssm_state_gt - ssm_state_).pow(2).mean() / ssm_state_gt.pow(2).mean()
        assert r2 < 1e-3
        assert torch.allclose(ssm_state_gt, ssm_state_, rtol=1e-2, atol=1e-2)


    def test_qchunkscan_group_heads_forward(self, batch, seqlen, d_ssm, headdim, dstate, ngroups, chunk_size, dt_softplus, seq_idx, dtype):
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt_limit=(0.0, float("inf"))
        nheads = d_ssm // headdim

        # Initialize log dt bias
        dt_init = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        dt_bias = nn.Parameter(inv_dt).cuda()

        A_init_range=(1, dstate)
        A = torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        A_log = nn.Parameter(A_log).cuda()
        A = -torch.exp(A_log.float())

        dt = torch.rand((batch, seqlen, nheads), dtype=torch.float16).cuda()
        dA_cumsum = torch.rand((batch, nheads, seqlen//chunk_size, chunk_size), dtype=torch.float32).cuda()
        x = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        z = torch.rand((batch, seqlen, nheads, headdim), dtype=dtype).cuda()
        B = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        C = torch.rand((batch, seqlen, ngroups, dstate), dtype=dtype).cuda()
        D = torch.rand((nheads), dtype=dtype).cuda()

        nhead_group = 4
        nhead_group_size = (nheads // ngroups) // nhead_group
        x_head_group_range = torch.arange(nhead_group_size, (nheads // ngroups) + nhead_group_size, nhead_group_size).to(torch.int32).cuda()
        x_head_group_range = x_head_group_range.repeat(ngroups, 1) # [n_ssd_groups, n_head_groups]
        ndim_group = 4
        ndim_group_size = headdim // ndim_group
        x_dim_group_range = torch.arange(ndim_group_size, headdim+ndim_group_size, ndim_group_size).to(torch.int32).cuda()
        x_dim_group_range = x_dim_group_range.repeat(ngroups, nhead_group, 1) # [n_ssd_groups, n_head_groups, n_dim_groups]

        out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
                                                                    x, dt, A, B, C, chunk_size,
                                                                    D=D, z=None, dt_bias=dt_bias,
                                                                    dt_softplus=dt_softplus,
                                                                    dt_limit=dt_limit)
        out = rearrange(out, "b s h p -> b s (h p)", p=headdim)

        dt_scale = dt.float().abs().max() / 127. # float32 scaling factor
        q_dt = (dt.float() / dt_scale).round().clamp(-128, 127).to(torch.int8)
        q_x, x_scales = quantize_tensor_head_channel_grouping(x, x_head_group_range, x_dim_group_range, n_bits=8)
        q_x = q_x.to(torch.int8)
        B_scale = B.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_B = (B / B_scale[None, None, :, None]).clamp(-128, 127).round().to(torch.int8)
        C_scale = C.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_C = (C / C_scale[None, None, :, None]).clamp(-128, 127).round().to(torch.int8)
        ssm_state_scale = final_states.float().abs().amax() / 127.
        _, ssm_state_scale = quantize_states_head_channel_grouping(final_states, x_head_group_range, x_dim_group_range, n_bits=8)
        qchunk_scan = Quamba2ChunkScan.from_fp16(d_ssm, headdim, dstate, ngroups, x_scales, x_head_group_range, x_dim_group_range,
                    A_log, chunk_size, D=D, dt_bias=dt_bias, delta_softplus=True,
                    dt_scale=dt_scale, B_scale=B_scale, C_scale=C_scale, ssm_state_scale=ssm_state_scale,
                    dt_limit=(0.0, float("inf")))
        # use scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        q_out, q_final_states = qchunk_scan(q_x, q_dt, q_B, q_C, z=None, return_final_states=True)
        deq_final_states = dequantize_states_head_channel_grouping(q_final_states, x_head_group_range, x_dim_group_range, ssm_state_scale)
        q_out = rearrange(q_out, "b s h p -> b s (h p)", p=headdim)
        # amax = (out - q_out).abs().max()
        r2 = (out - q_out).pow(2).mean() / out.pow(2).mean()
        assert r2 < 1e-3
        # assert torch.allclose(out, q_out, rtol=1e-2, atol=1e-2)
        r2 = (final_states - deq_final_states).pow(2).mean() / final_states.pow(2).mean()
        assert r2 < 1e-3
        # assert torch.allclose(final_states, q_final_states, rtol=1e-2, atol=1e-2)


    def test_qchunkscan_group_heads_update(self, batch, seqlen, d_ssm, headdim, dstate, ngroups, chunk_size, dt_softplus, seq_idx, dtype):
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt_limit=(0.0, float("inf"))
        nheads = d_ssm // headdim

        # Initialize log dt bias
        dt_init = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        dt_bias = nn.Parameter(inv_dt).cuda()

        A_init_range=(1, dstate)
        A = torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        A_log = nn.Parameter(A_log).cuda()
        A = -torch.exp(A_log.float())

        dt = torch.rand((batch, nheads), dtype=torch.float16).cuda()
        x = torch.rand((batch, nheads*headdim), dtype=dtype).cuda()
        z = torch.rand((batch, nheads*headdim), dtype=dtype).cuda()
        B = torch.rand((batch, ngroups*dstate), dtype=dtype).cuda()
        C = torch.rand((batch, ngroups*dstate), dtype=dtype).cuda()
        D = torch.rand((nheads), dtype=dtype).cuda()

        ssm_state = torch.rand((batch, nheads, headdim, dstate), dtype=torch.float16).cuda()

        nhead_group = 4
        nhead_group_size = (nheads // ngroups) // nhead_group
        x_head_group_range = torch.arange(nhead_group_size, (nheads // ngroups) + nhead_group_size, nhead_group_size).to(torch.int32).cuda()
        x_head_group_range = x_head_group_range.repeat(ngroups, 1) # [n_ssd_groups, n_head_groups]
        ndim_group = 4
        ndim_group_size = headdim // ndim_group
        x_dim_group_range = torch.arange(ndim_group_size, headdim+ndim_group_size, ndim_group_size).to(torch.int32).cuda()
        x_dim_group_range = x_dim_group_range.repeat(ngroups, ndim_group, 1) # [n_ssd_groups, n_head_groups, n_dim_groups]

        A_log_scale = A_log.float().abs().max() / 127. # float32 scaling factor
        q_A_log = (A_log.float() / A_log_scale).round().clamp(-128, 127).to(torch.int8)
        dt_scale = dt.float().abs().max() / 127. # float32 scaling factor
        q_dt = (dt.float() / dt_scale).round().clamp(-128, 127).to(torch.int8)
        x_ = rearrange(x, "b (h p) -> b h p", p=headdim)
        q_x, x_scales = quantize_tensor_head_channel_grouping(
            rearrange(x, "b (h p) -> b h p", p=headdim),
            x_head_group_range, x_dim_group_range, n_bits=8)
        q_x = q_x.to(torch.int8).reshape(batch, nheads*headdim)
        B_scale = B.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_B = (B.reshape(-1, ngroups, dstate) / B_scale[None, :, None]).clamp(-128, 127).round().to(torch.int8).reshape(-1, ngroups*dstate)
        C_scale = C.reshape(-1, ngroups, dstate).permute(1, 0, 2).reshape(ngroups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        q_C = (C.reshape(-1, ngroups, dstate) / C_scale[None, :, None]).clamp(-128, 127).round().to(torch.int8).reshape(-1, ngroups*dstate)
        D_scale = D.float().abs().max() / 127. # float32 scaling factor
        q_D = (D.float() / D_scale).round().clamp(-128, 127).to(torch.int8)

        A_ = -torch.exp((q_A_log*A_log_scale).float())
        A_ = repeat(A_, "h -> h p n", p=headdim, n=dstate).to(dtype=torch.float32)
        dt_ = repeat(q_dt*dt_scale, "b h -> b h p", p=headdim).to(dtype=dtype)
        dt_bias_ = repeat(dt_bias, "h -> h p", p=headdim).to(dtype=dtype)
        D_ = repeat(q_D*D_scale, "h -> h p", p=headdim).to(dtype=dtype)
        B_ = (rearrange(q_B, "b (g n) -> b g n", g=ngroups)*B_scale[None, :, None]).to(dtype=dtype)
        C_ = (rearrange(q_C, "b (g n) -> b g n", g=ngroups)*C_scale[None, :, None]).to(dtype=dtype)

        x_, x_scales = quantize_tensor_head_channel_grouping(
            rearrange(x, "b (h p) -> b h p", p=headdim),
            x_head_group_range, x_dim_group_range, n_bits=8, fake_quant=True)
        x_ = x_.reshape(batch, nheads, headdim)
        # if not self.rmsnorm:
        #     z = rearrange(z, "b (h p) -> b h p", p=headdim)

        # in-place modify ssm_state
        ssm_state_gt = ssm_state.clone()
        y = selective_state_update(
            ssm_state_gt, x_, dt_, A_, B_, C_, D_, z=None,
            dt_bias=dt_bias_, dt_softplus=dt_softplus
        )
        y = rearrange(y, "b h p -> b (h p)")

        q_ssm_state, ssm_state_scale = quantize_states_head_channel_grouping(ssm_state, x_head_group_range, x_dim_group_range, n_bits=8)
        qchunk_scan = Quamba2ChunkScan.from_fp16(d_ssm, headdim, dstate, ngroups, x_scales, x_head_group_range, x_dim_group_range,
                    A_log, chunk_size, D=D, dt_bias=dt_bias, delta_softplus=True,
                    dt_scale=dt_scale, B_scale=B_scale, C_scale=C_scale,
                    ssm_state_scale=ssm_state_scale, dt_limit=(0.0, float("inf")))
        # use scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        # in-place modify ssm_state
        q_ssm_state = q_ssm_state.to(torch.int8)
        y_ = qchunk_scan.update(q_ssm_state, q_x, q_dt, q_B, q_C, z=None)
        y_ = rearrange(y_, "b h p -> b (h p)")
        r2 = (y - y_).pow(2).mean() / y.pow(2).mean()
        assert r2 < 1e-3
        assert torch.allclose(y, y_, rtol=1e-2, atol=1e-2)
        deq_ssm_state= dequantize_states_head_channel_grouping(q_ssm_state, x_head_group_range, x_dim_group_range, ssm_state_scale)
        r2 = (ssm_state_gt - deq_ssm_state).pow(2).mean() / ssm_state_gt.pow(2).mean()
        assert r2 < 1e-3
        assert torch.allclose(ssm_state_gt.float(), deq_ssm_state.float(), rtol=1e-2, atol=1e-2)

