import pytest
import copy
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from quamba import W4A16B16O16Linear
from quamba import W4A8B8O8Linear, W4A8B8O8LinearParallel, W4A8B16O16Linear
from quamba import W8A8B8O8Linear, W8A8B8O8LinearParallel, W8A8B16O16Linear
from quamba import HadLinear
from quamba import HadLinear, Hadamard, QHadamard # editable installation
from quamba.marlin_utils import MarlinWorkspace
from quamba.marlin_utils import w4a8_quantize, w4a16_quantize
import quant_linear_cuda

torch.manual_seed(0)
torch.set_printoptions(sci_mode=False)

def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))

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
        "test_W8A8B16O16Linear": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W8A8B8O8Linear": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W8A8B8O8Linear_to_seqlen_last": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W8A8B8O8LinearParallel": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W4A16B16O16Linear": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W4A16B16O16Linear_to_seqlen_last": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1024, hidden_dim=2560, model_dim=2560*4, d_state=128, nheads=24),  # mamba-2.8b
        ],
        "test_W4A8B16O16Linear": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W4A8B16O16Linear_to_seqlen_last": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1024, hidden_dim=2560, model_dim=2560*4, d_state=128, nheads=24),  # mamba-2.8b
        ],
        "test_W4A8B8O8Linear": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m
        ],
        "test_W4A8B8O8Linear_to_seqlen_last": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1024, hidden_dim=2560, model_dim=2560*4, d_state=128, nheads=24),  # mamba-2.8b
        ],
        "test_W4A8B8O8LinearParallel": [
            dict(batch=4, seqlen=1024, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),    # mamba2-130m
            dict(batch=4, seqlen=89, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),      # mamba2-130m
            dict(batch=1, seqlen=128, hidden_dim=2560, model_dim=2560*2, d_state=128, nheads=24),   # mamba-2.8b
            dict(batch=1, seqlen=1, hidden_dim=768, model_dim=768*2, d_state=128, nheads=24),       # mamba2-130m 
        ],
    }

    @torch.no_grad()
    def test_W8A8B16O16Linear(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        y_gt = linear(x)
        y_gt = y_gt.float()

        x_scale = x.clone().abs().max() / 127
        qx = (x.clone() / x_scale).round().to(torch.int8)
        qx = qx.cuda()
        y_scale = y_gt.abs().max() / 127

        linear_int8_fp16 = W8A8B16O16Linear.from_fp16(linear, x_scale).cuda()
        q_y = linear_int8_fp16(qx)
        assert q_y.dtype == torch.float16, f"{q_y.shape, q_y.dtype}"
        y_hat = q_y.float()
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

    @torch.no_grad()
    def test_W8A8B8O8Linear(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        y_gt = linear(x)
        y_gt = y_gt.float()

        x_scale = x.clone().abs().max() / 127
        qx = (x.clone() / x_scale).round().to(torch.int8)
        qx = qx.cuda()
        y_scale = y_gt.abs().max() / 127

        linear_int8 = W8A8B8O8Linear.from_fp16(linear, x_scale, y_scale).cuda()
        q_y = linear_int8(qx)
        assert q_y.dtype == torch.int8, f"{q_y.shape, q_y.dtype}"
        y_hat = q_y.float() * y_scale
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

    @torch.no_grad()
    def test_W8A8B8O8Linear_to_seqlen_last(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        y_gt = linear(x)
        y_gt = y_gt.float()

        x_scale = x.clone().abs().max() / 127
        qx = (x.clone() / x_scale).round().to(torch.int8)
        qx = qx.cuda()
        y_scale = y_gt.abs().max() / 127

        linear_int8 = W8A8B8O8Linear.from_fp16(linear, x_scale, y_scale).cuda()
        q_y = linear_int8.to_seqlen_last(qx)
        y_hat = (q_y.float() * y_scale).transpose(1, 2) # B D L -> B L D
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

    @torch.no_grad()
    def test_W8A8B8O8LinearParallel(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        zxbcdt = linear(x)
        zxbcdt = zxbcdt.float()
        z, xBC, dt = torch.split(
            zxbcdt,
            [model_dim, model_dim + 2 * d_state, nheads],
            dim=-1
        )
        x_scale = x.clone().abs().max() / 127
        z_scale = z.abs().max() / 127
        xBC_scale = xBC.abs().max() / 127
        dt_scale = dt.abs().max() / 127


        qx = (x.clone() / x_scale).round().to(torch.int8)
        qx = qx.cuda()

        linear_int8 = W8A8B8O8LinearParallel.from_fp16(
                        linear, x_scale, [z_scale, xBC_scale, dt_scale],
                        [model_dim, model_dim + 2 * d_state, nheads]).cuda()

        q_zxbcdt = linear_int8(qx)
        q_z, q_xBC, q_dt = torch.split(
            q_zxbcdt,
            [model_dim, model_dim + 2 * d_state, nheads],
            dim=-1
        )
        z_hat = q_z.float() * z_scale
        amax = (z - z_hat).abs().max()
        r2 = (z - z_hat).pow(2).mean() / z.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

        xBC_hat = q_xBC.float() * xBC_scale
        amax = (xBC - xBC_hat).abs().max()
        r2 = (xBC - xBC_hat).pow(2).mean() / xBC.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

        dt_hat = q_dt.float() * dt_scale
        amax = (dt - dt_hat).abs().max()
        r2 = (dt - dt_hat).pow(2).mean() / dt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

    @torch.no_grad()
    def test_W4A16B16O16Linear(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        linear_w4 = W4A16B16O16Linear.from_fp16(linear)

        w_ref, _, _ = w4a16_quantize(
            linear.weight.data.t(), num_bits=4, group_size=128, scale=None, pad_out=linear_w4.pad_out)
        if linear_w4.pad_out > 0:
            w_ref = w_ref[:, :-linear_w4.pad_out] # [Din, Dout]
        linear.weight.data = w_ref.t().contiguous()

        y_gt = linear(x)
        q_y = linear_w4(x)
        assert q_y.dtype == torch.float16, f"{q_y.dtype}"
        y_gt = y_gt.float()
        y_hat = q_y.float()
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1

    @torch.no_grad()
    def test_W4A16B16O16Linear_to_seqlen_last(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        linear_w4 = W4A16B16O16Linear.from_fp16(linear).cuda()

        w_ref, _, _ = w4a16_quantize(
            linear.weight.data.t(), num_bits=4, group_size=128, scale=None, pad_out=linear_w4.pad_out)
        if linear_w4.pad_out > 0:
            w_ref = w_ref[:, :-linear_w4.pad_out] # [Din, Dout]
        linear.weight.data = w_ref.t().contiguous()

        y_gt = linear(x)
        q_y = linear_w4.to_seqlen_last(x)
        assert q_y.dtype == torch.float16, f"{q_y.dtype}"
        y_gt = y_gt.transpose(1, 2).contiguous()
        y_gt = y_gt.float()
        y_hat = q_y.float()
        print(y_gt.shape, q_y.shape)
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1
    
    @torch.no_grad()
    def test_W4A8B16O16Linear(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        int8_traits = torch.iinfo(torch.int8)
        # s_x = x.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(torch.float32) # per-token scaling
        s_x = x.abs().amax().div(int8_traits.max).to(torch.float32) # per-tensor scaling
        w4linear = W4A8B16O16Linear.from_fp16(linear, s_x) # hardcoded num_bits=4, group_size=128

        # using the original weight will not pass,
        # so we use the dequantized weight instead
        w_ref, marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel = \
            w4a8_quantize(linear.weight.data.t().contiguous(), num_bits=4, group_size=128, pad_out=w4linear.pad_out) # [Dout, Din] -> [Din, Dout]
        if w4linear.pad_out > 0:
            w_ref = w_ref[:, :-w4linear.pad_out] # [Din, Dout]
        linear.weight.data = w_ref.t().contiguous()  # [Din, Dout] > [Dout, Din]

        xq = (x / s_x).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
        y_gt = linear(xq.half() * s_x.half())
        y_hat = w4linear(xq)
        max_diff = compute_max_diff(y_hat, y_gt)
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1, f"amax={amax}, r2={r2}"


    @torch.no_grad()
    def test_W4A8B16O16Linear_to_seqlen_last(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        int8_traits = torch.iinfo(torch.int8)
        # s_x = x.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(torch.float32) # per-token scaling
        s_x = x.abs().amax().div(int8_traits.max).to(torch.float32) # per-tensor scaling
        w4linear = W4A8B16O16Linear.from_fp16(linear, s_x) # hardcoded num_bits=4, group_size=128

        # using the original weight will not pass,
        # so we use the dequantized weight instead
        w_ref, marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel = \
            w4a8_quantize(linear.weight.data.t().contiguous(), num_bits=4, group_size=128, pad_out=w4linear.pad_out) # [Dout, Din] -> [Din, Dout]
        if w4linear.pad_out > 0:
            w_ref = w_ref[:, :-w4linear.pad_out] # [Din, Dout]
        linear.weight.data = w_ref.t().contiguous()  # [Din, Dout] > [Dout, Din]

        xq = (x / s_x).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
        y_gt = linear(xq.half() * s_x.half())
        y_hat = w4linear.to_seqlen_last(xq)
        y_gt = y_gt.transpose(1, 2).contiguous()
        max_diff = compute_max_diff(y_hat, y_gt)
        amax = (y_gt - y_hat).abs().max()
        r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1, f"amax={amax}, r2={r2}"

    @torch.no_grad()
    def test_W4A8B8O8Linear(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        int8_traits = torch.iinfo(torch.int8)
        # s_x = x.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(torch.float) # per-token scaling
        s_x = x.abs().amax().div(int8_traits.max).to(torch.float) # per-tensor scaling
        xq = (x / s_x).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
        y_gt = linear(xq.half() * s_x.half())
        # s_o = y_gt.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(torch.float) # per-token scaling
        s_o = y_gt.abs().amax().div(int8_traits.max).to(torch.float)
        y_q = (y_gt / s_o).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
        y_gt = y_q.float() * s_o.float()

        w4linear = W4A8B8O8Linear.from_fp16(linear, s_x, s_o) # hardcoded num_bits=4, group_size=128
        y_hat = w4linear(xq)
        y_hat = y_hat.float() * s_o.float()
        max_diff = compute_max_diff(y_hat, y_gt)
        # amax = (y_gt - y_hat).abs().max()
        # r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        # assert r2 < 5e-3 and amax < 0.25, f"amax={amax}, r2={r2}"
        assert  max_diff < 0.08, f"max_diff={max_diff}"

    @torch.no_grad()
    def test_W4A8B8O8Linear_to_seqlen_last(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        int8_traits = torch.iinfo(torch.int8)
        # s_x = x.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(torch.float) # per-token scaling
        s_x = x.abs().amax().div(int8_traits.max).to(torch.float) # per-tensor scaling
        xq = (x / s_x).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
        y_gt = linear(xq.half() * s_x.half())
        # s_o = y_gt.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(torch.float) # per-token scaling
        s_o = y_gt.abs().amax().div(int8_traits.max).to(torch.float)
        y_q = (y_gt / s_o).round().clamp(int8_traits.min, int8_traits.max).to(torch.int8)
        y_gt = y_q.float() * s_o.float()

        w4linear = W4A8B8O8Linear.from_fp16(linear, s_x, s_o) # hardcoded num_bits=4, group_size=128
        y_hat = w4linear.to_seqlen_last(xq)
        y_hat = y_hat.float() * s_o.float()
        y_gt = y_gt.transpose(1, 2).contiguous()
        max_diff = compute_max_diff(y_hat, y_gt)
        # amax = (y_gt - y_hat).abs().max()
        # r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
        # assert r2 < 5e-3 and amax < 0.25, f"amax={amax}, r2={r2}"
        assert  max_diff < 0.08, f"max_diff={max_diff}"

    @torch.no_grad()
    def test_W4A8B8O8LinearParallel(self, batch, seqlen, hidden_dim, model_dim, d_state, nheads):
        x = torch.randn(batch, seqlen, hidden_dim).to(torch.float16).cuda()
        d_in_proj = 2 * model_dim + 2 * d_state + nheads
        linear = torch.nn.Linear(hidden_dim, d_in_proj, bias=False, dtype=torch.float16).cuda()
        zxbcdt = linear(x)
        zxbcdt = zxbcdt.float()
        z, xBC, dt = torch.split(
            zxbcdt,
            [model_dim, model_dim + 2 * d_state, nheads],
            dim=-1
        )
        x_scale = x.clone().abs().max() / 127
        z_scale = z.abs().max() / 127
        xBC_scale = xBC.abs().max() / 127
        dt_scale = dt.abs().max() / 127


        qx = (x.clone() / x_scale).round().to(torch.int8)
        qx = qx.cuda()

        linear_int4 = W4A8B8O8LinearParallel.from_fp16(
                        linear, x_scale.float(), [z_scale, xBC_scale, dt_scale],
                        [model_dim, model_dim + 2 * d_state, nheads]).cuda()

        q_zxbcdt = linear_int4(qx)
        q_z, q_xBC, q_dt = torch.split(
            q_zxbcdt,
            [model_dim, model_dim + 2 * d_state, nheads],
            dim=-1
        )
        z_hat = q_z.float() * z_scale
        max_diff = compute_max_diff(z_hat, z)
        assert  max_diff < 0.095, f"max_diff={max_diff}"

        xBC_hat = q_xBC.float() * xBC_scale
        max_diff = compute_max_diff(xBC_hat, xBC)
        assert  max_diff < 0.095, f"max_diff={max_diff}"

        dt_hat = q_dt.float() * dt_scale
        max_diff = compute_max_diff(dt_hat, dt)
        assert  max_diff < 0.095, f"max_diff={max_diff}"  