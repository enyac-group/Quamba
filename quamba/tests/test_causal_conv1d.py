import pytest
import random
import copy

import torch
import torch.utils.benchmark as benchmark

from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from quamba import QCausalConv1D, Quamb2Conv1D # editable installation
from quamba.quant_utils import quantize_tensor_head_channel_grouping, dequantize_tensor_head_channel_grouping
import quant_causal_conv1d_cuda
import quamba2_conv1d_cuda


torch.manual_seed(1234)
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)


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
        "test_seqlast_forward": [
            dict(batch=1, d_inner=768*2,  conv_bias=True, d_conv=4, seqlen=1024), # mamba-130m
            dict(batch=1, d_inner=2560*2, conv_bias=True, d_conv=4, seqlen=1024), # mamba-2.8b
        ],
        "test_channellast_forward": [
            dict(batch=1, d_inner=768*2,  conv_bias=True, d_conv=4, seqlen=1024), # mamba-130m
            dict(batch=1, d_inner=2560*2, conv_bias=True, d_conv=4, seqlen=1024), # mamba-2.8b
        ],
        "test_causal_conv1d_update": [
            dict(batch=1, d_inner=768*2,  conv_bias=True, d_conv=4), # mamba-130m
            dict(batch=1, d_inner=2560*2, conv_bias=True, d_conv=4), # mamba-2.8b
        ],
        "test_quamba2_conv1d_channellast_fwd": [
            dict(batch=1, x_dim=768*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1, seqlen=1024),   # mamba2-130m
            dict(batch=1, x_dim=2560*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1, seqlen=1024),  # mamba2-2.7B
            dict(batch=16, x_dim=8192, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=8, seqlen=1024),   # mamba2-8B
        ],
        "test_quamba2_conv1d_channellast_group_heads_fwd": [
            dict(batch=1, x_dim=768*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1, seqlen=1024),   # mamba2-130m
            dict(batch=1, x_dim=2560*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1, seqlen=1024),  # mamba2-2.7B
            dict(batch=1, x_dim=8192, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=8, seqlen=1024),   # mamba2-8B
        ],
        "test_quamba2_conv1d_update": [
            dict(batch=1, x_dim=768*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1),   # mamba2-130m
            dict(batch=1, x_dim=2560*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1),  # mamba2-2.7B
            dict(batch=16, x_dim=8192, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=8),   # mamba2-8B
        ],
        "test_quamba2_conv1d_group_heads_update": [
            dict(batch=1, x_dim=768*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1),   # mamba2-130m
            dict(batch=1, x_dim=2560*2, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=1),  # mamba2-2.7B
            dict(batch=16, x_dim=8192, x_headdim=64, d_state=128, conv_bias=True, d_conv=4, n_groups=8),   # mamba2-8B
        ],
    }

    @torch.no_grad()
    def test_seqlast_forward(self, batch, d_inner,  conv_bias, d_conv, seqlen):
        rtol=1e-02
        atol=1e-01

        # random input, sequence length in the last dimension
        x = torch.rand((batch, d_inner, seqlen)).cuda() # B, D, L

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        act = torch.nn.SiLU()
        y = act(conv1d(x)[..., :seqlen])
        y_scale = y.abs().max() / 127
        qy = (y / y_scale).round() * y_scale

        # causal_conv1d
        y_ = causal_conv1d_fn(
            x=x,
            weight=rearrange(conv1d.weight, "d 1 w -> d w"),
            bias=conv1d.bias,
            activation="silu",
        )
        assert torch.allclose(y, y_)

        """
            testing quant_causal_conv1d_cuda
        """
        # quantize weights and bias
        w_scale = conv1d.weight.abs().max() / 127
        qw = (conv1d.weight / w_scale).round().to(torch.int8)
        qw = rearrange(qw, "d 1 w -> d w")
        b_scale = conv1d.bias.abs().max() / 127
        qb = (conv1d.bias / b_scale).round().to(torch.int8)
        # quantize x
        x_scale = x.abs().max() / 127
        qx = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        # run quant_causal_conv1d_cuda.fwd
        y_ = quant_causal_conv1d_cuda.fwd(
                qx, x_scale,
                qw, w_scale,
                y_scale,
                b_scale, qb,
                None, None, None, True
            )
        y_ = y_.float() * y_scale
        assert torch.allclose(qy, y_, rtol=rtol, atol=atol)

        """
            testing QCausalConv1D
        """
        qconv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(conv1d),
            input_scale=x_scale,
            output_scale=y_scale,            
        )
        y_ = qconv1d(qx)
        y_ = y_.float() * y_scale
        assert torch.allclose(qy, y_, rtol=rtol, atol=atol)


    @torch.no_grad()
    def test_channellast_forward(self, batch, d_inner,  conv_bias, d_conv, seqlen):
        rtol=1e-02
        atol=1e-01

        # random input, channel in the last dimension
        x = torch.rand((batch, seqlen, d_inner)).cuda() # B, L, D
        x = x.transpose(1, 2) # (B, L, D) -> (B, D, L), but not contiguous

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        # causal_conv1d
        y_ = causal_conv1d_fn(
            x=x,
            weight=rearrange(conv1d.weight, "d 1 w -> d w").contiguous(),
            bias=conv1d.bias,
            activation="silu",
        )
        y_scale = y_.abs().max() / 127
        qy = (y_ / y_scale).round() * y_scale
        """
            testing quant_causal_conv1d_cuda
        """
        # quantize weights and bias
        w_scale = conv1d.weight.abs().max() / 127
        qw = (conv1d.weight / w_scale).round().to(torch.int8)
        qw = rearrange(qw, "d 1 w -> d w")
        b_scale = conv1d.bias.abs().max() / 127
        qb = (conv1d.bias / b_scale).round().to(torch.int8)
        # quantize x
        x_scale = x.abs().max() / 127
        qx = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        # run quant_causal_conv1d_cuda.fwd
        y_ = quant_causal_conv1d_cuda.fwd(
                qx, x_scale,
                qw, w_scale,
                y_scale,
                b_scale, qb,
                None, None, None, True
            )
        y_ = y_.float() * y_scale
        assert torch.allclose(qy, y_, rtol=rtol, atol=atol)

        """
            testing QCausalConv1D
        """
        qconv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(conv1d),
            input_scale=x_scale,
            output_scale=y_scale,            
        )
        y_ = qconv1d(qx)
        y_ = y_.float() * y_scale
        assert torch.allclose(qy, y_, rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_causal_conv1d_update(self, batch, d_inner,  conv_bias, d_conv):
        rtol=1e-02
        atol=1e-01

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        # create generation inputs: x and conv_state
        x = torch.rand((batch, d_inner)).cuda() # B, D
        conv_state = torch.rand(
            batch,
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
        y_scale = y.abs().max() / 127

        """
            testing quant_causal_conv1d_cuda
        """
        # quantize weights and bias
        w_scale = conv1d.weight.abs().max() / 127
        qw = (conv1d.weight / w_scale).round().to(torch.int8)
        qw = rearrange(qw, "d 1 w -> d w")
        b_scale = conv1d.bias.abs().max() / 127
        qb = (conv1d.bias / b_scale).round().to(torch.int8)
        # quantize x
        x_scale = x.abs().max() / 127
        qx = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        qconv_state = (conv_state.clone() / x_scale).round().clamp(-128, 127).to(torch.int8)
        qc_s = qconv_state.clone()
        # run quant_causal_conv1d_cuda.update
        y_ = quant_causal_conv1d_cuda.update(qx, qc_s, x_scale, qw, w_scale, y_scale, b_scale, qb, True) # update conv_state in-place
        qy_ = y_.float() * y_scale
        qy = (y / y_scale).round().clamp(-128, 127) * y_scale
        assert torch.allclose(qy, qy_, rtol=rtol, atol=atol)

        """
            testing QCausalConv1D
        """
        qconv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(conv1d),
            input_scale=x_scale,
            output_scale=y_scale,            
        )
        qc_s = qconv_state.clone()
        y_ = qconv1d.update(qx, qc_s) # update conv_state in-place
        # check y
        qy_ = y_.float() * y_scale
        qy = (y / y_scale).round().clamp(-128, 127) * y_scale
        assert torch.allclose(qy, qy_, rtol=rtol, atol=atol)
        # check conv_state
        qc_s = qc_s.float() * x_scale
        assert torch.allclose(qc_s, c_s, rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_quamba2_conv1d_channellast_fwd(self, batch, x_dim, x_headdim, d_state, conv_bias, d_conv, n_groups, seqlen):

        rtol=1e-02
        atol=1e-01

        n_head = x_dim // (x_headdim*n_groups)
        d_inner = x_dim + (d_state*n_groups)*2

        # random input, channel in the last dimension
        xBC = torch.rand((batch, seqlen, d_inner)).cuda() # B, L, D
        xBC = xBC.transpose(1, 2) # (B, L, D) -> (B, D, L), but not contiguous

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        # causal_conv1d
        y_ = causal_conv1d_fn(
            x=xBC,
            weight=rearrange(conv1d.weight, "d 1 w -> d w").contiguous(),
            bias=conv1d.bias,
            activation="silu",
        )

        """
            testing quamba2_conv1d_cuda_fwd
        """
        # input scales
        x_in, B_in, C_in = torch.split(xBC, [x_dim, d_state*n_groups, d_state*n_groups], dim=1)
        x_scale = (x_in.abs().max() / 127.).to(torch.float32)
        qx_in = (x_in / x_scale).round().to(torch.int8)
        B_scale = (B_in.abs().max() / 127.).to(torch.float32)
        qB_in = (B_in / B_scale).round().to(torch.int8)
        C_scale = (C_in.abs().max() / 127.).to(torch.float32)
        qC_in = (C_in / C_scale).round().to(torch.int8)
        qxBC = torch.cat([qx_in, qB_in, qC_in], dim=1) # implicit contiguous
        qxBC = qxBC.transpose(1, 2).contiguous().transpose(1, 2) # to channellast
        # output scales and tensors
        x, B, C = torch.split(y_, [x_dim, d_state*n_groups, d_state*n_groups], dim=1)
        # quantize output x
        x_head_group_range = None
        x_dim_group_range = None
        x_out_scale = (x.abs().max() / 127.).to(torch.float32)
        qx_gt = (x / x_out_scale).clamp(-128, 127).round() * x_out_scale
        # quantize output B
        B_out_scale = B.abs().amax().to(torch.float32) / 127.
        qB_gt = ((rearrange(B, "b (g d) l -> b g d l", g=n_groups, d=d_state) / B_out_scale).clamp(-128, 127).round()) * B_out_scale
        qB_gt = rearrange(qB_gt, "b g d l -> b (g d) l")
        # quantize output C
        C_out_scale = C.abs().amax().to(torch.float32) / 127.
        qC_gt = ((rearrange(C, "b (g d) l -> b g d l", g=n_groups, d=d_state) / C_out_scale).clamp(-128, 127).round()) * C_out_scale
        qC_gt = rearrange(qC_gt, "b g d l -> b (g d) l")
        qumaba2_conv = Quamb2Conv1D.from_fp16(
                conv1d,
                x_dim, x_headdim, d_state, n_groups,
                x_scale, B_scale, C_scale,
                x_out_scale,       # [n_ssd_groups, n_head_groups, n_dim_groups]
                B_out_scale,
                C_out_scale,
                x_head_group_range, # [n_ssd_groups, n_head_groups]
                x_dim_group_range,  # [n_ssd_groups, n_dim_groups]
            )
        qx, qB, qC = qumaba2_conv(qxBC)

        # check B and C
        qB = rearrange(qB, "b (g d) l -> b g d l", g=n_groups, d=d_state).float() * B_out_scale
        qB = rearrange(qB, "b g d l -> b (g d) l")
        amax = (qB - qB_gt).abs().max()
        r2 = (qB - qB_gt).pow(2).mean() / qB_gt.pow(2).mean()
        assert torch.allclose(qB, qB_gt, rtol=rtol, atol=atol)
        qC = rearrange(qC, "b (g d) l -> b g d l", g=n_groups, d=d_state).float() * C_out_scale
        qC = rearrange(qC, "b g d l -> b (g d) l")
        amax = (qC - qC_gt).abs().max()
        r2 = (qC - qC_gt).pow(2).mean() / qC_gt.pow(2).mean()
        assert torch.allclose(qC, qC_gt, rtol=rtol, atol=atol)
        # check x
        qx = qx.float() * x_out_scale
        assert torch.allclose(qx, qx_gt, rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_quamba2_conv1d_channellast_group_heads_fwd(self, batch, x_dim, x_headdim, d_state, conv_bias, d_conv, n_groups, seqlen):

        rtol=1e-02
        atol=1e-01

        n_head = x_dim // (x_headdim*n_groups)
        d_inner = x_dim + (d_state*n_groups)*2

        # random input, channel in the last dimension
        xBC = torch.rand((batch, seqlen, d_inner)).cuda() # B, L, D
        xBC = xBC.transpose(1, 2) # (B, L, D) -> (B, D, L), but not contiguous

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        # causal_conv1d
        y_ = causal_conv1d_fn(
            x=xBC,
            weight=rearrange(conv1d.weight, "d 1 w -> d w").contiguous(),
            bias=conv1d.bias,
            activation="silu",
        )

        """
            testing quamba2_conv1d_cuda_fwd
        """
        # input scales
        x_in, B_in, C_in = torch.split(xBC, [x_dim, d_state*n_groups, d_state*n_groups], dim=1)
        x_scale = (x_in.abs().max() / 127.).to(torch.float32)
        qx_in = (x_in / x_scale).round().to(torch.int8)
        B_scale = (B_in.abs().max() / 127.).to(torch.float32)
        qB_in = (B_in / B_scale).round().to(torch.int8)
        C_scale = (C_in.abs().max() / 127.).to(torch.float32)
        qC_in = (C_in / C_scale).round().to(torch.int8)
        qxBC = torch.cat([qx_in, qB_in, qC_in], dim=1) # implicit contiguous
        qxBC = qxBC.transpose(1, 2).contiguous().transpose(1, 2) # to channellast
        # output scales and tensors
        x, B, C = torch.split(y_, [x_dim, d_state*n_groups, d_state*n_groups], dim=1)
        # quantize output x
        nhead_group = 4
        nhead_group_size = n_head // nhead_group
        x_head_group_range = torch.arange(nhead_group_size, n_head+nhead_group_size, nhead_group_size).to(torch.int32).cuda()
        x_head_group_range = x_head_group_range.repeat(n_groups, 1) # [n_ssd_groups, n_head_groups]
        ndim_group = 4
        ndim_group_size = x_headdim // ndim_group
        x_dim_group_range = torch.arange(ndim_group_size, x_headdim+ndim_group_size, ndim_group_size).to(torch.int32).cuda()
        x_dim_group_range = x_dim_group_range.repeat(n_groups, nhead_group, 1) # [n_ssd_groups, nhead_group, n_dim_groups]

        x_t = x.transpose(1, 2) # B, D, L -> B, L, D
        x_reshape = x_t.reshape(batch, seqlen, -1, x_headdim) # x_reshape = [b, l, nh, hd]
        qx_gt, x_out_scales = quantize_tensor_head_channel_grouping(
            x_reshape, x_head_group_range, x_dim_group_range, n_bits=8,
            scales=None, fake_quant=True, clip_ratio=1.0)
        qx_gt = qx_gt.reshape(batch, seqlen, x_dim).transpose(1, 2) # B, L, D -> B, D, L
        assert torch.allclose(qx_gt, x, rtol=rtol, atol=atol)

        # quantize output B
        B_reshape = rearrange(B, "b (g d) l -> g b d l", g=n_groups, d=d_state)
        B_out_scale = B_reshape.reshape(n_groups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        qB_gt = ((rearrange(B, "b (g d) l -> b g d l", g=n_groups, d=d_state) / B_out_scale[None, :, None, None]).clamp(-128, 127).round()) * B_out_scale[None, :, None, None]
        qB_gt = rearrange(qB_gt, "b g d l -> b (g d) l")
        # quantize output C
        C_reshape = rearrange(C, "b (g d) l -> g b d l", g=n_groups, d=d_state)
        C_out_scale = C_reshape.reshape(n_groups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        qC_gt = ((rearrange(C, "b (g d) l -> b g d l", g=n_groups, d=d_state) / C_out_scale[None, :, None, None]).clamp(-128, 127).round()) * C_out_scale[None, :, None, None]
        qC_gt = rearrange(qC_gt, "b g d l -> b (g d) l")

        qumaba2_conv = Quamb2Conv1D.from_fp16(
                conv1d,
                x_dim, x_headdim, d_state, n_groups,
                x_scale, B_scale, C_scale,
                x_out_scales,       # [n_ssd_groups, n_head_groups, n_dim_groups]
                B_out_scale,
                C_out_scale,
                x_head_group_range, # [n_ssd_groups, n_head_groups]
                x_dim_group_range,  # [n_ssd_groups, n_dim_groups]
            )
        qx, qB, qC = qumaba2_conv(qxBC)

        # check B and C
        qB = rearrange(qB, "b (g d) l -> b g d l", g=n_groups, d=d_state).float() * B_out_scale[None, :, None, None]
        qB = rearrange(qB, "b g d l -> b (g d) l")
        amax = (qB - qB_gt).abs().max()
        r2 = (qB - qB_gt).pow(2).mean() / qB_gt.pow(2).mean()
        assert torch.allclose(qB, qB_gt, rtol=rtol, atol=atol)
        qC = rearrange(qC, "b (g d) l -> b g d l", g=n_groups, d=d_state).float() * C_out_scale[None, :, None, None]
        qC = rearrange(qC, "b g d l -> b (g d) l")
        amax = (qC - qC_gt).abs().max()
        r2 = (qC - qC_gt).pow(2).mean() / qC_gt.pow(2).mean()
        assert torch.allclose(qC, qC_gt, rtol=rtol, atol=atol)
        # check x
        qx_t = qx.transpose(1, 2) # B, D, L -> B, L, D
        qx_reshape = qx_t.reshape(batch, seqlen, -1, x_headdim)
        qx = dequantize_tensor_head_channel_grouping(
            qx_reshape, x_head_group_range, x_dim_group_range, x_out_scales)
        qx = qx.reshape(batch, seqlen, x_dim).transpose(1, 2) # B, L, D -> B, D, L
        amax = (qx - qx_gt).abs().max()
        r2 = (qx - qx_gt).pow(2).mean() / qx_gt.pow(2).mean()
        assert amax < 0.1, f"amax = {amax}"
        assert r2 < 0.001, f"r2 = {r2}"
        assert torch.allclose(qx, qx_gt, rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_quamba2_conv1d_update(self, batch, x_dim, x_headdim, d_state, conv_bias, d_conv, n_groups):
        rtol=1e-02
        atol=1e-01

        n_head = x_dim // (x_headdim*n_groups)
        d_inner = x_dim + (d_state*n_groups)*2

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        conv1d.weight.data = torch.ones_like(conv1d.weight)
        conv1d.bias.data = torch.zeros_like(conv1d.bias)
        
        # create generation inputs: x and conv_state
        # xBC = torch.ones((batch, d_inner)).cuda() # B, D
        xBC = torch.rand((batch, d_inner)).cuda()*2 # B, D
        conv_state = torch.ones(
        # conv_state = torch.rand(
            batch,
            d_inner,
            d_conv,
            device=conv1d.weight.device,
            dtype=conv1d.weight.dtype,
        )
        c_s = conv_state.clone()
        # update conv_state in-place
        y = causal_conv1d_update(
            xBC,
            c_s,
            rearrange(conv1d.weight, "d 1 w -> d w"),
            conv1d.bias,
            activation="silu",
        )

        """
            testing quamba2_conv1d_cuda_update
        """
        # input scales
        x_in, B_in, C_in = torch.split(xBC, [x_dim, d_state*n_groups, d_state*n_groups], dim=1) # [B, D]
        x_scale = (x_in.abs().max() / 127.).to(torch.float32)
        qx_in = (x_in / x_scale).round().to(torch.int8)
        B_scale = (B_in.abs().max() / 127.).to(torch.float32)
        qB_in = (B_in / B_scale).round().to(torch.int8)
        C_scale = (C_in.abs().max() / 127.).to(torch.float32)
        qC_in = (C_in / C_scale).round().to(torch.int8)
        print(qB_in)
        qxBC = torch.cat([qx_in, qB_in, qC_in], dim=1) # implicit contiguous

        # quant conv_state
        qconv_state = conv_state.clone()
        x_cs_in, B_cs_in, C_cs_in = torch.split(qconv_state, [x_dim, d_state*n_groups, d_state*n_groups], dim=1) # [B, D]
        qx_cs_in = (x_cs_in / x_scale).clamp(-128, 127).round().to(torch.int8)
        qB_cs_in = (B_cs_in / B_scale).clamp(-128, 127).round().to(torch.int8)
        qC_cs_in = (C_cs_in / C_scale).clamp(-128, 127).round().to(torch.int8)
        qconv_state = torch.cat([qx_cs_in, qB_cs_in, qC_cs_in], dim=1) # implicit contiguous

        # output scales and tensors
        x, B, C = torch.split(y, [x_dim, d_state*n_groups, d_state*n_groups], dim=1)
        # quantize output x
        x_head_group_range = None
        x_dim_group_range = None
        x_out_scale = (x.abs().max() / 127.).to(torch.float32)
        qx_gt = (x / x_out_scale).clamp(-128, 127).round() * x_out_scale
        # quantize output B
        B_out_scale = B.abs().amax().to(torch.float32) / 127.
        qB_gt = ((rearrange(B, "b (g d) -> b g d", g=n_groups, d=d_state) / B_out_scale).clamp(-128, 127).round()) * B_out_scale
        qB_gt = rearrange(qB_gt, "b g d -> b (g d)")
        # quantize output C
        C_out_scale = C.abs().amax().to(torch.float32) / 127.
        qC_gt = ((rearrange(C, "b (g d) -> b g d", g=n_groups, d=d_state) / C_out_scale).clamp(-128, 127).round()) * C_out_scale
        qC_gt = rearrange(qC_gt, "b g d -> b (g d)")
        """
            testing Quamb2Conv1D
        """
        qumaba2_conv = Quamb2Conv1D.from_fp16(
                copy.deepcopy(conv1d),
                x_dim, x_headdim, d_state, n_groups,
                x_scale, B_scale, C_scale,
                x_out_scale,       # [n_ssd_groups, n_head_groups, n_dim_groups]
                B_out_scale,
                C_out_scale,
                x_head_group_range, # [n_ssd_groups, n_head_groups]
                x_dim_group_range,  # [n_ssd_groups, n_dim_groups]
            )
        
        qc_s = qconv_state.clone()
        qx, qB, qC = qumaba2_conv.update(qxBC, qc_s) # update conv_state in-place

        # check B and C
        qB = rearrange(qB, "b (g d) -> b g d", g=n_groups, d=d_state).float() * B_out_scale
        qB = rearrange(qB, "b g d -> b (g d)")
        amax = (qB - qB_gt).abs().max()
        r2 = (qB - qB_gt).pow(2).mean() / qB_gt.pow(2).mean()
        assert torch.allclose(qB, qB_gt, rtol=rtol, atol=atol)
        qC = rearrange(qC, "b (g d) -> b g d", g=n_groups, d=d_state).float() * C_out_scale
        qC = rearrange(qC, "b g d -> b (g d)")
        amax = (qC - qC_gt).abs().max()
        r2 = (qC - qC_gt).pow(2).mean() / qC_gt.pow(2).mean()
        assert torch.allclose(qC, qC_gt, rtol=rtol, atol=atol)
        # check x
        qx = qx.float() * x_out_scale
        assert torch.allclose(qx, qx_gt, rtol=rtol, atol=atol)
        # check conv_state
        x_cs_in, B_cs_in, C_cs_in = torch.split(qc_s, [x_dim, d_state*n_groups, d_state*n_groups], dim=1) # [B, D]
        qx_cs_in = x_cs_in.float() * x_scale
        qB_cs_in = B_cs_in.float() * B_scale
        qC_cs_in = C_cs_in.float() * C_scale
        qc_s = torch.cat([qx_cs_in, qB_cs_in, qC_cs_in], dim=1) # implicit contiguous
        assert torch.allclose(qc_s, c_s, rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_quamba2_conv1d_group_heads_update(self, batch, x_dim, x_headdim, d_state, conv_bias, d_conv, n_groups):
        rtol=1e-02
        atol=1e-01

        n_head = x_dim // (x_headdim*n_groups)
        d_inner = x_dim + (d_state*n_groups)*2

        # pytorch convolution
        conv1d = torch.nn.Conv1d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=d_inner,
                    padding=d_conv - 1,
                ).cuda()
        conv1d.weight.data = torch.ones_like(conv1d.weight)
        conv1d.bias.data = torch.zeros_like(conv1d.bias)
        
        # create generation inputs: x and conv_state
        # xBC = torch.ones((batch, d_inner)).cuda() # B, D
        xBC = torch.rand((batch, d_inner)).cuda()*2 # B, D
        conv_state = torch.ones(
        # conv_state = torch.rand(
            batch,
            d_inner,
            d_conv,
            device=conv1d.weight.device,
            dtype=conv1d.weight.dtype,
        )
        c_s = conv_state.clone()
        # update conv_state in-place
        y = causal_conv1d_update(
            xBC,
            c_s,
            rearrange(conv1d.weight, "d 1 w -> d w"),
            conv1d.bias,
            activation="silu",
        )

        """
            testing quamba2_conv1d_cuda_update
        """
        # input scales
        x_in, B_in, C_in = torch.split(xBC, [x_dim, d_state*n_groups, d_state*n_groups], dim=1) # [B, D]
        x_scale = (x_in.abs().max() / 127.).to(torch.float32)
        qx_in = (x_in / x_scale).round().to(torch.int8)
        B_scale = (B_in.abs().max() / 127.).to(torch.float32)
        qB_in = (B_in / B_scale).round().to(torch.int8)
        C_scale = (C_in.abs().max() / 127.).to(torch.float32)
        qC_in = (C_in / C_scale).round().to(torch.int8)
        qxBC = torch.cat([qx_in, qB_in, qC_in], dim=1) # implicit contiguous

        # quant conv_state
        qconv_state = conv_state.clone()
        x_cs_in, B_cs_in, C_cs_in = torch.split(qconv_state, [x_dim, d_state*n_groups, d_state*n_groups], dim=1) # [B, D]
        qx_cs_in = (x_cs_in / x_scale).clamp(-128, 127).round().to(torch.int8)
        qB_cs_in = (B_cs_in / B_scale).clamp(-128, 127).round().to(torch.int8)
        qC_cs_in = (C_cs_in / C_scale).clamp(-128, 127).round().to(torch.int8)
        qconv_state = torch.cat([qx_cs_in, qB_cs_in, qC_cs_in], dim=1) # implicit contiguous

        # output scales and tensors
        x, B, C = torch.split(y, [x_dim, d_state*n_groups, d_state*n_groups], dim=1)
        # quantize output x
        nhead_group = 4
        nhead_group_size = n_head // nhead_group
        x_head_group_range = torch.arange(nhead_group_size, n_head+nhead_group_size, nhead_group_size).to(torch.int32).cuda()
        x_head_group_range = x_head_group_range.repeat(n_groups, 1) # [n_ssd_groups, n_head_groups]
        ndim_group = 4
        ndim_group_size = x_headdim // ndim_group
        x_dim_group_range = torch.arange(ndim_group_size, x_headdim+ndim_group_size, ndim_group_size).to(torch.int32).cuda()
        x_dim_group_range = x_dim_group_range.repeat(n_groups, nhead_group, 1) # [n_ssd_groups, n_dim_groups]

        x_reshape = x.reshape(batch, -1, x_headdim) # x_reshape = [b, nh, hd]
        qx_gt, x_out_scales = quantize_tensor_head_channel_grouping(
            x_reshape, x_head_group_range, x_dim_group_range, n_bits=8,
            scales=None, fake_quant=True, clip_ratio=1.0)
        qx_gt = qx_gt.reshape(batch, x_dim)
        assert torch.allclose(qx_gt, x, rtol=rtol, atol=atol)
        # quantize output B
        B_reshape = rearrange(B, "b (g d) -> g b d", g=n_groups, d=d_state)
        B_out_scale = B_reshape.reshape(n_groups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        qB_gt = ((rearrange(B, "b (g d) -> b g d", g=n_groups, d=d_state) / B_out_scale[None, :, None]).clamp(-128, 127).round()) * B_out_scale[None, :, None]
        qB_gt = rearrange(qB_gt, "b g d -> b (g d)")
        # quantize output C
        C_reshape = rearrange(C, "b (g d) -> g b d", g=n_groups, d=d_state)
        C_out_scale = C_reshape.reshape(n_groups, -1).abs().amax(dim=1).to(torch.float32) / 127.
        qC_gt = ((rearrange(C, "b (g d) -> b g d", g=n_groups, d=d_state) / C_out_scale[None, :, None]).clamp(-128, 127).round()) * C_out_scale[None, :, None]
        qC_gt = rearrange(qC_gt, "b g d -> b (g d)")
        """
            testing Quamb2Conv1D
        """
        qumaba2_conv = Quamb2Conv1D.from_fp16(
                copy.deepcopy(conv1d),
                x_dim, x_headdim, d_state, n_groups,
                x_scale, B_scale, C_scale,
                x_out_scales,       # [n_ssd_groups, n_head_groups, n_dim_groups]
                B_out_scale,
                C_out_scale,
                x_head_group_range, # [n_ssd_groups, n_head_groups]
                x_dim_group_range,  # [n_ssd_groups, n_dim_groups]
            )
        
        qc_s = qconv_state.clone()
        qx, qB, qC = qumaba2_conv.update(qxBC, qc_s) # update conv_state in-place

        # check B and C
        qB = rearrange(qB, "b (g d) -> b g d", g=n_groups, d=d_state).float() * B_out_scale[None, :, None]
        qB = rearrange(qB, "b g d -> b (g d)")
        amax = (qB - qB_gt).abs().max()
        r2 = (qB - qB_gt).pow(2).mean() / qB_gt.pow(2).mean()
        assert torch.allclose(qB, qB_gt, rtol=rtol, atol=atol)
        qC = rearrange(qC, "b (g d) -> b g d", g=n_groups, d=d_state).float() * C_out_scale[None, :, None]
        qC = rearrange(qC, "b g d -> b (g d)")
        amax = (qC - qC_gt).abs().max()
        r2 = (qC - qC_gt).pow(2).mean() / qC_gt.pow(2).mean()
        assert torch.allclose(qC, qC_gt, rtol=rtol, atol=atol)
        # check x
        qx = dequantize_tensor_head_channel_grouping(
            qx.reshape(batch, -1, x_headdim), x_head_group_range, x_dim_group_range, x_out_scales)
        qx = qx.reshape(batch, x_dim)
        assert torch.allclose(qx, qx_gt, rtol=rtol, atol=atol)
        # check conv_state
        x_cs_in, B_cs_in, C_cs_in = torch.split(qc_s, [x_dim, d_state*n_groups, d_state*n_groups], dim=1) # [B, D]
        qx_cs_in = x_cs_in.float() * x_scale
        qB_cs_in = B_cs_in.float() * B_scale
        qC_cs_in = C_cs_in.float() * C_scale
        qc_s = torch.cat([qx_cs_in, qB_cs_in, qC_cs_in], dim=1) # implicit contiguous
        assert torch.allclose(qc_s, c_s, rtol=rtol, atol=atol)
