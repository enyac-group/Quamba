import torch

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from quamba import QRMSNormGated

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
        "test_QNormGated_FP16Output": [
            dict(batch=1, seqlen=33, dim=768*2, with_z=True),    # mamba-130m
            dict(batch=4, seqlen=1024, dim=768*2, with_z=False),  # mamba-130m
            dict(batch=1, seqlen=44, dim=2560*2, with_z=False),   # mamba-2.8b
            dict(batch=4, seqlen=1024, dim=2560*2, with_z=True), # mamba-2.8b
        ],
        "test_QNormGated_StaticPerTensor": [
            dict(batch=1, seqlen=33, dim=768*2, with_z=True),    # mamba-130m
            dict(batch=4, seqlen=1024, dim=768*2, with_z=False),  # mamba-130m
            dict(batch=1, seqlen=44, dim=2560*2, with_z=False),   # mamba-2.8b
            dict(batch=4, seqlen=1024, dim=2560*2, with_z=True), # mamba-2.8b
        ],
        "test_QNormGated_DynamicPerToken": [
            dict(batch=1, seqlen=33, dim=768*2, with_z=True),    # mamba-130m
            dict(batch=4, seqlen=1024, dim=768*2, with_z=False),  # mamba-130m
            dict(batch=1, seqlen=44, dim=2560*2, with_z=False),   # mamba-2.8b
            dict(batch=4, seqlen=1024, dim=2560*2, with_z=True), # mamba-2.8b
        ],
    }

    def test_QNormGated_FP16Output(self, batch, seqlen, dim, with_z):
        rtol=1e-03
        atol=1e-02
        torch.manual_seed(1234)

        dtype = torch.float16
        ngroups = 1
        norm_before_gate=False

        x = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
        if with_z:
            z = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
            z_scale = (z.float().abs().max() / 127.).item()
            q_z = (z.float() / z_scale).round().clip(-128, 127).to(torch.int8)
        else:
            z = None
            z_scale = None
            q_z = None

        norm = RMSNormGated(
                    dim, eps=1e-5,
                    norm_before_gate=norm_before_gate,
                    group_size=dim // ngroups).cuda()
        with torch.no_grad():
            y = norm(x, z=z)

        # qnorm = QRMSNormGated(norm, z_scale, use_float16_output=True).cuda()
        qnorm = QRMSNormGated.from_fp16(norm, z_scale, use_float16_output=True).cuda()
        with torch.no_grad():
            y_ = qnorm(x, q_z=q_z)

        amax = (y - y_).abs().max()
        r2 = (y - y_).pow(2).mean() / y.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1
        # assert torch.allclose(y.float(), y_.float(), rtol=rtol, atol=atol)  # this will not pass if adding q_z

    def test_QNormGated_StaticPerTensor(self, batch, seqlen, dim, with_z):
        rtol=1e-03
        atol=1e-02
        torch.manual_seed(1234)

        dtype = torch.float16
        ngroups = 1
        norm_before_gate=False

        x = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
        if with_z:
            z = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
            z_scale = (z.float().abs().max() / 127.).item()
            q_z = (z.float() / z_scale).round().clip(-128, 127).to(torch.int8)
        else:
            z = None
            z_scale = None
            q_z = None

        norm = RMSNormGated(dim, eps=1e-5, norm_before_gate=norm_before_gate, group_size=dim // ngroups).cuda()
        with torch.no_grad():
            y = norm(x, z=z)

        out_scale = (y.float().abs().max() / 127.).item()
        qnorm = QRMSNormGated.from_fp16(norm, z_scale, out_scale).cuda()
        with torch.no_grad():
            q_y = qnorm(x, q_z=q_z)

        y_ = q_y * out_scale
        amax = (y - y_).abs().max()
        r2 = (y - y_).pow(2).mean() / y.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1
        # assert torch.allclose(y.float(), y_.float(), rtol=rtol, atol=atol)  # this will not pass if adding q_z

    def test_QNormGated_DynamicPerToken(self, batch, seqlen, dim, with_z):
        rtol=1e-03
        atol=1e-02
        torch.manual_seed(1234)

        dtype = torch.float16
        ngroups = 1
        norm_before_gate=False

        x = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
        if with_z:
            z = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
            z_scale = (z.float().abs().max() / 127.).item()
            q_z = (z.float() / z_scale).round().clip(-128, 127).to(torch.int8)
        else:
            z = None
            z_scale = None
            q_z = None

        norm = RMSNormGated(dim, eps=1e-5, norm_before_gate=norm_before_gate, group_size=dim // ngroups).cuda()
        with torch.no_grad():
            y = norm(x, z=z)

        qnorm = QRMSNormGated.from_fp16(norm, z_scale).cuda()
        with torch.no_grad():
            q_y, token_scale = qnorm(x, q_z=q_z)

        y_ = q_y * token_scale[..., None]
        amax = (y - y_).abs().max()
        r2 = (y - y_).pow(2).mean() / y.pow(2).mean()
        assert r2 < 1e-3 and amax < 0.1
        # assert torch.allclose(y.float(), y_.float(), rtol=rtol, atol=atol) # this will not pass if adding q_z
