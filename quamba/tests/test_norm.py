import torch

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from quamba import QRMSNorm

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
        "test_QRMSNorm_StaticPerTensor": [
            dict(batch=1, seqlen=33, dim=768*2, has_residual=True),    # mamba-130m
            dict(batch=4, seqlen=1024, dim=768*2, has_residual=False),  # mamba-130m
            dict(batch=1, seqlen=44, dim=2560*2, has_residual=False),   # mamba-2.8b
            dict(batch=4, seqlen=1024, dim=2560*2, has_residual=True), # mamba-2.8b
        ],
        "test_QRMSNorm_DynamicPerToken": [
            dict(batch=1, seqlen=33, dim=768*2, has_residual=True),    # mamba-130m
            dict(batch=4, seqlen=1024, dim=768*2, has_residual=False),  # mamba-130m
            dict(batch=1, seqlen=44, dim=2560*2, has_residual=False),   # mamba-2.8b
            dict(batch=4, seqlen=1024, dim=2560*2, has_residual=True), # mamba-2.8b
        ],
    }
    def test_QRMSNorm_StaticPerTensor(self, batch, seqlen, dim, has_residual):
        dtype = torch.float16
        rtol=1e-03
        atol=1e-02
        torch.manual_seed(1234)

        prenorm=True # return y and residual
        residual_in_fp32=False
        x = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
        if has_residual:
            residual = None
        else:
            residual = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()

        norm = RMSNorm(dim, eps=1e-5).cuda()
        with torch.no_grad():
            y, res = norm(x, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)
        out_scale = y.abs().max() / 127.

        qnorm = QRMSNorm.from_fp16(norm, output_scale=out_scale.item()).cuda()
        with torch.no_grad():
            q_y, res_ = qnorm(x, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)

        y_ = q_y * out_scale
        assert torch.allclose(y.float(), y_.float(), rtol=rtol, atol=atol)
        assert torch.allclose(res.float(), res_.float(), rtol=rtol, atol=atol)

    def test_QRMSNorm_DynamicPerToken(self, batch, seqlen, dim, has_residual):
        dtype = torch.float16
        rtol=1e-03
        atol=1e-02
        torch.manual_seed(1234)

        prenorm=True # return y and residual
        residual_in_fp32=False
        x = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()
        if has_residual:
            residual = None
        else:
            residual = torch.rand((batch, seqlen, dim), dtype=dtype).cuda()

        norm = RMSNorm(dim, eps=1e-5).cuda()
        with torch.no_grad():
            y, res = norm(x, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)

        qnorm = QRMSNorm.from_fp16(norm, output_scale=None).cuda()
        with torch.no_grad():
            q_y, res_, per_token_scale = qnorm(x, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)

        y_ = q_y * per_token_scale[..., None]
        assert torch.allclose(y.float(), y_.float(), rtol=rtol, atol=atol)
        assert (y-y_).abs().max() < 0.01, f"(y-y_).abs().max() = {(y-y_).abs().max()}"
        assert torch.allclose(res.float(), res_.float(), rtol=rtol, atol=atol)
        assert (res-res_).abs().max() < 0.001, f"(res-res_).abs().max() = {(res-res_).abs().max()}"
