import torch
import numpy as np

from quamba import W8O16Embedding, W4O16Embedding

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
        "test_batched_w8o16_qembedding": [
            dict(batch=1, seqlen=33, vocab_size=50288, embedding_dim=768),      # mamba2-130m
            dict(batch=16, seqlen=33, vocab_size=50288, embedding_dim=2560),    # mamba2-2.7b
            dict(batch=16, seqlen=1024, vocab_size=256000, embedding_dim=4096), # mamba2-8b
        ],
        "test_w8o16_qembedding": [
            dict(seqlen=33, vocab_size=50288, embedding_dim=768),       # mamba2-130m
            dict(seqlen=33, vocab_size=50288, embedding_dim=2560),      # mamba2-2.7b
            dict(seqlen=1024, vocab_size=256000, embedding_dim=4096),   # mamba2-8b
        ],
        "test_batched_w4o16_qembedding": [
            dict(batch=1, seqlen=33, vocab_size=50288, embedding_dim=768),      # mamba2-130m
            dict(batch=16, seqlen=33, vocab_size=50288, embedding_dim=2560),    # mamba2-2.7b
            dict(batch=16, seqlen=1024, vocab_size=256000, embedding_dim=4096), # mamba2-8b
        ],
        "test_w4o16_qembedding": [
            dict(seqlen=33, vocab_size=50288, embedding_dim=768),       # mamba2-130m
            dict(seqlen=33, vocab_size=50288, embedding_dim=2560),      # mamba2-2.7b
            dict(seqlen=1024, vocab_size=256000, embedding_dim=4096),   # mamba2-8b
        ],
    }

    def test_batched_w8o16_qembedding(self, batch, seqlen, vocab_size, embedding_dim):

        dtype = torch.float16
        device = torch.device("cuda")
        embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype, device=device)
        quant_embedding = W8O16Embedding.from_fp16(embedding)

        embed_w = embedding.weight.data
        token_scale = embed_w.amax(dim=-1, keepdim=True) / 127.
        embed_i8 = torch.round(embed_w / token_scale).clamp(-128, 127).to(torch.int8)
        embed_dqw = (embed_i8 * token_scale).to(torch.float16)

        x_idx = torch.randint(low=0, high=vocab_size, size=(batch, seqlen), dtype=torch.int64, device=device)
        x_gt = torch.nn.functional.embedding(x_idx, embed_dqw)
        x_q = quant_embedding(x_idx)
        amax = (x_gt - x_q).abs().max()
        r2 = (x_gt - x_q).pow(2).mean() / x_gt.pow(2).mean()
        assert torch.allclose(x_gt, x_q), f"amax = {amax}, r2 = {r2}"

    def test_w8o16_qembedding(self, seqlen, vocab_size, embedding_dim):

        dtype = torch.float16
        device = torch.device("cuda")
        embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype, device=device)
        quant_embedding = W8O16Embedding.from_fp16(embedding)

        embed_w = embedding.weight.data
        token_scale = embed_w.amax(dim=-1, keepdim=True) / 127.
        embed_i8 = torch.round(embed_w / token_scale).clamp(-128, 127).to(torch.int8)
        embed_dqw = (embed_i8 * token_scale).to(torch.float16)

        x_idx = torch.randint(low=0, high=vocab_size, size=(seqlen,), dtype=torch.int64, device=device)
        x_gt = torch.nn.functional.embedding(x_idx, embed_dqw)
        x_q = quant_embedding(x_idx)
        amax = (x_gt - x_q).abs().max()
        r2 = (x_gt - x_q).pow(2).mean() / x_gt.pow(2).mean()
        assert torch.allclose(x_gt, x_q), f"amax = {amax}, r2 = {r2}"


    def test_batched_w4o16_qembedding(self, batch, seqlen, vocab_size, embedding_dim):

        dtype = torch.float16
        device = torch.device("cuda")
        embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype, device=device)
        quant_embedding = W4O16Embedding.from_fp16(embedding)

        embed_w = embedding.weight.data
        num_bits = 4
        max_val = 2**num_bits # 16
        zero_point = 2**(num_bits-1) # 8
        token_scale = embed_w.amax(dim=-1, keepdim=True) / (max_val / 2)
        embed_dqw = torch.round((embed_w / token_scale) + zero_point).clamp(0, max_val-1)
        embed_dqw = (embed_dqw - zero_point) * token_scale
        
        x_idx = torch.randint(low=0, high=vocab_size, size=(batch, seqlen), dtype=torch.int64, device=device)
        x_gt = torch.nn.functional.embedding(x_idx, embed_dqw)
        x_q = quant_embedding(x_idx)
        amax = (x_gt - x_q).abs().max()
        r2 = (x_gt - x_q).pow(2).mean() / x_gt.pow(2).mean()
        assert torch.allclose(x_gt, x_q), f"amax = {amax}, r2 = {r2}"

    def test_w4o16_qembedding(self, seqlen, vocab_size, embedding_dim):

        dtype = torch.float16
        device = torch.device("cuda")
        embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype, device=device)
        quant_embedding = W4O16Embedding.from_fp16(embedding)

        embed_w = embedding.weight.data
        num_bits = 4
        max_val = 2**num_bits # 16
        zero_point = 2**(num_bits-1) # 8
        token_scale = embed_w.amax(dim=-1, keepdim=True) / (max_val / 2)
        embed_dqw = torch.round((embed_w / token_scale) + zero_point).clamp(0, max_val-1)
        embed_dqw = (embed_dqw - zero_point) * token_scale
        
        x_idx = torch.randint(low=0, high=vocab_size, size=(seqlen,), dtype=torch.int64, device=device)
        x_gt = torch.nn.functional.embedding(x_idx, embed_dqw)
        x_q = quant_embedding(x_idx)
        amax = (x_gt - x_q).abs().max()
        r2 = (x_gt - x_q).pow(2).mean() / x_gt.pow(2).mean()
        assert torch.allclose(x_gt, x_q), f"amax = {amax}, r2 = {r2}"
