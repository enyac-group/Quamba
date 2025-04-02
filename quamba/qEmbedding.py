import numpy as np

import torch
import torch.nn as nn

import quant_embedding_cuda

class W4O16Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.num_bits = 4
        self.pack_factor = 32 // self.num_bits # 8
        # to adapt next(iter(model.parameters())) in many places, we use parameter instead of buffer here
        self.register_parameter('weight', nn.Parameter(
            torch.empty((num_embeddings, embedding_dim // self.pack_factor),
                        dtype=torch.int32, **factory_kwargs),
            requires_grad=False))
        self.register_buffer('scales', torch.empty((num_embeddings),
            dtype=torch.float16, **factory_kwargs))

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear):
        device = originalLayer.weight.device
        qembed = cls(originalLayer.num_embeddings, originalLayer.embedding_dim, device=device)
        embed_w = originalLayer.weight.data
        vocab_size = originalLayer.num_embeddings
        max_val = 2**qembed.num_bits # 16
        zero_point = 2**(qembed.num_bits-1) # 8
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])

        with torch.no_grad():
            scales = embed_w.amax(dim=-1, keepdim=True) / (max_val / 2)
            embed_qw = torch.round((embed_w / scales) + zero_point).clamp(0, max_val-1)
            embed_qw = embed_qw.cpu().to(torch.int32).numpy().astype(np.uint32) # move to cpu to save memory
            embed_qw = embed_qw.reshape(vocab_size, -1, 8)
            embed_qw = embed_qw[:, :, interleave].reshape(vocab_size, -1)
            embed_qw_packed = np.zeros((embed_qw.shape[0], embed_qw.shape[1] // qembed.pack_factor), dtype=np.int32)
            for i in range(qembed.pack_factor):
                embed_qw_packed |= (embed_qw[:, i::qembed.pack_factor] & 0xF) << qembed.num_bits * i

        qembed.scales = scales.flatten().to(torch.float16).to(device)
        qembed.weight.data = torch.from_numpy(embed_qw_packed).to(torch.int32).to(device)
        return qembed

    @torch.no_grad()
    def forward(self, x):
        return quant_embedding_cuda.w4a16_embedding_lookup(x, self.weight, self.scales)

    # def to(self, *args, **kwargs):
    #     super(W4O16Embedding, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self
    
    def __repr__(self):
        return f"W4O16Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


class W8O16Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # to adapt next(iter(model.parameters())) in many places, we use parameter instead of buffer here
        self.register_parameter('weight', nn.Parameter(
            torch.empty((num_embeddings, embedding_dim),
                        dtype=torch.int8, **factory_kwargs),
            requires_grad=False))
        self.register_buffer('scales', torch.empty((num_embeddings),
            dtype=torch.float16, **factory_kwargs))

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear):
        device = originalLayer.weight.device
        qembed = cls(originalLayer.num_embeddings, originalLayer.embedding_dim, device=device)
        embed_w = originalLayer.weight.data
        with torch.no_grad():
            scales = embed_w.amax(dim=-1, keepdim=True) / 127.
            embed_i8 = torch.round(embed_w / scales).clamp(-128, 127).to(torch.int8)
        qembed.scales = scales.flatten().to(torch.float16).to(device)
        qembed.weight.data = embed_i8
        return qembed

    @torch.no_grad()
    def forward(self, x):
        return quant_embedding_cuda.w8a16_embedding_lookup(x, self.weight, self.scales)

    # def to(self, *args, **kwargs):
    #     super(W8O16Embedding, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self
    
    def __repr__(self):
        return f"W8O16Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
