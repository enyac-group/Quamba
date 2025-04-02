import torch
import torch.nn as nn
import torch.nn.functional as F

import quant_linear_cuda

from .hadamard_utils import get_had_fn
from .quant_utils import quantize_tensor_per_tensor_absmax
from .marlin_utils import w4a8_quantize, w4a16_quantize
from .marlin_utils import MARLIN_QQQ_MIN_THREAD_N, MARLIN_QQQ_MAX_PARALLEL
from .marlin_utils import MarlinWorkspace

class W4A16B16O16Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, group_size=128, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        self.size_n, self.size_k = out_features, in_features
        if self.size_n % 256 != 0:
            self.pad_out = 256 - self.size_n % 256
        self.size_n = self.size_n + self.pad_out

        self.max_par = 16
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        max_workspace_size = ((self.size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
        self.register_buffer('workspace', torch.zeros(
            max_workspace_size, dtype=torch.int32, **factory_kwargs))
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty(
            (self.size_k//16, self.size_n*16 // 8),
            dtype=torch.int32, **factory_kwargs))
        self.register_buffer('scale', torch.empty(
            (self.size_k//group_size, self.size_n),
            dtype=torch.float16, **factory_kwargs))

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear):
        assert originalLayer.bias is None, "Not support bias yet"
        # The linear kernel only supports symmetric quantization, so we only have scales
        bits = 4
        group_size = 128 if originalLayer.in_features > 128 else -1
        group_scale = None
        if hasattr(originalLayer, "apply_gptq") and originalLayer.apply_gptq == True:
            bits = originalLayer.bits
            group_size = originalLayer.group_size
            group_scale = originalLayer.group_scale # [n_groups, out_dim]
            # Marlin requires float16 scaling factors
            group_scale = group_scale.to(torch.float16)
        assert bits == 4, "Only support 4-bit quantization"
        if group_size not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        
        device = originalLayer.weight.device
        qlinear = cls(originalLayer.in_features, originalLayer.out_features, device=device)
        qlinear.pad_out = 0
        W = originalLayer.weight # [Dout, Din]
        qlinear.size_n, qlinear.size_k = W.shape
        if W.shape[0] % 256 != 0:
            qlinear.pad_out = 256 - W.shape[0] % 256

        # W16 per-channel per-group quantization to W4
        W_t = W.cpu().to(torch.float16).t().contiguous() # [Dout, Din] -> [Din, Dout], move to CPU to save memory
        group_scale = group_scale.cpu() if group_scale is not None else None
        w_ref, q_w, scale = w4a16_quantize(
            W_t, bits, group_size, group_scale, pad_out=qlinear.pad_out)
        # TODO: remove w_ref to save memory, since it creates a copy of the weight
        qlinear.size_k, qlinear.size_n = w_ref.shape
        qlinear.max_par = 16
        qlinear.weight = q_w.to(device)
        qlinear.scale = scale.to(device) # weight scale
        return qlinear

    # def to(self, *args, **kwargs):
    #     super(W4A16B16O16Linear, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # this contiguous is necessary for batch size > 1 for lm_head
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L281
        x = x.view(-1, x_shape[-1]).contiguous() # must squeeze the tensor first
        y = quant_linear_cuda.w4a16o16_gemm(
            x,
            self.weight,
            self.scale,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            False, -1, -1, -1, self.max_par
        )
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)
        return y

    @torch.no_grad()
    def to_seqlen_last(self, x):
        x_shape = x.shape # [B, L, D]
        x = x.view(-1, x.shape[-1]) # must squeeze the tensor first
        bsize, seqlen = x_shape[0], x_shape[1] 
        y = quant_linear_cuda.w4a16o16_gemm(
            x,
            self.weight,
            self.scale,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features (padded n)
            self.size_k,    # k: in_features
            True, -1, -1, -1, self.max_par
        )
        if self.pad_out != 0:
            y = y[0:-self.pad_out, :]
        if seqlen % 8 !=0:
            y = y[..., 0:bsize*seqlen]
        y = y.view(self.out_features, bsize, -1) # [D, B*L] -> [D, B, L]
        # For better efficiency, we do not need contiguous here, we just
        # have to keep the stride size of the last dimension (seqlen) to be 1
        # see quant_causal_conv1d_fwd
        y = y.transpose(0, 1)
        return y

    def __repr__(self):
        return f"W4A16B16O16Linear(in_features={self.in_features}, out_features={self.out_features})"


class W4A8B16O16Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, group_size=128, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        self.size_n, self.size_k = out_features, in_features
        if self.size_n % 64 != 0:
            self.pad_out = 64 - self.size_n % 64
        self.size_n = self.size_n + self.pad_out

        self.max_par = 16
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        max_workspace_size = ((self.size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
        self.register_buffer('workspace', torch.zeros(
            max_workspace_size, dtype=torch.int, **factory_kwargs))
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty(
            (self.size_k//16, self.size_n*16 // 8),
            dtype=torch.int32, **factory_kwargs))
        self.register_buffer('input_scale', torch.empty(
            (), dtype=torch.float32, **factory_kwargs)) # no-shape
        self.register_buffer('s_channel', torch.empty(
            (1, self.size_n), dtype=torch.float32, **factory_kwargs))
        self.register_buffer('s_group', torch.empty(
            (self.size_k//group_size, self.size_n),
            dtype=torch.float16, **factory_kwargs))

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear, input_scale: torch.Tensor):
        assert input_scale.numel() == 1, "Only support per-tensor input scale"
        assert originalLayer.bias is None, "Not support bias yet"
        # The linear kernel only supports symmetric quantization, so we only have scales
        bits = 4
        group_size = 128 if originalLayer.in_features > 128 else -1
        group_scale = None
        if hasattr(originalLayer, "apply_gptq") and originalLayer.apply_gptq == True:
            bits = originalLayer.bits
            group_size = originalLayer.group_size
            group_scale = originalLayer.group_scale # [n_groups, out_dim]
            # Marlin requires float16 scaling factors
            group_scale = group_scale.to(torch.float16)
        assert bits == 4, "Only support 4-bit quantization"
        if group_size not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')

        device = originalLayer.weight.device
        qlinear = cls(originalLayer.in_features, originalLayer.out_features, group_size=128, device=device)
        qlinear.pad_out = 0
        # W is the fake quantized weight from GPTQ
        W = originalLayer.weight # [Dout, Din]
        qlinear.size_n, qlinear.size_k = W.shape
        if W.shape[0] % 64 != 0:
            qlinear.pad_out = 64 - W.shape[0] % 64

        # Get per-channel scale and zero: 4-bit -> 8-bit
        channel_scale = torch.max(torch.abs(W), 1, keepdim=True)[0] # [Dout, Din]
        channel_scale /= 127.0
        channel_scale = channel_scale.reshape(1, -1).to(dtype=torch.float) # QQQ requires [1, out_dim]

        # W16 per-channel per-group quantization to W4
        W_t = W.cpu().to(torch.float16).t().contiguous() # [Dout, Din] -> [Din, Dout], move to CPU to save memory
        group_scale = group_scale.cpu() if group_scale is not None else None
        channel_scale = channel_scale.cpu()
        w_ref, q_w, s_group, s_channel = w4a8_quantize(
            W_t, bits, group_size, group_scale, channel_scale, pad_out=qlinear.pad_out)

        qlinear.size_k, qlinear.size_n = w_ref.shape
        qlinear.weight = q_w.to(device)
        qlinear.input_scale = input_scale.to(device)
        qlinear.s_group = s_group.to(device)
        qlinear.s_channel = s_channel.to(device)
        return qlinear

    # def to(self, *args, **kwargs):
    #     super(W4A8B16O16Linear, self).to(*args, **kwargs)
    #     self.weight = self.weight.to(*args, **kwargs)
    #     self.input_scale = self.input_scale.to(*args, **kwargs)
    #     self.s_group = self.s_group.to(*args, **kwargs)
    #     self.s_channel = self.s_channel.to(*args, **kwargs)
    #     if self.bias is not None:
    #         self.bias = self.bias.to(*args, **kwargs)
    #     return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # this contiguous is necessary for batch size > 1 for lm_head
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L281
        x = x.view(-1, x_shape[-1]).contiguous()
        y = quant_linear_cuda.w4a8o16_gemm(
            x,
            self.weight,
            self.input_scale,
            self.s_channel,
            self.s_group,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            False           # transpose output
        )
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1) # [B*L, D] -> [B, L, D]
        return y

    @torch.no_grad()
    def to_seqlen_last(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]) # [B*L, D]
        y = quant_linear_cuda.w4a8o16_gemm(
            x,
            self.weight,
            self.input_scale,
            self.s_channel,
            self.s_group,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            True            # transpose output
        )
        if self.pad_out != 0:
            y = y[0:-self.pad_out, :]
        y = y.view(-1, *x_shape[:-1]) # [D, B*L] -> [D, B, L]
        # For better efficiency, we do not need contiguous here, we just
        # have to keep the stride size of the last dimension (seqlen) to be 1
        # see quant_causal_conv1d_fwd
        y = y.transpose(0, 1)
        return y
    
    def __repr__(self):
        return f"W4A8B16O16Linear(in_features={self.in_features}, out_features={self.out_features})"


class W4A8B8O8Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, group_size=128, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        self.size_n, self.size_k = out_features, in_features
        if self.size_n % 64 != 0:
            self.pad_out = 64 - self.size_n % 64
        self.size_n = self.size_n + self.pad_out

        self.max_par = 16
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        max_workspace_size = ((self.size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
        self.register_buffer('workspace', torch.zeros(
            max_workspace_size, dtype=torch.int, **factory_kwargs))
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty(
            (self.size_k//16, self.size_n*16 // 8),
            dtype=torch.int32, device=torch.device('cuda')))
        self.register_buffer('input_scale', torch.empty(
            (), dtype=torch.float32, device=torch.device('cuda')))
        self.register_buffer('s_channel', torch.empty(
            (1, self.size_n), dtype=torch.float32, device=torch.device('cuda')))
        self.register_buffer('s_group', torch.empty(
            (self.size_k//group_size, self.size_n),
            dtype=torch.float16, device=torch.device('cuda')))
        
    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear, input_scale: torch.Tensor, output_scale: torch.Tensor):

        assert input_scale.numel() == 1, "Only support per-tensor input scale"
        assert output_scale.numel() == 1, "Only support per-tensor output scale"
        assert originalLayer.bias is None, "Not support bias yet"
        # The linear kernel only supports symmetric quantization, so we only have scales
        bits = 4
        group_size = 128 if originalLayer.in_features > 128 else -1
        group_scale = None
        if hasattr(originalLayer, "apply_gptq") and originalLayer.apply_gptq == True:
            bits = originalLayer.bits
            group_size = originalLayer.group_size
            group_scale = originalLayer.group_scale # [n_groups, out_dim]
            # Marlin requires float16 scaling factors
            group_scale = group_scale.to(torch.float16)
        assert bits == 4, "Only support 4-bit quantization"
        if group_size not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')

        device = originalLayer.weight.device
        qlinear = cls(originalLayer.in_features, originalLayer.out_features, group_size=128, device=device)
        qlinear.pad_out = 0
        # W is the fake quantized weight from GPTQ
        W = originalLayer.weight # [Dout, Din]
        qlinear.size_n, qlinear.size_k = W.shape
        if W.shape[0] % 64 != 0:
            qlinear.pad_out = 64 - W.shape[0] % 64

        # Get per-channel scale and zero: 4-bit -> 8-bit
        channel_scale = torch.max(torch.abs(W), 1, keepdim=True)[0] # [Dout, Din]
        channel_scale /= 127.0
        channel_scale = channel_scale.reshape(1, -1).to(dtype=torch.float) # QQQ requires [1, out_dim]

        # W16 per-channel per-group quantization to W4
        W_t = W.cpu().to(torch.float16).t().contiguous() # [Dout, Din] -> [Din, Dout], move to CPU to save memory
        group_scale = group_scale.cpu() if group_scale is not None else None
        channel_scale = channel_scale.cpu()
        w_ref, q_w, s_group, s_channel = w4a8_quantize(
            W_t, bits, group_size, group_scale, channel_scale,
            out_scales=output_scale, pad_out=qlinear.pad_out)

        qlinear.size_k, qlinear.size_n = w_ref.shape
        qlinear.weight = q_w.to(device)
        qlinear.input_scale = input_scale.to(device)
        qlinear.s_group = s_group.to(device)
        qlinear.s_channel = s_channel.to(device)
        return qlinear

    # def to(self, *args, **kwargs):
    #     super(W4A8B8O8Linear, self).to(*args, **kwargs)
    #     self.weight = self.weight.to(*args, **kwargs)
    #     self.input_scale = self.input_scale.to(*args, **kwargs)
    #     self.s_group = self.s_group.to(*args, **kwargs)
    #     self.s_channel = self.s_channel.to(*args, **kwargs)
    #     if self.bias is not None:
    #         self.bias = self.bias.to(*args, **kwargs)
    #     return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = quant_linear_cuda.w4a8o8_gemm(
            x,
            self.weight,
            self.input_scale,
            self.s_channel,
            self.s_group,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            False           # transpose output
        )
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)
        return y

    @torch.no_grad()
    def to_seqlen_last(self, x):
        B, L, D = x.shape
        x = x.view(-1, D)
        y = quant_linear_cuda.w4a8o8_gemm(
            x,
            self.weight,
            self.input_scale,
            self.s_channel,
            self.s_group,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            True           # transpose output
        )
        if self.pad_out != 0:
            y = y[0:-self.pad_out, :]
        y = y.view(-1, B, L) # [D, B*L] -> [D, B, L]
        # For better efficiency, we do not need contiguous here, we just
        # have to keep the stride size of the last dimension (seqlen) to be 1
        # see quant_causal_conv1d_fwd
        y = y.transpose(0, 1)
        return y
    
    def __repr__(self):
        return f"W4A8B8O8Linear(in_features={self.in_features}, out_features={self.out_features})"


class W4A8B8O8LinearParallel(torch.nn.Module):

    def __init__(self, in_features, out_features, group_size=128, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        self.size_n, self.size_k = out_features, in_features
        if self.size_n % 64 != 0:
            self.pad_out = 64 - self.size_n % 64
        self.size_n = self.size_n + self.pad_out

        self.max_par = 16
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        max_workspace_size = ((self.size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
        self.register_buffer('workspace', torch.zeros(
            max_workspace_size, dtype=torch.int, **factory_kwargs))
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty(
            (self.size_k//16, self.size_n*16 // 8),
            dtype=torch.int32, **factory_kwargs))
        self.register_buffer('input_scale', torch.empty(
            (), dtype=torch.float32, **factory_kwargs)) # no-shape
        self.register_buffer('s_channel', torch.empty(
            (1, self.size_n), dtype=torch.float32, **factory_kwargs))
        self.register_buffer('s_group', torch.empty(
            (self.size_k//group_size, self.size_n),
            dtype=torch.float16, **factory_kwargs))


    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear, input_scale: torch.Tensor, output_scales, out_split_dims):

        assert input_scale.numel() == 1, "Only support per-tensor input scale"
        assert sum(out_split_dims) == originalLayer.out_features
        assert len(out_split_dims) == len(output_scales)
        assert originalLayer.bias is None, "Not support bias yet"
        # The linear kernel only supports symmetric quantization, so we only have scales
        bits = 4
        group_size = 128 if originalLayer.in_features > 128 else -1
        group_scale = None
        if hasattr(originalLayer, "apply_gptq") and originalLayer.apply_gptq == True:
            bits = originalLayer.bits
            group_size = originalLayer.group_size
            group_scale = originalLayer.group_scale # [n_groups, out_dim]
            # Marlin requires float16 scaling factors
            group_scale = group_scale.to(torch.float16)
        assert bits == 4, "Only support 4-bit quantization"
        if group_size not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')

        device = originalLayer.weight.device
        qlinear = cls(originalLayer.in_features, originalLayer.out_features, group_size=128, device=device)
        # W is the fake quantized weight from GPTQ
        W = originalLayer.weight # [Dout, Din]

        # Get per-channel scale and zero: 4-bit -> 8-bit
        channel_scale = torch.max(torch.abs(W), 1, keepdim=True)[0] # [Dout, Din]
        channel_scale /= 127.0
        channel_scale = channel_scale.reshape(1, -1).to(dtype=torch.float) # QQQ requires [1, out_dim]

        # get the output scales for each output channel
        channel_o_scales = [] # list of tensors
        for o_dim, o_scale in zip(out_split_dims, output_scales):
            o_scale = o_scale.repeat(o_dim)
            channel_o_scales.append(o_scale)
        # W16 per-channel per-group quantization to W4
        channel_o_scales = torch.cat(channel_o_scales, dim=0)
        W_t = W.cpu().to(torch.float16).t().contiguous() # [Dout, Din] -> [Din, Dout], move to CPU to save memory
        group_scale = group_scale.cpu() if group_scale is not None else None
        channel_scale = channel_scale.cpu()
        w_ref, q_w, s_group, s_channel = w4a8_quantize(
            W_t, bits, group_size, group_scale, channel_scale,
            out_scales=channel_o_scales, pad_out=qlinear.pad_out)

        qlinear.weight = q_w.to(device)
        qlinear.input_scale = input_scale.to(device)
        qlinear.s_group = s_group.to(device)
        qlinear.s_channel = s_channel.to(device)
        return qlinear

    def to(self, *args, **kwargs):
        super(W4A8B8O8LinearParallel, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.input_scale = self.input_scale.to(*args, **kwargs)
        self.s_group = self.s_group.to(*args, **kwargs)
        self.s_channel = self.s_channel.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = quant_linear_cuda.w4a8o8_gemm(
            x,
            self.weight,
            self.input_scale,
            self.s_channel,
            self.s_group,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            False           # transpose output
        )
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)
        return y
    
    def __repr__(self):
        return f"W4A8B8O8LinearParallel(in_features={self.in_features}, out_features={self.out_features})"
    

class W8A8B8O8Linear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        if self.out_features % 16 != 0:
            self.pad_out = 16 - (self.out_features % 16) 

        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty_strided(
            (in_features, self.out_features+self.pad_out),
            stride=(1, in_features), dtype=torch.int8, device=torch.device('cuda')))
        self.register_buffer('a', torch.empty(
            (1, 1), dtype=torch.float32, device=torch.device('cuda')))
        self.register_buffer('b', torch.empty(
            (1, 1), dtype=torch.float32, device=torch.device('cuda')))
        # for gemv
        self.alpha = 0.0
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear, input_scale: float = 1.0, output_scale: float =1.0):
        
        qlinear = cls(originalLayer.in_features, originalLayer.out_features)
        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight, n_bits = 8, clip_ratio = 1.0)
        if qlinear.pad_out != 0:
            int8_weight = torch.nn.functional.pad(int8_weight, (0, 0, 0, qlinear.pad_out), "constant", 0).contiguous()
            
        int8_weight = int8_weight.to(torch.int8).t()
        qlinear.weight = int8_weight
        qlinear.a = torch.tensor(
            [[input_scale / output_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight.device)
        qlinear.b = torch.tensor(
            [[weight_scale]],
            requires_grad=False,
            dtype=torch.float32,
            device=int8_weight.device)
        
        # for gemv
        qlinear.alpha = (weight_scale * input_scale / output_scale).item()
        return qlinear

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'alpha'] = self.alpha
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.alpha = state_dict[prefix + 'alpha']
        del state_dict[prefix + 'alpha']

    def to(self, *args, **kwargs):
        super(W8A8B8O8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.b = self.b.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # input [(bsize*seqlen) x in_dim] --> dim last, use row major
        # weight [out_dim x in_dim] --> use column major
        # output [(bsize*seqlen) x out_dim] --> dim last row major
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.out_features + self.pad_out), dtype=x.dtype, device=x.device)
        if x.shape[0] == 1:
            quant_linear_cuda.cutlass_scaled_mv_dq(y, self.weight.t(), x, self.alpha, 0.0)
        else:
            quant_linear_cuda.cutlass_scaled_mm_dq(y, x, self.weight, self.a, self.b)
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)
        return y

    @torch.no_grad()
    def to_seqlen_last(self, x):
        B, L, D = x.shape
        x = x.view(-1, D)
        # weight [out_dim x in_dim] --> use row major, weight.t() [in_dim x out_dim]
        # input [(bsize*seqlen) x in_dim] --> dim last, use col major, input.t() [in_dim x (bsize*seqlen)]
        # output out_dim x (bsize*seqlen) --> seqlen last --> row major
        pad_in = 0
        if x.shape[0] % 16 != 0: # cutlass alignment
            # (padding_left,padding_right, padding_top, padding_bottom)
            pad_in = 16 - x.shape[0] % 16
            x = nn.functional.pad(x, (0, 0, pad_in, 0), "constant", 0)
        y = torch.empty((self.out_features + self.pad_out, x.shape[0]), dtype=x.dtype, device=x.device)
        quant_linear_cuda.cutlass_scaled_mm_dq(y, self.weight.t(), x.t(), self.b, self.a)
        if self.pad_out != 0:
            y = y[0:-self.pad_out, ...]
        if pad_in != 0:
            y = y[..., pad_in:]
        y = y.view(-1, B, L) # [D, B*L] -> [D, B, L]
        # For better efficiency, we do not need contiguous here, we just
        # have to keep the stride size of the last dimension (seqlen) to be 1
        # see quant_causal_conv1d_fwd
        y = y.transpose(0, 1)
        return y
    
    def __repr__(self):
        return f"W8A8B8O8Linear(in_features={self.in_features}, out_features={self.out_features})"


class W8A8B8O8LinearParallel(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        if self.out_features % 16 != 0:
            self.pad_out = 16 - (self.out_features % 16) 

        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty_strided(
            (in_features, self.out_features+self.pad_out),
            stride=(1, in_features), dtype=torch.int8, device=torch.device('cuda')))
        self.register_buffer('a', torch.empty(
            [1, 1], dtype=torch.float32, device=torch.device('cuda')))
        self.register_buffer('b', torch.empty(
            [1, self.out_features+self.pad_out],
            dtype=torch.float32, device=torch.device('cuda')))

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear, input_scale: float,
        output_scales: list, out_split_dims: list):

        assert sum(out_split_dims) == originalLayer.out_features
        assert len(out_split_dims) == len(output_scales)
        qlinear = cls(originalLayer.in_features, originalLayer.out_features)

        d_start = 0
        split_int8_weights = []
        split_scales = []
        W = originalLayer.weight
        for _, (dim, o_scale) in enumerate(zip(out_split_dims, output_scales)):
            d_end = d_start + dim
            # w_split has shape [dout, din]
            w_split = W[d_start:d_end]
            int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
                w_split,
                n_bits = 8,
                clip_ratio = 1.0
            )
            split_int8_weights.append(int8_weight)
            weight_scale = weight_scale.item()
            split_scales.extend([weight_scale / o_scale] * dim)
            d_start = d_end
        cat_int8_weight = torch.cat(split_int8_weights, dim=0).to(torch.int8)
        if qlinear.pad_out != 0:
            cat_int8_weight = torch.nn.functional.pad(cat_int8_weight, (0, 0, 0, qlinear.pad_out), "constant", 0).contiguous()
            split_scales.extend([0]*qlinear.pad_out)
        
        qlinear.weight = cat_int8_weight.t() # [Dout, Din] -> [Din, Dout] (Not contiguous)
        qlinear.a = torch.tensor(
            [[input_scale]], requires_grad=False,
            dtype=torch.float32, device=int8_weight.device)
        qlinear.b = torch.tensor(
            [split_scales], requires_grad=False,
            dtype=torch.float32, device=int8_weight.device)

        return qlinear
        
    def to(self, *args, **kwargs):
        super(W8A8B8O8LinearParallel, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.b = self.b.to(*args, **kwargs)
        # alpha is a float, not a tensor
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # input [(bsize*seqlen) x in_dim] --> dim last, use row major
        # weight [out_dim x in_dim] --> use column major
        # output [(bsize*seqlen) x out_dim] --> dim last row major
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.out_features + self.pad_out), dtype=x.dtype, device=x.device)
        if x.shape[0] == 1:
            # raise NotImplementedError("W8A8B8O8LinearParallel has not supported gemv yet")
            # quant_linear_cuda.cutlass_scaled_mv_dq(y, self.weight.t(), x, 0, 0.0)
            quant_linear_cuda.w8a8o8_gemv(y, self.weight.t(), x, self.b, self.a)
        else:
            quant_linear_cuda.cutlass_scaled_mm_dq(y, x, self.weight, self.a, self.b)
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)
        return y

class W8A8B16O16Linear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        
        self.pad_out = 0
        if self.out_features % 16 != 0:
            self.pad_out = 16 - (self.out_features % 16) 

        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty_strided(
            (in_features, self.out_features+self.pad_out), # contiguous? We have [Din, Dout] (Not contiguous)
            stride=(1, in_features), dtype=torch.int8, device=torch.device('cuda')))
        self.register_buffer('a', torch.empty(
            [1, 1], dtype=torch.float32, device=torch.device('cuda')))
        self.register_buffer('b', torch.empty(
            [1, 1],
            dtype=torch.float32, device=torch.device('cuda')))
        # for gemv
        self.alpha = 0.0
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear, input_scale: float = 1.0):
        qlinear = cls(originalLayer.in_features, originalLayer.out_features)
        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight, n_bits = 8, clip_ratio = 1.0)
        if qlinear.pad_out != 0:
            int8_weight = torch.nn.functional.pad(int8_weight, (0, 0, 0, qlinear.pad_out), "constant", 0).contiguous()
        int8_weight = int8_weight.to(torch.int8).t() # shape [Dout, Din], stride [Din, 1] -> [Din, Dout], stride [Din, 1]
        qlinear.weight = int8_weight
        qlinear.a = torch.tensor([[input_scale]], requires_grad=False,
            dtype=torch.float32, device=int8_weight.device)
        qlinear.b = torch.tensor([[weight_scale]], requires_grad=False,
            dtype=torch.float32, device=int8_weight.device)
        # for gemv, alpha is a float, not a tensor
        qlinear.alpha = (weight_scale * input_scale).item()
        return qlinear

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'alpha'] = self.alpha
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.alpha = state_dict[prefix + 'alpha']
        del state_dict[prefix + 'alpha']

    def to(self, *args, **kwargs):
        super(W8A8B16O16Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.b = self.b.to(*args, **kwargs)
        # alpha is a float, not a tensor
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # this contiguous is necessary for batch size > 1 for lm_head
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L281
        x = x.view(-1, x_shape[-1]).contiguous() # must squeeze the tensor first
        y = torch.empty((x.shape[0], self.out_features + self.pad_out), dtype=torch.float16, device=x.device)
        if x.shape[0] == 1:
            quant_linear_cuda.cutlass_scaled_mv_dq(y, self.weight.t(), x, self.alpha, 0.0)
        else:
            # self.weight [Dout, Din], stride [1, Dout]
            quant_linear_cuda.cutlass_scaled_mm_dq(y, x, self.weight, self.a, self.b)
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)
        return y

    def __repr__(self):
        return f"W8A8B16O16Linear(in_features={self.in_features}, out_features={self.out_features})"


class HadLinear(torch.nn.Linear):

    def __init__(self,
        originalLayer: nn.Linear,
        input_transform=True,
        output_transform=False,
        fuse_had=False
    ):
        assert originalLayer.weight.is_cuda, "Hadamard transform must be on CUDA"
        super().__init__(
            originalLayer.in_features,
            originalLayer.out_features,
            True if originalLayer.bias is not None else False,
            originalLayer.weight.device,
            originalLayer.weight.dtype,
        )
        
        # Do not fuse the Hadamard matrix here, so we can do weight re-ordering
        self.input_transform = input_transform
        if input_transform:
            self.input_transform_fn, self.Nin, self.had_scale_in = get_had_fn(
                originalLayer.in_features)
        
        self.output_transform = output_transform
        if output_transform:
            self.output_transform_fn, self.Nout, self.had_scale_out = get_had_fn(
                originalLayer.out_features)
            
        self.weight.data = originalLayer.weight.data
        if originalLayer.bias is not None:
            self.bias.data = originalLayer.bias.data
        
        self.fuse_had = fuse_had

    def fuse_hadamard(self):
        W = self.weight.data.clone()
        if self.input_transform:
            W = self.input_transform_fn(W, self.had_scale_in)
        if self.output_transform:
            W_t = self.output_transform_fn(W.t(), self.had_scale_out)
            W = W_t.t()
        self.weight.data = W.contiguous()
        self.fuse_had = True
    
    def forward(self, x):
        w_H = self.weight.clone()
        if not self.fuse_had and self.input_transform:
            w_H = self.input_transform_fn(w_H, self.had_scale_in)
        if not self.fuse_had and self.output_transform:
            w_H_t = self.output_transform_fn(w_H.t(), self.had_scale_out)
            w_H = w_H_t.t()
        return F.linear(x, w_H, self.bias)

    def __repr__(self):
        return f"HadLinear(in_features={self.in_features}, out_features={self.out_features}, input_transform={self.input_transform}, output_transform={self.output_transform})"
    