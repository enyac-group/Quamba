import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from .quant_utils import quantize_tensor_per_tensor_absmax

import quant_causal_conv1d_cuda
import quamba2_conv1d_cuda

class QCausalConv1D(nn.Module):

    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        assert in_channels == out_channels == groups, "QCausalConv1D only supports in_channels == out_channels == groups"
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 1, "QCausalConv1D only supports 1D kernel_size"
            kernel_size = kernel_size[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer('weight', torch.empty(
            (in_channels, kernel_size), dtype=torch.int8, **factory_kwargs))
        if bias is not None:
            self.register_buffer('bias', torch.empty(
                (in_channels), dtype=torch.int8, **factory_kwargs))
        else:
            self.bias = None
        self.weight_scale = 0.0
        self.bias_scale = 0.0
        self.input_scale = 0.0
        self.output_scale = 0.0
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)
        
    @classmethod
    def from_fp16(cls, originalLayer: nn.Conv1d, input_scale=1.0, output_scale=1.0):
        device = originalLayer.weight.device
        qconv = cls(originalLayer.in_channels, originalLayer.out_channels,
                    originalLayer.kernel_size, originalLayer.stride,
                    originalLayer.padding, originalLayer.dilation,
                    originalLayer.groups, originalLayer.bias, device=device)
        
        qconv.input_scale = input_scale
        qconv.output_scale = output_scale

        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight.clone().detach(),
            n_bits = 8,
            clip_ratio = 1.0
        )
        int8_weight = rearrange(int8_weight.to(torch.int8), "d 1 w -> d w").contiguous()
        qconv.weight = int8_weight.to(device)
        qconv.weight_scale = weight_scale.item()
        if originalLayer.bias is not None:
            int8_bias, bias_scale = quantize_tensor_per_tensor_absmax(
                originalLayer.bias.clone().detach(),
                n_bits = 8,
                clip_ratio = 1.0
            )
            int8_bias = int8_bias.to(torch.int8).contiguous().to(device)
            qconv.bias = int8_bias
            qconv.bias_scale = bias_scale.item()
        else:
            qconv.bias = None
            qconv.bias_scale = 1.0
        return qconv

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'weight_scale'] = self.weight_scale
        state_dict[prefix + 'bias_scale'] = self.bias_scale
        state_dict[prefix + 'input_scale'] = self.input_scale
        state_dict[prefix + 'output_scale'] = self.output_scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.weight_scale = state_dict[prefix + 'weight_scale']
        self.bias_scale = state_dict[prefix + 'bias_scale']
        self.input_scale = state_dict[prefix + 'input_scale']
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'weight_scale']
        del state_dict[prefix + 'bias_scale']
        del state_dict[prefix + 'input_scale']
        del state_dict[prefix + 'output_scale']

    @torch.no_grad()
    def forward(self, x):
        y = quant_causal_conv1d_cuda.fwd(
                x, self.input_scale,
                self.weight, self.weight_scale,
                self.output_scale,
                self.bias_scale, self.bias,
                None, None, None, True
            )
        return y

    @torch.no_grad()
    def update(self, x, conv_state):
        # update conv_state in-place
        y = quant_causal_conv1d_cuda.update(
            x, conv_state, self.input_scale,
            self.weight, self.weight_scale,
            self.output_scale,
            self.bias_scale, self.bias, True
        ) 
        return y

    # def to(self, *args, **kwargs):
    #     super(QCausalConv1D, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self

    def __repr__(self):
        return f"QCausalConv1D({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"



class Quamb2Conv1D(nn.Module):

    def __init__ (self, x_dim, x_headdim, d_state, n_groups, x_nhead_group, x_ndim_group,
                  in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        # x and d_state
        self.x_dim = x_dim
        self.x_headdim = x_headdim
        self.d_state = d_state
        self.n_groups = n_groups
        self.x_nhead_group = x_nhead_group
        self.x_ndim_group = x_ndim_group

        assert in_channels == out_channels == groups, "QCausalConv1D only supports in_channels == out_channels == groups"
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 1, "QCausalConv1D only supports 1D kernel_size"
            kernel_size = kernel_size[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.x_scale = 0.0
        self.B_scale = 0.0
        self.C_scale = 0.0
        self.wx_scale = 0.0
        self.wB_scale = 0.0
        self.wC_scale = 0.0
        self.register_buffer('weight', torch.empty(
            (in_channels, kernel_size), dtype=torch.int8, **factory_kwargs))
        if bias is not None:
            self.register_buffer('bias', torch.empty(
                (in_channels), dtype=torch.int8, **factory_kwargs))
            self.bx_scale = 0.0
            self.bB_scale = 0.0
            self.bC_scale = 0.0
        else:
            self.bias = None
            self.bx_scale = None
            self.bB_scale = None
            self.bC_scale = None
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

        if x_nhead_group > 0 and x_ndim_group > 0:
            self.register_buffer('x_head_group_range', torch.empty(
                (n_groups, x_nhead_group), dtype=torch.int32, **factory_kwargs))
            self.register_buffer('x_dim_group_range', torch.empty(
                (n_groups, x_nhead_group, x_ndim_group), dtype=torch.int32, **factory_kwargs))
            self.register_buffer('x_out_scales', torch.empty(
                (n_groups, x_nhead_group, x_ndim_group), dtype=torch.float32, **factory_kwargs))
        elif x_nhead_group == 0 and x_ndim_group == 0:
            self.x_head_group_range = None
            self.x_dim_group_range = None
            self.register_buffer('x_out_scales', torch.empty(
                (1), dtype=torch.float32, **factory_kwargs))
        else:
            raise ValueError("""x_nhead_group and x_ndim_group must be both zero or both non-zero""")
        
        self.register_buffer('B_out_scales', torch.empty(
            (n_groups), dtype=torch.float32, **factory_kwargs))
        self.register_buffer('C_out_scales', torch.empty(
            (n_groups), dtype=torch.float32, **factory_kwargs))

    @classmethod
    def from_fp16(
        cls,
        originalLayer: nn.Conv1d,
        x_dim, x_headdim, d_state, n_groups,
        x_scale, B_scale, C_scale,
        x_out_scales, B_out_scales, C_out_scales,
        x_head_group_range, x_dim_group_range,
    ):

        if x_head_group_range is not None and x_dim_group_range is not None:
            assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
            assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
            x_nhead_group = x_head_group_range.shape[1] # [n_ssd_group, x_nhead_group]
            x_ndim_group = x_dim_group_range.shape[2]   # [n_ssd_group, x_nhead_group, n_dim_group]
        elif x_head_group_range is None and x_dim_group_range is None:
            x_nhead_group = 0
            x_ndim_group = 0 
        else:
            raise ValueError("""x_head_group_range and x_dim_group_range must be both None or both not None""")
    
        device = originalLayer.weight.device
        qconv = cls(x_dim, x_headdim, d_state, n_groups, x_nhead_group, x_ndim_group,
                    originalLayer.in_channels, originalLayer.out_channels,
                    originalLayer.kernel_size, originalLayer.stride,
                    originalLayer.padding, originalLayer.dilation,
                    originalLayer.groups, originalLayer.bias, device=device)

        # input scales for x, B, and C
        qconv.x_scale = x_scale.item()   # float scalar
        qconv.B_scale = B_scale.item()   # float scalar
        qconv.C_scale = C_scale.item()   # float scalar

        # output grouping params
        qconv.x_head_group_range = x_head_group_range.to(device) if x_head_group_range is not None else None # scale tensor must be on cuda
        qconv.x_dim_group_range = x_dim_group_range.to(device) if x_dim_group_range is not None else None    # scale tensor must be on cuda

        # output scales for x, B, and C
        qconv.x_out_scales = x_out_scales.to(device) # scale tensor must be on cuda
        qconv.B_out_scales = B_out_scales.to(device) # scale tensor must be on cuda
        qconv.C_out_scales = C_out_scales.to(device) # scale tensor must be on cuda

        weight = rearrange(originalLayer.weight.clone().detach().to(torch.float32), "d 1 w -> d w")
        d_start = 0
        split_int8_weight = []
        split_weight_scales = []
        for dim in [qconv.x_dim, qconv.d_state*qconv.n_groups, qconv.d_state*qconv.n_groups]:
            d_end = d_start + dim
            w_split = weight[d_start:d_end].contiguous()
            w_split_i8, w_split_scale = quantize_tensor_per_tensor_absmax(
                w_split, n_bits = 8, clip_ratio = 1.0)
            split_int8_weight.append(w_split_i8.to(torch.int8).contiguous())
            split_weight_scales.append(w_split_scale.item())
            d_start = d_end
        cat_int8_weight = torch.cat(split_int8_weight, dim=0).contiguous()
        qconv.weight = cat_int8_weight
        qconv.wx_scale, qconv.wB_scale, qconv.wC_scale = split_weight_scales

        bias = originalLayer.bias.clone().detach().to(torch.float32) if originalLayer.bias is not None else None
        if bias is not None:
            d_start = 0
            split_int8_bias = []
            split_bias_scales = []
            for dim in [qconv.x_dim, qconv.d_state*qconv.n_groups, qconv.d_state*qconv.n_groups]:
                d_end = d_start + dim
                b_split = bias[d_start:d_end].contiguous()
                bias_split_i8, bias_split_scale = quantize_tensor_per_tensor_absmax(
                    b_split, n_bits = 8, clip_ratio = 1.0)
                split_int8_bias.append(bias_split_i8.to(torch.int8).contiguous())
                split_bias_scales.append(bias_split_scale.item())
                d_start = d_end
            cat_int8_bias = torch.cat(split_int8_bias, dim=0).contiguous()
            qconv.bias = cat_int8_bias
            qconv.bx_scale, qconv.bB_scale, qconv.bC_scale = split_bias_scales
        else:
            qconv.bias = None
            qconv.bx_scale, qconv.bB_scale, qconv.bC_scale = None, None, None
        return qconv

    def store_hook(self, module, state_dict, prefix, local_metadata):
        # Define all scales to store
        scale_names = ['x_scale', 'B_scale', 'C_scale', 'wx_scale', 'wB_scale', 'wC_scale']
        # Store each scale dynamically
        for name in scale_names:
            state_dict[prefix + name] = getattr(self, name)
        # Handle bias-related scales if bias exists
        if self.bias is not None:
            for name in ['bx_scale', 'bB_scale', 'bC_scale']:
                state_dict[prefix + name] = getattr(self, name)
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Define all scale names
        scales = ['x_scale', 'B_scale', 'C_scale', 'wx_scale', 'wB_scale', 'wC_scale']
        bias_scales = ['bx_scale', 'bB_scale', 'bC_scale']
        # Load and remove regular scales
        for scale in scales:
            setattr(self, scale, state_dict[prefix + scale])
            del state_dict[prefix + scale]
        # Handle bias-related scales if bias exists
        if self.bias is not None:
            for scale in bias_scales:
                setattr(self, scale, state_dict[prefix + scale])
                del state_dict[prefix + scale]

    @torch.no_grad()
    def forward(self, xBC):
        x, B, C = quamba2_conv1d_cuda.fwd(
                xBC, self.x_scale, self.B_scale, self.C_scale,
                self.x_dim, self.x_headdim, self.d_state, self.n_groups,
                self.x_head_group_range, self.x_dim_group_range,
                self.x_out_scales, self.B_out_scales, self.C_out_scales,
                self.weight, self.wx_scale, self.wB_scale, self.wC_scale,
                self.bias, self.bx_scale, self.bB_scale, self.bC_scale,
                None, None, None, True
            )
        return x, B, C

    @torch.no_grad()
    def update(self, xBC, conv_state):
        # update conv_state in-place
        x, B, C = quamba2_conv1d_cuda.update(
                xBC, conv_state, self.x_scale, self.B_scale, self.C_scale,
                self.x_dim, self.x_headdim, self.d_state, self.n_groups,
                self.x_head_group_range, self.x_dim_group_range,
                self.x_out_scales, self.B_out_scales, self.C_out_scales,
                self.weight, self.wx_scale, self.wB_scale, self.wC_scale,
                self.bias, self.bx_scale, self.bB_scale, self.bC_scale,
                None, None, None, True
        ) 
        return x, B, C

    # def to(self, *args, **kwargs):
    #     super(QCausalConv1D, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self

    def __repr__(self):
        return f"Quamb2Conv1D({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"