# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import copy
from functools import partial
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
# Only use pytorch to study quantization
causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.modules.mamba_simple import Mamba

from .qActLayer import QAct
from .qLinearLayer import W8A8B8O8Linear, W8A8BF16OF16Linear, HadW8A8BF16OF16Linear, HadLinear
from .qConvLayer import QCausalConv1D
from .qSelectiveScan import QSScan
from .qHadamard import Hadamard, QHadamard


class QMamba(nn.Module):
    def __init__(
        self,
        originalLayer: Mamba,
        act_scales: Dict,
        use_had_transform: bool = True
    ):
        super().__init__()
        self.d_model = originalLayer.d_model
        self.d_state = originalLayer.d_state
        self.d_conv = originalLayer.d_conv
        self.expand = originalLayer.expand
        self.d_inner = originalLayer.d_inner
        self.dt_rank = originalLayer.dt_rank
        self.use_fast_path = False # DO NOT USE FAST PATH for quantization experiments
        self.layer_idx = originalLayer.layer_idx

        # input proj
        self.in_proj = W8A8B8O8Linear(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"].item(),
            output_scale=act_scales["in_proj:output"].item(),
        )

        # causal conv
        # no used, silu is fused in causal_conv1d
        self.activation = "silu"
        assert self.activation == "silu"
        self.conv1d = QCausalConv1D(
            originalLayer=copy.deepcopy(originalLayer.conv1d),
            input_scale=act_scales["in_proj:output"].item(),
            output_scale=act_scales["x_proj:input"].item(),            
        )

        # x_proj
        self.x_proj = W8A8B8O8Linear(
            originalLayer=copy.deepcopy(originalLayer.x_proj),
            input_scale=act_scales["x_proj:input"].item(),
            output_scale=act_scales["x_proj:output"].item(),
        )

        # dt_proj
        original_dt_proj = copy.deepcopy(originalLayer.dt_proj)
        dt_proj_bias = original_dt_proj.bias.clone()
        original_dt_proj.bias = None
        self.dt_proj = W8A8B8O8Linear(
            originalLayer=original_dt_proj,
            input_scale=act_scales["x_proj:output"].item(), # use x_proj_scale to avoid additional quantization operations
            output_scale=act_scales["dt_proj:output"].item(),
        )

        # ascan
        self.selective_scan = QSScan(
            originalLayer.A_log.clone(), D=originalLayer.D.clone(),
            dt_bias=dt_proj_bias, delta_softplus=True,
            u_scale=act_scales["x_proj:input"],
            dt_scale=act_scales["dt_proj:output"],
            B_scale=act_scales["x_proj:output"],
            C_scale=act_scales["x_proj:output"],
            z_scale=act_scales["in_proj:output"],
            output_scale=act_scales["out_proj:input"]
        )

        # output proj
        if use_had_transform:
            self.had = QHadamard(originalLayer.out_proj.in_features, act_scales["out_proj:input"].item())
            self.out_proj = HadW8A8BF16OF16Linear(
                originalLayer=copy.deepcopy(originalLayer.out_proj),
                input_scale=act_scales["out_proj:input"].item(),
            )
        else:
            self.had = QAct(act_scales["out_proj:input"].item())
            self.out_proj = W8A8BF16OF16Linear(
                originalLayer=copy.deepcopy(originalLayer.out_proj),
                input_scale=act_scales["out_proj:input"].item(),
            )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        #assert hidden_states.shape[0] == 1, "Current only support bsz=1"
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
            
        #NOTE(brian1009): Simplified of original implementation 
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L134
        # Input projection for x, z
        # xz = self.in_proj(hidden_states) #(B, L, 2*D)
        # xz = rearrange(xz, "b l d -> b d l").contiguous() 
        # x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)

        xz = self.in_proj.to_seqlen_last(hidden_states) #(B, D, L)
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Perform causal conv1d and return conv_state
        if conv_state is not None:
            # store quantized x into conv_state
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        x = self.conv1d.forward(x)

        # Compute dt, B, C
        x_reshape = rearrange(x, "b d l -> b l d").contiguous()
        x_dbl = self.x_proj(x_reshape)  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Compute dt proj with x_proj_scale
        # dt = self.dt_proj(dt.contiguous())
        dt = self.dt_proj.to_seqlen_last(dt.contiguous())

        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        # dt = rearrange(dt, "b l d -> b d l", l=seqlen).contiguous()
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        # SSM step and return ssm_state
        # using quantized x will hurt the performance
        # acc 42.34 -> 37.28
        y = self.selective_scan.forward(x, dt, B, C, z=z, return_last_state=ssm_state is not None)
        if ssm_state is not None:
            y, last_state = y # y: fp16, last_state: fp32
            ssm_state.copy_(last_state) # last_state: fp32 copy to ssm_state: fp16
        
        # Output projection
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y) # HadW8A8BF16OF16Linear: input int8, output is fp16
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        # Input projection for x, z
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Perform causal conv1d and update conv_state in-place
        x = self.conv1d.update(x, conv_state)

        # Compute dt, B, C 
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt.contiguous())

        # SSM step and update ssm_state in-place
        y, ssm_state = self.selective_scan.update(ssm_state, x.contiguous(), dt, B, C, z=z)

        # Output projection
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = torch.float16
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=torch.int8,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=torch.float16,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state



class MambaSimple(nn.Module):
    def __init__(
        self,
        originalLayer: Mamba,
        use_had_transform: bool = True
    ):
        super().__init__()
        self.d_model = originalLayer.d_model
        self.d_state = originalLayer.d_state
        self.d_conv = originalLayer.d_conv
        self.expand = originalLayer.expand
        self.d_inner = originalLayer.d_inner
        self.dt_rank = originalLayer.dt_rank
        # self.use_fast_path = originalLayer.use_fast_path
        self.use_fast_path = False # DO NOT USE FAST PATH for quantization experiments
        self.layer_idx = originalLayer.layer_idx
        self.use_had_transform = use_had_transform
        
        # input proj
        self.in_proj = copy.deepcopy(originalLayer.in_proj)
        # causal conv
        self.conv1d = copy.deepcopy(originalLayer.conv1d)
        self.activation = "silu"
        self.act = nn.SiLU()
        # B, C, dt
        self.x_proj = copy.deepcopy(originalLayer.x_proj)
        self.dt_proj = copy.deepcopy(originalLayer.dt_proj)
        self.dt_proj.bias = None
        self.dt_proj_bias = originalLayer.dt_proj.bias.clone().float()
        # ascan
        self.A_log = copy.deepcopy(originalLayer.A_log)
        self.D = copy.deepcopy(originalLayer.D)
        self.H_trans = nn.Identity()
        if use_had_transform:
            self.H_trans = Hadamard(originalLayer.out_proj.in_features)

        # output proj
        self.out_proj = copy.deepcopy(originalLayer.out_proj)
        if use_had_transform:
            self.out_proj = HadLinear(self.out_proj)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        #assert hidden_states.shape[0] == 1, "Current only support bsz=1"
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
            
        #NOTE(brian1009): Simplified of original implementation 
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L134
        xz = self.in_proj(hidden_states) #(B, L, 2*D)
        xz = rearrange(xz, "b l d -> b d l") 
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        x = self.conv1d(x)
        x = self.act(x[...,:seqlen])
        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        x_reshape = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_reshape)  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        #NOTE(brian1009): Comment this line and do the inference directly with the forward in the module
        dt = self.dt_proj(dt)

        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        dt = rearrange(dt, "b l d -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        assert self.activation in ["silu", "swish"]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj_bias,
                # delta_bias=None,  # delta_bias has been added in dt_proj
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d") 
        y = self.H_trans(y)
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            w_quant, w_scales = self.conv1d.quant_weight
            x = torch.sum(conv_state * rearrange(w_quant, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        # Discretize A and B
        dt = F.softplus(dt+self.dt_proj_bias)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)

        y = self.H_trans(y)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


if __name__ == "__main__":
    from .observer import PerTensorMinmaxObserver
    from mamba_ssm.modules.mamba_simple import Mamba
    torch.manual_seed(1234)

    bsize = 1
    seqlen = 64
    dim = 768

    mamba = Mamba(d_model=768, use_fast_path=False).cuda()
    mamba_simple = MambaSimple(mamba).cuda()

    act_scales = {}
    def stat_hook(m, inputs, outputs, op):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        act_scales[op + ":input"] = inputs.clone().detach().abs().max()  / 127

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        act_scales[op + ":output"] = outputs.clone().detach().abs().max()  / 127

    hooks = []
    for name, m in mamba_simple.named_modules():
        if isinstance(m, torch.nn.Linear):
            op = name.split(".")[-1]
            hooks.append(
                m.register_forward_hook(partial(stat_hook, op=op))
            )

    x = torch.rand((bsize, seqlen, dim)).cuda()
    y_gt = mamba_simple(x)
    print(act_scales)
    mamba_quant = QMamba(mamba, act_scales).cuda()
    y_q = mamba_quant(x)
    print(y_q.shape, y_q.dtype)
    r2 = (y_gt - y_q).pow(2).mean() / y_gt.pow(2).mean()
    print(r2)