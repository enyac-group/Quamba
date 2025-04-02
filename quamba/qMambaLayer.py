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
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.modules.mamba_simple import Mamba

from .qActLayer import QAct, ActIdentity
from .qLinearLayer import W4A16B16O16Linear
from .qLinearLayer import W4A8B8O8Linear, W4A8B16O16Linear
from .qLinearLayer import W8A8B8O8Linear, W8A8B16O16Linear
from .qLinearLayer import HadLinear
from .qConvLayer import QCausalConv1D
from .qSelectiveScan import QSScan
from .qHadamard import Hadamard, QHadamard


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
        if use_had_transform:
            self.in_proj = HadLinear(originalLayer.in_proj, input_transform=True, output_transform=False)
        else:
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
        self.ssm_state_act = ActIdentity(tensor_name="ssm_state_act")
        # output proj
        if use_had_transform:
            self.had = Hadamard(originalLayer.out_proj.in_features)
            self.out_proj = HadLinear(originalLayer.out_proj, input_transform=True, output_transform=True)
        else:
            self.had = nn.Identity()
            self.out_proj = copy.deepcopy(originalLayer.out_proj)

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
            ssm_state = self.ssm_state_act(ssm_state)
        y = rearrange(y, "b d l -> b l d") 
        y = self.had(y)
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

        y = self.had(y)
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


class W4A16QMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_had_transform=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"

        self.in_proj = W4A16B16O16Linear(self.d_model, self.d_inner * 2, group_size=128)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = W4A16B16O16Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, group_size=128
        )
        # we seperate the bias, so we set bias=False here
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=False, **factory_kwargs)
        self.register_buffer("dt_proj_bias", torch.empty(
            self.d_inner, device=factory_kwargs["device"], dtype=torch.float32))

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj_bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        # output proj
        if use_had_transform:
            self.had = Hadamard(self.d_inner)
        else:
            self.had = nn.Identity()
        self.out_proj = W4A16B16O16Linear(self.d_inner, self.d_model, group_size=128)

    @classmethod
    def from_fp16(cls, originalLayer: Mamba, use_had_transform: bool = True):
        
        qmixer = cls(
            d_model=originalLayer.d_model,
            d_state=originalLayer.d_state,
            d_conv=originalLayer.d_conv,
            expand=originalLayer.expand,
            dt_rank=originalLayer.dt_rank,
            use_had_transform = use_had_transform,
            use_fast_path=False,  # Fused kernel options
            layer_idx=originalLayer.layer_idx,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
        
        # input proj, weight group_size=128
        qmixer.in_proj = W4A16B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
        )
        # causal conv
        qmixer.conv1d = copy.deepcopy(originalLayer.conv1d)
        qmixer.activation = "silu"
        qmixer.act = nn.SiLU()
        # B, C, dt
        qmixer.x_proj = W4A16B16O16Linear.from_fp16(copy.deepcopy(originalLayer.x_proj))
        # We use FP16 dt_proj, becuase w4a16o16 does not support M=bsize, K=48, N=1536
        qmixer.dt_proj = copy.deepcopy(originalLayer.dt_proj)
        qmixer.dt_proj_bias = originalLayer.dt_proj_bias.clone() # MambaSimple has separated bias 
        # ascan
        qmixer.A_log = copy.deepcopy(originalLayer.A_log)
        qmixer.D = copy.deepcopy(originalLayer.D)
        # output proj
        if use_had_transform:
            qmixer.had = Hadamard(originalLayer.out_proj.in_features)
        else:
            qmixer.had = nn.Identity()
        qmixer.out_proj = W4A16B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
        )
        return qmixer

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
            
        # xz = self.in_proj(hidden_states) #(B, 2*D, L)
        # xz = rearrange(xz, "b l d -> b d l")
        xz = self.in_proj.to_seqlen_last(hidden_states) #(B, 2*D, L)
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        assert self.activation in ["silu", "swish"]
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )

        # we need a contiguous here for W4A16B16O16
        x_reshape = rearrange(x, "b d l -> (b l) d").contiguous()
        x_dbl = self.x_proj(x_reshape)  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
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
        y = self.had(y)
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        x = causal_conv1d_update(
            x,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt) # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj_bias, dt_softplus=True
        )
        y = self.had(y)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = torch.float16
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
        # conv_dtype is torch.float16
        conv_dtype = torch.float16
        # ssm_dtype is torch.float16
        ssm_dtype = torch.float16
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=conv_dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=ssm_dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class W4A8QMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_had_transform=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"

        self.in_proj = W4A8B8O8Linear(self.d_model, self.d_inner * 2, group_size=128)

        self.conv1d = QCausalConv1D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = W4A8B8O8Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, group_size=128
        )
        # we seperate the bias and put the bias in the QSScan
        self.dt_proj = W8A8B8O8Linear(self.dt_rank, self.d_inner)

        # Quantized selective scan
        self.selective_scan = QSScan(d_state=self.d_state, d_inner=self.d_inner, delta_softplus=True)

        # output proj
        if use_had_transform:
            self.had = QHadamard(self.d_inner, x_H_scale=1.0)
        else:
            self.had = QAct(scale=1.0)
        self.out_proj = W4A8B16O16Linear(self.d_inner, self.d_model, group_size=128)

    @classmethod
    def from_fp16(cls, originalLayer: MambaSimple, act_scales: Dict, use_had_transform: bool = True):

        qmixer = cls(
            d_model=originalLayer.d_model,
            d_state=originalLayer.d_state,
            d_conv=originalLayer.d_conv,
            expand=originalLayer.expand,
            dt_rank=originalLayer.dt_rank,
            use_had_transform = use_had_transform,
            use_fast_path=False,  # Fused kernel options
            layer_idx=originalLayer.layer_idx,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

        # input proj, weight group_size=128
        qmixer.in_proj = W4A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"],
            output_scale=act_scales["in_proj:output"],
        )

        # causal conv
        # no used, silu is fused in causal_conv1d
        qmixer.activation = "silu"
        assert qmixer.activation == "silu"
        qmixer.conv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.conv1d),
            input_scale=act_scales["in_proj:output"].item(),
            output_scale=act_scales["x_proj:input"].item(),            
        )

        # x_proj
        qmixer.x_proj = W4A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.x_proj),
            input_scale=act_scales["x_proj:input"],
            output_scale=act_scales["x_proj:output"],
        )

        # We use W8A8B8O8 dt_proj, becuase W4A8B8O8 does not support M=bsize, K=48, N=1536
        qmixer.dt_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.dt_proj),
            input_scale=act_scales["x_proj:output"].item(), # use x_proj_scale to avoid additional quantization operations
            output_scale=act_scales["dt_proj:output"].item(),
        )

        # ascan
        qmixer.selective_scan = QSScan.from_fp16(
            originalLayer.d_state, originalLayer.d_inner,
            originalLayer.A_log.clone(), D=originalLayer.D.clone(),
            dt_bias=originalLayer.dt_proj_bias.clone(), delta_softplus=True,
            ssm_state_scale=act_scales["ssm_state_act:input"],
            u_scale=act_scales["x_proj:input"],
            dt_scale=act_scales["dt_proj:output"],
            B_scale=act_scales["x_proj:output"],
            C_scale=act_scales["x_proj:output"],
            z_scale=act_scales["in_proj:output"],
        )

        # output proj
        if use_had_transform:
            qmixer.had.x_H_scale = act_scales["out_proj:input"].item()
        else:
            qmixer.had.scale = act_scales["out_proj:input"].item()
        qmixer.out_proj = W4A8B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
            input_scale=act_scales["out_proj:input"],
        )
        return qmixer

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

        xz = self.in_proj.to_seqlen_last(hidden_states) # (B, D, L) 
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
        dt = self.dt_proj.to_seqlen_last(dt.contiguous())
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        # SSM step and return ssm_state
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
        conv_dtype = torch.int8
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = torch.int8
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
                dtype=torch.int8,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class W8A8QMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_had_transform=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"

        self.in_proj = W8A8B8O8Linear(self.d_model, self.d_inner * 2)

        self.conv1d = QCausalConv1D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = W8A8B8O8Linear(self.d_inner, self.dt_rank + self.d_state * 2)
        # we seperate the bias and put the bias in the QSScan
        self.dt_proj = W8A8B8O8Linear(self.dt_rank, self.d_inner)

        # Quantized selective scan
        self.selective_scan = QSScan(d_state=self.d_state, d_inner=self.d_inner, delta_softplus=True)

        # output proj
        if use_had_transform:
            self.had = QHadamard(self.d_inner, x_H_scale=1.0)
        else:
            self.had = QAct(scale=1.0)
        self.out_proj = W8A8B16O16Linear(self.d_inner, self.d_model)

    @classmethod
    def from_fp16(cls, originalLayer: MambaSimple, act_scales: Dict, use_had_transform: bool = True):

        qmixer = cls(
            d_model=originalLayer.d_model,
            d_state=originalLayer.d_state,
            d_conv=originalLayer.d_conv,
            expand=originalLayer.expand,
            dt_rank=originalLayer.dt_rank,
            use_had_transform = use_had_transform,
            use_fast_path=False,  # Fused kernel options
            layer_idx=originalLayer.layer_idx,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

        # input proj
        qmixer.in_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"].item(),
            output_scale=act_scales["in_proj:output"].item(),
        )

        # causal conv
        # no used, silu is fused in causal_conv1d
        qmixer.activation = "silu"
        assert qmixer.activation == "silu"
        qmixer.conv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.conv1d),
            input_scale=act_scales["in_proj:output"].item(),
            output_scale=act_scales["x_proj:input"].item(),            
        )

        # x_proj
        qmixer.x_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.x_proj),
            input_scale=act_scales["x_proj:input"].item(),
            output_scale=act_scales["x_proj:output"].item(),
        )

        # dt_proj
        original_dt_proj = copy.deepcopy(originalLayer.dt_proj)
        dt_proj_bias = originalLayer.dt_proj_bias.clone() # MambaSimple has separated bias 
        # original_dt_proj.bias = None
        qmixer.dt_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=original_dt_proj,
            input_scale=act_scales["x_proj:output"].item(), # use x_proj_scale to avoid additional quantization operations
            output_scale=act_scales["dt_proj:output"].item(),
        )

        # ascan
        qmixer.selective_scan = QSScan.from_fp16(
            originalLayer.d_state, originalLayer.d_inner,
            originalLayer.A_log.clone(), D=originalLayer.D.clone(),
            dt_bias=dt_proj_bias, delta_softplus=True,
            ssm_state_scale=act_scales["ssm_state_act:input"],
            u_scale=act_scales["x_proj:input"],
            dt_scale=act_scales["dt_proj:output"],
            B_scale=act_scales["x_proj:output"],
            C_scale=act_scales["x_proj:output"],
            z_scale=act_scales["in_proj:output"],
        )

        # output proj
        if use_had_transform:
            qmixer.had.x_H_scale = act_scales["out_proj:input"].item()
        else:
            qmixer.had.scale = act_scales["out_proj:input"].item()
        qmixer.out_proj = W8A8B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
            input_scale=act_scales["out_proj:input"].item(),
        )
        return qmixer

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
        dt = self.dt_proj.to_seqlen_last(dt.contiguous())
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        # SSM step and return ssm_state
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
        conv_dtype = torch.int8
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = torch.int8
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
                dtype=torch.int8,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state