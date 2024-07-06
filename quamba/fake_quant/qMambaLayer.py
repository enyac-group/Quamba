# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

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

# try:
#     from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None
# # Only use pytorch to study quantization
# selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.modules import mamba_simple


from quamba.fake_quant.qActLayer import QAct
from quamba.fake_quant.qLinearLayer import QLinearLayer
from quamba.fake_quant.qConvLayer import QConv1D
from quamba.fake_quant.qSelectiveScan import QSScan
from quamba.fake_quant.rotation_utils import HadamardTransform
from quamba.fake_quant.smooth_quant_utils import SmoothModule

class QMamba(nn.Module):
    def __init__(
        self,
        originalLayer: mamba_simple,
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
        
        self.in_proj = QLinearLayer(
            originalLayer = originalLayer.in_proj
        )

        self.conv1d = QConv1D(
            originalLayer=originalLayer.conv1d
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = QLinearLayer(
            originalLayer = originalLayer.x_proj
        )
        self.dt_proj = QLinearLayer(
            originalLayer = originalLayer.dt_proj
        )
        
        self.q_sscan = QSScan(
            originalLayer.A_log, originalLayer.D,
            dt_bias=None, delta_softplus=True
        )

        self.out_proj = QLinearLayer(
            originalLayer = originalLayer.out_proj
        )

        # SmoothQuant Modules
        self.mamba_in_smooth = SmoothModule(weight_to_smooth="in_proj", tensor_name="mamba_in_smooth")
        self.u_smooth = SmoothModule(weight_to_smooth="x_proj",tensor_name="u_smooth")
        self.dt_smooth = SmoothModule(weight_to_smooth="dt_proj", tensor_name="dt_smooth")
        # Quantization Modules
        self.mamba_in_quant = QAct(tensor_name="mamba_in_quant")
        self.conv_in_quant = QAct(tensor_name="conv_in_quant")
        self.dt_proj_in_quant = QAct(tensor_name="dt_proj_in_quant")

        self.dt_quant = QAct(tensor_name="dt_quant")
        self.B_quant = QAct(tensor_name="B_quant")
        self.C_quant = QAct(tensor_name="C_quant")
        #self.D_quant = QAct(n_bits=self.w_bits, clip_ratio=w_clip_ratio, real_quant=False)
        self.z_quant = QAct(tensor_name="z_quant")
        self.u_quant = QAct(tensor_name="u_quant")
        #NOTE(brian1009): Move into qSelectiveScan.py
        #self.ssm_out_quant = QAct(tensor_name="ssm_out_quant")
        #self.ssm_out_had = HadamardTransform()
        
        #self.c_quant = QAct(n_bits=self.a_bits, clip_ratio=a_clip_ratio, real_quant=False)
        

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
            
        #[Quantized Point]: Input of Mamba block
        # hidden_states, _, _ = quantize_tensor(hidden_states, n_bits=self.a_bits, quant_type=self.a_quant_type,
        #                                 sym=self.a_sym, clip_ratio=self.a_clip_ratio)
        
        #NOTE(expr/smoothquant brian1009 05/16):  Do smoothing here
        hidden_states = self.mamba_in_smooth(hidden_states)
        hidden_states = self.mamba_in_quant(hidden_states)
        #NOTE(brian1009): Simplified of original implementation
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L134
        xz = self.in_proj(hidden_states) #(B, L, 2*D)
        #NOTE(brian1009): Do not reshape here
        #xz = rearrange(xz, "b l d -> b d l")
        #x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)        
        x, z = xz.chunk(2, dim=-1)  # (B L D), (B L D)
        #[Quantized Point]: conv_1d input (x)
        x = self.conv_in_quant(x)
        #[Quantized Point]: z input quant (z) 
        z = self.z_quant(z)
        #NOTE(brian1009): Reshape after quantization to ensure that we always quantized alone the last dimension
        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        #x = self.act(self.conv1d(x)[..., :seqlen])
        x = self.conv1d(x)
        # We will handle here in FP32
        x = self.act(x[...,:seqlen])
        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d) we divide into a few steps to apply quantization
        
        
        #NOTE(brian1009): 2024/04/16: This is very ugly...., ensure that we always quantized alone the last dimension (b, l, d)
        #[Quantized Point]: x_proj (u_proj) input
        x_reshape = rearrange(x, "b d l -> b l d")
        
        #NOTE(expr/smoothquant brian1009 05/16):  Do smoothing here
        x_reshape = self.u_smooth(x_reshape)
        x_reshape = self.u_quant(x_reshape)
        #print(torch.all(u == x_reshape), self.u_quant.act_quantizer, self.u_proj_in_quat.act_quantizer)
        
        #NOTE(expr/smoothquant brian1009 05/16):  
        # This is an adhoc un-smoothing for u tensor (the branch forward into AScan)
        u = self.u_smooth(x_reshape, reverse=True)
        u = rearrange(u, "b l d -> b d l")
        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        #x_reshape = rearrange(x, "b d l -> (b l) d") 
        #x_reshape = rearrange(x, "b d l -> b l d")#NOTE(brian1009): 2024/04/16: See the NOTE above
        x_dbl  = self.x_proj(x_reshape)  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        #NOTE(brian1009): Comment this line and do the inference directly with the forward in the module
        # dt = self.dt_proj.weight @ dt.t()
        # dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        #[Quantized Point]: dt_proj input
        #NOTE(expr/smoothquant brian1009 05/16):  Do smoothing here
        dt = self.dt_smooth(dt)
        dt = self.dt_proj_in_quant(dt)
        dt = self.dt_proj(dt)
        dt = self.dt_quant(dt)
        B = self.B_quant(B)
        C = self.C_quant(C)

        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        #dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
        #B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        #C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        dt = rearrange(dt, "b l d -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        
        assert self.activation in ["silu", "swish"]

        y = self.q_sscan(
            u,
            dt,
            B,
            C,
            z=z,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
            
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        hidden_states, mamba_in_scales = self.mamba_in_quant(hidden_states)
        xz, in_proj_out_scales = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)
        #[Quantized Point]: conv_1d input (x)
        x, conv_in_scales = self.conv_in_quant(x, i_scales=in_proj_out_scales)
        #[Quantized Point]: z input quant (z) 
        z, z_scale = self.z_quant(z, i_scales=in_proj_out_scales)

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

        x_db, u_proj_out_scales = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt, dt_proj_in_scale = self.dt_proj_in_quant(dt, i_scales=u_proj_out_scales)
        dt, dt_proj_out_scale = self.dt_proj(dt, i_scales=dt_proj_in_scale)
        dt, dt_scale = self.dt_quant(dt, i_scales=dt_proj_out_scale)
        B, B_scale = self.B_quant(B, i_scales=u_proj_out_scales)
        C, C_scale = self.C_quant(C, i_scales=u_proj_out_scales)
        A_log, A_log_scale = self.q_sscan.quant_A_log
        D, D_scale = self.q_sscan.quant_D
        A = -torch.exp(A_log.float())  # (d_inner, d_state)

        # SSM step
        # Discretize A and B
        dt = F.softplus(dt)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + D.to(dtype) * x
        y = y * self.act(z)  # (B D)

        out, out_proj_scale = self.out_proj(y)
        out = out * out_proj_scale
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
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
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

