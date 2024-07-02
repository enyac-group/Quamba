# This code is adapted from ~/.cache/huggingface/modules/transformers_modules/Jamba-v0.1/modeling_jamba.py
# slightly different from https://github.com/huggingface/transformers/blob/12b1620e615592fbf099d4ec44af7b9f2d1b48aa/src/transformers/models/jamba/modeling_jamba.py
import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from transformers.cache_utils import DynamicCache
from transformers.models.jamba.modeling_jamba import JambaMambaMixer

# try except block so it'll work with trust_remote_code. Later we can have `if is_mamba_ssm_available():`
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

# try except block so it'll work with trust_remote_code. Later we can have `if is_causal_conv1d_available():`
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

from .qActLayer import QAct
from .qLinearLayer import W8A8B8O8Linear, W8A8BF16OF16Linear, HadW8A8BF16OF16Linear
from .qConvLayer import QCausalConv1D
from .qSelectiveScan import QSScan
from .qHadamard import QHadamard

logger = logging.getLogger(__name__)

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Jamba
class QJambaRMSNorm(nn.Module):
    def __init__(self,
        originalLayer,
        output_scale: float
        ):
        super().__init__()
        
        self.variance_epsilon = originalLayer.variance_epsilon
        self.dim = tuple(originalLayer.weight.shape)
        self.register_buffer('weight', originalLayer.weight.clone())
        self.output_scale = output_scale

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        return (hidden_states / self.output_scale).round().clamp(min=-128, max=127).to(torch.int8) # quant



class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    It stores the Key and Value states as a list of tensors, one for each layer.
    The expected shape for each tensor for attention layers is `[batch_size, num_heads, seq_len, head_dim]`.
    For the mamba layers, the `key_cache` represents the convolution state and has a shape of `[batch_size, d_inner, 1, d_conv]`,
    and the `value_cache` represents the ssm state and has a shape of `[batch_size, d_inner, 1, d_state]`. Mamba cache
    shape[2] is a dummy "seqlen" dimension to match the number of attention cache dimensions. For mamba, the cache
    doesn't grow with seqlen so this dimension is always 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_layer_idx = None  # used to know which layer has data on seqlen in the cache shape

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `HybridMambaAttentionDynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if self.attention_layer_idx is None and self._is_attn_layer(key_states, value_states):
            self.attention_layer_idx = layer_idx
        if self.attention_layer_idx is not None and layer_idx == self.attention_layer_idx:
            if hasattr(self, "_seen_tokens"):
                self._seen_tokens += key_states.shape[-2]
            else:
                self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                # attention layer - append the new states to the existing cache on the seqlen dimension
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            else:
                # mamba layer - replace the cache with the new states
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is not None:
            if len(self.key_cache) <= layer_idx:
                return 0
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                return self.key_cache[layer_idx].shape[-2]
            else:
                warnings.warn(
                    f"Asked to get the sequence length from cache of layer {layer_idx} which is not an attention layer. "
                    f"Ignoring that and using an attention layer cache"
                )
        if self.attention_layer_idx is None or len(self.key_cache) <= self.attention_layer_idx:
            return 0
        return self.key_cache[self.attention_layer_idx].shape[-2]

    @staticmethod
    def _is_attn_layer(key_states: torch.Tensor, value_states: torch.Tensor):
        return key_states.shape[-1] == value_states.shape[-1]


@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)

# Adapted from ~/.cache/huggingface/modules/transformers_modules/Jamba-v0.1/modeling_jamba.py
# slightly different from transformers.models.mamba.modeling_mamba.MambaMixer
class QJambaMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, originalLayer: JambaMambaMixer,
                 act_scales: Dict,
                 use_had_transform: bool = True):
        super().__init__()
        self.config = originalLayer.config
        self.layer_idx = originalLayer.layer_idx
        self.hidden_size = originalLayer.hidden_size
        self.ssm_state_size = originalLayer.ssm_state_size
        self.conv_kernel_size = originalLayer.conv_kernel_size
        self.intermediate_size = originalLayer.intermediate_size
        self.time_step_rank = originalLayer.time_step_rank
        self.use_conv_bias = originalLayer.use_conv_bias
        self.use_bias = originalLayer.use_bias
        self.conv1d = originalLayer.conv1d
        self.activation = originalLayer.activation
        self.act = originalLayer.act
        self.apply_inner_layernorms = originalLayer.apply_inner_layernorms
        self.use_fast_kernels = originalLayer.use_fast_kernels

        # projection of the input hidden states
        self.in_proj = W8A8B8O8Linear(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"].item(),
            output_scale=act_scales["in_proj:output"].item(),
        )

        # no used, silu is fused in causal_conv1d
        self.activation = self.config.hidden_act
        assert self.activation == "silu"
        self.conv1d = QCausalConv1D(
            originalLayer=copy.deepcopy(originalLayer.conv1d),
            input_scale=act_scales["in_proj:output"].item(),
            output_scale=act_scales["x_proj:input"].item(),            
        )

        # x_proj
        self.x_proj = W8A8BF16OF16Linear(
            originalLayer=copy.deepcopy(originalLayer.x_proj),
            input_scale=act_scales["x_proj:input"].item(),
        )

        self.dt_layernorm = QJambaRMSNorm(
            originalLayer=originalLayer.dt_layernorm,
            output_scale=act_scales["dt_layernorm:output"].item())
        # ~/.cache/huggingface/modules/transformers_modules/Jamba-v0.1/modeling_jamba.py
        self.B_layernorm = QJambaRMSNorm(
            originalLayer=originalLayer.B_layernorm,
            output_scale=act_scales["B_layernorm:output"].item())
        # ~/.cache/huggingface/modules/transformers_modules/Jamba-v0.1/modeling_jamba.py
        self.C_layernorm = QJambaRMSNorm(
            originalLayer=originalLayer.C_layernorm,
            output_scale=act_scales["C_layernorm:output"].item())

        # dt_proj
        original_dt_proj = copy.deepcopy(originalLayer.dt_proj)
        dt_proj_bias = original_dt_proj.bias.clone()
        original_dt_proj.bias = None
        self.dt_proj = W8A8B8O8Linear(
            originalLayer=original_dt_proj,
            input_scale=act_scales["dt_proj:input"].item(),
            output_scale=act_scales["dt_proj:output"].item(),
        )
        
        # ascan
        self.selective_scan = QSScan(
            originalLayer.A_log.clone(), D=originalLayer.D.clone(),
            dt_bias=dt_proj_bias, delta_softplus=True,
            u_scale=act_scales["x_proj:input"],
            dt_scale=act_scales["dt_proj:output"],
            B_scale=act_scales["B_layernorm:output"],
            C_scale=act_scales["C_layernorm:output"],
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


        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: HybridMambaAttentionDynamicCache = None):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
        )
        # # 1. Gated MLP's linear projection
        # projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # # We can't use `mamba_inner_fn` even if in training and without cache params because we have the
        # # inner layernorms which isn't supported by this fused kernel
        # hidden_states, gate = projected_states.chunk(2, dim=1)
        projected_states = self.in_proj.to_seqlen_last(hidden_states) #(B, D, L)
        hidden_states, gate = projected_states.chunk(2, dim=1) #(B, D, L), #(B, D, L)

        # 2. Convolution sequence transformation
        # conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if use_precomputed_states:
            # hidden_states = causal_conv1d_update(
            #     hidden_states.squeeze(-1),
            #     cache_params.conv_states[self.layer_idx],
            #     conv_weights,
            #     self.conv1d.bias,
            #     self.activation,
            # )
            # hidden_states = hidden_states.unsqueeze(-1)
            hidden_states = self.conv1d.update(hidden_states, cache_params.conv_states[self.layer_idx])
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            # hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
            hidden_states = self.conv1d.forward(hidden_states)


        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        # ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        # time_step, B, C = torch.split(
        #     ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        # )
        ssm_parameters = rearrange(hidden_states, "b d l -> b l d").contiguous()
        ssm_parameters = self.x_proj(ssm_parameters)  # (bl d)
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)

        time_step = self.dt_layernorm(time_step)
        B = self.B_layernorm(B)
        C = self.C_layernorm(C)

        # # Here we need to apply dt_proj without the bias, as the bias is added in the selective scan kernel.
        # # This is a hack to apply dt_proj while still using the forward pass of `torch.nn.Linear`, which is needed
        # # in order to make quantization work. Quantization code replaces `torch.nn.Linear` layers with quantized
        # # linear layers, and requires to call the forward pass directly.
        # # The original code here was: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        # time_proj_bias = self.dt_proj.bias
        # self.dt_proj.bias = None
        # discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        # self.dt_proj.bias = time_proj_bias
        discrete_time_step = self.dt_proj.to_seqlen_last(time_step.contiguous())

        # A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        # time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            # scan_outputs = selective_state_update(
            #     cache_params.ssm_states[self.layer_idx],
            #     hidden_states[..., 0],
            #     discrete_time_step[..., 0],
            #     A,
            #     B[:, 0],
            #     C[:, 0],
            #     self.D,
            #     gate[..., 0],
            #     time_proj_bias,
            #     dt_softplus=True,
            # ).unsqueeze(-1)
            y, ssm_state = self.selective_scan.update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0].contiguous(),
                discrete_time_step[..., 0],
                B[:, 0], C[:, 0], z=gate[..., 0])
        else:
            # scan_outputs, ssm_state = selective_scan_fn(
            #     hidden_states,
            #     discrete_time_step,
            #     A,
            #     B.transpose(1, 2),
            #     C.transpose(1, 2),
            #     self.D.float(),
            #     gate,
            #     time_proj_bias,
            #     delta_softplus=True,
            #     return_last_state=True,
            # )
            B = B.transpose(1, 2).contiguous()
            C = C.transpose(1, 2).contiguous()
            scan_outputs, ssm_state = self.selective_scan.forward(
                hidden_states, discrete_time_step, B, C, z=gate, return_last_state=True)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        # contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        scan_outputs = self.had(scan_outputs) # input fp16, output is int8
        contextualized_states = self.out_proj(scan_outputs) # HadW8A8BF16OF16Linear: input int8, output is fp16

        return contextualized_states


    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        if past_key_value is not None:
            cache_params = MambaCacheParams(
                seqlen_offset=0 if hidden_states.shape[1] > 1 else past_key_value.seen_tokens,
            )
            if len(past_key_value.key_cache) > self.layer_idx:
                # we already have cache for this layer, use it
                # remove the dummy seqlen dim (dim=2)
                cache_params.conv_states[self.layer_idx] = past_key_value.key_cache[self.layer_idx].squeeze(2)
                cache_params.ssm_states[self.layer_idx] = past_key_value.value_cache[self.layer_idx].squeeze(2)
            else:
                # we don't have cache for this layer, initialize it with zeros
                batch_size = hidden_states.shape[0]
                cache_params.conv_states[self.layer_idx] = torch.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                cache_params.ssm_states[self.layer_idx] = torch.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
        else:
            cache_params = None

        res = self.cuda_kernels_forward(hidden_states, cache_params)

        if past_key_value is not None:
            past_key_value.update(
                # add dummy seqlen dim (dim=2) to match the number of dimensions of the attention cache
                cache_params.conv_states[self.layer_idx].unsqueeze(2),
                cache_params.ssm_states[self.layer_idx].unsqueeze(2),
                self.layer_idx,
            )

        return res, past_key_value