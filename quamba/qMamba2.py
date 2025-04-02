import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


from .qActLayer import QAct, ActIdentity
from .qLinearLayer import W4A16B16O16Linear
from .qLinearLayer import W4A8B8O8LinearParallel, W4A8B16O16Linear # editable installation
from .qLinearLayer import W8A8B8O8LinearParallel, W8A8B16O16Linear
from .qLinearLayer import HadLinear
from .qConvLayer import QCausalConv1D, Quamb2Conv1D
from .qHadamard import Hadamard, QHadamard
from .qNorm import QRMSNormGated
from .qChunkScan import Quamba2ChunkScan

def get_group_params(scales, ngroups, device):
    if isinstance(scales, list):
        x_head_group_range = []
        x_dim_group_range = []
        x_out_scales = []
        for ssd_g in range(ngroups):
            head_group_size = []
            dim_group_size = []
            out_scales = []
            for (h_gsize, ch_gsize, ch_scales) in scales[ssd_g]:
                # h_gsize: int, ch_gsize: List[int], ch_scales: List[float]
                head_group_size.append(h_gsize)
                dim_group_size.append(ch_gsize)
                out_scales.append(ch_scales)
            head_group_size = torch.stack(head_group_size, dim=0).to(device)
            x_head_group_range.append(torch.cumsum(head_group_size, dim=0).to(torch.int32).to(device))
            dim_group_size = torch.stack(dim_group_size, dim=0).to(device)
            x_dim_group_range.append(torch.cumsum(dim_group_size, dim=1).to(torch.int32).to(device))
            x_out_scales.append(torch.stack(out_scales, dim=0))
        x_head_group_range = torch.stack(x_head_group_range, dim=0) # [n_ssd_groups, n_head_groups]
        x_dim_group_range = torch.stack(x_dim_group_range, dim=0)   # [n_ssd_groups, n_dim_groups]
        x_out_scales = torch.stack(x_out_scales, dim=0)  # [n_ssd_groups, n_head_groups, n_dim_groups]
        return x_head_group_range, x_dim_group_range, x_out_scales
    else:
        return None, None, scales.to(device)



class Mamba2Simple(nn.Module):
    def __init__(
        self,
        originalLayer: Mamba2,
        use_had_transform: bool = True
    ):
        #factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = originalLayer.d_model
        self.d_state = originalLayer.d_state
        self.d_conv = originalLayer.d_conv
        self.conv_init = originalLayer.conv_init
        self.expand = originalLayer.expand
        self.process_group = originalLayer.process_group
        assert self.process_group is None, "Only support process_group=None for now"
        #NOTE(brian1009): We will not use `sequence_parallel` flag, 
        # as we support only single process inference only for now.
        self.sequence_parallel = originalLayer.sequence_parallel 
        self.world_size = 1 # NOTE: ad-hoc
        self.local_rank = 0 # NOTE: ad-hoc
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = originalLayer.headdim
        self.d_ssm = originalLayer.d_ssm
        #NOTE(brian1009): We don't need this assertation, as it will always be true due to the ad-hoc fix of world_size to be 1
        #assert ngroups % self.world_size == 0
        #self.ngroups = ngroups // self.world_size
        self.ngroups = originalLayer.ngroups
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = originalLayer.D_has_hdim
        self.rmsnorm = originalLayer.rmsnorm
        self.norm_before_gate = originalLayer.norm_before_gate
        self.dt_limit = originalLayer.dt_limit
        self.activation = "silu"
        self.chunk_size = originalLayer.chunk_size
        #NOTE(brian1009): Disable mem_eff_path for now
        self.use_mem_eff_path = False 
        self.layer_idx = originalLayer.layer_idx

        # Order: [z, x, B, C, dt]
        # input proj
        if use_had_transform:
            self.in_proj = HadLinear(originalLayer.in_proj, input_transform=True, output_transform=False)
        else:
            self.in_proj = copy.deepcopy(originalLayer.in_proj)

        self.conv1d = originalLayer.conv1d
        self.act = nn.SiLU()

        
        # Initialize log dt bias
        self.dt_bias = originalLayer.dt_bias #NOTE(brain1009): Copy directly
        self.A_log = originalLayer.A_log #NOTE(brain1009): Copy directly
        # D "skip" parameter
        self.D = originalLayer.D #NOTE(brain1009): Copy directly

        if self.rmsnorm:
            self.norm = originalLayer.norm #NOTE(brain1009): Copy directly

        ### Initialization of ActIdentity module for calibrating scales
        self.z_act = ActIdentity(tensor_name="z_act")
        self.x_conv_in = ActIdentity(tensor_name="x_conv_in")
        self.B_conv_in = ActIdentity(tensor_name="B_conv_in")
        self.C_conv_in = ActIdentity(tensor_name="C_conv_in")
        self.x_conv_out = ActIdentity(tensor_name="x_conv_out")
        self.B_conv_out = ActIdentity(tensor_name="B_conv_out")
        self.C_conv_out = ActIdentity(tensor_name="C_conv_out")
        self.dt_act = ActIdentity(tensor_name="dt_act")
        self.ssm_state_act = ActIdentity(tensor_name="ssm_state_act")
        self.ssd_out_act = ActIdentity(tensor_name="ssd_out_act")
        # output proj
        if use_had_transform:
            self.had = Hadamard(originalLayer.out_proj.in_features)
            self.out_proj = HadLinear(originalLayer.out_proj, input_transform=True, output_transform=True)
        else:
            self.had = nn.Identity()
            self.out_proj = copy.deepcopy(originalLayer.out_proj)
        
    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        #NOTE(brian1009): We will not use the inference_params for now, this will only be used durign generation stage.
        # Please ignore reviewing the code under this if block.
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        #NOTE(brian1009) d_mlp is 0 for Mamba2.
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        #NOTE(brian1009) Hence, z0, x0 will also be none...
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        ) #NOTE(brian1009): z0, x0 will have shape of (B, L, 0) for Mamba2

        #NOTE(brian1009): Only need to be considered in generation stage. Skip for now.
        if conv_state is not None:
            if cu_seqlens is None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
            else:
                assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                assert batch == 1, "varlen inference only supports batch dimension 1"
                conv_varlen_states = causal_conv1d_varlen_states(
                    xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                )
                conv_state.copy_(conv_varlen_states)
        assert self.activation in ["silu", "swish"]
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            xBC = self.conv_in_act(xBC) # ActIdentity
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            xBC = self.conv_out_act(xBC)
        else:
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            x = self.x_conv_in(x)
            B = self.B_conv_in(B)
            C = self.C_conv_in(C)
            xBC = torch.cat([x, B, C], dim=-1)
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2), #NOTE(brian1009): (B, L, D) -> (B, D, L) for efficient conv1d
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2)
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x = self.x_conv_out(x) # ActIdentity
        B = self.B_conv_out(B) # ActIdentity
        C = self.C_conv_out(C) # ActIdentity
        dt = self.dt_act(dt) # ActIdentity
        z = self.z_act(z) # ActIdentity
                
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
            return_final_states=ssm_state is not None,
            return_varlen_states=cu_seqlens is not None and inference_params is not None,
        )
        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
            ssm_state = self.ssm_state_act(ssm_state)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.ssd_out_act(y) # ActIdentity
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        y = self.had(y) # HadamardTransform
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        y = self.had(y) # HadamardTransform
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class W4A16QMamba2(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        use_had_transform=True,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        assert self.process_group is None, "Only support process_group=None for now"
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"
        self.act = nn.SiLU()

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = W4A16B16O16Linear(self.d_model, d_in_proj, group_size=128, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.register_buffer('dt_bias', inv_dt)
        # Initialize A 
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.register_buffer('A_log', A_log)
        # D "skip" parameter
        self.register_buffer('D', torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))

        assert self.rmsnorm == True
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                    group_size=self.d_ssm // ngroups, **factory_kwargs)

        # output proj
        if use_had_transform:
            self.had = Hadamard(self.d_inner)
        else:
            self.had = nn.Identity()
        self.out_proj = W4A16B16O16Linear(self.d_inner, self.d_model, group_size=128, **factory_kwargs)
    
    @classmethod
    def from_fp16(
        cls,
        originalLayer: Mamba2Simple,
        use_had_transform=True
    ):
        qmixer = cls(
            d_model = originalLayer.d_model,
            d_state = originalLayer.d_state,
            d_conv = originalLayer.d_conv,
            conv_init = originalLayer.conv_init,
            expand = originalLayer.expand,
            headdim = originalLayer.headdim,
            d_ssm = originalLayer.d_ssm,
            ngroups = originalLayer.ngroups*originalLayer.world_size,
            rmsnorm = originalLayer.rmsnorm,
            norm_before_gate = originalLayer.norm_before_gate,
            use_had_transform = use_had_transform,
            dt_limit = originalLayer.dt_limit,
            chunk_size = originalLayer.chunk_size,
            use_mem_eff_path = False,
            layer_idx = originalLayer.layer_idx,
            sequence_parallel = originalLayer.sequence_parallel,
            process_group = originalLayer.process_group,
        )

        # # input proj, weight group_size=128
        qmixer.in_proj = W4A16B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
        )
        qmixer.conv1d = copy.deepcopy(originalLayer.conv1d)
        # Initialize log dt bias
        qmixer.dt_bias = originalLayer.dt_bias.clone()
        qmixer.A_log = originalLayer.A_log.clone()
        # D "skip" parameter
        qmixer.D = originalLayer.D.clone()

        if qmixer.rmsnorm:
            qmixer.norm = copy.deepcopy(originalLayer.norm)

        qmixer.out_proj = W4A16B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
        )
        return qmixer

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        #NOTE(brian1009): We will not use the inference_params for now, this will only be used durign generation stage.
        # Please ignore reviewing the code under this if block.
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        #NOTE(brian1009) d_mlp is 0 for Mamba2.
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        #NOTE(brian1009) Hence, z0, x0 will also be none...
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        ) #NOTE(brian1009): z0, x0 will have shape of (B, L, 0) for Mamba2

        #NOTE(brian1009): Only need to be considered in generation stage. Skip for now.
        if conv_state is not None:
            if cu_seqlens is None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
            else:
                assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                assert batch == 1, "varlen inference only supports batch dimension 1"
                conv_varlen_states = causal_conv1d_varlen_states(
                    xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                )
                conv_state.copy_(conv_varlen_states)
        assert self.activation in ["silu", "swish"]
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2), #NOTE(brian1009): (B, L, D) -> (B, D, L) for efficient conv1d
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2)
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
            return_final_states=ssm_state is not None,
            return_varlen_states=cu_seqlens is not None and inference_params is not None,
        )
        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        y = self.had(y) # HadamardTransform
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step, inplace modify conv_state
        xBC = causal_conv1d_update(
            xBC,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # Triton SSD step
        A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=self.headdim)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
        D = repeat(self.D, "h -> h p", p=self.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        if not self.rmsnorm:
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
        y = selective_state_update(
            ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
            dt_bias=dt_bias, dt_softplus=True
        )
        y = rearrange(y, "b h p -> b (h p)")
        
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        # conv_dtype is torch.float16
        conv_dtype = torch.float16
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)

        # ssm_dtype is torch.float16
        ssm_dtype = torch.float16
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
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
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=conv_dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
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


class W4A8QMamba2(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        use_had_transform=True,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16} # dtype is for norm layers
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        assert self.process_group is None, "Only support process_group=None for now"
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"
        self.act = nn.SiLU()

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = W4A8B8O8LinearParallel(self.d_model, d_in_proj, group_size=128, **factory_kwargs)
        
        # causal conv
        assert self.activation == "silu"
        x_nhead_group = 4
        x_ndim_group = 4
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = Quamb2Conv1D(
            self.d_ssm, self.headdim, self.d_state, self.ngroups, x_nhead_group, x_ndim_group,
            conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, bias=True, **factory_kwargs)
        # SSD
        self.qchunk_scan = Quamba2ChunkScan(
            self.d_ssm, self.headdim, self.d_state, self.ngroups, self.D_has_hdim, self.chunk_size,
                 x_nhead_group, x_ndim_group, delta_softplus=True, dt_limit=self.dt_limit, **factory_kwargs)
        # Norm
        assert self.rmsnorm, "Only support Mamba2 block with rmsnorm"
        self.norm = QRMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                    group_size=self.d_ssm // ngroups, use_float16_output=True,
                                    **factory_kwargs)
        # output proj
        if use_had_transform:
            self.had = QHadamard(self.d_inner, x_H_scale=1.0)
        else:
            self.had = QAct(scale=1.0)
        self.out_proj = W4A8B16O16Linear(self.d_inner, self.d_model, group_size=128, **factory_kwargs)

    @classmethod
    def from_fp16(
        cls,
        originalLayer: Mamba2Simple,
        act_scales: Dict,
        use_had_transform: bool = True
    ):
        qmixer = cls(
            d_model = originalLayer.d_model,
            d_state = originalLayer.d_state,
            d_conv = originalLayer.d_conv,
            conv_init = originalLayer.conv_init,
            expand = originalLayer.expand,
            headdim = originalLayer.headdim,
            d_ssm = originalLayer.d_ssm,
            ngroups = originalLayer.ngroups*originalLayer.world_size,
            rmsnorm = originalLayer.rmsnorm,
            norm_before_gate = originalLayer.norm_before_gate,
            use_had_transform = use_had_transform,
            dt_limit = originalLayer.dt_limit,
            chunk_size = originalLayer.chunk_size,
            use_mem_eff_path = False,
            layer_idx = originalLayer.layer_idx,
            sequence_parallel = originalLayer.sequence_parallel,
            process_group = originalLayer.process_group,
        )

        # input proj, weight group_size=128
        qmixer.in_proj = W4A8B8O8LinearParallel.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"],
            output_scales=[
                act_scales["z_act:input"],          # z scale
                act_scales["x_conv_in:input"],      # x scale
                act_scales["B_conv_in:input"],      # B scale
                act_scales["C_conv_in:input"],      # C scale
                act_scales["dt_act:input"],         # dt scale
            ],
            out_split_dims=[
                qmixer.d_ssm, qmixer.d_ssm, qmixer.ngroups*qmixer.d_state,
                qmixer.ngroups*qmixer.d_state, qmixer.nheads],
        )
        
        device = originalLayer.conv1d.weight.device
        x_head_group_range, x_dim_group_range, x_out_scales = get_group_params(act_scales["x_conv_out:input"], qmixer.ngroups, device)
        qmixer.conv1d = Quamb2Conv1D.from_fp16(
                copy.deepcopy(originalLayer.conv1d),
                qmixer.d_ssm, qmixer.headdim,
                qmixer.d_state, qmixer.ngroups,
                act_scales["x_conv_in:output"],
                act_scales["B_conv_in:output"],
                act_scales["C_conv_in:output"],
                x_out_scales, # [n_ssd_groups, n_head_groups, n_dim_groups] or torch.tensor([])
                act_scales["B_conv_out:input"],
                act_scales["C_conv_out:input"],
                x_head_group_range, # [n_ssd_groups, n_head_groups] or None
                x_dim_group_range,  # [n_ssd_groups, n_head_groups, n_dim_groups] or None
            )
        # SSD
        qmixer.qchunk_scan = Quamba2ChunkScan.from_fp16(
            qmixer.d_ssm, qmixer.headdim,
            qmixer.d_state, qmixer.ngroups,
            x_out_scales,       # [n_ssd_groups, n_head_groups, n_dim_groups] or torch.tensor([])
            x_head_group_range, # [n_ssd_groups, n_head_groups] or None
            x_dim_group_range,  # [n_ssd_groups, n_head_groups, n_dim_groups] or None
            originalLayer.A_log,
            originalLayer.chunk_size,
            D=originalLayer.D,
            D_has_hdim=qmixer.D_has_hdim,
            dt_bias=originalLayer.dt_bias,
            delta_softplus=True,
            dt_scale=act_scales["dt_act:output"],
            B_scale=act_scales["B_conv_out:input"],
            C_scale=act_scales["C_conv_out:input"],
            ssm_state_scale=act_scales["ssm_state_act:input"],
            dt_limit=originalLayer.dt_limit
        )
        # Norm
        assert originalLayer.rmsnorm, "Only support Mamba2 block with rmsnorm"
        qmixer.norm = QRMSNormGated.from_fp16(
                        originalLayer.norm,
                        z_scale=act_scales["z_act:output"].item(),
                        use_float16_output=True)
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

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        #NOTE(brian1009): We will not use the inference_params for now, this will only be used durign generation stage.
        # Please ignore reviewing the code under this if block.
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # # If the model is loaded in fp16, without the .float() here, A might be -inf
        # A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        #NOTE(brian1009) d_mlp is 0 for Mamba2.
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        #NOTE(brian1009) Hence, z0, x0 will also be none...
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        ) #NOTE(brian1009): z0, x0 will have shape of (B, L, 0) for Mamba2

        assert self.activation in ["silu", "swish"]
        # Perform causal conv1d and return conv_state
        if conv_state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC, "b l d -> b d l")
            conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
        x, B, C = self.conv1d.forward(xBC.transpose(1, 2))
        x = rearrange(x, "b (h p) l -> b l h p", p=self.headdim)
        B = rearrange(B, "b (g n) l -> b l g n", g=self.ngroups)
        C = rearrange(C, "b (g n) l -> b l g n", g=self.ngroups)
        y = self.qchunk_scan(x, dt, B, C, z=None, return_final_states=ssm_state is not None)

        if ssm_state is not None:
            y, last_state = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                raise NotImplementedError("Not implemented for cu_seqlens yet")
        
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        # Output projection
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y) # HadW8A8BF16OF16Linear: input int8, output is fp16
        return out


    def step(self, hidden_states, conv_state, ssm_state):
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step, inplace modify conv_state
        x, B, C = self.conv1d.update(xBC, conv_state)

        # SSM step, inplace modify ssm_state
        y = self.qchunk_scan.update(ssm_state, x, dt, B, C, z=None)
        y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        # conv_dtype is torch.int8
        conv_dtype = torch.int8
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)

        # ssm_dtype is torch.int8
        ssm_dtype = torch.int8
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        # conv_dtype is torch.int8
        conv_dtype = torch.int8
        # ssm_dtype is torch.float16
        ssm_dtype = torch.float16
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=conv_dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
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


class W8A8QMamba2(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        use_had_transform=True,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        assert self.process_group is None, "Only support process_group=None for now"
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"
        self.act = nn.SiLU()

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = W8A8B8O8LinearParallel(self.d_model, d_in_proj)
        
        # causal conv
        assert self.activation == "silu"
        x_nhead_group = 4
        x_ndim_group = 4
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = Quamb2Conv1D(
            self.d_ssm, self.headdim, self.d_state, self.ngroups, x_nhead_group, x_ndim_group,
            conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, bias=True)
        # SSD
        self.qchunk_scan = Quamba2ChunkScan(
            self.d_ssm, self.headdim, self.d_state, self.ngroups, self.D_has_hdim, self.chunk_size,
                 x_nhead_group, x_ndim_group, delta_softplus=True, dt_limit=self.dt_limit)
        # Norm
        assert self.rmsnorm, "Only support Mamba2 block with rmsnorm"
        self.norm = QRMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                    group_size=self.d_ssm // ngroups, use_float16_output=True,
                                    device=factory_kwargs["device"])
        # output proj
        if use_had_transform:
            self.had = QHadamard(self.d_inner, x_H_scale=1.0)
        else:
            self.had = QAct(scale=1.0)
        self.out_proj = W8A8B16O16Linear(self.d_inner, self.d_model)

    @classmethod
    def from_fp16(
        cls,
        originalLayer: Mamba2Simple,
        act_scales: Dict,
        use_had_transform: bool = True
    ):

        qmixer = cls(
            d_model = originalLayer.d_model,
            d_state = originalLayer.d_state,
            d_conv = originalLayer.d_conv,
            conv_init = originalLayer.conv_init,
            expand = originalLayer.expand,
            headdim = originalLayer.headdim,
            d_ssm = originalLayer.d_ssm,
            ngroups = originalLayer.ngroups*originalLayer.world_size,
            rmsnorm = originalLayer.rmsnorm,
            norm_before_gate = originalLayer.norm_before_gate,
            use_had_transform = use_had_transform,
            dt_limit = originalLayer.dt_limit,
            chunk_size = originalLayer.chunk_size,
            use_mem_eff_path = False,
            layer_idx = originalLayer.layer_idx,
            sequence_parallel = originalLayer.sequence_parallel,
            process_group = originalLayer.process_group,
        )

        qmixer.in_proj = W8A8B8O8LinearParallel.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"].item(),
            output_scales=[
                act_scales["z_act:input"].item(),          # z scale
                act_scales["x_conv_in:input"].item(),      # x scale
                act_scales["B_conv_in:input"].item(),      # B scale
                act_scales["C_conv_in:input"].item(),      # C scale
                act_scales["dt_act:input"].item(),         # dt scale
            ],
            out_split_dims=[
                qmixer.d_ssm, qmixer.d_ssm, qmixer.ngroups*qmixer.d_state,
                qmixer.ngroups*qmixer.d_state, qmixer.nheads
            ],
        )

        # causal conv
        # no used, silu is fused in causal_conv1d
        device = originalLayer.conv1d.weight.device
        x_head_group_range, x_dim_group_range, x_out_scales = get_group_params(act_scales["x_conv_out:input"], qmixer.ngroups, device)
        qmixer.conv1d = Quamb2Conv1D.from_fp16(
                copy.deepcopy(originalLayer.conv1d),
                qmixer.d_ssm, qmixer.headdim,
                qmixer.d_state, qmixer.ngroups,
                act_scales["x_conv_in:output"],
                act_scales["B_conv_in:output"],
                act_scales["C_conv_in:output"],
                x_out_scales, # [n_ssd_groups, n_head_groups, n_dim_groups] or torch.tensor([])
                act_scales["B_conv_out:input"],
                act_scales["C_conv_out:input"],
                x_head_group_range, # [n_ssd_groups, n_head_groups] or None
                x_dim_group_range,  # [n_ssd_groups, n_head_groups, n_dim_groups] or None
            )
        # SSD
        qmixer.qchunk_scan = Quamba2ChunkScan.from_fp16(
            qmixer.d_ssm, qmixer.headdim,
            qmixer.d_state, qmixer.ngroups,
            x_out_scales,       # [n_ssd_groups, n_head_groups, n_dim_groups] or torch.tensor([])
            x_head_group_range, # [n_ssd_groups, n_head_groups] or None
            x_dim_group_range,  # [n_ssd_groups, n_head_groups, n_dim_groups] or None
            originalLayer.A_log,
            originalLayer.chunk_size,
            D=originalLayer.D,
            D_has_hdim=qmixer.D_has_hdim,
            dt_bias=originalLayer.dt_bias,
            delta_softplus=True,
            dt_scale=act_scales["dt_act:output"],
            B_scale=act_scales["B_conv_out:input"],
            C_scale=act_scales["C_conv_out:input"],
            ssm_state_scale=act_scales["ssm_state_act:input"],
            dt_limit=originalLayer.dt_limit
        )
        # Norm
        assert qmixer.rmsnorm, "Only support Mamba2 block with rmsnorm"
        qmixer.norm = QRMSNormGated.from_fp16(
                        originalLayer.norm,
                        z_scale=act_scales["z_act:output"].item(),
                        use_float16_output=True)
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

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        #NOTE(brian1009): We will not use the inference_params for now, this will only be used durign generation stage.
        # Please ignore reviewing the code under this if block.
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # # If the model is loaded in fp16, without the .float() here, A might be -inf
        # A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        #NOTE(brian1009) d_mlp is 0 for Mamba2.
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        #NOTE(brian1009) Hence, z0, x0 will also be none...
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        ) #NOTE(brian1009): z0, x0 will have shape of (B, L, 0) for Mamba2

        assert self.activation in ["silu", "swish"]
        # Perform causal conv1d and return conv_state
        if conv_state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC, "b l d -> b d l")
            conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
        x, B, C = self.conv1d.forward(xBC.transpose(1, 2))
        x = rearrange(x, "b (h p) l -> b l h p", p=self.headdim)
        B = rearrange(B, "b (g n) l -> b l g n", g=self.ngroups)
        C = rearrange(C, "b (g n) l -> b l g n", g=self.ngroups)
        y = self.qchunk_scan(x, dt, B, C, z=None, return_final_states=ssm_state is not None)

        if ssm_state is not None:
            y, last_state = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                raise NotImplementedError("Not implemented for cu_seqlens yet")
        
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        # Output projection
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y) # HadW8A8BF16OF16Linear: input int8, output is fp16
        return out


    def step(self, hidden_states, conv_state, ssm_state):
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step, inplace modify conv_state
        x, B, C = self.conv1d.update(xBC, conv_state)

        # SSM step, inplace modify ssm_state
        y = self.qchunk_scan.update(ssm_state, x, dt, B, C, z=None)
        y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        # conv_dtype is torch.int8
        conv_dtype = torch.int8
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)

        # ssm_dtype is torch.int8
        ssm_dtype = torch.int8
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        # conv_dtype is torch.int8
        conv_dtype = torch.int8
        # ssm_dtype is torch.float16
        ssm_dtype = torch.float16
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=conv_dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
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