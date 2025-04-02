# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
from typing import Optional, Dict, Type, Tuple
import json
import os
import gc
import copy

from collections import namedtuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel 
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel 
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

import quamba.qBlock

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import quamba

# To use quamba, set cache_dir="./configs"
def load_config_hf(model_name, cache_dir=None):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, cache_dir=cache_dir, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

# To use quamba, set cache_dir="./configs"
def load_state_dict_hf(model_name, device=None, cache_dir=None):
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, cache_dir=cache_dir, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=device, weights_only=True)

@dataclass
class QuambaConfig(MambaConfig):
    # inherited all fields from MambaConfig
    # we add a norm_config field for Quamba
    norm_cfg: dict = field(default_factory=dict)
    embedding_cfg: dict = field(default_factory=dict)
    lm_head_cfg: dict = field(default_factory=dict)

    is_hybrid: bool = False

def create_quantized_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    norm_cfg=None,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    is_hybrid=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        supported_layers = [
            "W8A8QMamba", "W4A16QMamba", "W4A8QMamba",
            "W8A8QMamba2", "W4A16QMamba2", "W4A8QMamba2",
        ]
        if is_hybrid:
            # generate a dictionary of mixer class
            ssm_layer_dicts = ssm_cfg.pop("layer") # this will be a list
            ssm_layer_dict = ssm_layer_dicts[layer_idx]
        
            mixer_classes = {}
            for name, mixer_cls_name in ssm_layer_dict.items():
                if mixer_cls_name not in supported_layers:
                    raise ValueError(f"Invalid ssm_layer: {mixer_cls_name}, only support {supported_layers}")
                mixer_classes[name] = partial(
                    getattr(quamba, mixer_cls_name),
                    layer_idx=layer_idx,
                    **ssm_cfg,
                    **factory_kwargs
                )
        else:
            ssm_layer = ssm_cfg.pop("layer", "W8A8QMamba2")
            if ssm_layer not in supported_layers:
                raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support {supported_layers}")
            mixer_cls = partial(
                getattr(quamba, ssm_layer),
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    assert rms_norm is True, "Only RMSNorm is supported in Quamba"
    norm_cfg = copy.deepcopy(norm_cfg) if norm_cfg is not None else {}
    
    if is_hybrid:
        norm_dicts = norm_cfg.pop("norm") # this will be a list
        norm_dict = norm_dicts[layer_idx]
        supported_norms = ["QRMSNorm", "RMSNorm"]
        norm_classes = {}
        for name, norm_cls_name in norm_dict.items():
            if norm_cls_name not in supported_norms:
                raise ValueError(f"Invalid block_norm: {norm_cls_name}, only support {supported_norms}")
            norm_classes[name] = partial(
                getattr(quamba, norm_cls_name),
                eps=norm_epsilon,
                **factory_kwargs
            )
        block = quamba.qBlock.HybridQuambaBlock(
            d_model,
            mixer_classes=mixer_classes,
            norm_classes=norm_classes,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
    else:
        block_norm = norm_cfg.pop("norm", "QRMSNorm")
        supported_norms = ["QRMSNorm", "RMSNorm"]
        if block_norm not in supported_norms:
            raise ValueError(f"Invalid block_norm: {block_norm}, only support {supported_norms}")
        norm_cls = partial(getattr(quamba, block_norm), eps=norm_epsilon, **factory_kwargs)
        if d_intermediate == 0:
            mlp_cls = nn.Identity
        else:
            mlp_cls = partial(
                GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
            )
        block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
    return block

#NOTE(brian1009): Temporary workaround for hybrid model's creation
#Idealy, it should be placed under the 'HybridQuambaMixerModel' class
def set_default_configuration(model):
    for layer in model.backbone.layers:
        if hasattr(layer, "set_default_mixer"):
            layer.set_default_mixer()
        else:
            raise ValueError("Layer does not support set_default_mixer")
def set_layer_wise_configuration(model, config_names):
    for config_name, layer in zip(config_names, model.backbone.layers):
        if hasattr(layer, "set_mixer"):
            layer.set_mixer(config_name)
        else:
            raise ValueError("Layer does not support set_mixer")
        
class HybridQuambaMixerModel(MixerModel, nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        norm_cfg=None,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        self._validate_configuration(ssm_cfg, norm_cfg)
        factory_kwargs = {"device": device, "dtype": dtype}
        # only init nn.Module here, because we don't want to create fp16 modules in MixerModel
        nn.Module.__init__(self)
        self.residual_in_fp32 = residual_in_fp32

        # FIXME: hardcode the with w4 for now.
        self.embedding = quamba.W4O16Embedding(vocab_size, d_model)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_quantized_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    norm_cfg=norm_cfg,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    is_hybrid=True,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        assert rms_norm is True, "Only RMSNorm is supported in Quamba"
        norm_cfg = copy.deepcopy(norm_cfg) if norm_cfg is not None else {}
        
        #FIXME: hardcode the for w4a8 for now.
        self.norm_f = quamba.QRMSNorm(d_model, eps=norm_epsilon,**factory_kwargs)
    
    def _validate_configuration(self, ssm_cfg, norm_cfg):
        ssm_cfg = copy.deepcopy(ssm_cfg)
        ssm_layers_config = ssm_cfg.pop("layer", {})
        
        if not ssm_layers_config:
            raise ValueError("No configuration provided for HybridQuambaMixerModel")
        
        norm_cfg = copy.deepcopy(norm_cfg)
        norm_dict = norm_cfg.pop("norm", {})
        
        if not norm_dict:
            raise ValueError("No configuration provided for HybridQuambaMixerModel")
        
        # check the keys match
        if type(ssm_layers_config) == dict:
            if set(ssm_layers_config.keys()) != set(norm_dict.keys()):
                raise ValueError("Configuration keys do not match between ssm and norm")
        elif type(ssm_layers_config) == list:
            for ssm_layer_dict, norm_layer_dict in zip(ssm_layers_config, norm_dict):
                if set(ssm_layer_dict.keys()) != set(norm_layer_dict.keys()):
                    raise ValueError("Configuration keys do not match between ssm and norm")

    
class QuambaMixerModel(MixerModel, nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        embedding_cfg=None,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        norm_cfg=None,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # only init nn.Module here, because we don't want to create fp16 modules in MixerModel
        nn.Module.__init__(self)
        self.residual_in_fp32 = residual_in_fp32

        embedding_cfg = copy.deepcopy(embedding_cfg) if embedding_cfg is not None else {}
        embed_layer = embedding_cfg.pop("layer", "Embedding")
        supported_embedding = ["Embedding", "W4O16Embedding", "W8O16Embedding"]
        if embed_layer not in supported_embedding:
            raise ValueError(f"Invalid embedding layer: {embed_layer}, only support {supported_embedding}")

        if embed_layer == "Embedding":
            self.embedding = torch.nn.Embedding(vocab_size, d_model, **factory_kwargs)
        elif embed_layer == "W8O16Embedding" or embed_layer == "W4O16Embedding":
            self.embedding = getattr(quamba, embed_layer)(vocab_size, d_model, **factory_kwargs)
        else:
            raise ValueError(f"Invalid embed_layer: {embed_layer}")

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_quantized_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    norm_cfg=norm_cfg,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        assert rms_norm is True, "Only RMSNorm is supported in Quamba"
        norm_cfg = copy.deepcopy(norm_cfg) if norm_cfg is not None else {}
        block_norm = norm_cfg.pop("norm", "QRMSNorm")
        supported_norms = ["QRMSNorm", "RMSNorm"]
        if block_norm not in supported_norms:
            raise ValueError(f"Invalid block_norm: {block_norm}, only support {supported_norms}")
        self.norm_f = getattr(quamba, block_norm)(d_model, eps=norm_epsilon,**factory_kwargs)


class QuambaLMHeadModel(MambaLMHeadModel, nn.Module):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        **kwargs
    ) -> None:

        # only init nn.Module here, because we don't want to create fp16 modules in MambaLMHeadModel
        nn.Module.__init__(self)
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        embedding_cfg = config.embedding_cfg
        lm_head_cfg = config.lm_head_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        norm_cfg = config.norm_cfg
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        is_hybrid = config.is_hybrid
        factory_kwargs = {"device": device, "dtype": torch.float16} # use fp16 for non-quantized ops

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        if is_hybrid:
            self.backbone = HybridQuambaMixerModel(
                d_model=d_model,
                n_layer=n_layer,
                d_intermediate=d_intermediate,
                vocab_size=vocab_size,
                ssm_cfg=ssm_cfg,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                rms_norm=rms_norm,
                norm_cfg=norm_cfg,
                initializer_cfg=initializer_cfg,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                **factory_kwargs,
            )
        else:
            self.backbone = QuambaMixerModel(
                d_model=d_model,
                n_layer=n_layer,
                d_intermediate=d_intermediate,
                vocab_size=vocab_size,
                embedding_cfg=embedding_cfg,
                ssm_cfg=ssm_cfg,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                rms_norm=rms_norm,
                norm_cfg=norm_cfg,
                initializer_cfg=initializer_cfg,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                **factory_kwargs,
            )

        if is_hybrid: #FIXME: hardcode the for w4a8 for now.
            self.lm_head = quamba.W4A8B16O16Linear(d_model, vocab_size, group_size=128)
        else:
            lm_head_cfg = copy.deepcopy(lm_head_cfg) if lm_head_cfg is not None else {}
            lm_head_layer = lm_head_cfg.pop("layer", "Linear")
            supported_lm_heads = ["Linear", "W4A16B16O16Linear", "W4A8B16O16Linear", "W8A8B16O16Linear"]
            if lm_head_layer not in supported_lm_heads:
                raise ValueError(f"Invalid lm_head layer: {lm_head_layer}, only support {supported_lm_heads}")
            if lm_head_layer == "Linear":
                self.lm_head = torch.nn.Linear(d_model, vocab_size)
            elif lm_head_layer == "W4A16B16O16Linear" or lm_head_layer == "W4A8B16O16Linear" or lm_head_layer == "W8A8B16O16Linear":
                self.lm_head = getattr(quamba, lm_head_layer)(d_model, vocab_size)
            else:
                raise ValueError(f"Invalid lm_head layer: {lm_head_layer}")
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        config_data = load_config_hf(pretrained_model_name, cache_dir=cache_dir)
        config = QuambaConfig(**config_data)
        model = cls(config, device="cpu", **kwargs) # we always load model on cpu first to save memory
        loaded_model = load_state_dict_hf(pretrained_model_name, device="cpu", cache_dir=cache_dir)
        model.load_state_dict(loaded_model)
        del loaded_model
        torch.cuda.empty_cache()
        gc.collect()
        return model.to(device)

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)
        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)


if __name__ == "__main__":

    # To use quamba, set cache_dir="./configs"
    # pretrained_model_name = "ut-enyac/quamba2-130m-w4a16"
    pretrained_model_name = "ut-enyac/quamba2-130m-w4a8"
    # pretrained_model_name = "ut-enyac/quamba2-130m-w8a8"
    config_data = load_config_hf(pretrained_model_name, cache_dir="./configs")
    print(config_data)
    # model = load_state_dict_hf(pretrained_model_name, cache_dir="./configs")
    # print(model.keys())
    # model = QuambaLMHeadModel.from_pretrained(pretrained_model_name, device="cuda", cache_dir="./configs")
    config = QuambaConfig(**config_data)
    model = QuambaLMHeadModel(config, device="cuda")
    model.save_pretrained(pretrained_model_name)
    # model = QuambaLMHeadModel.from_pretrained(pretrained_model_name, device="cuda", cache_dir=pretrained_model_name)
