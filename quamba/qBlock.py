import torch
import torch.nn as nn
from typing import Optional, Dict, Type, Tuple, Callable
from mamba_ssm.modules.block import Block

class HybridQuambaBlock(nn.Module):
    def __init__(
        self, 
        dim: Optional[int] = None, 
        mixer_classes = None, 
        norm_classes = None,
        fused_add_norm: bool = False, 
        residual_in_fp32: bool = False
    ):
        """
        BlockWrapper that supports multiple mixer and norm class pairs, selectable dynamically.

        Args:
            dim: Dimension of the input.
            mixer_norm_classes: Dictionary of mixer class types and their corresponding norm classes.
            fused_add_norm: Whether to use fused add and norm (default: False).
            residual_in_fp32: Whether to keep residual in fp32 (default: False).
            default_mixer: The default mixer class to use.
        """
        super().__init__()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        self.current_mixer_key = None
        self.is_complete_class = False
        if mixer_classes is None or norm_classes is None:
            self.mixers = nn.ModuleDict()
            self.norms = nn.ModuleDict()
            self._support_mode = []
        else:
            self.is_complete_class = True
            assert dim is not None, "Dimension must be provided when mixer_norm_classes is not None."
            #check if the key of the mixer and norm classes are the same
            for key in mixer_classes.keys():
                if key not in norm_classes.keys():
                    raise ValueError(f"Key '{key}' is not found in the norm_classes. The norm_classes must have the same keys as the mixer_classes.")
            
            self.mixers = nn.ModuleDict({name: mixer_cls(dim) for name, mixer_cls in mixer_classes.items()})
            self.norms = nn.ModuleDict({name: norm_cls(dim) for name, norm_cls in norm_classes.items()})
            self._support_mode = list(mixer_classes.keys())
            # check if the mixer and norm classes have the same keys
            assert set(self.mixers.keys()) == set(self.norms.keys()), "Mixer and Norm classes mismatch."
            self._validate_mixer_norm_classes()
            self.set_default_mixer()
            
    def _validate_mixer_norm_classes(self):
        """
        Validate the mixer and norm classes.
        """
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Please provide the mixer_norm_classes in the constructor.")
        if not set(self.mixers.keys()) == set(self.norms.keys()):
            raise ValueError("Mixer and norm classes have mismatch keys.")

    def set_default_mixer(self):
        """
        Set the default mixer class to use.
        """
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Please provide the mixer_norm_classes in the constructor.")
        self.current_mixer_key = list(self.mixers.keys())[0]

    def set_mixer(self, mixer_key: str):
        """
        Set the current mixer and its corresponding norm class to use.

        Args:
            mixer_key: The key of the mixer to activate.
        """
        if mixer_key not in self.mixers:
            raise ValueError(f"Mixer key '{mixer_key}' is not available in the mixers: {list(self.mixers.keys())}")
        self.current_mixer_key = mixer_key

    def keep_only_current_mixer(self):
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Please provide the mixer_norm_classes in the constructor.")
        if self.current_mixer_key is None:
            raise ValueError("Mixer key is not set. Please call `set_mixer` to set the current mixer.")
        for key in self._support_mode:
            if key != self.current_mixer_key:
                del self.mixers[key]
                del self.norms[key]
        
        self._support_mode = [self.current_mixer_key]

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        residual: Optional[torch.Tensor] = None, 
        inference_params=None, 
        **mixer_kwargs
    ):
        """
        Forward pass through the block wrapper.

        Args:
            hidden_states: Input tensor.
            residual: Residual tensor (optional).
            inference_params: Parameters for inference (optional).
            mixer_kwargs: Additional arguments for the mixer.

        Returns:
            Tuple of (hidden_states, residual).
        """
        # Get the current mixer and norm module
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Please provide the mixer_norm_classes in the constructor.")
        if self.current_mixer_key is None:
            raise ValueError("Mixer key is not set. Please call `set_mixer` to set the current mixer.")
        
        mixer = self.mixers[self.current_mixer_key]
        norm = self.norms[self.current_mixer_key]

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = norm(residual.to(dtype=norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = norm(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32
            )

        hidden_states = mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        # NOTE(brian1009): MLP is removed for now, as it is not used in the current implementation.
        # Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
        return hidden_states, residual
    
    def set_module_dict(self, mixer_dict: Dict[str, Type[nn.Module]], norm_dict: Dict[str, Type[nn.Module]]):
        """
        Set the mixer and norm classes for the block.

        Args:
            mixer_classes: Dictionary of mixer class types.
            norm_classes: Dictionary of norm class types.
        """
        assert self.is_complete_class == False, "Mixer and norm classes are already set. Please create a new instance of the class."
        self.mixers = nn.ModuleDict(mixer_dict)
        self.norms = nn.ModuleDict(norm_dict)
        self._support_mode = list(self.mixers.keys())
        # check if the mixer and norm classes have the same keys
        self._validate_mixer_norm_classes()
        self.is_complete_class = True
    
    def get_supported_modes(self):
        """
        Get the supported modes for the block.

        Returns:
            List of supported modes.
        """
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Hence, get_supported_modes is non-available.\
                             Please provide the mixer_norm_classes in the constructor.")
        return self._support_mode
    
    def set_mode(self, mode: str):
        """
        Set the mode for the block.

        Args:
            mode: Mode to set.
        """
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Hence, set_mode is non-available.\
                             Please provide the mixer_norm_classes in the constructor.")
        if mode not in self._support_mode:
            raise ValueError(f"Mode '{mode}' is not supported. Supported modes are: {self._support_mode}")
        self.current_mixer_key = mode
    
    
    def get_class_info_dict_mixers_norms(self):
        """
        Get the dictionary of mixer and norm classes.

        Returns:
            Dictionary of configurable type to the name of its corresponding mixer and norm classes.
        """
        if not self.is_complete_class:
            raise ValueError("Mixer and norm classes are not set. Please provide the mixer_norm_classes in the constructor.")
        return {
            "mixers": {name: mixer.__class__.__name__ for name , mixer in self.mixers.items()},
            "norms": {name: norm.__class__.__name__ for name , norm in self.norms.items()}
        }
    
    def __repr__(self):
        """
        Custom string representation for the Block class.
        Displays the mode it is currently operating in and its child modules.
        """
        # Start with the main Block description
        repr_str = (
            f"{self.__class__.__name__}(current_mixer_key={self.current_mixer_key},\n"
        )

        # Include all child modules
        for name, module in self.named_children():
            module_str = repr(module).replace("\n", "\n    ")  # Indent submodule representation
            repr_str += f"    ({name}): {module_str},\n"

        # Close the representation
        repr_str += ")"
        return repr_str
    
    @classmethod
    def from_block_and_mixer_norm_dict(
            cls, block: Block,
            mixers_dict: Dict[str, nn.Module], 
            norms_dict: Dict[str, nn.Module]
        ):
        """
        Create a HybridQuambaBlock instance from an existing block and mixer norm classes.

        Args:
            block: Existing block to extract the dimension and MLP class from.
            mixer_norm_classes: Dictionary of function to quantized mixer and norms

        Returns:
            HybridQuambaBlock instance.
        """
        
        hybrid_block = cls(
            fused_add_norm = block.fused_add_norm,
            residual_in_fp32 = block.residual_in_fp32
        )
        hybrid_block.mixers = nn.ModuleDict(mixers_dict)
        hybrid_block.norms = nn.ModuleDict(norms_dict)
        hybrid_block.is_complete_class = True
            
        return hybrid_block

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if self.current_mixer_key is None:
            raise ValueError("Mixer key is not set. Please call `set_mixer` to set the current mixer.")
        return self.mixers[self.current_mixer_key].allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)