import torch
import torch.nn as nn

class QAct(nn.Module):
    def __init__(
        self,
        scale
    ):
        super().__init__()
        self.scale = scale
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'scale'] = self.scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.scale = state_dict[prefix + 'scale']
        del state_dict[prefix + 'scale']

    @torch.no_grad()
    def forward(self, x):
        return (x / self.scale).clamp(min=-128, max=127).to(torch.int8) # quant
    
    def __repr__(self):
        return f"QAct()"



class ActIdentity(nn.Module):
    def __init__(self, tensor_name):
        self.tensor_name = tensor_name
        super().__init__()
        
    @torch.no_grad()
    def forward(self, x):
        return x
    
    def __repr__(self):
        return f"ActIdentity({self.tensor_name})"
    
    
    