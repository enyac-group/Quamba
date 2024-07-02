import torch
import torch.nn as nn

class QAct(nn.Module):
    def __init__(
        self,
        scale
    ):
        super().__init__()
        self.scale = scale
        
    @torch.no_grad()
    def forward(self, x):
        return (x / self.scale).clamp(min=-128, max=127).to(torch.int8) # quant
    
    def __repr__(self):
        return f"QAct()"


    
    
    