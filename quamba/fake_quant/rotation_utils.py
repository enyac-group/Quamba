import torch
import torch.nn as nn
import math
from fast_hadamard_transform import hadamard_transform
from quamba.fake_quant.hadamard_utils import matmul_hadU_cuda, get_hadK

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')
    
    
class HadamardTransform(nn.Module):
    def __init__(self, do_rotate=False):
        super().__init__()
        self.do_rotate = do_rotate
    def forward(self, x, transpose=False):
       
        if self.do_rotate:
            dtype = x.dtype
            n = x.shape[-1]
            had_K, K = get_hadK(n)
            x = matmul_hadU_cuda(x.contiguous(), had_K, K, transpose=transpose).to(dtype)
        return x
    
    def configure(self, do_rotate):
        self.do_rotate = do_rotate
            
    
    