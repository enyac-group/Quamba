__version__ = "2.0.0a1"

from .modelutils_mamba import quantize_model_mamba
from .qMambaLayer import W4A8QMamba, W4A16QMamba, W8A8QMamba, MambaSimple
from .qMamba2 import W4A8QMamba2, W4A16QMamba2, W8A8QMamba2, Mamba2Simple
from .qActLayer import QAct
from .qHadamard import QHadamard, Hadamard
from .fusedNorm import FusedRMSNorm
from .qNorm import QRMSNorm, QRMSNormGated
from .qConvLayer import QCausalConv1D, Quamb2Conv1D
from .qLinearLayer import W4A16B16O16Linear
from .qLinearLayer import W4A8B8O8Linear, W4A8B8O8LinearParallel, W4A8B16O16Linear
from .qLinearLayer import W8A8B8O8Linear, W8A8B8O8LinearParallel, W8A8B16O16Linear
from .qLinearLayer import HadLinear
from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver
from .qSelectiveScan import QSScan
from .qChunkScan import Quamba2ChunkScan
from .qEmbedding import W8O16Embedding, W4O16Embedding
from .observer import PerTensorMinmaxObserver
from .quant_utils import quantize_tensor_per_tensor_absmax
from .triton.selective_state_update import quamba2_sscan_update_triton, quant_sscan_update_triton
from mamba_ssm.ops.triton.layer_norm import RMSNorm