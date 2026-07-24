from .bit_ops import (
    float16_to_uint16_numpy,
    uint16_to_float16_torch,
    extract_bit_plane_np,
    pack_bit_plane,
    unpack_bit_plane,
    reconstruct_tensor,
    get_packing_stats,
)
from .converter import convert_to_bitplanes
from .loader import BitPlaneModel
from .numerical import (
    compute_cosine_similarity,
    compute_kl_divergence,
    compute_numerical_metrics,
)
from .metrics import compute_weight_metrics
from .metadata import set_seed, collect_experiment_metadata, get_git_commit_hash
from .profiler import IsolatedTimer, MemoryProfiler
from .layer_hooks import LayerActivationCollector, compare_layer_activations

__version__ = "0.1.0"
__all__ = [
    "float16_to_uint16_numpy",
    "uint16_to_float16_torch",
    "extract_bit_plane_np",
    "pack_bit_plane",
    "unpack_bit_plane",
    "reconstruct_tensor",
    "get_packing_stats",
    "convert_to_bitplanes",
    "BitPlaneModel",
    "compute_weight_metrics",
    "compute_cosine_similarity",
    "compute_kl_divergence",
    "compute_numerical_metrics",
    "set_seed",
    "collect_experiment_metadata",
    "get_git_commit_hash",
    "IsolatedTimer",
    "MemoryProfiler",
    "LayerActivationCollector",
    "compare_layer_activations",
]
