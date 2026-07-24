"""
Universal Precision Runtime (UPR)
A single bit-plane runtime for dynamic precision reconstruction.
"""

from .bit_ops import (
    float16_to_uint16_numpy,
    uint16_to_float16_torch,
    extract_bit_plane_np,
    pack_bit_plane,
    unpack_bit_plane,
    reconstruct_tensor,
)
from .converter import convert_to_bitplanes
from .loader import BitPlaneModel
from .metrics import compute_weight_metrics

__version__ = "0.1.0"
__all__ = [
    "float16_to_uint16_numpy",
    "uint16_to_float16_torch",
    "extract_bit_plane_np",
    "pack_bit_plane",
    "unpack_bit_plane",
    "reconstruct_tensor",
    "convert_to_bitplanes",
    "BitPlaneModel",
    "compute_weight_metrics",
]
