import torch
import numpy as np
import gc
from typing import Tuple, Dict, Optional, Union

def float16_to_uint16_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Reinterprets float16 torch.Tensor as uint16 numpy ndarray (bit-exact).
    """
    np_f16 = tensor.detach().cpu().to(torch.float16).numpy()
    return np_f16.view(np.uint16)

def uint16_to_float16_torch(np_uint16: np.ndarray, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Reinterprets uint16 numpy ndarray as float16 torch.Tensor (bit-exact).
    """
    np_f16 = np_uint16.view(np.float16)
    tensor = torch.from_numpy(np_f16).to(device)
    # Sanitize NaNs and Infs for low-bit zero-filled exponents (e.g. 4-bit, 2-bit)
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return tensor

def extract_bit_plane_np(uint16_arr: np.ndarray, bit_index: int) -> np.ndarray:
    """
    Extracts a single bit (0 or 1) at bit_index (0..15) as uint8 numpy array.
    """
    assert 0 <= bit_index <= 15, f"bit_index must be between 0 and 15, got {bit_index}"
    return ((uint16_arr >> bit_index) & 1).astype(np.uint8)

def pack_bit_plane(bit_arr: np.ndarray) -> bytes:
    """
    Packs a 0/1 uint8 numpy array into packed uint8 bytes (8 bits per byte).
    """
    flat = bit_arr.ravel()
    packed = np.packbits(flat, bitorder='big')
    return packed.tobytes()

def unpack_bit_plane(packed_bytes: bytes, num_elements: int, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Unpacks packed uint8 bytes into a 0/1 uint8 numpy array with exact num_elements and shape.
    """
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8)
    unpacked = np.unpackbits(packed_np, bitorder='big')[:num_elements]
    if shape is not None:
        unpacked = unpacked.reshape(shape)
    return unpacked.astype(np.uint8)

def reconstruct_tensor(
    planes_dict: Dict[int, bytes],
    selected_bits: int,
    original_shape: Tuple[int, ...],
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Reconstructs a torch.float16 tensor from selected_bits Most Significant Bit planes.
    Frees temporary bit arrays immediately and sanitizes NaN/Inf numbers for extreme low-bit truncations.
    """
    assert 1 <= selected_bits <= 16, f"selected_bits must be between 1 and 16, got {selected_bits}"
    num_elements = int(np.prod(original_shape)) if len(original_shape) > 0 else 1

    accum = np.zeros(num_elements, dtype=np.uint32)

    start_bit = 15
    end_bit = 16 - selected_bits

    for b in range(start_bit, end_bit - 1, -1):
        if b in planes_dict:
            bits = unpack_bit_plane(planes_dict[b], num_elements)
            accum |= (bits.astype(np.uint32) << b)
            del bits  # Free unpacked bit array buffer immediately

    uint16_arr = accum.astype(np.uint16).reshape(original_shape)
    del accum  # Free uint32 accumulator immediately

    tensor = uint16_to_float16_torch(uint16_arr, device=device)
    del uint16_arr  # Free intermediate uint16 array
    return tensor
