import os
import json
import torch
import numpy as np
from typing import Union, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .bit_ops import float16_to_uint16_numpy, extract_bit_plane_np, pack_bit_plane

def convert_to_bitplanes(
    model_or_path: Union[str, torch.nn.Module],
    output_directory: str,
    torch_dtype: torch.dtype = torch.float16
) -> str:
    """
    Converts a Hugging Face FP16 model checkpoint sequentially into a BitPlane checkpoint.
    Creates packed binary bit-plane files (plane15.bin .. plane0.bin) and metadata.json.
    """
    os.makedirs(output_directory, exist_ok=True)
    tensors_dir = os.path.join(output_directory, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    if isinstance(model_or_path, str):
        model_name = model_or_path
        print(f"Loading Hugging Face model from: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
    else:
        model_name = getattr(model_or_path, "name_or_path", "custom_model")
        model = model_or_path

    state_dict = model.state_dict()
    metadata = {
        "model_name_or_path": model_name,
        "num_tensors": len(state_dict),
        "tensors": {}
    }

    print(f"Converting {len(state_dict)} tensors to bit-plane format in '{output_directory}'...")

    for idx, (tensor_name, tensor) in enumerate(tqdm(state_dict.items(), desc="BitPlane Conversion")):
        tensor_folder_name = f"tensor_{idx}"
        tensor_folder_path = os.path.join(tensors_dir, tensor_folder_name)
        os.makedirs(tensor_folder_path, exist_ok=True)

        original_shape = list(tensor.shape)
        dtype_str = str(tensor.dtype).replace("torch.", "")

        # Convert float16 to uint16 numpy array
        uint16_arr = float16_to_uint16_numpy(tensor)

        planes_meta = {}
        # Save all 16 bit-planes (bit 15 to bit 0)
        for bit_idx in range(16):
            plane_filename = f"plane{bit_idx}.bin"
            plane_path = os.path.join(tensor_folder_path, plane_filename)

            bit_arr = extract_bit_plane_np(uint16_arr, bit_idx)
            packed_bytes = pack_bit_plane(bit_arr)

            with open(plane_path, "wb") as f:
                f.write(packed_bytes)

            planes_meta[str(bit_idx)] = f"tensors/{tensor_folder_name}/{plane_filename}"

        metadata["tensors"][tensor_name] = {
            "shape": original_shape,
            "dtype": dtype_str,
            "numel": int(tensor.numel()),
            "folder": f"tensors/{tensor_folder_name}",
            "planes": planes_meta
        }

    metadata_path = os.path.join(output_directory, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully converted model to BitPlane format at: {output_directory}")
    return output_directory
