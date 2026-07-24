import os
import json
import csv
import gc
import torch
from typing import Optional, Union, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

from .bit_ops import reconstruct_tensor
from .numerical import compute_numerical_metrics

class BitPlaneModel:
    """
    Universal Precision Runtime (UPR) Model Loader.
    Reconstructs execution models dynamically at requested precision (16, 14, 12, 10, 8, 6, 4, 2 bits)
    from a single BitPlane checkpoint.
    """

    @classmethod
    def load_reconstructed_state_dict(
        cls,
        bitplane_directory: str,
        bits: int = 16,
        device: Union[str, torch.device] = 'cpu',
        export_reconstruction_csv: bool = False,
        original_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        csv_output_path: str = "results/reconstruction.csv"
    ) -> Dict[str, torch.Tensor]:
        """
        Loads and reconstructs parameter state dict from a BitPlane directory for requested precision bits.
        Fix 5 - Stores results/reconstruction.csv with one row per tensor containing torch.equal(), MAE, RMSE, Max Error, Cosine.
        """
        assert 1 <= bits <= 16, f"bits must be between 1 and 16, got {bits}"

        metadata_path = os.path.join(bitplane_directory, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in '{bitplane_directory}'")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        reconstructed_state_dict = {}
        tensors_meta = metadata["tensors"]

        start_bit = 15
        end_bit = 16 - bits
        csv_rows = []

        for idx, (tensor_name, info) in enumerate(tqdm(tensors_meta.items(), desc=f"Reconstructing ({bits}-bit)")):
            original_shape = tuple(info["shape"])
            planes_dict = {}

            # Read only the selected MSB plane binary files
            for b in range(start_bit, end_bit - 1, -1):
                plane_rel_path = info["planes"][str(b)]
                plane_full_path = os.path.join(bitplane_directory, plane_rel_path)

                if os.path.exists(plane_full_path):
                    with open(plane_full_path, "rb") as pf:
                        planes_dict[b] = pf.read()

            recon_tensor = reconstruct_tensor(
                planes_dict=planes_dict,
                selected_bits=bits,
                original_shape=original_shape,
                device=device
            )
            del planes_dict  # Free byte buffers immediately

            if export_reconstruction_csv and original_state_dict and tensor_name in original_state_dict:
                orig_t = original_state_dict[tensor_name]
                m = compute_numerical_metrics(orig_t, recon_tensor)
                csv_rows.append({
                    "tensor_name": tensor_name,
                    "bits": bits,
                    "torch_equal": m["torch_equal"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "max_error": m["max_abs_error"],
                    "mean_relative_error": m["mean_relative_error"],
                    "cosine_similarity": m["cosine_similarity"],
                    "num_elements": m["num_elements"]
                })

            reconstructed_state_dict[tensor_name] = recon_tensor

            if idx % 50 == 0:
                gc.collect()

        if export_reconstruction_csv and csv_rows:
            os.makedirs(os.path.dirname(csv_output_path) if os.path.dirname(csv_output_path) else ".", exist_ok=True)
            file_exists = os.path.exists(csv_output_path)
            with open(csv_output_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(csv_rows[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(csv_rows)

        gc.collect()
        return reconstructed_state_dict

    @classmethod
    def from_pretrained(
        cls,
        bitplane_directory: str,
        bits: int = 16,
        base_model_id: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ) -> torch.nn.Module:
        """
        Loads a Hugging Face Causal LM model reconstructed from a BitPlane directory at specified precision.
        """
        metadata_path = os.path.join(bitplane_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model_name = base_model_id or metadata.get("model_name_or_path")
        print(f"Instantiating model base architecture '{model_name}' for precision bits={bits}...")

        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

        state_dict = cls.load_reconstructed_state_dict(
            bitplane_directory=bitplane_directory,
            bits=bits,
            device='cpu'
        )

        model.load_state_dict(state_dict, strict=True)
        del state_dict  # Free state dict memory immediately
        gc.collect()

        if device_map is not None:
            model = model.to(device_map)

        return model
