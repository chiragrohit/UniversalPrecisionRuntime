import os
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional

from .bit_ops import unpack_bit_plane
from .numerical import compute_bit_density, compute_plane_entropy

def categorize_tensor(tensor_name: str) -> str:
    """
    Categorizes a model tensor name into one of 10 standard categories for Phase 1.2 Exp 4:
    Embedding, LM Head, Attn Q, Attn K, Attn V, Attn O, MLP Up, MLP Down, MLP Gate, Other.
    """
    name_lower = tensor_name.lower()
    if "embed" in name_lower or "wte" in name_lower or "wpe" in name_lower:
        return "Embedding"
    elif "lm_head" in name_lower or "output.weight" in name_lower:
        return "LM Head"
    elif "q_proj" in name_lower or "query" in name_lower:
        return "Attn Q"
    elif "k_proj" in name_lower or "key" in name_lower:
        return "Attn K"
    elif "v_proj" in name_lower or "value" in name_lower:
        return "Attn V"
    elif "o_proj" in name_lower or "out_proj" in name_lower or "dense" in name_lower and "attn" in name_lower:
        return "Attn O"
    elif "gate_proj" in name_lower or "w1" in name_lower or "gate" in name_lower:
        return "MLP Gate"
    elif "up_proj" in name_lower or "w3" in name_lower or "up" in name_lower:
        return "MLP Up"
    elif "down_proj" in name_lower or "w2" in name_lower or "down" in name_lower:
        return "MLP Down"
    elif "norm" in name_lower or "ln" in name_lower:
        return "LayerNorm"
    else:
        return "Other"

def analyze_representation_stats(bitplane_directory: str) -> List[Dict[str, Any]]:
    """
    Computes per-plane representation statistics (% ones, % zeros, entropy, compression ratio, bit density)
    across all parameter tensors in the checkpoint (Phase 1.2 Exp 6).
    """
    metadata_path = os.path.join(bitplane_directory, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in '{bitplane_directory}'")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tensors_meta = metadata["tensors"]
    plane_stats = {b: {"ones_count": 0, "total_bits": 0, "entropies": []} for b in range(16)}

    for idx, (tensor_name, info) in enumerate(tensors_meta.items()):
        numel = int(info["numel"])
        for b in range(16):
            plane_rel_path = info["planes"][str(b)]
            plane_full_path = os.path.join(bitplane_directory, plane_rel_path)
            if os.path.exists(plane_full_path):
                with open(plane_full_path, "rb") as pf:
                    packed_bytes = pf.read()
                unpacked = unpack_bit_plane(packed_bytes, numel)
                ones = int(np.sum(unpacked))
                plane_stats[b]["ones_count"] += ones
                plane_stats[b]["total_bits"] += numel
                plane_stats[b]["entropies"].append(compute_plane_entropy(unpacked))

    results = []
    total_fp16_bytes = metadata.get("num_tensors", len(tensors_meta)) * 2
    for b in range(15, -1, -1):
        st = plane_stats[b]
        tot = st["total_bits"]
        if tot > 0:
            pct_ones = (st["ones_count"] / tot) * 100.0
            pct_zeros = 100.0 - pct_ones
            avg_entropy = float(np.mean(st["entropies"]))
            packed_bytes = (tot + 7) // 8
            comp_ratio = (tot * 2) / packed_bytes if packed_bytes > 0 else 0.0
        else:
            pct_ones, pct_zeros, avg_entropy, comp_ratio = 0.0, 0.0, 0.0, 0.0

        results.append({
            "plane_index": b,
            "plane_name": f"Plane {b}" + (" (MSB)" if b == 15 else " (LSB)" if b == 0 else ""),
            "pct_ones": round(pct_ones, 4),
            "pct_zeros": round(pct_zeros, 4),
            "entropy": round(avg_entropy, 6),
            "compression_ratio": round(comp_ratio, 4)
        })

    return results
