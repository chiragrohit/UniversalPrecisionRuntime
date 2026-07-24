import torch
import numpy as np
from typing import Dict, Any

def compute_weight_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, Any]:
    """
    Computes Level 1 numerical reconstruction error metrics between original and reconstructed tensors.
    """
    # Ensure both tensors are float32 on CPU for high precision error calculations
    orig_f32 = original.detach().cpu().to(torch.float32)
    recon_f32 = reconstructed.detach().cpu().to(torch.float32)

    # 1. Exact bitwise equality
    is_exact = bool(torch.equal(original.detach().cpu(), reconstructed.detach().cpu()))

    # 2. Difference tensor
    diff = torch.abs(orig_f32 - recon_f32)
    mae = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((orig_f32 - recon_f32) ** 2)).item())
    max_error = float(diff.max().item())

    # 3. Cosine similarity
    orig_flat = orig_f32.view(-1)
    recon_flat = recon_f32.view(-1)

    norm_orig = torch.norm(orig_flat)
    norm_recon = torch.norm(recon_flat)

    if norm_orig == 0 or norm_recon == 0:
        cos_sim = 1.0 if norm_orig == norm_recon else 0.0
    else:
        cos_sim = float((torch.dot(orig_flat, recon_flat) / (norm_orig * norm_recon)).item())

    return {
        "exact_match": is_exact,
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "cosine_similarity": cos_sim,
        "num_elements": int(orig_f32.numel())
    }
