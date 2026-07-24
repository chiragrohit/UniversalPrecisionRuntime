import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

def compute_cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """
    Fix 1 — Audit & Cosine Similarity Implementation.
    Uses torch.nn.functional.cosine_similarity on float32 CPU tensors.
    Enforces mathematical bounds assertion: -1.0 - 1e-6 <= cosine <= 1.0 + 1e-6.
    """
    flat1 = t1.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
    flat2 = t2.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
    
    norm1 = torch.norm(flat1)
    norm2 = torch.norm(flat2)
    
    if norm1 == 0 and norm2 == 0:
        return 1.0
    elif norm1 == 0 or norm2 == 0:
        return 0.0
        
    sim = float(F.cosine_similarity(flat1, flat2, dim=1).item())
    
    # Assertions (Fix 1)
    assert sim <= 1.0 + 1e-6, f"Cosine similarity exceeds mathematical upper bound: {sim}"
    assert sim >= -1.0 - 1e-6, f"Cosine similarity below mathematical lower bound: {sim}"
    
    # Numerical clamping to strictly [-1.0, 1.0]
    return max(-1.0, min(1.0, sim))

def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """
    Computes KL divergence KL(P || Q) over logit probability distributions.
    """
    p_f32 = p_logits.detach().to(device="cpu", dtype=torch.float32)
    q_f32 = q_logits.detach().to(device="cpu", dtype=torch.float32)
    
    p_log_prob = F.log_softmax(p_f32, dim=-1)
    q_log_prob = F.log_softmax(q_f32, dim=-1)
    
    kl = F.kl_div(q_log_prob, p_log_prob, log_target=True, reduction="batchmean")
    val = float(kl.item())
    return val if not (np.isnan(val) or np.isinf(val)) else 0.0

def compute_numerical_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, Any]:
    """
    Fix 2 — Numerical Validation Framework module (upr/numerical.py).
    Computes: MAE, RMSE, Max Absolute Error, Mean Relative Error, Cosine Similarity, KL Divergence.
    """
    orig_f32 = original.detach().to(device="cpu", dtype=torch.float32)
    recon_f32 = reconstructed.detach().to(device="cpu", dtype=torch.float32)
    
    is_exact = bool(torch.equal(original.detach().cpu(), reconstructed.detach().cpu()))
    diff = torch.abs(orig_f32 - recon_f32)
    
    mae = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((orig_f32 - recon_f32) ** 2)).item())
    max_abs_error = float(diff.max().item())
    
    denom = torch.abs(orig_f32) + 1e-8
    mre = float((diff / denom).mean().item())
    
    cos_sim = compute_cosine_similarity(orig_f32, recon_f32)
    
    kl_div = 0.0
    if orig_f32.ndim >= 2 and orig_f32.shape[-1] > 1:
        kl_div = compute_kl_divergence(orig_f32, recon_f32)
        
    return {
        "torch_equal": is_exact,
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "mean_relative_error": mre,
        "cosine_similarity": cos_sim,
        "kl_divergence": kl_div,
        "num_elements": int(orig_f32.numel())
    }
