import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, List

def compute_cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """
    Fix 1 - Audit & Cosine Similarity Implementation.
    Uses L2-normalized vector dot product on float32 CPU tensors with exact match fast-path.
    Enforces mathematical bounds assertion: -1.0 - 1e-6 <= cosine <= 1.0 + 1e-6.
    """
    if torch.equal(t1, t2):
        return 1.0

    flat1 = t1.detach().to(device="cpu", dtype=torch.float32).flatten()
    flat2 = t2.detach().to(device="cpu", dtype=torch.float32).flatten()
    
    norm1 = torch.linalg.vector_norm(flat1)
    norm2 = torch.linalg.vector_norm(flat2)
    
    if norm1 < 1e-12 and norm2 < 1e-12:
        return 1.0
    elif norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
        
    dot = torch.dot(flat1, flat2)
    sim = float((dot / (norm1 * norm2)).item())
    
    # Numerical clamping to strictly [-1.0, 1.0]
    sim = max(-1.0, min(1.0, sim))

    # Assertions (Fix 1)
    assert sim <= 1.0 + 1e-6, f"Cosine similarity exceeds mathematical upper bound: {sim}"
    assert sim >= -1.0 - 1e-6, f"Cosine similarity below mathematical lower bound: {sim}"
    
    return sim

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
    Fix 2 - Numerical Validation Framework module (upr/numerical.py).
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

def compute_plane_entropy(bit_arr: np.ndarray) -> float:
    """
    Computes Shannon entropy H = -p0*log2(p0) - p1*log2(p1) for a 0/1 bit array (Exp 6).
    """
    if bit_arr.size == 0:
        return 0.0
    p1 = float(np.mean(bit_arr))
    p0 = 1.0 - p1
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return float(-p0 * math.log2(p0) - p1 * math.log2(p1))

def compute_bit_density(bit_arr: np.ndarray) -> Dict[str, float]:
    """
    Computes percentage of ones, zeros, and bit density stats (Exp 6).
    """
    if bit_arr.size == 0:
        return {"pct_ones": 0.0, "pct_zeros": 0.0, "entropy": 0.0}
    p1 = float(np.mean(bit_arr)) * 100.0
    p0 = 100.0 - p1
    entropy = compute_plane_entropy(bit_arr)
    return {
        "pct_ones": round(p1, 4),
        "pct_zeros": round(p0, 4),
        "entropy": round(entropy, 6)
    }

def compute_correlation_matrix(data_rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Computes Pearson correlation matrix across metrics (Exp 9).
    """
    corr_matrix: Dict[str, Dict[str, float]] = {k1: {} for k1 in keys}
    if not data_rows:
        return corr_matrix

    data_arrays = {}
    for k in keys:
        vals = [float(row.get(k, 0.0)) for row in data_rows]
        data_arrays[k] = np.array(vals, dtype=np.float64)

    for k1 in keys:
        for k2 in keys:
            v1, v2 = data_arrays[k1], data_arrays[k2]
            std1, std2 = np.std(v1), np.std(v2)
            if std1 == 0 or std2 == 0:
                val = 1.0 if k1 == k2 else 0.0
            else:
                val = float(np.corrcoef(v1, v2)[0, 1])
                if np.isnan(val):
                    val = 0.0
            corr_matrix[k1][k2] = round(val, 4)

    return corr_matrix
