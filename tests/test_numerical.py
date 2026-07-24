import torch
import pytest
from upr.numerical import (
    compute_cosine_similarity,
    compute_numerical_metrics,
    compute_kl_divergence
)

def test_cosine_similarity_bounds():
    # Identical tensors -> 1.0
    t1 = torch.randn(10, 10, dtype=torch.float32)
    assert abs(compute_cosine_similarity(t1, t1) - 1.0) < 1e-5

    # Orthogonal vectors -> 0.0
    t2 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    t3 = torch.tensor([0.0, 1.0], dtype=torch.float32)
    assert abs(compute_cosine_similarity(t2, t3)) < 1e-5

    # Opposite vectors -> -1.0
    assert abs(compute_cosine_similarity(t2, -t2) - (-1.0)) < 1e-5

def test_cosine_similarity_assertion_bound():
    t1 = torch.ones(5, 5, dtype=torch.float32)
    cos = compute_cosine_similarity(t1, t1)
    assert -1.0 - 1e-6 <= cos <= 1.0 + 1e-6

def test_numerical_metrics():
    t1 = torch.ones(5, 5, dtype=torch.float16)
    t2 = torch.ones(5, 5, dtype=torch.float16)
    
    res = compute_numerical_metrics(t1, t2)
    assert res["torch_equal"] is True
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["cosine_similarity"] == 1.0

def test_kl_divergence():
    p = torch.randn(1, 10, dtype=torch.float32)
    kl_self = compute_kl_divergence(p, p)
    assert abs(kl_self) < 1e-4
