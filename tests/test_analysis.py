import pytest
import torch
import numpy as np
import upr

def test_categorize_tensor():
    assert upr.categorize_tensor("model.embed_tokens.weight") == "Embedding"
    assert upr.categorize_tensor("lm_head.weight") == "LM Head"
    assert upr.categorize_tensor("model.layers.0.self_attn.q_proj.weight") == "Attn Q"
    assert upr.categorize_tensor("model.layers.0.self_attn.k_proj.weight") == "Attn K"
    assert upr.categorize_tensor("model.layers.0.self_attn.v_proj.weight") == "Attn V"
    assert upr.categorize_tensor("model.layers.0.self_attn.o_proj.weight") == "Attn O"
    assert upr.categorize_tensor("model.layers.0.mlp.gate_proj.weight") == "MLP Gate"
    assert upr.categorize_tensor("model.layers.0.mlp.up_proj.weight") == "MLP Up"
    assert upr.categorize_tensor("model.layers.0.mlp.down_proj.weight") == "MLP Down"

def test_entropy_and_density():
    arr = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    entropy = upr.compute_plane_entropy(arr)
    assert abs(entropy - 1.0) < 1e-5

    density = upr.compute_bit_density(arr)
    assert density["pct_ones"] == 50.0
    assert density["pct_zeros"] == 50.0
    assert abs(density["entropy"] - 1.0) < 1e-5

def test_correlation_matrix():
    rows = [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 4.0},
        {"x": 3.0, "y": 6.0},
    ]
    corr = upr.compute_correlation_matrix(rows, ["x", "y"])
    assert corr["x"]["y"] == 1.0
    assert corr["y"]["x"] == 1.0
    assert corr["x"]["x"] == 1.0
