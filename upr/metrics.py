from .numerical import (
    compute_cosine_similarity,
    compute_kl_divergence,
    compute_numerical_metrics
)

def compute_weight_metrics(original, reconstructed):
    """
    Alias mapping to compute_numerical_metrics for backward compatibility.
    """
    m = compute_numerical_metrics(original, reconstructed)
    return {
        "exact_match": m["torch_equal"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "max_error": m["max_abs_error"],
        "mean_relative_error": m["mean_relative_error"],
        "cosine_similarity": m["cosine_similarity"],
        "kl_divergence": m["kl_divergence"],
        "num_elements": m["num_elements"]
    }
