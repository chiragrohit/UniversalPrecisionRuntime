import os
import sys
import random
import subprocess
import time
import torch
import numpy as np
from typing import Dict, Any, Optional

def set_seed(seed: int = 42) -> None:
    """
    Fix 10 — Fixes Python, NumPy, PyTorch CPU/CUDA, and cuDNN random seeds for 100% deterministic evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"

def collect_experiment_metadata(
    precision_bits: int,
    dataset_name: str = "wikitext-2-raw-v1",
    seed: int = 42,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fix 11 — Automatically stores Git commit, timestamp, representation, precision, dataset,
    seed, model version, system/GPU information, PyTorch and Python versions.
    """
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit_hash(),
        "model_name": model_name,
        "representation": "bitplane_packed_uint16",
        "precision_bits": precision_bits,
        "dataset": dataset_name,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_device": gpu_info,
    }
    
    if extra_info:
        meta.update(extra_info)
        
    return meta
