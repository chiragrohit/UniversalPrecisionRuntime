import os
import tarfile
import modal

# ---------------------------------------------------------------------------
# Modal App, Volume & Images
# ---------------------------------------------------------------------------
app = modal.App("upr-pipeline")
vol = modal.Volume.from_name("upr-data-vol", create_if_missing=True)

# Base image for all stages
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "matplotlib>=3.7.0",
        "numpy>=1.22.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.19.0",
        "optimum",
        "bitsandbytes>=0.41.0",
        "compressed-tensors>=0.15.0",
        "auto-round",
        "auto-gptq",
        "autoawq",
    )
    .add_local_dir("upr", remote_path="/root/upr")
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_ID     = "Qwen/Qwen3.5-0.8B"
VOL_MOUNT    = "/vol"
BITPLANE_DIR = f"{VOL_MOUNT}/models/bitplane_qwen"
TAR_PATH     = f"{VOL_MOUNT}/models/bitplane_qwen.tar"
TMP_DIR      = "/tmp/bitplane_qwen"
RESULTS_DIR  = f"{VOL_MOUNT}/results"
PLOTS_DIR    = f"{RESULTS_DIR}/plots"
BASELINES_JSON = f"{RESULTS_DIR}/baselines.json"

# ---------------------------------------------------------------------------
# Shared helpers (all run INSIDE Modal containers)
# ---------------------------------------------------------------------------
def setup_hf_auth():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception as e:
            print(f"HF Login warning: {e}")


def get_fast_checkpoint_dir() -> str:
    """Extract single tar from Modal Volume to fast local NVMe (/tmp)."""
    meta_tmp = os.path.join(TMP_DIR, "metadata.json")
    if os.path.exists(meta_tmp):
        return TMP_DIR

    os.makedirs(TMP_DIR, exist_ok=True)

    if os.path.exists(TAR_PATH):
        print(f"Extracting tar: {TAR_PATH} -> {TMP_DIR} ...")
        with tarfile.open(TAR_PATH, "r:") as tar:
            tar.extractall(path=TMP_DIR)
        print("[OK] Extraction complete!")
    elif os.path.exists(os.path.join(BITPLANE_DIR, "metadata.json")):
        print("Creating tar from folder ...")
        os.makedirs(os.path.dirname(TAR_PATH), exist_ok=True)
        with tarfile.open(TAR_PATH, "w:") as tar:
            tar.add(BITPLANE_DIR, arcname="")
        vol.commit()
        with tarfile.open(TAR_PATH, "r:") as tar:
            tar.extractall(path=TMP_DIR)

    return TMP_DIR if os.path.exists(meta_tmp) else BITPLANE_DIR


def archive_tmp_to_vol():
    """Pack /tmp/bitplane_qwen into a single tar on Modal Volume."""
    meta_tmp = os.path.join(TMP_DIR, "metadata.json")
    if os.path.exists(meta_tmp):
        print(f"Packing {TMP_DIR} -> {TAR_PATH} ...")
        os.makedirs(os.path.dirname(TAR_PATH), exist_ok=True)
        with tarfile.open(TAR_PATH, "w:") as tar:
            tar.add(TMP_DIR, arcname="")
        vol.commit()
        print("[OK] Checkpoint archived to Modal Volume!")


def evaluate_perplexity(model, input_ids, seq_len: int = 512) -> float:
    """Standard WikiText-2 perplexity evaluation."""
    import torch
    model.eval()
    nlls = []
    total_len = input_ids.size(1)
    end_loc = 0
    for i in range(0, total_len, seq_len):
        end_loc = min(i + seq_len, total_len)
        if end_loc - i < 64:
            continue
        chunk_ids = input_ids[:, i:end_loc]
        with torch.no_grad():
            try:
                loss = model(chunk_ids, labels=chunk_ids).loss
                if torch.isnan(loss) or torch.isinf(loss):
                    return 9999.0
                nlls.append(loss * (end_loc - i))
            except Exception:
                return 9999.0
    if not nlls or end_loc == 0:
        return 9999.0
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    val = float(ppl.item())
    return val if not (torch.isnan(ppl) or torch.isinf(ppl)) else 9999.0
