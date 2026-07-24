# Universal Precision Runtime (UPR)

> **A single FP16 model checkpoint → multiple execution precisions (16, 14, 12, 10, 8, 6, 4, 2 bits) reconstructed at runtime — without storing separate quantized models.**

---

## What is UPR?

Current inference systems require separate checkpoints for each precision:

```
FP16 Model  →  stored separately
INT8 Model  →  stored separately
INT4 Model  →  stored separately
AWQ Model   →  stored separately
```

UPR proposes a single **BitPlane representation**: every FP16 weight tensor is decomposed into 16 bit-planes (`plane15.bin` → `plane0.bin`). At runtime, only the top-N bit-planes are loaded and reconstructed, producing a model at any desired precision — from a single checkpoint.

```
BitPlane Checkpoint (one, stored once)
         │
         ├── plane15 (MSB)
         ├── plane14
         ├── ...
         └── plane0  (LSB)
                │
                ▼
     Reconstruct at runtime:
     16-bit  = all 16 planes  (exact FP16)
     8-bit   = top 8 planes
     4-bit   = top 4 planes
     2-bit   = top 2 planes
```

---

## Project Status

**Phase 1.1 — Evaluation Audit & Infrastructure Fixes** *(current)*

The goal of this phase is to make every reported metric **trustworthy and reproducible** before any further representation research begins.

| Fix | Description | Status |
|-----|-------------|--------|
| Fix 1 | Cosine similarity bug — bounds assertion + verified F.cosine_similarity | ✅ |
| Fix 2 | Numerical validation framework (MAE, RMSE, KL, MRE) | ✅ |
| Fix 3 | Layer-wise activation comparison via forward hooks | ✅ |
| Fix 5 | Per-tensor reconstruction CSV export | ✅ |
| Fix 6 | Isolated timing per phase (reconstruction / init / forward / generation) | ✅ |
| Fix 7 | Memory profiler — CPU RAM, GPU VRAM, checkpoint size to `results/memory.csv` | ✅ |
| Fix 8 | Bit-packing assertion: 1 bit per element, never 1 byte per element | ✅ |
| Fix 9 | Deterministic WikiText-2 evaluation (fixed prompt, tokenizer, generation params) | ✅ |
| Fix 10 | `set_seed(42)` before every precision sweep step | ✅ |
| Fix 11/12 | Unified JSON schema with dataset, seed, git commit, GPU, metadata fields | ✅ |

---

## Repository Structure

```
UniversalPrecisionRuntime/
│
├── notebooks/
│   ├── 00_bootstrap_upr_files.ipynb        ← Run FIRST (writes upr/ to Drive)
│   ├── 01_bitplane_conversion_and_level1.ipynb
│   ├── 02_layer_outputs_and_logits_level2_3.ipynb
│   ├── 03_variable_precision_and_perplexity_level4_5.ipynb
│   └── 04_benchmarks_plots_and_comparison_level6.ipynb
│
├── upr/                                    ← Python package
│   ├── __init__.py
│   ├── bit_ops.py                          ← Bit-plane encode/decode
│   ├── converter.py                        ← FP16 → BitPlane checkpoint
│   ├── loader.py                           ← BitPlane → state dict (any precision)
│   ├── numerical.py                        ← Fix 1/2: CosSim, KL, MAE, RMSE
│   ├── metadata.py                         ← Fix 10/11: set_seed, experiment metadata
│   ├── profiler.py                         ← Fix 6/7: IsolatedTimer, MemoryProfiler
│   ├── layer_hooks.py                      ← Fix 3: LayerActivationCollector
│   └── metrics.py                          ← compute_weight_metrics (backward compat)
│
├── tests/
│   ├── test_numerical.py
│   ├── test_packing.py
│   └── test_validator.py
│
├── validate_results.py                     ← CLI schema + cosine bounds validator
├── idea.md                                 ← Full prototype specification
├── steps.md                                ← Phase plan
├── pyproject.toml
└── setup.py
```

---

## Running on Google Colab (VS Code Extension)

This project runs on **Google Colab GPU** via the VS Code Colab extension. All notebooks are opened locally in VS Code but executed on the remote Colab kernel, which reads `upr/` from Google Drive.

### Step 0 — Set up HF Token (once)

1. Open your notebook in a **browser** at [colab.research.google.com](https://colab.research.google.com)
2. Click **🔑 Secrets** in the left sidebar → **Add new secret**
3. Name: `HF_TOKEN`, Value: your HuggingFace token
4. Toggle **Notebook access** ON

The notebooks automatically read it via:
```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
```

### Step 1 — Bootstrap (once per session)

Run `00_bootstrap_upr_files.ipynb` **before anything else**.

This writes all 7 updated `upr/` Python files directly to your Google Drive project directory:

```
Bootstrap writes:                   Google Drive
─────────────────                   ─────────────────────────────────────────
upr/__init__.py    ──────────────►  /content/drive/MyDrive/
upr/bit_ops.py                      UniversalPrecisionRuntime/upr/
upr/numerical.py
upr/metadata.py
upr/profiler.py
upr/layer_hooks.py
upr/metrics.py
```

**Why is this needed?** Your local `C:\Projects\UniversalPrecisionRuntime` and Colab's Google Drive are separate filesystems. The bootstrap bridges that gap without requiring a `git push`.

**When to re-run bootstrap:**
- After a Colab session reset (runtime disconnects)
- After updating `upr/` source files locally

### Step 2 — Run Notebooks in Order

```
00_bootstrap_upr_files.ipynb              ← writes upr/ to Drive (once)
        │
        ▼
01_bitplane_conversion_and_level1.ipynb   ← converts Qwen3.5-0.8B → BitPlane checkpoint
        │                                    saves: models/bitplane_qwen/
        ▼
02_layer_outputs_and_logits_level2_3.ipynb ← layer hook verification + logit comparison
        │
        ▼
03_variable_precision_and_perplexity_level4_5.ipynb ← 16→2 bit sweep
        │                                              saves: results/*.json
        │                                                     results/memory.csv
        │                                                     results/reconstruction.csv
        ▼
04_benchmarks_plots_and_comparison_level6.ipynb ← 8 plots + comparison table
                                                   saves: results/plots/*.png
                                                          results/upr_evaluation_report.json
```

---

## What Each Notebook Does

### `00_bootstrap_upr_files.ipynb`
Writes all `upr/` module files to Google Drive. Run once per session before any other notebook.

### `01_bitplane_conversion_and_level1.ipynb`
- Downloads `Qwen/Qwen3.5-0.8B` in FP16
- Converts all weight tensors to packed bit-plane format (16 binary files per tensor)
- Validates **100% exact bitwise reconstruction** at 16-bit (`torch.equal == True` for all tensors)
- Exports per-tensor metrics to `results/reconstruction.csv`

### `02_layer_outputs_and_logits_level2_3.ipynb`
- Attaches forward hooks to compare layer activations between original and reconstructed models
- Computes logit cosine similarity (Fix 1 bounds assertion applied)
- Validates KL divergence and top-1 token accuracy at 16-bit

### `03_variable_precision_and_perplexity_level4_5.ipynb`
- Sets `upr.set_seed(42)` before every precision level (Fix 10)
- Sweeps 16 → 14 → 12 → 10 → 8 → 6 → 4 → 2 bit reconstruction
- Records isolated timing per phase: reconstruction / model init / forward / generation (Fix 6)
- Snapshots CPU RAM + GPU VRAM to `results/memory.csv` (Fix 7)
- Evaluates WikiText-2 perplexity with identical tokenizer/prompt/generation settings per precision (Fix 9)
- Saves unified JSON schema per precision with metadata (Fix 11/12)

### `04_benchmarks_plots_and_comparison_level6.ipynb`
- Loads `results/variable_precision_summary.json`
- Validates schema and cosine bounds on every result (Fix 1/2)
- Generates 8 publication-ready plots to `results/plots/`
- Builds competitive comparison table (UPR vs FP16 / AWQ / GPTQ published numbers)
- Exports `results/upr_evaluation_report.json` (full evaluation record)

---

## Results

All results are saved under `results/` on Google Drive:

| File | Contents |
|------|----------|
| `results/reconstruction.csv` | Per-tensor: MAE, RMSE, cosine sim, exact match |
| `results/memory.csv` | Per-precision: CPU RAM, GPU VRAM, checkpoint size |
| `results/{N}bit.json` | Full metrics for each precision level |
| `results/variable_precision_summary.json` | All 8 precisions in one file |
| `results/upr_evaluation_report.json` | Final report with comparison table |
| `results/plots/01_ppl_vs_bits.png` | Perplexity vs precision |
| `results/plots/02_cosine_vs_bits.png` | Logit cosine similarity vs precision |
| `results/plots/03_top1_acc_vs_bits.png` | Token accuracy vs precision |
| `results/plots/04_recon_time_vs_bits.png` | Reconstruction time vs precision |
| `results/plots/05_ppl_delta_vs_bits.png` | PPL degradation vs FP16 baseline |
| `results/plots/06_quality_cliff_map.png` | Heatmap of all metrics |
| `results/plots/07_cos_vs_ppl_scatter.png` | Quality vs PPL frontier |
| `results/plots/08_time_vs_ppl.png` | Speed vs quality tradeoff |

---

## `upr` Package API

```python
import upr

# Seed (Fix 10)
upr.set_seed(42)

# Convert FP16 model to BitPlane checkpoint
upr.convert_to_bitplanes(model_or_path="Qwen/Qwen3.5-0.8B", output_directory="models/bp")

# Reconstruct state dict at any precision
state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
    bitplane_directory="models/bp",
    bits=8,                          # 16, 14, 12, 10, 8, 6, 4, or 2
    export_reconstruction_csv=True,
    original_state_dict=orig_sd,
    csv_output_path="results/reconstruction.csv"
)

# Fix 1 — Verified cosine similarity (bounds asserted)
cos = upr.compute_cosine_similarity(tensor_a, tensor_b)

# Fix 2 — Full numerical metrics
metrics = upr.compute_numerical_metrics(original_tensor, reconstructed_tensor)
# Returns: mae, rmse, max_abs_error, mean_relative_error, cosine_similarity, kl_divergence

# Fix 6 — Isolated timing
timer = upr.IsolatedTimer()
timer.start("reconstruction")
# ... do work ...
elapsed = timer.stop("reconstruction")

# Fix 7 — Memory profiling
profiler = upr.MemoryProfiler()
profiler.record_memory_snapshot(precision_bits=8, checkpoint_dir="models/bp", output_csv_path="results/memory.csv")

# Fix 3 — Layer activation comparison
collector = upr.LayerActivationCollector(model)
collector.register_hooks()
# ... run forward pass ...
metrics = upr.compare_layer_activations(orig_collector, recon_collector)

# Fix 11 — Experiment metadata
meta = upr.collect_experiment_metadata(precision_bits=8)
# Returns: timestamp, git_commit, model_name, seed, pytorch_version, gpu_device, ...
```

---

## Running Tests Locally

```bash
python -m pytest tests/ -v
```

Or validate saved results:

```bash
python validate_results.py results/
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Bit-planes packed with `np.packbits` | 1 bit per element (not 1 byte) — Fix 8 asserts this |
| All `cosine_similarity` via `F.cosine_similarity` on float32 | Eliminates FP16 overflow causing values > 1.0 — Fix 1 |
| WikiText-2 with fixed 512-token windows | Reproducible PPL — Fix 9 |
| `set_seed(42)` before every precision level | Removes RNG variance from timing/accuracy comparisons — Fix 10 |
| `IsolatedTimer` with CUDA sync | Prevents GPU async from contaminating timing measurements — Fix 6 |
| NaN/Inf → clamped in `uint16_to_float16_torch` | 4-bit and 2-bit zero-fill exponent bits, producing invalid FP16 patterns |

---

## License

Research prototype — not for production use.
