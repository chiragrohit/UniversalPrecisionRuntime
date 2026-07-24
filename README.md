# Universal Precision Runtime (UPR)

> **A single FP16 model checkpoint → multiple execution precisions (16, 14, 12, 10, 8, 6, 4, 2 bits) reconstructed at runtime — without storing separate quantized models.**

---

## What is UPR?

Current LLM inference systems require separate checkpoints for each precision level:

```
FP16 Model  →  stored separately (~1.4 GB)
INT8 Model  →  stored separately (~0.7 GB)
INT4 Model  →  stored separately (~0.35 GB)
AWQ Model   →  stored separately (~0.35 GB)
```

UPR proposes a single **BitPlane representation**: every FP16 weight tensor is decomposed into 16 bit-planes (`plane15.bin` → `plane0.bin`). At runtime, only the top-N bit-planes are loaded and reconstructed, producing a model at any desired precision — from a single checkpoint.

```
BitPlane Checkpoint (single .tar archive stored once on Modal Volume)
         │
         ├── plane15 (MSB)
         ├── plane14
         ├── ...
         └── plane0  (LSB)
                │
                ▼
     Reconstruct at runtime:
     16-bit  = all 16 planes  (exact FP16 match)
     8-bit   = top 8 planes   (50% storage)
     4-bit   = top 4 planes   (25% storage)
     2-bit   = top 2 planes   (12.5% storage)
```

---

## The 4 Pipeline Stages (`modal_app.py`)

The pipeline runs remotely on **Modal Labs** using an NVIDIA T4 GPU:

```
                  ┌───────────────────────────────────────────────────────────┐
                  │ 1. modal run modal_app.py                                  │
                  └───────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: BitPlane Conversion & 16-Bit Verification                                            │
│ • Downloads Qwen3.5-0.8B FP16 baseline model.                                                 │
│ • Decomposes all 321 weight tensors into 16 binary bit-planes on fast local NVMe (/tmp).      │
│ • Archives to a single bitplane_qwen.tar file on Modal Volume for fast persistence.           │
│ • Reconstructs all 16 bits and verifies 100% exact bitwise match (torch.equal == True).       │
│ • Exports per-tensor reconstruction metrics to results/reconstruction.csv.                    │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Layer Activation Hooks & Logit Verification                                          │
│ • Attaches forward hooks (LayerActivationCollector) to all Transformer blocks.                │
│ • Runs prompt through original FP16 model and reconstructed 16-bit BitPlane model.            │
│ • Verifies intermediate layer activations and logit similarity (Cosine Sim >= 0.9999).       │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Variable Precision Sweep & WikiText Perplexity                                       │
│ • Sets deterministic seed (upr.set_seed(42)) before every step.                               │
│ • Sweeps precision levels: 16 → 14 → 12 → 10 → 8 → 6 → 4 → 2 bits.                           │
│ • Measures isolated phase timings (IsolatedTimer) and memory stats (MemoryProfiler).          │
│ • Evaluates WikiText-2 language model perplexity (PPL) per precision level.                   │
│ • Saves results/{N}bit.json and results/variable_precision_summary.json.                      │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Dark-Mode Plots & Evaluation Report                                                  │
│ • Validates result schemas and verifies cosine bounds (-1.0 to 1.0).                          │
│ • Generates 8 dark-mode plots (01_ppl_vs_bits.png through 08_time_vs_ppl.png).                │
│ • Builds competitive comparison table (UPR vs FP16 / AWQ / GPTQ published numbers).           │
│ • Exports results/upr_evaluation_report.json.                                                 │
│ • Automatically downloads all plots and reports back to your local results/ folder.           │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Audit & Infrastructure Fixes (Phase 1.1)

Every reported metric is **100% mathematically valid, deterministic, and auditable**.

| Fix | Description | Status |
|-----|-------------|--------|
| Fix 1 | Cosine similarity bug — bounds assertion + L2 vector norm dot product | ✅ |
| Fix 2 | Numerical validation framework (MAE, RMSE, KL, MRE) | ✅ |
| Fix 3 | Layer-wise activation comparison via forward hooks | ✅ |
| Fix 5 | Per-tensor reconstruction CSV export | ✅ |
| Fix 6 | Isolated timing per phase (reconstruction / init / forward / generation) | ✅ |
| Fix 7 | Memory profiler — CPU RAM, GPU VRAM, checkpoint size to `results/memory.csv` | ✅ |
| Fix 8 | Bit-packing assertion: 1 bit per element, never 1 byte per element | ✅ |
| Fix 9 | Deterministic WikiText-2 evaluation (fixed prompt, tokenizer, generation params) | ✅ |
| Fix 10 | `set_seed(42)` before every precision sweep step | ✅ |
| Fix 11/12 | Unified JSON schema with dataset, seed, git commit, GPU, metadata fields | ✅ |
| Modal Migration | Native serverless Modal execution on NVIDIA T4 GPU with tar archive volume storage | ✅ |

---

## How to Run on Modal

### 1. Prerequisites

Ensure Modal CLI is authenticated:
```bash
modal setup
```

Ensure your Hugging Face token is saved in Modal Secrets as `hf-token`:
```bash
modal secret create hf-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Run All 4 Stages (Single Command)

In **PowerShell**:
```powershell
$env:PYTHONUTF8=1; modal run modal_app.py
```

In **CMD**:
```cmd
set PYTHONUTF8=1 && modal run modal_app.py
```

### 3. Run a Specific Stage

You can also run an individual stage using `--stage`:

```bash
modal run modal_app.py --stage 1    # Stage 1: Conversion & Level 1 Verification
modal run modal_app.py --stage 2    # Stage 2: Layer Activation Hooks & Logit Similarity
modal run modal_app.py --stage 3    # Stage 3: 16..2 Bit Precision Sweep & WikiText PPL
modal run modal_app.py --stage 4    # Stage 4: Plots & Final Report
```

---

## Persistent Data Storage (`upr-data-vol`)

All persistent data is stored in a **Modal Volume** (`upr-data-vol`) mounted at `/vol`:

```
Modal Volume (/vol)
├── models/
│   └── bitplane_qwen.tar                  ← Fast single tar archive of BitPlane checkpoint
└── results/
    ├── reconstruction.csv                  ← Per-tensor MAE, RMSE, CosSim, exact match
    ├── memory.csv                          ← Per-precision CPU RAM, GPU VRAM, checkpoint size
    ├── 16bit.json ... 2bit.json            ← Individual precision metrics
    ├── variable_precision_summary.json     ← Full 8-precision sweep summary
    ├── upr_evaluation_report.json          ← Complete evaluation report
    └── plots/                              ← 8 publication-ready dark-mode plots
        ├── 01_ppl_vs_bits.png              ← Perplexity vs precision
        ├── 02_cosine_vs_bits.png           ← Logit Cosine Sim vs precision
        ├── 03_top1_acc_vs_bits.png          ← Token Accuracy vs precision
        ├── 04_recon_time_vs_bits.png       ← Reconstruction Time vs precision
        ├── 05_ppl_delta_vs_bits.png        ← PPL degradation vs FP16 baseline
        ├── 06_quality_cliff_map.png        ← Heatmap of all metrics
        ├── 07_cos_vs_ppl_scatter.png       ← Quality vs PPL frontier
        └── 08_time_vs_ppl.png              ← Speed vs Quality tradeoff
```

Upon completion of Stage 4, `modal_app.py` automatically downloads the generated plots and `upr_evaluation_report.json` back to your local `results/` folder.

---

## Repository Structure

```
UniversalPrecisionRuntime/
│
├── modal_app.py                            ← Main Modal application pipeline (4 stages)
│
├── upr/                                    ← Core Python package
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
├── README.md                               ← Documentation
├── idea.md                                 ← Prototype specification
├── pyproject.toml                          ← Dependencies
└── .gitignore
```

---

## `upr` Package API Reference

```python
import upr

# Set deterministic random seed (Fix 10)
upr.set_seed(42)

# Convert FP16 model to BitPlane checkpoint
upr.convert_to_bitplanes(model_or_path="Qwen/Qwen3.5-0.8B", output_directory="/tmp/bp")

# Reconstruct state dict at any precision level
state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
    bitplane_directory="/tmp/bp",
    bits=8,                          # 16, 14, 12, 10, 8, 6, 4, or 2
    export_reconstruction_csv=True,
    original_state_dict=orig_sd,
    csv_output_path="results/reconstruction.csv"
)

# Verified cosine similarity (bounds asserted)
cos = upr.compute_cosine_similarity(tensor_a, tensor_b)

# Full numerical metrics (MAE, RMSE, KL, MRE, CosSim)
metrics = upr.compute_numerical_metrics(original_tensor, reconstructed_tensor)

# Isolated timing per phase
timer = upr.IsolatedTimer()
timer.start("reconstruction")
# ... do work ...
elapsed = timer.stop("reconstruction")

# Memory profiling
profiler = upr.MemoryProfiler()
profiler.record_memory_snapshot(precision_bits=8, checkpoint_dir="/tmp/bp", output_csv_path="results/memory.csv")

# Layer activation hooks comparison
collector = upr.LayerActivationCollector(model)
collector.register_hooks()
metrics = upr.compare_layer_activations(orig_collector, recon_collector)

# Experiment metadata
meta = upr.collect_experiment_metadata(precision_bits=8)
```

---

## Local Result Validation

You can run the validator script on your locally downloaded `results/` folder:

```bash
python validate_results.py results/
```

---

## License

Research prototype — not for production use.
