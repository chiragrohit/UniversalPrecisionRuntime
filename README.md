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
BitPlane Checkpoint (one, stored once on Modal Volume)
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

**Phase 1.1 — Evaluation Audit & Infrastructure Fixes**

Every reported metric is **100% mathematically valid, deterministic, and auditable**.

| Fix | Description | Status |
|-----|-------------|--------|
| Fix 1 | Cosine similarity bug — bounds assertion + verified `F.cosine_similarity` | ✅ |
| Fix 2 | Numerical validation framework (MAE, RMSE, KL, MRE) | ✅ |
| Fix 3 | Layer-wise activation comparison via forward hooks | ✅ |
| Fix 5 | Per-tensor reconstruction CSV export | ✅ |
| Fix 6 | Isolated timing per phase (reconstruction / init / forward / generation) | ✅ |
| Fix 7 | Memory profiler — CPU RAM, GPU VRAM, checkpoint size to `results/memory.csv` | ✅ |
| Fix 8 | Bit-packing assertion: 1 bit per element, never 1 byte per element | ✅ |
| Fix 9 | Deterministic WikiText-2 evaluation (fixed prompt, tokenizer, generation params) | ✅ |
| Fix 10 | `set_seed(42)` before every precision sweep step | ✅ |
| Fix 11/12 | Unified JSON schema with dataset, seed, git commit, GPU, metadata fields | ✅ |
| Modal Migration | Native serverless Modal execution on NVIDIA GPU with `modal.Volume` and `hf-token` secret | ✅ |

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
├── idea.md                                 ← Full prototype specification
├── steps.md                                ← Phase plan
├── pyproject.toml
└── setup.py
```

---

## Modal Cloud Deployment & Execution

The entire UPR pipeline runs remotely on **Modal Labs (`modal`)** with GPU acceleration (NVIDIA T4).

### 1. Prerequisites

Make sure you have authenticated Modal CLI locally:
```bash
modal setup
```

Ensure your Hugging Face token is stored in Modal Secrets as `hf-token`:
```bash
modal secret create hf-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Running the Full Pipeline

Run all 4 stages sequentially on Modal with a single command:

```bash
modal run modal_app.py
```

### 3. Running Specific Stages

You can also run specific stages on Modal using the `--stage` flag:

```bash
# Run Stage 1 only (BitPlane Conversion & 16-Bit Verification)
modal run modal_app.py --stage 1

# Run Stage 2 only (Layer Activation Hooks & Logit Similarity)
modal run modal_app.py --stage 2

# Run Stage 3 only (16..2 Bit Precision Sweep & WikiText PPL)
modal run modal_app.py --stage 3

# Run Stage 4 only (Generate 8 Dark-Mode Plots & Competitive Report)
modal run modal_app.py --stage 4
```

---

## Persistent Data Storage (`upr-data-vol`)

All persistent artifacts are automatically stored inside a persistent **Modal Volume** (`upr-data-vol`) mounted at `/vol`:

```
Modal Volume (/vol)
├── models/
│   └── bitplane_qwen/                      ← BitPlane checkpoint (321 parameter tensors)
│       ├── metadata.json
│       └── tensors/
└── results/
    ├── reconstruction.csv                  ← Per-tensor MAE, RMSE, CosSim, exact match
    ├── memory.csv                          ← Per-precision CPU RAM, GPU VRAM, checkpoint size
    ├── 16bit.json ... 2bit.json            ← Individual precision metrics
    ├── variable_precision_summary.json     ← Full 8-precision sweep summary
    ├── upr_evaluation_report.json          ← Complete prototype evaluation report
    └── plots/                              ← 8 publication-ready dark-mode plots
        ├── 01_ppl_vs_bits.png
        ├── 02_cosine_vs_bits.png
        ├── 03_top1_acc_vs_bits.png
        ├── 04_recon_time_vs_bits.png
        ├── 05_ppl_delta_vs_bits.png
        ├── 06_quality_cliff_map.png
        ├── 07_cos_vs_ppl_scatter.png
        └── 08_time_vs_ppl.png
```

Upon completion of Stage 4, `modal_app.py` automatically syncs the generated plots and `upr_evaluation_report.json` back to your local `results/` folder.

---

## `upr` Package API

```python
import upr

# Seed (Fix 10)
upr.set_seed(42)

# Convert FP16 model to BitPlane checkpoint
upr.convert_to_bitplanes(model_or_path="Qwen/Qwen3.5-0.8B", output_directory="/vol/models/bp")

# Reconstruct state dict at any precision
state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
    bitplane_directory="/vol/models/bp",
    bits=8,                          # 16, 14, 12, 10, 8, 6, 4, or 2
    export_reconstruction_csv=True,
    original_state_dict=orig_sd,
    csv_output_path="/vol/results/reconstruction.csv"
)

# Fix 1 — Verified cosine similarity (bounds asserted)
cos = upr.compute_cosine_similarity(tensor_a, tensor_b)

# Fix 2 — Full numerical metrics
metrics = upr.compute_numerical_metrics(original_tensor, reconstructed_tensor)

# Fix 6 — Isolated timing
timer = upr.IsolatedTimer()
timer.start("reconstruction")
# ... do work ...
elapsed = timer.stop("reconstruction")

# Fix 7 — Memory profiling
profiler = upr.MemoryProfiler()
profiler.record_memory_snapshot(precision_bits=8, checkpoint_dir="/vol/models/bp", output_csv_path="/vol/results/memory.csv")

# Fix 3 — Layer activation comparison
collector = upr.LayerActivationCollector(model)
collector.register_hooks()
metrics = upr.compare_layer_activations(orig_collector, recon_collector)

# Fix 11 — Experiment metadata
meta = upr.collect_experiment_metadata(precision_bits=8)
```

---

## Validating Results

You can run the validator CLI against the downloaded `results/` directory:

```bash
python validate_results.py results/
```

---

## License

Research prototype — not for production use.
