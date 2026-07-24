# Phase 1.3 — Correct Memory & Bandwidth Accounting

Current memory metrics are misleading because the runtime always reconstructs a full FP16 model before inference.

Do NOT attempt to optimize memory yet.

Instead, report the correct theoretical and actual memory usage separately.

---

## 1. Keep Existing Metrics

Continue measuring

* CPU RAM
* GPU VRAM
* Peak GPU VRAM
* Reconstruction Time
* Model Initialization Time

These represent actual runtime usage.

---

## 2. Add UPR-Specific Memory Metrics

For every precision compute

### Planes Loaded

Example

16 bits

↓

16 planes

10 bits

↓

10 planes

4 bits

↓

4 planes

---

### Bytes Read From Checkpoint

Report

actual_bytes_loaded

instead of

full_checkpoint_size

Example

```text
16 bits

1920 MB

↓

10 bits

1200 MB

↓

8 bits

960 MB

↓

4 bits

480 MB

↓

2 bits

240 MB
```

---

### Effective Checkpoint Size

Compute

```text
effective_checkpoint_size =
checkpoint_size ×
(bits_loaded / total_planes)
```

Store

effective_checkpoint_size_mb

---

### Compression Ratio

Compute

```text
effective_checkpoint_size /
original_fp16_checkpoint_size
```

---

### Bandwidth Reduction

Compute

```text
bytes_loaded /
fp16_checkpoint_size
```

Example

```text
16 bits

100%

↓

8 bits

50%

↓

4 bits

25%

↓

2 bits

12.5%
```

---

## 3. Separate Runtime Memory From Representation Memory

Rename current fields

runtime_gpu_vram_mb

runtime_cpu_ram_mb

Add

representation_size_mb

effective_loaded_size_mb

planes_loaded

bytes_loaded

---

## 4. Add Summary Table

Generate

| Bits | Planes Loaded | Bytes Loaded | Effective Checkpoint | Runtime VRAM |
| ---- | ------------: | -----------: | -------------------: | -----------: |

This clearly separates

storage

from

runtime.

---

## 5. Do NOT Claim

Do not claim

* VRAM reduction
* GPU memory reduction
* Runtime memory reduction

Current implementation still reconstructs FP16 tensors.

These claims would be incorrect.

---

## 6. Add Future Metric Placeholder

Add fields

future_expected_vram_mb

future_expected_bandwidth

These should be calculated analytically only.

Clearly mark them as

"Theoretical"

not measured.

---

## Deliverable

The report should distinguish between

1. Actual runtime memory
2. Representation storage
3. Data loaded from checkpoint
4. Theoretical future execution memory

This prevents misleading conclusions while accurately demonstrating the storage advantages of UPR.
