# Universal Precision Runtime (UPR)

## Prototype Specification v0.2

---

# Objective

Validate the hypothesis:

> **A single FP16 model can be converted into a single bit-plane representation from which multiple execution precisions (16, 14, 12, 10, 8, 6, 4, 2 bits) can be reconstructed at runtime without storing separate quantized checkpoints.**

This prototype **does not attempt to optimize speed**.

The objective is to validate:

- correctness
- numerical behavior
- model quality
- feasibility of a single-representation runtime

---

# Core Principle

Current inference ecosystem

```text
FP16 Model

INT8 Model

INT4 Model

AWQ Model

GPTQ Model
```

Multiple checkpoints.

---

This project

```text
Original FP16 Model
        │
One-time conversion
        │
        ▼
Single Bit-Plane Model
        │
        │
Runtime reconstructs
        │
        ▼
16-bit

14-bit

12-bit

10-bit

8-bit

6-bit

4-bit

2-bit
```

There is **never** an INT8 model.

There is **never** an INT4 model.

There is only one BitPlane checkpoint.

---

# Deliverables

The project produces exactly two stored models.

```
models/

qwen_fp16/

bitplane_qwen/
```

Nothing else.

No

```
int8/

int4/

int2/
```

directories should ever exist.

---

# Pipeline

```
HF FP16 Model

↓

BitPlane Converter

↓

BitPlane Checkpoint

↓

BitPlane Loader(bits=N)

↓

Reconstructed FP16 Tensor

↓

Inference
```

---

# Experiments

---

## Experiment 1 — Original Baseline

Purpose

Reference implementation.

Pipeline

```
Original FP16 checkpoint

↓

Load with HuggingFace

↓

Inference
```

Collect

- Perplexity
- Loss
- Generated text
- Token probabilities
- Logits
- Memory usage
- Inference time

Store

```
results/original.json
```

---

## Experiment 2 — Full Reconstruction

Purpose

Verify that BitPlane storage is completely lossless.

Pipeline

```
BitPlane checkpoint

↓

Load ALL planes

↓

Reconstruct tensor

↓

Inference
```

Expected

- identical weights
- identical logits
- identical outputs
- identical perplexity

Validation

```
torch.equal(original, reconstructed)
```

must return

```
True
```

Store

```
results/full16.json
```

---

## Experiment 3 — Variable Precision Reconstruction

Purpose

Validate that one BitPlane checkpoint can materialize multiple precision variants.

Run

```
16 bits

14 bits

12 bits

10 bits

8 bits

6 bits

4 bits

2 bits
```

Each experiment

```
BitPlane checkpoint

↓

Load selected planes

↓

Reconstruct tensor

↓

Inference

↓

Save metrics
```

Store

```
results/

16bit.json

14bit.json

12bit.json

10bit.json

8bit.json

6bit.json

4bit.json

2bit.json
```

---

# Evaluation Metrics

For every precision

Collect

## Model Metrics

- Loss
- Perplexity
- Generated text
- Top-1 agreement
- Top-5 agreement

---

## Numerical Metrics

Compare reconstructed weights against original

Compute

- MAE
- RMSE
- Max error
- Cosine similarity

Layer-wise.

---

## Layer Metrics

Feed identical input.

Compare

Original activation

vs

Reconstructed activation

Collect

- L2 error
- Cosine similarity

---

## Runtime Metrics

Measure

- conversion time
- reconstruction time
- inference time

---

## Storage Metrics

Measure

Original checkpoint size

BitPlane checkpoint size

Bytes loaded for each precision

Memory usage

---

# BitPlane Storage Format

Input

```
torch.float16 tensor
```

Convert

```
float16

↓

uint16

↓

16 bits
```

Generate

```
plane15.bin

plane14.bin

...

plane0.bin
```

Each plane stores

```
one bit

for

every weight
```

Bit packing is mandatory.

Never store

```
1 bit

as

1 byte
```

Use packed bit arrays.

---

# Storage Layout

```
bitplane_qwen/

metadata.json

layers/

layer0/

plane15.bin

plane14.bin

...

plane0.bin

layer1/

...

...
```

metadata.json stores

- tensor name
- tensor shape
- dtype
- plane count
- tensor ordering

---

# Converter

Implement

```python
convert_to_bitplanes(
    input_model,
    output_directory
)
```

Responsibilities

- load HF checkpoint
- convert every tensor
- generate planes
- pack bits
- write metadata
- never keep entire model duplicated in RAM

Process tensors sequentially.

---

# Loader

Implement

```python
BitPlaneModel.from_pretrained(
    model_path,
    bits=16
)
```

Supported values

```
16

14

12

10

8

6

4

2
```

No separate checkpoints.

The loader always opens

```
bitplane_qwen/
```

Internally

```
Read metadata

↓

Read requested planes

↓

Reconstruct tensor

↓

Load into HF model

↓

Inference
```

The downstream inference code should remain unchanged.

---

# Reconstruction Algorithm

Input

```
requested_bits
```

Examples

```
16

14

12

10

8

6

4

2
```

Algorithm

```
Read required MSB planes

↓

Fill remaining planes with zero

↓

Reconstruct uint16

↓

reinterpret as float16

↓

return tensor
```

This is Version 1.

Future versions may experiment with different plane selection strategies.

---

# API

Normal usage

```python
model = BitPlaneModel.from_pretrained(
    "bitplane_qwen",
    bits=8
)
```

Changing precision

```python
model = BitPlaneModel.from_pretrained(
    "bitplane_qwen",
    bits=4
)
```

The checkpoint never changes.

Only

```
bits=
```

changes.

---

# Visualization

Generate

## Accuracy vs Precision

```
16
14
12
10
8
6
4
2
```

Plot

- perplexity
- loss

---

## Reconstruction Error

Plot

- MAE
- RMSE

vs precision.

---

## Memory Loaded

Plot

Bytes loaded

vs

precision.

---

## Layer Error

Heatmap

Layer

×

Precision

---

# Initial Model

Use

```
Qwen/Qwen3.5-0.8B
```

---

# Success Criteria

## Stage 1

BitPlane checkpoint can reconstruct

100%

identical weights.

---

## Stage 2

Experiment 2 produces

identical outputs

to original HF model.

---

## Stage 3

One BitPlane checkpoint successfully reconstructs

```
16

14

12

10

8

6

4

2
```

precision variants.

---

## Stage 4

Quality degradation across precision levels is quantified.

---

# Out of Scope

This prototype will **not** attempt:

- CUDA kernels
- Triton kernels
- custom Tensor Core operations
- speed optimization
- hardware acceleration
- adaptive query routing
- automatic precision selection

These are future phases.

---

# Future Roadmap

Phase 2

Compare reconstructed 8-bit and 4-bit variants against AWQ and GPTQ.

Phase 3

Implement optimized reconstruction kernels.

Phase 4

Reduce reconstruction overhead.

Phase 5

Investigate hardware-friendly memory layouts (e.g., bit-plane-aware memory systems).

---

# Prototype Goal

The prototype is successful if it demonstrates:

> **One stored bit-plane checkpoint can serve as a universal representation from which multiple execution precisions can be reconstructed on demand, while reproducing the original model exactly at full precision and exposing a measurable quality-versus-precision tradeoff at lower precisions.**

---

# Evaluation Methodology

The objective is **not** to compare different models.

The objective is to compare **different representations of the exact same model**.

Every experiment must use:

* identical tokenizer
* identical prompts
* identical architecture
* identical generation parameters
* identical random seed (where applicable)

Only the weight representation changes.

---

## Comparison Matrix

The following comparisons must be executed.

### Baseline

Original HuggingFace FP16 checkpoint.

---

### Reconstruction 16-bit

BitPlane checkpoint reconstructed using all 16 planes.

Purpose:

Validate that the BitPlane representation is completely lossless.

Expected:

* identical weights
* identical logits
* identical outputs

---

### Reduced Precision

Run all of the following:

```text
16 bits
14 bits
12 bits
10 bits
8 bits
6 bits
4 bits
2 bits
```

Each reconstruction must come from the **same BitPlane checkpoint**.

No additional checkpoints may exist.

---

# Evaluation Levels

## Level 1 — Weight Reconstruction

Compare reconstructed tensors against original tensors.

For every parameter tensor compute:

* torch.equal()
* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Maximum Absolute Error
* Cosine Similarity

For 16-bit reconstruction:

```python
torch.equal(original, reconstructed)
```

must return

```python
True
```

If not, reconstruction is incorrect and remaining experiments should not run.

---

## Level 2 — Layer Output Comparison

For a fixed prompt:

Run both models.

After every transformer block compare

Original Layer Output

vs

BitPlane Layer Output.

Compute

* L2 Error
* Cosine Similarity
* Maximum Difference

Store per-layer statistics.

---

## Level 3 — Logit Comparison

Feed identical tokenized inputs.

Compare

Original logits

vs

BitPlane logits.

Compute

* MAE
* RMSE
* Cosine Similarity
* KL Divergence

This is one of the primary evaluation metrics.

---

## Level 4 — Generation Comparison

Use fixed prompts.

Generate with

* identical sampling strategy
* identical temperature
* identical max tokens
* identical random seed

Record

* generated text
* token probabilities
* top-1 agreement
* top-5 agreement

---

## Level 5 — Perplexity

Evaluate all precision levels on the same dataset.

Recommended datasets

* WikiText-2
* WikiText-103

Record

Perplexity for

* Original
* BitPlane16
* BitPlane14
* BitPlane12
* BitPlane10
* BitPlane8
* BitPlane6
* BitPlane4
* BitPlane2

Generate

Perplexity vs Precision graph.

---

## Level 6 — Benchmark Accuracy

Evaluate every precision on identical benchmark datasets.

Recommended

* MMLU
* ARC
* HellaSwag
* GSM8K

Generate comparison tables.

---

# Competitive Comparison

After completing all experiments above,

compare against existing quantization methods.

Include

* FP16
* AWQ
* GPTQ
* (optional) BitsAndBytes

Comparison table should include

* Storage size
* Number of checkpoints required
* Perplexity
* Benchmark accuracy
* Runtime reconstruction overhead

One important comparison column should be

**Supports Multiple Runtime Precisions From One Checkpoint**

Expected values

| Method           | Single Checkpoint | Multiple Runtime Precisions |
| ---------------- | ----------------- | --------------------------- |
| FP16             | Yes               | No                          |
| AWQ              | No                | No                          |
| GPTQ             | No                | No                          |
| BitPlane Runtime | Yes               | Yes                         |

---

# Figures To Generate

Generate the following plots automatically.

1. Perplexity vs Precision
2. MAE vs Precision
3. RMSE vs Precision
4. Cosine Similarity vs Precision
5. Memory Loaded vs Precision
6. Reconstruction Time vs Precision
7. Benchmark Accuracy vs Precision
8. Layer-wise Error Heatmap

All figures should be saved under

```
results/plots/
```

for inclusion in future reports or papers.

