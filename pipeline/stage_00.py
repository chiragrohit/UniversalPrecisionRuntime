"""
Stage 0 — Baseline Setup (Parallelized with concurrency_limit=10)
Evaluates exact WikiText-2 perplexity for standard quantization baselines on Qwen/Qwen3.5-0.8B:
- FP16 Baseline (16-bit)
- BitsAndBytes NF4 (4-bit)
- BitsAndBytes INT8 (8-bit)
"""
import os
import json
import gc
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT, RESULTS_DIR,
    BASELINES_JSON, setup_hf_auth, evaluate_perplexity
)


@app.function(
    image=base_image,
    gpu="T4",
    max_containers=10,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=1800
)
def eval_single_baseline(method_type: str):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import upr

    setup_hf_auth()
    upr.set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    try:
        ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    test_texts = [t for t in ds["text"] if len(t.strip()) > 100][:32]
    enc = tokenizer("\n\n".join(test_texts), return_tensors="pt")
    seq_len = 512
    input_ids = enc.input_ids[:, :seq_len * 4].to(device)

    print(f"=== Evaluating Baseline: {method_type} on {device} ===")

    if method_type == "fp16":
        fp16_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        ppl = evaluate_perplexity(fp16_model, input_ids, seq_len=seq_len)
        return {
            "key": "fp16",
            "bits": 16,
            "perplexity": float(ppl),
            "method": "FP16 (baseline)",
            "model_id": MODEL_ID,
            "status": "measured"
        }

    elif method_type == "bitsandbytes_4bit":
        try:
            bnb_4bit_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb_4bit_config, device_map="auto"
            )
            ppl = evaluate_perplexity(model, input_ids, seq_len=seq_len)
            return {
                "key": "bitsandbytes_4bit",
                "bits": 4,
                "perplexity": float(ppl),
                "method": "BitsAndBytes NF4 4-bit",
                "model_id": MODEL_ID,
                "status": "measured"
            }
        except Exception as e:
            return {"key": "bitsandbytes_4bit", "bits": 4, "perplexity": None, "status": f"failed: {e}"}

    elif method_type == "bitsandbytes_8bit":
        try:
            bnb_8bit_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb_8bit_config, device_map="auto"
            )
            ppl = evaluate_perplexity(model, input_ids, seq_len=seq_len)
            return {
                "key": "bitsandbytes_8bit",
                "bits": 8,
                "perplexity": float(ppl),
                "method": "BitsAndBytes INT8 8-bit",
                "model_id": MODEL_ID,
                "status": "measured"
            }
        except Exception as e:
            return {"key": "bitsandbytes_8bit", "bits": 8, "perplexity": None, "status": f"failed: {e}"}


@app.function(
    image=base_image,
    gpu="any",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=1800
)
def stage_00_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Stage 0: Launching Parallel Baseline Evaluations (FP16, NF4 4-bit, INT8 8-bit) ===")
    methods = ["fp16", "bitsandbytes_4bit", "bitsandbytes_8bit"]

    # Map across methods in parallel on 3 GPU containers
    results_list = list(eval_single_baseline.map(methods))

    baselines = {
        "model": MODEL_ID,
        "dataset": "wikitext-2-raw-v1",
        "seed": 42,
        "fp16": {},
        "bitsandbytes_4bit": {},
        "bitsandbytes_8bit": {}
    }

    for res in results_list:
        baselines[res["key"]] = res

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY (PARALLEL)")
    for key, data in baselines.items():
        if isinstance(data, dict) and "perplexity" in data:
            ppl_val = data.get("perplexity")
            ppl_str = f"{ppl_val:.4f}" if ppl_val is not None else "FAILED"
            print(f"  {key:<22} : PPL = {ppl_str}")
    print("=" * 60)

    with open(BASELINES_JSON, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    vol.commit()
    return json.loads(json.dumps(baselines))
