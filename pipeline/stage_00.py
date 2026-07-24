"""
Stage 0 — Baseline Setup
Evaluates exact WikiText-2 perplexity for standard quantization baselines on Qwen/Qwen3.5-0.8B:
- FP16 Baseline (16-bit)
- BitsAndBytes NF4 (4-bit)
- BitsAndBytes INT8 (8-bit)
- AWQ 4-bit (SubSir/Qwen3.5-0.8B-AWQ)
- GPTQ 4-bit (Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-GPTQ)

Saves all measured results into results/baselines.json.
Subsequent stages read from baselines.json.
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

AWQ_MODEL_ID = "SubSir/Qwen3.5-0.8B-AWQ"
GPTQ_MODEL_ID = "Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-GPTQ"


@app.function(
    image=base_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=1800
)
def stage_00_baselines():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import upr

    setup_hf_auth()
    upr.set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading WikiText-2 test split for baseline evaluation...")
    try:
        ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    test_texts = [t for t in ds["text"] if len(t.strip()) > 100][:32]
    enc = tokenizer("\n\n".join(test_texts), return_tensors="pt")
    seq_len = 512
    input_ids = enc.input_ids[:, :seq_len * 4].to(device)

    # ===================================================================
    # 1. FP16 BASELINE
    # ===================================================================
    print("\n=== 1. Evaluating FP16 Baseline (Qwen/Qwen3.5-0.8B) ===")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    fp16_ppl = evaluate_perplexity(fp16_model, input_ids, seq_len=seq_len)
    print(f"[FP16 Baseline]  PPL = {fp16_ppl:.4f}")
    del fp16_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===================================================================
    # 2. BITSANDBYTES NF4 4-BIT BASELINE
    # ===================================================================
    print("\n=== 2. Evaluating BitsAndBytes NF4 4-bit Baseline ===")
    bnb_4bit_ppl = None
    try:
        bnb_4bit_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        bnb_4bit_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_4bit_config,
            device_map="auto"
        )
        bnb_4bit_ppl = evaluate_perplexity(bnb_4bit_model, input_ids, seq_len=seq_len)
        print(f"[BitsAndBytes NF4 4-bit] PPL = {bnb_4bit_ppl:.4f}")
        del bnb_4bit_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] BitsAndBytes NF4 evaluation error: {e}")

    # ===================================================================
    # 3. BITSANDBYTES INT8 8-BIT BASELINE
    # ===================================================================
    print("\n=== 3. Evaluating BitsAndBytes INT8 8-bit Baseline ===")
    bnb_8bit_ppl = None
    try:
        bnb_8bit_config = BitsAndBytesConfig(load_in_8bit=True)
        bnb_8bit_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_8bit_config,
            device_map="auto"
        )
        bnb_8bit_ppl = evaluate_perplexity(bnb_8bit_model, input_ids, seq_len=seq_len)
        print(f"[BitsAndBytes INT8 8-bit] PPL = {bnb_8bit_ppl:.4f}")
        del bnb_8bit_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] BitsAndBytes INT8 evaluation error: {e}")

    # ===================================================================
    # 4. AWQ 4-BIT (SubSir/Qwen3.5-0.8B-AWQ)
    # ===================================================================
    print(f"\n=== 4. Evaluating AWQ 4-bit Baseline ({AWQ_MODEL_ID}) ===")
    awq_ppl = None
    try:
        awq_model = AutoModelForCausalLM.from_pretrained(
            AWQ_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        awq_ppl = evaluate_perplexity(awq_model, input_ids, seq_len=seq_len)
        if awq_ppl < 9000.0:
            print(f"[AWQ 4-bit]  PPL = {awq_ppl:.4f}")
        else:
            print(f"[WARN] AWQ evaluation yielded un-decompressed weight error (PPL = {awq_ppl:.4f})")
            awq_ppl = None
        del awq_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] AWQ loading/evaluation error: {e}")

    # ===================================================================
    # 5. GPTQ 4-BIT (Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-GPTQ)
    # ===================================================================
    print(f"\n=== 5. Evaluating GPTQ 4-bit Baseline ({GPTQ_MODEL_ID}) ===")
    gptq_ppl = None
    try:
        gptq_model = AutoModelForCausalLM.from_pretrained(
            GPTQ_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        gptq_ppl = evaluate_perplexity(gptq_model, input_ids, seq_len=seq_len)
        if gptq_ppl < 9000.0:
            print(f"[GPTQ 4-bit] PPL = {gptq_ppl:.4f}")
        else:
            gptq_ppl = None
        del gptq_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] GPTQ loading/evaluation error: {e}")

    # ===================================================================
    # 6. Save baselines.json
    # ===================================================================
    baselines = {
        "model": MODEL_ID,
        "dataset": "wikitext-2-raw-v1",
        "seed": 42,
        "fp16": {
            "bits": 16,
            "perplexity": float(fp16_ppl),
            "method": "FP16 (baseline)",
            "model_id": MODEL_ID,
            "status": "measured"
        },
        "bitsandbytes_4bit": {
            "bits": 4,
            "perplexity": float(bnb_4bit_ppl) if bnb_4bit_ppl is not None else None,
            "method": "BitsAndBytes NF4 4-bit",
            "model_id": MODEL_ID,
            "status": "measured" if bnb_4bit_ppl is not None else "failed"
        },
        "bitsandbytes_8bit": {
            "bits": 8,
            "perplexity": float(bnb_8bit_ppl) if bnb_8bit_ppl is not None else None,
            "method": "BitsAndBytes INT8 8-bit",
            "model_id": MODEL_ID,
            "status": "measured" if bnb_8bit_ppl is not None else "failed"
        },
        "awq": {
            "bits": 4,
            "perplexity": float(awq_ppl) if awq_ppl is not None else None,
            "method": "AWQ 4-bit (SubSir)",
            "model_id": AWQ_MODEL_ID,
            "status": "measured" if awq_ppl is not None else "failed"
        },
        "gptq": {
            "bits": 4,
            "perplexity": float(gptq_ppl) if gptq_ppl is not None else None,
            "method": "GPTQ 4-bit (AutoRound)",
            "model_id": GPTQ_MODEL_ID,
            "status": "measured" if gptq_ppl is not None else "failed"
        }
    }

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY")
    print(f"  FP16  Baseline      PPL : {fp16_ppl:.4f}")
    print(f"  BitsAndBytes NF4 4-bit  : {bnb_4bit_ppl:.4f}" if bnb_4bit_ppl is not None else "  BitsAndBytes NF4 4-bit  : FAILED")
    print(f"  BitsAndBytes INT8 8-bit : {bnb_8bit_ppl:.4f}" if bnb_8bit_ppl is not None else "  BitsAndBytes INT8 8-bit : FAILED")
    print(f"  AWQ 4-bit               : {awq_ppl:.4f}" if awq_ppl is not None else "  AWQ 4-bit               : FAILED")
    print(f"  GPTQ 4-bit              : {gptq_ppl:.4f}" if gptq_ppl is not None else "  GPTQ 4-bit              : FAILED")
    print("=" * 60)

    with open(BASELINES_JSON, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    vol.commit()
    return json.loads(json.dumps(baselines))
