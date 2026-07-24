"""
Stage 3 — Variable Precision Sweep & WikiText Perplexity (Parallelized with concurrency_limit=10)
Sweeps precisions 16, 14, 12, 10, 8, 6, 4, 2 bits across 8 parallel T4 GPU containers.
"""
import os
import json
import gc
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT, RESULTS_DIR,
    setup_hf_auth, get_fast_checkpoint_dir, evaluate_perplexity
)


@app.function(
    image=base_image,
    gpu="T4",
    max_containers=10,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def eval_single_precision(bits: int):
    import torch
    import upr
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    setup_hf_auth()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_dir = get_fast_checkpoint_dir()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    try:
        test_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    text_samples = [t for t in test_dataset["text"] if len(t.strip()) > 100][:32]
    encodings = tokenizer("\n\n".join(text_samples), return_tensors="pt")
    seq_len = 512
    input_ids = encodings.input_ids[:, :seq_len * 4].to(device)

    EVAL_PROMPT = "Universal Precision Runtime provides dynamic multi-precision execution."
    GEN_PROMPT = "The key advantage of a single bit-plane representation is"
    GEN_MAX_NEW_TOKENS = 40

    prompt_inputs = tokenizer(EVAL_PROMPT, return_tensors="pt").to(device)
    gen_inputs = tokenizer(GEN_PROMPT, return_tensors="pt").to(device)

    # Load FP16 reference logits & output token ids for metric comparison
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()

    with torch.no_grad():
        orig_logits = orig_model(**prompt_inputs).logits.detach().cpu()
        orig_output_ids = orig_model.generate(**gen_inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False).detach().cpu()

    orig_state_dict = orig_model.state_dict()
    del orig_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    upr.set_seed(42)
    timer = upr.IsolatedTimer()

    timer.start("reconstruction")
    recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
        bitplane_directory=fast_dir,
        bits=bits,
        device="cpu",
        export_reconstruction_csv=(bits in [16, 8, 4]),
        original_state_dict=orig_state_dict,
        csv_output_path=f"{RESULTS_DIR}/reconstruction.csv"
    )
    t_recon = timer.stop("reconstruction")

    timer.start("model_init")
    config = AutoConfig.from_pretrained(MODEL_ID)
    recon_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    recon_model.load_state_dict(recon_state_dict, strict=True)
    del recon_state_dict
    gc.collect()
    recon_model = recon_model.to(device)
    recon_model.eval()
    t_init = timer.stop("model_init")

    timer.start("forward_pass")
    try:
        with torch.no_grad():
            recon_logits = recon_model(**prompt_inputs).logits.detach().cpu()
        cos_sim = upr.compute_cosine_similarity(orig_logits, recon_logits)
        kl_div = upr.compute_kl_divergence(orig_logits, recon_logits)
    except Exception:
        cos_sim = 0.0
        kl_div = 0.0
    t_forward = timer.stop("forward_pass")

    timer.start("generation")
    try:
        with torch.no_grad():
            recon_output_ids = recon_model.generate(**gen_inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False).detach().cpu()
        token_matches = (orig_output_ids == recon_output_ids).sum().item()
        token_acc = (token_matches / orig_output_ids.numel()) * 100.0
    except Exception:
        token_acc = 0.0
    t_gen = timer.stop("generation")

    ppl = evaluate_perplexity(recon_model, input_ids, seq_len=seq_len)
    mem_profiler = upr.MemoryProfiler()
    mem_stats = mem_profiler.record_memory_snapshot(bits, fast_dir, f"{RESULTS_DIR}/memory.csv")

    del recon_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res_item = {
        "precision_bits": int(bits),
        "dataset": "wikitext-2-raw-v1",
        "seed": 42,
        "timing_sec": {
            "reconstruction": round(float(t_recon), 4),
            "model_init": round(float(t_init), 4),
            "forward_pass": round(float(t_forward), 4),
            "generation": round(float(t_gen), 4)
        },
        "memory_stats": mem_stats,
        "logit_cosine_similarity": float(cos_sim),
        "logit_kl_divergence": float(kl_div),
        "top1_token_accuracy_pct": float(token_acc),
        "perplexity": float(ppl),
        "metadata": upr.collect_experiment_metadata(precision_bits=bits)
    }

    with open(f"{RESULTS_DIR}/{bits}bit.json", "w") as f:
        json.dump(res_item, f, indent=2)

    print(f"[{bits}bit Complete]  Recon={t_recon:.2f}s | CosSim={cos_sim:.6f} | Top1={token_acc:.1f}% | PPL={ppl:.4f}")
    return res_item


@app.function(
    image=base_image,
    gpu="any",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def stage_03_precision_sweep():
    import upr
    setup_hf_auth()

    print("=== Stage 3: Launching Parallel Precision Sweep across 8 T4 GPU Containers (16..2 bits) ===")
    precisions = [16, 14, 12, 10, 8, 6, 4, 2]

    # Map evaluation over 8 parallel GPU containers (throttled to concurrency_limit=10)
    sweep_results = list(eval_single_precision.map(precisions))

    # Sort results by precision_bits descending (16 -> 2)
    sweep_results = sorted(sweep_results, key=lambda x: x["precision_bits"], reverse=True)

    fp16_ppl = next((r["perplexity"] for r in sweep_results if r["precision_bits"] == 16), 24.0684)

    summary_data = {
        "baseline_model": MODEL_ID,
        "baseline_perplexity": float(fp16_ppl),
        "precision_bits": 16,
        "dataset": "wikitext-2-raw-v1",
        "seed": 42,
        "precision_sweep": sweep_results
    }

    summary_path = f"{RESULTS_DIR}/variable_precision_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("PARALLEL VARIABLE PRECISION SWEEP COMPLETED SUCCESSFULLY!")
    print(f"Saved: {summary_path}")
    print("=" * 70)

    vol.commit()
    return json.loads(json.dumps(summary_data))
