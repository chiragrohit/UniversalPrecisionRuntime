"""
Stage 3 — Variable Precision Sweep & WikiText Perplexity (Level 4 & 5)
Sweeps precisions 16, 14, 12, 10, 8, 6, 4, 2 bits, profiling memory, isolated timings, and WikiText PPL.
"""
import os
import json
import gc
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT, RESULTS_DIR, BITPLANE_DIR,
    setup_hf_auth, get_fast_checkpoint_dir, evaluate_perplexity
)


@app.function(
    image=base_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def stage_03_precision_sweep():
    import torch
    import upr
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    setup_hf_auth()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fast_dir = get_fast_checkpoint_dir()

    print(f"=== Stage 3: Variable Precision Sweep & WikiText PPL on {device} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading WikiText-2 test split...")
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

    print("Evaluating FP16 Baseline...")
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()

    prompt_inputs = tokenizer(EVAL_PROMPT, return_tensors="pt").to(device)
    gen_inputs = tokenizer(GEN_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        orig_logits = orig_model(**prompt_inputs).logits.detach().cpu()
        orig_output_ids = orig_model.generate(**gen_inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False).detach().cpu()

    ppl_baseline = evaluate_perplexity(orig_model, input_ids, seq_len=seq_len)
    orig_state_dict = orig_model.state_dict()

    del orig_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"FP16 BASELINE RESULT | Perplexity (PPL): {ppl_baseline:.4f}")

    precisions = [16, 14, 12, 10, 8, 6, 4, 2]
    sweep_results = []
    config = AutoConfig.from_pretrained(MODEL_ID)
    mem_profiler = upr.MemoryProfiler()
    mem_csv = f"{RESULTS_DIR}/memory.csv"

    print(f"\n{'Bits':<6} | {'Recon(s)':<10} | {'Init(s)':<8} | {'CosSim':<12} | {'Top-1%':<10} | {'PPL'}")
    print("-" * 72)

    for bits in precisions:
        upr.set_seed(42)  # Enforce deterministic seed per step
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
        mem_stats = mem_profiler.record_memory_snapshot(bits, fast_dir, mem_csv)

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
        sweep_results.append(res_item)

        with open(f"{RESULTS_DIR}/{bits}bit.json", "w") as f:
            json.dump(res_item, f, indent=2)

        print(f"{bits:<6} | {t_recon:<10.3f} | {t_init:<8.3f} | {cos_sim:<12.6f} | {token_acc:<9.2f}% | {ppl:.4f}")

    summary_data = {
        "baseline_model": MODEL_ID,
        "baseline_perplexity": float(ppl_baseline),
        "precision_bits": 16,
        "dataset": "wikitext-2-raw-v1",
        "seed": 42,
        "precision_sweep": sweep_results
    }

    summary_path = f"{RESULTS_DIR}/variable_precision_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("VARIABLE PRECISION SWEEP COMPLETED SUCCESSFULLY!")
    print(f"Saved: {summary_path}")
    print("=" * 70)

    vol.commit()
    return json.loads(json.dumps(summary_data))
