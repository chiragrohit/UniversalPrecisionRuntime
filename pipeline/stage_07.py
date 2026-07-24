"""
Stage 7 — Phase 1.2 Progressive 1-Bit Resolution Sweep (Parallelized with concurrency_limit=10)
Evaluates perplexity, logit cosine similarity, and KL divergence across all 15 bit resolution steps (16..2 bits) in parallel.
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
def eval_single_1bit_step(bits: int):
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

    prompt_inputs = tokenizer("Universal Precision Runtime provides dynamic multi-precision execution.", return_tensors="pt").to(device)

    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()
    baseline_ppl = evaluate_perplexity(orig_model, input_ids, seq_len=seq_len)

    with torch.no_grad():
        orig_logits = orig_model(**prompt_inputs).logits.detach().cpu()

    del orig_model
    gc.collect()

    upr.set_seed(42)
    recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
        bitplane_directory=fast_dir,
        bits=bits,
        device="cpu"
    )

    config = AutoConfig.from_pretrained(MODEL_ID)
    recon_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    recon_model.load_state_dict(recon_state_dict, strict=True)
    recon_model = recon_model.to(device)
    recon_model.eval()

    with torch.no_grad():
        recon_logits = recon_model(**prompt_inputs).logits.detach().cpu()

    cos_sim = upr.compute_cosine_similarity(orig_logits, recon_logits)
    kl_div = upr.compute_kl_divergence(orig_logits, recon_logits)
    ppl = evaluate_perplexity(recon_model, input_ids, seq_len=seq_len)

    del recon_model, recon_state_dict
    gc.collect()

    print(f"[{bits}bit Step Complete]  CosSim={cos_sim:.6f} | KLDiv={kl_div:.6f} | PPL={ppl:.4f}")

    return {
        "bits": int(bits),
        "cosine_similarity": round(float(cos_sim), 6),
        "kl_divergence": round(float(kl_div), 6),
        "perplexity": round(float(ppl), 4),
        "delta_perplexity": round(float(ppl - baseline_ppl), 4)
    }


@app.function(
    image=base_image,
    gpu="any",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def stage_07_full_1bit_sweep():
    setup_hf_auth()

    print("=== Stage 7: Launching Parallel Progressive 1-Bit Resolution Sweep (16..2 bits) ===")
    bits_range = list(range(16, 1, -1))

    # Map evaluation over 15 precision steps (throttled to concurrency_limit=10)
    sweep_1bit = list(eval_single_1bit_step.map(bits_range))

    # Sort results by bits descending (16 -> 2)
    sweep_1bit = sorted(sweep_1bit, key=lambda x: x["bits"], reverse=True)

    baseline_ppl = next((r["perplexity"] for r in sweep_1bit if r["bits"] == 16), 24.0684)

    json_path = f"{RESULTS_DIR}/progressive_1bit_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_ppl": baseline_ppl, "sweep_1bit": sweep_1bit}, f, indent=2)

    print("\n" + "=" * 65)
    print("PARALLEL PROGRESSIVE 1-BIT SWEEP COMPLETED SUCCESSFULLY!")
    print(f"Saved: {json_path}")
    print("=" * 65)

    vol.commit()
    return json.loads(json.dumps({"json": json_path, "sweep": sweep_1bit}))
