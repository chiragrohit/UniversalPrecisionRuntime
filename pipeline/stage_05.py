"""
Stage 5 — Phase 1.2 Bit Plane Importance Analysis (Parallelized with concurrency_limit=10)
Performs leave-one-out plane ablation across 16 bit planes in parallel across GPU containers.
"""
import os
import json
import csv
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
def eval_single_plane_drop(plane_to_drop: int):
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

    # FP16 reference model
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()
    baseline_ppl = evaluate_perplexity(orig_model, input_ids, seq_len=seq_len)
    orig_state_dict = orig_model.state_dict()

    with torch.no_grad():
        orig_logits = orig_model(**prompt_inputs).logits.detach().cpu()

    del orig_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    upr.set_seed(42)

    recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
        bitplane_directory=fast_dir,
        bits=16,
        device="cpu",
        drop_planes=[plane_to_drop]
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
    delta_ppl = ppl - baseline_ppl

    total_mae, total_rmse, count = 0.0, 0.0, 0
    for name, orig_t in orig_state_dict.items():
        diff = torch.abs(orig_t.cpu().float() - recon_state_dict[name].cpu().float())
        total_mae += float(diff.mean().item())
        total_rmse += float(torch.sqrt((diff ** 2).mean()).item())
        count += 1

    avg_mae = total_mae / max(1, count)
    avg_rmse = total_rmse / max(1, count)

    del recon_model, recon_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Drop Plane {plane_to_drop:2d} Complete]  CosSim={cos_sim:.6f} | PPL={ppl:.4f} | DeltaPPL=+{delta_ppl:.4f}")

    return {
        "dropped_plane": int(plane_to_drop),
        "plane_name": f"Plane {plane_to_drop}" + (" (MSB)" if plane_to_drop == 15 else " (LSB)" if plane_to_drop == 0 else ""),
        "perplexity": round(float(ppl), 4),
        "delta_perplexity": round(float(delta_ppl), 4),
        "cosine_similarity": round(float(cos_sim), 6),
        "kl_divergence": round(float(kl_div), 6),
        "mae": round(float(avg_mae), 6),
        "rmse": round(float(avg_rmse), 6)
    }


@app.function(
    image=base_image,
    gpu="any",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def stage_05_bit_importance():
    setup_hf_auth()

    print("=== Stage 5: Launching Parallel Bit Plane Importance Analysis (16 Planes) ===")
    planes_to_drop = list(range(15, -1, -1))

    # Map evaluation over 16 plane drop runs (throttled to concurrency_limit=10)
    ablation_results = list(eval_single_plane_drop.map(planes_to_drop))

    # Sort results by dropped_plane descending (15 -> 0)
    ablation_results = sorted(ablation_results, key=lambda x: x["dropped_plane"], reverse=True)

    baseline_ppl = 24.0684

    csv_path = f"{RESULTS_DIR}/bit_importance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_results)

    json_path = f"{RESULTS_DIR}/bit_importance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_ppl": baseline_ppl, "ablation_results": ablation_results}, f, indent=2)

    print("\n" + "=" * 65)
    print("PARALLEL BIT PLANE IMPORTANCE ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Saved: {csv_path}")
    print("=" * 65)

    vol.commit()
    return json.loads(json.dumps({"csv": csv_path, "json": json_path, "results": ablation_results}))
