"""
Stage 5 — Phase 1.2 Bit Plane Importance Analysis (Exp 1 & 2)
Performs leave-one-out plane ablation across all 16 bit planes (plane 15 to plane 0),
evaluating perplexity, logit cosine similarity, KL divergence, and weight MAE/RMSE per dropped plane.
"""
import os
import json
import csv
import gc
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT, RESULTS_DIR,
    setup_hf_auth, get_fast_checkpoint_dir
)


@app.function(
    image=base_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def stage_05_bit_importance():
    import torch
    import upr
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    setup_hf_auth()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_dir = get_fast_checkpoint_dir()

    print("=== Phase 1.2 Stage 5: Bit Plane Importance Analysis (Leave-One-Out Ablation) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    try:
        test_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    text_samples = [t for t in test_dataset["text"] if len(t.strip()) > 100][:32]
    encodings = tokenizer("\n\n".join(text_samples), return_tensors="pt")
    seq_len = 512
    input_ids = encodings.input_ids[:, :seq_len * 4].to(device)

    def eval_ppl(model):
        model.eval()
        nlls = []
        total_len = input_ids.size(1)
        end_loc = 0
        for i in range(0, total_len, seq_len):
            end_loc = min(i + seq_len, total_len)
            if end_loc - i < 64:
                continue
            trg_len = end_loc - i
            chunk_ids = input_ids[:, i:end_loc]
            with torch.no_grad():
                try:
                    loss = model(chunk_ids, labels=chunk_ids).loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        return 9999.0
                    nlls.append(loss * trg_len)
                except Exception:
                    return 9999.0
        if not nlls or end_loc == 0:
            return 9999.0
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        val = float(ppl.item())
        return val if not (torch.isnan(ppl) or torch.isinf(ppl)) else 9999.0

    print("Loading FP16 baseline...")
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()
    baseline_ppl = eval_ppl(orig_model)
    orig_state_dict = orig_model.state_dict()

    prompt_inputs = tokenizer("Universal Precision Runtime provides dynamic multi-precision execution.", return_tensors="pt").to(device)
    with torch.no_grad():
        orig_logits = orig_model(**prompt_inputs).logits.detach().cpu()

    del orig_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(MODEL_ID)
    ablation_results = []
    csv_rows = []

    print(f"\n{'Dropped Plane':<15} | {'CosSim':<12} | {'KL Div':<10} | {'PPL':<12} | {'Delta PPL'}")
    print("-" * 65)

    for plane_to_drop in range(15, -1, -1):
        upr.set_seed(42)

        recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
            bitplane_directory=fast_dir,
            bits=16,
            device="cpu",
            drop_planes=[plane_to_drop]
        )

        recon_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        recon_model.load_state_dict(recon_state_dict, strict=True)
        recon_model = recon_model.to(device)
        recon_model.eval()

        with torch.no_grad():
            recon_logits = recon_model(**prompt_inputs).logits.detach().cpu()

        cos_sim = upr.compute_cosine_similarity(orig_logits, recon_logits)
        kl_div = upr.compute_kl_divergence(orig_logits, recon_logits)
        ppl = eval_ppl(recon_model)
        delta_ppl = ppl - baseline_ppl

        total_mae, total_rmse, count = 0.0, 0.0, 0
        for name, orig_t in orig_state_dict.items():
            diff = torch.abs(orig_t.cpu().float() - recon_state_dict[name].cpu().float())
            total_mae += float(diff.mean().item())
            total_rmse += float(torch.sqrt((diff ** 2).mean()).item())
            count += 1

        avg_mae = total_mae / max(1, count)
        avg_rmse = total_rmse / max(1, count)

        row = {
            "dropped_plane": plane_to_drop,
            "plane_name": f"Plane {plane_to_drop}" + (" (MSB)" if plane_to_drop == 15 else " (LSB)" if plane_to_drop == 0 else ""),
            "perplexity": round(ppl, 4),
            "delta_perplexity": round(delta_ppl, 4),
            "cosine_similarity": round(cos_sim, 6),
            "kl_divergence": round(kl_div, 6),
            "mae": round(avg_mae, 6),
            "rmse": round(avg_rmse, 6)
        }
        ablation_results.append(row)
        csv_rows.append(row)

        print(f"Drop Plane {plane_to_drop:<5} | {cos_sim:<12.6f} | {kl_div:<10.6f} | {ppl:<12.4f} | +{delta_ppl:.4f}")

        del recon_model, recon_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = f"{RESULTS_DIR}/bit_importance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = f"{RESULTS_DIR}/bit_importance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_ppl": baseline_ppl, "ablation_results": ablation_results}, f, indent=2)

    vol.commit()
    return json.loads(json.dumps({"csv": csv_path, "json": json_path, "results": ablation_results}))
