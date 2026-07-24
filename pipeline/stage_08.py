"""
Stage 8 — Phase 1.2 Representation Statistics, Error Propagation & Correlation Matrix
Computes per-plane entropy/bit-density, tracks error cascading across network layers, and generates Pearson correlation matrix.
"""
import os
import json
import csv
import gc
import numpy as np
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
def stage_08_rep_stats_and_error_propagation():
    import torch
    import upr
    setup_hf_auth()
    fast_dir = get_fast_checkpoint_dir()

    print("=== Phase 1.2 Stage 8: Representation Stats, Error Cascade & Correlation Matrix ===")

    # 1. Representation Stats (Exp 6)
    rep_stats = upr.analyze_representation_stats(fast_dir)
    stats_csv = f"{RESULTS_DIR}/representation_statistics.csv"
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rep_stats[0].keys()))
        writer.writeheader()
        writer.writerows(rep_stats)

    # 2. Error Propagation Cascade (Exp 8)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_inputs = tokenizer("Universal Precision Runtime provides dynamic multi-precision execution.", return_tensors="pt").to(device)

    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()
    orig_state_dict = orig_model.state_dict()

    orig_collector = upr.LayerActivationCollector(orig_model)
    orig_collector.register_hooks()
    with torch.no_grad():
        orig_outputs = orig_model(**prompt_inputs)
        orig_logits = orig_outputs.logits.detach().cpu()
        orig_pred = torch.argmax(orig_logits, dim=-1)

    config = AutoConfig.from_pretrained(MODEL_ID)
    error_cascade_rows = []

    for bits in [16, 14, 12, 10, 8, 6, 4, 2]:
        upr.set_seed(42)
        recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
            bitplane_directory=fast_dir,
            bits=bits,
            device="cpu"
        )
        total_w_err = sum(float(torch.abs(orig_state_dict[k].cpu().float() - recon_state_dict[k].cpu().float()).mean().item()) for k in orig_state_dict)
        weight_mae = total_w_err / len(orig_state_dict)

        recon_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        recon_model.load_state_dict(recon_state_dict, strict=True)
        recon_model = recon_model.to(device)
        recon_model.eval()

        recon_collector = upr.LayerActivationCollector(recon_model)
        recon_collector.register_hooks()

        with torch.no_grad():
            recon_outputs = recon_model(**prompt_inputs)
            recon_logits = recon_outputs.logits.detach().cpu()
            recon_pred = torch.argmax(recon_logits, dim=-1)

        layer_diffs = upr.compare_layer_activations(orig_collector, recon_collector)
        act_mae = float(np.mean([m["mae"] for m in layer_diffs.values()]))

        logit_mae = float(torch.abs(orig_logits - recon_logits).mean().item())
        pred_match_pct = float((orig_pred == recon_pred).float().mean().item()) * 100.0

        error_cascade_rows.append({
            "precision_bits": bits,
            "weight_error_mae": round(weight_mae, 6),
            "activation_error_mae": round(act_mae, 6),
            "logit_error_mae": round(logit_mae, 6),
            "prediction_match_pct": round(pred_match_pct, 2)
        })

        orig_collector.clear()
        recon_collector.clear()
        del recon_model, recon_state_dict
        gc.collect()

    del orig_model
    gc.collect()

    err_csv = f"{RESULTS_DIR}/error_propagation.csv"
    with open(err_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_cascade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_cascade_rows)

    # 3. Correlation Matrix (Exp 9)
    summary_path = f"{RESULTS_DIR}/variable_precision_summary.json"
    corr_rows = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for r in summary.get("precision_sweep", []):
            corr_rows.append({
                "precision_bits": r["precision_bits"],
                "perplexity": r["perplexity"],
                "cosine_similarity": r["logit_cosine_similarity"],
                "top1_accuracy": r["top1_token_accuracy_pct"],
                "kl_divergence": r["logit_kl_divergence"],
                "reconstruction_time": r["timing_sec"]["reconstruction"]
            })

    keys = ["precision_bits", "perplexity", "cosine_similarity", "top1_accuracy", "kl_divergence"]
    corr_dict = upr.compute_correlation_matrix(corr_rows, keys)
    corr_csv = f"{RESULTS_DIR}/correlation_matrix.csv"
    with open(corr_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + keys)
        for k in keys:
            writer.writerow([k] + [corr_dict[k][k2] for k2 in keys])

    vol.commit()
    return json.loads(json.dumps({"stats_csv": stats_csv, "err_csv": err_csv, "corr_csv": corr_csv}))
