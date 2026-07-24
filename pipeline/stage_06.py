"""
Stage 6 — Phase 1.2 Sensitivity Analysis (Layer & Tensor Sensitivity)
Maps activation error per layer and weight error per tensor category (Embedding, LM Head, Q/K/V/O, MLP) at 10, 8, 6 bits.
"""
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
def stage_06_sensitivity_analysis():
    import torch
    import upr
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    setup_hf_auth()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_dir = get_fast_checkpoint_dir()

    print("=== Phase 1.2 Stage 6: Layer & Tensor Sensitivity Analysis ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_inputs = tokenizer("Universal Precision Runtime provides dynamic multi-precision execution.", return_tensors="pt").to(device)

    print("Loading FP16 model baseline...")
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()
    orig_collector = upr.LayerActivationCollector(orig_model)
    orig_collector.register_hooks()

    with torch.no_grad():
        orig_model(**prompt_inputs)

    orig_state_dict = orig_model.state_dict()
    config = AutoConfig.from_pretrained(MODEL_ID)

    layer_csv_rows = []
    tensor_csv_rows = []
    test_precisions = [10, 8, 6]

    for bits in test_precisions:
        upr.set_seed(42)
        recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
            bitplane_directory=fast_dir,
            bits=bits,
            device="cpu"
        )

        recon_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        recon_model.load_state_dict(recon_state_dict, strict=True)
        recon_model = recon_model.to(device)
        recon_model.eval()

        recon_collector = upr.LayerActivationCollector(recon_model)
        recon_collector.register_hooks()

        with torch.no_grad():
            recon_model(**prompt_inputs)

        layer_metrics = upr.compare_layer_activations(orig_collector, recon_collector)
        for layer_name, m in layer_metrics.items():
            layer_csv_rows.append({
                "precision_bits": bits,
                "layer_name": layer_name,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "cosine_similarity": m["cosine_similarity"],
                "kl_divergence": m["kl_divergence"]
            })

        cat_diffs = {}
        for t_name, orig_t in orig_state_dict.items():
            cat = upr.categorize_tensor(t_name)
            if cat not in cat_diffs:
                cat_diffs[cat] = {"maes": [], "rmses": [], "coss": []}
            diff = torch.abs(orig_t.cpu().float() - recon_state_dict[t_name].cpu().float())
            cat_diffs[cat]["maes"].append(float(diff.mean().item()))
            cat_diffs[cat]["rmses"].append(float(torch.sqrt((diff ** 2).mean()).item()))
            cat_diffs[cat]["coss"].append(upr.compute_cosine_similarity(orig_t, recon_state_dict[t_name]))

        for cat, vals in cat_diffs.items():
            tensor_csv_rows.append({
                "precision_bits": bits,
                "tensor_category": cat,
                "mean_mae": round(float(np.mean(vals["maes"])), 6),
                "mean_rmse": round(float(np.mean(vals["rmses"])), 6),
                "mean_cosine_similarity": round(float(np.mean(vals["coss"])), 6)
            })

        orig_collector.clear()
        recon_collector.clear()
        del recon_model, recon_state_dict
        gc.collect()

    del orig_model
    gc.collect()

    layer_csv = f"{RESULTS_DIR}/layer_sensitivity.csv"
    with open(layer_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_csv_rows)

    tensor_csv = f"{RESULTS_DIR}/tensor_sensitivity.csv"
    with open(tensor_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tensor_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tensor_csv_rows)

    vol.commit()
    return json.loads(json.dumps({"layer_csv": layer_csv, "tensor_csv": tensor_csv}))
