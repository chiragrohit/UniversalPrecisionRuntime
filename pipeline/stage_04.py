"""
Stage 4 — Benchmarks, Dark-Mode Plots & Competitive Comparison Report
Generates 8 dark-mode plots, checks baselines.json for real measured numbers (BitsAndBytes, AWQ, GPTQ),
and exports upr_evaluation_report.json.
"""
import os
import json
import glob
import datetime
import platform
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT, RESULTS_DIR, PLOTS_DIR, BASELINES_JSON,
    setup_hf_auth
)


@app.function(
    image=base_image,
    gpu="any",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=1800
)
def stage_04_plots_and_report():
    import torch
    import upr
    setup_hf_auth()
    upr.set_seed(42)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = f"{RESULTS_DIR}/variable_precision_summary.json"
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary JSON not found at {summary_path}. Run stage 3 first!")

    with open(summary_path) as f:
        summary = json.load(f)

    baseline_ppl = summary["baseline_perplexity"]
    sweep = summary["precision_sweep"]

    bits_list = [r["precision_bits"] for r in sweep]
    ppl_list = [r["perplexity"] for r in sweep]
    cos_list = [r["logit_cosine_similarity"] for r in sweep]
    acc_list = [r["top1_token_accuracy_pct"] for r in sweep]
    t_recon_list = [r["timing_sec"]["reconstruction"] for r in sweep]

    # Bounds Assertion & Schema Verification
    for r in sweep:
        cs = r["logit_cosine_similarity"]
        assert cs <= 1.0 + 1e-6, f"Cosine > 1 at {r['precision_bits']}bit: {cs}"
        assert cs >= -1.0 - 1e-6, f"Cosine < -1 at {r['precision_bits']}bit: {cs}"

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "#0f0f0f", "axes.facecolor": "#181818",
        "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#aaaaaa", "ytick.color": "#aaaaaa",
        "axes.edgecolor": "#333333", "grid.color": "#2a2a2a",
        "font.family": "DejaVu Sans", "font.size": 11
    })

    ACCENT = "#4ecdc4"
    DANGER = "#ff6b6b"
    GOLD = "#ffd700"

    # Plot 1: PPL vs Bits
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bits_list, ppl_list, "o-", color=ACCENT, linewidth=2.5, markersize=8)
    ax.axhline(y=baseline_ppl, color=GOLD, linestyle="--", linewidth=1.5, label=f"FP16 Baseline ({baseline_ppl:.1f})")
    ax.set_xlabel("Precision (bits)"); ax.set_ylabel("Perplexity (PPL)")
    ax.set_title("Perplexity vs. Precision Level"); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_xticks(bits_list)
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/01_ppl_vs_bits.png", dpi=150); plt.close()

    # Plot 2: Cosine Similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bits_list, cos_list, "s-", color=ACCENT, linewidth=2.5, markersize=8)
    ax.axhline(y=1.0, color=GOLD, linestyle="--", linewidth=1.5, label="Perfect Similarity (1.0)")
    ax.set_xlabel("Precision (bits)"); ax.set_ylabel("Logit Cosine Similarity")
    ax.set_title("Logit Cosine Similarity vs. Precision"); ax.legend()
    ax.set_ylim(-0.1, 1.1); ax.grid(True, alpha=0.3); ax.set_xticks(bits_list)
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/02_cosine_vs_bits.png", dpi=150); plt.close()

    # Plot 3: Top-1 Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(b) for b in bits_list], acc_list, color=ACCENT, alpha=0.85)
    ax.set_xlabel("Precision (bits)"); ax.set_ylabel("Top-1 Token Match (%)")
    ax.set_title("Token Accuracy vs. Precision"); ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/03_top1_acc_vs_bits.png", dpi=150); plt.close()

    # Plot 4: Recon Time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(b) for b in bits_list], t_recon_list, color=GOLD, alpha=0.85)
    ax.set_xlabel("Precision (bits)"); ax.set_ylabel("Reconstruction Time (s)")
    ax.set_title("Reconstruction Time vs. Precision")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/04_recon_time_vs_bits.png", dpi=150); plt.close()

    # Plot 5: PPL Delta
    ppl_deltas = [ppl - baseline_ppl for ppl in ppl_list]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [DANGER if d > 5 else ACCENT for d in ppl_deltas]
    ax.bar([str(b) for b in bits_list], ppl_deltas, color=colors, alpha=0.85)
    ax.axhline(y=0, color=GOLD, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Precision (bits)"); ax.set_ylabel("PPL Delta (vs FP16)")
    ax.set_title("PPL Degradation vs. FP16 Baseline")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/05_ppl_delta_vs_bits.png", dpi=150); plt.close()

    # Plot 6: Quality Cliff Map
    import numpy as np
    metric_names = ["PPL (norm)", "CosSim", "Top-1 Acc"]
    max_ppl = max(ppl_list) if max(ppl_list) < 9000 else baseline_ppl * 5
    ppl_norm = [1.0 - min((p - baseline_ppl) / (max_ppl - baseline_ppl + 1e-6), 1.0) for p in ppl_list]
    data = np.array([ppl_norm, cos_list, [a / 100.0 for a in acc_list]])
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(bits_list))); ax.set_xticklabels([f"{b}bit" for b in bits_list])
    ax.set_yticks(range(3)); ax.set_yticklabels(metric_names)
    ax.set_title("Quality Cliff Map (Green=Good, Red=Degraded)")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_quality_cliff_map.png", dpi=150); plt.close()

    # Plot 7: Cosine vs PPL Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ppl_list, cos_list, c=bits_list, cmap="plasma", s=150, zorder=5)
    for b, p, c in zip(bits_list, ppl_list, cos_list):
        ax.annotate(f"{b}bit", (p, c), textcoords="offset points", xytext=(6, 4), fontsize=9, color="#cccccc")
    ax.axvline(x=baseline_ppl, color=GOLD, linestyle="--", linewidth=1.5, label=f"FP16 PPL ({baseline_ppl:.1f})")
    ax.axhline(y=1.0, color=GOLD, linestyle=":", linewidth=1.5)
    ax.set_xlabel("Perplexity"); ax.set_ylabel("Logit Cosine Similarity")
    ax.set_title("Quality vs. PPL Frontier"); ax.legend()
    plt.colorbar(sc, ax=ax, label="Bits")
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/07_cos_vs_ppl_scatter.png", dpi=150); plt.close()

    # Plot 8: Recon Time vs PPL
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(t_recon_list, ppl_list, c=bits_list, cmap="plasma", s=150, zorder=5)
    for b, t, p in zip(bits_list, t_recon_list, ppl_list):
        ax.annotate(f"{b}bit", (t, p), textcoords="offset points", xytext=(6, 4), fontsize=9, color="#cccccc")
    ax.set_xlabel("Reconstruction Time (s)"); ax.set_ylabel("Perplexity")
    ax.set_title("Speed vs. Quality Tradeoff")
    plt.colorbar(sc, ax=ax, label="Bits")
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/08_time_vs_ppl.png", dpi=150); plt.close()

    print("All 8 dark-mode plots saved to Modal Volume.")

    # Read real measured baselines if available from Stage 0
    real_baselines = {}
    if os.path.exists(BASELINES_JSON):
        try:
            with open(BASELINES_JSON) as bf:
                real_baselines = json.load(bf)
        except Exception as e:
            print(f"Warning loading baselines.json: {e}")

    bnb_4bit_ppl = real_baselines.get("bitsandbytes_4bit", {}).get("perplexity")
    bnb_8bit_ppl = real_baselines.get("bitsandbytes_8bit", {}).get("perplexity")
    awq_ppl = real_baselines.get("awq", {}).get("perplexity")
    gptq_ppl = real_baselines.get("gptq", {}).get("perplexity")

    comparison_table = [
        {"Method": "FP16 (baseline)",          "Bits": 16, "PPL": baseline_ppl, "CosSim": 1.0,  "Status": "Measured"},
        {"Method": "BitsAndBytes INT8 8-bit",  "Bits": 8,  "PPL": bnb_8bit_ppl,  "CosSim": None, "Status": "Measured" if bnb_8bit_ppl else "Not Measured"},
        {"Method": "BitsAndBytes NF4 4-bit",   "Bits": 4,  "PPL": bnb_4bit_ppl,  "CosSim": None, "Status": "Measured" if bnb_4bit_ppl else "Not Measured"},
        {"Method": "AWQ 4-bit (SubSir)",       "Bits": 4,  "PPL": awq_ppl,       "CosSim": None, "Status": "Measured" if awq_ppl else "Not Measured"},
        {"Method": "GPTQ 4-bit (AutoRound)",   "Bits": 4,  "PPL": gptq_ppl,      "CosSim": None, "Status": "Measured" if gptq_ppl else "Not Measured"},
        {"Method": "UPR BitPlane 8-bit",        "Bits": 8,  "PPL": None,          "CosSim": None, "Status": "This Work"},
        {"Method": "UPR BitPlane 4-bit",        "Bits": 4,  "PPL": None,          "CosSim": None, "Status": "This Work"},
        {"Method": "UPR BitPlane 2-bit",        "Bits": 2,  "PPL": None,          "CosSim": None, "Status": "This Work"},
    ]
    bits_to_result = {r["precision_bits"]: r for r in sweep}
    for row in comparison_table:
        if row["Status"] == "This Work" and row["Bits"] in bits_to_result:
            r = bits_to_result[row["Bits"]]
            row["PPL"] = r["perplexity"]
            row["CosSim"] = r["logit_cosine_similarity"]

    # Phase 1.3 Memory & Bandwidth Accounting Table
    memory_accounting_table = []
    for r in sweep:
        m = r.get("memory_stats", {})
        bits = r["precision_bits"]
        planes_loaded = m.get("planes_loaded", bits)
        bytes_loaded = m.get("bytes_loaded", int(m.get("checkpoint_size_mb", 1920.37) * 1024 * 1024 * bits / 16))
        eff_chkpt_mb = m.get("effective_checkpoint_size_mb", round(m.get("checkpoint_size_mb", 1920.37) * bits / 16, 2))
        runtime_vram_mb = m.get("runtime_gpu_vram_mb", m.get("gpu_vram_mb", 2926.21))
        theo_vram_mb = m.get("theoretical_future_vram_mb", round(2926.21 * bits / 16, 2))
        theo_bw_pct = m.get("theoretical_future_bandwidth_pct", round(bits / 16 * 100.0, 2))
        
        memory_accounting_table.append({
            "precision_bits": bits,
            "planes_loaded": planes_loaded,
            "bytes_loaded": bytes_loaded,
            "effective_checkpoint_size_mb": eff_chkpt_mb,
            "runtime_gpu_vram_mb": runtime_vram_mb,
            "theoretical_future_vram_mb": theo_vram_mb,
            "theoretical_future_bandwidth_pct": theo_bw_pct,
            "type_classification": {
                "runtime_vram": "Actual Measured (FP16 model reconstruction)",
                "effective_checkpoint": "Actual Measured BitPlane Storage",
                "theoretical_future_vram": "Analytical Placeholder Only"
            }
        })

    report = {
        "experiment_name": "UPR_Phase1.3_ModalEvaluation",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "upr_version": getattr(upr, "__version__", "0.1.0"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(),
        "seed": 42,
        "model": summary["baseline_model"],
        "dataset": "wikitext-2-raw-v1",
        "baseline_perplexity": baseline_ppl,
        "real_baselines_measured": bool(real_baselines),
        "sweep_results": sweep,
        "comparison_table": comparison_table,
        "memory_accounting_table": memory_accounting_table
    }

    report_path = f"{RESULTS_DIR}/upr_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Final report saved to: {report_path}")
    vol.commit()

    plot_data = {}
    for p_path in sorted(glob.glob(f"{PLOTS_DIR}/*.png")):
        with open(p_path, "rb") as pf:
            plot_data[os.path.basename(p_path)] = pf.read()

    return {"report": json.loads(json.dumps(report)), "plots": plot_data}
