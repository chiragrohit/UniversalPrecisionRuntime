"""
Universal Precision Runtime (UPR) — Modal Execution Entrypoint

Modular architecture:
- pipeline/common.py     : Shared Modal app, volume, container images, paths, helpers
- pipeline/stage_00.py   : Stage 0 Baseline Setup (FP16 + AWQ + GPTQ quantize & eval)
- pipeline/stage_01.py   : Stage 1 BitPlane Conversion & Level 1 Verification
- pipeline/stage_02.py   : Stage 2 Layer Activation Hooks & Logit Verification
- pipeline/stage_03.py   : Stage 3 Variable Precision Sweep & WikiText PPL
- pipeline/stage_04.py   : Stage 4 Dark-Mode Plots & Evaluation Report (uses real baselines.json)
- pipeline/stage_05.py   : Stage 5 Bit Plane Importance Ablation (Phase 1.2 Exp 1 & 2)
- pipeline/stage_06.py   : Stage 6 Layer & Tensor Sensitivity (Phase 1.2 Exp 3 & 4)
- pipeline/stage_07.py   : Stage 7 Progressive 1-Bit Sweep (Phase 1.2 Exp 5)
- pipeline/stage_08.py   : Stage 8 Rep Stats, Error Propagation & Correlation (Phase 1.2 Exp 6-9)
- pipeline/stage_09.py   : Stage 9 Phase 1.2 Plots & Characterization Report
"""
import os
import json
from pipeline.common import app, vol, BASELINES_JSON
from pipeline.stage_00 import stage_00_baselines
from pipeline.stage_01 import stage_01_conversion
from pipeline.stage_02 import stage_02_layer_logits
from pipeline.stage_03 import stage_03_precision_sweep
from pipeline.stage_04 import stage_04_plots_and_report
from pipeline.stage_05 import stage_05_bit_importance
from pipeline.stage_06 import stage_06_sensitivity_analysis
from pipeline.stage_07 import stage_07_full_1bit_sweep
from pipeline.stage_08 import stage_08_rep_stats_and_error_propagation
from pipeline.stage_09 import stage_09_phase1_2_report_and_plots


@app.local_entrypoint()
def main(stage: int = -1, phase: str = ""):
    print("=" * 70)
    print("UNIVERSAL PRECISION RUNTIME (UPR) - MODAL EXECUTION PIPELINE")
    print("=" * 70)

    # Full Phase 1.1 Trigger (Stages 0 -> 4)
    if phase == "1.1":
        print("\n>>> Launching Full Phase 1.1 Execution Pipeline (Stages 0 -> 4)...")
        stage = 11

    # Full Phase 1.2 Trigger (Stages 5 -> 9)
    if phase == "1.2":
        print("\n>>> Launching Full Phase 1.2: Representation Characterization (Stages 5 -> 9)...")
        stage = 55

    # Default run (no flags passed): Run full Phase 1.1 + Phase 1.2
    run_all = (stage == -1 and phase == "")

    # --- Stage 0: Baselines (FP16, AWQ, GPTQ) ---
    if run_all or stage == 11 or stage == 0:
        print("\n>>> Launching Stage 0: Baseline Setup (FP16, AWQ 4-bit, GPTQ 4-bit)...")
        s0_res = stage_00_baselines.remote()
        print(f"Stage 0 Result: {s0_res}")

    # --- Stage 1: Conversion & Level 1 ---
    if run_all or stage == 11 or stage == 1:
        print("\n>>> Launching Stage 1: BitPlane Conversion & Level 1 Verification...")
        s1_res = stage_01_conversion.remote()
        print(f"Stage 1 Result: {s1_res}")

    # --- Stage 2: Layer Hooks ---
    if run_all or stage == 11 or stage == 2:
        print("\n>>> Launching Stage 2: Layer Activation Hooks & Logit Verification...")
        s2_res = stage_02_layer_logits.remote()
        print(f"Stage 2 Result: {s2_res}")

    # --- Stage 3: Precision Sweep ---
    if run_all or stage == 11 or stage == 3:
        print("\n>>> Launching Stage 3: Variable Precision Sweep & WikiText PPL...")
        s3_res = stage_03_precision_sweep.remote()
        print(f"Stage 3 Complete. Evaluated {len(s3_res.get('precision_sweep', []))} precisions.")

    # --- Stage 4: Phase 1.1 Plots & Report ---
    if run_all or stage == 11 or stage == 4:
        print("\n>>> Launching Stage 4: Dark-Mode Plots & Evaluation Report...")
        s4_res = stage_04_plots_and_report.remote()
        _download_local_artifacts(s4_res)

    # --- Stage 5: Bit Importance Ablation ---
    if run_all or stage == 55 or stage == 5:
        print("\n>>> Launching Stage 5: Bit Plane Importance Analysis (Leave-One-Out Ablation)...")
        s5_res = stage_05_bit_importance.remote()
        print(f"Stage 5 Complete. Evaluated {len(s5_res.get('results', []))} bit plane ablations.")

    # --- Stage 6: Sensitivity Analysis ---
    if run_all or stage == 55 or stage == 6:
        print("\n>>> Launching Stage 6: Layer & Tensor Sensitivity Analysis...")
        s6_res = stage_06_sensitivity_analysis.remote()
        print(f"Stage 6 Complete: {s6_res}")

    # --- Stage 7: Progressive 1-Bit Sweep ---
    if run_all or stage == 55 or stage == 7:
        print("\n>>> Launching Stage 7: Progressive 1-Bit Resolution Sweep (16..2 bits)...")
        s7_res = stage_07_full_1bit_sweep.remote()
        print(f"Stage 7 Complete. Evaluated {len(s7_res.get('sweep', []))} precision steps.")

    # --- Stage 8: Rep Stats & Error Cascade ---
    if run_all or stage == 55 or stage == 8:
        print("\n>>> Launching Stage 8: Representation Stats, Error Propagation & Correlation...")
        s8_res = stage_08_rep_stats_and_error_propagation.remote()
        print(f"Stage 8 Complete: {s8_res}")

    # --- Stage 9: Phase 1.2 Plots & Characterization Report ---
    if run_all or stage == 55 or stage == 9:
        print("\n>>> Launching Stage 9: Phase 1.2 Plots & Final Characterization Report...")
        s9_res = stage_09_phase1_2_report_and_plots.remote()
        _download_local_artifacts(s9_res)

    print("\n" + "=" * 70)
    print("MODAL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def _download_local_artifacts(res_dict):
    local_results = "results"
    local_plots = f"{local_results}/plots"
    os.makedirs(local_plots, exist_ok=True)

    if "report" in res_dict:
        report_file = f"{local_results}/representation_analysis_report.json" if res_dict["report"].get("phase") == "1.2" else f"{local_results}/upr_evaluation_report.json"
        with open(report_file, "w") as f:
            json.dump(res_dict["report"], f, indent=2)

    if "plots" in res_dict:
        for plot_name, plot_bytes in res_dict["plots"].items():
            with open(f"{local_plots}/{plot_name}", "wb") as pf:
                pf.write(plot_bytes)

        print(f"\n[OK] Downloaded {len(res_dict['plots'])} plots to '{local_plots}/'")
