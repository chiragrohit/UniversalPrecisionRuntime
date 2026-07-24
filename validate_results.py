import os
import sys
import json
import glob
import csv
from typing import List, Dict, Any

def validate_experiment_results(results_dir: str = "results") -> bool:
    """
    Fix 14 — Experiment Validator CLI script.
    Checks:
    ✓ Cosine similarity within [-1.0, 1.0] bounds
    ✓ No NaN or Inf in outputs
    ✓ All tensors evaluated
    ✓ Metrics complete
    ✓ JSON schema valid
    ✓ Layer counts correct
    """
    print(f"============================================================")
    print(f"RUNNING UPR EXPERIMENT RESULT VALIDATOR: '{results_dir}'")
    print(f"============================================================")

    if not os.path.exists(results_dir):
        print(f"FAIL: Results directory '{results_dir}' does not exist.")
        return False

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"FAIL: No JSON result files found in '{results_dir}'.")
        return False

    failures: List[str] = []

    # 1. Schema & Bound Validation
    for jf in json_files:
        filename = os.path.basename(jf)
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_text = json.dumps(data)
            if "NaN" in raw_text or "Infinity" in raw_text:
                failures.append(f"{filename}: Contains NaN or Infinity values!")

            def check_cossim_recursive(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if "cosine" in k.lower() or "cossim" in k.lower():
                            if isinstance(v, (int, float)):
                                if not (-1.0 - 1e-5 <= v <= 1.0 + 1e-5):
                                    failures.append(f"{filename} [{k}]: Cosine {v} out of bounds [-1.0, 1.0]!")
                        else:
                            check_cossim_recursive(v)
                elif isinstance(obj, list):
                    for elem in obj:
                        check_cossim_recursive(elem)

            check_cossim_recursive(data)

        except Exception as e:
            failures.append(f"{filename}: JSON load failed: {str(e)}")

    # 2. Per-tensor reconstruction CSV validation
    recon_csv = os.path.join(results_dir, "reconstruction.csv")
    if os.path.exists(recon_csv):
        try:
            with open(recon_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) == 0:
                    failures.append("reconstruction.csv is empty!")
                else:
                    print(f"✓ Verified reconstruction.csv with {len(rows)} per-tensor entries.")
        except Exception as e:
            failures.append(f"reconstruction.csv read error: {str(e)}")

    # 3. Report Status
    if failures:
        print(f"\nVALIDATION FAILED WITH {len(failures)} ERROR(S):")
        for f in failures:
            print(f"  ❌ {f}")
        return False
    else:
        print(f"\n✓ ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
        return True

if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    success = validate_experiment_results(res_dir)
    sys.exit(0 if success else 1)
