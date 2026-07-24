"""
Stage 1 — BitPlane Conversion & Level 1 Weight Verification
Converts FP16 model weights into 16 binary bit-plane files per tensor on fast container NVMe (/tmp),
archives to /vol/models/bitplane_qwen.tar, and verifies 100% exact bitwise match (torch.equal == True).
"""
import os
import json
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT, RESULTS_DIR, TMP_DIR,
    setup_hf_auth, archive_tmp_to_vol
)


@app.function(
    image=base_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=3600
)
def stage_01_conversion():
    import torch
    import upr
    from transformers import AutoModelForCausalLM

    setup_hf_auth()
    upr.set_seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    print(f"=== Stage 1: Converting {MODEL_ID} -> {TMP_DIR} (fast local NVMe) ===")
    timer = upr.IsolatedTimer()
    timer.start("bitplane_conversion")

    # 1. Fast conversion to local NVMe /tmp/bitplane_qwen
    upr.convert_to_bitplanes(
        model_or_path=MODEL_ID,
        output_directory=TMP_DIR,
        torch_dtype=torch.float16
    )
    conv_time = timer.stop("bitplane_conversion")
    print(f"[OK] BitPlane conversion completed in {conv_time:.2f} seconds.")

    # 2. Archive to single tar file on Modal Volume for fast persistence
    archive_tmp_to_vol()

    print("\n--- Level 1 Verification: Reconstructing 16-bit FP16 weights ---")

    print("Loading original FP16 baseline model for comparison...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    orig_state_dict = orig_model.state_dict()

    timer.start("tensor_reconstruction")
    csv_path = f"{RESULTS_DIR}/reconstruction.csv"
    recon_state_dict = upr.BitPlaneModel.load_reconstructed_state_dict(
        bitplane_directory=TMP_DIR,
        bits=16,
        device="cpu",
        export_reconstruction_csv=True,
        original_state_dict=orig_state_dict,
        csv_output_path=csv_path
    )
    recon_time = timer.stop("tensor_reconstruction")

    total_tensors = len(orig_state_dict)
    exact_matches = sum(
        1 for name, orig_t in orig_state_dict.items()
        if torch.equal(orig_t, recon_state_dict[name])
    )

    print("=" * 60)
    print("LEVEL 1 RECONSTRUCTION RESULTS (16-bit Full Reconstruction)")
    print(f"Total Parameter Tensors Verified: {total_tensors}")
    print(f"Exact Bitwise Matches (torch.equal == True): {exact_matches} / {total_tensors} ({exact_matches/total_tensors*100:.2f}%)")
    print(f"Reconstruction Time: {recon_time:.2f} seconds")
    print(f"Per-tensor metrics exported to: {csv_path}")
    print("=" * 60)

    assert exact_matches == total_tensors, f"FAIL: Expected {total_tensors} bitwise matches, got {exact_matches}"

    vol.commit()
    res = {"status": "PASSED", "exact_matches": int(exact_matches), "total_tensors": int(total_tensors)}
    return json.loads(json.dumps(res))
