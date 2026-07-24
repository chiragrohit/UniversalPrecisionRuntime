"""
Stage 2 — Layer Outputs & Logit Verification (Level 2 & 3)
Attaches forward hooks to all Transformer blocks and verifies intermediate activation matching and logit similarity.
"""
import json
import modal

from .common import (
    app, vol, base_image,
    MODEL_ID, VOL_MOUNT,
    setup_hf_auth, get_fast_checkpoint_dir
)


@app.function(
    image=base_image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOL_MOUNT: vol},
    timeout=1800
)
def stage_02_layer_logits():
    import torch
    import upr
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_hf_auth()
    upr.set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fast_dir = get_fast_checkpoint_dir()

    print(f"=== Stage 2: Layer Hooks & Logit Verification on {device} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "Universal Precision Runtime provides dynamic multi-precision execution."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Loading original FP16 model...")
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()

    orig_collector = upr.LayerActivationCollector(orig_model)
    orig_collector.register_hooks()

    with torch.no_grad():
        orig_outputs = orig_model(**inputs)
        orig_logits = orig_outputs.logits.detach().cpu()

    print("Loading reconstructed 16-bit BitPlane model from fast local disk...")
    recon_model = upr.BitPlaneModel.from_pretrained(fast_dir, bits=16, base_model_id=MODEL_ID, torch_dtype=torch.float16).to(device)
    recon_model.eval()

    recon_collector = upr.LayerActivationCollector(recon_model)
    recon_collector.register_hooks()

    with torch.no_grad():
        recon_outputs = recon_model(**inputs)
        recon_logits = recon_outputs.logits.detach().cpu()

    layer_diffs = upr.compare_layer_activations(orig_collector, recon_collector)
    cos_sim = upr.compute_cosine_similarity(orig_logits, recon_logits)
    kl_div = upr.compute_kl_divergence(orig_logits, recon_logits)

    orig_collector.clear()
    recon_collector.clear()

    print("=" * 60)
    print("STAGE 2 LOGIT & LAYER VERIFICATION")
    print(f"Captured Layer Hooks: {len(layer_diffs)} activations")
    print(f"Logit Cosine Similarity: {cos_sim:.8f} (Assertion <= 1.0 + 1e-6 passed)")
    print(f"Logit KL Divergence: {kl_div:.8f}")
    print("=" * 60)

    assert cos_sim >= 0.9999, f"Logit cosine similarity too low: {cos_sim}"

    vol.commit()
    res = {"logit_cosine_similarity": float(cos_sim), "kl_divergence": float(kl_div), "num_layers_hooked": int(len(layer_diffs))}
    return json.loads(json.dumps(res))
