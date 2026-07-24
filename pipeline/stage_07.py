"""
Stage 7 — Phase 1.2 Progressive 1-Bit Resolution Sweep (16..2 bits)
Evaluates perplexity, logit cosine similarity, and KL divergence across all 15 bit resolution steps (16 down to 2).
"""
import json
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
def stage_07_full_1bit_sweep():
    import torch
    import upr
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    setup_hf_auth()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_dir = get_fast_checkpoint_dir()

    print("=== Phase 1.2 Stage 7: Progressive 1-Bit Resolution Sweep (16..2 bits) ===")
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
            if end_loc - i < 64: continue
            trg_len = end_loc - i
            chunk_ids = input_ids[:, i:end_loc]
            with torch.no_grad():
                try:
                    loss = model(chunk_ids, labels=chunk_ids).loss
                    if torch.isnan(loss) or torch.isinf(loss): return 9999.0
                    nlls.append(loss * trg_len)
                except Exception: return 9999.0
        if not nlls or end_loc == 0: return 9999.0
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        val = float(ppl.item())
        return val if not (torch.isnan(ppl) or torch.isinf(ppl)) else 9999.0

    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    orig_model.eval()
    baseline_ppl = eval_ppl(orig_model)

    prompt_inputs = tokenizer("Universal Precision Runtime provides dynamic multi-precision execution.", return_tensors="pt").to(device)
    with torch.no_grad():
        orig_logits = orig_model(**prompt_inputs).logits.detach().cpu()

    del orig_model
    gc.collect()

    config = AutoConfig.from_pretrained(MODEL_ID)
    bits_range = list(range(16, 1, -1))
    sweep_1bit = []

    print(f"\n{'Bits':<6} | {'CosSim':<12} | {'KL Div':<10} | {'Perplexity':<12} | {'Delta PPL'}")
    print("-" * 65)

    for bits in bits_range:
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

        with torch.no_grad():
            recon_logits = recon_model(**prompt_inputs).logits.detach().cpu()

        cos_sim = upr.compute_cosine_similarity(orig_logits, recon_logits)
        kl_div = upr.compute_kl_divergence(orig_logits, recon_logits)
        ppl = eval_ppl(recon_model)

        res = {
            "bits": bits,
            "cosine_similarity": round(cos_sim, 6),
            "kl_divergence": round(kl_div, 6),
            "perplexity": round(ppl, 4),
            "delta_perplexity": round(ppl - baseline_ppl, 4)
        }
        sweep_1bit.append(res)
        print(f"{bits:<6} | {cos_sim:<12.6f} | {kl_div:<10.6f} | {ppl:<12.4f} | +{ppl - baseline_ppl:.4f}")

        del recon_model, recon_state_dict
        gc.collect()

    json_path = f"{RESULTS_DIR}/progressive_1bit_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_ppl": baseline_ppl, "sweep_1bit": sweep_1bit}, f, indent=2)

    vol.commit()
    return json.loads(json.dumps({"json": json_path, "sweep": sweep_1bit}))
