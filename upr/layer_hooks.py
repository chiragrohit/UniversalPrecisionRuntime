import torch
import torch.nn as nn
from typing import Dict, List, Any
from .numerical import compute_numerical_metrics

class LayerActivationCollector:
    """
    Fix 3 — Layer Comparison module.
    Attaches forward hooks to every Transformer block to collect:
    - input activation
    - output activation
    - hidden state
    - attention output
    - MLP output
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[Any] = []
        self.activations: Dict[str, torch.Tensor] = {}

    def register_hooks(self) -> None:
        self.clear()
        for name, module in self.model.named_modules():
            # Identify transformer blocks, attention, MLP, and layer norms
            if any(k in name for k in ["layers.", "block", "h.", "self_attn", "mlp"]):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook(module, input_tensor, output_tensor):
            inp = input_tensor[0] if isinstance(input_tensor, tuple) and len(input_tensor) > 0 else input_tensor
            out = output_tensor[0] if isinstance(output_tensor, tuple) and len(output_tensor) > 0 else output_tensor
            
            if isinstance(inp, torch.Tensor):
                self.activations[f"{layer_name}.input"] = inp.detach().cpu()
            if isinstance(out, torch.Tensor):
                self.activations[f"{layer_name}.output"] = out.detach().cpu()
        return hook

    def clear(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

def compare_layer_activations(orig_collector: LayerActivationCollector, recon_collector: LayerActivationCollector) -> Dict[str, Any]:
    """
    Fix 3 — Compares original vs BitPlane layer-wise activations and stores metrics.
    """
    layer_metrics = {}
    common_keys = set(orig_collector.activations.keys()).intersection(set(recon_collector.activations.keys()))
    
    for key in sorted(common_keys):
        orig_act = orig_collector.activations[key]
        recon_act = recon_collector.activations[key]
        layer_metrics[key] = compute_numerical_metrics(orig_act, recon_act)
        
    return layer_metrics
