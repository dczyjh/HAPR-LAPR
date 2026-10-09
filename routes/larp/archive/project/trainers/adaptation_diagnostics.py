"""Read-only training-probe diagnostics. These are NOT accuracy evidence."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import torch
from torch.cuda.amp import autocast
from .efficient_adaptation import (
    HighLevelAdapter, QKVLoRAParametrization, WeightLoRAParametrization, adaptation_parameter,
)

ADAPTER_TYPES = (HighLevelAdapter, QKVLoRAParametrization, WeightLoRAParametrization)


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def norm(tensors):
    terms = [tensor.detach().float().norm().square() for tensor in tensors]
    return torch.stack(terms).sum().sqrt().item() if terms else 0.


@contextmanager
def adaptation_disabled(model):
    modules = [(module, getattr(module, "diagnostic_disabled", False))
               for module in model.modules() if isinstance(module, ADAPTER_TYPES)]
    try:
        for module, _ in modules:
            module.diagnostic_disabled = True
        yield
    finally:
        for module, previous in modules:
            module.diagnostic_disabled = previous


@torch.no_grad()
def lora_weight_diagnostics(model, precision):
    records = {}
    for name, module in model.named_modules():
        if not hasattr(module, "parametrizations"):
            continue
        for key, chain in module.parametrizations.items():
            adapter = chain[0]
            if not isinstance(adapter, (QKVLoRAParametrization, WeightLoRAParametrization)):
                continue
            base = chain.original.detach()
            if isinstance(adapter, QKVLoRAParametrization):
                deltas = [torch.zeros_like(part, dtype=torch.float32)
                          for part in base.split(adapter.embed_dim)]
                for target in adapter.targets:
                    a = getattr(adapter, "lora_A_" + target).float()
                    b = getattr(adapter, "lora_B_" + target).float()
                    deltas[adapter._TARGET_TO_INDEX[target]] = (b @ a) * adapter.scaling
                delta = torch.cat(deltas)
            else:
                delta = (adapter.lora_B_o.float() @ adapter.lora_A_o.float()) * adapter.scaling
            # Reproduce parametrization under the actual training autocast.
            with autocast(enabled=precision == "amp"):
                effective = chain().detach()
            compute_dtype = torch.float16 if precision in ("amp", "fp16") else torch.float32
            # This models the projection weight cast, not every kernel's
            # rounding. Small lost elements alone do not prove lost accuracy.
            actual = effective.to(compute_dtype).float() - base.to(compute_dtype).float()
            active = delta != 0
            records[f"{name}.{key}"] = {
                "ideal_delta_to_weight_norm": finite_number(delta.norm() / base.float().norm().clamp_min(1e-12)),
                "compute_delta_to_weight_norm": finite_number(actual.norm() / base.float().norm().clamp_min(1e-12)),
                "surviving_nonzero_fraction": finite_number((actual[active] != 0).float().mean()) if active.any() else None,
                "relative_rounding_error": finite_number((actual - delta).norm() / delta.norm()) if active.any() else None,
                "weight_cast_dtype": str(compute_dtype),
            }
    return records


@torch.no_grad()
def probe_contribution(model, image, text, precision):
    """Same current checkpoint on/off, never an independently trained baseline."""
    states = [(module, module.training) for module in model.modules()]
    handles, residuals, block_norms = [], {}, {}
    device_ids = [image.device.index] if image.is_cuda else []
    try:
        model.eval()
        for name, module in model.named_modules():
            if isinstance(getattr(module, "harp_adapter", None), HighLevelAdapter):
                handles.append(module.harp_adapter.register_forward_hook(
                    lambda m, x, y, key=name: residuals.update({key: y.detach().float().norm()})))
                handles.append(module.register_forward_hook(
                    lambda m, x, y, key=name: block_norms.update({key: y.detach().float().norm()})))
        with torch.random.fork_rng(devices=device_ids), autocast(enabled=precision == "amp"):
            features, scale = model(image)
            logits = (scale * features @ text.t()).float()
            for handle in handles:
                handle.remove()
            handles.clear()
            with adaptation_disabled(model):
                off_features, off_scale = model(image)
                off_logits = (off_scale * off_features @ text.t()).float()
        return {
            "probe_size": len(image), "probe_mode": "eval_on_fixed_training_images",
            "probe_precision": precision,
            "mean_abs_logit_change": finite_number((logits - off_logits).abs().mean()),
            "max_abs_logit_change": finite_number((logits - off_logits).abs().max()),
            "argmax_disagreement_fraction": finite_number((logits.argmax(1) != off_logits.argmax(1)).float().mean()),
            "harp_residual_to_block_output_norm": {
                key: finite_number(value / block_norms[key].clamp_min(1e-12))
                for key, value in residuals.items()
            },
        }
    finally:
        for handle in handles:
            handle.remove()
        for module, training in states:
            module.training = training


class AdaptationDiagnostics:
    def __init__(self, model, cfg, output_dir):
        self.model = model
        self.precision = cfg.TRAINER.PROMPTKD.PREC
        self.interval = cfg.TRAINER.PROMPTKD.DIAGNOSTICS.INTERVAL
        self.probe_size = cfg.TRAINER.PROMPTKD.DIAGNOSTICS.PROBE_SIZE
        self.path = Path(output_dir) / "diagnostics.jsonl"
        self.probe = None
        self.before = None

    def due(self, step):
        return self.interval > 0 and (step == 1 or step % self.interval == 0)

    def before_step(self, step, image, optimizer):
        if not self.due(step):
            return
        if self.probe is None:
            self.probe = image[:self.probe_size].detach().clone()
        self.before = {name: parameter.detach().clone()
                       for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.record = {"step": step, "source": "training_probe", "not_accuracy_evidence": True,
                       "optimizer_groups": [], "groups": {}}
        for group in optimizer.param_groups:
            self.record["optimizer_groups"].append({"name": group.get("name", "unnamed"), "lr": group["lr"]})
        for group in ("promptkd", "adaptation"):
            parameters = [p for name, p in self.model.named_parameters()
                          if p.requires_grad and adaptation_parameter(name) == (group == "adaptation")]
            self.record["groups"][group] = {
                "parameter_norm": finite_number(norm(parameters)),
                "gradient_norm": finite_number(norm([p.grad for p in parameters if p.grad is not None])),
                "has_gradient": any(p.grad is not None for p in parameters),
            }

    def after_step(self, step, text, skipped=False):
        if not self.due(step):
            return
        self.record["optimizer_step_skipped"] = bool(skipped)
        for group in ("promptkd", "adaptation"):
            pairs = [(p.detach(), self.before[name]) for name, p in self.model.named_parameters()
                     if p.requires_grad and adaptation_parameter(name) == (group == "adaptation")]
            denominator = self.record["groups"][group]["parameter_norm"]
            update = norm([after.float() - before.float() for after, before in pairs])
            self.record["groups"][group]["update_norm"] = finite_number(update)
            self.record["groups"][group]["relative_update_norm"] = (
                finite_number(update / max(denominator, 1e-12)) if denominator is not None else None)
        self.record["contribution"] = probe_contribution(self.model, self.probe, text, self.precision)
        self.record["lora_weights"] = lora_weight_diagnostics(self.model, self.precision)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(self.record, allow_nan=False) + "\n")
        self.before = None
