"""Keep PromptKD's optimizer while explicitly grouping new parameters."""
import math
from .efficient_adaptation import adaptation_parameter


def adaptation_param_groups(model, cfg):
    multiplier = float(cfg.TRAINER.PROMPTKD.ADAPTATION.LR_MULT)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Adaptation LR multiplier must be finite and positive.")
    if cfg.OPTIM.STAGED_LR:
        raise ValueError("Use ADAPTATION.LR_MULT, not OPTIM.STAGED_LR.")
    original, added = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (added if adaptation_parameter(name) else original).append(parameter)
    if not added and multiplier != 1.:
        raise ValueError("A baseline has no adaptation LR group; multiplier must be 1.")
    groups = [{"params": original, "lr": cfg.OPTIM.LR,
               "name": "promptkd", "warmup_factor": 1.}]
    if added:
        groups.append({"params": added, "lr": cfg.OPTIM.LR * multiplier,
                       "name": "adaptation", "warmup_factor": multiplier})
    return groups
