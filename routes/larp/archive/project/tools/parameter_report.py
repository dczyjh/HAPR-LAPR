"""Count actual ViT-B/16 student trainable parameters; no weights/data required."""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dassl.config import get_cfg_default
from train import extend_cfg
from clip.model import VisionTransformer
from trainers.promptkd import CustomCLIP
from trainers.efficient_adaptation import adaptation_parameter


def main():
    for method in ("none", "harp", "larp"):
        cfg = get_cfg_default()
        extend_cfg(cfg)
        cfg.TRAINER.PROMPTKD.ADAPTATION.TYPE = method
        visual = VisionTransformer(224, 16, 768, 12, 12, 512,
                                   {"trainer": "IVLP", "vision_depth": 9,
                                    "vision_ctx": 4, "language_ctx": 4})
        backbone = SimpleNamespace(visual=visual, dtype=torch.float32,
                                   logit_scale=torch.nn.Parameter(torch.tensor(1.)))
        model = CustomCLIP(cfg, ["a", "b"], backbone)
        counts = {"VPT": 0, "projector": 0, "adaptation": 0}
        for name, parameter in model.named_parameters():
            if adaptation_parameter(name):
                counts["adaptation"] += parameter.numel()
            elif "VPT_image_trans" in name:
                counts["projector"] += parameter.numel()
            elif "VPT" in name:
                counts["VPT"] += parameter.numel()
        print(method, counts, "total_trainable=", sum(counts.values()))
        del model, backbone, visual
        gc.collect()


if __name__ == "__main__":
    main()
