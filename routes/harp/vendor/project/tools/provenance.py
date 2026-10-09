"""Run identity needed to compare small accuracy improvements fairly."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torchvision


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_identity(cfg, dataset, teacher_path):
    # Hash the actual split membership/class ordering, not dataset display names.
    membership = {}
    for split in ("train_x", "val", "test"):
        items = getattr(dataset, split, []) or []
        membership[split] = [
            (os.path.relpath(os.path.abspath(item.impath), os.path.abspath(cfg.DATASET.ROOT)),
             item.label, item.classname)
            for item in items
        ]
    split_hash = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    identity = {
        "dataset": cfg.DATASET.NAME, "split_sha256": split_hash,
        "classnames": list(dataset.classnames),
        "teacher_sha256": sha256_file(teacher_path),
        "student_backbone_sha256": sha256_file(Path(cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT) / "ViT-B-16.pt"),
        "teacher_backbone_sha256": sha256_file(Path(cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT) / "ViT-L-14.pt"),
        "precision": cfg.TRAINER.PROMPTKD.PREC, "selection": cfg.TEST.FINAL_MODEL,
        "optimizer": cfg.OPTIM.dump(), "input": cfg.INPUT.dump(),
        "train_batch": cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        "temperature": cfg.TRAINER.PROMPTKD.TEMPERATURE,
        "kd_weight": cfg.TRAINER.PROMPTKD.KD_WEIGHT,
        "prompt_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
        "prompt_context": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
        "reproducible": cfg.TRAINER.PROMPTKD.REPRODUCIBLE,
        "evaluation_role": "development_base" if cfg.TRAINER.PROMPTKD.DEVELOPMENT else "benchmark",
        "diagnostics": cfg.TRAINER.PROMPTKD.DIAGNOSTICS.dump(),
        "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                        "sdp_flash": torch.backends.cuda.flash_sdp_enabled(),
                        "sdp_memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                        "sdp_math": torch.backends.cuda.math_sdp_enabled(),
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
    }
    source_root = Path(__file__).resolve().parents[1]
    identity["source_sha256"] = {
        name: sha256_file(source_root / name)
        for name in ("train.py", "clip/model.py", "trainers/promptkd.py",
                     "trainers/efficient_adaptation.py", "tools/provenance.py",
                     "trainers/adaptation_diagnostics.py", "trainers/adaptation_optim.py",
                     "trainers/development.py", "Dassl.pytorch/dassl/optim/lr_scheduler.py",
                     "Dassl.pytorch/dassl/optim/optimizer.py",
                     "Dassl.pytorch/dassl/engine/trainer.py",
                     "Dassl.pytorch/dassl/data/data_manager.py")
    }
    identity["source_sha256"].update({
        str(path.relative_to(source_root)): sha256_file(path)
        for path in sorted((source_root / "datasets").glob("*.py"))
    })
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, signature
