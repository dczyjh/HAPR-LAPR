"""Official PromptKD baseline, with paths relocated in an isolated file view.

The official source, optimizer and forward_backward are unchanged. The file view
supplies the two directories which its original loaders address relative to cwd.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT.parent / "larp/archive/official_reference"


def _helper():
    name = "r0_material_helpers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT.parent / "larp/portable.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def resolve_config(dataset, seed, variant, paths):
    if variant != "r0" or seed not in (1,2,3):
        raise ValueError("R0 has only the official baseline variant and seeds 1/2/3")
    templates = json.loads((ROOT / "templates.json").read_text())
    if dataset not in templates:
        raise ValueError(dataset)
    paths = paths if isinstance(paths, dict) else vars(paths)
    cfg = copy.deepcopy(templates[dataset])
    cfg["DATASET"]["ROOT"] = str(Path(paths["data_root"]).expanduser().resolve())
    output = paths.get("output_dir") or paths.get("output")
    if output is None and paths.get("output_root") is not None:
        output = Path(paths["output_root"]) / dataset / f"seed_{seed}" / variant
    if output is None:
        raise ValueError("output_dir (phase directory) or output_root is required")
    cfg["OUTPUT_DIR"] = str(Path(output).expanduser().resolve())
    cfg["SEED"] = seed
    # Paths for the official relative loader belong to the execution registration,
    # not to its scientific configuration. Carry them outside the trainer cfg.
    return {"official_cfg": cfg, "clip_root": str(Path(paths["clip_root"]).expanduser().resolve()),
            "teacher_root": str(Path(paths["teacher_root"]).expanduser().resolve()), "dataset_key": dataset}


def config_identity(resolved):
    """Reject all scientific overrides; normalize Yacs tuple/list roundtrips."""
    resolved = json.loads(json.dumps(resolved))
    cfg = resolved["official_cfg"]
    dataset = resolved["dataset_key"]
    paths = {"data_root": cfg["DATASET"]["ROOT"], "clip_root": resolved["clip_root"],
             "teacher_root": resolved["teacher_root"], "output_dir": cfg["OUTPUT_DIR"]}
    if resolve_config(dataset, cfg["SEED"], "r0", paths) != resolved:
        raise ValueError("Official scientific config differs from the frozen resolved protocol")
    return dataset, cfg["SEED"], "r0"


def source_checks():
    return _helper().source_checks()


def data_identity(cfg, dataset):
    """Hash actual ordered reader rows using the shared original identity rule.

    This is metadata only, not a new reader or a claim of whole-image byte
    identity. Logical relative paths intentionally preserve external image
    symlinks; the supervisor separately checks original split-file bytes.
    """
    membership = {}
    for split in ("train_x", "val", "test"):
        items = getattr(dataset, split, []) or []
        membership[split] = [
            (os.path.relpath(os.path.abspath(item.impath), os.path.abspath(cfg.DATASET.ROOT)),
             item.label, item.classname) for item in items]
    return {"dataset": cfg.DATASET.NAME,
            "split_sha256": hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest(),
            "classnames": list(dataset.classnames), "image_bytes_verified": False}


def _framework(view):
    """Import the exact official source in a fresh process (also CPU-testable)."""
    view = Path(view).resolve()
    for name in ("train", "dassl", "datasets", "trainers", "clip", "tools"):
        module = sys.modules.get(name)
        if module is not None:
            filename = getattr(module, "__file__", None)
            if not filename or not Path(filename).resolve().is_relative_to(view):
                raise RuntimeError(f"{name} already imported from another source; use a fresh official-baseline process")
    for path in reversed((view / "Dassl.pytorch", view)):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    train = importlib.import_module("train")
    from dassl.config import get_cfg_default
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed, setup_logger
    from yacs.config import CfgNode
    return SimpleNamespace(train=train, get_cfg_default=get_cfg_default, build_trainer=build_trainer,
                           set_random_seed=set_random_seed, setup_logger=setup_logger, CfgNode=CfgNode)


def _make_cfg(framework, resolved):
    config_identity(resolved)
    cfg = framework.get_cfg_default()
    framework.train.extend_cfg(cfg)
    cfg.merge_from_other_cfg(framework.CfgNode(resolved["official_cfg"]))
    cfg.freeze()
    # The official defaults must not silently add/change scientific settings.
    config_identity(dict(resolved, official_cfg=cfg))
    return cfg


def _create_view(resolved):
    """Copy code only; external assets stay symlinked and outside the release."""
    view = Path(resolved["official_cfg"]["OUTPUT_DIR"]) / "official_framework"
    shutil.copytree(OFFICIAL, view)
    for name in ("ViT-B-16.pt", "ViT-L-14.pt"):
        (view / "clip" / name).symlink_to(Path(resolved["clip_root"]) / name)
    (view / "teacher_model").symlink_to(resolved["teacher_root"], target_is_directory=True)
    return view


def added(name):
    return False


def canonical(name):
    return name


def tensor_hash(value):
    return _helper().tensor_hash(value)


def capture_initial(tr):
    import torch
    return {"dataset": tr.release_dataset, "seed": tr.cfg.SEED, "source_variant": "r0", "variant": "r0",
            "common": {n: tensor_hash(p) for n,p in tr.model.named_parameters()}, "added": {},
            "frozen_hashes": {n: tensor_hash(p) for n,p in tr.model.named_parameters() if not p.requires_grad},
            "rng": {"cpu": tensor_hash(torch.get_rng_state()), "cuda": tensor_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}}


def apply_initial_reference(tr, reference):
    """Validate natural official initialization without overwriting any tensor."""
    current = capture_initial(tr)
    for key in ("dataset", "seed", "variant", "source_variant", "common", "rng", "added"):
        if current[key] != reference.get(key):
            raise ValueError(f"Official initialization differs from registered initialization: {key}")


def build(resolved, *, device="cuda", init_reference=None):
    import torch
    helper = _helper()
    helper._runtime(torch, device)
    assert source_checks()["status"] == "passed", "Official frozen source mismatch"
    dataset, seed, _ = config_identity(resolved)
    cfg_dict = resolved["official_cfg"]
    inspection = copy.deepcopy(cfg_dict)
    inspection["TRAINER"]["PROMPTKD"].update(WEIGHTS_ROOT=resolved["clip_root"], TEACHER_ROOT=resolved["teacher_root"])
    assets = helper.validate_assets(inspection, dataset)
    output = Path(cfg_dict["OUTPUT_DIR"])
    for name in ("train", "dassl", "datasets", "trainers", "clip", "tools"):
        if name in sys.modules:
            raise RuntimeError(f"{name} already imported; use a fresh official-baseline process")
    view = _create_view(resolved)
    os.chdir(view)
    framework = _framework(view)
    framework.set_random_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    cfg = _make_cfg(framework, resolved)
    framework.setup_logger(str(output))
    tr = framework.build_trainer(cfg)
    tr.release_dataset, tr.release_variant = dataset, "r0"
    tr.release_seed, tr.release_assets = seed, assets
    tr.run_identity = data_identity(cfg, tr.dm.dataset)
    tr.release_reader = helper.validate_reader(tr, dataset)
    assert sum(p.numel() for p in tr.model.parameters() if p.requires_grad) == 1013760
    assert len(tr.optim.param_groups) == 1 and tr.scaler is None
    if init_reference is not None:
        apply_initial_reference(tr, init_reference)
    return tr


def install_schedule(tr, variant, count):
    """Record the unmodified official SGD step; no projector ramp is installed."""
    if variant != "r0" or count != len(tr.train_loader_x) or getattr(tr, "release_schedule_installed", False):
        raise ValueError("Invalid/repeated official optimizer attachment")
    import torch
    if not isinstance(tr.optim, torch.optim.SGD) or len(tr.optim.param_groups) != 1:
        raise ValueError("The preserved official optimizer is single-group SGD")
    original = tr.optim.step
    def step(*args, **kwargs):
        rates = [group["lr"] for group in tr.optim.param_groups]
        result = original(*args, **kwargs)
        tr.actual_step = {"epoch": tr.epoch + 1, "batch": tr.batch_idx + 1, "scheduled_lr": rates,
                          "prompt_lr": rates[0], "projector_lr": rates[0], "added_lr": None, "added_held": False}
        return result
    tr.optim.step = step
    tr.release_schedule_installed = True


def evidence_module():
    return _helper().evidence_module()
