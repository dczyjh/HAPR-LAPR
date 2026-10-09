"""Portable wiring for the preserved LARP scientific implementation.

This adapter constructs the actual PromptKD trainer. The process supervisor is
responsible for output ownership, pairing, four-step gates, training/evaluation
process separation and checkpoint audits. No historical run/log directory is
required. Archive sources and their numerical functions are never modified.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "archive/project"
DATASET_DIRECTORIES = {
    "dtd": "dtd", "oxford_pets": "oxford_pets", "fgvc_aircraft": "fgvc_aircraft",
    "oxford_flowers": "oxford_flowers", "caltech101": "caltech-101",
    "stanford_cars": "stanford_cars", "ucf101": "ucf101", "eurosat": "eurosat",
    "sun397": "sun397", "food101": "food-101",
}
TOLERANCES = {"forward": {"atol": .002, "rtol": .002},
              "gradient": {"atol": .0001, "rtol": .01},
              "update": {"atol": .00001, "rtol": .001}}


def _json(name):
    return json.loads((ROOT / name).read_text())


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_checks():
    """Verify every frozen scientific entry plus the original schedule body."""
    inventory = _json("provenance/source_inventory.json")
    failures = []
    for row in inventory["files"] + inventory["workers"] + inventory["supplementary_assets"]:
        path = ROOT / row["release_path"]
        if path.is_symlink() or not path.is_file() or _sha(path) != row["sha256"]:
            failures.append(row["release_path"])
    source = ROOT / "protocol/original_sgd.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install")
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if digest != inventory["extracted_function"]["ast_sha256"]:
        failures.append("protocol/original_sgd.py:install")
    expected_schedule = _json("provenance/final_scientific_manifest.json")[
        "larp_pets_projector_ramp_20260929_setup2/projector_schedule.py"]
    if _sha(ROOT / "protocol/projector_schedule.py") != expected_schedule:
        failures.append("protocol/projector_schedule.py")
    # A normalized template must match every original contract field except
    # the four declared external path substitutions.
    original = _json("archive/source_contract.json.txt")
    templates = _json("protocol/resolved_templates.json")
    fields = _json("protocol/protocol.json")["path_substitutions"]
    for dataset, roles in templates.items():
        for role, actual in roles.items():
            expected = copy.deepcopy(original["configs"][dataset][role])
            for dotted, value in fields.items():
                keys = dotted.split(".")
                parent = expected
                for key in keys[:-1]:
                    parent = parent[key]
                parent[keys[-1]] = value
            if actual != expected:
                failures.append(f"protocol/resolved_templates.json:{dataset}/{role}")
    return {"status": "passed" if not failures else "failed", "failures": failures,
            "frozen_scientific_sources": len(inventory["files"])}


def resolve_config(dataset, seed, variant, paths):
    """Resolve the full frozen dictionary; paths is a dict or Namespace.

    Required keys: data_root, clip_root, teacher_root, and output_root. An
    explicit output_dir overrides the computed dataset/seed/variant directory,
    allowing the supervisor to isolate gate and formal outputs.
    """
    paths = dict(paths) if isinstance(paths, dict) else vars(paths)
    variants = _json("protocol/variants.json")
    templates = _json("protocol/resolved_templates.json")
    if dataset not in templates or variant not in variants or type(seed) is not int or seed not in (1, 2, 3):
        raise ValueError("Unsupported dataset, variant or seed in the fixed LARP protocol")
    resolved = {}
    for key in ("data_root", "clip_root", "teacher_root"):
        value = paths.get(key)
        if value is None or not str(value).strip() or "${" in str(value):
            raise ValueError(f"An explicit {key} is required")
        resolved[key] = str(Path(value).expanduser().resolve())
    if paths.get("output_dir"):
        output = Path(paths["output_dir"])
    elif paths.get("output_root"):
        output = Path(paths["output_root"]) / dataset / f"seed_{seed}" / variant
    else:
        raise ValueError("An explicit output_root or output_dir is required")
    if "${" in str(output):
        raise ValueError("Output path contains an unresolved placeholder")
    cfg = copy.deepcopy(templates[dataset]["r1" if variant == "r1" else "o4"])
    cfg["DATASET"]["ROOT"] = resolved["data_root"]
    cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"] = resolved["clip_root"]
    cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"] = resolved["teacher_root"]
    cfg["OUTPUT_DIR"] = str(output.expanduser().resolve())
    cfg["SEED"] = seed
    if variant != "r1":
        spec = variants[variant]
        cfg["TRAINER"]["PROMPTKD"]["ADAPTATION"].update(LORA_R=spec["rank"], LAYERS=spec["layers"])
    return cfg


def config_identity(cfg):
    """Reject arbitrary scientific overrides, including unnoticed old V defaults."""
    cfg = json.loads(json.dumps(cfg))
    datasets = _json("protocol/datasets.json")
    found = [key for key, data in datasets.items() if data["dataset_class"] == cfg["DATASET"]["NAME"]]
    if len(found) != 1:
        raise ValueError("Configuration does not name a registered dataset")
    dataset = found[0]
    a = cfg["TRAINER"]["PROMPTKD"]["ADAPTATION"]
    if a["TYPE"] == "none":
        variant = "r1"
    else:
        matches = [key for key, spec in _json("protocol/variants.json").items()
                   if spec["type"] == a["TYPE"] and spec["rank"] == a["LORA_R"] and spec["layers"] == a["LAYERS"]]
        if len(matches) != 1:
            raise ValueError("Configuration does not match one registered O variant")
        variant = matches[0]
    p = cfg["TRAINER"]["PROMPTKD"]
    expected = resolve_config(dataset, cfg["SEED"], variant, {
        "data_root": cfg["DATASET"]["ROOT"], "clip_root": p["WEIGHTS_ROOT"],
        "teacher_root": p["TEACHER_ROOT"], "output_dir": cfg["OUTPUT_DIR"]})
    if cfg != expected:
        raise ValueError("Scientific configuration differs from the frozen resolved protocol")
    return dataset, cfg["SEED"], variant


def _framework():
    """Load only this route's exact source; fail on another route's imports."""
    for name in ("train", "dassl", "datasets", "trainers", "clip", "tools"):
        module = sys.modules.get(name)
        if module is not None:
            filename = getattr(module, "__file__", None)
            if not filename or not Path(filename).resolve().is_relative_to(PROJECT):
                raise RuntimeError(f"{name} is already imported from another project; use a fresh process")
    for path in reversed((PROJECT / "Dassl.pytorch", PROJECT)):
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


def _runtime(torch, device):
    if device not in ("cuda", "cuda:0"):
        raise ValueError("The preserved FP16 scientific path requires CUDA; no implicit CPU/FP32 fallback")
    if platform.system() != "Linux" or platform.python_version() != "3.10.8":
        raise RuntimeError("The portable model execution environment requires Linux and Python 3.10.8")
    import torchvision
    import numpy
    import PIL
    if torch.__version__ != "2.0.1+cu118" or torchvision.__version__ != "0.15.2+cu118" or torch.version.cuda != "11.8":
        raise RuntimeError("Expected torch 2.0.1+cu118, torchvision 0.15.2+cu118 and CUDA 11.8")
    if numpy.__version__ != "1.24.4" or PIL.__version__ != "9.5.0":
        raise RuntimeError("Expected numpy 1.24.4 and Pillow 9.5.0 for original input decoding")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("The historical engineering protocol targets NVIDIA GeForce RTX 4090")


def validate_assets(cfg, dataset):
    """Check the actual assets before any weight deserialization/model build."""
    spec = _json("protocol/datasets.json")[dataset]
    data_root = Path(cfg["DATASET"]["ROOT"])
    directory = data_root / DATASET_DIRECTORIES[dataset]
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
    # Never let the upstream reader create a new random split silently.
    if dataset != "fgvc_aircraft":
        split = directory / f"split_zhou_{spec['dataset_class']}.json"
        if not split.is_file():
            raise FileNotFoundError(f"Existing frozen dataset split required: {split}")
    p = cfg["TRAINER"]["PROMPTKD"]
    assets = {}
    for name, expected in spec["assets_sha256"].items():
        asset = (Path(p["TEACHER_ROOT"]) / spec["dataset_class"] / "VLPromptLearner/model-best.pth.tar"
                 if name == "teacher" else Path(p["WEIGHTS_ROOT"]) / name)
        if not asset.is_file() or _sha(asset) != expected:
            raise RuntimeError(f"Missing or different frozen model asset: {asset}")
        assets[name] = expected
    return assets


def validate_reader(tr, dataset):
    """Validate counts, class order and material existence; return portable IDs.

    Exact split-file SHA verification is a separate supervisor preflight. The
    normalized membership hash is checked against explicitly sourced shared
    dataset evidence; six records also have LARP-native identity cross-checks.
    """
    spec = _json("protocol/datasets.json")[dataset]
    data = tr.dm.dataset
    groups = (data.train_x, data.val, data.test)
    counts = [len(group) for group in groups]
    if counts != spec["counts_train_base_novel"] or list(data.classnames) != spec["classnames"]:
        raise RuntimeError("Dataset counts or class order differs from the frozen protocol")
    root = Path(tr.cfg.DATASET.ROOT).resolve()
    records = []
    for group in groups:
        rows = []
        for item in group:
            path = Path(item.impath)
            if not path.is_file():
                raise FileNotFoundError(f"Dataset reader references a missing image: {path}")
            # Keep the logical path relative to the requested root; datasets may
            # legitimately consist of read-only symlinks to external images.
            logical = Path(path.absolute()).relative_to(root).as_posix()
            rows.append([logical, int(item.label), item.classname])
        records.append(rows)
    if len(tr.train_loader_x) != spec["batches_per_epoch"]:
        raise RuntimeError("Actual train_loader_x batch count differs from frozen protocol")
    reference = _json("dataset_reference.json")["records"][dataset]
    identity = getattr(tr, "run_identity", {})
    if identity.get("split_sha256") != reference["split_sha256"] or identity.get("classnames") != spec["classnames"]:
        raise RuntimeError("Actual ordered split membership differs from the fixed-material reference")
    return {"counts": counts, "classnames": list(data.classnames),
            "portable_reader_records_sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
            "split_sha256": identity["split_sha256"], "reference_basis": reference["basis"],
            "ordered_membership_verified": True, "image_bytes_verified": False,
            "metadata_file_sha_preflight_responsibility": "common supervisor before reader construction"}


def added(name):
    return any(value in name for value in ("harp_adapter", "lora_A_", "lora_B_"))


def canonical(name):
    return name.replace(".parametrizations.in_proj_weight.original", ".in_proj_weight").replace(
        ".out_proj.parametrizations.weight.original", ".out_proj.weight")


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def common_hashes(model, frozen_only=False):
    return {canonical(name): tensor_hash(parameter) for name, parameter in model.named_parameters()
            if not added(name) and (not frozen_only or not parameter.requires_grad)}


def backend(torch=None):
    if torch is None:
        import torch
    return {"benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled()}


def capture_initial(tr):
    """A torch.save-compatible pairing record captured before any iterator."""
    import torch
    dataset, seed, variant = config_identity(tr.cfg)
    return {"dataset": dataset, "seed": seed, "variant": variant, "source_variant": variant,
            "common": common_hashes(tr.model),
            "added": {n: p.detach().cpu().clone() for n, p in tr.model.named_parameters() if added(n)},
            "frozen_hashes": common_hashes(tr.model, True),
            "rng": {"cpu": tensor_hash(torch.get_rng_state()),
                    "cuda": tensor_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}}


def apply_initial_reference(tr, reference, *, dataset, seed, variant):
    """Validate natural common initialization; map only declared rank2 A/B."""
    import torch
    if reference.get("dataset") != dataset or reference.get("seed") != seed:
        raise RuntimeError("Initial reference has a different dataset or seed")
    params = {n: p for n, p in tr.model.named_parameters() if p.requires_grad}
    expected = reference.get("common", {})
    if common_hashes(tr.model) != expected:
        raise RuntimeError("Natural common initialization differs; no silent common-weight replacement")
    before_cpu = torch.get_rng_state().clone()
    before_cuda = [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    rng = reference.get("rng")
    current_rng = {"cpu": tensor_hash(before_cpu), "cuda": tensor_hash(before_cuda[0]) if before_cuda else None}
    if rng != current_rng:
        raise RuntimeError("Post-build CPU/CUDA RNG differs from paired reference")
    source_variant = reference.get("source_variant", reference.get("variant"))
    live_added = {n: p for n, p in params.items() if added(n)}
    if variant == "o3-r2" and source_variant not in ("o4-r2", "o3-r2"):
        raise RuntimeError("Last-three-layer rank2 initialization requires the same-seed O4/rank2 anchor")
    if variant in ("o4-r2", "o3-r2") and source_variant in ("o4-r2", "o3-r2"):
        ref_added = reference.get("added", {})
        for name, parameter in live_added.items():
            value = ref_added.get(name)
            if value is None or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise RuntimeError(f"Missing, incompatible or nonfinite physical-layer initializer: {name}")
            if "lora_B_o" in name and torch.count_nonzero(value):
                raise RuntimeError("B must be zero in an initialization reference")
        with torch.no_grad():
            for name, parameter in live_added.items():
                parameter.copy_(ref_added[name].to(parameter))
    elif source_variant == variant and variant != "r1":
        expected_added = reference.get("added", {})
        if live_added.keys() != expected_added.keys() or any(not torch.equal(p.detach().cpu(), expected_added[n]) for n, p in live_added.items()):
            raise RuntimeError("The same-rank initializer differs from its gate; rank1/4 are not resampled")
    if not torch.equal(before_cpu, torch.get_rng_state()) or any(not torch.equal(a, b) for a, b in zip(
            before_cuda, torch.cuda.get_rng_state_all() if before_cuda else [])):
        raise RuntimeError("Initialization mapping unexpectedly consumed RNG")


def build(cfg, *, device="cuda", init_reference=None):
    """Build the real frozen trainer with supplied assets, never a placeholder."""
    dataset, seed, variant = config_identity(cfg)
    checks = source_checks()
    if checks["status"] != "passed":
        raise RuntimeError(f"Source check failed: {checks['failures']}")
    if variant == "o3-r2" and init_reference is None:
        raise RuntimeError("Build the same-seed O4/rank2 gate anchor before the last-three-layer candidate")
    import torch
    _runtime(torch, device)
    asset_hashes = validate_assets(cfg, dataset)
    framework = _framework()
    framework.set_random_seed(seed)
    framework.train.configure_cuda_reproducibility(False)
    torch.set_num_threads(4)
    config = framework.get_cfg_default()
    framework.train.extend_cfg(config)
    config.merge_from_other_cfg(framework.CfgNode(copy.deepcopy(cfg)))
    config.freeze()
    framework.setup_logger(config.OUTPUT_DIR)
    tr = framework.build_trainer(config)
    if backend(torch) != _json("protocol/protocol.json")["backend"]:
        raise RuntimeError("LARP requires the exact declared CUDA backend, including efficient=True")
    if hasattr(tr, "boundary_reference") or tr.reference_encoder is not None or tr.scaler is not None:
        raise RuntimeError("Unexpected extra objective, representation branch or mixed-precision scaler")
    expected_count = 1013760 + _json("protocol/variants.json")[variant]["added_parameters"]
    if sum(p.numel() for p in tr.model.parameters() if p.requires_grad) != expected_count:
        raise RuntimeError("Trainable parameter count differs from the declared variant")
    for name, parameter in tr.model.named_parameters():
        if added(name):
            if ".out_proj.parametrizations.weight.0.lora_" not in name or parameter.dtype != torch.float16 or not parameter.requires_grad:
                raise RuntimeError("Adaptation is not the frozen FP16 O-projection A/B path")
            if "lora_B_o" in name and torch.count_nonzero(parameter):
                raise RuntimeError("New LoRA B must start at zero")
    if any(p.requires_grad for p in tr.model_teacher.parameters()) or tr.model_teacher.training:
        raise RuntimeError("Teacher must be frozen and in eval mode")
    if init_reference is not None:
        apply_initial_reference(tr, init_reference, dataset=dataset, seed=seed, variant=variant)
    tr.release_dataset = dataset
    tr.release_seed = seed
    tr.release_variant = variant
    tr.release_assets = asset_hashes
    tr.release_reader = validate_reader(tr, dataset)
    return tr


def install_schedule(tr, variant, count):
    """Install the byte/AST-checked original FP16 SGD/projector-ramp wrapper."""
    if variant not in _json("protocol/variants.json") or not isinstance(count, int) or count <= 1:
        raise ValueError("Invalid frozen variant or batch count")
    if getattr(tr, "release_schedule_installed", False):
        raise RuntimeError("The original schedule must only be installed once per trainer")
    if hasattr(tr, "train_loader_x") and len(tr.train_loader_x) != count:
        raise ValueError("Schedule batch count differs from the actual train_loader_x")
    if hasattr(tr, "cfg") and config_identity(tr.cfg)[2] != variant:
        raise ValueError("Schedule variant differs from the trainer's frozen configuration")
    import torch
    schedule = _load("larp_portable_projector_schedule", ROOT / "protocol/projector_schedule.py")
    worker = _load(f"larp_portable_original_sgd_{id(tr)}", ROOT / "protocol/original_sgd.py")
    worker.q = SimpleNamespace(added=added)
    worker.f = SimpleNamespace(ps=schedule, COUNT=count)
    worker.install(torch, tr, "r1" if variant == "r1" else "larp")
    tr.release_schedule_installed = True
    return tr


def evaluation_helpers():
    """Import the exact frozen attach/compare implementation without its worker."""
    path = ROOT / "archive/helpers/larp_two_dataset_ramp_20260929/evaluation_evidence.py.txt"
    namespace = {"__name__": "larp_portable_evaluation_evidence"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return SimpleNamespace(attach=namespace["attach"], compare=namespace["compare"])


def evidence_module():
    """Common-supervisor API alias for the exact original evidence functions."""
    return evaluation_helpers()


def compare_first(a, b, label):
    """Exact retained first-step comparison, with legacy element diagnostics."""
    import math
    import torch
    worker = ROOT / "archive/final_worker/retained_worker.py.txt"
    node = next(n for n in ast.parse(worker.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "compare_first")
    helper = ROOT / "archive/helpers/larp_research_20260928_fix1/research_queue.py.txt"
    reference = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "compare_tensor")
    namespace = {"torch": torch, "math": math, "TOLERANCES": TOLERANCES}
    exec(compile(ast.Module(body=[reference], type_ignores=[]), str(helper), "exec"), namespace)
    namespace["q"] = SimpleNamespace(compare_tensor=namespace["compare_tensor"])
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(worker), "exec"), namespace)
    return namespace["compare_first"](a, b, label)
