"""Portable wiring for HARP's frozen scientific implementation.

Only external paths are relocated. The original trainer, loss, data readers,
fix2 VJP and two SGD ramps are used unchanged. A fresh-process supervisor owns
the gates, output ownership, training, and first independent reload. This
module neither resumes training nor manufactures historical gate evidence.
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

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "vendor/project"
VARIANTS = {"r1": ([8, 9, 10, 11], 32, 0),
            "harp": ([8, 9, 10, 11], 32, 199808),
            "mid4_d32": ([4, 5, 6, 7], 32, 199808),
            "last4_d16": ([8, 9, 10, 11], 16, 101440),
            "last4_d64": ([8, 9, 10, 11], 64, 396544)}
DATASET_DIRECTORIES = {"caltech101": "caltech-101", "oxford_pets": "oxford_pets",
                       "stanford_cars": "stanford_cars", "oxford_flowers": "oxford_flowers",
                       "food101": "food-101", "fgvc_aircraft": "fgvc_aircraft",
                       "sun397": "sun397", "dtd": "dtd", "eurosat": "eurosat", "ucf101": "ucf101"}
EXPECTED_BACKEND = {"benchmark": True, "deterministic": False,
                    "deterministic_algorithms": False, "flash": True,
                    "efficient": False, "math": True}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_checks():
    """Verify frozen bytes, extracted scientific ASTs, and path-only templates."""
    failures = []
    manifest = _read(ROOT / "provenance/generated_files.json")
    for relative, entry in manifest.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or _sha(path) != entry["sha256"]:
            failures.append(relative)
    tree = ast.parse((ROOT / "runtime/frozen_install.py").read_text())
    for name, entry in _read(ROOT / "provenance/scientific_functions.json").items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        if actual != entry["ast_sha256"]:
            failures.append(f"runtime/frozen_install.py:{name}")
    # All 150 snapshots were mechanically converted from YAML to JSON. Verify
    # the entire dictionary, not just a short whitelist of hyperparameters.
    import yaml
    for row in _read(ROOT / "configs/index.json"):
        original = yaml.safe_load((ROOT / row["recorded"]).read_text())
        original["OUTPUT_DIR"] = "${OUTPUT_DIR}"
        original["DATASET"]["ROOT"] = "${DATA_ROOT}"
        original["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"] = "${CLIP_ROOT}"
        original["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"] = "${TEACHER_ROOT}"
        if original != _read(ROOT / row["portable"]):
            failures.append(row["portable"])
    return {"status": "passed" if not failures else "failed", "failures": failures,
            "frozen_scientific_sources": len(manifest), "resolved_configurations": 150}


def resolve_config(dataset, seed, variant, paths):
    """Return the full recorded configuration with exactly four path changes."""
    if dataset not in DATASET_DIRECTORIES or seed not in (1, 2, 3) or variant not in VARIANTS:
        raise ValueError("Unsupported HARP dataset, seed or variant")
    paths = dict(paths) if isinstance(paths, dict) else vars(paths)
    output = paths.get("output_dir") or paths.get("output")
    required = {"DATA_ROOT": paths.get("data_root"), "CLIP_ROOT": paths.get("clip_root"),
                "TEACHER_ROOT": paths.get("teacher_root"), "OUTPUT_DIR": output}
    if any(not value or "${" in str(value) for value in required.values()):
        raise ValueError("data_root, clip_root, teacher_root and output_dir/output must be explicit")
    resolved = {key: str(Path(value).expanduser().resolve()) for key, value in required.items()}
    row = next(row for row in _read(ROOT / "configs/index.json")
               if (row["dataset"], row["seed"], row["variant"]) == (dataset, seed, variant))
    cfg = copy.deepcopy(_read(ROOT / row["portable"]))
    cfg["DATASET"]["ROOT"] = resolved["DATA_ROOT"]
    cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"] = resolved["CLIP_ROOT"]
    cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"] = resolved["TEACHER_ROOT"]
    cfg["OUTPUT_DIR"] = resolved["OUTPUT_DIR"]
    return cfg


def config_identity(cfg):
    """Reject every undeclared scientific change, including non-key defaults."""
    if hasattr(cfg, "dump"):
        import yaml
        cfg = yaml.safe_load(cfg.dump())
    refs = _read(ROOT / "dataset_reference.json")["records"]
    dataset = next((key for key, value in refs.items() if value["dataset_name"] == cfg["DATASET"]["NAME"]), None)
    adaptation = cfg["TRAINER"]["PROMPTKD"]["ADAPTATION"]
    candidates = [name for name, (layers, width, _) in VARIANTS.items()
                  if adaptation["LAYERS"] == layers and adaptation["HARP_DIM"] == width
                  and adaptation["TYPE"] == ("none" if name == "r1" else "harp")]
    if dataset is None or len(candidates) != 1:
        raise ValueError("Configuration does not identify a frozen HARP role")
    variant, seed = candidates[0], cfg["SEED"]
    expected = resolve_config(dataset, seed, variant,
                              {"data_root": cfg["DATASET"]["ROOT"],
                               "clip_root": cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"],
                               "teacher_root": cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"],
                               "output_dir": cfg["OUTPUT_DIR"]})
    if cfg != expected:
        raise ValueError("Full configuration differs from its recorded template (only four paths may change)")
    return dataset, seed, variant


def _framework():
    # These packages use global registries: mixing HARP and LARP imports in a
    # process would be unsafe. The supervisor uses a fresh process per action.
    for prefix in ("train", "dassl", "clip", "trainers", "datasets", "tools"):
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if (name == prefix or name.startswith(prefix + ".")) and path:
                if not Path(path).resolve().is_relative_to(PROJECT):
                    raise RuntimeError(f"Foreign scientific module already imported: {name}; use a fresh process")
    for path in (PROJECT / "Dassl.pytorch", PROJECT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("train")


def _runtime(torch, device):
    if str(device) not in ("cuda", "cuda:0"):
        raise RuntimeError("Full HARP builds require CUDA; CPU tiny tests do not replace FP16 GPU gates")
    import torchvision
    import numpy
    import PIL
    reference = _read(ROOT / "environment/provenance.json")
    if platform.system() != "Linux" or platform.python_version() != reference["python"] or torch.__version__ != reference["torch"] or torchvision.__version__ != reference["torchvision"]:
        raise RuntimeError("Python/PyTorch/torchvision do not match the recorded CUDA environment")
    if torch.version.cuda != "11.8" or numpy.__version__ != "1.24.4" or PIL.__version__ != "9.5.0":
        raise RuntimeError("Expected CUDA 11.8, NumPy 1.24.4 and Pillow 9.5.0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    if torch.cuda.get_device_name(0) != reference["hardware_reference"]:
        raise RuntimeError("This frozen protocol requires the recorded RTX 4090 class")


def validate_assets(cfg, dataset):
    """Hash the actual weights every fresh build; forbid auto-created splits."""
    assets = _read(ROOT / "provenance/external_assets.json")
    values = {}
    for entry in assets["clip"]:
        path = Path(cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"]) / entry["relative_path"]
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise RuntimeError(f"CLIP asset differs: {entry['relative_path']}")
        values[entry["relative_path"]] = entry["sha256"]
    teacher = next(row for row in assets["teachers"] if row["dataset_name"] == cfg["DATASET"]["NAME"])
    path = Path(cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"]) / teacher["relative_path"]
    if not path.is_file() or _sha(path) != teacher["sha256"]:
        raise RuntimeError("Teacher checkpoint differs from the recorded asset")
    values[teacher["relative_path"]] = teacher["sha256"]
    folder = Path(cfg["DATASET"]["ROOT"]) / DATASET_DIRECTORIES[dataset]
    if not (folder / "split_fewshot").is_dir():
        raise RuntimeError(f"Prepare the empty/existing {folder / 'split_fewshot'} directory explicitly; readers may not mutate source data")
    required = (["variants.txt", "images_variant_train.txt", "images_variant_val.txt", "images_variant_test.txt"]
                if dataset == "fgvc_aircraft" else [f"split_zhou_{cfg['DATASET']['NAME']}.json"])
    if any(not (folder / name).is_file() for name in required):
        raise RuntimeError("Missing fixed dataset split; refusing the reader's random split-generation fallback")
    return values


def validate_reader(tr, dataset):
    """Check the unchanged reader's root-relative full membership identity.

    This verifies path/label/classname ordering and all referenced image paths.
    It is not a byte-for-byte image archive audit; the supervisor's separately
    frozen asset manifest provides that evidence.
    """
    reference = _read(ROOT / "dataset_reference.json")["records"][dataset]
    identity = tr.run_identity
    data = tr.dm.dataset
    counts = [len(getattr(data, split)) for split in ("train_x", "val", "test")]
    class_hash = hashlib.sha256(json.dumps(list(data.classnames)).encode()).hexdigest()
    if (identity["split_sha256"] != reference["split_sha256"] or counts != reference["counts"]
            or class_hash != reference["classnames_sha256"] or len(data.classnames) != reference["class_count"]):
        raise RuntimeError("Dataset membership/class ordering/counts differ from the recorded experiment")
    root = Path(tr.cfg.DATASET.ROOT).resolve()
    for split in ("train_x", "val", "test"):
        for item in getattr(data, split):
            # resolve() verifies availability; membership itself deliberately
            # uses the original abspath/relpath rule, not symlink resolution.
            if not Path(item.impath).is_file():
                raise RuntimeError(f"Missing image: {item.impath}")
            if not Path(item.impath).absolute().is_relative_to(root):
                raise RuntimeError("Dataset membership escapes the configured root")
    if len(tr.train_loader_x) != counts[0] // tr.cfg.DATALOADER.TRAIN_X.BATCH_SIZE:
        raise RuntimeError("Unexpected training batch count/drop-last behavior")
    return {"status": "passed", "split_sha256": identity["split_sha256"], "counts": counts,
            "classnames_sha256": class_hash, "image_bytes_verified_here": False,
            "reference_source_sha256": reference["source_sha256"]}


def canonical(name):
    return name.replace(".parametrizations.in_proj_weight.original", ".in_proj_weight")


def added(name):
    return any(value in name for value in ("harp_adapter", "lora_A_", "lora_B_"))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def common_hashes(model, frozen_only=False):
    return {canonical(name): tensor_hash(parameter) for name, parameter in model.named_parameters()
            if not added(name) and (not frozen_only or not parameter.requires_grad)}


def backend(torch=None):
    if torch is None:
        import torch
    return {"benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled()}


def capture_initial(tr):
    """Capture before creating any data-loader iterator or performing a step."""
    import torch
    dataset, seed, variant = config_identity(tr.cfg)
    return {"common": common_hashes(tr.model),
            "added": {name: parameter.detach().cpu().clone() for name, parameter in tr.model.named_parameters() if added(name)},
            "frozen_hashes": common_hashes(tr.model, True), "seed": seed, "dataset": dataset,
            "source_variant": variant, "variant": variant,
            "rng": {"cpu": tensor_hash(torch.get_rng_state()),
                    "cuda": tensor_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}}


def apply_initial_reference(tr, reference, *, dataset, seed, variant):
    """Validate natural initial state without replacing or resampling weights.

    The original mid-four experiment checked (not copied) the corresponding
    same-width initial draws at physical layers 8..11 against layers 4..7.
    Width changes preserve their natural initialization; each subsequent
    action must match its own saved initialization reference exactly.
    """
    import torch
    if reference.get("dataset") != dataset or reference.get("seed") != seed:
        raise RuntimeError("Initialization reference has a different dataset or seed")
    if common_hashes(tr.model) != reference.get("common"):
        raise RuntimeError("Natural common initialization differs; no silent replacement is permitted")
    current_rng = {"cpu": tensor_hash(torch.get_rng_state()),
                   "cuda": tensor_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}
    if current_rng != reference.get("rng"):
        raise RuntimeError("Post-build CPU/CUDA RNG differs from its initialization reference")
    source_variant = reference.get("source_variant", reference.get("variant"))
    if source_variant not in VARIANTS:
        raise RuntimeError("Unknown initialization reference role")
    allowed = {"r1": {"r1"}, "harp": {"r1", "harp"},
               "mid4_d32": {"harp", "mid4_d32"},
               "last4_d16": {"r1", "harp", "last4_d16"},
               "last4_d64": {"r1", "harp", "last4_d64"}}
    if source_variant not in allowed[variant]:
        raise RuntimeError("Unsupported cross-role initialization reference")
    live = {name: p for name, p in tr.model.named_parameters() if added(name)}
    prior = reference.get("added", {})
    if variant == "mid4_d32" and source_variant not in ("harp", "mid4_d32"):
        raise RuntimeError("Mid-four width32 requires a same-seed main HARP initialization anchor")
    if source_variant == variant or (variant == "mid4_d32" and source_variant == "harp"):
        mapped = {}
        for name, parameter in live.items():
            prior_name = name
            if variant == "mid4_d32" and source_variant == "harp":
                layer = int(name.split("resblocks.", 1)[1].split(".", 1)[0])
                prior_name = name.replace(f"resblocks.{layer}.", f"resblocks.{layer + 4}.")
            mapped[name] = prior_name
            value = prior.get(prior_name)
            if (value is None or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not torch.equal(parameter.detach().cpu(), value)):
                raise RuntimeError(f"Natural added initialization differs: {name}")
        if set(mapped.values()) != set(prior):
            raise RuntimeError("Initialization reference has unexpected added parameters")


def build(cfg, *, device="cuda", init_reference=None):
    """Construct the original full trainer; never run gates/train/eval here."""
    dataset, seed, variant = config_identity(cfg)
    check = source_checks()
    if check["status"] != "passed":
        raise RuntimeError(f"Frozen source check failed: {check['failures']}")
    if variant == "mid4_d32" and init_reference is None:
        raise RuntimeError("Prepare the same-seed main HARP initial anchor before mid4_d32")
    import torch
    _runtime(torch, device)
    assets = validate_assets(cfg, dataset)
    train = _framework()
    from yacs.config import CfgNode
    from dassl.utils import set_random_seed, setup_logger
    from dassl.engine import build_trainer
    config = train.get_cfg_default()
    train.extend_cfg(config)
    config.merge_from_other_cfg(CfgNode(copy.deepcopy(cfg)))
    config.freeze()
    config_identity(config)
    torch.set_num_threads(4)
    setup_logger(config.OUTPUT_DIR)
    set_random_seed(seed)
    train.configure_cuda_reproducibility(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    tr = build_trainer(config)
    if backend(torch) != EXPECTED_BACKEND:
        raise RuntimeError("Backend drift: HARP requires efficient=False with benchmark/Flash/math=True")
    if tr.reference_encoder is not None or tr.scaler is not None:
        raise RuntimeError("Unexpected representation branch or AMP scaler")
    layers, width, added_count = VARIANTS[variant]
    if sum(p.numel() for p in tr.model.parameters() if p.requires_grad) != 1013760 + added_count:
        raise RuntimeError("Trainable parameter budget differs from the recorded variant")
    adapters = {name: p for name, p in tr.model.named_parameters() if added(name)}
    if sum(p.numel() for p in adapters.values()) != added_count:
        raise RuntimeError("Added parameter budget differs")
    if adapters and sorted({int(n.split("resblocks.")[1].split(".")[0]) for n in adapters}) != layers:
        raise RuntimeError("Added physical layers differ")
    for name, parameter in adapters.items():
        if "harp_adapter" not in name or not parameter.requires_grad or parameter.dtype != torch.float16:
            raise RuntimeError("Unexpected added path or precision")
        if (".up." in name or name.endswith(".down.bias")) and torch.count_nonzero(parameter):
            raise RuntimeError("HARP output and down bias must have the original zero initialization")
    if any(p.requires_grad for p in tr.model_teacher.parameters()) or tr.model_teacher.training:
        raise RuntimeError("Teacher must remain frozen in eval mode")
    tr._portable_dataset, tr._portable_variant, tr._portable_reference = dataset, variant, init_reference
    tr.release_dataset, tr.release_seed, tr.release_variant = dataset, seed, variant
    tr.release_assets, tr.release_reader = assets, validate_reader(tr, dataset)
    if init_reference is not None:
        apply_initial_reference(tr, init_reference, dataset=dataset, seed=seed, variant=variant)
    return tr


def install_schedule(tr, variant, count):
    """Install the verbatim frozen dual-ramp function once, without new maths."""
    if variant not in VARIANTS or not isinstance(count, int) or count <= 1:
        raise ValueError("Invalid HARP variant or train batch count")
    if getattr(tr, "release_schedule_installed", False):
        raise RuntimeError("The original schedule must only be installed once")
    if hasattr(tr, "train_loader_x") and len(tr.train_loader_x) != count:
        raise ValueError("Schedule count differs from actual training batches")
    if hasattr(tr, "cfg") and config_identity(tr.cfg)[2] != variant:
        raise ValueError("Schedule variant differs from the full configuration")
    # Each module instance owns COUNT; a later trainer must not alter the
    # closure of an already constructed trainer with a different batch count.
    for name in ("policy", "projector_schedule"):
        path = PROJECT / (name + ".py")
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != path:
            raise RuntimeError(f"Foreign schedule module {name}; use a fresh process")
        if loaded is None:
            _load(name, path)
    import torch
    worker = _load(f"harp_portable_install_{id(tr)}", ROOT / "runtime/frozen_install.py")
    worker.COUNT = count
    worker.install(torch, tr, "r1" if variant == "r1" else "harp")
    tr.release_schedule_installed = True
    return tr


def evidence_module():
    """Original attach/compare with unchanged row-scale hard bounds."""
    return _load("harp_portable_evaluation_evidence", PROJECT / "evaluation_evidence.py")


evaluation_helpers = evidence_module
