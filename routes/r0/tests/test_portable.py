"""Offline CPU tests; no dataset, real model weights, CUDA or training job."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tested_r0_portable", ROOT / "portable.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def load_module(name, path, package=False):
    kw = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kw)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_config_framework():
    """Real defaults + exact original extend_cfg, without unrelated WILDS imports."""
    from yacs.config import CfgNode
    defaults = load_module("tested_r0_defaults", adapter.OFFICIAL / "Dassl.pytorch/dassl/config/defaults.py")
    source = adapter.OFFICIAL / "train.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "extend_cfg")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(get_cfg_default=lambda: defaults._C.clone(),
                           train=SimpleNamespace(extend_cfg=namespace["extend_cfg"]), CfgNode=CfgNode)


class PortableOfficialTests(unittest.TestCase):
    def paths(self):
        return dict(data_root="/fixture/data", clip_root="/fixture/clip",
                    teacher_root="/fixture/teacher", output_root="/fixture/output")

    def test_all_thirty_actual_yacs_configs(self):
        framework = official_config_framework()
        templates = json.loads((ROOT / "templates.json").read_text())
        self.assertEqual(len(templates), 10)
        for dataset in templates:
            for seed in (1, 2, 3):
                resolved = adapter.resolve_config(dataset, seed, "r0", self.paths())
                cfg = adapter._make_cfg(framework, resolved)
                self.assertEqual(adapter.config_identity(dict(resolved, official_cfg=cfg)), (dataset, seed, "r0"))
                self.assertTrue(cfg.is_frozen())
                self.assertEqual(cfg.OPTIM.NAME, "sgd")
                self.assertEqual(cfg.OPTIM.MAX_EPOCH, 20)
                self.assertEqual(cfg.TRAINER.PROMPTKD.PREC, "fp16")
                self.assertNotIn("ADAPTATION", cfg.TRAINER.PROMPTKD)
        direct = adapter.resolve_config("dtd", 1, "r0", SimpleNamespace(**dict(self.paths(), output_dir="/fixture/phase")))
        self.assertEqual(direct["official_cfg"]["OUTPUT_DIR"], "/fixture/phase")

    def test_scientific_overrides_rejected(self):
        cfg = adapter.resolve_config("dtd", 1, "r0", self.paths())
        for group, key, value in (("OPTIM", "LR", .002), ("OPTIM", "MAX_EPOCH", 19),
                                  ("DATASET", "NUM_SHOTS", 16)):
            changed = copy.deepcopy(cfg)
            changed["official_cfg"][group][key] = value
            with self.assertRaises(ValueError):
                adapter.config_identity(changed)
        for variant, seed in (("r1", 1), ("r0", 4)):
            with self.assertRaises(ValueError):
                adapter.resolve_config("dtd", seed, variant, self.paths())
        with self.assertRaises(ValueError):
            adapter.resolve_config("dtd", 1, "r0", {k:v for k,v in self.paths().items() if k != "output_root"})

    def test_reader_identity_matches_exact_original_rule(self):
        cfg = SimpleNamespace(DATASET=SimpleNamespace(ROOT="/fixture/data", NAME="TinyDataset"))
        datum = lambda path, label, classname: SimpleNamespace(impath=path, label=label, classname=classname)
        data = SimpleNamespace(classnames=["a", "b"],
            train_x=[datum("/fixture/data/tiny/link/a.jpg", 0, "a"), datum("/fixture/data/tiny/b.jpg", 1, "b")],
            val=[datum("/fixture/data/tiny/val.jpg", 0, "a")],
            test=[datum("/fixture/data/tiny/test.jpg", 0, "b")])
        membership = {split: [(os.path.relpath(os.path.abspath(x.impath), os.path.abspath(cfg.DATASET.ROOT)), x.label, x.classname)
                              for x in getattr(data, split)] for split in ("train_x", "val", "test")}
        expected = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
        actual = adapter.data_identity(cfg, data)
        self.assertEqual(actual["split_sha256"], expected)
        self.assertEqual(actual["classnames"], ["a", "b"])
        self.assertFalse(actual["image_bytes_verified"])
        data.train_x.reverse()
        self.assertNotEqual(adapter.data_identity(cfg, data)["split_sha256"], expected)

    def test_initial_reference_checks_identity_without_mutation(self):
        import torch
        model = torch.nn.Linear(2, 2)
        model.bias.requires_grad_(False)
        tr = SimpleNamespace(model=model, cfg=SimpleNamespace(SEED=1), release_dataset="dtd")
        with patch.object(torch.cuda, "is_available", return_value=False):
            reference = adapter.capture_initial(tr)
            before = {n:p.detach().clone() for n,p in model.named_parameters()}
            adapter.apply_initial_reference(tr, reference)
            for key, value in (("dataset", "food101"), ("seed", 2), ("variant", "r1"),
                               ("source_variant", "r1"), ("rng", {}), ("common", {}), ("added", {"wrong": "not_empty"})):
                changed = dict(reference, **{key: value})
                with self.assertRaises(ValueError):
                    adapter.apply_initial_reference(tr, changed)
            for name, value in model.named_parameters():
                self.assertTrue(torch.equal(value, before[name]))

    def test_official_sgd_recorder_is_numerically_transparent(self):
        import torch
        framework = official_config_framework()
        cfg = adapter._make_cfg(framework, adapter.resolve_config("dtd", 1, "r0", self.paths()))
        optim_module = load_module("tested_r0_optim", adapter.OFFICIAL / "Dassl.pytorch/dassl/optim/__init__.py", package=True)
        models = [torch.nn.Linear(2, 1, dtype=torch.float64) for _ in range(2)]
        models[1].load_state_dict(models[0].state_dict())
        optimizers = [optim_module.build_optimizer(m, cfg.OPTIM) for m in models]
        schedulers = [optim_module.build_lr_scheduler(o, cfg.OPTIM) for o in optimizers]
        tr = SimpleNamespace(optim=optimizers[0], train_loader_x=[None, None], epoch=0, batch_idx=0)
        adapter.install_schedule(tr, "r0", 2)
        with self.assertRaises(ValueError):
            adapter.install_schedule(tr, "r0", 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for epoch in range(20):
                tr.epoch = epoch
                for batch in range(2):
                    tr.batch_idx = batch
                    rate = optimizers[1].param_groups[0]["lr"]
                    for model in models:
                        for p in model.parameters():
                            p.grad = torch.full_like(p, .25 + epoch / 100 + batch / 10)
                    for optimizer in optimizers:
                        optimizer.step()
                    self.assertEqual(tr.actual_step["scheduled_lr"], [rate])
                    self.assertEqual(tr.actual_step["projector_lr"], rate)
                    self.assertEqual(tr.actual_step["prompt_lr"], rate)
                    self.assertIsNone(tr.actual_step["added_lr"])
                    for a,b in zip(models[0].parameters(), models[1].parameters()):
                        self.assertTrue(torch.equal(a, b))
                        self.assertTrue(torch.equal(optimizers[0].state[a]["momentum_buffer"], optimizers[1].state[b]["momentum_buffer"]))
                for scheduler in schedulers:
                    scheduler.step()
                self.assertEqual(optimizers[0].param_groups[0]["lr"], optimizers[1].param_groups[0]["lr"])

    def test_relocated_view_copies_code_and_links_only_requested_assets(self):
        with tempfile.TemporaryDirectory(prefix="r0-view-test-") as directory:
            paths = dict(self.paths(), output_dir=directory)
            resolved = adapter.resolve_config("dtd", 1, "r0", paths)
            view = adapter._create_view(resolved)
            self.assertEqual((view / "train.py").read_bytes(), (adapter.OFFICIAL / "train.py").read_bytes())
            self.assertEqual(os.readlink(view / "teacher_model"), resolved["teacher_root"])
            for name in ("ViT-B-16.pt", "ViT-L-14.pt"):
                self.assertEqual(os.readlink(view / "clip" / name), str(Path(resolved["clip_root"]) / name))
            with self.assertRaises(FileExistsError):
                adapter._create_view(resolved)

    def test_no_implicit_cpu_precision_fallback(self):
        cfg = adapter.resolve_config("dtd", 1, "r0", self.paths())
        with self.assertRaisesRegex(ValueError, "CUDA"):
            adapter.build(cfg, device="cpu")

    def test_full_official_import_when_declared_optional_dependency_available(self):
        # Do not fake an optional package merely to claim full import success.
        if importlib.util.find_spec("wilds") is None:
            self.skipTest("Local audit environment lacks wilds==1.2.2; declared Linux lock includes it")
        code = ("import importlib.util; from pathlib import Path; "
                f"p=Path({str(ROOT / 'portable.py')!r}); "
                "s=importlib.util.spec_from_file_location('r0_full_import',p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m._framework(m.OFFICIAL)")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
