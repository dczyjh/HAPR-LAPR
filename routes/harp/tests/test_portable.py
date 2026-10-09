"""CPU-only adapter tests; no external data, full CLIP build or CUDA action."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harp_portable_tested", ROOT / "portable.py")
hp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hp
spec.loader.exec_module(hp)
PATHS = dict(data_root="/tmp/harp-test-data", clip_root="/tmp/harp-test-clip",
             teacher_root="/tmp/harp-test-teacher", output="/tmp/harp-test-output")


class PortableTests(unittest.TestCase):
    def test_frozen_source_checks_and_all_templates(self):
        self.assertEqual(hp.source_checks()["status"], "passed")
        for dataset in hp.DATASET_DIRECTORIES:
            for seed in (1, 2, 3):
                for variant in hp.VARIANTS:
                    cfg = hp.resolve_config(dataset, seed, variant, PATHS)
                    self.assertEqual(hp.config_identity(cfg), (dataset, seed, variant))
        self.assertTrue(hp.resolve_config("dtd", 2, "harp", {**PATHS, "output_dir": "/tmp/preferred"})["OUTPUT_DIR"].endswith("/preferred"))

    def test_full_configuration_drift_rejected(self):
        cfg = hp.resolve_config("dtd", 1, "harp", PATHS)
        cfg["TRAIN"]["PRINT_FREQ"] += 1
        with self.assertRaisesRegex(ValueError, "Full configuration differs"):
            hp.config_identity(cfg)
        with self.assertRaises(ValueError):
            hp.resolve_config("dtd", 4, "harp", PATHS)
        with self.assertRaises(ValueError):
            hp.resolve_config("dtd", 1, "harp", {**PATHS, "clip_root": "${CLIP_ROOT}"})

    def test_yacs_full_roundtrip_and_cpu_full_build_rejected(self):
        import torch
        from yacs.config import CfgNode
        train = hp._framework()
        raw = hp.resolve_config("eurosat", 2, "last4_d64", PATHS)
        cfg = train.get_cfg_default()
        train.extend_cfg(cfg)
        cfg.merge_from_other_cfg(CfgNode(raw))
        cfg.freeze()
        self.assertEqual(hp.config_identity(cfg), ("eurosat", 2, "last4_d64"))
        with self.assertRaisesRegex(RuntimeError, "require CUDA"):
            hp._runtime(torch, "cpu")

    def make_initial_trainer(self, variant):
        import torch
        hp._framework()
        from trainers.efficient_adaptation import HighLevelAdapter
        torch.manual_seed(71)
        model = torch.nn.Module()
        model.shared = torch.nn.Parameter(torch.ones(2))
        model.image_encoder = torch.nn.Module()
        model.image_encoder.transformer = torch.nn.Module()
        model.image_encoder.transformer.resblocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
        layers, width, count = hp.VARIANTS[variant]
        with torch.random.fork_rng(devices=[]):
            if count:
                for layer in layers:
                    model.image_encoder.transformer.resblocks[layer].harp_adapter = HighLevelAdapter(8, width, .001, torch.float32)
        return SimpleNamespace(model=model, cfg=hp.resolve_config("dtd", 2, variant, PATHS))

    def test_initial_mapping_validates_without_copy_or_rng_change(self):
        import torch
        anchor = hp.capture_initial(self.make_initial_trainer("harp"))
        middle = self.make_initial_trainer("mid4_d32")
        rng = torch.get_rng_state().clone()
        before = {n: p.detach().clone() for n, p in middle.model.named_parameters()}
        hp.apply_initial_reference(middle, anchor, dataset="dtd", seed=2, variant="mid4_d32")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(before[n], p) for n, p in middle.model.named_parameters()))
        own = hp.capture_initial(middle)
        hp.apply_initial_reference(middle, own, dataset="dtd", seed=2, variant="mid4_d32")
        broken = copy.deepcopy(own)
        key = next(iter(broken["added"]))
        broken["added"][key].flatten()[0] += 1
        with self.assertRaisesRegex(RuntimeError, "added initialization differs"):
            hp.apply_initial_reference(middle, broken, dataset="dtd", seed=2, variant="mid4_d32")
        broken = copy.deepcopy(own)
        broken["rng"]["cpu"] = "different"
        with self.assertRaisesRegex(RuntimeError, "RNG differs"):
            hp.apply_initial_reference(middle, broken, dataset="dtd", seed=2, variant="mid4_d32")
        broken = copy.deepcopy(own)
        broken["common"]["shared"] = "different"
        with self.assertRaisesRegex(RuntimeError, "common initialization differs"):
            hp.apply_initial_reference(middle, broken, dataset="dtd", seed=2, variant="mid4_d32")

    def test_width_variant_retains_natural_initialization(self):
        import torch
        anchor = hp.capture_initial(self.make_initial_trainer("harp"))
        narrow = self.make_initial_trainer("last4_d16")
        before = {n: p.detach().clone() for n, p in narrow.model.named_parameters()}
        hp.apply_initial_reference(narrow, anchor, dataset="dtd", seed=2, variant="last4_d16")
        self.assertTrue(all(torch.equal(before[n], p) for n, p in narrow.model.named_parameters()))
        own = hp.capture_initial(narrow)
        rebuilt = self.make_initial_trainer("last4_d16")
        hp.apply_initial_reference(rebuilt, own, dataset="dtd", seed=2, variant="last4_d16")

    def test_schedule_instances_do_not_share_count_and_reject_double_install(self):
        import torch
        def tiny():
            model = torch.nn.Module()
            model.VPT = torch.nn.Parameter(torch.ones(1))
            model.VPT_image_trans = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(6)])
            model.harp_adapter = torch.nn.Linear(1, 1)
            shared = [p for n, p in model.named_parameters() if not hp.added(n)]
            added = [p for n, p in model.named_parameters() if hp.added(n)]
            return SimpleNamespace(model=model, epoch=1, batch_idx=1,
                optim=torch.optim.SGD([dict(params=shared, lr=.005), dict(params=added, lr=.0005)], momentum=.9))
        first, second = tiny(), tiny()
        hp.install_schedule(first, "harp", 3)
        hp.install_schedule(second, "harp", 5)
        for trainer, count in ((first, 3), (second, 5)):
            for parameter in trainer.model.parameters():
                parameter.grad = torch.full_like(parameter, .1)
            trainer.optim.step()
            self.assertAlmostEqual(trainer.actual_step["projector_lr"], 1e-5 + (.005 - 1e-5) / (count - 1))
            self.assertAlmostEqual(trainer.actual_step["added_lr"], 1e-6 + (.0005 - 1e-6) / (count - 1))
        with self.assertRaisesRegex(RuntimeError, "only be installed once"):
            hp.install_schedule(first, "harp", 3)

    def test_original_evidence_hard_bound_and_identity_fail_closed(self):
        import torch
        evidence = hp.evidence_module()
        a = dict(inputs=[dict(image="same", label="same")], teacher_text_hash="same",
                 labels=torch.tensor([0]), logits=torch.tensor([[1., 0.]]), accuracy=100.)
        b = copy.deepcopy(a)
        b["logits"][0, 1] = .003
        result = evidence.compare(a, b, torch, {"forward": {"atol": .002, "rtol": .002}})
        self.assertEqual(result["failed_elements"], 1)
        self.assertEqual(result["row_scaled_failed_elements"], 0)
        b["logits"][0, 1] = .005
        with self.assertRaisesRegex(AssertionError, "outside registered"):
            evidence.compare(a, b, torch, {"forward": {"atol": .002, "rtol": .002}})
        b = copy.deepcopy(a)
        b["inputs"][0]["image"] = "different"
        with self.assertRaisesRegex(AssertionError, "input identity"):
            evidence.compare(a, b, torch, {"forward": {"atol": .002, "rtol": .002}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
