"""Portable wiring tests; all tensors are tiny CPU fixtures, never CLIP runs."""
import copy
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("larp_portable_test", ROOT / "portable.py")
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)


def config(variant="o4-r2", seed=1):
    return portable.resolve_config("dtd", seed, variant, {
        "data_root": "/example/data", "clip_root": "/example/clip", "teacher_root": "/example/teacher",
        "output_dir": "/example/isolated/phase/output"})


class PortableConfigTests(unittest.TestCase):
    def test_source_and_function_checks(self):
        self.assertEqual(portable.source_checks()["status"], "passed")

    def test_resolves_every_final_role_and_direct_output_path(self):
        for dataset in portable._json("protocol/datasets.json"):
            for variant in portable._json("protocol/variants.json"):
                for seed in (1, 2, 3):
                    cfg = portable.resolve_config(dataset, seed, variant, {
                        "data_root": "/data spaces", "clip_root": "/clip", "teacher_root": "/teachers",
                        "output_dir": "/isolated/phase"})
                    self.assertEqual(portable.config_identity(cfg), (dataset, seed, variant))
                    self.assertEqual(cfg["OUTPUT_DIR"], "/isolated/phase")

    def test_old_targets_or_scientific_overrides_are_rejected(self):
        for field, value in (("LORA_TARGETS", ["v"]), ("LR_MULT", 1.), ("REP_WEIGHT", .1)):
            cfg = config()
            cfg["TRAINER"]["PROMPTKD"]["ADAPTATION"][field] = value
            with self.assertRaises(ValueError):
                portable.config_identity(cfg)
        cfg = config()
        cfg["OPTIM"]["MAX_EPOCH"] = 1
        with self.assertRaises(ValueError):
            portable.config_identity(cfg)

    def test_cpu_build_cannot_silently_change_fp16_protocol(self):
        with self.assertRaisesRegex(ValueError, "no implicit CPU/FP32"):
            portable.build(config(), device="cpu")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Already-installed torch required for CPU fixtures")
class PortableInitialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.adaptation = portable._load("larp_portable_fixture_adaptation", ROOT / "archive/project/trainers/efficient_adaptation.py")

    def make_trainer(self, variant, seed=1):
        torch = self.torch
        torch.manual_seed(seed)
        specs = portable._json("protocol/variants.json")
        model = torch.nn.Module()
        model.VPT = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
        model.image_encoder = torch.nn.Module()
        model.image_encoder.transformer = torch.nn.Module()
        blocks = torch.nn.ModuleList()
        for _ in range(12):
            block = torch.nn.Module()
            block.attn = torch.nn.MultiheadAttention(4, 1, dtype=torch.float16)
            blocks.append(block)
        model.image_encoder.transformer.resblocks = blocks
        if variant != "r1":
            with torch.random.fork_rng(devices=[]):
                self.adaptation.inject_larp(model.image_encoder, specs[variant]["layers"], specs[variant]["rank"], 1., ["o"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(portable.added(name) or name == "VPT")
        return types.SimpleNamespace(model=model, cfg=config(variant, seed))

    def test_same_seed_last3_maps_exact_physical_o4_layers(self):
        torch = self.torch
        anchor = self.make_trainer("o4-r2")
        reference = portable.capture_initial(anchor)
        tail = self.make_trainer("o3-r2")
        before = torch.get_rng_state().clone()
        natural = portable.capture_initial(tail)
        self.assertTrue(any(not torch.equal(value, reference["added"][name]) for name, value in natural["added"].items()))
        portable.apply_initial_reference(tail, reference, dataset="dtd", seed=1, variant="o3-r2")
        mapped = portable.capture_initial(tail)
        self.assertEqual(mapped["common"], reference["common"])
        self.assertTrue(all(isinstance(value, str) for value in mapped["common"].values()))
        self.assertEqual(mapped["rng"], reference["rng"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, reference["added"][name]) for name, value in mapped["added"].items()))

    def test_same_variant_tail_initialization_reloads_its_saved_mapping(self):
        anchor = portable.capture_initial(self.make_trainer("o4-r2"))
        tail = self.make_trainer("o3-r2")
        portable.apply_initial_reference(tail, anchor, dataset="dtd", seed=1, variant="o3-r2")
        own = portable.capture_initial(tail)
        later = self.make_trainer("o3-r2")
        portable.apply_initial_reference(later, own, dataset="dtd", seed=1, variant="o3-r2")
        actual = portable.capture_initial(later)
        self.assertTrue(all(self.torch.equal(v, own["added"][n]) for n, v in actual["added"].items()))

    def test_rank1_and_rank4_do_not_copy_rank2_initializers(self):
        torch = self.torch
        reference = portable.capture_initial(self.make_trainer("o4-r2"))
        for variant in ("o4-r1", "o4-r4"):
            tr = self.make_trainer(variant)
            before = portable.capture_initial(tr)
            portable.apply_initial_reference(tr, reference, dataset="dtd", seed=1, variant=variant)
            after = portable.capture_initial(tr)
            self.assertTrue(all(torch.equal(value, before["added"][name]) for name, value in after["added"].items()))

    def test_common_seed_rng_and_missing_layer_errors_are_hard(self):
        base = portable.capture_initial(self.make_trainer("o4-r2"))
        cases = []
        wrong = copy.deepcopy(base)
        wrong["seed"] = 2
        cases.append(wrong)
        wrong = copy.deepcopy(base)
        first = next(iter(wrong["common"]))
        wrong["common"][first] = "0" * 64
        cases.append(wrong)
        wrong = copy.deepcopy(base)
        wrong["rng"]["cpu"] = "0" * 64
        cases.append(wrong)
        wrong = copy.deepcopy(base)
        del wrong["added"][next(name for name in wrong["added"] if ".resblocks.9." in name)]
        cases.append(wrong)
        for reference in cases:
            tr = self.make_trainer("o3-r2")
            with self.assertRaises(RuntimeError):
                portable.apply_initial_reference(tr, reference, dataset="dtd", seed=1, variant="o3-r2")

    def test_original_evidence_bound_is_not_weakened(self):
        torch = self.torch
        x = torch.tensor([[4., 1.]], dtype=torch.float32)
        y = x.clone()
        y[0, 1] += .02
        self.assertTrue(portable.compare_first(x, x, "student_logits")["passed"])
        self.assertFalse(portable.compare_first(x, y, "student_logits")["passed"])
        record = {"inputs": [{"image": "same", "label": "same"}], "teacher_text_hash": "same",
                  "labels": torch.tensor([0]), "logits": x, "accuracy": 100.}
        other = dict(record, logits=y)
        with self.assertRaises(AssertionError):
            portable.evidence_module().compare(record, other, torch, portable.TOLERANCES)


if __name__ == "__main__":
    unittest.main()
