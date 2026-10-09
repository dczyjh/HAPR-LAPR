import copy
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from torch.nn import functional as F
from dassl.data.datasets import Datum, DatasetBase, build_dataset
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.adaptation_diagnostics import AdaptationDiagnostics, probe_contribution, lora_weight_diagnostics, adaptation_disabled
from trainers.adaptation_optim import adaptation_param_groups
from trainers.development import base_development_dataset
from tools.report_results import read_runs
from tools.preflight import DATASET_DIRS, TEACHER_DIRS
from tools.run_experiment import parser, make_command
from tools.report_development import read_development
from test_pipeline import config, student


class DiagnosticsTest(unittest.TestCase):
    def test_probe_preserves_rng_buffers_modes_and_parameters(self):
        for method in ("none", "harp", "larp"):
            model = student(method).train()
            # Nonzero adapters ensure this checks the real on/off path.
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B_" in name or "harp_adapter.up.weight" in name:
                        p.fill_(.05)
            image, text = torch.randn(4, 3, 8, 8), F.normalize(torch.randn(5, 768), dim=-1)
            before = copy.deepcopy(model.state_dict())
            modes = [m.training for m in model.modules()]
            rng = torch.get_rng_state().clone()
            result = probe_contribution(model, image, text, "fp32")
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(modes, [m.training for m in model.modules()])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            if method == "none":
                self.assertEqual(result["max_abs_logit_change"], 0.)
            else:
                self.assertGreater(result["max_abs_logit_change"], 0.)
            if method == "harp":
                self.assertEqual(len(result["harp_residual_to_block_output_norm"]), 2)

    def test_lora_diagnostic_detects_half_weight_rounding(self):
        model = student("larp")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B_" in name:
                    parameter.fill_(1e-7)
        # fp16 mode here tests the cast while keeping CPU matmul in fp32.
        records = lora_weight_diagnostics(model, "fp16")
        self.assertEqual(len(records), 2)
        for row in records.values():
            self.assertGreater(row["ideal_delta_to_weight_norm"], 0.)
            self.assertLess(row["surviving_nonzero_fraction"], 1.)

    def test_disabled_context_restores_on_exception(self):
        model = student("larp")
        with self.assertRaises(RuntimeError):
            with adaptation_disabled(model):
                raise RuntimeError("test")
        self.assertFalse(any(getattr(m, "diagnostic_disabled", False) for m in model.modules()))

    def test_diagnostics_do_not_change_training_trajectory(self):
        image, text = torch.randn(4, 3, 8, 8), F.normalize(torch.randn(5, 768), dim=-1)
        for method in ("none", "harp", "larp"):
            reference, instrumented = student(method), student(method)
            optimizers = [torch.optim.SGD(m.parameters(), lr=.005, momentum=.9)
                          for m in (reference, instrumented)]
            with tempfile.TemporaryDirectory() as folder:
                cfg = config(method)
                cfg.TRAINER.PROMPTKD.DIAGNOSTICS.INTERVAL = 1
                diagnostic = AdaptationDiagnostics(instrumented, cfg, folder)
                for step in (1, 2, 3):
                    for index, model in enumerate((reference, instrumented)):
                        optimizer = optimizers[index]
                        optimizer.zero_grad()
                        (model(image)[0] @ text.t()).square().mean().backward()
                        if index == 1:
                            diagnostic.before_step(step, image, optimizer)
                        optimizer.step()
                        if index == 1:
                            diagnostic.after_step(step, text)
                    for key, value in reference.state_dict().items():
                        torch.testing.assert_close(value, instrumented.state_dict()[key], rtol=0, atol=0)
                rows = [json.loads(line) for line in (Path(folder) / "diagnostics.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                self.assertTrue(rows[0]["not_accuracy_evidence"])


class OptimizerGroupsTest(unittest.TestCase):
    def test_group_ratio_survives_warmup_and_cosine(self):
        cfg = config("larp")
        cfg.OPTIM.LR = .005
        cfg.OPTIM.MAX_EPOCH = 5
        cfg.OPTIM.LR_SCHEDULER = "cosine"
        cfg.OPTIM.WARMUP_EPOCH = 1
        cfg.OPTIM.WARMUP_TYPE = "constant"
        cfg.OPTIM.WARMUP_CONS_LR = 1e-5
        cfg.TRAINER.PROMPTKD.ADAPTATION.LR_MULT = 3.
        model = student("larp")
        optimizer = build_optimizer(model, cfg.OPTIM, param_groups=adaptation_param_groups(model, cfg))
        scheduler = build_lr_scheduler(optimizer, cfg.OPTIM)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        for _ in range(5):
            self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 3 * optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

    def test_default_groups_match_original_optimizer_updates(self):
        cfg = config("harp")
        original, grouped = student("harp"), student("harp")
        optimizers = [build_optimizer(original, cfg.OPTIM),
                      build_optimizer(grouped, cfg.OPTIM, param_groups=adaptation_param_groups(grouped, cfg))]
        image = torch.randn(4, 3, 8, 8)
        for _ in range(3):
            for model, optimizer in zip((original, grouped), optimizers):
                optimizer.zero_grad()
                model(image)[0][:, :10].sum().backward()
                optimizer.step()
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, grouped.state_dict()[key], rtol=0, atol=0)


class DevelopmentSplitTest(unittest.TestCase):
    def test_all_ten_actual_dataset_readers(self):
        image_subdirs = {"caltech101": "101_ObjectCategories", "oxford_pets": "images",
                         "stanford_cars": "", "oxford_flowers": "jpg", "food101": "images",
                         "fgvc_aircraft": "images", "sun397": "SUN397", "dtd": "images",
                         "eurosat": "2750", "ucf101": "UCF-101-midframes"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for slug, directory in DATASET_DIRS.items():
                with self.subTest(dataset=slug):
                    data_root = root / directory
                    images = data_root / image_subdirs[slug]
                    images.mkdir(parents=True)
                    metadata = {}
                    for split in ("train", "val", "test"):
                        metadata[split] = []
                        for label, name in enumerate(("a", "b", "c", "d")):
                            filename = f"{split}_{label}.jpg"
                            (images / filename).touch()
                            metadata[split].append([filename, label, name])
                    if slug == "fgvc_aircraft":
                        (data_root / "variants.txt").write_text("a\nb\nc\nd\n")
                        for split, rows in metadata.items():
                            (data_root / f"images_variant_{split}.txt").write_text(
                                "".join(f"{Path(path).stem} {name}\n" for path, label, name in rows))
                    else:
                        (data_root / f"split_zhou_{TEACHER_DIRS[slug]}.json").write_text(json.dumps(metadata))
                    cfg = config()
                    cfg.DATASET.ROOT = str(root)
                    cfg.DATASET.NAME = TEACHER_DIRS[slug]
                    cfg.DATASET.NUM_SHOTS = 0
                    cfg.TRAINER.NAME = "PromptKD"
                    dataset = build_dataset(cfg)
                    result = base_development_dataset(dataset)
                    self.assertEqual(len(result.val), 2)
                    self.assertTrue(all(Path(item.impath).name.startswith("val_") for item in result.val))
                    self.assertEqual(len(result.train_x), 4)

    def fixture(self, root):
        names = ["a", "b", "c", "d"]
        sources = {}
        for split in ("train", "dev", "base_test", "novel_test"):
            sources[split] = []
            for label, name in enumerate(names):
                path = root / f"{split}_{label}.png"
                path.touch()
                sources[split].append(Datum(str(path), label=label, classname=name))
        dataset = DatasetBase(train_x=sources["train"], val=sources["base_test"], test=sources["novel_test"])
        dataset.split_path = root / "split.json"
        dataset.image_dir = root
        rows = [[Path(x.impath).name, x.label, x.classname] for x in sources["dev"]]
        dataset.split_path.write_text(json.dumps({"val": rows}))
        return dataset

    def test_original_val_only_base_no_training_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(Path(folder))
            result = base_development_dataset(dataset)
            self.assertIs(result.train_x, dataset.train_x)
            self.assertEqual([item.label for item in result.val], [0, 1])
            self.assertTrue(all("dev_" in item.impath for item in result.test))
            self.assertEqual(result.classnames, dataset.classnames)

    def test_overlapping_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(Path(folder))
            metadata = json.loads(dataset.split_path.read_text())
            metadata["val"][0][0] = Path(dataset.train_x[0].impath).name
            dataset.split_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "overlap"):
                base_development_dataset(dataset)

    def test_development_cannot_enter_benchmark_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "metrics.json").write_text(json.dumps({"schema_version": 3, "status": "complete",
                                                           "evaluation_role": "development_base"}))
            self.assertEqual(read_runs(root, "larp", 0.), {})

    def test_benchmark_values_cannot_enter_development_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "development_metrics.json").write_text(json.dumps({
                "schema_version": 3, "status": "complete", "evaluation_role": "development_base",
                "novel": 90.}))
            with self.assertRaisesRegex(ValueError, "isolated"):
                read_development(root, 20)

    def test_mixed_variants_across_seeds_are_rejected(self):
        from test_report_results import record
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed in (1, 2):
                row = record("larp")
                row.update(schema_version=3, evaluation_role="benchmark", seed=seed,
                           adaptation_spec=f"LR_MULT: {seed}")
                directory = root / str(seed)
                directory.mkdir()
                (directory / "metrics.json").write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Mixed adaptation"):
                read_runs(root, "larp", 0.)
            candidate_id = hashlib.sha256("LR_MULT: 1".encode()).hexdigest()[:12]
            selected = read_runs(root, "larp", 0., candidate_id)
            self.assertEqual(list(selected), [("larp", "caltech101", 1)])

    def test_controlled_launch_options_merge_and_isolate_paths(self):
        args = parser().parse_args(["--method", "larp", "--data-root", "/tmp/data",
                                   "--stage", "development", "--adapt-lr-mult", "3",
                                   "--lora-targets", "q", "v"])
        command, output = make_command(args, "dtd", 1)
        cfg = config()
        cfg.merge_from_file(command[command.index("--config-file") + 1])
        cfg.merge_from_list(command[command.index("DATASET.NUM_SHOTS"):])
        self.assertTrue(cfg.TRAINER.PROMPTKD.DEVELOPMENT)
        self.assertEqual(cfg.TRAINER.PROMPTKD.ADAPTATION.LR_MULT, 3.)
        self.assertEqual(cfg.TRAINER.PROMPTKD.ADAPTATION.LORA_TARGETS, ["q", "v"])
        self.assertIn("development", output.parts)
        self.assertIn("lr_3_targets_qv", output.parts)
