"""Exercise the real Dassl/PromptKD lifecycle using synthetic data/weights."""
import copy
import json
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from clip.model import CLIP
from dassl.data.datasets import Datum
from trainers.promptkd import PromptKD, CustomCLIP_teacher
from test_pipeline import config


def tiny_clip(output_dim):
    return CLIP(output_dim, 8, 3, 64, 4, 77, 49408, 64, 1, 3,
                {"trainer": "IVLP", "vision_depth": 2, "language_depth": 2,
                 "vision_ctx": 2, "language_ctx": 2})


class TrainerLifecycleTest(unittest.TestCase):
    def test_three_methods_complete_actual_trainer_lifecycle(self):
        paired_signatures = {}
        for method, development in itertools.product(("none", "harp", "larp"), (False, True)):
            with self.subTest(method=method, development=development), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                cfg = config(method, 0.05 if method != "none" else 0.)
                cfg.USE_CUDA = False
                cfg.TRAINER.PROMPTKD.DEVELOPMENT = development
                cfg.SEED = 1
                cfg.OUTPUT_DIR = str(root / "output")
                cfg.DATASET.ROOT = str(root)
                cfg.DATASET.NAME = "Caltech101"
                cfg.DATASET.NUM_SHOTS = 0
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 4
                cfg.DATALOADER.TEST.BATCH_SIZE = 4
                cfg.TRAINER.NAME = "PromptKD"
                cfg.TRAINER.PROMPTKD.N_CTX_TEXT = 2
                cfg.TRAINER.PROMPTKD.N_CTX_VISION = 2
                cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT = str(root)
                cfg.TRAINER.PROMPTKD.TEACHER_ROOT = str(root / "teachers")
                cfg.INPUT.SIZE = (8, 8)
                cfg.OPTIM.MAX_EPOCH = 1
                cfg.OPTIM.LR_SCHEDULER = "cosine"
                cfg.TEST.FINAL_MODEL = "last_step"
                cfg.TEST.NO_TEST = False
                classnames = ["cat", "dog", "car", "plane"]
                torch.manual_seed(123)
                backbone = tiny_clip(512).float()
                teacher_backbone = tiny_clip(768).float()
                teacher = CustomCLIP_teacher(cfg, classnames, copy.deepcopy(teacher_backbone))
                teacher_dir = root / "teachers/Caltech101/VLPromptLearner"
                teacher_dir.mkdir(parents=True)
                checkpoint = dict(state_dict=teacher.state_dict(), epoch=1)
                torch.save(checkpoint, teacher_dir / "model-best.pth.tar")
                torch.save(backbone.state_dict(), root / "ViT-B-16.pt")
                torch.save(teacher_backbone.state_dict(), root / "ViT-L-14.pt")
                # Exercise the actual dataset class, split reader, transforms,
                # sampler, and DataManager with real tiny image fixtures.
                image_dir = root / "caltech-101/101_ObjectCategories"
                image_dir.mkdir(parents=True)
                splits = {}
                for split in ("train", "val", "test"):
                    splits[split] = []
                    for label, classname in enumerate(classnames):
                        for index in range(2):
                            filename = f"{split}_{label}_{index}.png"
                            pixels = torch.randint(0, 256, (12, 12, 3), dtype=torch.uint8).numpy()
                            Image.fromarray(pixels).save(image_dir / filename)
                            splits[split].append([filename, label, classname])
                (root / "caltech-101/split_zhou_Caltech101.json").write_text(json.dumps(splits))

                with patch("trainers.promptkd.load_clip_to_cpu", return_value=copy.deepcopy(backbone)), \
                     patch("trainers.promptkd.load_clip_to_cpu_teacher", return_value=copy.deepcopy(teacher_backbone)):
                    trainer = PromptKD(cfg)
                    trainer.train()
                    if development:
                        with self.assertRaisesRegex(ValueError, "Development mode"):
                            trainer.test(split="test")
                        self.assertTrue(all("val_" in item.impath for item in trainer.dm.dataset.val))
                filename = "development_metrics.json" if development else "metrics.json"
                metrics = json.loads((root / "output" / filename).read_text())
                self.assertEqual(metrics["status"], "complete")
                self.assertEqual(metrics["method"], method)
                self.assertEqual(metrics["selection"], "last_step")
                self.assertEqual(metrics["selected_epoch"], 1)
                self.assertTrue((root / "output/VLPromptLearner/model.pth.tar-1").exists())
                self.assertIn("comparison_signature", metrics)
                expected_signature = paired_signatures.setdefault(development, metrics["comparison_signature"])
                self.assertEqual(metrics["comparison_signature"], expected_signature)
                self.assertEqual(metrics["evaluation_role"], "development_base" if development else "benchmark")
                if development:
                    self.assertNotIn("novel", metrics)
                    self.assertFalse((root / "output/metrics.json").exists())
                diagnostics = [json.loads(line) for line in (root / "output/diagnostics.jsonl").read_text().splitlines()]
                self.assertEqual(len(diagnostics), 1)
                for metric in (("base_development",) if development else ("base", "novel", "hm")):
                    self.assertGreaterEqual(metrics[metric], 0.)
                    self.assertLessEqual(metrics[metric], 100.)


if __name__ == "__main__":
    unittest.main()
