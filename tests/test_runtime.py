"""CPU integration fixtures for the supervisor, not research model runs.

The fixture invokes the retained epoch loop, checkpoint selection, scheduler,
SGD wrapper and evaluation-evidence code. Its tiny tensors are synthetic and
cannot validate the full CUDA/FP16 scientific implementation.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


execution = load("tested_portable_execution", ROOT / "runtime/execution.py")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Already-installed torch required for CPU integration fixtures")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.lr_module = load("runtime_fixture_original_scheduler", ROOT / "routes/larp/archive/project/Dassl.pytorch/dassl/optim/lr_scheduler.py")
        cls.schedule = load("runtime_fixture_original_projector_schedule", ROOT / "routes/larp/protocol/projector_schedule.py")
        cls.installer = load("runtime_fixture_original_install", ROOT / "routes/larp/protocol/original_sgd.py")
        cls.evidence = load("runtime_fixture_original_evidence", ROOT / "routes/harp/vendor/project/evaluation_evidence.py")
        source = ROOT / "routes/harp/vendor/project/Dassl.pytorch/dassl/engine/trainer.py"
        tree = ast.parse(source.read_text())
        namespace = {}
        for class_name, method, alias in (("TrainerBase", "train", "epoch_loop"), ("SimpleTrainer", "after_epoch", "select_checkpoint")):
            parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            node = copy.deepcopy(next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == method))
            node.name = alias
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cls.epoch_loop = staticmethod(namespace["epoch_loop"])
        cls.select_checkpoint = staticmethod(namespace["select_checkpoint"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="promptkd-runtime-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.adapter = self.make_adapter()
        self.reg = dict(route="larp", output_root=str(self.folder / "run"),
                        data_root=str(self.folder / "data"), clip_root=str(self.folder / "clip"),
                        teacher_root=str(self.folder / "teachers"), disk_reserve_bytes=0,
                        expected_counts={"dtd": [8, 4, 4]}, release_manifest_sha256="tiny-fixture")

    def make_adapter(self):
        torch = self.torch
        outer = self

        class LoRA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A_o = torch.nn.Parameter(torch.randn(2, 2) * .05)
                self.lora_B_o = torch.nn.Parameter(torch.zeros(2, 2))

            def forward(self, weight):
                return weight + self.lora_B_o @ self.lora_A_o

        class Student(torch.nn.Module):
            def __init__(self, variant):
                super().__init__()
                self.image_encoder = torch.nn.Module()
                self.image_encoder.prompt = torch.nn.Parameter(torch.ones(2) * .03)
                self.image_encoder.transformer = torch.nn.Module()
                block = torch.nn.Module()
                block.attn = torch.nn.Module()
                block.attn.out_proj = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    block.attn.out_proj.weight.copy_(torch.eye(2) * .2)
                block.attn.out_proj.weight.requires_grad_(False)
                self.image_encoder.transformer.resblocks = torch.nn.ModuleList([block])
                if variant != "r1":
                    with torch.random.fork_rng(devices=[]):
                        torch.nn.utils.parametrize.register_parametrization(block.attn.out_proj, "weight", LoRA())
                self.VPT_image_trans = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2) * .01) for _ in range(6)])
                self.logit_scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)

            def forward(self, image, label=None):
                feature = self.image_encoder.transformer.resblocks[0].attn.out_proj(image)
                feature = feature + self.image_encoder.prompt + sum(self.VPT_image_trans) * .1
                return feature, self.logit_scale

        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text = torch.nn.Parameter(torch.eye(2), requires_grad=False)
                self.eval()

            def get_text_features(self):
                return self.text

            def forward(self, image):
                feature = image * .7
                return feature, self.text, feature @ self.text.t()

        class Data(torch.utils.data.Dataset):
            def __init__(self, split, shift=False):
                self.split, self.shift = split, shift

            def __len__(self):
                return 8 if self.split == "train" else 4

            def __getitem__(self, index):
                label = index % 2
                image = torch.tensor([1., .1] if label == 0 else [.1, 1.])
                if self.split == "train":
                    image = image + torch.rand(2) * .01
                if self.shift:
                    image = image + .1
                return dict(img=image, label=label, impath=str(outer.folder / "data" / self.split / f"{index}.png"))

        class Evaluator:
            def reset(self):
                self._total, self._correct = 0, 0

            def process(self, output, label):
                self._total += len(label)
                self._correct += int((output.argmax(1) == label).sum())

        class Trainer:
            def __init__(self, cfg, adapter):
                torch.manual_seed(cfg["seed"])
                self.config, self.adapter = cfg, adapter
                self.output_dir = cfg["output"]
                self.cfg = SimpleNamespace(TEST=SimpleNamespace(NO_TEST=False, FINAL_MODEL="best_val"),
                                           TRAIN=SimpleNamespace(CHECKPOINT_FREQ=0))
                self.model_teacher, self.model = Teacher(), Student(cfg["variant"])
                common = [p for n, p in self.model.named_parameters() if p.requires_grad and not adapter.added(n)]
                extra = [p for n, p in self.model.named_parameters() if p.requires_grad and adapter.added(n)]
                groups = [dict(params=common, lr=.005)]
                if extra:
                    groups.append(dict(params=extra, lr=.0005, warmup_factor=.1))
                self.optim = torch.optim.SGD(groups, momentum=.9)
                successor = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 20)
                self.sched = outer.lr_module.ConstantWarmupScheduler(self.optim, successor, 1, 1e-5)
                shift = adapter.shift_training_input and "/formal/" in cfg["output"] and cfg["variant"] != "r1"
                self.train_loader_x = torch.utils.data.DataLoader(Data("train", shift), batch_size=2, shuffle=True, drop_last=True)
                self.val_loader = torch.utils.data.DataLoader(Data("val"), batch_size=2)
                self.test_loader = torch.utils.data.DataLoader(Data("test"), batch_size=2)
                self.evaluator = Evaluator()
                self.start_epoch, self.max_epoch, self.best_result = 0, 20, -1.
                self.num_batches = len(self.train_loader_x)
                self.epoch, self.batch_idx = 0, 0

            def set_model_mode(self, mode):
                self.model.train(mode == "train")

            def forward_backward(self, batch):
                with torch.no_grad():
                    _, text, expected = self.model_teacher(batch["img"])
                feature, scale = self.model(batch["img"])
                loss = torch.nn.functional.mse_loss(scale * feature @ text.t(), expected)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.batch_idx + 1 == self.num_batches:
                    self.sched.step()
                return dict(loss=float(loss), loss_kd=float(loss))

            def parse_batch_test(self, batch):
                return batch["img"], batch["label"]

            def test(self, split=None):
                self.set_model_mode("eval")
                self.evaluator.reset()
                for batch in self.val_loader if split == "val" else self.test_loader:
                    image, label = self.parse_batch_test(batch)
                    with torch.no_grad():
                        feature, scale = self.model(image)
                        logits = scale * feature @ self.model_teacher.get_text_features().t()
                        if self.adapter.perturb_reload and "/evaluation/" in self.output_dir:
                            logits = logits + .1
                    self.evaluator.process(logits, label)
                return 100. * self.evaluator._correct / self.evaluator._total

            def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
                path = Path(directory) / "VLPromptLearner" / (model_name or f"model.pth.tar-{epoch + 1}")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=self.model.state_dict(), optimizer=self.optim.state_dict(),
                                scheduler=self.sched.state_dict(), epoch=epoch + 1, val_result=val_result), path)

            def load_model(self, directory, epoch=None):
                path = Path(directory) / "VLPromptLearner" / ("model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}")
                self.model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"], strict=True)

            def before_train(self):
                pass

            def before_epoch(self):
                self.set_model_mode("train")

            def run_epoch(self):
                for self.batch_idx, batch in enumerate(self.train_loader_x):
                    self.forward_backward(batch)

            def after_epoch(self):
                return outer.select_checkpoint(self)

            def after_train(self):
                self.load_model(self.output_dir)
                self.test("val")
                self.test("test")
                self.close_writer()

            def train(self):
                return outer.epoch_loop(self, self.start_epoch, self.max_epoch)

            def close_writer(self):
                pass

        class Adapter:
            shift_training_input = False
            perturb_reload = False
            builds = []

            @staticmethod
            def added(name):
                return "lora_A_" in name or "lora_B_" in name

            @staticmethod
            def canonical(name):
                return name.replace(".out_proj.parametrizations.weight.original", ".out_proj.weight")

            @staticmethod
            def tensor_hash(tensor):
                import hashlib
                return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

            @staticmethod
            def resolve_config(dataset, seed, variant, paths):
                return dict(dataset=dataset, seed=seed, variant=variant, output=paths["output_dir"])

            def capture_initial(self, trainer):
                return dict(dataset=trainer.config["dataset"], seed=trainer.config["seed"],
                            variant=trainer.config["variant"], source_variant=trainer.config["variant"],
                            common=execution.hash_model(self, trainer.model), rng=execution.rng_hash(self, torch),
                            added={n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if self.added(n)})

            def build(self, cfg, device="cpu", init_reference=None):
                assert device == "cpu"
                trainer = Trainer(cfg, self)
                self.builds.append(trainer)
                if init_reference:
                    current = self.capture_initial(trainer)
                    assert current["common"] == init_reference["common"]
                    assert current["rng"] == init_reference["rng"]
                    if current["variant"] == init_reference["variant"]:
                        assert current["added"].keys() == init_reference["added"].keys()
                        assert all(torch.equal(v, init_reference["added"][n]) for n, v in current["added"].items())
                return trainer

            def install_schedule(self, trainer, variant, count):
                outer.installer.q = SimpleNamespace(added=self.added)
                outer.installer.f = SimpleNamespace(ps=outer.schedule, COUNT=count)
                outer.installer.install(torch, trainer, "r1" if variant == "r1" else "larp")

            @staticmethod
            def evidence_module():
                return outer.evidence

        return Adapter()

    def work(self, action, variant="r1"):
        return execution.worker(self.reg, action, "dtd", 1, variant, adapter=self.adapter)

    def prepare_training(self, variant="r1"):
        self.work("init", variant)
        self.work("gate", variant)
        return self.work("train", variant)

    def test_nominal_probe_preserves_scheduler_links_and_future_trajectory(self):
        cfg = self.adapter.resolve_config("dtd", 1, "r1", {"output_dir": str(self.folder / "probe")})
        trainer = self.adapter.build(cfg)
        successor = trainer.sched.successor
        initial_rng = self.torch.get_rng_state().clone()
        expected = execution.nominal_rates(trainer)
        self.assertIs(trainer.sched.successor, successor)
        self.assertIs(trainer.sched.successor.optimizer, trainer.optim)
        self.assertTrue(self.torch.equal(initial_rng, self.torch.get_rng_state()))
        observed = []
        for _ in range(20):
            observed.append([g["lr"] for g in trainer.optim.param_groups])
            for p in trainer.model.parameters():
                if p.requires_grad:
                    p.grad = self.torch.ones_like(p) * .01
            trainer.optim.step()
            trainer.sched.step()
        self.assertEqual(observed, expected)

    def test_manual_evaluation_uses_shared_environment_and_lock(self):
        reg = dict(self.reg, gpu="3")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "PYTHONPATH": "/foreign", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            environment = execution.worker_environment(reg)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(environment["OMP_NUM_THREADS"], "4")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", environment)
        item = dict(action="evaluate", dataset="dtd", seed=1, variant="r1")
        lock = Mock()
        with patch.object(execution, "verify_registration") as verified, patch.object(execution, "gpu_lock", return_value=lock) as acquired, patch.object(execution, "execute_item") as launched:
            execution.first_evaluation(reg, item)
        verified.assert_called_once_with(reg)
        acquired.assert_called_once_with(reg)
        self.assertEqual(launched.call_args.args[:2], (reg, item))
        lock.close.assert_called_once()

    def test_paired_fresh_workers_execute_original_loop_and_first_reload(self):
        plan = execution.execution_plan("larp", ["dtd"], [1], ["r1", "o4-r2"])
        self.reg["plan"] = plan
        for item in plan["actions"]:
            execution.worker(self.reg, **item, adapter=self.adapter)
        self.assertEqual(len(self.adapter.builds), 8)
        self.assertEqual(len({id(tr) for tr in self.adapter.builds}), 8)
        pair = execution.pair_dir(self.reg, "dtd", 1)
        traces = []
        for variant in ("r1", "o4-r2"):
            gate = execution.read(pair / "gate" / variant / "audit.json")
            self.assertEqual(len(gate["steps"]), 4)
            self.assertEqual([(r["epoch"], r["batch"]) for r in gate["steps"]], [(1, 1), (2, 1), (2, 4), (3, 1)])
            formal = pair / "formal" / variant
            complete = execution.read(formal / "train_complete.json")
            self.assertEqual((complete["epochs"], complete["batches"]), (20, 80))
            metrics = execution.read(formal / "audited_metrics.json")
            self.assertEqual(metrics["canonical_evaluation"], "first_independent_reload")
            self.assertEqual(set(metrics["audits"]), {"best", "last"})
            self.assertTrue(all(r["row_scaled_failed_elements"] == 0 for r in metrics["reload_audit"].values()))
            traces.append([json.loads(line) for line in (formal / "batches.jsonl").read_text().splitlines()])
        self.assertEqual([r["identity"] for r in traces[0]], [r["identity"] for r in traces[1]])
        self.assertEqual([(r["prompt_lr"], r["projector_lr"]) for r in traces[0]],
                         [(r["prompt_lr"], r["projector_lr"]) for r in traces[1]])

    def test_changed_candidate_input_stops_before_first_formal_update(self):
        self.prepare_training()
        self.work("evaluate")
        self.work("init", "o4-r2")
        self.work("gate", "o4-r2")
        self.adapter.shift_training_input = True
        with self.assertRaisesRegex(AssertionError, "different batch|input mismatch"):
            self.work("train", "o4-r2")
        formal = execution.pair_dir(self.reg, "dtd", 1) / "formal/o4-r2"
        self.assertFalse((formal / "batches.jsonl").exists())
        self.assertFalse((formal / "train_complete.json").exists())
        self.assertTrue((formal / "first_failure_state.pt").is_file())
        self.assertFalse(execution.read(formal / "failure.json")["automatic_retry"])

    def test_first_reload_outside_original_bound_is_preserved_and_not_retried(self):
        self.prepare_training()
        self.adapter.perturb_reload = True
        with self.assertRaisesRegex(AssertionError, "outside registered row-scaled"):
            self.work("evaluate")
        out = execution.pair_dir(self.reg, "dtd", 1) / "evaluation/r1"
        first = out / "first_independent_metrics.json"
        digest = execution.sha(first)
        self.assertEqual(execution.read(first)["status"], "recorded_before_hard_checks")
        self.assertTrue((out / "canonical_val_evidence.pt").is_file())
        self.assertTrue((out / "reload_pair_native_val.pt").is_file())
        self.assertFalse((out / "audited_metrics.json").exists())
        self.assertFalse(execution.read(out / "failure.json")["automatic_retry"])
        with self.assertRaises(FileExistsError):
            self.work("evaluate")
        self.assertEqual(execution.sha(first), digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
