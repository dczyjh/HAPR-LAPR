"""Small CPU arithmetic tests, not model training or GPU reproducibility checks.

The source archive can be checked with the standard library alone. These tests
skip explicitly if torch/yacs are absent; they never install dependencies.
"""
import ast
import copy
import importlib.util
import json
import math
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "yacs"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_DEPS, "Optional CPU tests require already-installed torch and yacs")
class RetainedScienceCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.ps = load("larp_release_cpu_schedule", ROOT / "protocol/projector_schedule.py")
        cls.sgd = load("larp_release_cpu_sgd", ROOT / "protocol/original_sgd.py")
        cls.adaptation = load("larp_release_cpu_adaptation", ROOT / "archive/project/trainers/efficient_adaptation.py")
        cls.scheduler = load("larp_release_cpu_scheduler", ROOT / "archive/project/Dassl.pytorch/dassl/optim/lr_scheduler.py")
        raw = (ROOT / "archive/helpers/larp_research_20260928_fix1/research_queue.py.txt").read_text()
        node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "added")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "retained_added", "exec"), namespace)
        cls.sgd.q = types.SimpleNamespace(added=namespace["added"])

    def test_original_projector_schedule_and_momentum(self):
        result = self.ps.cpu_test(self.torch)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["momentum_preserved"])

    def test_original_sgd_against_independent_per_parameter_reference(self):
        torch = self.torch
        from yacs.config import CfgNode
        configs = json.loads((ROOT / "protocol/resolved_templates.json").read_text())
        optim_cfg = CfgNode(configs["dtd"]["o4"]["OPTIM"])

        class Tiny(torch.nn.Module):
            def __init__(self, adapted):
                super().__init__()
                self.VPT = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
                self.VPT_image_trans = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64)) for _ in range(6)])
                if adapted:
                    self.lora_A_o = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
                    self.lora_B_o = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

        for adapted in (False, True):
            model = Tiny(adapted)
            common = [p for n, p in model.named_parameters() if not self.sgd.q.added(n)]
            added = [p for n, p in model.named_parameters() if self.sgd.q.added(n)]
            groups = [{"params": common, "lr": .005, "warmup_factor": 1.}]
            if adapted:
                groups.append({"params": added, "lr": .0005, "warmup_factor": .1})
            opt = torch.optim.SGD(groups, lr=.005, momentum=.9, weight_decay=.0005)
            scheduler = self.scheduler.build_lr_scheduler(opt, optim_cfg)
            tr = types.SimpleNamespace(model=model, optim=opt, epoch=0, batch_idx=0)
            count = 368
            self.sgd.f = types.SimpleNamespace(ps=self.ps, COUNT=count)
            self.sgd.install(torch, tr, "larp" if adapted else "r1")
            reference = copy.deepcopy(model)
            refs = [torch.optim.SGD([p], lr=.005, momentum=.9, weight_decay=.0005) for p in reference.parameters()]
            for epoch in range(20):
                tr.epoch = epoch
                if epoch == 0:
                    self.assertEqual(opt.param_groups[0]["lr"], 1e-5)
                if epoch == 1:
                    self.assertEqual(opt.param_groups[0]["lr"], .005)
                for batch in (0, count - 1):
                    tr.batch_idx = batch
                    group_ids = [id(g) for g in opt.param_groups]
                    for p in model.parameters():
                        p.grad = torch.full_like(p, .1)
                    opt.step()
                    self.assertEqual([id(g) for g in opt.param_groups], group_ids)
                    for (name, p), (_, expected), ref_opt in zip(model.named_parameters(), reference.named_parameters(), refs):
                        expected.grad = torch.full_like(expected, .1)
                        if self.sgd.q.added(name):
                            rate = tr.actual_step["added_lr"]
                            self.assertTrue(math.isclose(rate, .1 * tr.actual_step["prompt_lr"], rel_tol=1e-12))
                        elif name.startswith("VPT_image_trans."):
                            rate = tr.actual_step["projector_lr"]
                        else:
                            rate = tr.actual_step["prompt_lr"]
                        ref_opt.param_groups[0]["lr"] = rate
                        ref_opt.step()
                        self.assertTrue(torch.equal(p, expected), (adapted, epoch, batch, name))
                        self.assertTrue(torch.equal(opt.state[p]["momentum_buffer"], ref_opt.state[expected]["momentum_buffer"]))
                scheduler.step()

    def test_o_projection_rank_zero_start_and_updates(self):
        torch = self.torch
        for rank in (1, 2, 4):
            torch.manual_seed(17)
            weight = torch.randn(16, 16)
            module = self.adaptation.WeightLoRAParametrization(16, 16, rank, 1., torch.float32, torch.device("cpu"))
            self.assertEqual(module.scaling, 1 / math.sqrt(rank))
            self.assertEqual(sum(p.numel() for p in module.parameters()), 32 * rank)
            self.assertTrue(torch.equal(module(weight), weight))
            opt = torch.optim.SGD(module.parameters(), lr=.01, momentum=.9, weight_decay=.0005)
            for _ in range(4):
                opt.zero_grad()
                module(weight).square().sum().backward()
                opt.step()
            self.assertGreater(torch.count_nonzero(module.lora_B_o @ module.lora_A_o).item(), 0)
            clone = copy.deepcopy(module)
            clone.load_state_dict(module.state_dict())
            self.assertTrue(torch.equal(module(weight), clone(weight)))


if __name__ == "__main__":
    unittest.main()
