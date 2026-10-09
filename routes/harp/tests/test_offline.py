"""Offline CPU tests of frozen HARP functions; never instantiate a full CLIP model."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "vendor/project"
sys.path[:0] = [str(PROJECT), str(PROJECT / "Dassl.pytorch"), str(ROOT / "runtime")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Tests(unittest.TestCase):
    def test_full_framework_import_only(self):
        import train
        import dassl
        self.assertEqual(Path(train.__file__).resolve(), PROJECT / "train.py")
        self.assertEqual(Path(dassl.__file__).resolve(), PROJECT / "Dassl.pytorch/dassl/__init__.py")

    def test_extracted_scientific_functions_identical(self):
        records = json.loads((ROOT / "provenance/scientific_functions.json").read_text())
        destination = ast.parse((ROOT / "runtime/frozen_install.py").read_text())
        for name, record in records.items():
            node = next(n for n in destination.body if isinstance(n, ast.FunctionDef) and n.name == name)
            value = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(value, record["ast_sha256"])

    def test_zero_branch_and_parameter_counts(self):
        import torch
        adaptation = load("harp_offline_adaptation", PROJECT / "trainers/efficient_adaptation.py")
        with torch.random.fork_rng(devices=[]):
            for width, expected in ((16, 101440), (32, 199808), (64, 396544)):
                model = adaptation.HighLevelAdapter(768, width, .001, torch.float32)
                self.assertEqual(sum(p.numel() for p in model.parameters()) * 4, expected)
                self.assertEqual(torch.count_nonzero(model(torch.ones(2, 3, 768))).item(), 0)

    def test_fix2_vjp_not_ordinary_scaled_derivative(self):
        import torch
        adaptation = load("harp_offline_vjp", PROJECT / "trainers/efficient_adaptation.py")
        for cls, forward_scale, input_scale in ((adaptation.InputScaledLinear, 1., .001),
                                                 (adaptation.ScaledResidualLinear, .001, 1.)):
            layer = cls(3, 2, .001).double()
            with torch.no_grad():
                layer.weight.fill_(2.); layer.bias.fill_(.5)
            x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
            out = layer(x)
            torch.testing.assert_close(out, torch.full((2, 2), 6.5 * forward_scale, dtype=torch.float64))
            out.sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 4. * input_scale))
            torch.testing.assert_close(layer.weight.grad, torch.full_like(layer.weight, 2.))
            torch.testing.assert_close(layer.bias.grad, torch.full_like(layer.bias, 2.))

    def test_original_projector_schedule_reference(self):
        import torch
        import projector_schedule
        self.assertEqual(projector_schedule.cpu_test(torch)["status"], "passed")

    def test_frozen_double_ramp_installer(self):
        import torch
        import frozen_install
        frozen_install.COUNT = 3
        for method in ("r1", "harp"):
            class Tiny(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.VPT = torch.nn.Parameter(torch.ones(1))
                    self.VPT_image_trans = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(6)])
                    if method == "harp":
                        self.harp_adapter = torch.nn.Linear(1, 1)
            model = Tiny()
            shared = [p for n, p in model.named_parameters() if not frozen_install.added(n)]
            added = [p for n, p in model.named_parameters() if frozen_install.added(n)]
            groups = [dict(params=shared, lr=1e-5)]
            if added:
                groups.append(dict(params=added, lr=1e-6))
            opt = torch.optim.SGD(groups, momentum=.9, weight_decay=.0005)
            tr = SimpleNamespace(model=model, optim=opt, epoch=0, batch_idx=0)
            frozen_install.install(torch, tr, method)
            for epoch, nominal in ((0, 1e-5), (1, .005), (2, .0049)):
                tr.epoch = epoch
                for batch in range(3):
                    tr.batch_idx = batch
                    opt.param_groups[0]["lr"] = nominal
                    if added:
                        opt.param_groups[1]["lr"] = nominal * .1
                    for p in model.parameters():
                        p.grad = torch.full_like(p, .1)
                    opt.step()
                    expected = (1e-5 + (.005 - 1e-5) * batch / 2) if epoch == 1 else nominal
                    self.assertAlmostEqual(tr.actual_step["projector_lr"], expected, places=14)
                    if added and epoch == 0:
                        self.assertTrue(all(p.grad is None and not opt.state.get(p, {}) for p in added))
                    if added and epoch == 1:
                        self.assertAlmostEqual(tr.actual_step["added_lr"], 1e-6 + (.0005 - 1e-6) * batch / 2, places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
