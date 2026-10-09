"""Real torch forward/backward/checkpoint tests with tiny random CLIP weights.

These test engineering, not recognition accuracy. No dataset or network needed.
"""
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch.nn import functional as F
from dassl.config import get_cfg_default
from clip.model import VisionTransformer, convert_weights
from train import extend_cfg
from trainers.promptkd import CustomCLIP, PromptKD
from trainers.efficient_adaptation import adaptation_parameter
from tools.run_experiment import parser, make_command
from trainers.adaptation_diagnostics import probe_contribution, lora_weight_diagnostics

torch.set_num_threads(1)


def config(method="none", rep=0.0):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.TRAINER.PROMPTKD.PREC = "fp32"
    cfg.TRAINER.PROMPTKD.ADAPTATION.TYPE = method
    cfg.TRAINER.PROMPTKD.ADAPTATION.LAYERS = [1, 2]
    cfg.TRAINER.PROMPTKD.ADAPTATION.REP_LAYERS = [1, 2]
    cfg.TRAINER.PROMPTKD.ADAPTATION.REP_WEIGHT = rep
    cfg.TRAINER.PROMPTKD.ADAPTATION.HARP_DIM = 4
    cfg.TRAINER.PROMPTKD.KD_WEIGHT = 1000.0
    return cfg


def clip_model():
    visual = VisionTransformer(8, 4, 16, 3, 4, 512,
                               {"trainer": "IVLP", "vision_depth": 2,
                                "vision_ctx": 2, "language_ctx": 2})
    return SimpleNamespace(visual=visual, logit_scale=torch.nn.Parameter(torch.tensor(1.)),
                           dtype=torch.float32)


def student(method):
    torch.manual_seed(11)
    model = CustomCLIP(config(method), ["a", "b"], clip_model())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("VPT" in name or adaptation_parameter(name))
    return model


class PipelineTest(unittest.TestCase):
    def test_harp_respects_clip_mixed_layernorm_dtype(self):
        backbone = clip_model()
        convert_weights(backbone.visual)
        backbone.dtype = torch.float16
        model = CustomCLIP(config("harp"), ["a", "b"], backbone)
        block = model.image_encoder.transformer.resblocks[1]
        self.assertEqual(block.ln_1.weight.dtype, torch.float32)
        self.assertEqual(block.harp_adapter.down.weight.dtype, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA hardware")
    def test_cuda_fp16_actual_forward_backward(self):
        for method in ("none", "harp", "larp"):
            for precision in ("fp16", "amp"):
                with self.subTest(method=method, precision=precision):
                    backbone = clip_model()
                    if precision == "fp16":
                        convert_weights(backbone.visual)
                        backbone.dtype = torch.float16
                    model = CustomCLIP(config(method), ["a", "b"], backbone).cuda()
                    for name, parameter in model.named_parameters():
                        parameter.requires_grad_("VPT" in name or adaptation_parameter(name))
                    image = torch.randn(4, 3, 8, 8, device="cuda")
                    text = F.normalize(torch.randn(5, 768, device="cuda"), dim=-1)
                    if precision == "fp16":
                        text = text.half()
                    teacher_logits = torch.randn(4, 5, device="cuda", dtype=text.dtype)
                    trainer = object.__new__(PromptKD)
                    trainer.cfg = config(method)
                    trainer.model = model
                    trainer.reference_encoder = None
                    trainer.rep_weight = 0.
                    trainer.rep_layers = (1, 2)
                    trainer.temperature = 1.
                    optimizer = torch.optim.SGD(model.parameters(), lr=.005)
                    with torch.cuda.amp.autocast(enabled=precision == "amp"):
                        loss, _, _ = trainer._compute_student_loss(image, text, teacher_logits)
                    if precision == "amp":
                        scaler = torch.cuda.amp.GradScaler(init_scale=128.)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    self.assertTrue(torch.isfinite(loss).item())
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    optimizer.step()
                    probe = probe_contribution(model, image, text, precision)
                    self.assertIsNotNone(probe["mean_abs_logit_change"])
                    if method == "larp":
                        self.assertEqual(len(lora_weight_diagnostics(model, precision)), 2)

    def test_nonfinite_gradient_stops_before_optimizer_corruption(self):
        model = student("harp")
        trainer = object.__new__(PromptKD)
        trainer.cfg = config("harp")
        trainer.model = model
        trainer.reference_encoder = None
        trainer.rep_weight = 0.
        trainer.rep_layers = (1, 2)
        trainer.temperature = 1.
        trainer.device = torch.device("cpu")
        trainer.scaler = None
        trainer.optim = torch.optim.SGD(model.parameters(), lr=.005)
        text = F.normalize(torch.randn(5, 768), dim=-1)
        teacher_logits = torch.randn(4, 5)
        trainer.model_teacher = lambda image: (None, text, teacher_logits)
        target = model.VPT_image_trans.conv1[3].weight
        before = target.detach().clone()
        handle = target.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient"):
                trainer.forward_backward({"img": torch.randn(4, 3, 8, 8),
                                          "label": torch.zeros(4, dtype=torch.long)})
        finally:
            handle.remove()
        self.assertTrue(torch.equal(before, target))

    def test_same_seed_same_projector_rng_and_zero_update_outputs(self):
        results, next_random = [], []
        image = torch.ones(3, 3, 8, 8)
        for method in ("none", "harp", "larp"):
            model = student(method).eval()
            results.append(model(image)[0].detach())
            next_random.append(torch.rand(5))
        for i in (1, 2):
            torch.testing.assert_close(results[0], results[i], rtol=0, atol=0)
            self.assertTrue(torch.equal(next_random[0], next_random[i]))

    def test_training_updates_only_allowed_parameters_and_checkpoint_roundtrip(self):
        image = torch.randn(4, 3, 8, 8)
        text = F.normalize(torch.randn(5, 768), dim=-1)
        teacher_logits = torch.randn(4, 5)
        for method in ("none", "harp", "larp"):
            with self.subTest(method=method):
                model = student(method)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                trainer = object.__new__(PromptKD)
                trainer.cfg = config(method, 0.05)
                trainer.model = model
                trainer.rep_weight = 0.05
                trainer.rep_layers = (1, 2)
                trainer.reference_encoder = copy.deepcopy(student("none").image_encoder).eval()
                trainer.reference_encoder.requires_grad_(False)
                trainer.temperature = 1.0
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005)
                for _ in range(3):
                    optimizer.zero_grad()
                    loss, kd, rep = trainer._compute_student_loss(image, text, teacher_logits)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all().item())
                    optimizer.step()
                changed = [name for name, p in model.named_parameters()
                           if not torch.equal(p, before[name])]
                self.assertTrue(any("VPT_image_trans" in name for name in changed))
                if method != "none":
                    self.assertTrue(any(adaptation_parameter(name) for name in changed))
                self.assertTrue(all("VPT" in name or adaptation_parameter(name) for name in changed))
                self.assertTrue(all(p.grad is None for p in trainer.reference_encoder.parameters()))
                model.eval()
                expected = model(image)[0]
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                restored = student(method).eval()
                restored.load_state_dict(torch.load(stream), strict=True)
                torch.testing.assert_close(restored(image)[0], expected, rtol=0, atol=0)

    def test_intermediate_path_does_not_change_outputs(self):
        model = student("harp").eval()
        image = torch.randn(2, 3, 8, 8)
        standard = model(image)[0]
        collected = model(image, return_intermediates=True, intermediate_layers=[1, 2])
        torch.testing.assert_close(standard, collected[0], rtol=0, atol=0)
        self.assertEqual(set(collected[2]), {1, 2})

    def test_all_launchers_merge_into_real_yacs_configuration(self):
        for method in ("baseline", "harp", "larp"):
            args = parser().parse_args(["--method", method, "--data-root", "/tmp/data",
                                       "--output-root", "relative_output"])
            command, output = make_command(args, "caltech101", 1)
            self.assertTrue(output.is_absolute())
            cfg = config()
            cfg.merge_from_file(command[command.index("--config-file") + 1])
            cfg.merge_from_list(command[command.index("DATASET.NUM_SHOTS"):])
            self.assertEqual(cfg.TRAINER.PROMPTKD.KD_WEIGHT, 1000.0)
            self.assertEqual(cfg.TEST.FINAL_MODEL, "last_step")


if __name__ == "__main__":
    unittest.main()
