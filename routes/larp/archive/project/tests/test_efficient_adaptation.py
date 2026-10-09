import copy
import unittest

import torch
import torch.nn as nn

from trainers.efficient_adaptation import HighLevelAdapter, inject_larp, _ResidualScale


class _Block(nn.Module):
    def __init__(self, width=16, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([_Block(), _Block()])


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()


class EfficientAdaptationTest(unittest.TestCase):
    def test_fused_scale_avoids_half_precision_reciprocal_overflow(self):
        value = torch.ones(3, dtype=torch.float16, requires_grad=True)
        _ResidualScale.apply(value, 0.001).backward(torch.full_like(value, 100.))
        self.assertTrue(torch.isfinite(value.grad).all().item())
        torch.testing.assert_close(value.grad, torch.full_like(value, 100.))

    def setUp(self):
        torch.manual_seed(7)

    def test_harp_starts_as_exact_zero_residual_and_receives_gradient(self):
        adapter = HighLevelAdapter(width=16, bottleneck=4, scale=0.1, dtype=torch.float32)
        value = torch.randn(5, 2, 16, requires_grad=True)
        residual = adapter(value)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

        residual.sum().backward()
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0.0)

    def test_larp_zero_initialization_preserves_multihead_attention(self):
        encoder = _Encoder()
        original = copy.deepcopy(encoder.transformer.resblocks[1].attn)
        selected = inject_larp(
            encoder,
            layers=[1],
            rank=2,
            alpha=1.0,
            targets=["v"],
        )
        self.assertEqual(selected, (1,))

        query = torch.randn(6, 3, 16)
        expected, _ = original(query, query, query, need_weights=False)
        actual, _ = encoder.transformer.resblocks[1].attn(
            query, query, query, need_weights=False
        )
        self.assertTrue(torch.equal(expected, actual))

        actual.square().mean().backward()
        names_and_parameters = dict(encoder.named_parameters())
        lora_b = [
            parameter
            for name, parameter in names_and_parameters.items()
            if "lora_B_v" in name
        ]
        self.assertEqual(len(lora_b), 1)
        self.assertIsNotNone(lora_b[0].grad)
        self.assertGreater(lora_b[0].grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
