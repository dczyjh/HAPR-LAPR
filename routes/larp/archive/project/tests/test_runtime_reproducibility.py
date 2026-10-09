"""The controlled runtime must reject nondeterminism, not just warn about it."""
import unittest
import torch
from train import configure_cuda_reproducibility


class RuntimeReproducibilityTest(unittest.TestCase):
    def test_strict_math_attention_and_restore_legacy_mode(self):
        previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cuda.flash_sdp_enabled(),
                    torch.backends.cuda.mem_efficient_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled())
        try:
            configure_cuda_reproducibility(True)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
            self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            self.assertTrue(torch.backends.cuda.math_sdp_enabled())
            configure_cuda_reproducibility(False)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cuda.flash_sdp_enabled())
            self.assertTrue(torch.backends.cuda.mem_efficient_sdp_enabled())
        finally:
            torch.backends.cudnn.benchmark = previous[0]
            torch.backends.cudnn.deterministic = previous[1]
            torch.use_deterministic_algorithms(previous[2], warn_only=previous[3])
            torch.backends.cuda.enable_flash_sdp(previous[4])
            torch.backends.cuda.enable_mem_efficient_sdp(previous[5])
            torch.backends.cuda.enable_math_sdp(previous[6])
