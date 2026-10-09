"""Parameter-efficient visual adaptation used by HARP and LARP.

This module deliberately keeps the PromptKD teacher and text classifier
unchanged. HARP inserts zero-initialized bottleneck residuals in selected
student visual blocks. LARP parametrizes selected attention projections with
zero-initialized low-rank updates while preserving PyTorch's original
``nn.MultiheadAttention`` forward path.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientScale.apply(value, scale)


class _ResidualScale(torch.autograd.Function):
    """MMA-equivalent combined forward/gradient scaling without 1/s overflow."""
    @staticmethod
    def forward(ctx, value, scale):
        return value * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Equivalent to scale * value followed by gradient_scale(..., 1/scale),
        # but never creates an intermediate half-precision gradient / scale.
        return grad_output, None


class _ScaledLinearFunction(torch.autograd.Function):
    """Move residual scaling before FP16 GEMM; retain the registered MMA VJP.

    This is deliberately not the ordinary derivative of scaled linear: as in
    _ResidualScale, parameter-side gradients omit the forward scale. The
    adapter's input-side gradient_scale still applies its original factor.
    No master weights, AMP, clipping, or new optimizer settings are introduced.
    Finite-precision rounding differs from scaling the GEMM output afterward.
    """
    @staticmethod
    def forward(ctx, value, weight, bias, scale):
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        return F.linear(value, weight * scale,
                        None if bias is None else bias * scale)

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        grad_value = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        value_flat = value.reshape(-1, value.shape[-1])
        grad_weight = grad_flat.t().matmul(value_flat) if ctx.needs_input_grad[1] else None
        grad_bias = grad_flat.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_value, grad_weight, grad_bias, None


class ScaledResidualLinear(nn.Linear):
    def __init__(self, in_features, out_features, scale):
        super().__init__(in_features, out_features)
        self.residual_scale = float(scale)

    def forward(self, value):
        return _ScaledLinearFunction.apply(value, self.weight, self.bias,
                                          self.residual_scale)


class _InputScaledLinearFunction(torch.autograd.Function):
    """Preserve the input-only VJP scale without materializing an unsafe dx.

    Forward and parameter-side derivatives are unscaled. Only dx contains
    the existing adapter input factor. In real arithmetic this is identical
    to gradient_scale(value, scale) followed by an ordinary linear layer;
    FP16 rounding is deliberately different and must be disclosed.
    """
    @staticmethod
    def forward(ctx, value, weight, bias, scale):
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        ctx.scale = scale
        return F.linear(value, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        grad_value = grad_output.matmul(weight * ctx.scale) if ctx.needs_input_grad[0] else None
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        value_flat = value.reshape(-1, value.shape[-1])
        grad_weight = grad_flat.t().matmul(value_flat) if ctx.needs_input_grad[1] else None
        grad_bias = grad_flat.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_value, grad_weight, grad_bias, None


class InputScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, scale):
        super().__init__(in_features, out_features)
        self.input_gradient_scale = float(scale)

    def forward(self, value):
        return _InputScaledLinearFunction.apply(value, self.weight, self.bias,
                                               self.input_gradient_scale)


class HighLevelAdapter(nn.Module):
    """MMA-inspired parallel bottleneck with an identity-preserving start."""

    def __init__(self, width: int, bottleneck: int, scale: float, dtype: torch.dtype):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("HARP bottleneck must be positive")
        if scale <= 0:
            raise ValueError("HARP scale must be positive")

        self.scale = float(scale)
        self.down = InputScaledLinear(width, bottleneck, scale)
        self.activation = nn.ReLU(inplace=False)
        self.up = ScaledResidualLinear(bottleneck, width, scale)

        nn.init.kaiming_normal_(self.down.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.down.bias)
        # A zero output projection guarantees that HARP initially reproduces
        # the unmodified PromptKD student exactly.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.to(dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if getattr(self, "diagnostic_disabled", False):
            return torch.zeros_like(value)
        # MMA uses reciprocal gradient scaling so a small residual scale does
        # not also make the adapter parameter gradients vanishingly small.
        # The down layer incorporates the SAME input-only factor before its
        # backward GEMM; parameter gradients do not acquire this factor.
        adapted = self.up(self.activation(self.down(value)))
        return adapted


class QKVLoRAParametrization(nn.Module):
    """LoRA update for packed q/k/v weights of ``nn.MultiheadAttention``."""

    _TARGET_TO_INDEX = {"q": 0, "k": 1, "v": 2}

    def __init__(
        self,
        embed_dim: int,
        targets: Sequence[str],
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LARP rank must be positive")
        invalid = set(targets) - set(self._TARGET_TO_INDEX)
        if invalid:
            raise ValueError(f"Unsupported packed attention targets: {sorted(invalid)}")

        self.embed_dim = int(embed_dim)
        self.targets = tuple(targets)
        self.scaling = float(alpha) / math.sqrt(rank)

        for target in self.targets:
            a = nn.Parameter(torch.empty(rank, embed_dim, dtype=dtype, device=device))
            b = nn.Parameter(torch.zeros(embed_dim, rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter(f"lora_A_{target}", a)
            self.register_parameter(f"lora_B_{target}", b)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if getattr(self, "diagnostic_disabled", False):
            return weight
        if weight.shape != (3 * self.embed_dim, self.embed_dim):
            raise RuntimeError(
                "LARP expects a packed q/k/v projection with shape "
                f"{(3 * self.embed_dim, self.embed_dim)}, got {tuple(weight.shape)}"
            )

        parts = list(weight.split(self.embed_dim, dim=0))
        for target in self.targets:
            index = self._TARGET_TO_INDEX[target]
            a = getattr(self, f"lora_A_{target}")
            b = getattr(self, f"lora_B_{target}")
            parts[index] = parts[index] + (b @ a) * self.scaling
        return torch.cat(parts, dim=0)


class WeightLoRAParametrization(nn.Module):
    """LoRA update for a square linear output projection."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LARP rank must be positive")
        self.expected_shape = (out_features, in_features)
        self.scaling = float(alpha) / math.sqrt(rank)
        self.lora_A_o = nn.Parameter(
            torch.empty(rank, in_features, dtype=dtype, device=device)
        )
        self.lora_B_o = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A_o, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if getattr(self, "diagnostic_disabled", False):
            return weight
        if tuple(weight.shape) != self.expected_shape:
            raise RuntimeError(
                f"Unexpected output projection shape {tuple(weight.shape)}; "
                f"expected {self.expected_shape}"
            )
        return weight + (self.lora_B_o @ self.lora_A_o) * self.scaling


def _validated_layers(layers: Iterable[int], number_of_blocks: int) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(layer) for layer in layers)))
    invalid = [layer for layer in normalized if layer < 0 or layer >= number_of_blocks]
    if invalid:
        raise ValueError(
            f"Adaptation layers {invalid} are outside the valid range "
            f"[0, {number_of_blocks - 1}]"
        )
    if not normalized:
        raise ValueError("At least one adaptation layer is required")
    return normalized


def inject_harp(
    image_encoder: nn.Module,
    layers: Iterable[int],
    bottleneck: int,
    scale: float,
) -> tuple[int, ...]:
    blocks = image_encoder.transformer.resblocks
    selected = _validated_layers(layers, len(blocks))
    for layer_index in selected:
        block = blocks[layer_index]
        width = block.ln_2.weight.numel()
        block.harp_adapter = HighLevelAdapter(
            width=width,
            bottleneck=bottleneck,
            scale=scale,
            # CLIP intentionally keeps LayerNorm in fp32 even in fp16 mode.
            # Adapter linear weights must match attention/activation dtype.
            dtype=block.attn.in_proj_weight.dtype,
        ).to(device=block.attn.in_proj_weight.device)
    return selected


def inject_larp(
    image_encoder: nn.Module,
    layers: Iterable[int],
    rank: int,
    alpha: float,
    targets: Sequence[str],
) -> tuple[int, ...]:
    blocks = image_encoder.transformer.resblocks
    selected = _validated_layers(layers, len(blocks))
    normalized_targets = tuple(dict.fromkeys(target.lower() for target in targets))
    if not normalized_targets:
        raise ValueError("At least one LARP target is required")
    invalid = set(normalized_targets) - {"q", "k", "v", "o"}
    if invalid:
        raise ValueError(f"Unsupported LARP targets: {sorted(invalid)}")

    packed_targets = tuple(target for target in normalized_targets if target != "o")
    for layer_index in selected:
        attention = blocks[layer_index].attn
        if packed_targets:
            weight = attention.in_proj_weight
            parametrization = QKVLoRAParametrization(
                embed_dim=attention.embed_dim,
                targets=packed_targets,
                rank=rank,
                alpha=alpha,
                dtype=weight.dtype,
                device=weight.device,
            )
            parametrize.register_parametrization(
                attention, "in_proj_weight", parametrization, unsafe=True
            )
        if "o" in normalized_targets:
            weight = attention.out_proj.weight
            parametrization = WeightLoRAParametrization(
                out_features=weight.shape[0],
                in_features=weight.shape[1],
                rank=rank,
                alpha=alpha,
                dtype=weight.dtype,
                device=weight.device,
            )
            parametrize.register_parametrization(
                attention.out_proj, "weight", parametrization, unsafe=True
            )
    return selected


def configure_student_adaptation(cfg, image_encoder: nn.Module) -> dict:
    adaptation = cfg.TRAINER.PROMPTKD.ADAPTATION
    method = str(adaptation.TYPE).lower()
    layers = tuple(int(layer) for layer in adaptation.LAYERS)

    if method == "none":
        return {"type": "none", "layers": ()}
    if method == "harp":
        selected = inject_harp(
            image_encoder,
            layers=layers,
            bottleneck=int(adaptation.HARP_DIM),
            scale=float(adaptation.HARP_SCALE),
        )
        return {
            "type": "harp",
            "layers": selected,
            "bottleneck": int(adaptation.HARP_DIM),
            "scale": float(adaptation.HARP_SCALE),
        }
    if method == "larp":
        if float(adaptation.LORA_DROPOUT) != 0:
            raise ValueError(
                "LARP keeps the original MultiheadAttention path and therefore "
                "requires LORA_DROPOUT=0 in this implementation"
            )
        selected = inject_larp(
            image_encoder,
            layers=layers,
            rank=int(adaptation.LORA_R),
            alpha=float(adaptation.LORA_ALPHA),
            targets=tuple(adaptation.LORA_TARGETS),
        )
        return {
            "type": "larp",
            "layers": selected,
            "rank": int(adaptation.LORA_R),
            "alpha": float(adaptation.LORA_ALPHA),
            "targets": tuple(adaptation.LORA_TARGETS),
        }
    raise ValueError(f"Unknown adaptation type: {adaptation.TYPE}")


def adaptation_parameter(name: str) -> bool:
    return "harp_adapter" in name or "lora_A_" in name or "lora_B_" in name
