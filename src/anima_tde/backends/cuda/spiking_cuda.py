"""CUDA-accelerated spiking operations with CPU fallback.

Provides drop-in replacements for Python LIF neuron and SDA attention
that use fused CUDA kernels for significant speedup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Try loading compiled CUDA extension
_cuda_ops = None
_kernel_dir = Path(__file__).parent / "kernels"
if _kernel_dir.exists():
    sys.path.insert(0, str(_kernel_dir))
    try:
        import tde_spiking_ops

        _cuda_ops = tde_spiking_ops
    except ImportError:
        pass
    finally:
        sys.path.pop(0)


def has_cuda_ops() -> bool:
    """Check if custom CUDA ops are available."""
    return _cuda_ops is not None


class FusedLIFFunction(torch.autograd.Function):
    """Fused LIF neuron using custom CUDA kernel."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        beta: float,
        threshold: float,
        surrogate_alpha: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold
        ctx.surrogate_alpha = surrogate_alpha
        return _cuda_ops.fused_lif_forward(x.contiguous(), beta, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        # Re-compute membrane potentials for backward
        T = x.shape[0]
        N = x.numel() // T
        x_flat = x.reshape(T, N)
        membrane = torch.zeros(T, N, device=x.device)
        mem = torch.zeros(N, device=x.device)
        for t in range(T):
            mem = ctx.beta * mem + x_flat[t]
            membrane[t] = mem
            spike = (mem >= ctx.threshold).float()
            mem = mem - ctx.threshold * spike

        membrane = membrane.reshape_as(x)
        grad_input = _cuda_ops.fused_lif_backward(
            grad_output.contiguous(),
            membrane.contiguous(),
            ctx.beta,
            ctx.threshold,
            ctx.surrogate_alpha,
        )
        return grad_input, None, None, None


class CUDALIFNeuron(nn.Module):
    """CUDA-accelerated LIF neuron.

    Uses fused CUDA kernel when available, falls back to Python loop.
    """

    def __init__(
        self,
        beta: float = 0.25,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _cuda_ops is not None and x.is_cuda:
            return FusedLIFFunction.apply(
                x, self.beta, self.threshold, self.surrogate_alpha
            )
        # CPU fallback
        return self._python_forward(x)

    def _python_forward(self, x: torch.Tensor) -> torch.Tensor:
        from anima_tde.neurons import LIFNeuron

        lif = LIFNeuron(beta=self.beta, threshold=self.threshold)
        return lif(x)


def fused_sda_attention(
    H: torch.Tensor,
    g_temp_spike: torch.Tensor,
    g_temp_float: torch.Tensor,
    g_ch_spike: torch.Tensor,
    g_ch_float: torch.Tensor,
    g_spa_spike: torch.Tensor,
    g_spa_float: torch.Tensor,
) -> torch.Tensor:
    """Fused Spike-Driven Attention using CUDA kernel.

    Falls back to PyTorch ops on CPU.
    """
    if _cuda_ops is not None and H.is_cuda:
        return _cuda_ops.fused_sda(
            H, g_temp_spike, g_temp_float,
            g_ch_spike, g_ch_float,
            g_spa_spike, g_spa_float,
        )
    # CPU fallback
    return (
        g_temp_spike * g_ch_float * g_spa_spike
        + g_ch_spike * g_spa_float * g_temp_spike
        + g_spa_spike * g_temp_float * g_ch_spike
        + H
    )


def fused_alpha_mix(
    input_feat: torch.Tensor,
    temporal_feat: torch.Tensor,
    alpha_sigmoid: float,
) -> torch.Tensor:
    """Fused alpha mixing for Spiking Encoder."""
    if _cuda_ops is not None and input_feat.is_cuda:
        return _cuda_ops.fused_alpha_mix(
            input_feat, temporal_feat, alpha_sigmoid
        )
    # CPU fallback
    return alpha_sigmoid * input_feat + (1.0 - alpha_sigmoid) * temporal_feat
