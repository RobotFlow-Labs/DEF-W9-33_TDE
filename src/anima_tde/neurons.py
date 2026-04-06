"""LIF neuron implementations for TDE.

Implements Leaky Integrate-and-Fire neurons with surrogate gradient for
direct training of spiking neural networks. Includes standard LIF,
LIF0 (top-k% firing), and LIF1 (dual output).

Reference: arXiv 2512.02447, Section 3 (Spike-Driven Attention)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ATanSurrogate(torch.autograd.Function):
    """Surrogate gradient using arctangent function."""

    alpha: float = 2.0

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (membrane,) = ctx.saved_tensors
        grad = (
            ATanSurrogate.alpha
            / 2
            / (1 + (torch.pi * ATanSurrogate.alpha * (membrane - ctx.threshold)).pow(2))
        )
        return grad_output * grad, None


def spike_function(membrane: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Generate binary spike from membrane potential using surrogate gradient."""
    return ATanSurrogate.apply(membrane, threshold)


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with soft reset.

    At each timestep t:
        membrane[t] = beta * membrane[t-1] + input[t] - V_th * spike[t-1]
        spike[t] = Heaviside(membrane[t] - V_th)  (with surrogate gradient)

    Args:
        beta: Membrane potential decay factor (0, 1]. Default 0.25.
        threshold: Firing threshold V_th. Default 1.0.
        reset: Reset mode, "soft" subtracts V_th, "hard" resets to 0.
    """

    def __init__(
        self,
        beta: float = 0.25,
        threshold: float = 1.0,
        reset: str = "soft",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset = reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input over T timesteps.

        Args:
            x: Input tensor (T, B, C, H, W) or (T, B, C).

        Returns:
            Spike tensor of same shape, values in {0, 1}.
        """
        T = x.shape[0]

        membrane = torch.zeros_like(x[0])
        spikes = []

        for t in range(T):
            # Charge: leak previous membrane + new input
            membrane = self.beta * membrane + x[t]

            # Fire
            spike = spike_function(membrane, self.threshold)
            spikes.append(spike)

            # Reset
            if self.reset == "soft":
                membrane = membrane - self.threshold * spike
            else:
                membrane = membrane * (1.0 - spike)

        return torch.stack(spikes, dim=0)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}, reset={self.reset}"


class LIF0Neuron(nn.Module):
    """LIF0: Top-k% firing strategy without threshold.

    Instead of threshold-based firing, selects the top k% of activations
    to fire. Used in SDA attention branches to maintain sparse activation
    while avoiding hard threshold.

    Args:
        beta: Membrane decay factor. Default 0.25.
        topk_percent: Percentage of activations that fire. Default 50.
    """

    def __init__(self, beta: float = 0.25, topk_percent: int = 50) -> None:
        super().__init__()
        self.beta = beta
        self.topk_percent = topk_percent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input with top-k% firing.

        Args:
            x: Input tensor (T, B, C, ...).

        Returns:
            Spike tensor (T, B, C, ...) with top-k% values set to 1.
        """
        T = x.shape[0]
        membrane = torch.zeros_like(x[0])
        spikes = []

        for t in range(T):
            membrane = self.beta * membrane + x[t]

            # Top-k% firing: flatten all dims after batch, find threshold
            flat = membrane.flatten(start_dim=1)  # (B, D)
            k = max(1, int(flat.shape[1] * self.topk_percent / 100))
            topk_vals, _ = flat.topk(k, dim=1)
            threshold_val = topk_vals[:, -1:]  # (B, 1) — min of top-k

            # Reshape threshold for broadcasting against membrane
            for _ in range(membrane.dim() - 2):
                threshold_val = threshold_val.unsqueeze(-1)

            spike = (membrane >= threshold_val).float()
            spikes.append(spike)

            # Soft reset
            membrane = membrane - membrane * spike

        return torch.stack(spikes, dim=0)


class LIF1Neuron(nn.Module):
    """LIF1: Dual-output neuron with threshold firing.

    Outputs both binary spike and floating-point membrane potential.
    Used in SDA to provide both spike (for accumulation) and float
    (for gating) outputs.

    Args:
        beta: Membrane decay factor. Default 0.25.
        threshold: Firing threshold. Default 1.0.
    """

    def __init__(self, beta: float = 0.25, threshold: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input, return both spike and membrane potential.

        Args:
            x: Input tensor (T, B, C, ...).

        Returns:
            Tuple of (spikes, membrane_potentials), each (T, B, C, ...).
        """
        T = x.shape[0]
        membrane = torch.zeros_like(x[0])
        spikes = []
        membranes = []

        for t in range(T):
            membrane = self.beta * membrane + x[t]
            spike = spike_function(membrane, self.threshold)
            spikes.append(spike)
            membranes.append(membrane.clone())
            membrane = membrane - self.threshold * spike

        return torch.stack(spikes, dim=0), torch.stack(membranes, dim=0)
