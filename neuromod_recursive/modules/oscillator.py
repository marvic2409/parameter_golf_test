"""Oscillatory gating - gamma/beta-inspired rhythm per transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class OscillatoryGating(nn.Module):
    """Per-block oscillatory gate: sigmoid(amplitude * sin(2π * freq * iteration + phase))."""

    def __init__(self, num_blocks: int):
        super().__init__()
        self.amplitude = nn.Parameter(torch.ones(num_blocks) * 1.0)
        self.frequency = nn.Parameter(torch.ones(num_blocks) * 0.5)
        self.phase = nn.Parameter(torch.zeros(num_blocks))

    def all_gates(self, num_iterations: int, dtype: torch.dtype | None = None) -> Tensor:
        """Return gates for every iteration/block pair as (num_iterations, num_blocks)."""
        iterations = torch.arange(
            num_iterations,
            device=self.amplitude.device,
            dtype=self.amplitude.dtype,
        ).unsqueeze(1)
        amp = self.amplitude.unsqueeze(0)
        freq = self.frequency.unsqueeze(0)
        phase = self.phase.unsqueeze(0)
        gates = torch.sigmoid(amp * torch.sin((2.0 * torch.pi * freq * iterations) + phase))
        if dtype is not None:
            gates = gates.to(dtype=dtype)
        return gates

    def forward(self, block_idx: int, iteration: int) -> Tensor:
        """Returns a scalar gate value for the given block at the given iteration."""
        return self.all_gates(iteration + 1)[iteration, block_idx]
