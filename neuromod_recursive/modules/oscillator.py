"""Oscillatory gating — gamma/beta-inspired rhythm per transformer block."""

from __future__ import annotations

import math

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

    def forward(self, block_idx: int, iteration: int) -> Tensor:
        """Returns a scalar gate value for the given block at the given iteration."""
        amp = self.amplitude[block_idx]
        freq = self.frequency[block_idx]
        ph = self.phase[block_idx]
        gate = torch.sigmoid(amp * torch.sin(2.0 * math.pi * freq * iteration + ph))
        return gate
