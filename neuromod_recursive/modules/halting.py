"""Halting mechanisms — attractor, learned/ACT, modulator, energy budget, inhibitory damping, synaptic depression."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import NeuroModConfig


class AttractorHalt(nn.Module):
    """Stop when hidden state change falls below threshold. No learnable params."""

    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold

    def forward(self, h_new: Tensor, h_old: Tensor) -> Tensor:
        # h shape: (B, T, D) — compute per-batch
        delta = (h_new - h_old).norm(dim=(-2, -1))  # (B,)
        denom = h_old.norm(dim=(-2, -1)) + 1e-8      # (B,)
        relative_delta = delta / denom
        return (relative_delta < self.threshold).float().unsqueeze(-1)  # (B, 1)


class LearnedHalt(nn.Module):
    """ACT-style learned halting probability (Graves 2016)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, T, D) — mean-pool over sequence
        pooled = h.mean(dim=1)  # (B, D)
        return torch.sigmoid(self.proj(pooled))  # (B, 1)


class ModulatorHalt(nn.Module):
    """Passes through the halt signal from the modulator network. No extra params."""

    def forward(self, modulation: dict) -> Tensor:
        return modulation.get("halt_signal", torch.zeros(1))  # (B, 1)


class EnergyBudgetHalt(nn.Module):
    """Learned energy cost per iteration; halts when budget depleted."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cost_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        pooled = h.mean(dim=1)  # (B, D)
        return torch.sigmoid(self.cost_head(pooled))  # (B, 1) — energy cost for this iteration


class SynapticDepression(nn.Module):
    """Computes weight depression multiplier. No learnable params."""

    def __init__(self, depression_rate: float = 0.05):
        super().__init__()
        self.depression_rate = depression_rate

    def forward(self, iteration: int) -> float:
        return (1.0 - self.depression_rate) ** iteration


class InhibitoryDamping(nn.Module):
    """Feedback inhibition that damps recurrent activity over time."""

    def __init__(self):
        super().__init__()
        self.inhibition_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_new: Tensor, h_old: Tensor, accumulator: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (damped_h, updated_accumulator)."""
        delta_norm = (h_new - h_old).norm(dim=(-2, -1), keepdim=True)  # (B, 1, 1)
        new_accum = accumulator + delta_norm
        damping = 1.0 / (1.0 + self.inhibition_gain.abs() * new_accum)
        damped_h = h_old + (h_new - h_old) * damping
        return damped_h, new_accum


class HaltCombiner(nn.Module):
    """Combines multiple halt signals into a single decision."""

    def __init__(self, config: NeuroModConfig):
        super().__init__()
        self.mode = config.halt_combination
        num_signals = config.count_active_halt_signals()
        if self.mode == "learned" and num_signals > 0:
            self.combiner = nn.Linear(num_signals, 1)
            nn.init.constant_(self.combiner.bias, -1.0)  # start biased toward not halting
        self.num_signals = num_signals

    def forward(
        self,
        halt_signals: dict[str, Tensor],
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (should_halt_binary, halt_probability).
        Both are (B, 1) tensors."""
        if not halt_signals:
            B = batch_size if batch_size is not None else 1
            device = device if device is not None else torch.device("cpu")
            return torch.zeros(B, 1, device=device), torch.zeros(B, 1, device=device)

        signals = list(halt_signals.values())
        B = signals[0].shape[0]
        device = signals[0].device
        stacked = torch.cat(signals, dim=-1)  # (B, num_signals)

        if self.mode == "any":
            halt_prob = stacked.max(dim=-1, keepdim=True).values
            should_halt = (halt_prob > 0.5).float()

        elif self.mode == "majority":
            binary = (stacked > 0.5).float()
            vote = binary.mean(dim=-1, keepdim=True)
            halt_prob = vote
            should_halt = (vote > 0.5).float()

        elif self.mode == "learned":
            halt_prob = torch.sigmoid(self.combiner(stacked))  # (B, 1)
            should_halt = (halt_prob > 0.5).float()

        else:
            halt_prob = torch.zeros(B, 1, device=device)
            should_halt = torch.zeros(B, 1, device=device)

        return should_halt, halt_prob
