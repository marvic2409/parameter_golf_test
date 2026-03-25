"""Modulator network - small side-network generating modulation signals."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import NeuroModConfig


class ModulatorNetwork(nn.Module):
    """Generates modulation signals conditioned on input, iteration, and hidden state."""

    def __init__(self, config: NeuroModConfig):
        super().__init__()
        self.config = config
        mod_dim = config.mod_dim
        hidden_dim = config.hidden_dim
        num_blocks = config.num_shared_blocks

        # Input projection: pool hidden_dim -> mod_dim
        self.input_proj = nn.Linear(hidden_dim, mod_dim)

        # Iteration encoding (sinusoidal, projected)
        if config.use_iteration_encoding:
            self.iter_proj = nn.Linear(mod_dim, mod_dim)

        # Adaptive modulation: project current hidden state
        if config.use_adaptive_modulation:
            self.hidden_proj = nn.Linear(hidden_dim, mod_dim)

        # Compute MLP input size
        mlp_in = mod_dim  # input summary always present
        if config.use_iteration_encoding:
            mlp_in += mod_dim
        if config.use_adaptive_modulation:
            mlp_in += mod_dim

        # Core MLP (2 layers, hidden 64)
        mlp_hidden = 64
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )

        # --- Output heads (conditional on config) ---
        if config.use_global_modulation:
            self.global_head = nn.Linear(mlp_hidden, 2)  # scale, shift

        if config.use_layer_modulation:
            # Per-block: (scale_vec, shift_vec) each of hidden_dim
            self.layer_head = nn.Linear(mlp_hidden, num_blocks * hidden_dim * 2)

        if config.use_channel_gating:
            self.gate_head = nn.Linear(mlp_hidden, num_blocks * hidden_dim)

        if config.use_modulator_halt:
            self.halt_head = nn.Linear(mlp_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize so modulation starts as identity."""
        cfg = self.config
        if cfg.use_global_modulation:
            nn.init.constant_(self.global_head.bias, 0)
            # scale near 1, shift near 0
            with torch.no_grad():
                self.global_head.bias[0] = 1.0  # scale
                self.global_head.bias[1] = 0.0  # shift

        if cfg.use_layer_modulation:
            nn.init.zeros_(self.layer_head.weight)
            with torch.no_grad():
                # Initialize scales to 1, shifts to 0
                bias = self.layer_head.bias.data
                hidden_dim = cfg.hidden_dim
                num_blocks = cfg.num_shared_blocks
                for b in range(num_blocks):
                    offset = b * hidden_dim * 2
                    bias[offset:offset + hidden_dim] = 1.0       # scale
                    bias[offset + hidden_dim:offset + 2 * hidden_dim] = 0.0  # shift

        if cfg.use_channel_gating:
            # Positive bias so sigmoid ~ 0.7
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, 0.85)  # sigmoid(0.85) ≈ 0.7

    def _sinusoidal_encoding(self, iteration: int, dim: int, device: torch.device) -> Tensor:
        positions = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        denom = torch.pow(10000.0, positions / dim)
        phase = torch.full_like(positions, float(iteration)) / denom
        pe = torch.zeros(dim, device=device, dtype=torch.float32)
        pe[0::2] = torch.sin(phase)
        pe[1::2] = torch.cos(phase[: pe[1::2].numel()])
        return pe

    def forward(
        self,
        input_summary: Tensor,     # (B, hidden_dim) — mean-pooled input embeddings
        iteration: int,
        hidden_summary: Optional[Tensor] = None,  # (B, hidden_dim) — mean-pooled current hidden
    ) -> dict[str, Tensor]:
        cfg = self.config
        B = input_summary.shape[0]
        device = input_summary.device

        # Build modulator input
        parts = [self.input_proj(input_summary)]  # (B, mod_dim)

        if cfg.use_iteration_encoding:
            iter_enc = self._sinusoidal_encoding(iteration, cfg.mod_dim, device)
            iter_enc = self.iter_proj(iter_enc).unsqueeze(0).expand(B, -1)
            parts.append(iter_enc)

        if cfg.use_adaptive_modulation and hidden_summary is not None:
            parts.append(self.hidden_proj(hidden_summary))

        mlp_in = torch.cat(parts, dim=-1)
        features = self.mlp(mlp_in)  # (B, 64)

        outputs = {}

        if cfg.use_global_modulation:
            gs = self.global_head(features)  # (B, 2)
            outputs["global_scale"] = gs[:, 0:1].unsqueeze(1)  # (B, 1, 1)
            outputs["global_shift"] = gs[:, 1:2].unsqueeze(1)  # (B, 1, 1)

        if cfg.use_layer_modulation:
            raw = self.layer_head(features)  # (B, num_blocks * hidden_dim * 2)
            raw = raw.view(B, cfg.num_shared_blocks, cfg.hidden_dim * 2)
            layer_scale = raw[:, :, :cfg.hidden_dim]   # (B, num_blocks, hidden_dim)
            layer_shift = raw[:, :, cfg.hidden_dim:]   # (B, num_blocks, hidden_dim)
            outputs["layer_scale"] = layer_scale
            outputs["layer_shift"] = layer_shift

        if cfg.use_channel_gating:
            raw = self.gate_head(features)  # (B, num_blocks * hidden_dim)
            raw = raw.view(B, cfg.num_shared_blocks, cfg.hidden_dim)
            outputs["channel_gate"] = torch.sigmoid(raw)  # (B, num_blocks, hidden_dim)

        if cfg.use_modulator_halt:
            outputs["halt_signal"] = torch.sigmoid(self.halt_head(features))  # (B, 1)

        return outputs


def extract_block_modulation(
    modulation: dict[str, Tensor],
    block_idx: int,
    config: NeuroModConfig,
) -> dict[str, Tensor]:
    """Extract per-block modulation from the full modulation dict."""
    block_mod: dict[str, Tensor] = {}

    if config.use_global_modulation and "global_scale" in modulation:
        block_mod["attn_scale"] = modulation["global_scale"]
        block_mod["attn_shift"] = modulation["global_shift"]
        block_mod["ffn_scale"] = modulation["global_scale"]
        block_mod["ffn_shift"] = modulation["global_shift"]

    if config.use_layer_modulation and "layer_scale" in modulation:
        ls = modulation["layer_scale"][:, block_idx, :].unsqueeze(1)  # (B, 1, D)
        lsh = modulation["layer_shift"][:, block_idx, :].unsqueeze(1)
        # Combine with global if present
        if "attn_scale" in block_mod:
            block_mod["attn_scale"] = block_mod["attn_scale"] * ls
            block_mod["attn_shift"] = block_mod["attn_shift"] + lsh
            block_mod["ffn_scale"] = block_mod["ffn_scale"] * ls
            block_mod["ffn_shift"] = block_mod["ffn_shift"] + lsh
        else:
            block_mod["attn_scale"] = ls
            block_mod["attn_shift"] = lsh
            block_mod["ffn_scale"] = ls
            block_mod["ffn_shift"] = lsh

    if config.use_channel_gating and "channel_gate" in modulation:
        cg = modulation["channel_gate"][:, block_idx, :].unsqueeze(1)  # (B, 1, D)
        block_mod["channel_gate"] = cg

    return block_mod
