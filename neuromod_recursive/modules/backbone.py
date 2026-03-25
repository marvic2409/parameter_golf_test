"""Shared recursive transformer backbone with modulation hooks."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.token_emb(input_ids) + self.pos_emb(positions)


class SharedTransformerBlock(nn.Module):
    """Single pre-norm transformer block with modulation hooks.

    Modulation dict keys (all optional):
      - attn_scale: (B, 1, hidden_dim) or scalar
      - attn_shift: (B, 1, hidden_dim) or scalar
      - ffn_scale:  (B, 1, hidden_dim) or scalar
      - ffn_shift:  (B, 1, hidden_dim) or scalar
      - weight_scale: scalar multiplier on projection weights
      - channel_gate: (B, 1, hidden_dim) in [0,1]
      - residual_scale: scalar multiplier on residual connections
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: float = 2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Attention projections
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Feedforward
        ff_dim = int(hidden_dim * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim, bias=False),
        )

    def forward(self, x: Tensor, modulation: Optional[dict] = None) -> Tensor:
        if modulation is None:
            modulation = {}

        weight_scale = modulation.get("weight_scale")
        if weight_scale is None:
            weight_scale = x.new_tensor(1.0)
        elif not torch.is_tensor(weight_scale):
            weight_scale = x.new_tensor(float(weight_scale))
        else:
            weight_scale = weight_scale.to(device=x.device, dtype=x.dtype)

        residual_scale = modulation.get("residual_scale")
        if residual_scale is None:
            residual_scale = x.new_tensor(1.0)
        elif not torch.is_tensor(residual_scale):
            residual_scale = x.new_tensor(float(residual_scale))
        else:
            residual_scale = residual_scale.to(device=x.device, dtype=x.dtype)

        # --- Pre-attention modulation ---
        h = self.norm1(x)
        if "attn_scale" in modulation:
            h = h * modulation["attn_scale"]
        if "attn_shift" in modulation:
            h = h + modulation["attn_shift"]

        # --- Self-attention ---
        B, T, C = h.shape
        qkv = self.qkv(h) * weight_scale
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn) * weight_scale

        # Channel gating on attention output
        if "channel_gate" in modulation:
            attn_out = attn_out * modulation["channel_gate"]

        x = x + attn_out * residual_scale

        # --- Pre-FFN modulation ---
        h = self.norm2(x)
        if "ffn_scale" in modulation:
            h = h * modulation["ffn_scale"]
        if "ffn_shift" in modulation:
            h = h + modulation["ffn_shift"]

        # --- Feedforward ---
        ff_out = self.ff(h) * weight_scale

        if "channel_gate" in modulation:
            ff_out = ff_out * modulation["channel_gate"]

        x = x + ff_out * residual_scale
        return x


class OutputHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))
