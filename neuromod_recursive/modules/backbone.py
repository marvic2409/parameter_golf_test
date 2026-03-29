"""Shared recursive transformer backbone with modulation hooks."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        use_pos_emb: bool = True,
        init_std: float = 0.005,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim) if use_pos_emb else None
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=init_std)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: Tensor) -> Tensor:
        _, T = input_ids.shape
        x = self.token_emb(input_ids)
        if self.pos_emb is not None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(positions)
        return x


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class ReLUSquaredMLP(nn.Module):
    def __init__(self, hidden_dim: int, ff_mult: float = 2.0):
        super().__init__()
        ff_dim = int(hidden_dim * ff_mult)
        self.fc = CastedLinear(hidden_dim, ff_dim, bias=False)
        self.proj = CastedLinear(ff_dim, hidden_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SmearGate(nn.Module):
    """Blend each token embedding with the previous token's embedding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1.0 - gate) * x + gate * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""

    def __init__(self, bigram_vocab_size: int, bigram_dim: int, hidden_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, hidden_dim, bias=False) if bigram_dim != hidden_dim else None
        if self.proj is not None:
            self.proj._zero_init = True
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, token_ids: Tensor) -> Tensor:
        token_int = token_ids.to(torch.int32)
        mod = max(self.bigram_vocab_size - 1, 1)
        out = torch.empty_like(token_int)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * token_int[..., 1:], 27191 * token_int[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            hidden = self.proj(hidden)
        return hidden * self.scale.to(dtype=hidden.dtype)


class LatentWorkspace(nn.Module):
    """Small recurrent latent workspace for iterative introspection and control.

    The controller can optionally keep a short memory of past slow/controller
    summaries and attend over that memory before applying each recurrent update.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, latent_layers: int = 1, latent_memory_slots: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.latent_layers = latent_layers
        self.latent_memory_slots = latent_memory_slots
        self.input_proj = CastedLinear(hidden_dim, latent_dim, bias=False)
        self.hidden_proj = CastedLinear(hidden_dim, latent_dim, bias=False)
        self.delta_proj = CastedLinear(hidden_dim, latent_dim, bias=False)
        adapter_in_dim = latent_dim * (4 if latent_memory_slots > 0 else 3)
        self.input_adapter = CastedLinear(adapter_in_dim, latent_dim, bias=False)
        self.cores = nn.ModuleList(
            [nn.GRUCell(latent_dim, latent_dim) for _ in range(latent_layers)]
        )
        if latent_memory_slots > 0:
            self.memory_query = CastedLinear(latent_dim, latent_dim, bias=False)
            self.memory_key = CastedLinear(latent_dim, latent_dim, bias=False)
            self.memory_value = CastedLinear(latent_dim, latent_dim, bias=False)
            self.memory_update = CastedLinear(hidden_dim, latent_dim, bias=False)
        self.to_hidden = CastedLinear(latent_dim, hidden_dim, bias=False)
        self.to_gate = nn.Linear(latent_dim, hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.to_hidden.weight)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.constant_(self.to_gate.bias, -2.0)

    def init_state(self, input_summary: Tensor) -> dict[str, Tensor]:
        base = torch.tanh(self.input_proj(input_summary))
        layers = base.unsqueeze(1).expand(-1, self.latent_layers, -1).contiguous().clone()
        state = {"layers": layers}
        if self.latent_memory_slots > 0:
            memory = torch.zeros(
                input_summary.size(0),
                self.latent_memory_slots,
                self.latent_dim,
                device=input_summary.device,
                dtype=input_summary.dtype,
            )
            memory[:, -1, :] = torch.tanh(self.memory_update(input_summary))
            memory_counts = torch.ones(input_summary.size(0), device=input_summary.device, dtype=torch.long)
            state["memory"] = memory
            state["memory_counts"] = memory_counts
        return state

    def _memory_context(
        self,
        query_source: Tensor,
        memory: Tensor,
        memory_counts: Tensor,
    ) -> Tensor:
        if self.latent_memory_slots <= 0:
            return query_source.new_zeros(query_source.shape)

        query = self.memory_query(query_source).unsqueeze(1)
        keys = self.memory_key(memory)
        values = self.memory_value(memory)
        scores = torch.matmul(query, keys.transpose(-1, -2)).squeeze(1)
        positions = torch.arange(self.latent_memory_slots, device=memory.device).unsqueeze(0)
        valid_from = (self.latent_memory_slots - memory_counts).unsqueeze(1)
        valid_mask = positions >= valid_from
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(valid_mask, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        return torch.bmm(attn.unsqueeze(1), values).squeeze(1)

    def _update_memory(
        self,
        memory: Tensor,
        memory_counts: Tensor,
        hidden_summary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        memory_token = torch.tanh(self.memory_update(hidden_summary)).unsqueeze(1)
        updated_memory = torch.cat([memory[:, 1:, :], memory_token], dim=1)
        updated_counts = torch.clamp(memory_counts + 1, max=self.latent_memory_slots)
        return updated_memory, updated_counts

    def forward(
        self,
        latent_state: dict[str, Tensor] | Tensor,
        input_summary: Tensor,
        hidden_summary: Tensor,
        delta_summary: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        if isinstance(latent_state, dict):
            layer_state = latent_state["layers"]
            memory = latent_state.get("memory")
            memory_counts = latent_state.get("memory_counts")
        else:
            layer_state = latent_state
            memory = None
            memory_counts = None

        input_proj = self.input_proj(input_summary)
        hidden_proj = self.hidden_proj(hidden_summary)
        delta_proj = self.delta_proj(delta_summary)
        parts = [input_proj, hidden_proj, delta_proj]
        if self.latent_memory_slots > 0:
            if memory is None or memory_counts is None:
                memory = torch.zeros(
                    input_summary.size(0),
                    self.latent_memory_slots,
                    self.latent_dim,
                    device=input_summary.device,
                    dtype=input_summary.dtype,
                )
                memory_counts = torch.zeros(input_summary.size(0), device=input_summary.device, dtype=torch.long)
            memory_context = self._memory_context(hidden_proj, memory, memory_counts)
            parts.append(memory_context)
        update_in = torch.cat(parts, dim=-1)
        x = torch.tanh(self.input_adapter(update_in))
        next_states: list[Tensor] = []
        for layer_idx, core in enumerate(self.cores):
            prev_state = layer_state[:, layer_idx, :]
            x = core(x, prev_state)
            x = F.rms_norm(x, (x.size(-1),))
            next_states.append(x)
        stacked_state = torch.stack(next_states, dim=1)
        latent_readout = stacked_state[:, -1, :]
        next_state: dict[str, Tensor] = {"layers": stacked_state}
        if self.latent_memory_slots > 0 and memory is not None and memory_counts is not None:
            next_memory, next_counts = self._update_memory(memory, memory_counts, hidden_summary)
            next_state["memory"] = next_memory
            next_state["memory_counts"] = next_counts
        latent_hidden = self.to_hidden(latent_readout)
        latent_gate = torch.sigmoid(self.to_gate(latent_readout))
        latent_residual = latent_hidden * latent_gate * self.residual_scale.to(dtype=latent_hidden.dtype)
        return next_state, latent_readout, latent_residual


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

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        ff_mult: float = 2.0,
        use_rotary_embeddings: bool = True,
        qk_gain_init: float = 1.5,
        use_residual_mix: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        if use_rotary_embeddings and self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even when rotary embeddings are enabled")

        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = CastedLinear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = CastedLinear(hidden_dim, kv_dim, bias=False)
        self.v_proj = CastedLinear(hidden_dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(hidden_dim, hidden_dim, bias=False)
        self.out_proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.attn_scale = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim) if use_rotary_embeddings else None

        self.ff = ReLUSquaredMLP(hidden_dim, ff_mult)
        self.resid_mix = (
            nn.Parameter(torch.stack((torch.ones(hidden_dim), torch.zeros(hidden_dim))).float())
            if use_residual_mix
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, x: Tensor, x0: Tensor | None = None, modulation: Optional[dict] = None) -> Tensor:
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

        if self.resid_mix is not None and x0 is not None:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # --- Pre-attention modulation ---
        h = self.norm1(x)
        if "attn_scale" in modulation:
            h = h * modulation["attn_scale"]
        if "attn_shift" in modulation:
            h = h + modulation["attn_shift"]

        # --- Self-attention ---
        B, T, C = h.shape
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if self.rotary is not None:
            cos, sin = self.rotary(T, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn) * weight_scale

        # Channel gating on attention output
        if "channel_gate" in modulation:
            attn_out = attn_out * modulation["channel_gate"]

        x = x + attn_out * residual_scale * self.attn_scale.to(dtype=x.dtype)[None, None, :]

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

        x = x + ff_out * residual_scale * self.mlp_scale.to(dtype=x.dtype)[None, None, :]
        return x


class OutputHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        tie_embeddings: bool = True,
        logit_softcap: float = 30.0,
    ):
        super().__init__()
        self.norm = RMSNorm()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.proj = None if tie_embeddings else CastedLinear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: Tensor, token_emb_weight: Optional[Tensor] = None) -> Tensor:
        h = self.norm(x)
        if self.tie_embeddings:
            if token_emb_weight is None:
                raise ValueError("token_emb_weight must be provided when embeddings are tied")
            logits = F.linear(h, token_emb_weight)
        else:
            if self.proj is None:
                raise RuntimeError("proj is required when embeddings are untied")
            logits = self.proj(h)
        if self.logit_softcap > 0.0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits
