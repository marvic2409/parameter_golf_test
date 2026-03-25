"""Full NeuroModulated Recursive Model — assembles backbone, modulator, and halting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import NeuroModConfig
from .modules.backbone import Embedding, OutputHead, SharedTransformerBlock
from .modules.halting import (
    AttractorHalt, EnergyBudgetHalt, HaltCombiner, InhibitoryDamping,
    LearnedHalt, ModulatorHalt, SynapticDepression,
)
from .modules.modulator import ModulatorNetwork, extract_block_modulation
from .modules.oscillator import OscillatoryGating


def _summarize_modulation(modulation: dict[str, Tensor], batch_size: int, device: torch.device) -> dict[str, Tensor]:
    """Reduce modulation tensors to small per-sample statistics for profiling."""
    feature_parts: list[Tensor] = []
    if "global_scale" in modulation:
        feature_parts.append((modulation["global_scale"] - 1.0).reshape(batch_size, -1).abs())
    if "global_shift" in modulation:
        feature_parts.append(modulation["global_shift"].reshape(batch_size, -1).abs())
    if "layer_scale" in modulation:
        feature_parts.append((modulation["layer_scale"] - 1.0).reshape(batch_size, -1).abs())
    if "layer_shift" in modulation:
        feature_parts.append(modulation["layer_shift"].reshape(batch_size, -1).abs())
    if "channel_gate" in modulation:
        feature_parts.append((modulation["channel_gate"] - 1.0).reshape(batch_size, -1).abs())

    if feature_parts:
        flat = torch.cat(feature_parts, dim=-1)
        magnitude = flat.mean(dim=-1)
        variance = flat.var(dim=-1, unbiased=False)
    else:
        magnitude = torch.zeros(batch_size, device=device)
        variance = torch.zeros(batch_size, device=device)

    if "channel_gate" in modulation:
        gates = modulation["channel_gate"].reshape(batch_size, -1).clamp(1e-6, 1.0 - 1e-6)
        sparsity = (gates < 0.1).float().mean(dim=-1)
        gate_entropy = -(gates * gates.log() + (1.0 - gates) * (1.0 - gates).log()).mean(dim=-1)
    else:
        sparsity = torch.zeros(batch_size, device=device)
        gate_entropy = torch.zeros(batch_size, device=device)

    return {
        "magnitude": magnitude,
        "variance": variance,
        "channel_gate_sparsity": sparsity,
        "channel_gate_entropy": gate_entropy,
    }


class NeuroModRecursiveModel(nn.Module):
    def __init__(self, config: NeuroModConfig):
        super().__init__()
        self.config = config

        # --- Embedding (applied once) ---
        self.embedding = Embedding(config.vocab_size, config.hidden_dim, config.seq_len)

        # --- Shared transformer blocks ---
        self.shared_blocks = nn.ModuleList([
            SharedTransformerBlock(config.hidden_dim, config.num_heads, config.ff_mult)
            for _ in range(config.num_shared_blocks)
        ])

        # --- Modulator ---
        self.modulator = ModulatorNetwork(config)

        # --- Halting mechanisms ---
        if config.use_attractor_halt:
            self.attractor_halt = AttractorHalt(config.attractor_threshold)
        if config.use_learned_halt:
            self.learned_halt = LearnedHalt(config.hidden_dim)
        if config.use_modulator_halt:
            self.modulator_halt = ModulatorHalt()
        if config.use_energy_budget:
            self.energy_halt = EnergyBudgetHalt(config.hidden_dim)
        if config.use_synaptic_depression:
            self.synaptic_depression = SynapticDepression(config.depression_rate)
        if config.use_inhibitory_damping:
            self.inhibitory_damping = InhibitoryDamping()
        if config.use_oscillatory_gating:
            self.oscillatory_gating = OscillatoryGating(config.num_shared_blocks)

        # --- Halt combiner ---
        self.halt_combiner = HaltCombiner(config)

        # --- Inter-iteration normalization (prevents activation explosion in recursive loop) ---
        self.iter_norm = nn.LayerNorm(config.hidden_dim)

        # --- Output head ---
        self.output_head = OutputHead(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        return_details: bool = False,
    ) -> tuple[Tensor, dict]:
        """Forward pass with recursive processing.

        During training: runs all iterations, weights outputs by halt probabilities (ACT).
        During eval: can hard-halt early.

        Returns:
            logits: (B, T, vocab_size)
            details: dict with iteration info, halt signals, etc.
        """
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device

        # 1. Embed
        h = self.embedding(input_ids)  # (B, T, D)

        # 2. Initial modulator context
        input_summary = h.mean(dim=1)  # (B, D)

        # 3. Initialize tracking
        h_prev = h
        energy = torch.full((B, 1), cfg.energy_budget, device=device)
        inhibition_accum = torch.zeros(B, 1, 1, device=device)
        cumulative_halt_prob = torch.zeros(B, 1, device=device)
        epsilon = 1e-2

        hidden_states = []
        halt_probs = []
        iteration_details = []
        depression_schedule = None
        if cfg.use_synaptic_depression:
            exponents = torch.arange(cfg.max_iterations, device=device, dtype=h.dtype)
            base = h.new_tensor(1.0 - self.synaptic_depression.depression_rate)
            depression_schedule = torch.pow(base, exponents)
        oscillation_schedule = None
        if cfg.use_oscillatory_gating:
            oscillation_schedule = self.oscillatory_gating.all_gates(cfg.max_iterations, dtype=h.dtype)

        # 4. Recursive loop
        for i in range(cfg.max_iterations):
            # 4a. Normalize hidden state between iterations (prevents compounding explosion)
            h = self.iter_norm(h)

            # 4b. Generate modulation
            hidden_summary = h.mean(dim=1) if cfg.use_adaptive_modulation else None
            modulation = self.modulator(input_summary, i, hidden_summary)

            # Clamp modulation outputs to safe ranges
            if "global_scale" in modulation:
                modulation["global_scale"] = modulation["global_scale"].clamp(-2.0, 2.0)
            if "global_shift" in modulation:
                modulation["global_shift"] = modulation["global_shift"].clamp(-1.0, 1.0)
            if "layer_scale" in modulation:
                modulation["layer_scale"] = modulation["layer_scale"].clamp(0.1, 3.0)
            if "layer_shift" in modulation:
                modulation["layer_shift"] = modulation["layer_shift"].clamp(-1.0, 1.0)

            # 4c. Apply synaptic depression
            if cfg.use_synaptic_depression:
                depression_mult = depression_schedule[i]
                base_weight_scale = modulation.get("weight_scale")
                if base_weight_scale is None:
                    base_weight_scale = depression_mult.new_tensor(1.0)
                elif not torch.is_tensor(base_weight_scale):
                    base_weight_scale = depression_mult.new_tensor(float(base_weight_scale))
                modulation["weight_scale"] = base_weight_scale * depression_mult
            modulation_stats = _summarize_modulation(modulation, B, device) if return_details else None

            # 4d. Run shared blocks with modulation
            for block_idx, block in enumerate(self.shared_blocks):
                block_mod = extract_block_modulation(modulation, block_idx, cfg)

                # Oscillatory gating
                if cfg.use_oscillatory_gating:
                    osc_gate = oscillation_schedule[i, block_idx]
                    base_residual_scale = block_mod.get("residual_scale")
                    if base_residual_scale is None:
                        base_residual_scale = osc_gate.new_tensor(1.0)
                    elif not torch.is_tensor(base_residual_scale):
                        base_residual_scale = osc_gate.new_tensor(float(base_residual_scale))
                    else:
                        base_residual_scale = base_residual_scale.to(device=osc_gate.device, dtype=osc_gate.dtype)
                    block_mod["residual_scale"] = base_residual_scale * osc_gate

                # Weight scale from depression
                if "weight_scale" in modulation:
                    block_mod["weight_scale"] = modulation["weight_scale"]

                h = block(h, modulation=block_mod)

            # 4d. Inhibitory damping
            if cfg.use_inhibitory_damping:
                h, inhibition_accum = self.inhibitory_damping(h, h_prev, inhibition_accum)

            delta = (h - h_prev).norm(dim=(-2, -1))
            delta_denom = h_prev.norm(dim=(-2, -1)) + 1e-8
            relative_delta = delta / delta_denom
            hidden_norm = h.norm(dim=(-2, -1))

            # 4e. Collect halt signals
            halt_signals = {}
            if cfg.use_attractor_halt:
                halt_signals["attractor"] = self.attractor_halt(h, h_prev)
            if cfg.use_learned_halt:
                halt_signals["learned"] = self.learned_halt(h)
            if cfg.use_modulator_halt:
                halt_signals["modulator"] = self.modulator_halt(modulation)
            if cfg.use_energy_budget:
                cost = self.energy_halt(h)  # (B, 1)
                energy = energy - cost
                halt_signals["energy"] = (energy <= 0).float()

            # 4f. Combine halt signals
            should_halt, halt_prob = self.halt_combiner(halt_signals, batch_size=B, device=device)

            # 4g. ACT bookkeeping
            # Clamp so cumulative doesn't exceed 1
            remainder = 1.0 - cumulative_halt_prob
            used_prob = torch.min(halt_prob, remainder)
            cumulative_halt_prob = cumulative_halt_prob + used_prob

            hidden_states.append(h)
            halt_probs.append(used_prob)

            if return_details:
                iteration_details.append({
                    "halt_signals": {k: v.detach() for k, v in halt_signals.items()},
                    "should_halt": should_halt.detach(),
                    "halt_prob": halt_prob.detach(),
                    "used_prob": used_prob.detach(),
                    "cumulative_halt_prob": cumulative_halt_prob.detach().clone(),
                    "energy": energy.detach().clone() if cfg.use_energy_budget else None,
                    "delta_norm": relative_delta.detach(),
                    "hidden_norm": hidden_norm.detach(),
                    "modulation_stats": {k: v.detach() for k, v in modulation_stats.items()} if modulation_stats else None,
                })

            h_prev = h

            # During eval, hard-halt early if all samples halted
            if not self.training:
                if (cumulative_halt_prob >= 1.0 - epsilon).all():
                    break

        # 5. Weighted output (ACT-style)
        if halt_probs:
            # Ensure weights sum to 1 per sample
            prob_stack = torch.cat(halt_probs, dim=-1)  # (B, num_iters)
            # Add remainder to last iteration
            remainder = torch.clamp_min(1.0 - prob_stack.sum(dim=-1, keepdim=True), 0.0)
            halt_distribution = torch.cat([
                prob_stack[:, :-1],
                prob_stack[:, -1:] + remainder,
            ], dim=-1)

            # Weighted combination
            states = torch.stack(hidden_states, dim=1)  # (B, num_iters, T, D)
            weights = halt_distribution.unsqueeze(-1).unsqueeze(-1)  # (B, num_iters, 1, 1)
            output_h = (states * weights).sum(dim=1)  # (B, T, D)
            step_ids = torch.arange(1, halt_distribution.size(-1) + 1, device=device, dtype=halt_distribution.dtype)
            expected_iterations = (halt_distribution * step_ids.unsqueeze(0)).sum(dim=-1)
        else:
            output_h = h
            halt_distribution = torch.ones(B, 1, device=device, dtype=h.dtype)
            expected_iterations = torch.ones(B, device=device, dtype=h.dtype)

        # 6. Output head
        logits = self.output_head(output_h)

        details = {
            "expected_iterations": expected_iterations,
            "num_iterations": expected_iterations.detach().mean(),
            "iterations_executed": len(hidden_states),
            "halt_probs": halt_probs,
            "halt_distribution": halt_distribution,
            "cumulative_halt_prob": cumulative_halt_prob,
        }
        if return_details:
            details["iteration_details"] = iteration_details
            details["final_hidden_summary"] = output_h.mean(dim=1).detach()

        return logits, details


def compute_loss(
    logits: Tensor,
    targets: Tensor,
    details: dict,
    config: NeuroModConfig,
) -> tuple[Tensor, dict]:
    """Compute total loss: task + ponder cost + ACT remainder loss."""
    task_loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1),
    )

    # Ponder cost
    ponder_cost = config.iteration_cost * details["expected_iterations"].mean()

    # ACT remainder loss — encourages crisp halting decisions
    act_loss = torch.tensor(0.0, device=logits.device)
    if config.use_learned_halt and details["halt_probs"]:
        probs = torch.cat(details["halt_probs"], dim=-1)  # (B, num_iters)
        remainder = torch.clamp_min(1.0 - probs.sum(dim=-1), 0.0)
        act_loss = remainder.mean()

    total_loss = task_loss + ponder_cost + 0.01 * act_loss
    avg_iterations = details["num_iterations"]
    if isinstance(avg_iterations, torch.Tensor):
        avg_iterations = float(avg_iterations.detach().item())
    else:
        avg_iterations = float(avg_iterations)

    return total_loss, {
        "task_loss": task_loss.item(),
        "ponder_cost": float(ponder_cost.detach().item()),
        "act_loss": act_loss.item(),
        "total_loss": total_loss.item(),
        "avg_iterations": avg_iterations,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
