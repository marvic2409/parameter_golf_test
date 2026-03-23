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
        num_iterations_used = torch.zeros(B, device=device)

        # 4. Recursive loop
        for i in range(cfg.max_iterations):
            # 4a. Generate modulation
            hidden_summary = h.mean(dim=1) if cfg.use_adaptive_modulation else None
            modulation = self.modulator(input_summary, i, hidden_summary)

            # 4b. Apply synaptic depression
            if cfg.use_synaptic_depression:
                depression_mult = self.synaptic_depression(i)
                modulation["weight_scale"] = modulation.get("weight_scale", 1.0) * depression_mult

            # 4c. Run shared blocks with modulation
            for block_idx, block in enumerate(self.shared_blocks):
                block_mod = extract_block_modulation(modulation, block_idx, cfg)

                # Oscillatory gating
                if cfg.use_oscillatory_gating:
                    osc_gate = self.oscillatory_gating(block_idx, i)
                    block_mod["residual_scale"] = block_mod.get("residual_scale", 1.0) * osc_gate

                # Weight scale from depression
                if "weight_scale" in modulation:
                    block_mod["weight_scale"] = modulation["weight_scale"]

                h = block(h, modulation=block_mod)

            # 4d. Inhibitory damping
            if cfg.use_inhibitory_damping:
                h, inhibition_accum = self.inhibitory_damping(h, h_prev, inhibition_accum)

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
            should_halt, halt_prob = self.halt_combiner(halt_signals)

            # 4g. ACT bookkeeping
            # Clamp so cumulative doesn't exceed 1
            remainder = 1.0 - cumulative_halt_prob
            used_prob = torch.min(halt_prob, remainder)
            cumulative_halt_prob = cumulative_halt_prob + used_prob

            hidden_states.append(h)
            halt_probs.append(used_prob)
            num_iterations_used = num_iterations_used + 1.0

            if return_details:
                iteration_details.append({
                    "halt_signals": {k: v.detach() for k, v in halt_signals.items()},
                    "halt_prob": halt_prob.detach(),
                    "cumulative_halt_prob": cumulative_halt_prob.detach().clone(),
                    "energy": energy.detach().clone() if cfg.use_energy_budget else None,
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
            remainder = 1.0 - prob_stack.sum(dim=-1, keepdim=True)
            prob_stack = torch.cat([
                prob_stack[:, :-1],
                prob_stack[:, -1:] + remainder,
            ], dim=-1)

            # Weighted combination
            states = torch.stack(hidden_states, dim=1)  # (B, num_iters, T, D)
            weights = prob_stack.unsqueeze(-1).unsqueeze(-1)  # (B, num_iters, 1, 1)
            output_h = (states * weights).sum(dim=1)  # (B, T, D)
        else:
            output_h = h

        # 6. Output head
        logits = self.output_head(output_h)

        details = {
            "num_iterations": num_iterations_used.mean().item(),
            "halt_probs": [hp.detach() for hp in halt_probs],
            "cumulative_halt_prob": cumulative_halt_prob.detach(),
        }
        if return_details:
            details["iteration_details"] = iteration_details

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
    ponder_cost = config.iteration_cost * details["num_iterations"]

    # ACT remainder loss — encourages crisp halting decisions
    act_loss = torch.tensor(0.0, device=logits.device)
    if config.use_learned_halt and details["halt_probs"]:
        probs = torch.cat(details["halt_probs"], dim=-1)  # (B, num_iters)
        remainder = (1.0 - probs.sum(dim=-1)).abs()
        act_loss = remainder.mean()

    total_loss = task_loss + ponder_cost + 0.01 * act_loss

    return total_loss, {
        "task_loss": task_loss.item(),
        "ponder_cost": ponder_cost,
        "act_loss": act_loss.item(),
        "total_loss": total_loss.item(),
        "avg_iterations": details["num_iterations"],
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
