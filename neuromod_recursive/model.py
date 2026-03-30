"""Full NeuroModulated Recursive Model — assembles backbone, modulator, and halting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import NeuroModConfig
from .modules.backbone import (
    BigramHashEmbedding,
    CastedLinear,
    DynamicCoordinator,
    Embedding,
    LatentWorkspace,
    OutputHead,
    SharedTransformerBlock,
    SmearGate,
)
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
        self.modulation_enabled = any([
            config.use_global_modulation,
            config.use_layer_modulation,
            config.use_channel_gating,
            config.use_modulator_halt,
        ])
        block_param_count = (
            config.num_shared_blocks
            if config.share_block_weights
            else config.num_shared_blocks * config.max_iterations
        )

        # --- Embedding (applied once) ---
        self.embedding = Embedding(
            config.vocab_size,
            config.hidden_dim,
            config.seq_len,
            use_pos_emb=not config.use_rotary_embeddings,
            init_std=config.tied_embed_init_std,
        )
        self.bigram = (
            BigramHashEmbedding(config.bigram_hash_buckets, config.bigram_hash_dim, config.hidden_dim)
            if config.bigram_hash_buckets > 0
            else None
        )
        self.smear_gate = SmearGate(config.hidden_dim) if config.use_smear_gate else None
        self.latent_workspace = (
            LatentWorkspace(
                config.hidden_dim,
                config.latent_dim,
                config.latent_layers,
                config.latent_memory_slots,
            )
            if config.use_latent_workspace
            else None
        )
        self.slow_blocks = None
        self.fast_to_slow = None
        self.fast_gate = None
        self.dynamic_coordinator = None
        self.slow_to_fast = None
        self.slow_gate = None
        self.fast_residual_scale = None
        self.slow_residual_scale = None

        self.num_encoder_blocks = config.num_shared_blocks // 2
        self.num_decoder_blocks = config.num_shared_blocks - self.num_encoder_blocks
        self.num_skip_weights = (
            min(self.num_encoder_blocks, self.num_decoder_blocks)
            if config.use_block_skip_connections
            else 0
        )
        self.skip_weights = (
            nn.Parameter(torch.ones(self.num_skip_weights, config.hidden_dim, dtype=torch.float32))
            if self.num_skip_weights > 0
            else None
        )

        # --- Shared transformer blocks ---
        self.shared_blocks = nn.ModuleList([
            SharedTransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.num_kv_heads,
                config.ff_mult,
                use_rotary_embeddings=config.use_rotary_embeddings,
                qk_gain_init=config.qk_gain_init,
                use_residual_mix=config.use_residual_mix,
            )
            for _ in range(block_param_count)
        ])
        if config.use_fast_slow_hierarchy:
            slow_block_param_count = (
                config.num_slow_blocks
                if config.share_block_weights
                else config.num_slow_blocks * config.max_iterations
            )
            self.slow_blocks = nn.ModuleList([
                SharedTransformerBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.ff_mult,
                    use_rotary_embeddings=config.use_rotary_embeddings,
                    qk_gain_init=config.qk_gain_init,
                    use_residual_mix=False,
                )
                for _ in range(slow_block_param_count)
            ])
            self.fast_to_slow = CastedLinear(config.hidden_dim, config.hidden_dim, bias=False)
            self.fast_to_slow._zero_init = True
            self.fast_gate = nn.Linear(config.hidden_dim, config.hidden_dim)
            if config.use_dynamic_coordinator:
                self.dynamic_coordinator = DynamicCoordinator(
                    hidden_dim=config.hidden_dim,
                    max_fast_steps=config.slow_update_interval,
                    coordinator_dim=config.coordinator_dim,
                    latent_dim=config.latent_dim if config.use_latent_workspace else None,
                )
            self.slow_to_fast = CastedLinear(config.hidden_dim, config.hidden_dim, bias=False)
            self.slow_to_fast._zero_init = True
            self.slow_gate = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.fast_residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.slow_residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # --- Modulator ---
        self.modulator = ModulatorNetwork(config) if self.modulation_enabled else None

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
        self.iter_norm = nn.Identity()

        # --- Output head ---
        self.output_head = OutputHead(
            config.hidden_dim,
            config.vocab_size,
            tie_embeddings=config.tie_embeddings,
            logit_softcap=config.logit_softcap,
        )
        self._init_weights()
        if self.fast_gate is not None:
            nn.init.zeros_(self.fast_gate.weight)
            nn.init.constant_(self.fast_gate.bias, -2.0)
        if self.slow_gate is not None:
            nn.init.zeros_(self.slow_gate.weight)
            nn.init.constant_(self.slow_gate.bias, -2.0)

    def _init_weights(self) -> None:
        total_layers = max(len(self.shared_blocks), 1)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                    continue
                if module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if name.endswith("out_proj") or name.endswith("ff.proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * total_layers))
            elif isinstance(module, nn.Embedding) and name.endswith("pos_emb"):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def _block_param_index(self, cycle_idx: int, block_idx: int, blocks_per_cycle: int) -> int:
        if self.config.share_block_weights:
            return block_idx
        return cycle_idx * blocks_per_cycle + block_idx

    def _clamp_modulation(self, modulation: dict[str, Tensor]) -> dict[str, Tensor]:
        if "global_scale" in modulation:
            modulation["global_scale"] = modulation["global_scale"].clamp(-2.0, 2.0)
        if "global_shift" in modulation:
            modulation["global_shift"] = modulation["global_shift"].clamp(-1.0, 1.0)
        if "layer_scale" in modulation:
            modulation["layer_scale"] = modulation["layer_scale"].clamp(0.1, 3.0)
        if "layer_shift" in modulation:
            modulation["layer_shift"] = modulation["layer_shift"].clamp(-1.0, 1.0)
        return modulation

    def _apply_weight_scale(
        self,
        modulation: dict[str, Tensor],
        depression_schedule: Tensor | None,
        schedule_idx: int,
    ) -> dict[str, Tensor]:
        if not self.config.use_synaptic_depression or depression_schedule is None:
            return modulation
        depression_mult = depression_schedule[schedule_idx]
        base_weight_scale = modulation.get("weight_scale")
        if base_weight_scale is None:
            base_weight_scale = depression_mult.new_tensor(1.0)
        elif not torch.is_tensor(base_weight_scale):
            base_weight_scale = depression_mult.new_tensor(float(base_weight_scale))
        modulation["weight_scale"] = base_weight_scale * depression_mult
        return modulation

    def _run_block_stack(
        self,
        h: Tensor,
        *,
        blocks: nn.ModuleList,
        num_blocks: int,
        cycle_idx: int,
        x0: Tensor,
        modulation: dict[str, Tensor],
        block_offset: int = 0,
        oscillation_step: int | None = None,
        oscillation_schedule: Tensor | None = None,
        use_skip_connections: bool = False,
    ) -> Tensor:
        cfg = self.config
        skips: list[Tensor] = []
        for block_idx in range(num_blocks):
            param_block_idx = self._block_param_index(cycle_idx, block_idx, num_blocks)
            block = blocks[param_block_idx]
            block_mod = extract_block_modulation(modulation, block_idx, cfg, block_offset=block_offset)

            if use_skip_connections and self.num_skip_weights > 0 and block_idx >= self.num_encoder_blocks:
                decoder_idx = block_idx - self.num_encoder_blocks
                if decoder_idx < self.num_skip_weights and skips:
                    skip_weight = self.skip_weights[decoder_idx].to(dtype=h.dtype)[None, None, :]
                    h = h + skip_weight * skips.pop()

            if oscillation_schedule is not None and oscillation_step is not None:
                osc_gate = oscillation_schedule[oscillation_step, block_idx]
                base_residual_scale = block_mod.get("residual_scale")
                if base_residual_scale is None:
                    base_residual_scale = osc_gate.new_tensor(1.0)
                elif not torch.is_tensor(base_residual_scale):
                    base_residual_scale = osc_gate.new_tensor(float(base_residual_scale))
                else:
                    base_residual_scale = base_residual_scale.to(device=osc_gate.device, dtype=osc_gate.dtype)
                block_mod["residual_scale"] = base_residual_scale * osc_gate

            if "weight_scale" in modulation:
                block_mod["weight_scale"] = modulation["weight_scale"]

            h = block(h, x0=x0, modulation=block_mod)
            if use_skip_connections and block_idx < self.num_encoder_blocks and len(skips) < self.num_skip_weights:
                skips.append(h)
        return h

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
        if self.bigram is not None:
            h = h + self.bigram(input_ids)
        h = F.rms_norm(h, (h.size(-1),))
        if self.smear_gate is not None:
            h = self.smear_gate(h)
        x0 = h

        # 2. Initial modulator context
        input_summary = h.mean(dim=1)  # (B, D)
        latent_state = self.latent_workspace.init_state(input_summary) if self.latent_workspace is not None else None
        prev_summary = input_summary

        # 3. Initialize tracking
        fast_h = h
        slow_h = h
        controller_prev = fast_h if self.slow_blocks is None else 0.5 * (fast_h + slow_h)
        energy = torch.full((B, 1), cfg.energy_budget, device=device)
        inhibition_accum = torch.zeros(B, 1, 1, device=device)
        cumulative_halt_prob = torch.zeros(B, 1, device=device)
        epsilon = 1e-2

        hidden_states = []
        halt_probs = []
        iteration_details = []
        latent_norms = []
        slow_norms = []
        cycle_compute_costs = []
        fast_steps_per_cycle = cfg.slow_update_interval if self.slow_blocks is not None else 1
        total_fast_steps = cfg.max_iterations * fast_steps_per_cycle
        total_modulation_steps = cfg.max_iterations * (fast_steps_per_cycle + (1 if self.slow_blocks is not None else 0))
        equivalent_units_per_cycle = (
            1.0
            if self.slow_blocks is None
            else fast_steps_per_cycle + (cfg.num_slow_blocks / max(cfg.num_shared_blocks, 1))
        )

        depression_schedule = None
        if cfg.use_synaptic_depression:
            exponents = torch.arange(total_modulation_steps, device=device, dtype=fast_h.dtype)
            base = h.new_tensor(1.0 - self.synaptic_depression.depression_rate)
            depression_schedule = torch.pow(base, exponents)
        oscillation_schedule = None
        if cfg.use_oscillatory_gating:
            oscillation_schedule = self.oscillatory_gating.all_gates(total_fast_steps, dtype=fast_h.dtype)
        iteration_schedule = None
        if self.modulator is not None and cfg.use_iteration_encoding:
            iteration_schedule = self.modulator.iteration_features(
                total_modulation_steps,
                device=device,
                dtype=input_summary.dtype,
            )
        modulation_step = 0
        fast_step_idx = 0

        # 4. Recursive loop
        for slow_cycle in range(cfg.max_iterations):
            fast_h = self.iter_norm(fast_h)
            if self.slow_blocks is not None:
                slow_h = self.iter_norm(slow_h)
            latent_residual = None
            latent_readout = None
            slow_summary = None
            slow_modulation_stats = None
            fast_modulation_stats = None
            slow_mix = torch.ones(B, 1, device=device, dtype=fast_h.dtype)
            fast_gate_schedule = torch.ones(B, fast_steps_per_cycle, device=device, dtype=fast_h.dtype)

            fast_summary_pre = fast_h.mean(dim=1)
            if self.slow_blocks is not None:
                slow_summary = slow_h.mean(dim=1)
            controller_summary = (
                fast_summary_pre
                if slow_summary is None
                else 0.5 * (fast_summary_pre + slow_summary)
            )
            if self.latent_workspace is not None and latent_state is not None:
                delta_summary = controller_summary - prev_summary
                latent_state, latent_readout, latent_residual = self.latent_workspace(
                    latent_state,
                    input_summary=input_summary,
                    hidden_summary=controller_summary,
                    delta_summary=delta_summary,
                )
                latent_norms.append(latent_readout.norm(dim=-1).detach())
            prev_summary = controller_summary

            if self.dynamic_coordinator is not None and slow_summary is not None:
                coordinator_outputs = self.dynamic_coordinator(
                    fast_summary_pre,
                    slow_summary,
                    latent_readout=latent_readout,
                )
                slow_mix = coordinator_outputs["slow_mix"].to(dtype=fast_h.dtype)
                fast_gate_schedule = coordinator_outputs["fast_gates"].to(dtype=fast_h.dtype)

            if self.slow_blocks is not None:
                prev_slow_h = slow_h
                slow_seed = slow_h
                fast_to_slow = self.fast_to_slow(fast_summary_pre)
                fast_gate = torch.sigmoid(self.fast_gate(fast_summary_pre))
                slow_seed = slow_seed + (
                    fast_to_slow.unsqueeze(1)
                    * fast_gate.unsqueeze(1)
                    * self.fast_residual_scale.to(dtype=slow_seed.dtype)
                )
                if latent_residual is not None:
                    slow_seed = slow_seed + latent_residual.unsqueeze(1).to(dtype=slow_seed.dtype)

                slow_iter_features = iteration_schedule[modulation_step] if iteration_schedule is not None else None
                if self.modulator is not None:
                    slow_modulation = self.modulator(
                        input_summary,
                        hidden_summary=controller_summary if cfg.use_adaptive_modulation else None,
                        iteration_features=slow_iter_features,
                        latent_state=latent_readout,
                    )
                else:
                    slow_modulation = {}
                self._clamp_modulation(slow_modulation)
                self._apply_weight_scale(slow_modulation, depression_schedule, modulation_step)
                if return_details:
                    slow_modulation_stats = _summarize_modulation(slow_modulation, B, device)
                modulation_step += 1
                slow_candidate = self._run_block_stack(
                    slow_seed,
                    blocks=self.slow_blocks,
                    num_blocks=cfg.num_slow_blocks,
                    cycle_idx=slow_cycle,
                    x0=x0,
                    modulation=slow_modulation,
                    block_offset=cfg.num_shared_blocks,
                    use_skip_connections=False,
                )
                slow_h = prev_slow_h + slow_mix.unsqueeze(1) * (slow_candidate - prev_slow_h)
                slow_summary = slow_h.mean(dim=1)
                slow_norms.append(slow_summary.norm(dim=-1).detach())

            if latent_residual is not None:
                fast_h = fast_h + latent_residual.unsqueeze(1).to(dtype=fast_h.dtype)

            latest_modulation = {}
            for _fast_inner in range(fast_steps_per_cycle):
                fast_h = self.iter_norm(fast_h)
                fast_summary = fast_h.mean(dim=1)
                controller_hidden = (
                    fast_summary
                    if slow_summary is None
                    else 0.5 * (fast_summary + slow_summary)
                )
                iter_features = iteration_schedule[modulation_step] if iteration_schedule is not None else None
                if self.modulator is not None:
                    fast_modulation = self.modulator(
                        input_summary,
                        hidden_summary=controller_hidden if cfg.use_adaptive_modulation else None,
                        iteration_features=iter_features,
                        latent_state=latent_readout,
                    )
                else:
                    fast_modulation = {}
                self._clamp_modulation(fast_modulation)
                self._apply_weight_scale(fast_modulation, depression_schedule, modulation_step)
                if return_details:
                    fast_modulation_stats = _summarize_modulation(fast_modulation, B, device)
                modulation_step += 1
                latest_modulation = fast_modulation

                if slow_summary is not None:
                    slow_bridge = self.slow_to_fast(slow_summary)
                    slow_gate = torch.sigmoid(self.slow_gate(slow_summary))
                    fast_h = fast_h + (
                        slow_bridge.unsqueeze(1)
                        * slow_gate.unsqueeze(1)
                        * self.slow_residual_scale.to(dtype=fast_h.dtype)
                    )

                fast_prev = fast_h
                fast_candidate = self._run_block_stack(
                    fast_h,
                    blocks=self.shared_blocks,
                    num_blocks=cfg.num_shared_blocks,
                    cycle_idx=slow_cycle,
                    x0=x0,
                    modulation=fast_modulation,
                    oscillation_step=fast_step_idx if oscillation_schedule is not None else None,
                    oscillation_schedule=oscillation_schedule,
                    use_skip_connections=True,
                )

                if cfg.use_inhibitory_damping:
                    fast_candidate, inhibition_accum = self.inhibitory_damping(fast_candidate, fast_prev, inhibition_accum)
                fast_gate_step = fast_gate_schedule[:, _fast_inner].view(B, 1, 1).to(dtype=fast_candidate.dtype)
                fast_h = fast_prev + fast_gate_step * (fast_candidate - fast_prev)
                fast_step_idx += 1

            controller_h = fast_h if slow_summary is None else 0.5 * (fast_h + slow_h)
            slow_compute = (
                slow_mix.squeeze(-1) * (cfg.num_slow_blocks / max(cfg.num_shared_blocks, 1))
                if self.slow_blocks is not None
                else torch.zeros(B, device=device, dtype=fast_h.dtype)
            )
            fast_compute = fast_gate_schedule.sum(dim=-1)
            cycle_compute = fast_compute + slow_compute
            cycle_compute_costs.append(cycle_compute)
            delta = (controller_h - controller_prev).norm(dim=(-2, -1))
            delta_denom = controller_prev.norm(dim=(-2, -1)) + 1e-8
            relative_delta = delta / delta_denom
            hidden_norm = controller_h.norm(dim=(-2, -1))

            # 4e. Collect halt signals after each slow cycle.
            halt_signals = {}
            if cfg.use_attractor_halt:
                halt_signals["attractor"] = self.attractor_halt(controller_h, controller_prev)
            if cfg.use_learned_halt:
                halt_signals["learned"] = self.learned_halt(controller_h)
            if cfg.use_modulator_halt:
                halt_signals["modulator"] = self.modulator_halt(latest_modulation)
            if cfg.use_energy_budget:
                cost = self.energy_halt(controller_h)  # (B, 1)
                energy = energy - cost
                halt_signals["energy"] = (energy <= 0).float()

            # 4f. Combine halt signals
            should_halt, halt_prob = self.halt_combiner(halt_signals, batch_size=B, device=device)
            if (slow_cycle + 1) < cfg.min_iterations_before_halt:
                halt_prob = torch.zeros_like(halt_prob)
                should_halt = torch.zeros_like(should_halt)

            # 4g. ACT bookkeeping
            # Clamp so cumulative doesn't exceed 1
            remainder = 1.0 - cumulative_halt_prob
            used_prob = torch.min(halt_prob, remainder)
            cumulative_halt_prob = cumulative_halt_prob + used_prob

            hidden_states.append(controller_h)
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
                    "halt_enabled": torch.full((B, 1), float((slow_cycle + 1) >= cfg.min_iterations_before_halt), device=device),
                    "modulation_stats": {k: v.detach() for k, v in fast_modulation_stats.items()} if fast_modulation_stats else None,
                    "fast_modulation_stats": {k: v.detach() for k, v in fast_modulation_stats.items()} if fast_modulation_stats else None,
                    "slow_modulation_stats": {k: v.detach() for k, v in slow_modulation_stats.items()} if slow_modulation_stats else None,
                    "latent_norm": latent_readout.norm(dim=-1).detach() if latent_readout is not None else None,
                    "latent_residual_norm": latent_residual.norm(dim=-1).detach() if latent_residual is not None else None,
                    "slow_norm": slow_summary.norm(dim=-1).detach() if slow_summary is not None else None,
                    "slow_update": slow_mix.detach(),
                    "fast_micro_steps": fast_gate_schedule.detach(),
                })

            controller_prev = controller_h

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
            expected_slow_cycles = (halt_distribution * step_ids.unsqueeze(0)).sum(dim=-1)
            cycle_compute_tensor = torch.stack(cycle_compute_costs, dim=1)
            expected_iterations = (halt_distribution * cycle_compute_tensor).sum(dim=-1)
        else:
            output_h = fast_h if self.slow_blocks is None else 0.5 * (fast_h + slow_h)
            halt_distribution = torch.ones(B, 1, device=device, dtype=h.dtype)
            expected_slow_cycles = torch.ones(B, device=device, dtype=fast_h.dtype)
            if cycle_compute_costs:
                expected_iterations = cycle_compute_costs[-1]
            else:
                expected_iterations = torch.full((B,), equivalent_units_per_cycle, device=device, dtype=fast_h.dtype)

        # 6. Output head
        logits = self.output_head(
            output_h,
            token_emb_weight=self.embedding.token_emb.weight if cfg.tie_embeddings else None,
        )

        details = {
            "expected_iterations": expected_iterations,
            "expected_slow_cycles": expected_slow_cycles,
            "num_iterations": expected_iterations.detach().mean(),
            "iterations_executed": len(hidden_states),
            "fast_steps_executed": fast_step_idx,
            "slow_cycles_executed": len(hidden_states),
            "equivalent_iterations_per_cycle": (
                torch.stack(cycle_compute_costs, dim=1).mean().detach()
                if cycle_compute_costs
                else torch.tensor(equivalent_units_per_cycle, device=device, dtype=fast_h.dtype)
            ),
            "halt_probs": halt_probs,
            "halt_distribution": halt_distribution,
            "cumulative_halt_prob": cumulative_halt_prob,
            "latent_state_norm": (
                torch.stack(latent_norms, dim=1).mean(dim=1)
                if latent_norms
                else None
            ),
            "slow_state_norm": (
                torch.stack(slow_norms, dim=1).mean(dim=1)
                if slow_norms
                else None
            ),
        }
        if return_details:
            details["iteration_details"] = iteration_details
            details["final_hidden_summary"] = output_h.mean(dim=1).detach()

        return logits, details

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self(input_ids, return_details=False)[0]


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
