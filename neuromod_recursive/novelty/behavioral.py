"""Behavioral characterization system — profiles HOW a network solves tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from ..config import NeuroModConfig


@dataclass
class BehavioralProfile:
    # Iteration dynamics
    mean_iterations: float = 0.0
    iteration_variance: float = 0.0
    iteration_by_difficulty: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    iteration_entropy: float = 0.0

    # Halting behavior
    halt_trigger_rates: Dict[str, float] = field(default_factory=dict)
    halt_timing_profile: List[float] = field(default_factory=list)

    # Modulation dynamics
    modulation_magnitude: float = 0.0
    modulation_variance: float = 0.0
    modulation_iteration_drift: float = 0.0
    channel_gate_sparsity: float = 0.0
    channel_gate_entropy: float = 0.0

    # Hidden state dynamics
    convergence_rate: float = 0.0
    hidden_state_rank: float = 0.0

    # Information flow
    residual_stream_norm_profile: List[float] = field(default_factory=list)

    # Output behavior
    confidence_profile: float = 0.0
    output_diversity: float = 0.0

    # Fixed keys for halt trigger rates to ensure consistent vector length
    HALT_KEYS = ["attractor", "energy", "learned", "modulator"]

    def to_vector(self) -> np.ndarray:
        """Flatten all scalar features into a fixed-length vector for distance computation."""
        parts = [
            self.mean_iterations,
            self.iteration_variance,
            *self.iteration_by_difficulty,
            self.iteration_entropy,
            self.modulation_magnitude,
            self.modulation_variance,
            self.modulation_iteration_drift,
            self.channel_gate_sparsity,
            self.channel_gate_entropy,
            self.convergence_rate,
            self.hidden_state_rank,
            self.confidence_profile,
            self.output_diversity,
        ]
        # Add halt trigger rates (fixed order, 0.0 for missing)
        for key in self.HALT_KEYS:
            parts.append(self.halt_trigger_rates.get(key, 0.0))
        # Pad halt timing to 8
        timing = list(self.halt_timing_profile) + [0.0] * 8
        parts.extend(timing[:8])
        # Pad residual norm profile to 8
        resid = list(self.residual_stream_norm_profile) + [0.0] * 8
        parts.extend(resid[:8])
        return np.array(parts, dtype=np.float32)


def generate_diagnostic_probes(
    vocab_size: int,
    seq_len: int,
    num_probes: int = 500,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, list[str]]:
    """Generate a fixed diagnostic probe set for behavioral characterization.

    Returns (probe_inputs, probe_categories) where categories describe difficulty.
    """
    if num_probes <= 0:
        empty = torch.empty((0, seq_len), dtype=torch.long, device=device)
        return empty, []

    probes = []
    categories = []

    num_categories = 5
    base_n = num_probes // num_categories
    remainder = num_probes % num_categories
    counts = [base_n + (1 if idx < remainder else 0) for idx in range(num_categories)]

    # 1. Uniform random
    for _ in range(counts[0]):
        seq = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]
        probes.append(seq)
        categories.append("random")

    # 2. Deep nesting — high recursion demand
    for _ in range(counts[1]):
        depth = random.randint(8, min(seq_len // 2, 20))
        seq = _make_nested_seq(seq_len, vocab_size, depth)
        probes.append(seq)
        categories.append("hard")

    # 3. Shallow nesting — low recursion demand
    for _ in range(counts[2]):
        depth = random.randint(1, 3)
        seq = _make_nested_seq(seq_len, vocab_size, depth)
        probes.append(seq)
        categories.append("easy")

    # 4. Repetitive patterns
    for _ in range(counts[3]):
        period = random.randint(2, 5)
        base = [random.randint(1, min(vocab_size - 1, 50)) for _ in range(period)]
        seq = [base[i % period] for i in range(seq_len)]
        probes.append(seq)
        categories.append("medium")

    # 5. Adversarial: oscillating sequences
    for _ in range(counts[4]):
        a = random.randint(1, min(vocab_size - 1, 50))
        b = random.randint(1, min(vocab_size - 1, 50))
        noise_rate = 0.1
        seq = []
        for i in range(seq_len):
            if random.random() < noise_rate:
                seq.append(random.randint(1, min(vocab_size - 1, 50)))
            else:
                seq.append(a if i % 2 == 0 else b)
        probes.append(seq)
        categories.append("medium")

    probe_tensor = torch.tensor(probes, dtype=torch.long, device=device)
    return probe_tensor, categories


def _make_nested_seq(seq_len: int, vocab_size: int, depth: int) -> list[int]:
    OPEN, CLOSE = 1, 2
    seq = []
    d = 0
    for _ in range(seq_len):
        if d < depth and random.random() < 0.4:
            seq.append(OPEN)
            d += 1
        elif d > 0 and random.random() < 0.5:
            seq.append(CLOSE)
            d -= 1
        else:
            seq.append(random.randint(3, min(vocab_size - 1, 50)))
    return seq


@torch.no_grad()
def compute_behavioral_profile(
    model: torch.nn.Module,
    probes: Tensor,
    categories: list[str],
    config: NeuroModConfig,
) -> BehavioralProfile:
    """Run model on diagnostic probes and extract behavioral features."""
    model.eval()
    device = next(model.parameters()).device
    probes = probes.to(device)
    profile = BehavioralProfile()

    if len(probes) == 0:
        return profile

    batch_size = min(64, len(probes))
    all_iterations = []
    all_halt_triggers: dict[str, list[float]] = {}
    all_mod_magnitudes = []
    all_mod_variances = []
    all_mod_drifts = []
    all_gate_sparsities = []
    all_gate_entropies = []
    all_residual_norms: list[list[float]] = []
    all_confidences = []
    all_top1 = []
    all_convergence_rates = []
    all_hidden_summaries: list[Tensor] = []
    halt_timing_sum = np.zeros(config.max_iterations, dtype=np.float64)
    halt_timing_count = 0.0

    for start in range(0, len(probes), batch_size):
        batch = probes[start:start + batch_size]
        logits, details = model(batch, return_details=True)

        # Iteration count
        expected_iterations = details.get("expected_iterations")
        if isinstance(expected_iterations, torch.Tensor):
            all_iterations.extend(expected_iterations.detach().cpu().tolist())
        else:
            n_iters = details.get("num_iterations", config.max_iterations)
            if isinstance(n_iters, torch.Tensor):
                n_iters = float(n_iters.detach().cpu().item())
            else:
                n_iters = float(n_iters)
            all_iterations.extend([n_iters] * len(batch))

        halt_distribution = details.get("halt_distribution")
        if isinstance(halt_distribution, torch.Tensor):
            halt_distribution = halt_distribution.detach().cpu().numpy()
            width = min(halt_distribution.shape[1], config.max_iterations)
            halt_timing_sum[:width] += halt_distribution[:, :width].sum(axis=0)
            halt_timing_count += halt_distribution.shape[0]

        # Halt triggers
        if "iteration_details" in details:
            batch_mod_magnitudes = []
            for iter_idx, iter_detail in enumerate(details["iteration_details"]):
                for name, signal in iter_detail.get("halt_signals", {}).items():
                    if name not in all_halt_triggers:
                        all_halt_triggers[name] = []
                    all_halt_triggers[name].append(
                        (signal > 0.5).float().mean().item()
                    )
                mod_stats = iter_detail.get("modulation_stats")
                if mod_stats is not None:
                    mag = mod_stats["magnitude"].cpu().numpy()
                    var = mod_stats["variance"].cpu().numpy()
                    sparse = mod_stats["channel_gate_sparsity"].cpu().numpy()
                    ent = mod_stats["channel_gate_entropy"].cpu().numpy()
                    all_mod_magnitudes.extend(mag.tolist())
                    all_mod_variances.extend(var.tolist())
                    all_gate_sparsities.extend(sparse.tolist())
                    all_gate_entropies.extend(ent.tolist())
                    batch_mod_magnitudes.append(mag)
                delta_norm = iter_detail.get("delta_norm")
                if isinstance(delta_norm, torch.Tensor):
                    all_convergence_rates.extend(delta_norm.cpu().tolist())
                hidden_norm = iter_detail.get("hidden_norm")
                if isinstance(hidden_norm, torch.Tensor):
                    while len(all_residual_norms) <= iter_idx:
                        all_residual_norms.append([])
                    all_residual_norms[iter_idx].extend(hidden_norm.cpu().tolist())
            if len(batch_mod_magnitudes) >= 2:
                stacked = np.stack(batch_mod_magnitudes, axis=1)
                all_mod_drifts.extend(np.abs(np.diff(stacked, axis=1)).mean(axis=1).tolist())

        hidden_summary = details.get("final_hidden_summary")
        if isinstance(hidden_summary, torch.Tensor):
            all_hidden_summaries.append(hidden_summary.cpu())

        # Confidence
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values.mean(dim=-1)  # (B,)
        all_confidences.extend(max_probs.cpu().tolist())

        # Top-1 diversity
        top1 = logits.argmax(dim=-1)[:, -1]  # last position
        all_top1.extend(top1.cpu().tolist())

    # --- Compute profile ---
    iters_arr = np.array(all_iterations, dtype=np.float32)
    profile.mean_iterations = float(iters_arr.mean())
    profile.iteration_variance = float(iters_arr.var())

    # Iteration by difficulty
    diff_map = {"easy": 0, "medium": 1, "hard": 2, "random": 1}
    buckets = [[], [], []]
    for cat, it in zip(categories, all_iterations):
        buckets[diff_map.get(cat, 1)].append(it)
    profile.iteration_by_difficulty = [
        float(np.mean(b)) if b else 0.0 for b in buckets
    ]

    # Iteration entropy
    iter_bins = np.clip(
        np.rint(np.array(all_iterations, dtype=np.float32)).astype(np.int64),
        1,
        config.max_iterations,
    )
    counts = np.bincount(iter_bins, minlength=config.max_iterations + 1)
    counts = counts.astype(np.float64)
    p = counts / (counts.sum() + 1e-8)
    p = p[p > 0]
    profile.iteration_entropy = float(-np.sum(p * np.log(p + 1e-10)))

    # Halt trigger rates
    profile.halt_trigger_rates = {
        name: float(np.mean(vals)) for name, vals in all_halt_triggers.items()
    }

    # Confidence and diversity
    profile.confidence_profile = float(np.mean(all_confidences))
    profile.output_diversity = float(len(set(all_top1)) / max(len(all_top1), 1))

    # Halt timing (which iteration halt fires)
    if halt_timing_count > 0:
        profile.halt_timing_profile = (halt_timing_sum / halt_timing_count).tolist()
    else:
        profile.halt_timing_profile = [0.0] * config.max_iterations

    profile.modulation_magnitude = float(np.mean(all_mod_magnitudes)) if all_mod_magnitudes else 0.0
    profile.modulation_variance = float(np.mean(all_mod_variances)) if all_mod_variances else 0.0
    profile.modulation_iteration_drift = float(np.mean(all_mod_drifts)) if all_mod_drifts else 0.0
    profile.channel_gate_sparsity = float(np.mean(all_gate_sparsities)) if all_gate_sparsities else 0.0
    profile.channel_gate_entropy = float(np.mean(all_gate_entropies)) if all_gate_entropies else 0.0
    profile.convergence_rate = float(np.mean(all_convergence_rates)) if all_convergence_rates else 0.0
    profile.residual_stream_norm_profile = [
        float(np.mean(vals)) if vals else 0.0 for vals in all_residual_norms[: config.max_iterations]
    ]
    if len(profile.residual_stream_norm_profile) < config.max_iterations:
        profile.residual_stream_norm_profile.extend([0.0] * (config.max_iterations - len(profile.residual_stream_norm_profile)))

    if all_hidden_summaries:
        hidden = torch.cat(all_hidden_summaries, dim=0).float().numpy()
        hidden = hidden - hidden.mean(axis=0, keepdims=True)
        if hidden.shape[0] > 1 and np.any(hidden):
            singular_values = np.linalg.svd(hidden, compute_uv=False)
            spectral_norm_sq = float((singular_values[0] ** 2) + 1e-8)
            profile.hidden_state_rank = float((singular_values ** 2).sum() / spectral_norm_sq)

    model.train()
    return profile
