"""Evolutionary search harness — MAP-Elites + Speciation + Novelty-driven search.

Every architecture is evaluated AFTER int8 quantization + zlib compression to ensure
it remains viable under the Parameter Golf challenge constraints (16MB artifact, BPB scoring).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from typing import Optional

import torch

from .config import (
    NeuroModConfig, mutate, crossover,
    make_all_on_config, make_minimal_config,
    make_modulation_only_config, make_halting_only_config,
    make_random_config, BOOLEAN_PARAMS,
)
from .compression import measure_compressed_size, quantize_state_dict_int8, dequantize_state_dict_int8
from .model import NeuroModRecursiveModel, count_parameters
from .train import train_single_config
from .evaluate import evaluate_model
from .novelty.behavioral import (
    BehavioralProfile, compute_behavioral_profile, generate_diagnostic_probes,
)
from .novelty.map_elites import MAPElitesArchive
from .novelty.speciation import SpeciationManager
from .novelty.novelty import compute_novelty
from .utils import set_seed, config_to_dict, save_config, get_device


def compute_composite_fitness(
    score_value: float,
    avg_iterations: float,
    stability: float,
    novelty_score: float,
    config: NeuroModConfig,
    compressed_bytes: int = 0,
    quant_degradation: float = 0.0,
    w_quality: float = 1.0,
    w_novelty: float = 0.5,
    w_efficiency: float = 0.1,
    w_simplicity: float = 0.05,
    w_size_penalty: float = 2.0,
    w_quant_penalty: float = 1.0,
) -> float:
    quality = -score_value
    efficiency = -0.1 * avg_iterations
    stability_bonus = -0.05 * stability
    active_mechanisms = config.count_active_mechanisms()
    simplicity = -active_mechanisms

    # Size penalty: harsh penalty for exceeding 16MB, gentle pressure otherwise
    size_penalty = 0.0
    if compressed_bytes > 0:
        mb = compressed_bytes / 1_000_000
        if mb > 16.0:
            size_penalty = -w_size_penalty * (mb - 16.0)  # big penalty per MB over
        else:
            size_penalty = -0.01 * mb  # gentle pressure to be smaller

    # Quantization degradation penalty: penalize architectures that break after int8
    quant_penalty = -w_quant_penalty * quant_degradation

    return (
        w_quality * quality
        + w_novelty * novelty_score
        + w_efficiency * efficiency
        + w_simplicity * simplicity
        + stability_bonus
        + size_penalty
        + quant_penalty
    )


def tournament_select(
    members: list[tuple[NeuroModConfig, float]],
    k: int = 3,
) -> NeuroModConfig:
    """Tournament selection: pick k random members, return the best."""
    selected = random.sample(members, min(k, len(members)))
    best = max(selected, key=lambda x: x[1])
    return copy.deepcopy(best[0])


def run_evolutionary_search(
    population_size: int = 30,
    num_generations: int = 20,
    training_steps_per_eval: int = 2000,
    novelty_k: int = 15,
    novelty_weight: float = 0.5,
    seed: int = 42,
    output_dir: str = "search_results",
    device: Optional[torch.device] = None,
    quiet: bool = False,
    fineweb_setup: Optional[dict] = None,
    base_config: Optional[NeuroModConfig] = None,
) -> MAPElitesArchive:
    """Run the full evolutionary search with MAP-Elites + speciation + novelty."""
    set_seed(seed)
    if device is None:
        device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    archive = MAPElitesArchive()
    speciation_mgr = SpeciationManager(threshold=4.0)
    score_label = "val_bpb" if fineweb_setup is not None else "val_loss"

    # If using FineWeb, override vocab/seq in all configs
    fw_vocab = fineweb_setup["vocab_size"] if fineweb_setup else None
    fw_seq = fineweb_setup["seq_len"] if fineweb_setup else None

    # Generate fixed diagnostic probes
    probe_vocab = fw_vocab or 512
    probe_seq = min(fw_seq or 64, 64)  # keep probes short for speed
    probes, probe_categories = generate_diagnostic_probes(
        vocab_size=probe_vocab, seq_len=probe_seq, num_probes=500, device=device
    )

    # --- Initialize population ---
    seed_config = copy.deepcopy(base_config) if base_config is not None else NeuroModConfig()
    population = [
        make_all_on_config(seed_config),
        make_minimal_config(seed_config),
        make_modulation_only_config(seed_config),
        make_halting_only_config(seed_config),
    ]
    while len(population) < population_size:
        population.append(make_random_config(seed_config))

    coverage_history = []
    generation_logs = []

    for gen in range(num_generations):
        gen_start = time.time()
        results = []
        speciation_mgr.clear_members()

        if not quiet:
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{num_generations}")
            print(f"{'='*60}")

        for idx, config in enumerate(population):
            eval_start = time.time()

            # Override vocab/seq if using FineWeb
            if fineweb_setup:
                config.vocab_size = fw_vocab
                config.seq_len = fw_seq

            # Train
            train_result = train_single_config(
                config,
                num_steps=training_steps_per_eval,
                seed=seed + gen * 1000 + idx,
                device=device,
                quiet=True,
                fineweb_setup=fineweb_setup,
            )
            model = train_result["model"]
            score_pre = train_result["val_bpb"] if train_result.get("val_bpb") is not None else train_result["val_loss"]
            avg_iters = train_result["avg_iterations"]

            # --- Challenge constraint checks ---
            # 1. Measure compressed size
            size_stats = measure_compressed_size(model)
            compressed_bytes = size_stats["zlib_compressed_bytes"]

            # 2. Quantization roundtrip: evaluate AFTER int8 quantize+dequantize
            quant_obj, _ = quantize_state_dict_int8(model.state_dict())
            dequant_state = dequantize_state_dict_int8(quant_obj)
            model.load_state_dict(dequant_state)

            # Re-evaluate after quantization to get the actual challenge score
            if fineweb_setup is not None:
                from .fineweb_eval import eval_fineweb_bpb
                val_loss_post, val_bpb_post = eval_fineweb_bpb(
                    model=model,
                    val_tokens=fineweb_setup["val_tokens"],
                    seq_len=config.seq_len,
                    vocab_size=config.vocab_size,
                    base_bytes_lut=fineweb_setup["base_bytes_lut"].to(device),
                    has_leading_space_lut=fineweb_setup["has_leading_space_lut"].to(device),
                    is_boundary_token_lut=fineweb_setup["is_boundary_token_lut"].to(device),
                    batch_size=max(1, 65536 // config.seq_len),
                    device=device,
                )
                score_post = val_bpb_post
            else:
                eval_post = evaluate_model(model, config, num_batches=3, device=device)
                val_loss_post = eval_post["val_loss"]
                score_post = val_loss_post

            # Use the post-quantization challenge metric as the real score.
            score_value = score_post
            quant_degradation = max(0.0, score_post - score_pre)

            # Behavioral characterization (on quantized model)
            profile = compute_behavioral_profile(model, probes, probe_categories, config)

            # Novelty score
            novelty = compute_novelty(profile, archive.all_profiles, k=novelty_k)

            # Handle NaN losses
            if math.isnan(score_value) or math.isinf(score_value):
                score_value = 100.0
                avg_iters = config.max_iterations
                quant_degradation = 10.0

            # Composite fitness with challenge constraints
            fitness = compute_composite_fitness(
                score_value, avg_iters, 0.0, novelty, config,
                compressed_bytes=compressed_bytes,
                quant_degradation=quant_degradation,
                w_novelty=novelty_weight,
            )

            # Update archive
            is_new = archive.add(config, fitness, profile)

            # Species assignment
            species = speciation_mgr.assign_species(config)
            species.update_fitness(fitness)

            results.append((config, fitness, profile, species, score_value, novelty))

            # Free model memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            eval_time = time.time() - eval_start
            if not quiet:
                status = "NEW_NICHE" if is_new else ""
                mb = compressed_bytes / 1_000_000
                size_ok = "OK" if mb < 16.0 else "OVER"
                print(
                    f"  [{idx + 1}/{len(population)}] "
                    f"{score_label}={score_value:.4f} fitness={fitness:.4f} "
                    f"novelty={novelty:.3f} iters={avg_iters:.1f} "
                    f"size={mb:.2f}MB({size_ok}) qdeg={quant_degradation:.4f} "
                    f"species={species.id} {status} "
                    f"({eval_time:.1f}s)"
                )

        # --- Log generation ---
        gen_time = time.time() - gen_start
        archive_stats = archive.stats()
        species_stats = speciation_mgr.stats()
        fitnesses = [f for _, f, _, _, _, _ in results]
        scores = [score for _, _, _, _, score, _ in results]
        novelties = [n for _, _, _, _, _, n in results]

        gen_log = {
            "generation": gen + 1,
            "archive": archive_stats,
            "species": species_stats,
            "fitness_best": max(fitnesses),
            "fitness_mean": sum(fitnesses) / len(fitnesses),
            "score_name": score_label,
            "score_best": min(scores),
            "score_mean": sum(scores) / len(scores),
            "novelty_mean": sum(novelties) / len(novelties),
            "time_seconds": gen_time,
            "novelty_weight": novelty_weight,
        }
        generation_logs.append(gen_log)
        coverage_history.append(archive_stats["coverage"])

        if not quiet:
            print(f"\n  Gen {gen + 1} Summary:")
            print(f"    Archive coverage: {archive_stats['coverage']:.1%} ({archive_stats['num_filled']} cells)")
            print(f"    Best fitness: {max(fitnesses):.4f} | Best {score_label}: {min(scores):.4f}")
            print(f"    Species: {species_stats['num_species']} | Novelty mean: {sum(novelties)/len(novelties):.3f}")
            print(f"    Time: {gen_time:.0f}s")

        # --- Selection and reproduction ---
        offspring_allocation = speciation_mgr.allocate_offspring(population_size - 2)
        new_population = []

        # Elitism: carry top 2 from archive
        top_configs = archive.best_configs(2)
        for cfg, _, _ in top_configs:
            new_population.append(copy.deepcopy(cfg))

        # Per-species reproduction
        for species, num_offspring in offspring_allocation.items():
            members_ranked = [
                (cfg, fit)
                for cfg, fit, _, sp, _, _ in results
                if sp == species
            ]
            members_ranked.sort(key=lambda x: x[1], reverse=True)

            if not members_ranked:
                continue

            for _ in range(num_offspring):
                if len(members_ranked) >= 2 and random.random() < 0.7:
                    p1 = tournament_select(members_ranked, k=3)
                    p2 = tournament_select(members_ranked, k=3)
                    child = crossover(p1, p2)
                else:
                    parent = tournament_select(members_ranked, k=3)
                    child = mutate(parent)

                child = mutate(child)
                new_population.append(child)

        # Inject archive samples for exploration
        num_archive_samples = max(2, population_size // 10)
        for _ in range(num_archive_samples):
            if archive.grid:
                parent = archive.sample_parent()
                child = mutate(mutate(copy.deepcopy(parent)))
                new_population.append(child)

        population = new_population[:population_size]

        # --- Adaptive novelty weight ---
        if gen > 5 and len(coverage_history) >= 5:
            recent = coverage_history[-5:]
            if max(recent) - min(recent) < 0.02:
                novelty_weight = min(1.5, novelty_weight * 1.2)
                if not quiet:
                    print(f"  Coverage stalling, novelty_weight -> {novelty_weight:.2f}")

        # Save checkpoint
        _save_checkpoint(output_dir, gen, archive, generation_logs)

    # Final save
    _save_checkpoint(output_dir, num_generations - 1, archive, generation_logs, final=True)

    return archive


def _save_checkpoint(
    output_dir: str,
    gen: int,
    archive: MAPElitesArchive,
    generation_logs: list,
    final: bool = False,
):
    """Save search state to disk."""
    # Save generation logs
    log_path = os.path.join(output_dir, "generation_logs.json")
    with open(log_path, "w") as f:
        json.dump(generation_logs, f, indent=2, default=str)

    # Save top configs from archive
    top = archive.best_configs(10)
    configs_path = os.path.join(output_dir, "top_configs.json")
    configs_data = []
    for cfg, fitness, profile in top:
        configs_data.append({
            "fitness": fitness,
            "config": config_to_dict(cfg),
            "mean_iterations": profile.mean_iterations,
            "iteration_variance": profile.iteration_variance,
        })
    with open(configs_path, "w") as f:
        json.dump(configs_data, f, indent=2)

    # Save archive stats
    stats_path = os.path.join(output_dir, "archive_stats.json")
    with open(stats_path, "w") as f:
        json.dump(archive.stats(), f, indent=2)

    if final:
        # Mechanism frequency across archive
        freq = {param: 0 for param in BOOLEAN_PARAMS}
        total = len(archive.grid)
        for cfg, _, _ in archive.grid.values():
            for param in BOOLEAN_PARAMS:
                if getattr(cfg, param):
                    freq[param] += 1
        if total > 0:
            freq = {k: v / total for k, v in freq.items()}
        freq_path = os.path.join(output_dir, "mechanism_frequency.json")
        with open(freq_path, "w") as f:
            json.dump(freq, f, indent=2)
