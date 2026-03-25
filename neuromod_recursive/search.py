"""Evolutionary search harness — MAP-Elites + Speciation + Novelty-driven search.

Every architecture is evaluated AFTER int8 quantization + zlib compression to ensure
it remains viable under the Parameter Golf challenge constraints (16MB artifact, BPB scoring).
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import os
import random
import time
from typing import Optional

import torch

from .config import (
    MutationSettings, NeuroModConfig, mutate, crossover,
    make_all_on_config, make_minimal_config,
    make_modulation_only_config, make_halting_only_config,
    make_random_config, BOOLEAN_PARAMS,
)
from .compression import dequantize_state_dict_int8, quantize_and_measure_model
from .model import NeuroModRecursiveModel
from .train import evaluate_trained_model, train_single_config
from .evaluate import evaluate_model
from .novelty.behavioral import (
    BehavioralProfile, compute_behavioral_profile, generate_diagnostic_probes,
)
from .novelty.map_elites import MAPElitesArchive
from .novelty.speciation import SpeciationManager
from .novelty.novelty import compute_novelty
from .utils import config_to_dict, export_state_dict, get_device, set_seed


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


def _score_from_eval_result(eval_result: dict) -> float:
    if eval_result.get("val_bpb") is not None:
        return float(eval_result["val_bpb"])
    return float(eval_result["val_loss"])


def _snapshot_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in export_state_dict(model).items()
    }


def _build_eval_stages(
    fineweb_setup: Optional[dict],
    screen_val_sequences: Optional[int],
    promote_val_sequences: Optional[int],
    elite_val_sequences: Optional[int],
) -> dict[str, Optional[dict]]:
    if fineweb_setup is None:
        return {"screen": None, "promote": None, "elite": None}

    from .fineweb_eval import make_eval_subset

    total_sequences = (fineweb_setup["val_tokens"].numel() - 1) // fineweb_setup["seq_len"]

    def centered_offset(num_sequences: int | None) -> int:
        if num_sequences is None:
            return 0
        used = min(num_sequences, total_sequences)
        return max(0, (total_sequences - used) // 2)

    def tail_offset(num_sequences: int | None) -> int:
        if num_sequences is None:
            return 0
        used = min(num_sequences, total_sequences)
        return max(0, total_sequences - used)

    return {
        "screen": make_eval_subset(fineweb_setup, screen_val_sequences, sequence_offset=0)
        if screen_val_sequences is not None else fineweb_setup,
        "promote": make_eval_subset(fineweb_setup, promote_val_sequences, sequence_offset=centered_offset(promote_val_sequences))
        if promote_val_sequences is not None else fineweb_setup,
        "elite": make_eval_subset(fineweb_setup, elite_val_sequences, sequence_offset=tail_offset(elite_val_sequences))
        if elite_val_sequences is not None else fineweb_setup,
    }


def _seed_population(
    population_size: int,
    base_config: NeuroModConfig,
    search_space: str,
) -> list[NeuroModConfig]:
    population = [
        make_all_on_config(base_config, search_space=search_space),
        make_minimal_config(base_config, search_space=search_space),
    ]
    if search_space != "halting_only":
        population.append(make_modulation_only_config(base_config, search_space=search_space))
    if search_space != "modulation_only":
        population.append(make_halting_only_config(base_config, search_space=search_space))
    while len(population) < population_size:
        population.append(make_random_config(base_config, search_space=search_space))
    return population[:population_size]


def _linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def _rerank_archive_elites(
    archive: MAPElitesArchive,
    top_k: int,
    rerank_steps: int,
    rerank_seeds: int,
    device: torch.device,
    training_seed: int,
    fineweb_setup: Optional[dict],
    eval_setup: Optional[dict],
    amp_dtype: str | None,
    compile_model: bool,
) -> list[dict]:
    if top_k <= 0 or rerank_seeds <= 0:
        return []

    elite_entries = []
    for archive_rank, (cfg, _, profile) in enumerate(archive.best_configs(top_k), start=1):
        cfg = copy.deepcopy(cfg)
        seed_scores = []
        seed_sizes = []
        seed_iterations = []
        for seed_idx in range(rerank_seeds):
            result = train_single_config(
                cfg,
                num_steps=rerank_steps,
                seed=training_seed + archive_rank * 100 + seed_idx,
                device=device,
                quiet=True,
                fineweb_setup=fineweb_setup,
                eval_setup=eval_setup,
                amp_dtype=amp_dtype,
                compile_model=compile_model,
            )
            model = result["model"]
            quant_obj, size_stats = quantize_and_measure_model(model)
            dequant_state = dequantize_state_dict_int8(quant_obj)
            model.load_state_dict(dequant_state)
            eval_result = evaluate_trained_model(
                model,
                cfg,
                device=device,
                fineweb_setup=eval_setup,
                amp_dtype=amp_dtype,
            )
            seed_scores.append(_score_from_eval_result(eval_result))
            seed_sizes.append(size_stats["zlib_compressed_bytes"])
            seed_iterations.append(result["avg_iterations"])
            del model

        mean_score = sum(seed_scores) / len(seed_scores)
        variance = 0.0
        if len(seed_scores) > 1:
            variance = sum((score - mean_score) ** 2 for score in seed_scores) / len(seed_scores)
        elite_entries.append({
            "archive_rank": archive_rank,
            "config": config_to_dict(cfg),
            "score_mean": mean_score,
            "score_std": math.sqrt(variance),
            "avg_iterations_mean": sum(seed_iterations) / len(seed_iterations),
            "compressed_bytes_mean": sum(seed_sizes) / len(seed_sizes),
            "profile_mean_iterations": profile.mean_iterations,
            "profile_iteration_variance": profile.iteration_variance,
            "seed_scores": seed_scores,
        })

    elite_entries.sort(key=lambda item: item["score_mean"])
    return elite_entries


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
    search_space: str = "motif_only",
    screen_val_sequences: Optional[int] = None,
    promote_top_k: int = 0,
    promote_val_sequences: Optional[int] = None,
    elite_rerank_top_k: int = 0,
    elite_rerank_steps: Optional[int] = None,
    elite_rerank_seeds: int = 1,
    elite_val_sequences: Optional[int] = None,
    amp_dtype: str | None = None,
    compile_model: bool = False,
    compile_search_candidates: bool = False,
    mutation_settings: MutationSettings | None = None,
    exploration_start: float = 1.5,
    exploration_end: float = 0.75,
    random_immigrants: int = 0,
    archive_samples: Optional[int] = None,
) -> MAPElitesArchive:
    """Run the full evolutionary search with MAP-Elites + speciation + novelty."""
    set_seed(seed)
    if device is None:
        device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    archive = MAPElitesArchive()
    speciation_mgr = SpeciationManager(threshold=4.0)
    score_label = "val_bpb" if fineweb_setup is not None else "val_loss"
    promote_top_k = min(max(promote_top_k, 0), population_size)
    elite_rerank_top_k = max(elite_rerank_top_k, 0)
    elite_rerank_steps = elite_rerank_steps or max(training_steps_per_eval * 2, training_steps_per_eval)
    mutation_settings = mutation_settings or MutationSettings()
    archive_samples = max(0, archive_samples if archive_samples is not None else max(2, population_size // 10))
    random_immigrants = max(0, random_immigrants)

    # If using FineWeb, override vocab/seq in all configs
    fw_vocab = fineweb_setup["vocab_size"] if fineweb_setup else None
    fw_seq = fineweb_setup["seq_len"] if fineweb_setup else None
    eval_stages = _build_eval_stages(
        fineweb_setup,
        screen_val_sequences=screen_val_sequences,
        promote_val_sequences=promote_val_sequences,
        elite_val_sequences=elite_val_sequences,
    )

    # Generate fixed diagnostic probes
    probe_vocab = fw_vocab or 512
    probe_seq = min(fw_seq or 64, 64)  # keep probes short for speed
    probes, probe_categories = generate_diagnostic_probes(
        vocab_size=probe_vocab, seq_len=probe_seq, num_probes=500, device=device
    )

    # --- Initialize population ---
    seed_config = copy.deepcopy(base_config) if base_config is not None else NeuroModConfig()
    population = _seed_population(population_size, seed_config, search_space)

    coverage_history = []
    generation_logs = []

    for gen in range(num_generations):
        gen_start = time.time()
        progress = 0.0 if num_generations <= 1 else gen / (num_generations - 1)
        exploration_multiplier = _linear_schedule(exploration_start, exploration_end, progress)
        active_mutation = mutation_settings.scaled(exploration_multiplier)
        elite_count = min(2, population_size)
        active_archive_samples = min(archive_samples, max(0, population_size - elite_count))
        active_random_immigrants = min(
            max(0, int(round(random_immigrants * exploration_multiplier))),
            max(0, population_size - elite_count - active_archive_samples),
        )
        results = []
        promoted_heap: list[tuple[float, int]] = []
        promoted_snapshots: dict[int, dict[str, torch.Tensor]] = {}
        speciation_mgr.clear_members()

        if not quiet:
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{num_generations}")
            print(f"{'='*60}")
            print(
                f"  exploration={exploration_multiplier:.2f} "
                f"mutate(bool={active_mutation.boolean_prob:.2f}, "
                f"cont={active_mutation.continuous_prob:.2f}, "
                f"cont_scale={active_mutation.continuous_scale:.2f}, "
                f"cat={active_mutation.categorical_prob:.2f}) "
                f"immigrants={active_random_immigrants} archive_samples={active_archive_samples}"
            )

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
                eval_setup=eval_stages["screen"],
                amp_dtype=amp_dtype,
                compile_model=compile_search_candidates,
            )
            model = train_result["model"]
            score_pre = train_result["val_bpb"] if train_result.get("val_bpb") is not None else train_result["val_loss"]
            avg_iters = train_result["avg_iterations"]

            # --- Challenge constraint checks ---
            # 1. Quantize once and reuse the object for size measurement and reload.
            quant_obj, size_stats = quantize_and_measure_model(model)
            compressed_bytes = size_stats["zlib_compressed_bytes"]

            # 2. Quantization roundtrip: evaluate AFTER int8 quantize+dequantize
            dequant_state = dequantize_state_dict_int8(quant_obj)
            model.load_state_dict(dequant_state)

            screen_eval = evaluate_trained_model(
                model,
                config,
                device=device,
                fineweb_setup=eval_stages["screen"],
                eval_batches=3,
                amp_dtype=amp_dtype,
            )
            score_post = _score_from_eval_result(screen_eval)

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

            results.append({
                "config": copy.deepcopy(config),
                "fitness": fitness,
                "profile": profile,
                "score_value": score_value,
                "screen_score": score_value,
                "score_pre": score_pre,
                "novelty": novelty,
                "avg_iters": avg_iters,
                "compressed_bytes": compressed_bytes,
                "quant_degradation": quant_degradation,
                "promoted": False,
                "eval_time_seconds": time.time() - eval_start,
            })

            if promote_top_k > 0:
                provisional = (fitness, idx)
                should_snapshot = len(promoted_heap) < promote_top_k or fitness > promoted_heap[0][0]
                if should_snapshot:
                    snapshot = _snapshot_state_dict_cpu(model)
                    if len(promoted_heap) < promote_top_k:
                        heapq.heappush(promoted_heap, provisional)
                    else:
                        _, dropped_idx = heapq.heapreplace(promoted_heap, provisional)
                        promoted_snapshots.pop(dropped_idx, None)
                    promoted_snapshots[idx] = snapshot

            # Free model memory
            del model

            if not quiet:
                mb = compressed_bytes / 1_000_000
                size_ok = "OK" if mb < 16.0 else "OVER"
                print(
                    f"  [{idx + 1}/{len(population)}] "
                    f"screen_{score_label}={score_value:.4f} fitness={fitness:.4f} "
                    f"novelty={novelty:.3f} iters={avg_iters:.1f} "
                    f"size={mb:.2f}MB({size_ok}) qdeg={quant_degradation:.4f} "
                    f"stage=screen ({results[-1]['eval_time_seconds']:.1f}s)"
                )

        for idx in sorted(promoted_snapshots, key=lambda i: results[i]["fitness"], reverse=True):
            entry = results[idx]
            config = entry["config"]
            model = NeuroModRecursiveModel(config).to(device)
            model.load_state_dict(promoted_snapshots[idx])
            promote_eval = evaluate_trained_model(
                model,
                config,
                device=device,
                fineweb_setup=eval_stages["promote"],
                eval_batches=8,
                amp_dtype=amp_dtype,
            )
            promoted_score = _score_from_eval_result(promote_eval)
            entry["score_value"] = promoted_score
            entry["fitness"] = compute_composite_fitness(
                promoted_score,
                entry["avg_iters"],
                0.0,
                entry["novelty"],
                config,
                compressed_bytes=entry["compressed_bytes"],
                quant_degradation=entry["quant_degradation"],
                w_novelty=novelty_weight,
            )
            entry["promoted"] = True
            del model
            if not quiet:
                print(
                    f"    promoted [{idx + 1}/{len(population)}] "
                    f"{score_label}: {entry['screen_score']:.4f}->{promoted_score:.4f} "
                    f"fitness={entry['fitness']:.4f}"
                )

        for entry in results:
            config = entry["config"]
            fitness = entry["fitness"]
            profile = entry["profile"]
            is_new = archive.add(config, fitness, profile)
            species = speciation_mgr.assign_species(config)
            species.update_fitness(fitness)
            entry["species"] = species
            entry["is_new"] = is_new
            if not quiet and entry["promoted"]:
                print(
                    f"      archive species={species.id} "
                    f"{'NEW_NICHE' if is_new else ''}".rstrip()
                )

        # --- Log generation ---
        gen_time = time.time() - gen_start
        archive_stats = archive.stats()
        species_stats = speciation_mgr.stats()
        fitnesses = [entry["fitness"] for entry in results]
        scores = [entry["score_value"] for entry in results]
        novelties = [entry["novelty"] for entry in results]

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
            "promoted_count": sum(1 for entry in results if entry["promoted"]),
            "search_space": search_space,
            "exploration_multiplier": exploration_multiplier,
            "mutation_boolean_prob": active_mutation.boolean_prob,
            "mutation_continuous_prob": active_mutation.continuous_prob,
            "mutation_continuous_scale": active_mutation.continuous_scale,
            "mutation_categorical_prob": active_mutation.categorical_prob,
            "random_immigrants": active_random_immigrants,
            "archive_samples": active_archive_samples,
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
        species_budget = max(0, population_size - elite_count - active_archive_samples - active_random_immigrants)
        offspring_allocation = speciation_mgr.allocate_offspring(species_budget)
        new_population = []

        # Elitism: carry top archive members forward unchanged.
        top_configs = archive.best_configs(elite_count)
        for cfg, _, _ in top_configs:
            new_population.append(copy.deepcopy(cfg))
        species_target_size = elite_count + species_budget

        # Per-species reproduction
        for species, num_offspring in offspring_allocation.items():
            members_ranked = [
                (entry["config"], entry["fitness"])
                for entry in results
                if entry["species"] == species
            ]
            members_ranked.sort(key=lambda x: x[1], reverse=True)

            if not members_ranked:
                continue

            for _ in range(num_offspring):
                if len(new_population) >= species_target_size:
                    break
                if len(members_ranked) >= 2 and random.random() < 0.7:
                    p1 = tournament_select(members_ranked, k=3)
                    p2 = tournament_select(members_ranked, k=3)
                    child = crossover(p1, p2)
                else:
                    parent = tournament_select(members_ranked, k=3)
                    child = mutate(parent, search_space=search_space, settings=active_mutation)

                child = mutate(child, search_space=search_space, settings=active_mutation)
                new_population.append(child)
            if len(new_population) >= species_target_size:
                break

        while len(new_population) < species_target_size:
            if archive.grid:
                parent = archive.sample_parent()
                child = mutate(copy.deepcopy(parent), search_space=search_space, settings=active_mutation)
            else:
                child = make_random_config(seed_config, search_space=search_space)
            new_population.append(child)

        archive_target_size = species_target_size + active_archive_samples
        while len(new_population) < archive_target_size:
            if not archive.grid:
                break
            parent = archive.sample_parent()
            child = mutate(
                mutate(copy.deepcopy(parent), search_space=search_space, settings=active_mutation),
                search_space=search_space,
                settings=active_mutation,
            )
            new_population.append(child)

        immigrant_target_size = archive_target_size + active_random_immigrants
        while len(new_population) < immigrant_target_size:
            new_population.append(make_random_config(seed_config, search_space=search_space))

        while len(new_population) < population_size:
            if archive.grid:
                parent = archive.sample_parent()
                child = mutate(copy.deepcopy(parent), search_space=search_space, settings=active_mutation)
            else:
                child = make_random_config(seed_config, search_space=search_space)
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

    elite_rerank_results = _rerank_archive_elites(
        archive,
        top_k=elite_rerank_top_k,
        rerank_steps=elite_rerank_steps,
        rerank_seeds=elite_rerank_seeds,
        device=device,
        training_seed=seed + num_generations * 1000,
        fineweb_setup=fineweb_setup,
        eval_setup=eval_stages["elite"],
        amp_dtype=amp_dtype,
        compile_model=compile_model,
    )

    if elite_rerank_results and not quiet:
        print("\nElite rerank:")
        for item in elite_rerank_results[:5]:
            print(
                f"  archive_rank={item['archive_rank']} "
                f"score_mean={item['score_mean']:.4f} "
                f"score_std={item['score_std']:.4f} "
                f"bytes={item['compressed_bytes_mean'] / 1_000_000:.2f}MB"
            )

    # Final save
    _save_checkpoint(
        output_dir,
        num_generations - 1,
        archive,
        generation_logs,
        final=True,
        elite_rerank_results=elite_rerank_results,
    )

    return archive


def _save_checkpoint(
    output_dir: str,
    gen: int,
    archive: MAPElitesArchive,
    generation_logs: list,
    final: bool = False,
    elite_rerank_results: Optional[list[dict]] = None,
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

        if elite_rerank_results:
            rerank_path = os.path.join(output_dir, "elite_rerank.json")
            with open(rerank_path, "w") as f:
                json.dump(elite_rerank_results, f, indent=2)
