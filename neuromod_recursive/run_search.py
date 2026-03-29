#!/usr/bin/env python3
"""Entry point: launch evolutionary architecture search.

Usage:
  # Smoke test on CPU with synthetic data:
  python -m neuromod_recursive.run_search --smoke-test --device cpu

  # Single config on FineWeb (actual BPB scoring):
  python -m neuromod_recursive.run_search --single --use-fineweb --preset fineweb_medium --steps 5000

  # Full evolutionary search on FineWeb with GPU:
  python -m neuromod_recursive.run_search --use-fineweb --preset fineweb_large --population 30 --generations 20 --steps 2000

  # Multi-GPU:
  torchrun --standalone --nproc_per_node=4 -m neuromod_recursive.run_search --distributed --single --use-fineweb --preset fineweb_medium
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from .config import (
    MutationSettings,
    NeuroModConfig,
    SEARCH_SPACE_SPECS,
    load_config_json,
    make_preset_config,
    normalize_config,
)
from .compression import measure_compressed_size
from .model import NeuroModRecursiveModel, count_parameters
from .search import run_evolutionary_search
from .train import train_single_config, train_distributed
from .utils import set_seed, get_device, format_param_count


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroMod Evolutionary Architecture Search")

    # Search parameters
    parser.add_argument("--population", type=int, default=30, help="Population size per generation")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per evaluation")
    parser.add_argument("--novelty-weight", type=float, default=0.2, help="Initial novelty weight")
    parser.add_argument("--novelty-end-weight", type=float, default=0.05, help="Final novelty weight after annealing")
    parser.add_argument("--novelty-k", type=int, default=15, help="k for k-nearest novelty")
    parser.add_argument(
        "--novelty-transform",
        choices=["identity", "log1p", "sqrt", "clamp2"],
        default="log1p",
        help="Transform applied to raw novelty before fitness weighting.",
    )
    parser.add_argument("--fitness-efficiency-weight", type=float, default=0.02, help="Penalty weight for average iterations in search fitness")
    parser.add_argument("--fitness-simplicity-weight", type=float, default=0.01, help="Penalty weight for active mechanisms in search fitness")
    parser.add_argument("--fitness-quant-penalty", type=float, default=1.0, help="Penalty weight for post-quant degradation in search fitness")
    parser.add_argument("--target-val-bpb", type=float, default=None, help="Optional BPB target used to reward candidates below a baseline and punish candidates above it")
    parser.add_argument("--target-reward-weight", type=float, default=0.0, help="Extra fitness weight for beating target-val-bpb")
    parser.add_argument("--target-penalty-weight", type=float, default=0.0, help="Extra fitness weight for missing target-val-bpb")
    parser.add_argument("--mutation-boolean-prob", type=float, default=0.15, help="Base mutation probability for boolean genes")
    parser.add_argument("--mutation-continuous-prob", type=float, default=0.20, help="Base mutation probability for continuous genes")
    parser.add_argument("--mutation-continuous-scale", type=float, default=0.10, help="Gaussian mutation scale as a fraction of each continuous range")
    parser.add_argument("--mutation-categorical-prob", type=float, default=0.10, help="Base mutation probability for categorical genes")
    parser.add_argument("--exploration-start", type=float, default=1.5, help="Exploration multiplier at generation 1")
    parser.add_argument("--exploration-end", type=float, default=0.75, help="Exploration multiplier at the final generation")
    parser.add_argument("--random-immigrants", type=int, default=None, help="Fresh random architectures injected each generation before annealing")
    parser.add_argument("--archive-samples", type=int, default=None, help="Mutated archive samples injected each generation")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output-dir", type=str, default="search_results", help="Output directory")
    parser.add_argument(
        "--search-space",
        type=str,
        default="motif_only",
        choices=sorted(SEARCH_SPACE_SPECS.keys()),
        help="Which subset of the recursive genome to evolve.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "fineweb_medium", "fineweb_large", "fineweb_competitive", "fineweb_latent_competitive", "fineweb_baseline_parity"],
        help="Base config preset before applying any explicit overrides.",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Load the base config from a JSON file instead of a preset, then apply any CLI overrides.",
    )

    # Mode
    parser.add_argument("--single", action="store_true", help="Train a single config instead of search")
    parser.add_argument("--distributed", action="store_true", help="Use DDP for training")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test (5 pop, 3 gen, 200 steps)")

    # Data
    parser.add_argument("--use-fineweb", action="store_true",
                        help="Use real FineWeb data + BPB scoring (requires downloaded dataset)")
    parser.add_argument("--data-path", type=str, default="./data/datasets/fineweb10B_sp1024",
                        help="Path to FineWeb dataset directory")
    parser.add_argument("--tokenizer-path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to sentencepiece tokenizer model")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size for FineWeb")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length for FineWeb training")

    # Base-config overrides.
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override model hidden size")
    parser.add_argument("--num-heads", type=int, default=None, help="Override number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="Override number of KV heads for grouped attention")
    parser.add_argument("--ff-mult", type=float, default=None, help="Override FFN expansion multiplier")
    parser.add_argument("--bigram-hash-buckets", type=int, default=None, help="Override hashed bigram vocabulary size (0 disables)")
    parser.add_argument("--bigram-hash-dim", type=int, default=None, help="Override hashed bigram embedding dimension")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent workspace width")
    parser.add_argument("--latent-layers", type=int, default=None, help="Override latent workspace depth")
    parser.add_argument("--mod-dim", type=int, default=None, help="Override modulation code dimension")
    parser.add_argument("--num-shared-blocks", type=int, default=None, help="Override number of shared blocks")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max recursive iterations")
    parser.add_argument("--min-iterations-before-halt", type=int, default=None, help="Minimum recursive iterations before halting can trigger")
    parser.add_argument("--untie-block-weights", action="store_true", help="Use unique block weights at each recursion iteration instead of sharing them")
    parser.add_argument("--enable-smear-gate", action="store_true", help="Enable pre-attention token smear gating")
    parser.add_argument("--disable-smear-gate", action="store_true", help="Disable pre-attention token smear gating")
    parser.add_argument("--enable-latent-workspace", action="store_true", help="Enable the recurrent latent workspace")
    parser.add_argument("--disable-latent-workspace", action="store_true", help="Disable the recurrent latent workspace")
    parser.add_argument("--disable-residual-mix", action="store_true", help="Disable trainable residual mixing with the input stream")
    parser.add_argument("--disable-block-skips", action="store_true", help="Disable U-Net style skip connections across recursive blocks")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-process batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate")
    parser.add_argument("--matrix-lr", type=float, default=None, help="Override Muon matrix learning rate")
    parser.add_argument("--scalar-lr", type=float, default=None, help="Override scalar/vector Adam learning rate")
    parser.add_argument("--embed-lr", type=float, default=None, help="Override untied embedding learning rate")
    parser.add_argument("--tied-embed-lr", type=float, default=None, help="Override tied embedding learning rate")
    parser.add_argument("--head-lr", type=float, default=None, help="Override output head learning rate")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override LR warmup steps")
    parser.add_argument("--num-cycles", type=int, default=None, help="Override LR restart cycle count")
    parser.add_argument("--min-lr-ratio", type=float, default=None, help="Override LR floor ratio")
    parser.add_argument("--iteration-cost", type=float, default=None, help="Override recursive ponder cost")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Clip gradient norm to stabilize longer runs (0 disables)")
    parser.add_argument("--eval-stride", type=int, default=None, help="Use sliding-window BPB evaluation with this stride (0 disables)")
    parser.add_argument("--enable-swa", action="store_true", help="Enable stochastic weight averaging before final evaluation")
    parser.add_argument("--disable-swa", action="store_true", help="Disable stochastic weight averaging before final evaluation")
    parser.add_argument("--swa-start-frac", type=float, default=None, help="Begin SWA once LR scale falls below this fraction")
    parser.add_argument("--swa-every", type=int, default=None, help="Capture an SWA checkpoint every N steps")
    parser.add_argument("--best-checkpoint-every", type=int, default=None, help="Subset-eval cadence for restoring the best checkpoint in single runs (0 disables)")
    parser.add_argument("--best-checkpoint-val-seqs", type=int, default=None, help="Validation sequences used for best-checkpoint selection in single runs")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["none", "bf16", "fp16"],
        help="Autocast dtype for CUDA training/eval.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile on fixed-shape training paths (single runs and elite rerank in search mode).",
    )
    parser.add_argument(
        "--compile-search-candidates",
        action="store_true",
        help="Also compile every mutated search candidate. Usually slower due to per-candidate compile overhead.",
    )

    # Staged search controls.
    parser.add_argument("--screen-val-seqs", type=int, default=None, help="Validation sequences for cheap screening eval")
    parser.add_argument("--promote-top-k", type=int, default=None, help="Promote top-K screen candidates each generation")
    parser.add_argument("--quality-promote-top-k", type=int, default=None, help="Reserve this many promotion slots for the lowest raw screen score")
    parser.add_argument("--promote-val-seqs", type=int, default=None, help="Validation sequences for promoted candidates")
    parser.add_argument("--elite-rerank-top-k", type=int, default=None, help="Final archive elites to retrain from scratch")
    parser.add_argument("--elite-rerank-steps", type=int, default=None, help="Training steps for elite reranking")
    parser.add_argument("--elite-rerank-seeds", type=int, default=2, help="Seeds per elite rerank candidate")
    parser.add_argument("--elite-val-seqs", type=int, default=None, help="Validation sequences for elite rerank eval")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, cuda:0, etc.")

    return parser.parse_args()


def build_base_config(args) -> NeuroModConfig:
    config = load_config_json(args.config_json) if args.config_json else make_preset_config(args.preset)
    overrides = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "ff_mult": args.ff_mult,
        "bigram_hash_buckets": args.bigram_hash_buckets,
        "bigram_hash_dim": args.bigram_hash_dim,
        "latent_dim": args.latent_dim,
        "latent_layers": args.latent_layers,
        "mod_dim": args.mod_dim,
        "num_shared_blocks": args.num_shared_blocks,
        "max_iterations": args.max_iterations,
        "min_iterations_before_halt": args.min_iterations_before_halt,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "matrix_lr": args.matrix_lr,
        "scalar_lr": args.scalar_lr,
        "embed_lr": args.embed_lr,
        "tied_embed_lr": args.tied_embed_lr,
        "head_lr": args.head_lr,
        "warmup_steps": args.warmup_steps,
        "num_cycles": args.num_cycles,
        "min_lr_ratio": args.min_lr_ratio,
        "iteration_cost": args.iteration_cost,
        "grad_clip_norm": args.grad_clip_norm,
        "eval_stride": args.eval_stride,
        "swa_start_frac": args.swa_start_frac,
        "swa_every": args.swa_every,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    if args.untie_block_weights:
        config.share_block_weights = False
    if args.enable_smear_gate:
        config.use_smear_gate = True
    if args.disable_smear_gate:
        config.use_smear_gate = False
    if args.enable_latent_workspace:
        config.use_latent_workspace = True
    if args.disable_latent_workspace:
        config.use_latent_workspace = False
    if args.disable_residual_mix:
        config.use_residual_mix = False
    if args.disable_block_skips:
        config.use_block_skip_connections = False
    if args.enable_swa:
        config.swa_enabled = True
    if args.disable_swa:
        config.swa_enabled = False
    return normalize_config(config)


def main():
    args = parse_args()

    # Smoke test overrides
    if args.smoke_test:
        args.population = 5
        args.generations = 3
        args.steps = 200

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    base_config = build_base_config(args)
    compile_search_candidates = args.compile_model and args.compile_search_candidates
    compile_elite_rerank = args.compile_model
    mutation_settings = MutationSettings(
        boolean_prob=args.mutation_boolean_prob,
        continuous_prob=args.mutation_continuous_prob,
        continuous_scale=args.mutation_continuous_scale,
        categorical_prob=args.mutation_categorical_prob,
    )

    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({getattr(torch.cuda.get_device_properties(i), 'total_memory', getattr(torch.cuda.get_device_properties(i), 'total_mem', 0)) / 1e9:.1f} GB)")

    # --- FineWeb setup ---
    fineweb_setup = None
    if args.use_fineweb:
        from .fineweb_eval import setup_fineweb_eval
        print(f"\nSetting up FineWeb evaluation...")
        fineweb_setup = setup_fineweb_eval(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            device=device,
        )
        print(f"  FineWeb ready: vocab={args.vocab_size}, seq_len={args.seq_len}")

        total_val_seqs = (fineweb_setup["val_tokens"].numel() - 1) // fineweb_setup["seq_len"]
        if args.screen_val_seqs is None:
            args.screen_val_seqs = min(1024, total_val_seqs)
        if args.promote_val_seqs is None:
            args.promote_val_seqs = min(4096, total_val_seqs)
        if args.promote_top_k is None:
            args.promote_top_k = max(1, args.population // 5)
        if args.quality_promote_top_k is None:
            args.quality_promote_top_k = min(1, args.promote_top_k)
        if args.elite_rerank_top_k is None:
            args.elite_rerank_top_k = min(3, args.population)
        if args.elite_rerank_steps is None:
            args.elite_rerank_steps = max(args.steps * 2, args.steps)
        if args.single and args.best_checkpoint_every is None and args.steps >= 5000:
            args.best_checkpoint_every = max(1000, args.steps // 20)
        if args.single and args.best_checkpoint_every and args.best_checkpoint_val_seqs is None:
            args.best_checkpoint_val_seqs = min(1024, total_val_seqs)
        if args.target_val_bpb is None and args.preset == "fineweb_baseline_parity":
            args.target_val_bpb = 1.22436570
        if args.target_val_bpb is not None and args.target_reward_weight == 0.0 and args.target_penalty_weight == 0.0:
            args.target_reward_weight = 2.0
            args.target_penalty_weight = 1.0
    else:
        if args.promote_top_k is None:
            args.promote_top_k = 0
        if args.quality_promote_top_k is None:
            args.quality_promote_top_k = 0
        if args.elite_rerank_top_k is None:
            args.elite_rerank_top_k = 0

    if args.random_immigrants is None:
        args.random_immigrants = 1 if args.population >= 8 else 0
    if args.archive_samples is None:
        args.archive_samples = max(2, args.population // 10)

    preview_config = NeuroModConfig(**vars(base_config))
    if fineweb_setup:
        preview_config.vocab_size = fineweb_setup["vocab_size"]
        preview_config.seq_len = fineweb_setup["seq_len"]
    preview_model = NeuroModRecursiveModel(preview_config)
    preview_params = count_parameters(preview_model)
    preview_size_stats = measure_compressed_size(preview_model)
    preview_size_mb = preview_size_stats["zlib_compressed_bytes"] / 1_000_000
    del preview_model

    print(
        f"Base config: source={Path(args.config_json).name if args.config_json else args.preset} "
        f"hidden_dim={base_config.hidden_dim} "
        f"heads={base_config.num_heads}/{base_config.num_kv_heads} ff_mult={base_config.ff_mult} "
        f"bigram={base_config.bigram_hash_buckets}x{base_config.bigram_hash_dim} "
        f"latent={base_config.use_latent_workspace}:{base_config.latent_dim}x{base_config.latent_layers} "
        f"shared_blocks={base_config.num_shared_blocks} max_iterations={base_config.max_iterations} "
        f"min_halt={base_config.min_iterations_before_halt} "
        f"shared_weights={base_config.share_block_weights} "
        f"smear={base_config.use_smear_gate} resid_mix={base_config.use_residual_mix} "
        f"skips={base_config.use_block_skip_connections} eval_stride={base_config.eval_stride} "
        f"swa={base_config.swa_enabled} "
        f"batch_size={base_config.batch_size} lr={base_config.lr:g} "
        f"params={format_param_count(preview_params)} size={preview_size_mb:.2f}MB"
    )

    # --- Single config training ---
    if args.single:
        print("\n--- Training single default config ---")
        config = NeuroModConfig(**vars(base_config))
        if fineweb_setup:
            config.vocab_size = fineweb_setup["vocab_size"]
            config.seq_len = fineweb_setup["seq_len"]
        model = NeuroModRecursiveModel(config)
        params = count_parameters(model)
        size_stats = measure_compressed_size(model)
        print(f"Parameters: {format_param_count(params)} | size={size_stats['zlib_compressed_bytes'] / 1_000_000:.2f}MB")
        del model

        if args.distributed:
            result = train_distributed(
                config, num_steps=args.steps, seed=args.seed,
                fineweb_setup=fineweb_setup,
                amp_dtype=args.amp_dtype,
                compile_model=args.compile_model,
            )
        else:
            result = train_single_config(
                config, num_steps=args.steps, seed=args.seed, device=device,
                fineweb_setup=fineweb_setup,
                amp_dtype=args.amp_dtype,
                compile_model=args.compile_model,
                best_checkpoint_every=max(0, args.best_checkpoint_every or 0),
                best_checkpoint_val_sequences=args.best_checkpoint_val_seqs,
            )

        print(f"\nFinal val_loss: {result['val_loss']:.4f}")
        if result.get("val_bpb") is not None:
            print(f"Final val_bpb:  {result['val_bpb']:.4f}  (this is the challenge score)")
        if result.get("compressed_mb") is not None:
            print(f"Final size:     {result['compressed_mb']:.2f}MB")
        return

    # --- Evolutionary search ---
    print(f"\n--- Evolutionary Search ---")
    print(f"Population: {args.population} | Generations: {args.generations}")
    print(f"Steps/eval: {args.steps} | Novelty weight: {args.novelty_weight}")
    if compile_search_candidates:
        compile_mode = "all-candidates"
    elif compile_elite_rerank:
        compile_mode = "rerank-only"
    else:
        compile_mode = "off"
    print(f"Search space: {args.search_space} | AMP: {args.amp_dtype} | compile={compile_mode}")
    if args.compile_model and not compile_search_candidates:
        print("Compile note: mutated search candidates stay eager to avoid CPU-bound torch.compile churn.")
    print(
        f"Exploration: start={args.exploration_start:.2f} end={args.exploration_end:.2f} "
        f"mutate(bool={mutation_settings.boolean_prob:.2f}, cont={mutation_settings.continuous_prob:.2f}, "
        f"cont_scale={mutation_settings.continuous_scale:.2f}, cat={mutation_settings.categorical_prob:.2f}) "
        f"immigrants={args.random_immigrants} archive_samples={args.archive_samples}"
    )
    print(
        f"Fitness: novelty={args.novelty_weight:.2f}->{args.novelty_end_weight:.2f} "
        f"transform={args.novelty_transform} eff={args.fitness_efficiency_weight:.2f} "
        f"simplicity={args.fitness_simplicity_weight:.2f} quant={args.fitness_quant_penalty:.2f}"
    )
    if args.target_val_bpb is not None:
        print(
            f"Target shaping: val_bpb<={args.target_val_bpb:.6f} "
            f"reward={args.target_reward_weight:.2f} penalty={args.target_penalty_weight:.2f}"
        )
    if fineweb_setup:
        print(
            f"Staged eval: screen={args.screen_val_seqs} seqs | "
            f"promote_top_k={args.promote_top_k} quality_promote={args.quality_promote_top_k} "
            f"promote_eval={args.promote_val_seqs} seqs | "
            f"elite_top_k={args.elite_rerank_top_k} elite_steps={args.elite_rerank_steps} "
            f"elite_seeds={args.elite_rerank_seeds}"
        )
    print(f"Data: {'FineWeb (real BPB)' if fineweb_setup else 'Synthetic'}")
    print(f"Output: {args.output_dir}")

    t0 = time.time()
    archive = run_evolutionary_search(
        population_size=args.population,
        num_generations=args.generations,
        training_steps_per_eval=args.steps,
        novelty_k=args.novelty_k,
        novelty_weight=args.novelty_weight,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        fineweb_setup=fineweb_setup,
        base_config=base_config,
        search_space=args.search_space,
        screen_val_sequences=args.screen_val_seqs,
        promote_top_k=args.promote_top_k,
        quality_promote_top_k=args.quality_promote_top_k,
        promote_val_sequences=args.promote_val_seqs,
        elite_rerank_top_k=args.elite_rerank_top_k,
        elite_rerank_steps=args.elite_rerank_steps,
        elite_rerank_seeds=args.elite_rerank_seeds,
        elite_val_sequences=args.elite_val_seqs,
        amp_dtype=args.amp_dtype,
        compile_model=compile_elite_rerank,
        compile_search_candidates=compile_search_candidates,
        mutation_settings=mutation_settings,
        exploration_start=args.exploration_start,
        exploration_end=args.exploration_end,
        novelty_end_weight=args.novelty_end_weight,
        novelty_transform=args.novelty_transform,
        fitness_efficiency_weight=args.fitness_efficiency_weight,
        fitness_simplicity_weight=args.fitness_simplicity_weight,
        fitness_quant_penalty=args.fitness_quant_penalty,
        random_immigrants=args.random_immigrants,
        archive_samples=args.archive_samples,
        target_score=args.target_val_bpb,
        target_reward_weight=args.target_reward_weight,
        target_penalty_weight=args.target_penalty_weight,
    )
    total_time = time.time() - t0

    # Print final results
    print(f"\n{'='*60}")
    print(f"Search Complete — {total_time:.0f}s total")
    print(f"{'='*60}")
    stats = archive.stats()
    print(f"Archive coverage: {stats['coverage']:.1%} ({stats['num_filled']} cells)")
    print(f"Best fitness: {stats['best_fitness']:.4f}")

    print("\nTop 5 Configs:")
    for i, (cfg, fitness, profile) in enumerate(archive.best_configs(5)):
        active = cfg.count_active_mechanisms()
        print(f"  #{i+1}: fitness={fitness:.4f} | "
              f"iters={profile.mean_iterations:.1f} | "
              f"mechanisms={active} | "
              f"blocks={cfg.num_shared_blocks} | "
              f"max_iter={cfg.max_iterations} | "
              f"min_halt={cfg.min_iterations_before_halt}")


if __name__ == "__main__":
    main()
