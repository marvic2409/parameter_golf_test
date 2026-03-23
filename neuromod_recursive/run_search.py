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

import torch

from .config import NeuroModConfig, make_preset_config
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
    parser.add_argument("--novelty-weight", type=float, default=0.5, help="Initial novelty weight")
    parser.add_argument("--novelty-k", type=int, default=15, help="k for k-nearest novelty")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output-dir", type=str, default="search_results", help="Output directory")
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "fineweb_medium", "fineweb_large"],
        help="Base config preset before applying any explicit overrides.",
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
    parser.add_argument("--ff-mult", type=float, default=None, help="Override FFN expansion multiplier")
    parser.add_argument("--mod-dim", type=int, default=None, help="Override modulation code dimension")
    parser.add_argument("--num-shared-blocks", type=int, default=None, help="Override number of shared blocks")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max recursive iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-process batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override LR warmup steps")
    parser.add_argument("--num-cycles", type=int, default=None, help="Override LR restart cycle count")
    parser.add_argument("--min-lr-ratio", type=float, default=None, help="Override LR floor ratio")
    parser.add_argument("--iteration-cost", type=float, default=None, help="Override recursive ponder cost")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, cuda:0, etc.")

    return parser.parse_args()


def build_base_config(args) -> NeuroModConfig:
    config = make_preset_config(args.preset)
    overrides = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "ff_mult": args.ff_mult,
        "mod_dim": args.mod_dim,
        "num_shared_blocks": args.num_shared_blocks,
        "max_iterations": args.max_iterations,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "num_cycles": args.num_cycles,
        "min_lr_ratio": args.min_lr_ratio,
        "iteration_cost": args.iteration_cost,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    return config


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

    preview_config = NeuroModConfig(**vars(base_config))
    if fineweb_setup:
        preview_config.vocab_size = fineweb_setup["vocab_size"]
        preview_config.seq_len = fineweb_setup["seq_len"]
    preview_model = NeuroModRecursiveModel(preview_config)
    preview_params = count_parameters(preview_model)
    del preview_model

    print(
        f"Base config: preset={args.preset} hidden_dim={base_config.hidden_dim} "
        f"heads={base_config.num_heads} ff_mult={base_config.ff_mult} "
        f"shared_blocks={base_config.num_shared_blocks} max_iterations={base_config.max_iterations} "
        f"batch_size={base_config.batch_size} lr={base_config.lr:g} "
        f"params={format_param_count(preview_params)}"
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
        print(f"Parameters: {format_param_count(params)}")
        del model

        if args.distributed:
            result = train_distributed(
                config, num_steps=args.steps, seed=args.seed,
                fineweb_setup=fineweb_setup,
            )
        else:
            result = train_single_config(
                config, num_steps=args.steps, seed=args.seed, device=device,
                fineweb_setup=fineweb_setup,
            )

        print(f"\nFinal val_loss: {result['val_loss']:.4f}")
        if result.get("val_bpb") is not None:
            print(f"Final val_bpb:  {result['val_bpb']:.4f}  (this is the challenge score)")
        return

    # --- Evolutionary search ---
    print(f"\n--- Evolutionary Search ---")
    print(f"Population: {args.population} | Generations: {args.generations}")
    print(f"Steps/eval: {args.steps} | Novelty weight: {args.novelty_weight}")
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
              f"max_iter={cfg.max_iterations}")


if __name__ == "__main__":
    main()
