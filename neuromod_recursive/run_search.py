#!/usr/bin/env python3
"""Entry point: launch evolutionary architecture search.

Usage:
  # Smoke test on CPU with synthetic data:
  python -m neuromod_recursive.run_search --smoke-test --device cpu

  # Single config on FineWeb (actual BPB scoring):
  python -m neuromod_recursive.run_search --single --use-fineweb --steps 5000

  # Full evolutionary search on FineWeb with GPU:
  python -m neuromod_recursive.run_search --use-fineweb --population 30 --generations 20 --steps 2000

  # Multi-GPU:
  torchrun --standalone --nproc_per_node=4 -m neuromod_recursive.run_search --distributed --single --use-fineweb
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from .config import NeuroModConfig
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

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, cuda:0, etc.")

    return parser.parse_args()


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

    # --- Single config training ---
    if args.single:
        print("\n--- Training single default config ---")
        config = NeuroModConfig()
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
