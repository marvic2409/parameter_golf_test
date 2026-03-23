#!/usr/bin/env python3
"""Short end-to-end benchmark for the recursive search pipeline.

This benchmarks one candidate evaluation and extrapolates to a larger search.

Example:
  python -m neuromod_recursive.benchmark_search \
    --use-fineweb \
    --preset fineweb_large \
    --steps 200 \
    --search-steps 2000 \
    --population 30 \
    --generations 20 \
    --num-probes 128 \
    --target-probes 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .compression import dequantize_state_dict_int8, measure_compressed_size, quantize_state_dict_int8
from .config import NeuroModConfig, make_preset_config
from .evaluate import evaluate_model
from .model import NeuroModRecursiveModel, count_parameters
from .novelty.behavioral import compute_behavioral_profile, generate_diagnostic_probes
from .train import train_single_config
from .utils import format_param_count, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one recursive-search candidate and extrapolate")
    parser.add_argument("--preset", choices=["default", "fineweb_medium", "fineweb_large"], default="fineweb_medium")
    parser.add_argument("--use-fineweb", action="store_true", help="Benchmark with real FineWeb data")
    parser.add_argument("--data-path", type=str, default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200, help="Short benchmark training length")
    parser.add_argument("--search-steps", type=int, default=2000, help="Target steps per candidate in the real search")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--num-probes", type=int, default=128, help="Probe count used in the short benchmark")
    parser.add_argument("--target-probes", type=int, default=500, help="Probe count used in the real search")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the JSON summary")
    return parser.parse_args()


def build_config(args) -> NeuroModConfig:
    return make_preset_config(args.preset)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_rate(count: float, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return count / seconds


def eval_workload_size(
    config: NeuroModConfig,
    fineweb_setup,
    synthetic_eval_batches: int,
) -> tuple[int, int]:
    if fineweb_setup is not None:
        token_count = int(fineweb_setup["val_tokens"].numel() - 1)
        seq_count = token_count // config.seq_len
        return token_count, seq_count
    seq_count = synthetic_eval_batches * config.batch_size
    token_count = seq_count * config.seq_len
    return token_count, seq_count


def timed_eval(
    model,
    config: NeuroModConfig,
    fineweb_setup,
    device: torch.device,
    synthetic_eval_batches: int = 5,
) -> tuple[float, float, int, int]:
    sync_device(device)
    t0 = time.perf_counter()
    if fineweb_setup is not None:
        from .fineweb_eval import eval_fineweb_bpb

        val_loss, val_bpb = eval_fineweb_bpb(
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
        score = val_bpb
    else:
        eval_result = evaluate_model(model, config, num_batches=synthetic_eval_batches, device=device)
        val_loss = eval_result["val_loss"]
        score = val_loss
    sync_device(device)
    token_count, seq_count = eval_workload_size(config, fineweb_setup, synthetic_eval_batches)
    return score, time.perf_counter() - t0, token_count, seq_count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else get_device()
    config = build_config(args)
    fineweb_setup = None

    if args.use_fineweb:
        from .fineweb_eval import setup_fineweb_eval

        fineweb_setup = setup_fineweb_eval(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            device=device,
        )
        config.vocab_size = fineweb_setup["vocab_size"]
        config.seq_len = fineweb_setup["seq_len"]

    preview_model = NeuroModRecursiveModel(config)
    param_count = count_parameters(preview_model)
    del preview_model

    print(
        f"Benchmarking preset={args.preset} device={device} "
        f"params={format_param_count(param_count)} steps={args.steps} search_steps={args.search_steps}"
    )

    sync_device(device)
    t_train0 = time.perf_counter()
    train_result = train_single_config(
        config=config,
        num_steps=args.steps,
        device=device,
        quiet=True,
        fineweb_setup=fineweb_setup,
    )
    sync_device(device)
    train_plus_eval_s = time.perf_counter() - t_train0

    model = train_result["model"]
    prequant_score = train_result["val_bpb"] if train_result.get("val_bpb") is not None else train_result["val_loss"]

    # Re-time the same eval once so we can estimate and subtract the fixed eval cost
    # already included inside train_single_config().
    synthetic_eval_batches = 5
    prequant_score_recheck, prequant_eval_s, eval_tokens, eval_sequences = timed_eval(
        model,
        config,
        fineweb_setup,
        device,
        synthetic_eval_batches=synthetic_eval_batches,
    )

    sync_device(device)
    t_comp0 = time.perf_counter()
    size_stats = measure_compressed_size(model)
    sync_device(device)
    compression_s = time.perf_counter() - t_comp0

    sync_device(device)
    t_quant0 = time.perf_counter()
    quant_obj, _ = quantize_state_dict_int8(model.state_dict())
    dequant_state = dequantize_state_dict_int8(quant_obj)
    model.load_state_dict(dequant_state)
    sync_device(device)
    quant_reload_s = time.perf_counter() - t_quant0

    post_quant_score, post_quant_eval_s, _, _ = timed_eval(
        model,
        config,
        fineweb_setup,
        device,
        synthetic_eval_batches=synthetic_eval_batches,
    )

    probe_vocab = config.vocab_size
    probe_seq = min(config.seq_len, 64)
    probes, probe_categories = generate_diagnostic_probes(
        vocab_size=probe_vocab,
        seq_len=probe_seq,
        num_probes=args.num_probes,
        device=device,
    )
    sync_device(device)
    t_prof0 = time.perf_counter()
    profile = compute_behavioral_profile(model, probes, probe_categories, config)
    sync_device(device)
    profile_s = time.perf_counter() - t_prof0

    train_only_s = max(train_plus_eval_s - prequant_eval_s, 0.0)
    train_s_per_step = train_only_s / max(args.steps, 1)
    scaled_profile_s = profile_s * (args.target_probes / max(args.num_probes, 1))
    train_sequences = args.steps * config.batch_size
    train_tokens = train_sequences * config.seq_len
    train_token_passes = train_tokens * train_result["avg_iterations"]
    profile_tokens = args.num_probes * probe_seq
    scaled_profile_tokens = args.target_probes * probe_seq
    estimated_train_s = train_s_per_step * args.search_steps
    estimated_train_tokens = args.search_steps * config.batch_size * config.seq_len
    estimated_train_token_passes = estimated_train_tokens * train_result["avg_iterations"]
    estimated_candidate_tokens = estimated_train_tokens + 2 * eval_tokens + scaled_profile_tokens
    estimated_eval_seconds = prequant_eval_s + post_quant_eval_s

    estimated_candidate_s = (
        estimated_train_s
        + prequant_eval_s
        + compression_s
        + quant_reload_s
        + post_quant_eval_s
        + scaled_profile_s
    )
    estimated_search_hours = estimated_candidate_s * args.population * args.generations / 3600.0

    train_tokens_per_second = safe_rate(train_tokens, train_only_s)
    train_sequences_per_second = safe_rate(train_sequences, train_only_s)
    train_token_passes_per_second = safe_rate(train_token_passes, train_only_s)
    prequant_eval_tokens_per_second = safe_rate(eval_tokens, prequant_eval_s)
    prequant_eval_sequences_per_second = safe_rate(eval_sequences, prequant_eval_s)
    post_quant_eval_tokens_per_second = safe_rate(eval_tokens, post_quant_eval_s)
    post_quant_eval_sequences_per_second = safe_rate(eval_sequences, post_quant_eval_s)
    profile_tokens_per_second = safe_rate(profile_tokens, profile_s)
    profile_sequences_per_second = safe_rate(args.num_probes, profile_s)
    estimated_candidate_tokens_per_second = safe_rate(estimated_candidate_tokens, estimated_candidate_s)

    summary = {
        "preset": args.preset,
        "device": str(device),
        "use_fineweb": args.use_fineweb,
        "param_count": param_count,
        "benchmark_steps": args.steps,
        "search_steps": args.search_steps,
        "population": args.population,
        "generations": args.generations,
        "num_probes": args.num_probes,
        "target_probes": args.target_probes,
        "train_plus_eval_seconds": train_plus_eval_s,
        "prequant_eval_seconds": prequant_eval_s,
        "estimated_train_only_seconds": train_only_s,
        "train_seconds_per_step": train_s_per_step,
        "train_sequences": train_sequences,
        "train_tokens": train_tokens,
        "train_token_passes": train_token_passes,
        "train_sequences_per_second": train_sequences_per_second,
        "train_tokens_per_second": train_tokens_per_second,
        "train_token_passes_per_second": train_token_passes_per_second,
        "compression_seconds": compression_s,
        "quant_reload_seconds": quant_reload_s,
        "eval_sequences": eval_sequences,
        "eval_tokens": eval_tokens,
        "prequant_eval_sequences_per_second": prequant_eval_sequences_per_second,
        "prequant_eval_tokens_per_second": prequant_eval_tokens_per_second,
        "post_quant_eval_seconds": post_quant_eval_s,
        "post_quant_eval_sequences_per_second": post_quant_eval_sequences_per_second,
        "post_quant_eval_tokens_per_second": post_quant_eval_tokens_per_second,
        "profile_seconds": profile_s,
        "profile_probe_seq_len": probe_seq,
        "profile_probe_tokens": profile_tokens,
        "profile_sequences_per_second": profile_sequences_per_second,
        "profile_tokens_per_second": profile_tokens_per_second,
        "scaled_profile_seconds": scaled_profile_s,
        "scaled_profile_tokens": scaled_profile_tokens,
        "prequant_score": prequant_score,
        "prequant_score_recheck": prequant_score_recheck,
        "post_quant_score": post_quant_score,
        "avg_iterations": train_result["avg_iterations"],
        "compressed_model_bytes": size_stats["zlib_compressed_bytes"],
        "estimated_train_seconds_at_search_steps": estimated_train_s,
        "estimated_train_tokens_at_search_steps": estimated_train_tokens,
        "estimated_train_token_passes_at_search_steps": estimated_train_token_passes,
        "estimated_eval_seconds_per_candidate": estimated_eval_seconds,
        "estimated_candidate_tokens": estimated_candidate_tokens,
        "estimated_candidate_tokens_per_second": estimated_candidate_tokens_per_second,
        "estimated_candidate_seconds_at_search_steps": estimated_candidate_s,
        "estimated_total_search_hours": estimated_search_hours,
        "profile_mean_iterations": profile.mean_iterations,
        "profile_iteration_variance": profile.iteration_variance,
    }

    print(
        f"Train throughput: {train_tokens_per_second:,.0f} tok/s "
        f"({train_sequences_per_second:,.1f} seq/s, "
        f"{train_token_passes_per_second:,.0f} token-passes/s)"
    )
    print(
        f"Eval throughput: pre={prequant_eval_tokens_per_second:,.0f} tok/s "
        f"post={post_quant_eval_tokens_per_second:,.0f} tok/s"
    )
    print(
        f"Profile throughput: {profile_tokens_per_second:,.0f} tok/s "
        f"({profile_sequences_per_second:,.1f} probe seq/s)"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"saved_json:{output_path}")


if __name__ == "__main__":
    main()
