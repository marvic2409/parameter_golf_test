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
from .model import NeuroModRecursiveModel, count_parameters
from .novelty.behavioral import compute_behavioral_profile, generate_diagnostic_probes
from .train import evaluate_trained_model, train_single_config
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
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--compile-search-candidates",
        action="store_true",
        help="Also compile the screened search candidate. Usually not worthwhile for evolutionary search.",
    )
    parser.add_argument("--screen-val-seqs", type=int, default=None)
    parser.add_argument("--promote-top-k", type=int, default=None)
    parser.add_argument("--promote-val-seqs", type=int, default=None)
    parser.add_argument("--elite-rerank-top-k", type=int, default=None)
    parser.add_argument("--elite-rerank-steps", type=int, default=None)
    parser.add_argument("--elite-rerank-seeds", type=int, default=2)
    parser.add_argument("--elite-val-seqs", type=int, default=None)
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


def build_eval_stages(
    fineweb_setup,
    screen_val_sequences: int | None,
    promote_val_sequences: int | None,
    elite_val_sequences: int | None,
):
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


def timed_eval(
    model,
    config: NeuroModConfig,
    fineweb_setup,
    device: torch.device,
    synthetic_eval_batches: int = 5,
    amp_dtype: str | None = "none",
) -> tuple[float, float, int, int]:
    sync_device(device)
    t0 = time.perf_counter()
    eval_result = evaluate_trained_model(
        model,
        config,
        device=device,
        fineweb_setup=fineweb_setup,
        eval_batches=synthetic_eval_batches,
        amp_dtype=amp_dtype,
    )
    sync_device(device)
    token_count, seq_count = eval_workload_size(config, fineweb_setup, synthetic_eval_batches)
    score = eval_result["val_bpb"] if eval_result.get("val_bpb") is not None else eval_result["val_loss"]
    return score, time.perf_counter() - t0, token_count, seq_count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else get_device()
    config = build_config(args)
    compile_search_candidates = args.compile_model and args.compile_search_candidates
    compile_elite_rerank = args.compile_model
    fineweb_setup = None
    eval_stages = {"screen": None, "promote": None, "elite": None}

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
        total_val_sequences = (fineweb_setup["val_tokens"].numel() - 1) // fineweb_setup["seq_len"]
        if args.screen_val_seqs is None:
            args.screen_val_seqs = min(1024, total_val_sequences)
        if args.promote_val_seqs is None:
            args.promote_val_seqs = min(4096, total_val_sequences)
        if args.promote_top_k is None:
            args.promote_top_k = max(1, args.population // 5)
        if args.elite_rerank_top_k is None:
            args.elite_rerank_top_k = min(3, args.population)
        if args.elite_rerank_steps is None:
            args.elite_rerank_steps = max(args.search_steps * 2, args.search_steps)
        eval_stages = build_eval_stages(
            fineweb_setup,
            screen_val_sequences=args.screen_val_seqs,
            promote_val_sequences=args.promote_val_seqs,
            elite_val_sequences=args.elite_val_seqs,
        )
    else:
        if args.promote_top_k is None:
            args.promote_top_k = 0
        if args.elite_rerank_top_k is None:
            args.elite_rerank_top_k = 0

    preview_model = NeuroModRecursiveModel(config)
    param_count = count_parameters(preview_model)
    del preview_model

    print(
        f"Benchmarking preset={args.preset} device={device} "
        f"params={format_param_count(param_count)} steps={args.steps} search_steps={args.search_steps} "
        f"compile={'all-candidates' if compile_search_candidates else ('rerank-only' if compile_elite_rerank else 'off')}"
    )

    sync_device(device)
    t_train0 = time.perf_counter()
    train_result = train_single_config(
        config=config,
        num_steps=args.steps,
        device=device,
        quiet=True,
        fineweb_setup=fineweb_setup,
        eval_setup=eval_stages["screen"],
        amp_dtype=args.amp_dtype,
        compile_model=compile_search_candidates,
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
        eval_stages["screen"],
        device,
        synthetic_eval_batches=synthetic_eval_batches,
        amp_dtype=args.amp_dtype,
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
        eval_stages["screen"],
        device,
        synthetic_eval_batches=synthetic_eval_batches,
        amp_dtype=args.amp_dtype,
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
    estimated_screen_candidate_tokens = estimated_train_tokens + 2 * eval_tokens + scaled_profile_tokens
    estimated_screen_candidate_s = (
        estimated_train_s
        + prequant_eval_s
        + compression_s
        + quant_reload_s
        + post_quant_eval_s
        + scaled_profile_s
    )
    estimated_eval_seconds = prequant_eval_s + post_quant_eval_s

    promote_eval_s = 0.0
    promote_eval_tokens = 0
    promote_eval_sequences = 0
    if args.promote_top_k > 0:
        _, promote_eval_s, promote_eval_tokens, promote_eval_sequences = timed_eval(
            model,
            config,
            eval_stages["promote"],
            device,
            synthetic_eval_batches=max(8, synthetic_eval_batches),
            amp_dtype=args.amp_dtype,
        )

    rerank_candidate_s = 0.0
    rerank_eval_s = 0.0
    rerank_tokens = 0
    if args.elite_rerank_top_k > 0 and args.elite_rerank_seeds > 0:
        rerank_config = NeuroModConfig(**vars(config))
        rerank_result = train_single_config(
            config=rerank_config,
            num_steps=args.elite_rerank_steps,
            device=device,
            quiet=True,
            fineweb_setup=fineweb_setup,
            eval_setup=eval_stages["elite"],
            amp_dtype=args.amp_dtype,
            compile_model=compile_elite_rerank,
        )
        rerank_model = rerank_result["model"]
        _, rerank_pre_eval_s, rerank_tokens, _ = timed_eval(
            rerank_model,
            rerank_config,
            eval_stages["elite"],
            device,
            synthetic_eval_batches=synthetic_eval_batches,
            amp_dtype=args.amp_dtype,
        )
        sync_device(device)
        t_rerank_comp0 = time.perf_counter()
        rerank_size_stats = measure_compressed_size(rerank_model)
        sync_device(device)
        rerank_comp_s = time.perf_counter() - t_rerank_comp0

        sync_device(device)
        t_rerank_quant0 = time.perf_counter()
        rerank_quant_obj, _ = quantize_state_dict_int8(rerank_model.state_dict())
        rerank_dequant_state = dequantize_state_dict_int8(rerank_quant_obj)
        rerank_model.load_state_dict(rerank_dequant_state)
        sync_device(device)
        rerank_quant_s = time.perf_counter() - t_rerank_quant0

        _, rerank_post_eval_s, _, _ = timed_eval(
            rerank_model,
            rerank_config,
            eval_stages["elite"],
            device,
            synthetic_eval_batches=synthetic_eval_batches,
            amp_dtype=args.amp_dtype,
        )
        rerank_train_only_s = max(rerank_result["elapsed_seconds"] - rerank_pre_eval_s, 0.0)
        rerank_eval_s = rerank_pre_eval_s + rerank_post_eval_s
        rerank_candidate_s = rerank_train_only_s + rerank_comp_s + rerank_quant_s + rerank_eval_s
        del rerank_model
        _ = rerank_size_stats

    estimated_generation_s = args.population * estimated_screen_candidate_s + args.promote_top_k * promote_eval_s
    estimated_search_s = (
        args.generations * estimated_generation_s
        + args.elite_rerank_top_k * args.elite_rerank_seeds * rerank_candidate_s
    )
    estimated_search_hours = estimated_search_s / 3600.0

    train_tokens_per_second = safe_rate(train_tokens, train_only_s)
    train_sequences_per_second = safe_rate(train_sequences, train_only_s)
    train_token_passes_per_second = safe_rate(train_token_passes, train_only_s)
    prequant_eval_tokens_per_second = safe_rate(eval_tokens, prequant_eval_s)
    prequant_eval_sequences_per_second = safe_rate(eval_sequences, prequant_eval_s)
    post_quant_eval_tokens_per_second = safe_rate(eval_tokens, post_quant_eval_s)
    post_quant_eval_sequences_per_second = safe_rate(eval_sequences, post_quant_eval_s)
    promote_eval_tokens_per_second = safe_rate(promote_eval_tokens, promote_eval_s)
    profile_tokens_per_second = safe_rate(profile_tokens, profile_s)
    profile_sequences_per_second = safe_rate(args.num_probes, profile_s)
    estimated_screen_candidate_tokens_per_second = safe_rate(estimated_screen_candidate_tokens, estimated_screen_candidate_s)

    summary = {
        "preset": args.preset,
        "device": str(device),
        "use_fineweb": args.use_fineweb,
        "amp_dtype": args.amp_dtype,
        "compile_model": args.compile_model,
        "compile_search_candidates": compile_search_candidates,
        "compile_elite_rerank": compile_elite_rerank,
        "param_count": param_count,
        "benchmark_steps": args.steps,
        "search_steps": args.search_steps,
        "population": args.population,
        "generations": args.generations,
        "num_probes": args.num_probes,
        "target_probes": args.target_probes,
        "screen_val_sequences": args.screen_val_seqs,
        "promote_val_sequences": args.promote_val_seqs,
        "elite_val_sequences": args.elite_val_seqs,
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
        "promote_top_k": args.promote_top_k,
        "promote_eval_seconds": promote_eval_s,
        "promote_eval_sequences": promote_eval_sequences,
        "promote_eval_tokens": promote_eval_tokens,
        "promote_eval_tokens_per_second": promote_eval_tokens_per_second,
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
        "estimated_candidate_tokens": estimated_screen_candidate_tokens,
        "estimated_candidate_tokens_per_second": estimated_screen_candidate_tokens_per_second,
        "estimated_candidate_seconds_at_search_steps": estimated_screen_candidate_s,
        "estimated_screen_candidate_seconds_at_search_steps": estimated_screen_candidate_s,
        "estimated_promoted_candidate_extra_seconds": promote_eval_s,
        "estimated_generation_seconds": estimated_generation_s,
        "elite_rerank_top_k": args.elite_rerank_top_k,
        "elite_rerank_steps": args.elite_rerank_steps,
        "elite_rerank_seeds": args.elite_rerank_seeds,
        "estimated_elite_rerank_candidate_seconds": rerank_candidate_s,
        "estimated_elite_rerank_total_seconds": args.elite_rerank_top_k * args.elite_rerank_seeds * rerank_candidate_s,
        "elite_rerank_eval_seconds": rerank_eval_s,
        "elite_rerank_eval_tokens": rerank_tokens,
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
        f"Screen eval throughput: pre={prequant_eval_tokens_per_second:,.0f} tok/s "
        f"post={post_quant_eval_tokens_per_second:,.0f} tok/s"
    )
    if args.promote_top_k > 0:
        print(f"Promote eval throughput: {promote_eval_tokens_per_second:,.0f} tok/s")
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
