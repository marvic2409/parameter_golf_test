"""Single-config training loop with multi-GPU support via DDP.

Supports two modes:
  - Synthetic data (default, for fast architecture search)
  - FineWeb real data (--use-fineweb, for actual BPB scoring)
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import NeuroModConfig
from .data import generate_mixed_batch
from .evaluate import evaluate_model
from .model import NeuroModRecursiveModel, compute_loss, count_parameters
from .utils import set_seed, get_device, format_param_count


def setup_distributed():
    """Initialize DDP if running in distributed mode."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank
    return False, 0, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_single_config(
    config: NeuroModConfig,
    num_steps: int = 2000,
    seed: int = 42,
    device: Optional[torch.device] = None,
    quiet: bool = False,
    log_every: int = 100,
    eval_batches: int = 5,
    fineweb_setup: Optional[dict] = None,
) -> dict:
    """Train a single config and return results.

    Args:
        fineweb_setup: If provided, train on real FineWeb data and evaluate BPB.
            Should be the dict returned by fineweb_eval.setup_fineweb_eval().
            If None, uses synthetic data.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    # When using FineWeb, override config to match challenge tokenizer
    if fineweb_setup is not None:
        config.vocab_size = fineweb_setup["vocab_size"]
        config.seq_len = fineweb_setup["seq_len"]

    model = NeuroModRecursiveModel(config).to(device)
    n_params = count_parameters(model)

    if not quiet:
        print(f"Model parameters: {format_param_count(n_params)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Set up data source
    token_stream = None
    if fineweb_setup is not None:
        from .fineweb_eval import TokenStream
        token_stream = TokenStream(fineweb_setup["train_pattern"])

    model.train()
    train_losses = []
    start_time = time.time()

    for step in range(num_steps):
        if token_stream is not None:
            # Real FineWeb data
            total_tokens = config.batch_size * config.seq_len + 1
            raw = token_stream.take(total_tokens).to(device=device, dtype=torch.long)
            inputs = raw[:-1].reshape(config.batch_size, config.seq_len)
            targets = raw[1:].reshape(config.batch_size, config.seq_len)
        else:
            # Synthetic data
            inputs, targets = generate_mixed_batch(
                config.batch_size, config.seq_len, config.vocab_size, device=device
            )

        logits, details = model(inputs, return_details=False)
        loss, loss_dict = compute_loss(logits, targets, details, config)

        if math.isnan(loss.item()):
            if not quiet:
                print(f"  NaN loss at step {step}, stopping early")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss_dict["task_loss"])

        if not quiet and (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(train_losses[-log_every:]) / min(log_every, len(train_losses))
            print(
                f"  step {step + 1}/{num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"iters={loss_dict['avg_iterations']:.1f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | "
                f"time={elapsed:.1f}s"
            )

    # --- Final evaluation ---
    elapsed = time.time() - start_time

    if fineweb_setup is not None:
        # Evaluate on actual FineWeb BPB
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
        avg_iters = details.get("num_iterations", config.max_iterations) if details else config.max_iterations

        result = {
            "val_loss": val_loss,
            "val_bpb": val_bpb,
            "task_loss": val_loss,
            "avg_iterations": avg_iters,
            "train_loss_final": train_losses[-1] if train_losses else float("inf"),
            "n_params": n_params,
            "elapsed_seconds": elapsed,
            "model": model,
        }
        if not quiet:
            print(f"\n  Final: val_loss={val_loss:.4f} | val_bpb={val_bpb:.4f} | "
                  f"avg_iters={avg_iters:.1f} | time={elapsed:.1f}s")
    else:
        eval_result = evaluate_model(model, config, num_batches=eval_batches, device=device)
        result = {
            "val_loss": eval_result["val_loss"],
            "val_bpb": None,
            "task_loss": eval_result["task_loss"],
            "avg_iterations": eval_result["avg_iterations"],
            "train_loss_final": train_losses[-1] if train_losses else float("inf"),
            "n_params": n_params,
            "elapsed_seconds": elapsed,
            "model": model,
        }
        if not quiet:
            print(f"\n  Final: val_loss={result['val_loss']:.4f} | "
                  f"avg_iters={result['avg_iterations']:.1f} | "
                  f"time={elapsed:.1f}s")

    return result


def train_distributed(
    config: NeuroModConfig,
    num_steps: int = 2000,
    seed: int = 42,
    log_every: int = 100,
    eval_batches: int = 5,
    fineweb_setup: Optional[dict] = None,
) -> dict:
    """Train with DDP on multiple GPUs.

    Launch via: torchrun --standalone --nproc_per_node=N -m neuromod_recursive.run_search --distributed --single
    """
    is_dist, rank, local_rank = setup_distributed()

    if is_dist:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device()

    set_seed(seed + rank)

    if fineweb_setup is not None:
        config.vocab_size = fineweb_setup["vocab_size"]
        config.seq_len = fineweb_setup["seq_len"]

    model = NeuroModRecursiveModel(config).to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank])
        base_model = model.module
    else:
        base_model = model

    n_params = count_parameters(base_model)
    is_master = rank == 0

    if is_master:
        print(f"Model parameters: {format_param_count(n_params)}")
        if is_dist:
            print(f"World size: {dist.get_world_size()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    token_stream = None
    if fineweb_setup is not None:
        from .fineweb_eval import TokenStream
        token_stream = TokenStream(fineweb_setup["train_pattern"])

    model.train()
    train_losses = []
    start_time = time.time()

    for step in range(num_steps):
        if token_stream is not None:
            total_tokens = config.batch_size * config.seq_len + 1
            raw = token_stream.take(total_tokens).to(device=device, dtype=torch.long)
            inputs = raw[:-1].reshape(config.batch_size, config.seq_len)
            targets = raw[1:].reshape(config.batch_size, config.seq_len)
        else:
            inputs, targets = generate_mixed_batch(
                config.batch_size, config.seq_len, config.vocab_size, device=device
            )

        logits, details = model(inputs, return_details=False)
        loss, loss_dict = compute_loss(logits, targets, details, config)

        if math.isnan(loss.item()):
            if is_master:
                print(f"  NaN loss at step {step}, stopping early")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss_dict["task_loss"])

        if is_master and (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(train_losses[-log_every:]) / min(log_every, len(train_losses))
            print(
                f"  step {step + 1}/{num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"iters={loss_dict['avg_iterations']:.1f} | "
                f"time={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time

    if is_master:
        if fineweb_setup is not None:
            from .fineweb_eval import eval_fineweb_bpb
            val_loss, val_bpb = eval_fineweb_bpb(
                model=base_model,
                val_tokens=fineweb_setup["val_tokens"],
                seq_len=config.seq_len,
                vocab_size=config.vocab_size,
                base_bytes_lut=fineweb_setup["base_bytes_lut"].to(device),
                has_leading_space_lut=fineweb_setup["has_leading_space_lut"].to(device),
                is_boundary_token_lut=fineweb_setup["is_boundary_token_lut"].to(device),
                batch_size=max(1, 65536 // config.seq_len),
                device=device,
            )
            eval_result = {"val_loss": val_loss, "val_bpb": val_bpb, "task_loss": val_loss, "avg_iterations": 0}
        else:
            eval_result = evaluate_model(base_model, config, num_batches=eval_batches, device=device)
            eval_result["val_bpb"] = None
    else:
        eval_result = {"val_loss": 0, "val_bpb": None, "task_loss": 0, "avg_iterations": 0}

    if is_dist:
        cleanup_distributed()

    result = {
        "val_loss": eval_result["val_loss"],
        "val_bpb": eval_result.get("val_bpb"),
        "task_loss": eval_result["task_loss"],
        "avg_iterations": eval_result["avg_iterations"],
        "train_loss_final": train_losses[-1] if train_losses else float("inf"),
        "n_params": n_params,
        "elapsed_seconds": elapsed,
        "model": base_model,
    }

    if is_master:
        bpb_str = f" | val_bpb={result['val_bpb']:.4f}" if result["val_bpb"] is not None else ""
        print(f"\n  Final: val_loss={result['val_loss']:.4f}{bpb_str} | time={elapsed:.1f}s")

    return result
