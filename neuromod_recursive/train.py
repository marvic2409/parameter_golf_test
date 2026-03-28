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
from .utils import (
    autocast_context,
    enable_fast_cuda_math,
    export_state_dict,
    format_param_count,
    get_device,
    maybe_compile_model,
    normalize_amp_dtype,
    set_seed,
    unwrap_model,
)


def _make_cyclical_lr(
    total_steps: int,
    warmup_steps: int = 200,
    num_cycles: int = 4,
    min_lr_ratio: float = 0.05,
):
    r"""Cyclical cosine LR with warm restarts -- learning/consolidation oscillations.

    Each cycle:
      1. Spike LR back to peak (warm restart)
      2. Cosine decay down to min_lr_ratio * peak

    Cycles get progressively longer (T_mult=1.5), so early cycles are fast
    exploration and later cycles are longer consolidation.

    Looks like:
      LR  ^
          |  /\      /\          /\
          | /  \    /  \        /  \
          |/    \  /    \      /    \____
          |      \/      \    /
          |               \  /
          +----------------------------> steps
          warmup  cycle1  cycle2  cycle3
    """
    t_mult = 1.5  # each cycle 1.5x longer than the last

    # Compute cycle boundaries
    remaining = total_steps - warmup_steps
    if remaining <= 0 or num_cycles <= 0:
        # Fall back to simple warmup + cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_lambda

    # Geometric series: first_len * (1 + t_mult + t_mult^2 + ... + t_mult^(n-1)) = remaining
    geo_sum = sum(t_mult ** i for i in range(num_cycles))
    first_cycle_len = remaining / geo_sum
    cycle_lengths = [int(first_cycle_len * t_mult ** i) for i in range(num_cycles)]
    # Fix rounding so they sum to remaining
    cycle_lengths[-1] = remaining - sum(cycle_lengths[:-1])

    # Build boundaries
    cycle_starts = [warmup_steps]
    for length in cycle_lengths[:-1]:
        cycle_starts.append(cycle_starts[-1] + length)

    def lr_lambda(step):
        # Warmup phase
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        # Find which cycle we're in
        cycle_idx = 0
        for i in range(len(cycle_starts) - 1, -1, -1):
            if step >= cycle_starts[i]:
                cycle_idx = i
                break

        cycle_start = cycle_starts[cycle_idx]
        cycle_len = cycle_lengths[cycle_idx]
        progress = (step - cycle_start) / max(1, cycle_len)
        progress = min(progress, 1.0)

        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


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


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


def _make_adam_optimizer(params, lr: float, config: NeuroModConfig, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {
        "betas": (config.beta1, config.beta2),
        "eps": config.adam_eps,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.Adam(
        [{"params": params, "lr": lr, "base_lr": lr}],
        **kwargs,
    )


def _build_optimizers(
    model: nn.Module,
    config: NeuroModConfig,
    device: torch.device,
) -> list[torch.optim.Optimizer]:
    embed_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    matrix_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in {"embedding.token_emb.weight", "embedding.pos_emb.weight", "bigram.embed.weight"}:
            embed_params.append(param)
        elif name == "output_head.proj.weight":
            head_params.append(param)
        elif name in {"skip_weights", "bigram.scale"} or name.endswith("resid_mix"):
            scalar_params.append(param)
        elif name == "bigram.proj.weight":
            matrix_params.append(param)
        elif param.ndim == 2:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    optimizers: list[torch.optim.Optimizer] = []
    if embed_params:
        embed_lr = config.tied_embed_lr if config.tie_embeddings else config.embed_lr
        optimizers.append(_make_adam_optimizer(embed_params, embed_lr, config, device))
    if head_params:
        optimizers.append(_make_adam_optimizer(head_params, config.head_lr, config, device))
    if matrix_params:
        muon = Muon(
            matrix_params,
            lr=config.matrix_lr,
            momentum=config.muon_momentum_warmup_start,
            backend_steps=config.muon_backend_steps,
        )
        for group in muon.param_groups:
            group["base_lr"] = config.matrix_lr
        optimizers.append(muon)
    if scalar_params:
        optimizers.append(_make_adam_optimizer(scalar_params, config.scalar_lr, config, device))
    return optimizers


def _make_schedulers(
    optimizers: list[torch.optim.Optimizer],
    config: NeuroModConfig,
    total_steps: int,
) -> list[torch.optim.lr_scheduler.LambdaLR]:
    lr_lambda = _make_cyclical_lr(total_steps, config.warmup_steps, config.num_cycles, config.min_lr_ratio)
    return [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizers]


def _zero_grad_all(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def _step_all(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()


def _update_muon_momentum(
    optimizers: list[torch.optim.Optimizer],
    config: NeuroModConfig,
    step: int,
) -> None:
    if config.muon_momentum_warmup_steps > 0:
        frac = min(step / config.muon_momentum_warmup_steps, 1.0)
        momentum = (
            (1.0 - frac) * config.muon_momentum_warmup_start
            + frac * config.muon_momentum
        )
    else:
        momentum = config.muon_momentum
    for opt in optimizers:
        if isinstance(opt, Muon):
            for group in opt.param_groups:
                group["momentum"] = momentum


def _maybe_capture_swa(
    swa_state: dict[str, Tensor] | None,
    swa_count: int,
    model: nn.Module,
) -> tuple[dict[str, Tensor] | None, int]:
    state = export_state_dict(model)
    if swa_state is None:
        return (
            {name: tensor.detach().to(device="cpu").clone() for name, tensor in state.items()},
            1,
        )
    for name, tensor in state.items():
        swa_state[name] += tensor.detach().to(device="cpu")
    return swa_state, swa_count + 1


def _maybe_apply_swa(
    model: nn.Module,
    swa_state: dict[str, Tensor] | None,
    swa_count: int,
) -> bool:
    if swa_state is None or swa_count <= 1:
        return False
    current_state = export_state_dict(model)
    avg_state = {
        name: (tensor / swa_count).to(dtype=current_state[name].dtype)
        for name, tensor in swa_state.items()
    }
    model.load_state_dict(avg_state, strict=True)
    return True


def evaluate_trained_model(
    model: torch.nn.Module,
    config: NeuroModConfig,
    device: torch.device,
    fineweb_setup: Optional[dict] = None,
    eval_batches: int = 5,
    amp_dtype: str | None = "none",
) -> dict:
    """Evaluate a trained model on either FineWeb or synthetic validation."""
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
            amp_dtype=amp_dtype,
            stride=config.eval_stride,
        )
        return {
            "val_loss": val_loss,
            "val_bpb": val_bpb,
            "task_loss": val_loss,
            "avg_iterations": None,
        }

    eval_result = evaluate_model(
        model,
        config,
        num_batches=eval_batches,
        device=device,
        amp_dtype=amp_dtype,
    )
    return {
        "val_loss": eval_result["val_loss"],
        "val_bpb": None,
        "task_loss": eval_result["task_loss"],
        "avg_iterations": eval_result["avg_iterations"],
    }


def train_single_config(
    config: NeuroModConfig,
    num_steps: int = 2000,
    seed: int = 42,
    device: Optional[torch.device] = None,
    quiet: bool = False,
    log_every: int = 100,
    eval_batches: int = 5,
    fineweb_setup: Optional[dict] = None,
    eval_setup: Optional[dict] = None,
    amp_dtype: str | None = None,
    compile_model: bool = False,
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
    enable_fast_cuda_math()
    amp_dtype = normalize_amp_dtype(amp_dtype, device)

    # When using FineWeb, override config to match challenge tokenizer
    if fineweb_setup is not None:
        config.vocab_size = fineweb_setup["vocab_size"]
        config.seq_len = fineweb_setup["seq_len"]
    elif eval_setup is not None:
        config.vocab_size = eval_setup["vocab_size"]
        config.seq_len = eval_setup["seq_len"]

    final_eval_setup = eval_setup if eval_setup is not None else fineweb_setup

    base_model = NeuroModRecursiveModel(config).to(device)
    n_params = count_parameters(base_model)
    optimizers = _build_optimizers(base_model, config, device)
    schedulers = _make_schedulers(optimizers, config, num_steps)
    model = maybe_compile_model(base_model, enabled=compile_model)

    if not quiet:
        print(f"Model parameters: {format_param_count(n_params)}")

    # Set up data source
    token_stream = None
    if fineweb_setup is not None:
        from .fineweb_eval import TokenStream
        token_stream = TokenStream(fineweb_setup["train_pattern"])

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == "fp16"))
    train_losses = []
    iteration_means = []
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
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

        with autocast_context(device, amp_dtype):
            logits, details = model(inputs, return_details=False)
            loss, loss_dict = compute_loss(logits, targets, details, config)

        if math.isnan(loss.item()):
            if not quiet:
                print(f"  NaN loss at step {step}, stopping early")
            break

        _zero_grad_all(optimizers)
        _update_muon_momentum(optimizers, config, step)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.unscale_(opt)
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
            _step_all(optimizers)
        for scheduler in schedulers:
            scheduler.step()

        if config.swa_enabled and optimizers and (step + 1) % config.swa_every == 0:
            base_lr = float(optimizers[0].param_groups[0].get("base_lr", 1.0))
            current_lr = float(schedulers[0].get_last_lr()[0])
            lr_scale = current_lr / max(base_lr, 1e-12)
            if lr_scale <= config.swa_start_frac:
                swa_state, swa_count = _maybe_capture_swa(swa_state, swa_count, base_model)

        train_losses.append(loss_dict["task_loss"])
        iteration_means.append(float(loss_dict["avg_iterations"]))

        if not quiet and (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(train_losses[-log_every:]) / min(log_every, len(train_losses))
            print(
                f"  step {step + 1}/{num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"iters={loss_dict['avg_iterations']:.1f} | "
                f"lr={schedulers[0].get_last_lr()[0]:.6f} | "
                f"time={elapsed:.1f}s"
            )

    # --- Final evaluation ---
    elapsed = time.time() - start_time
    avg_iterations_run = (
        sum(iteration_means) / len(iteration_means) if iteration_means else float(config.max_iterations)
    )
    applied_swa = _maybe_apply_swa(base_model, swa_state, swa_count)
    if applied_swa and not quiet:
        print(f"  Applied SWA over {swa_count} checkpoints")

    eval_result = evaluate_trained_model(
        model,
        config,
        device=device,
        fineweb_setup=final_eval_setup,
        eval_batches=eval_batches,
        amp_dtype=amp_dtype,
    )
    exported_model = unwrap_model(model)

    if final_eval_setup is not None:
        result = {
            "val_loss": eval_result["val_loss"],
            "val_bpb": eval_result["val_bpb"],
            "task_loss": eval_result["task_loss"],
            "avg_iterations": avg_iterations_run,
            "train_loss_final": train_losses[-1] if train_losses else float("inf"),
            "n_params": n_params,
            "elapsed_seconds": elapsed,
            "model": exported_model,
        }
        if not quiet:
            print(f"\n  Final: val_loss={result['val_loss']:.4f} | val_bpb={result['val_bpb']:.4f} | "
                  f"avg_iters={avg_iterations_run:.1f} | time={elapsed:.1f}s")
    else:
        result = {
            "val_loss": eval_result["val_loss"],
            "val_bpb": eval_result["val_bpb"],
            "task_loss": eval_result["task_loss"],
            "avg_iterations": eval_result["avg_iterations"] if eval_result["avg_iterations"] is not None else avg_iterations_run,
            "train_loss_final": train_losses[-1] if train_losses else float("inf"),
            "n_params": n_params,
            "elapsed_seconds": elapsed,
            "model": exported_model,
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
    eval_setup: Optional[dict] = None,
    amp_dtype: str | None = None,
    compile_model: bool = False,
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
    enable_fast_cuda_math()
    amp_dtype = normalize_amp_dtype(amp_dtype, device)

    if fineweb_setup is not None:
        config.vocab_size = fineweb_setup["vocab_size"]
        config.seq_len = fineweb_setup["seq_len"]
    elif eval_setup is not None:
        config.vocab_size = eval_setup["vocab_size"]
        config.seq_len = eval_setup["seq_len"]

    final_eval_setup = eval_setup if eval_setup is not None else fineweb_setup

    base_model = NeuroModRecursiveModel(config).to(device)
    n_params = count_parameters(base_model)

    if compile_model and is_dist and rank == 0:
        print("compile_model is ignored in distributed mode")
    if compile_model and not is_dist:
        model = maybe_compile_model(base_model, enabled=True)
    else:
        model = base_model

    if is_dist:
        model = DDP(model, device_ids=[local_rank])
        base_model = model.module
    optimizers = _build_optimizers(base_model, config, device)
    schedulers = _make_schedulers(optimizers, config, num_steps)

    is_master = rank == 0

    if is_master:
        print(f"Model parameters: {format_param_count(n_params)}")
        if is_dist:
            print(f"World size: {dist.get_world_size()}")

    token_stream = None
    if fineweb_setup is not None:
        from .fineweb_eval import DistributedTokenLoader, TokenStream
        if is_dist:
            token_stream = DistributedTokenLoader(
                fineweb_setup["train_pattern"],
                rank=rank,
                world_size=dist.get_world_size(),
                device=device,
            )
        else:
            token_stream = TokenStream(fineweb_setup["train_pattern"])

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == "fp16"))
    train_losses = []
    iteration_means = []
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    start_time = time.time()

    for step in range(num_steps):
        if token_stream is not None:
            if is_dist:
                inputs, targets = token_stream.next_batch(config.batch_size, config.seq_len)
            else:
                total_tokens = config.batch_size * config.seq_len + 1
                raw = token_stream.take(total_tokens).to(device=device, dtype=torch.long)
                inputs = raw[:-1].reshape(config.batch_size, config.seq_len)
                targets = raw[1:].reshape(config.batch_size, config.seq_len)
        else:
            inputs, targets = generate_mixed_batch(
                config.batch_size, config.seq_len, config.vocab_size, device=device
            )

        with autocast_context(device, amp_dtype):
            logits, details = model(inputs, return_details=False)
            loss, loss_dict = compute_loss(logits, targets, details, config)

        if math.isnan(loss.item()):
            if is_master:
                print(f"  NaN loss at step {step}, stopping early")
            break

        _zero_grad_all(optimizers)
        _update_muon_momentum(optimizers, config, step)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.unscale_(opt)
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
            _step_all(optimizers)
        for scheduler in schedulers:
            scheduler.step()

        if (
            config.swa_enabled
            and optimizers
            and (not is_dist or is_master)
            and (step + 1) % config.swa_every == 0
        ):
            base_lr = float(optimizers[0].param_groups[0].get("base_lr", 1.0))
            current_lr = float(schedulers[0].get_last_lr()[0])
            lr_scale = current_lr / max(base_lr, 1e-12)
            if lr_scale <= config.swa_start_frac:
                swa_state, swa_count = _maybe_capture_swa(swa_state, swa_count, base_model)

        train_losses.append(loss_dict["task_loss"])
        iteration_means.append(float(loss_dict["avg_iterations"]))

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
    avg_iterations_run = (
        sum(iteration_means) / len(iteration_means) if iteration_means else float(config.max_iterations)
    )
    if is_master and _maybe_apply_swa(base_model, swa_state, swa_count):
        print(f"  Applied SWA over {swa_count} checkpoints")

    if is_master:
        eval_result = evaluate_trained_model(
            base_model,
            config,
            device=device,
            fineweb_setup=final_eval_setup,
            eval_batches=eval_batches,
            amp_dtype=amp_dtype,
        )
        if eval_result["avg_iterations"] is None:
            eval_result["avg_iterations"] = avg_iterations_run
    else:
        eval_result = {"val_loss": 0, "val_bpb": None, "task_loss": 0, "avg_iterations": avg_iterations_run}

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
