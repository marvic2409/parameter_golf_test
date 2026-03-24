"""Evaluation: loss, iteration analysis, mechanism analysis."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .config import NeuroModConfig
from .data import generate_mixed_batch
from .model import NeuroModRecursiveModel, compute_loss
from .utils import autocast_context


@torch.no_grad()
def evaluate_model(
    model: NeuroModRecursiveModel,
    config: NeuroModConfig,
    num_batches: int = 10,
    device: torch.device = torch.device("cpu"),
    amp_dtype: str | None = "none",
) -> dict:
    """Evaluate a model on synthetic data. Returns loss and iteration statistics."""
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_iterations = 0.0
    total_samples = 0
    iteration_counts = []

    for _ in range(num_batches):
        inputs, targets = generate_mixed_batch(
            config.batch_size, config.seq_len, config.vocab_size, device=device
        )
        with autocast_context(device, amp_dtype):
            logits, details = model(inputs, return_details=False)
            loss, loss_dict = compute_loss(logits, targets, details, config)

        total_loss += loss_dict["total_loss"] * len(inputs)
        total_task_loss += loss_dict["task_loss"] * len(inputs)
        total_iterations += loss_dict["avg_iterations"] * len(inputs)
        total_samples += len(inputs)
        iteration_counts.append(details["num_iterations"])

    model.train()

    avg_iters = total_iterations / total_samples
    return {
        "val_loss": total_loss / total_samples,
        "task_loss": total_task_loss / total_samples,
        "avg_iterations": avg_iters,
        "iteration_counts": iteration_counts,
    }


def compute_stability(
    model_class,
    config: NeuroModConfig,
    num_runs: int = 3,
    training_steps: int = 500,
    device: torch.device = torch.device("cpu"),
    amp_dtype: str | None = "none",
) -> float:
    """Measure training stability by variance of val_loss across multiple short runs."""
    from .train import train_single_config  # lazy import to avoid circular

    losses = []
    for seed in range(num_runs):
        result = train_single_config(
            config,
            num_steps=training_steps,
            seed=seed,
            device=device,
            quiet=True,
            amp_dtype=amp_dtype,
        )
        losses.append(result["val_loss"])

    if len(losses) < 2:
        return 0.0
    mean = sum(losses) / len(losses)
    variance = sum((l - mean) ** 2 for l in losses) / len(losses)
    return variance
