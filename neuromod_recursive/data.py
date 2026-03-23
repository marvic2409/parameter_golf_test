"""Synthetic data generators for training and probing."""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch import Tensor


# --- Option A: Nested Parenthesis Sequences ---

def generate_parenthesis_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    max_depth: int = 10,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """Generate sequences with nested parentheses for next-token prediction.

    Token mapping:
      0 = PAD, 1 = OPEN '(', 2 = CLOSE ')', 3..vocab_size-1 = filler tokens
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        seq = _generate_single_paren_seq(seq_len, vocab_size, max_depth)
        inputs.append(seq[:-1])
        targets.append(seq[1:])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def _generate_single_paren_seq(seq_len: int, vocab_size: int, max_depth: int) -> list[int]:
    """Build a sequence mixing parens and filler tokens."""
    OPEN, CLOSE = 1, 2
    seq = []
    depth = 0
    target_depth = random.randint(1, max_depth)

    for _ in range(seq_len + 1):
        r = random.random()
        if depth < target_depth and r < 0.3:
            seq.append(OPEN)
            depth += 1
        elif depth > 0 and r < 0.6:
            seq.append(CLOSE)
            depth -= 1
        else:
            # Filler token
            seq.append(random.randint(3, min(vocab_size - 1, 50)))

    # Pad or truncate
    seq = seq[: seq_len + 1]
    while len(seq) < seq_len + 1:
        seq.append(0)
    return seq


# --- Option B: Algorithmic Sequence Prediction ---

def generate_algorithmic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """Generate algorithmic pattern sequences.

    Token 0 = PAD, tokens 1-4 = task markers (copy, reverse, sort, repeat),
    tokens 5+ = data tokens.
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        seq = _generate_single_algo_seq(seq_len, vocab_size)
        inputs.append(seq[:-1])
        targets.append(seq[1:])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def _generate_single_algo_seq(seq_len: int, vocab_size: int) -> list[int]:
    data_range = min(vocab_size - 5, 50)
    task = random.choice(["copy", "reverse", "sort", "repeat"])
    task_token = {"copy": 1, "reverse": 2, "sort": 3, "repeat": 4}[task]

    seg_len = random.randint(3, min(seq_len // 3, 12))
    data = [random.randint(5, 5 + data_range - 1) for _ in range(seg_len)]

    if task == "copy":
        result = list(data)
    elif task == "reverse":
        result = list(reversed(data))
    elif task == "sort":
        result = sorted(data)
    elif task == "repeat":
        result = data * 2
    else:
        result = data

    # Format: [task_token] [data...] [0 separator] [result...]
    seq = [task_token] + data + [0] + result
    seq = seq[: seq_len + 1]
    while len(seq) < seq_len + 1:
        seq.append(0)
    return seq


# --- Option C: Multi-Scale Pattern Completion ---

def generate_pattern_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """Generate sequences with patterns at multiple scales."""
    inputs = []
    targets = []
    for _ in range(batch_size):
        seq = _generate_pattern_seq(seq_len, vocab_size)
        inputs.append(seq[:-1])
        targets.append(seq[1:])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def _generate_pattern_seq(seq_len: int, vocab_size: int) -> list[int]:
    data_range = min(vocab_size - 1, 50)
    pattern_type = random.choice(["periodic", "fibonacci_mod", "counting", "alternating"])

    seq = []
    if pattern_type == "periodic":
        period = random.randint(2, 6)
        base = [random.randint(1, data_range) for _ in range(period)]
        for i in range(seq_len + 1):
            seq.append(base[i % period])
    elif pattern_type == "fibonacci_mod":
        a, b = random.randint(1, data_range), random.randint(1, data_range)
        seq.append(a)
        seq.append(b)
        for _ in range(seq_len - 1):
            c = (a + b) % data_range + 1
            seq.append(c)
            a, b = b, c
    elif pattern_type == "counting":
        start = random.randint(1, data_range // 2)
        step = random.randint(1, 3)
        for i in range(seq_len + 1):
            seq.append((start + i * step) % data_range + 1)
    elif pattern_type == "alternating":
        a = random.randint(1, data_range)
        b = random.randint(1, data_range)
        for i in range(seq_len + 1):
            seq.append(a if i % 2 == 0 else b)

    return seq[: seq_len + 1]


# --- Mixed task generator ---

def generate_mixed_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
    task_weights: Optional[dict] = None,
) -> tuple[Tensor, Tensor]:
    """Generate a mixed batch from all task types."""
    if task_weights is None:
        task_weights = {"paren": 0.4, "algo": 0.3, "pattern": 0.3}

    generators = {
        "paren": generate_parenthesis_batch,
        "algo": generate_algorithmic_batch,
        "pattern": generate_pattern_batch,
    }

    all_inputs = []
    all_targets = []
    remaining = batch_size

    for task, weight in task_weights.items():
        n = max(1, int(batch_size * weight)) if remaining > 0 else 0
        n = min(n, remaining)
        if n > 0:
            inp, tgt = generators[task](n, seq_len, vocab_size, device=device)
            all_inputs.append(inp)
            all_targets.append(tgt)
            remaining -= n

    # Handle rounding remainders
    if remaining > 0:
        inp, tgt = generate_parenthesis_batch(remaining, seq_len, vocab_size, device=device)
        all_inputs.append(inp)
        all_targets.append(tgt)

    return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)
