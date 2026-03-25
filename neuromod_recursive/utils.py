"""Utilities: parameter counting, seeding, config serialization."""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, fields
from typing import Optional

import numpy as np
import torch

from .config import NeuroModConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count parameters by component."""
    total = 0
    breakdown = {}
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        top = name.split(".")[0]
        breakdown[top] = breakdown.get(top, 0) + n
    return {"total": total, "breakdown": breakdown}


def config_to_dict(config: NeuroModConfig) -> dict:
    return asdict(config)


def config_from_dict(d: dict) -> NeuroModConfig:
    cfg = NeuroModConfig()
    for f in fields(NeuroModConfig):
        if f.name in d:
            setattr(cfg, f.name, d[f.name])
    return cfg


def save_config(config: NeuroModConfig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2)


def load_config(path: str) -> NeuroModConfig:
    with open(path) as f:
        return config_from_dict(json.load(f))


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_fast_cuda_math() -> None:
    """Enable fast matmul paths on modern NVIDIA GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def normalize_amp_dtype(amp_dtype: str | None, device: torch.device) -> str:
    if device.type != "cuda":
        return "none"
    if amp_dtype is None:
        return "bf16"
    amp_dtype = amp_dtype.lower()
    if amp_dtype not in {"none", "bf16", "fp16"}:
        raise ValueError(f"Unsupported amp dtype: {amp_dtype}")
    return amp_dtype


def autocast_context(device: torch.device, amp_dtype: str | None):
    resolved = normalize_amp_dtype(amp_dtype, device)
    if resolved == "none":
        return nullcontext()
    dtype = torch.bfloat16 if resolved == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def maybe_compile_model(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    from torch import _dynamo

    _dynamo.config.capture_scalar_outputs = True
    _dynamo.config.suppress_errors = True
    return compile_fn(model, mode="reduce-overhead")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def canonicalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(name.startswith(prefix) for name in state_dict):
        return state_dict
    return {
        (name[len(prefix):] if name.startswith(prefix) else name): tensor
        for name, tensor in state_dict.items()
    }


def export_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return canonicalize_state_dict(unwrap_model(model).state_dict())


def format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
