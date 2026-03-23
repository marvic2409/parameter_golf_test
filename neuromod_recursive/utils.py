"""Utilities: parameter counting, seeding, config serialization."""

from __future__ import annotations

import json
import os
import random
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


def format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
