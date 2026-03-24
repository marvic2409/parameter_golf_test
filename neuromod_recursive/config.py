"""Configuration dataclass — the 'genome' that the evolutionary search mutates."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, fields


@dataclass
class NeuroModConfig:
    # --- Core architecture ---
    vocab_size: int = 512
    hidden_dim: int = 128
    num_heads: int = 4
    num_shared_blocks: int = 2
    max_iterations: int = 6
    ff_mult: float = 2.0
    seq_len: int = 64

    # --- Modulation mechanisms (all toggleable) ---
    mod_dim: int = 16

    use_global_modulation: bool = True
    use_layer_modulation: bool = True
    use_channel_gating: bool = True
    use_iteration_encoding: bool = True
    use_adaptive_modulation: bool = True

    # --- Halting mechanisms (all toggleable, can stack) ---
    use_attractor_halt: bool = True
    attractor_threshold: float = 0.01

    use_learned_halt: bool = True
    use_modulator_halt: bool = True
    use_synaptic_depression: bool = True
    depression_rate: float = 0.05

    use_oscillatory_gating: bool = True
    use_energy_budget: bool = True
    energy_budget: float = 1.0

    use_inhibitory_damping: bool = True

    # --- Halt combination strategy ---
    halt_combination: str = "learned"  # 'any', 'majority', 'learned'

    # --- Training ---
    iteration_cost: float = 0.01
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 20
    warmup_steps: int = 200
    num_cycles: int = 4          # number of LR spike/decay cycles (warm restarts)
    min_lr_ratio: float = 0.05   # LR floor as fraction of peak (never fully stops learning)

    def count_active_halt_signals(self) -> int:
        count = 0
        if self.use_attractor_halt:
            count += 1
        if self.use_learned_halt:
            count += 1
        if self.use_modulator_halt:
            count += 1
        if self.use_energy_budget:
            count += 1
        return count

    def count_active_mechanisms(self) -> int:
        return sum([
            self.use_global_modulation, self.use_layer_modulation,
            self.use_channel_gating, self.use_iteration_encoding,
            self.use_adaptive_modulation, self.use_attractor_halt,
            self.use_learned_halt, self.use_modulator_halt,
            self.use_synaptic_depression, self.use_oscillatory_gating,
            self.use_energy_budget, self.use_inhibitory_damping,
        ])


# --- Evolutionary search parameter definitions ---

BOOLEAN_PARAMS = [
    "use_global_modulation", "use_layer_modulation", "use_channel_gating",
    "use_iteration_encoding", "use_adaptive_modulation",
    "use_attractor_halt", "use_learned_halt", "use_modulator_halt",
    "use_synaptic_depression", "use_oscillatory_gating",
    "use_energy_budget", "use_inhibitory_damping",
]

MODULATION_BOOLEAN_PARAMS = [
    "use_global_modulation", "use_layer_modulation", "use_channel_gating",
    "use_iteration_encoding", "use_adaptive_modulation",
]

HALTING_BOOLEAN_PARAMS = [
    "use_attractor_halt", "use_learned_halt", "use_modulator_halt",
    "use_synaptic_depression", "use_oscillatory_gating",
    "use_energy_budget", "use_inhibitory_damping",
]

CONTINUOUS_PARAMS = {
    "attractor_threshold": (0.001, 0.1),
    "depression_rate": (0.01, 0.2),
    "energy_budget": (0.5, 2.0),
    "iteration_cost": (0.001, 0.1),
}

CATEGORICAL_PARAMS = {
    "halt_combination": ["any", "majority", "learned"],
    "mod_dim": [8, 16, 32],
    "max_iterations": [3, 4, 6, 8],
    "num_shared_blocks": [1, 2, 3],
}

SEARCH_SPACE_SPECS = {
    "all": {
        "boolean": list(BOOLEAN_PARAMS),
        "continuous": list(CONTINUOUS_PARAMS.keys()),
        "categorical": list(CATEGORICAL_PARAMS.keys()),
    },
    "motif_only": {
        "boolean": list(BOOLEAN_PARAMS),
        "continuous": ["iteration_cost"],
        "categorical": ["halt_combination", "mod_dim", "max_iterations", "num_shared_blocks"],
    },
    "modulation_only": {
        "boolean": list(MODULATION_BOOLEAN_PARAMS),
        "continuous": [],
        "categorical": ["mod_dim", "max_iterations", "num_shared_blocks"],
    },
    "halting_only": {
        "boolean": list(HALTING_BOOLEAN_PARAMS),
        "continuous": ["attractor_threshold", "depression_rate", "energy_budget", "iteration_cost"],
        "categorical": ["halt_combination", "max_iterations", "num_shared_blocks"],
    },
}


def _base_config(base: NeuroModConfig | None = None) -> NeuroModConfig:
    return copy.deepcopy(base) if base is not None else NeuroModConfig()


def get_search_space_spec(search_space: str = "all") -> dict[str, list[str]]:
    if search_space not in SEARCH_SPACE_SPECS:
        raise ValueError(f"Unknown search space: {search_space}")
    return SEARCH_SPACE_SPECS[search_space]


def mutate(config: NeuroModConfig, search_space: str = "all") -> NeuroModConfig:
    cfg = copy.deepcopy(config)
    spec = get_search_space_spec(search_space)
    for param in spec["boolean"]:
        if random.random() < 0.15:
            setattr(cfg, param, not getattr(cfg, param))
    for param in spec["continuous"]:
        lo, hi = CONTINUOUS_PARAMS[param]
        if random.random() < 0.20:
            val = getattr(cfg, param)
            noise = random.gauss(0, 0.1 * (hi - lo))
            setattr(cfg, param, max(lo, min(hi, val + noise)))
    for param in spec["categorical"]:
        choices = CATEGORICAL_PARAMS[param]
        if random.random() < 0.10:
            setattr(cfg, param, random.choice(choices))
    return cfg


def crossover(cfg1: NeuroModConfig, cfg2: NeuroModConfig) -> NeuroModConfig:
    child = copy.deepcopy(cfg1)
    for f in fields(NeuroModConfig):
        if f.name in ("vocab_size", "hidden_dim", "num_heads", "seq_len", "ff_mult",
                       "lr", "batch_size", "num_epochs"):
            continue  # don't crossover fixed training params
        if random.random() < 0.5:
            setattr(child, f.name, getattr(cfg1, f.name))
        else:
            setattr(child, f.name, getattr(cfg2, f.name))
    return child


def make_random_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = _base_config(base)
    spec = get_search_space_spec(search_space)
    for param in spec["boolean"]:
        setattr(cfg, param, random.random() < 0.5)
    for param in spec["continuous"]:
        lo, hi = CONTINUOUS_PARAMS[param]
        setattr(cfg, param, random.uniform(lo, hi))
    for param in spec["categorical"]:
        choices = CATEGORICAL_PARAMS[param]
        setattr(cfg, param, random.choice(choices))
    return cfg


def make_all_on_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = _base_config(base)
    spec = get_search_space_spec(search_space)
    for param in spec["boolean"]:
        setattr(cfg, param, True)
    return cfg


def make_minimal_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = _base_config(base)
    spec = get_search_space_spec(search_space)
    for param in spec["boolean"]:
        setattr(cfg, param, False)
    if "max_iterations" in spec["categorical"]:
        cfg.max_iterations = 4
    if "num_shared_blocks" in spec["categorical"]:
        cfg.num_shared_blocks = 2
    return cfg


def make_modulation_only_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = make_minimal_config(base, search_space=search_space)
    spec = get_search_space_spec(search_space)
    for param in MODULATION_BOOLEAN_PARAMS:
        if param in spec["boolean"]:
            setattr(cfg, param, True)
    return cfg


def make_halting_only_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = make_minimal_config(base, search_space=search_space)
    spec = get_search_space_spec(search_space)
    for param in ("use_attractor_halt", "use_learned_halt", "use_energy_budget"):
        if param in spec["boolean"]:
            setattr(cfg, param, True)
    if "halt_combination" in spec["categorical"]:
        cfg.halt_combination = "learned"
    return cfg


def make_preset_config(name: str) -> NeuroModConfig:
    cfg = NeuroModConfig()
    if name == "default":
        return cfg
    if name == "fineweb_medium":
        cfg.hidden_dim = 384
        cfg.num_heads = 6
        cfg.ff_mult = 3.0
        cfg.mod_dim = 32
        cfg.num_shared_blocks = 2
        cfg.max_iterations = 4
        cfg.batch_size = 16
        cfg.lr = 2e-4
        cfg.warmup_steps = 400
        cfg.num_cycles = 3
        cfg.min_lr_ratio = 0.08
        cfg.iteration_cost = 0.005
        return cfg
    if name == "fineweb_large":
        cfg.hidden_dim = 512
        cfg.num_heads = 8
        cfg.ff_mult = 3.0
        cfg.mod_dim = 32
        cfg.num_shared_blocks = 2
        cfg.max_iterations = 4
        cfg.batch_size = 8
        cfg.lr = 1.5e-4
        cfg.warmup_steps = 500
        cfg.num_cycles = 3
        cfg.min_lr_ratio = 0.08
        cfg.iteration_cost = 0.005
        return cfg
    raise ValueError(f"unknown preset {name!r}")
