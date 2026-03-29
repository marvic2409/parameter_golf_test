"""Configuration dataclass - the "genome" that the evolutionary search mutates."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class NeuroModConfig:
    # --- Core architecture ---
    vocab_size: int = 512
    hidden_dim: int = 128
    num_heads: int = 4
    num_kv_heads: int = 4
    num_shared_blocks: int = 2
    max_iterations: int = 6
    min_iterations_before_halt: int = 1
    share_block_weights: bool = True
    ff_mult: float = 2.0
    seq_len: int = 64
    tie_embeddings: bool = True
    use_rotary_embeddings: bool = True
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5
    bigram_hash_buckets: int = 0
    bigram_hash_dim: int = 128
    use_smear_gate: bool = False
    use_latent_workspace: bool = False
    latent_dim: int = 64
    latent_layers: int = 1
    use_residual_mix: bool = True
    use_block_skip_connections: bool = True
    eval_stride: int = 0

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
    matrix_lr: float = 0.04
    scalar_lr: float = 0.04
    embed_lr: float = 0.6
    tied_embed_lr: float = 0.05
    head_lr: float = 0.008
    tied_embed_init_std: float = 0.005
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 0.0
    swa_enabled: bool = False
    swa_start_frac: float = 0.5
    swa_every: int = 50

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
            self.bigram_hash_buckets > 0,
            self.use_smear_gate,
            self.use_latent_workspace,
            self.use_residual_mix,
            self.use_block_skip_connections,
            self.use_global_modulation, self.use_layer_modulation,
            self.use_channel_gating, self.use_iteration_encoding,
            self.use_adaptive_modulation, self.use_attractor_halt,
            self.use_learned_halt, self.use_modulator_halt,
            self.use_synaptic_depression, self.use_oscillatory_gating,
            self.use_energy_budget, self.use_inhibitory_damping,
        ])


@dataclass(frozen=True)
class MutationSettings:
    boolean_prob: float = 0.15
    continuous_prob: float = 0.20
    continuous_scale: float = 0.10
    categorical_prob: float = 0.10

    def scaled(self, multiplier: float) -> "MutationSettings":
        multiplier = max(0.0, multiplier)
        return MutationSettings(
            boolean_prob=min(1.0, self.boolean_prob * multiplier),
            continuous_prob=min(1.0, self.continuous_prob * multiplier),
            continuous_scale=min(1.0, self.continuous_scale * multiplier),
            categorical_prob=min(1.0, self.categorical_prob * multiplier),
        )


# --- Evolutionary search parameter definitions ---

BOOLEAN_PARAMS = [
    "share_block_weights",
    "use_smear_gate",
    "use_latent_workspace",
    "use_residual_mix",
    "use_block_skip_connections",
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
    "iteration_cost": (0.0, 0.1),
}

CATEGORICAL_PARAMS = {
    "halt_combination": ["any", "majority", "learned"],
    "mod_dim": [8, 16, 32],
    "bigram_hash_buckets": [0, 2048, 4096, 8192],
    "bigram_hash_dim": [64, 128, 256],
    "latent_dim": [32, 64, 96, 128],
    "latent_layers": [1, 2, 3, 4],
    "max_iterations": [3, 4, 6, 8, 10, 12],
    "min_iterations_before_halt": [1, 2, 3, 4, 6, 8],
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
        "categorical": [
            "halt_combination", "mod_dim", "bigram_hash_buckets", "latent_dim", "latent_layers", "max_iterations",
            "min_iterations_before_halt", "num_shared_blocks",
        ],
    },
    "modulation_only": {
        "boolean": list(MODULATION_BOOLEAN_PARAMS),
        "continuous": [],
        "categorical": ["mod_dim", "max_iterations", "min_iterations_before_halt", "num_shared_blocks"],
    },
    "halting_only": {
        "boolean": list(HALTING_BOOLEAN_PARAMS),
        "continuous": ["attractor_threshold", "depression_rate", "energy_budget", "iteration_cost"],
        "categorical": ["halt_combination", "max_iterations", "min_iterations_before_halt", "num_shared_blocks"],
    },
}


def _base_config(base: NeuroModConfig | None = None) -> NeuroModConfig:
    return copy.deepcopy(base) if base is not None else NeuroModConfig()


def normalize_config(config: NeuroModConfig) -> NeuroModConfig:
    config.max_iterations = max(1, int(config.max_iterations))
    config.min_iterations_before_halt = max(1, int(config.min_iterations_before_halt))
    config.min_iterations_before_halt = min(config.min_iterations_before_halt, config.max_iterations)
    config.num_heads = max(1, int(config.num_heads))
    requested_kv = max(1, int(config.num_kv_heads))
    valid_kv = [d for d in range(1, config.num_heads + 1) if config.num_heads % d == 0]
    config.num_kv_heads = max((d for d in valid_kv if d <= requested_kv), default=1)
    config.num_shared_blocks = max(1, int(config.num_shared_blocks))
    config.bigram_hash_buckets = max(0, int(config.bigram_hash_buckets))
    config.bigram_hash_dim = max(1, int(config.bigram_hash_dim))
    config.latent_dim = max(8, int(config.latent_dim))
    config.latent_layers = max(1, int(config.latent_layers))
    config.eval_stride = max(0, int(config.eval_stride))
    config.swa_every = max(1, int(config.swa_every))
    return config


def config_from_mapping(mapping: dict) -> NeuroModConfig:
    """Build a config from a partial/complete dict, ignoring unknown keys."""
    valid_fields = {f.name for f in fields(NeuroModConfig)}
    filtered = {key: value for key, value in mapping.items() if key in valid_fields}
    return normalize_config(NeuroModConfig(**filtered))


def load_config_json(path: str | Path) -> NeuroModConfig:
    """Load a config from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config JSON must contain an object at top level: {path}")
    return config_from_mapping(data)


def get_search_space_spec(search_space: str = "all") -> dict[str, list[str]]:
    if search_space not in SEARCH_SPACE_SPECS:
        raise ValueError(f"Unknown search space: {search_space}")
    return SEARCH_SPACE_SPECS[search_space]


def mutate(
    config: NeuroModConfig,
    search_space: str = "all",
    settings: MutationSettings | None = None,
) -> NeuroModConfig:
    cfg = copy.deepcopy(config)
    spec = get_search_space_spec(search_space)
    settings = settings or MutationSettings()
    for param in spec["boolean"]:
        if random.random() < settings.boolean_prob:
            setattr(cfg, param, not getattr(cfg, param))
    for param in spec["continuous"]:
        lo, hi = CONTINUOUS_PARAMS[param]
        if random.random() < settings.continuous_prob:
            val = getattr(cfg, param)
            noise = random.gauss(0, settings.continuous_scale * (hi - lo))
            setattr(cfg, param, max(lo, min(hi, val + noise)))
    for param in spec["categorical"]:
        choices = CATEGORICAL_PARAMS[param]
        if random.random() < settings.categorical_prob:
            setattr(cfg, param, random.choice(choices))
    return normalize_config(cfg)


def crossover(cfg1: NeuroModConfig, cfg2: NeuroModConfig) -> NeuroModConfig:
    child = copy.deepcopy(cfg1)
    for f in fields(NeuroModConfig):
        if f.name in (
            "vocab_size", "hidden_dim", "num_heads", "num_kv_heads", "seq_len", "ff_mult",
            "tie_embeddings", "use_rotary_embeddings", "logit_softcap", "qk_gain_init",
            "lr", "batch_size", "num_epochs", "matrix_lr", "scalar_lr", "embed_lr",
            "tied_embed_lr", "head_lr", "tied_embed_init_std", "muon_momentum", "muon_backend_steps",
            "muon_momentum_warmup_start", "muon_momentum_warmup_steps", "beta1",
            "beta2", "adam_eps", "grad_clip_norm", "eval_stride",
            "swa_enabled", "swa_start_frac", "swa_every",
        ):
            continue  # don't crossover fixed training params
        if random.random() < 0.5:
            setattr(child, f.name, getattr(cfg1, f.name))
        else:
            setattr(child, f.name, getattr(cfg2, f.name))
    return normalize_config(child)


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
    return normalize_config(cfg)


def make_all_on_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = _base_config(base)
    spec = get_search_space_spec(search_space)
    for param in spec["boolean"]:
        setattr(cfg, param, True)
    return normalize_config(cfg)


def make_minimal_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = _base_config(base)
    spec = get_search_space_spec(search_space)
    for param in spec["boolean"]:
        setattr(cfg, param, False)
    if "bigram_hash_buckets" in spec["categorical"]:
        cfg.bigram_hash_buckets = 0
    if "max_iterations" in spec["categorical"]:
        cfg.max_iterations = 4
    if "min_iterations_before_halt" in spec["categorical"]:
        cfg.min_iterations_before_halt = 1
    if "num_shared_blocks" in spec["categorical"]:
        cfg.num_shared_blocks = 2
    return normalize_config(cfg)


def make_modulation_only_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = make_minimal_config(base, search_space=search_space)
    spec = get_search_space_spec(search_space)
    for param in MODULATION_BOOLEAN_PARAMS:
        if param in spec["boolean"]:
            setattr(cfg, param, True)
    return normalize_config(cfg)


def make_halting_only_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = make_minimal_config(base, search_space=search_space)
    spec = get_search_space_spec(search_space)
    for param in ("use_attractor_halt", "use_learned_halt", "use_energy_budget"):
        if param in spec["boolean"]:
            setattr(cfg, param, True)
    if "halt_combination" in spec["categorical"]:
        cfg.halt_combination = "learned"
    return normalize_config(cfg)


def make_deep_recursion_config(base: NeuroModConfig | None = None, search_space: str = "all") -> NeuroModConfig:
    cfg = make_all_on_config(base, search_space=search_space)
    spec = get_search_space_spec(search_space)
    if "max_iterations" in spec["categorical"]:
        cfg.max_iterations = max(CATEGORICAL_PARAMS["max_iterations"])
    if "min_iterations_before_halt" in spec["categorical"]:
        cfg.min_iterations_before_halt = max(2, cfg.max_iterations // 2)
    if "iteration_cost" in spec["continuous"]:
        cfg.iteration_cost = CONTINUOUS_PARAMS["iteration_cost"][0]
    if "use_energy_budget" in spec["boolean"]:
        cfg.use_energy_budget = False
    if "use_attractor_halt" in spec["boolean"]:
        cfg.use_attractor_halt = False
    if "halt_combination" in spec["categorical"]:
        cfg.halt_combination = "learned"
    return normalize_config(cfg)


def make_preset_config(name: str) -> NeuroModConfig:
    cfg = NeuroModConfig()
    if name == "default":
        return normalize_config(cfg)
    if name == "fineweb_medium":
        cfg.hidden_dim = 384
        cfg.num_heads = 6
        cfg.num_kv_heads = 3
        cfg.ff_mult = 3.0
        cfg.use_residual_mix = True
        cfg.use_block_skip_connections = True
        cfg.mod_dim = 32
        cfg.num_shared_blocks = 2
        cfg.max_iterations = 6
        cfg.min_iterations_before_halt = 2
        cfg.batch_size = 16
        cfg.lr = 2e-4
        cfg.warmup_steps = 400
        cfg.num_cycles = 3
        cfg.min_lr_ratio = 0.08
        cfg.iteration_cost = 0.003
        return normalize_config(cfg)
    if name == "fineweb_large":
        cfg.hidden_dim = 512
        cfg.num_heads = 8
        cfg.num_kv_heads = 4
        cfg.ff_mult = 3.0
        cfg.use_residual_mix = True
        cfg.use_block_skip_connections = True
        cfg.mod_dim = 32
        cfg.num_shared_blocks = 2
        cfg.max_iterations = 6
        cfg.min_iterations_before_halt = 2
        cfg.batch_size = 8
        cfg.lr = 1.5e-4
        cfg.warmup_steps = 500
        cfg.num_cycles = 3
        cfg.min_lr_ratio = 0.08
        cfg.iteration_cost = 0.0025
        return normalize_config(cfg)
    if name == "fineweb_competitive":
        cfg.hidden_dim = 640
        cfg.num_heads = 10
        cfg.num_kv_heads = 5
        cfg.ff_mult = 3.0
        cfg.bigram_hash_buckets = 4096
        cfg.bigram_hash_dim = 128
        cfg.use_smear_gate = True
        cfg.use_residual_mix = True
        cfg.use_block_skip_connections = True
        cfg.mod_dim = 32
        cfg.num_shared_blocks = 3
        cfg.max_iterations = 8
        cfg.min_iterations_before_halt = 2
        cfg.batch_size = 6
        cfg.lr = 1.2e-4
        cfg.warmup_steps = 700
        cfg.num_cycles = 3
        cfg.min_lr_ratio = 0.08
        cfg.iteration_cost = 0.0015
        cfg.eval_stride = 64
        cfg.swa_enabled = True
        cfg.swa_start_frac = 0.5
        cfg.swa_every = 50
        return normalize_config(cfg)
    if name == "fineweb_latent_competitive":
        cfg.hidden_dim = 640
        cfg.num_heads = 10
        cfg.num_kv_heads = 5
        cfg.ff_mult = 3.0
        cfg.bigram_hash_buckets = 4096
        cfg.bigram_hash_dim = 128
        cfg.use_smear_gate = True
        cfg.use_latent_workspace = True
        cfg.latent_dim = 64
        cfg.latent_layers = 2
        cfg.use_residual_mix = True
        cfg.use_block_skip_connections = True
        cfg.mod_dim = 32
        cfg.num_shared_blocks = 3
        cfg.max_iterations = 8
        cfg.min_iterations_before_halt = 2
        cfg.batch_size = 6
        cfg.lr = 1.2e-4
        cfg.warmup_steps = 700
        cfg.num_cycles = 3
        cfg.min_lr_ratio = 0.08
        cfg.iteration_cost = 0.0015
        cfg.eval_stride = 64
        cfg.swa_enabled = True
        cfg.swa_start_frac = 0.5
        cfg.swa_every = 50
        return normalize_config(cfg)
    if name == "fineweb_baseline_parity":
        cfg.hidden_dim = 512
        cfg.num_heads = 8
        cfg.num_kv_heads = 4
        cfg.ff_mult = 2.0
        cfg.use_residual_mix = True
        cfg.use_block_skip_connections = True
        cfg.mod_dim = 16
        cfg.num_shared_blocks = 3
        cfg.max_iterations = 3
        cfg.min_iterations_before_halt = 3
        cfg.share_block_weights = False
        cfg.use_global_modulation = False
        cfg.use_layer_modulation = False
        cfg.use_channel_gating = False
        cfg.use_iteration_encoding = False
        cfg.use_adaptive_modulation = False
        cfg.use_attractor_halt = False
        cfg.use_learned_halt = False
        cfg.use_modulator_halt = False
        cfg.use_synaptic_depression = False
        cfg.use_oscillatory_gating = False
        cfg.use_energy_budget = False
        cfg.use_inhibitory_damping = False
        cfg.iteration_cost = 0.0
        cfg.batch_size = 12
        cfg.warmup_steps = 40
        cfg.num_cycles = 1
        cfg.min_lr_ratio = 0.1
        cfg.tied_embed_init_std = 0.005
        cfg.matrix_lr = 0.04
        cfg.scalar_lr = 0.04
        cfg.embed_lr = 0.6
        cfg.tied_embed_lr = 0.05
        cfg.head_lr = 0.008
        return normalize_config(cfg)
    raise ValueError(f"unknown preset {name!r}")
