"""MAP-Elites quality-diversity archive."""

from __future__ import annotations

import random
from typing import Any, Optional

from ..config import NeuroModConfig
from .behavioral import BehavioralProfile

MAP_ELITES_DIMENSIONS = {
    "modulation_complexity": {
        "compute_from_config": lambda cfg: sum([
            cfg.use_global_modulation,
            cfg.use_layer_modulation,
            cfg.use_channel_gating,
            cfg.use_adaptive_modulation,
        ]),
        "bins": [0, 1, 2, 3, 4],
    },
    "halting_complexity": {
        "compute_from_config": lambda cfg: sum([
            cfg.use_attractor_halt,
            cfg.use_learned_halt,
            cfg.use_modulator_halt,
            cfg.use_energy_budget,
        ]),
        "bins": [0, 1, 2, 3, 4],
    },
    "effective_depth": {
        "compute_from_config": lambda cfg: cfg.num_shared_blocks * cfg.max_iterations,
        "bins": [0, 6, 12, 18, 24, 30, 36],
    },
    "iteration_variance": {
        "compute_from_profile": lambda profile: profile.iteration_variance,
        "bins": [0.0, 0.5, 1.0, 2.0, 5.0],
    },
}


class MAPElitesArchive:
    def __init__(self, dimensions: Optional[dict] = None):
        self.dimensions = dimensions or MAP_ELITES_DIMENSIONS
        self.grid: dict[tuple, tuple[NeuroModConfig, float, BehavioralProfile]] = {}
        self.all_profiles: list[BehavioralProfile] = []

    def _get_bin(self, value: float, bins: list) -> int:
        """Find which bin a value falls into."""
        for i in range(len(bins) - 1, -1, -1):
            if value >= bins[i]:
                return i
        return 0

    def _get_cell(self, config: NeuroModConfig, profile: BehavioralProfile) -> tuple:
        cell = []
        for dim_name, dim_spec in self.dimensions.items():
            if "compute_from_config" in dim_spec:
                value = dim_spec["compute_from_config"](config)
            elif "compute_from_profile" in dim_spec:
                value = dim_spec["compute_from_profile"](profile)
            else:
                value = 0
            cell.append(self._get_bin(value, dim_spec["bins"]))
        return tuple(cell)

    def add(
        self,
        config: NeuroModConfig,
        fitness: float,
        profile: BehavioralProfile,
    ) -> bool:
        """Add to archive. Returns True if new niche or improvement."""
        cell = self._get_cell(config, profile)
        self.all_profiles.append(profile)

        if cell not in self.grid or fitness > self.grid[cell][1]:
            self.grid[cell] = (config, fitness, profile)
            return True
        return False

    def coverage(self) -> float:
        """Fraction of cells filled."""
        total_cells = 1
        for dim_spec in self.dimensions.values():
            total_cells *= len(dim_spec["bins"])
        return len(self.grid) / total_cells

    def sample_parent(self) -> NeuroModConfig:
        """Sample a parent from the archive (uniform over filled cells)."""
        cells = list(self.grid.keys())
        cell = random.choice(cells)
        return self.grid[cell][0]

    def best_configs(self, n: int = 5) -> list[tuple[NeuroModConfig, float, BehavioralProfile]]:
        """Return top-n configs by fitness."""
        sorted_entries = sorted(self.grid.values(), key=lambda x: x[1], reverse=True)
        return sorted_entries[:n]

    def stats(self) -> dict:
        """Return archive statistics."""
        if not self.grid:
            return {"coverage": 0.0, "num_filled": 0, "best_fitness": 0.0, "mean_fitness": 0.0}
        fitnesses = [f for _, f, _ in self.grid.values()]
        return {
            "coverage": self.coverage(),
            "num_filled": len(self.grid),
            "best_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "worst_fitness": min(fitnesses),
        }
