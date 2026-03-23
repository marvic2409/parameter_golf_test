"""Speciation system — protects novel architectures from premature elimination."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import NeuroModConfig, BOOLEAN_PARAMS


def structural_distance(cfg1: NeuroModConfig, cfg2: NeuroModConfig) -> float:
    """Measure structural distance between two configs."""
    distance = 0.0

    # Boolean toggles: each mismatch adds 1.0
    for param in BOOLEAN_PARAMS:
        if getattr(cfg1, param) != getattr(cfg2, param):
            distance += 1.0

    # Categorical: mismatch adds 1.0
    if cfg1.halt_combination != cfg2.halt_combination:
        distance += 1.0

    # Continuous params: normalized absolute difference, weighted 0.5
    continuous = {
        "attractor_threshold": (0.001, 0.1),
        "depression_rate": (0.01, 0.2),
        "energy_budget": (0.5, 2.0),
        "max_iterations": (3, 8),
        "num_shared_blocks": (1, 3),
    }
    for param, (lo, hi) in continuous.items():
        v1 = getattr(cfg1, param)
        v2 = getattr(cfg2, param)
        distance += 0.5 * abs(v1 - v2) / (hi - lo + 1e-8)

    return distance


@dataclass
class Species:
    representative: NeuroModConfig
    members: list = field(default_factory=list)
    best_fitness: float = float("-inf")
    stagnation_counter: int = 0
    id: int = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Species) and self.id == other.id

    def update_fitness(self, fitness: float):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1


class SpeciationManager:
    def __init__(self, threshold: float = 4.0, max_stagnation: int = 10):
        self.species: list[Species] = []
        self.threshold = threshold
        self.max_stagnation = max_stagnation
        self._next_id = 0

    def assign_species(self, config: NeuroModConfig) -> Species:
        """Assign a config to an existing species or create a new one."""
        for sp in self.species:
            if structural_distance(config, sp.representative) < self.threshold:
                sp.members.append(config)
                return sp
        new_sp = Species(representative=config, id=self._next_id)
        self._next_id += 1
        new_sp.members.append(config)
        self.species.append(new_sp)
        return new_sp

    def clear_members(self):
        """Clear member lists for a new generation."""
        for sp in self.species:
            sp.members = []

    def prune_stagnant(self):
        """Remove species that haven't improved in max_stagnation generations."""
        self.species = [
            sp for sp in self.species
            if sp.stagnation_counter < self.max_stagnation
        ]

    def allocate_offspring(self, total_offspring: int) -> dict[Species, int]:
        """Allocate reproductive budget proportional to species, with minimums."""
        self.prune_stagnant()

        if not self.species:
            return {}

        min_per = 1
        remaining = total_offspring - len(self.species) * min_per
        remaining = max(0, remaining)

        # Use adjusted fitness (offset so all positive)
        fitnesses = [max(sp.best_fitness, -100.0) for sp in self.species]
        min_f = min(fitnesses)
        adjusted = [f - min_f + 0.1 for f in fitnesses]
        total_f = sum(adjusted) + 1e-8

        allocations = {}
        for sp, adj_f in zip(self.species, adjusted):
            share = int(remaining * adj_f / total_f)
            allocations[sp] = min_per + share
        return allocations

    def stats(self) -> dict:
        return {
            "num_species": len(self.species),
            "species_sizes": [len(sp.members) for sp in self.species],
            "species_best_fitness": [sp.best_fitness for sp in self.species],
            "stagnation": [sp.stagnation_counter for sp in self.species],
        }
