"""Novelty score computation — mean distance to k-nearest neighbors in behavioral space."""

from __future__ import annotations

import numpy as np

from .behavioral import BehavioralProfile


def compute_novelty(
    profile: BehavioralProfile,
    archive_profiles: list[BehavioralProfile],
    k: int = 15,
) -> float:
    """Novelty = mean distance to k-nearest neighbors in behavioral space."""
    if len(archive_profiles) < 2:
        return 1.0  # everything is novel when archive is nearly empty

    vec = profile.to_vector()
    archive_vecs = np.array([p.to_vector() for p in archive_profiles])

    # Replace any nan/inf with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    archive_vecs = np.nan_to_num(archive_vecs, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize each dimension by archive std
    stds = archive_vecs.std(axis=0) + 1e-8
    vec_norm = vec / stds
    archive_norm = archive_vecs / stds

    distances = np.linalg.norm(archive_norm - vec_norm[np.newaxis, :], axis=1)
    k_actual = min(k, len(distances))
    k_nearest = np.sort(distances)[:k_actual]
    # Clamp novelty to a reasonable range
    return float(min(k_nearest.mean(), 10.0))
