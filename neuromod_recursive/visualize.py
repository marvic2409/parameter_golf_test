"""Visualization: training curves, MAP-Elites heatmaps, mechanism frequencies, novelty plots.

All plots save to disk (no display required). Uses matplotlib if available, falls back to CSV.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def load_generation_logs(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, "generation_logs.json")
    with open(path) as f:
        return json.load(f)


def load_top_configs(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, "top_configs.json")
    with open(path) as f:
        return json.load(f)


def load_mechanism_frequency(output_dir: str) -> dict:
    path = os.path.join(output_dir, "mechanism_frequency.json")
    with open(path) as f:
        return json.load(f)


def plot_all(output_dir: str, save_dir: Optional[str] = None):
    """Generate all visualizations from search results."""
    if save_dir is None:
        save_dir = os.path.join(output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("matplotlib not available, saving CSV only")

    logs = load_generation_logs(output_dir)

    if HAS_MPL:
        _plot_fitness_curves(logs, save_dir, plt)
        _plot_coverage(logs, save_dir, plt)
        _plot_species(logs, save_dir, plt)
        _plot_novelty(logs, save_dir, plt)

    # Always try mechanism frequency
    try:
        freq = load_mechanism_frequency(output_dir)
        if HAS_MPL:
            _plot_mechanism_frequency(freq, save_dir, plt)
    except FileNotFoundError:
        pass

    # Save CSV summary
    _save_csv_summary(logs, save_dir)
    print(f"Plots saved to {save_dir}/")


def _plot_fitness_curves(logs, save_dir, plt):
    gens = [l["generation"] for l in logs]
    best_f = [l["fitness_best"] for l in logs]
    mean_f = [l["fitness_mean"] for l in logs]
    best_vl = [l["val_loss_best"] for l in logs]
    mean_vl = [l["val_loss_mean"] for l in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(gens, best_f, "b-o", label="Best fitness", markersize=4)
    ax1.plot(gens, mean_f, "r--", label="Mean fitness", alpha=0.7)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.set_title("Fitness over Generations")
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, best_vl, "g-o", label="Best val_loss", markersize=4)
    ax2.plot(gens, mean_vl, "orange", linestyle="--", label="Mean val_loss", alpha=0.7)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Validation Loss")
    ax2.legend()
    ax2.set_title("Validation Loss over Generations")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fitness_curves.png"), dpi=150)
    plt.close()


def _plot_coverage(logs, save_dir, plt):
    gens = [l["generation"] for l in logs]
    coverage = [l["archive"]["coverage"] for l in logs]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, coverage, "b-o", markersize=4)
    plt.xlabel("Generation")
    plt.ylabel("Archive Coverage")
    plt.title("MAP-Elites Archive Coverage")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "archive_coverage.png"), dpi=150)
    plt.close()


def _plot_species(logs, save_dir, plt):
    gens = [l["generation"] for l in logs]
    num_species = [l["species"]["num_species"] for l in logs]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, num_species, "m-o", markersize=4)
    plt.xlabel("Generation")
    plt.ylabel("Number of Species")
    plt.title("Species Diversity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "species_count.png"), dpi=150)
    plt.close()


def _plot_novelty(logs, save_dir, plt):
    gens = [l["generation"] for l in logs]
    novelty = [l["novelty_mean"] for l in logs]
    nw = [l.get("novelty_weight", 0.5) for l in logs]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(gens, novelty, "c-o", label="Mean novelty", markersize=4)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Novelty Score", color="c")

    ax2 = ax1.twinx()
    ax2.plot(gens, nw, "r--", label="Novelty weight", alpha=0.7)
    ax2.set_ylabel("Novelty Weight", color="r")

    ax1.set_title("Novelty Dynamics")
    ax1.grid(True, alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "novelty_dynamics.png"), dpi=150)
    plt.close()


def _plot_mechanism_frequency(freq, save_dir, plt):
    names = list(freq.keys())
    values = list(freq.values())

    # Shorten names for display
    short = [n.replace("use_", "").replace("_", " ") for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(short, values, color="steelblue")
    plt.xlabel("Frequency in Archive")
    plt.title("Mechanism Frequency (MAP-Elites Archive)")
    plt.xlim(0, 1)
    for bar, val in zip(bars, values):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0%}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mechanism_frequency.png"), dpi=150)
    plt.close()


def _save_csv_summary(logs, save_dir):
    path = os.path.join(save_dir, "generation_summary.csv")
    with open(path, "w") as f:
        f.write("gen,best_fitness,mean_fitness,best_val_loss,mean_val_loss,coverage,num_species,novelty_mean,time_s\n")
        for l in logs:
            f.write(
                f"{l['generation']},{l['fitness_best']:.4f},{l['fitness_mean']:.4f},"
                f"{l['val_loss_best']:.4f},{l['val_loss_mean']:.4f},"
                f"{l['archive']['coverage']:.4f},{l['species']['num_species']},"
                f"{l['novelty_mean']:.4f},{l['time_seconds']:.0f}\n"
            )


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "search_results"
    plot_all(output_dir)
