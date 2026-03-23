# NeuroMod Evolutionary Architecture Discovery — Implementation Status

## Overview

This project implements a **novelty-driven evolutionary search** over **neuromodulated recursive neural network architectures** for the OpenAI Parameter Golf challenge. The goal is to discover novel architectures that achieve competitive bits-per-byte (BPB) compression scores within a 16MB artifact limit.

### Core Idea
- **Shared recursive layers** (same weights looped N times) give effective depth without parameter cost
- **Neuromodulation** (a small side-network) dynamically retunes the main network per-input and per-iteration
- **Multiple halting mechanisms** decide when to stop recursing (adaptive computation time)
- **MAP-Elites + novelty search + speciation** discover genuinely novel architectures, not just optimize known ones

---

## File Structure

```
neuromod_recursive/
├── config.py          # NeuroModConfig dataclass (the "genome"), mutation, crossover
├── model.py           # Full model: assembles backbone + modulator + halting, forward pass, loss
├── modules/
│   ├── backbone.py    # SharedTransformerBlock, Embedding, OutputHead
│   ├── modulator.py   # ModulatorNetwork + extract_block_modulation
│   ├── halting.py     # 6 halt mechanisms + HaltCombiner
│   └── oscillator.py  # OscillatoryGating (gamma/beta-inspired per-block rhythm)
├── data.py            # Synthetic data: parenthesis, algorithmic, pattern, mixed
├── train.py           # Single-config training + DDP multi-GPU support
├── novelty/
│   ├── behavioral.py  # BehavioralProfile + diagnostic probes + profiling
│   ├── map_elites.py  # MAPElitesArchive (4D: modulation, halting, depth, iteration variance)
│   ├── speciation.py  # SpeciationManager + structural_distance
│   └── novelty.py     # k-nearest neighbor novelty computation
├── search.py          # Full evolutionary search loop (MAP-Elites + speciation + novelty)
├── evaluate.py        # Evaluation on synthetic data
├── visualize.py       # Plotting: fitness, coverage, species, novelty, mechanism frequency
├── utils.py           # Seeding, param counting, config serialization
└── run_search.py      # CLI entry point
```

---

## Architecture Components

### Backbone (SharedRecursiveTransformer)
- Token + positional embeddings (applied once)
- `num_shared_blocks` transformer blocks (pre-norm, multi-head attention, GELU FFN)
- Blocks expose modulation hooks: attn_scale/shift, ffn_scale/shift, channel_gate, weight_scale, residual_scale
- Output head: LayerNorm → Linear

### Modulator Network
- Small MLP (2 layers, hidden=64)
- Inputs: mean-pooled embeddings + (optional) sinusoidal iteration encoding + (optional) current hidden state
- Outputs: global scale/shift, per-layer FiLM scale/shift, per-channel sigmoid gates, halt signal
- Initialized as identity (scale=1, shift=0, gates~0.7)

### Halting Mechanisms (all toggleable)
1. **AttractorHalt** — convergence detection (no learnable params)
2. **LearnedHalt** — ACT-style cumulative probability (Graves 2016)
3. **ModulatorHalt** — modulator outputs explicit "done" signal
4. **EnergyBudgetHalt** — learned per-iteration cost depleting a budget
5. **SynapticDepression** — weight attenuation per iteration (makes convergence natural)
6. **InhibitoryDamping** — feedback inhibition damping residual connections
7. **OscillatoryGating** — learned sin-wave rhythm per block

### Halt Combiner
- 'any': OR logic
- 'majority': >50% vote
- 'learned': tiny neural network combines signals (recommended)

### Forward Pass
- Training: runs all iterations, ACT-style weighted combination of hidden states
- Inference: hard-halts when cumulative probability ≥ threshold

---

## Evolutionary Search

### Genome
The `NeuroModConfig` dataclass is the genome. It has:
- 12 boolean toggles (mechanisms on/off)
- 4 continuous hyperparameters
- 4 categorical/discrete choices

### Mutation & Crossover
- Booleans: 15% flip probability each
- Continuous: 20% Gaussian perturbation (σ = 10% of range)
- Categorical: 10% random reselection
- Crossover: uniform (each param from random parent)

### MAP-Elites Archive
4 dimensions × 5 bins each = 625 cells:
1. Modulation complexity (0-4 active modulation mechanisms)
2. Halting complexity (0-4 active halt mechanisms)
3. Effective depth (blocks × max_iterations)
4. Iteration variance (from behavioral profile)

### Speciation
- Structural distance based on boolean/continuous/categorical param differences
- Species protect novel architectures from premature elimination
- Stagnant species (no improvement for N generations) are pruned

### Novelty Score
- k-nearest neighbor distance in behavioral feature space
- Behavioral features: iteration dynamics, halt trigger rates, confidence, output diversity, etc.
- Clamped to [0, 10] to prevent explosion

### Composite Fitness
```
fitness = w_quality * (-val_loss)
        + w_novelty * novelty
        + w_efficiency * (-0.1 * avg_iterations)
        + w_simplicity * (-num_active_mechanisms)
```

### Adaptive Novelty Weight
If archive coverage stalls for 5 generations, novelty_weight increases by 20% (up to 1.5).

---

## Usage

```bash
# Quick smoke test (synthetic data, CPU):
python -m neuromod_recursive.run_search --smoke-test --device cpu

# Single config on FineWeb with BPB scoring (the challenge metric):
python -m neuromod_recursive.run_search --single --use-fineweb --steps 5000

# Mini search on FineWeb (~2 hours on 1x H100):
python -m neuromod_recursive.run_search --use-fineweb --population 10 --generations 5 --steps 1000

# Full search on FineWeb (~10-20 hours on 1x H100):
python -m neuromod_recursive.run_search --use-fineweb --population 30 --generations 20 --steps 2000

# Multi-GPU with DDP:
torchrun --standalone --nproc_per_node=4 -m neuromod_recursive.run_search --distributed --single --use-fineweb

# Generate visualizations from results:
python -m neuromod_recursive.visualize search_results/

# RunPod setup:
bash setup_runpod.sh        # Downloads 1 shard
bash setup_runpod.sh 10     # Downloads 10 shards
```

---

## Current Status

### DONE
- [x] Phase 1: Config dataclass with all toggles and evolutionary operators
- [x] Phase 2: Backbone (shared transformer blocks with modulation hooks)
- [x] Phase 3: Modulator network (global, layer, channel gating, iteration encoding)
- [x] Phase 4: All 7 halting mechanisms + halt combiner (any/majority/learned)
- [x] Phase 5: Full model assembly with ACT-weighted forward pass
- [x] Phase 6: Synthetic data generators (parenthesis, algorithmic, pattern, mixed)
- [x] Phase 7: Training loop (single GPU + DDP multi-GPU)
- [x] Phase 8: Behavioral characterization (profile extraction, diagnostic probes)
- [x] Phase 9: MAP-Elites archive (4D quality-diversity)
- [x] Phase 10: Speciation system (structural distance, species management)
- [x] Phase 11: Novelty score computation (k-NN in behavioral space)
- [x] Phase 12: Evolutionary search harness (selection, reproduction, adaptive novelty)
- [x] Phase 13: Visualization (fitness curves, coverage, species, mechanism frequency)
- [x] Phase 14: CLI entry point with smoke test mode

- [x] Phase 15: FineWeb data loading + BPB evaluation (matches official scoring)
- [x] Phase 16: --use-fineweb flag for real data training + eval
- [x] Phase 17: RunPod setup script (setup_runpod.sh)
- [x] Phase 18: NaN handling for diverged training configs

### TODO / Future Work
- [ ] Scale up to ~17M params (hidden_dim=512) for competitive runs
- [ ] Run full evolutionary search (30 pop × 20 gen) on GPU with FineWeb
- [ ] Retrain top-5 discovered architectures for full duration
- [ ] Generate MAP-Elites heatmaps and novelty vs quality Pareto plots
- [ ] Ablation studies on top configs
- [ ] Port winning architecture to train_gpt.py format for Parameter Golf submission
- [ ] Tighten search-time artifact accounting to include code bytes in addition to compressed model bytes
- [ ] Add attention entropy tracking to behavioral profile
- [ ] Add modulation statistics tracking (magnitude, drift) during profiling
- [ ] UMAP/t-SNE visualization of behavioral embedding space

### Known Issues
- Novelty score clamped to 10.0 to prevent explosion from extreme outlier profiles
- Search-time artifact accounting still measures compressed model bytes, not final submission code+model bytes
- On CPU, each config evaluation takes ~20-70s; a full search (600 evals) would take ~6-12 hours

---

## Key Design Decisions
- **ACT-style weighted output during training** makes halting differentiable
- **Modulation initialized as identity** prevents gradient death at start
- **Fixed-length behavioral vectors** (with padding) ensure consistent novelty computation
- **Species hash by ID** for use as dict keys in offspring allocation
- **Adaptive novelty weight** prevents both stalling exploration and sacrificing quality
