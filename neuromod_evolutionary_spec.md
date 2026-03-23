# Neuromodulated Recursive Neural Network — Evolutionary Architecture Discovery Spec

## Project Goal

Build a modular, configurable PyTorch framework for a ~1M parameter language model that uses:
1. **Shared recursive layers** (same weights looped multiple times per forward pass)
2. **Biologically-inspired neuromodulation** (a small side-network that dynamically retunes the main network's behavior per-input and per-iteration)
3. **Multiple halting mechanisms** (diverse ways to decide when to stop recursing)
4. **Novelty-driven evolutionary search** (MAP-Elites + novelty search + speciation to discover genuinely novel architectures, not just optimize known ones)

Every mechanism is independently toggleable via a config dataclass. A quality-diversity evolutionary harness explores the configuration space to find which combinations produce the best AND most novel performance, letting useful behaviors **emerge** from selection pressure rather than being hand-designed.

### Critical Design Philosophy

Standard NAS optimizes for performance and rediscovers known architectures. This project explicitly **decouples search from pure objective optimization** (following Kenneth Stanley's insight). We reward both quality AND behavioral novelty, protect immature-but-structurally-novel architectures via speciation, and maintain a MAP-Elites archive that forces exploration into weird corners of architecture space.

---

## Architecture Overview

There are 5 components: **Backbone**, **Modulator**, **Halt Controller**, **Novelty-Aware Search Harness**, and **Behavioral Characterization**.

```
Input tokens
    │
    ▼
[Embedding]
    │
    ▼
[Modulator] ◄── reads input embedding + iteration index + (optionally) evolving hidden state
    │ outputs: global_mod, layer_mods, channel_gates, halt_signal
    ▼
┌─────────────────────────────────────────────┐
│  Recursive Loop (up to max_iterations)      │
│                                             │
│   for i in range(max_iterations):           │
│     mod = modulator(context, i, h)          │
│     apply modulation to shared blocks       │
│     h = shared_blocks(h)                    │
│     apply synaptic depression               │
│     apply inhibitory damping                │
│     apply oscillatory gating                │
│     check all halt conditions               │
│     if halted: break                        │
│                                             │
└─────────────────────────────────────────────┘
    │
    ▼
[Output head] → logits over vocab
    │
    ▼
[Behavioral Characterization] → novelty score, MAP-Elites cell placement
```

---

## Component 1: Configuration Dataclass

Create a `NeuroModConfig` dataclass that holds every hyperparameter and toggle. This is the "genome" that the evolutionary search mutates.

```python
@dataclass
class NeuroModConfig:
    # --- Core architecture ---
    vocab_size: int = 512          # small vocab for testing; increase for real tasks
    hidden_dim: int = 128          # main hidden size
    num_heads: int = 4             # attention heads
    num_shared_blocks: int = 2     # number of unique transformer blocks (shared across iterations)
    max_iterations: int = 6        # max recursive passes
    ff_mult: float = 2.0           # feedforward expansion multiplier
    seq_len: int = 64              # sequence length for training

    # --- Modulation mechanisms (all toggleable) ---
    mod_dim: int = 16              # dimensionality of modulation code vector

    use_global_modulation: bool = True     # whole-model scale/shift from modulator
    use_layer_modulation: bool = True      # per-layer FiLM-style scale/shift
    use_channel_gating: bool = True        # per-channel multiplicative gating (like receptor density)
    use_iteration_encoding: bool = True    # modulator receives iteration index
    use_adaptive_modulation: bool = True   # modulator sees evolving hidden state (not just initial input)

    # --- Halting mechanisms (all toggleable, can stack) ---
    use_attractor_halt: bool = True        # stop when hidden state change < threshold
    attractor_threshold: float = 0.01

    use_learned_halt: bool = True          # ACT-style learned halting probability
    use_modulator_halt: bool = True        # modulator outputs explicit "done" signal
    use_synaptic_depression: bool = True   # weights weaken per iteration
    depression_rate: float = 0.05

    use_oscillatory_gating: bool = True    # gamma/beta-inspired oscillatory modulation per iteration
    use_energy_budget: bool = True         # fixed compute budget that depletes per iteration
    energy_budget: float = 1.0

    use_inhibitory_damping: bool = True    # feedback inhibition that damps recurrent activity over time

    # --- Halt combination strategy ---
    # 'any' = halt if any signal fires
    # 'majority' = halt if >50% of active halt signals fire
    # 'learned' = a tiny network combines all halt signals into one decision
    halt_combination: str = 'learned'

    # --- Training ---
    iteration_cost: float = 0.01   # penalty added to loss per iteration used (encourages efficiency)
    lr: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 20
```

---

## Component 2: Backbone (SharedRecursiveTransformer)

### 2.1 Embedding Layer
- Token embedding: `nn.Embedding(vocab_size, hidden_dim)`
- Learned positional embedding: `nn.Embedding(seq_len, hidden_dim)`
- These are NOT shared/recursive — applied once at input

### 2.2 Shared Transformer Blocks
- `num_shared_blocks` unique transformer blocks, each containing:
  - Multi-head self-attention (hidden_dim, num_heads)
  - LayerNorm (pre-norm style)
  - Feedforward: hidden_dim → hidden_dim * ff_mult → hidden_dim (with GELU)
  - LayerNorm
  - Residual connections
- During the recursive loop, ALL shared blocks are applied sequentially per iteration. So if you have 2 blocks and 4 iterations, the effective depth is 8 layers, but only 2 sets of weights.

### 2.3 Modulation Hooks
Each shared block must expose hooks for modulation to be applied:
- **Pre-attention modulation point**: scale/shift the input to attention
- **Pre-FFN modulation point**: scale/shift the input to the feedforward
- **Weight-level modulation**: multiplicative scaling on the attention projection weights and FFN weights

Implement this as each block having a method like:
```python
def forward(self, x, modulation=None):
    # modulation is a dict with keys like 'attn_scale', 'attn_shift', 'ffn_scale', 'ffn_shift', 'weight_scale'
    # Apply scale/shift to activations, apply weight_scale to weight matrices
```

### 2.4 Output Head
- LayerNorm → Linear(hidden_dim, vocab_size)
- Applied to the final hidden state after recursion completes

---

## Component 3: Modulator Network

A small network (~100-150K params) that generates modulation signals. It operates as follows:

### Inputs (concatenated into a single vector):
1. **Input summary**: mean-pooled embedding of the input sequence → dim hidden_dim, projected down to mod_dim
2. **Iteration index** (if `use_iteration_encoding`): sinusoidal encoding of the current iteration, dim mod_dim
3. **Current hidden state summary** (if `use_adaptive_modulation`): mean-pooled current hidden state → projected to mod_dim

These are concatenated and fed through a small MLP (2 layers, hidden size ~64).

### Outputs (all conditional on config toggles):

1. **Global modulation** (if `use_global_modulation`):
   - A scale scalar and shift scalar applied uniformly to all activations after each block
   - Shape: (2,) — one scale, one shift

2. **Layer modulation** (if `use_layer_modulation`):
   - Per-block scale and shift vectors
   - Shape: (num_shared_blocks, hidden_dim, 2) — FiLM-style per-channel scale and shift per block
   - Implemented as a learned linear projection from mod code → per-layer params

3. **Channel gating** (if `use_channel_gating`):
   - Sigmoid-gated per-channel multiplier per block (like receptor expression controlling gain per "neuron")
   - Shape: (num_shared_blocks, hidden_dim)
   - Values in [0, 1] via sigmoid

4. **Halt signal** (if `use_modulator_halt`):
   - A scalar sigmoid output representing "confidence that processing is complete"
   - Shape: (1,)

### Implementation Notes:
- The modulator should be lightweight. Use a 2-layer MLP with hidden size 64.
- All output heads are small linear projections from the MLP's output.
- Initialize scale outputs near 1.0 and shift outputs near 0.0 (so modulation starts as identity).
- Initialize channel gates with positive bias (so gates start ~open).

---

## Component 4: Halting Mechanisms

Implement each as a separate small module. The forward pass collects halt signals from all active mechanisms and combines them.

### 4.1 Attractor Convergence Halt (`use_attractor_halt`)
```
delta = (h_new - h_old).norm() / h_old.norm()
halt_signal = (delta < attractor_threshold).float()
```
- No learnable params. Pure convergence detection.
- The threshold is a config hyperparameter the search can tune.

### 4.2 Learned Halt / ACT (`use_learned_halt`)
- A tiny linear layer: hidden_dim → 1, with sigmoid activation
- Reads the mean-pooled hidden state
- Outputs a halting probability p_halt ∈ (0, 1)
- Accumulate p_halt across iterations. Halt when cumulative probability ≥ 1 - epsilon
- This is Adaptive Computation Time (Graves 2016). Use the remainder-based weighting scheme from that paper for the final output: the hidden states from each iteration are weighted by their halting probabilities.

### 4.3 Modulator-Driven Halt (`use_modulator_halt`)
- The modulator already outputs a halt signal (see Component 3)
- Treat as another halt probability input
- This is distinct from learned halt because it's driven by the modulator's view of the input + iteration, not just the hidden state

### 4.4 Synaptic Depression (`use_synaptic_depression`)
- NOT a halt signal directly — it's a mechanism that makes halting more likely over iterations
- Before each iteration, scale the shared block weights: `effective_weight = base_weight * (1 - depression_rate) ** iteration`
- This naturally attenuates the network's ability to change the hidden state, making attractor convergence happen faster
- The depression_rate is a tunable hyperparameter
- Implement by modifying the modulation dict to include a depression multiplier, NOT by modifying weight tensors in-place (that would break autograd)

### 4.5 Oscillatory Gating (`use_oscillatory_gating`)
- Generate a per-iteration oscillatory multiplier that modulates the residual stream
- Use a learned "frequency" and "phase" parameter per block
- `gate = sigmoid(amplitude * sin(2π * freq * iteration + phase))`
- This creates a rhythm: some iterations are "processing" iterations (gate open), others are "consolidation" (gate partially closed)
- 3 learnable params per block: amplitude, frequency, phase

### 4.6 Energy Budget (`use_energy_budget`)
- Start with `energy = energy_budget` (e.g., 1.0)
- Each iteration costs some energy: `cost = sigmoid(energy_cost_head(h_pooled))` — a learned estimate of how much energy this iteration used
- `energy -= cost`
- When energy ≤ 0, force halt
- This lets the network learn to spend more compute on harder inputs
- Learnable: the energy_cost_head (linear hidden_dim → 1)

### 4.7 Inhibitory Damping (`use_inhibitory_damping`)
- Maintain a running "inhibition" accumulator
- Each iteration, add the norm of the hidden state delta to the accumulator
- Multiply the residual connection by `1 / (1 + inhibition_gain * accumulator)` 
- `inhibition_gain` is a learnable scalar (initialized small, e.g., 0.1)
- This means early iterations have full residual strength, later iterations are increasingly damped
- Like attractor halt, this makes convergence happen naturally but through a different mechanism (damping vs. detection)

### 4.8 Halt Combination

When multiple halt mechanisms are active, combine their signals:

**'any' mode**: halt if any single mechanism says halt. Simple OR logic.

**'majority' mode**: halt if more than half of the active halt mechanisms say halt. Each mechanism's signal is thresholded at 0.5 to become binary.

**'learned' mode** (recommended): 
- Collect all active halt signals into a vector
- Pass through a tiny linear layer → sigmoid → single halt probability
- This lets the network learn which halt signals to trust and how to weight them
- Learnable params: one small linear layer (num_active_halt_signals → 1)

---

## Component 5: Full Forward Pass

Pseudocode for the complete forward pass:

```python
def forward(self, input_ids):
    # 1. Embed
    h = self.embed(input_ids) + self.pos_embed(positions)
    
    # 2. Initial modulator context
    mod_context = self.mod_input_proj(h.mean(dim=1))  # pool over sequence
    
    # 3. Initialize halt tracking
    cumulative_halt_prob = 0.0
    remainders = []
    hidden_states = []
    energy = self.config.energy_budget
    inhibition_accum = 0.0
    h_prev = h
    num_iterations_used = 0
    
    # 4. Recursive loop
    for i in range(self.config.max_iterations):
        # 4a. Generate modulation
        mod_inputs = [mod_context]
        if self.config.use_iteration_encoding:
            mod_inputs.append(self.iter_embed(i))
        if self.config.use_adaptive_modulation:
            mod_inputs.append(self.mod_hidden_proj(h.mean(dim=1)))
        
        modulation = self.modulator(torch.cat(mod_inputs, dim=-1))
        
        # 4b. Apply synaptic depression to modulation
        if self.config.use_synaptic_depression:
            depression_mult = (1 - self.config.depression_rate) ** i
            modulation['weight_scale'] *= depression_mult
        
        # 4c. Run shared blocks with modulation
        for block_idx, block in enumerate(self.shared_blocks):
            block_mod = extract_block_modulation(modulation, block_idx)
            
            # Apply oscillatory gating
            if self.config.use_oscillatory_gating:
                osc_gate = self.oscillatory_gate(block_idx, i)
                block_mod['residual_scale'] *= osc_gate
            
            h = block(h, modulation=block_mod)
        
        # 4d. Apply inhibitory damping
        if self.config.use_inhibitory_damping:
            delta = (h - h_prev).norm()
            inhibition_accum += delta.item()
            damping = 1.0 / (1.0 + self.inhibition_gain * inhibition_accum)
            h = h_prev + (h - h_prev) * damping  # damp the residual
        
        # 4e. Collect halt signals
        halt_signals = {}
        
        if self.config.use_attractor_halt:
            relative_delta = (h - h_prev).norm() / (h_prev.norm() + 1e-8)
            halt_signals['attractor'] = (relative_delta < self.config.attractor_threshold).float()
        
        if self.config.use_learned_halt:
            halt_signals['learned'] = self.halt_head(h.mean(dim=1)).sigmoid()
        
        if self.config.use_modulator_halt:
            halt_signals['modulator'] = modulation['halt_signal']
        
        if self.config.use_energy_budget:
            cost = self.energy_cost_head(h.mean(dim=1)).sigmoid()
            energy -= cost
            halt_signals['energy'] = (energy <= 0).float()
        
        # 4f. Combine halt signals
        should_halt, halt_prob = self.combine_halts(halt_signals)
        
        # 4g. ACT-style bookkeeping
        hidden_states.append(h)
        cumulative_halt_prob += halt_prob
        num_iterations_used += 1
        
        h_prev = h
        
        if should_halt or cumulative_halt_prob >= 1.0:
            break
    
    # 5. Weighted output (ACT-style)
    # Weight each iteration's hidden state by its halt probability contribution
    # This makes the output differentiable w.r.t. halting decisions
    output_h = weighted_combine(hidden_states, halt_probs)
    
    # 6. Output head
    logits = self.output_head(self.output_norm(output_h))
    
    return logits, num_iterations_used
```

---

## Component 6: Loss Function

```python
def compute_loss(logits, targets, num_iterations, config):
    # Standard language modeling loss
    task_loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    
    # Ponder cost: penalize using more iterations
    ponder_cost = config.iteration_cost * num_iterations
    
    # ACT remainder loss (if using learned halt): encourages crisp halting decisions
    # This is the standard ACT regularizer from Graves 2016
    act_loss = remainder_loss  # accumulated from the halt probability remainders
    
    total_loss = task_loss + ponder_cost + act_loss
    return total_loss, {'task': task_loss.item(), 'ponder': ponder_cost, 'act': act_loss.item()}
```

---

## Component 7: Behavioral Characterization System (NEW)

This is the key component that ensures the evolutionary search discovers genuinely novel architectures rather than converging on local optima.

### 7.1 Why Behavioral Characterization

Two networks with identical accuracy but radically different computational strategies are behaviorally similar from a loss perspective but structurally/dynamically novel. We need to characterize HOW a network solves a task, not just HOW WELL.

### 7.2 Behavioral Feature Vector

For each trained config, run a **diagnostic probe set** (a fixed set of ~500 diverse test inputs) and extract:

```python
@dataclass
class BehavioralProfile:
    # --- Iteration dynamics ---
    mean_iterations: float              # average iterations used
    iteration_variance: float           # variance of iterations across inputs
    iteration_by_difficulty: List[float] # mean iterations per difficulty bucket (easy/med/hard)
    iteration_entropy: float            # entropy of iteration count distribution
    
    # --- Halting behavior ---
    halt_trigger_rates: Dict[str, float]  # per-mechanism: fraction of inputs where this mechanism triggered halt
    halt_timing_profile: List[float]      # distribution of which iteration halt occurs at
    
    # --- Modulation dynamics ---
    modulation_magnitude: float         # mean L2 norm of modulation signals
    modulation_variance: float          # variance of modulation across inputs
    modulation_iteration_drift: float   # how much modulation changes across iterations (mean delta)
    channel_gate_sparsity: float        # fraction of gates < 0.1 (how many channels are "shut off")
    channel_gate_entropy: float         # entropy of gate distribution
    
    # --- Hidden state dynamics ---
    convergence_rate: float             # mean rate at which hidden state deltas decrease
    attractor_count: float              # estimated number of distinct attractors (cluster hidden states)
    hidden_state_rank: float            # effective rank of hidden state covariance matrix
    
    # --- Information flow ---
    residual_stream_norm_profile: List[float]  # L2 norm of residual updates per iteration
    attention_entropy_profile: List[float]     # entropy of attention distributions per iteration
    
    # --- Output behavior ---
    confidence_profile: float           # mean max softmax probability
    output_diversity: float             # number of distinct top-1 predictions across probe set
    
    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single normalized vector for distance computation."""
        ...
```

### 7.3 Novelty Score Computation

```python
def compute_novelty(profile: BehavioralProfile, archive: List[BehavioralProfile], k: int = 15) -> float:
    """
    Novelty = mean distance to k-nearest neighbors in behavioral space.
    Uses the behavioral feature vector, NOT the config genome.
    """
    vec = profile.to_vector()
    archive_vecs = np.array([p.to_vector() for p in archive])
    
    # Normalize each dimension by archive std to prevent scale domination
    stds = archive_vecs.std(axis=0) + 1e-8
    vec_norm = vec / stds
    archive_norm = archive_vecs / stds
    
    distances = np.linalg.norm(archive_norm - vec_norm, axis=1)
    k_nearest = np.sort(distances)[:k]
    return k_nearest.mean()
```

### 7.4 Diagnostic Probe Set Design

Create a fixed diagnostic probe set that tests diverse computational demands:

```python
def generate_diagnostic_probes(vocab_size, seq_len, num_probes=500):
    probes = []
    # Uniform random sequences (baseline)
    probes.extend(random_sequences(100))
    # Deeply nested parentheses (high recursion demand)
    probes.extend(deep_nesting(100, max_depth=20))
    # Shallow parentheses (low recursion demand)
    probes.extend(shallow_nesting(100, max_depth=3))
    # Repetitive patterns (tests whether model detects periodicity)
    probes.extend(repetitive_patterns(100))
    # Adversarial: sequences designed to confuse convergence detection
    probes.extend(adversarial_sequences(100))
    return probes
```

The probe set is generated ONCE and reused across all evaluations for consistency.

---

## Component 8: MAP-Elites Quality-Diversity Archive (NEW)

### 8.1 Feature Space Definition

Define a multi-dimensional feature space describing architectural PROPERTIES. MAP-Elites fills every cell with the best-performing architecture having those properties, forcing exploration into weird corners.

```python
MAP_ELITES_DIMENSIONS = {
    # Dimension 1: Modulation complexity (0 = no modulation, 3 = all modulation on)
    'modulation_complexity': {
        'compute': lambda cfg: sum([
            cfg.use_global_modulation,
            cfg.use_layer_modulation,
            cfg.use_channel_gating,
            cfg.use_adaptive_modulation,
        ]),
        'bins': [0, 1, 2, 3, 4],  # 5 bins
    },
    
    # Dimension 2: Halting complexity (number of active halt mechanisms)
    'halting_complexity': {
        'compute': lambda cfg: sum([
            cfg.use_attractor_halt,
            cfg.use_learned_halt,
            cfg.use_modulator_halt,
            cfg.use_energy_budget,
        ]),
        'bins': [0, 1, 2, 3, 4],  # 5 bins
    },
    
    # Dimension 3: Effective depth (num_shared_blocks * max_iterations)
    'effective_depth': {
        'compute': lambda cfg: cfg.num_shared_blocks * cfg.max_iterations,
        'bins': [0, 6, 12, 18, 24],  # 5 bins
    },
    
    # Dimension 4: Dynamic behavior (measured post-training)
    'iteration_variance': {
        'compute': lambda profile: profile.iteration_variance,
        'bins': [0.0, 0.5, 1.0, 2.0, 5.0],  # 5 bins, computed from behavioral profile
    },
}
# Total cells: 5 * 5 * 5 * 5 = 625 possible niches
```

### 8.2 MAP-Elites Update Logic

```python
class MAPElitesArchive:
    def __init__(self, dimensions):
        self.grid = {}  # (bin_tuple) -> (config, fitness, profile)
        self.dimensions = dimensions
        self.all_profiles = []  # for novelty computation
    
    def add(self, config, fitness, profile):
        cell = self._get_cell(config, profile)
        self.all_profiles.append(profile)
        
        if cell not in self.grid or fitness > self.grid[cell][1]:
            self.grid[cell] = (config, fitness, profile)
            return True  # new niche or improvement
        return False
    
    def coverage(self):
        """Fraction of cells filled — tracks exploration progress."""
        total_cells = 1
        for dim in self.dimensions.values():
            total_cells *= len(dim['bins'])
        return len(self.grid) / total_cells
    
    def sample_parent(self):
        """Sample a parent from the archive, preferring less-explored regions."""
        cells = list(self.grid.keys())
        # Uniform random from filled cells (encourages even exploration)
        cell = random.choice(cells)
        return self.grid[cell][0]  # return config
```

---

## Component 9: Speciation System (NEW)

### 9.1 Why Speciation

Novel architectures are initially bad at the task because their weights haven't been optimized yet. Without protection, they're immediately outcompeted by mature conventional architectures and die off before they have a chance to develop. Speciation (from NEAT) prevents this.

### 9.2 Structural Distance

```python
def structural_distance(cfg1: NeuroModConfig, cfg2: NeuroModConfig) -> float:
    """
    Measure how structurally different two configs are.
    Boolean differences weighted more than continuous differences.
    """
    distance = 0.0
    
    # Boolean mechanism toggles: each mismatch adds 1.0
    bool_params = [
        'use_global_modulation', 'use_layer_modulation', 'use_channel_gating',
        'use_iteration_encoding', 'use_adaptive_modulation',
        'use_attractor_halt', 'use_learned_halt', 'use_modulator_halt',
        'use_synaptic_depression', 'use_oscillatory_gating',
        'use_energy_budget', 'use_inhibitory_damping',
    ]
    for param in bool_params:
        if getattr(cfg1, param) != getattr(cfg2, param):
            distance += 1.0
    
    # Categorical params: mismatch adds 1.0
    if cfg1.halt_combination != cfg2.halt_combination:
        distance += 1.0
    
    # Continuous params: normalized absolute difference, weighted 0.5
    continuous_params = {
        'attractor_threshold': (0.001, 0.1),
        'depression_rate': (0.01, 0.2),
        'energy_budget': (0.5, 2.0),
        'max_iterations': (3, 8),
        'num_shared_blocks': (1, 3),
    }
    for param, (lo, hi) in continuous_params.items():
        v1 = getattr(cfg1, param)
        v2 = getattr(cfg2, param)
        distance += 0.5 * abs(v1 - v2) / (hi - lo + 1e-8)
    
    return distance
```

### 9.3 Species Management

```python
class Species:
    def __init__(self, representative: NeuroModConfig):
        self.representative = representative
        self.members = []
        self.best_fitness = float('-inf')
        self.stagnation_counter = 0
    
class SpeciationManager:
    def __init__(self, threshold=4.0, max_stagnation=10):
        self.species = []
        self.threshold = threshold
        self.max_stagnation = max_stagnation
    
    def assign_species(self, config):
        """Assign a config to a species or create a new one."""
        for sp in self.species:
            if structural_distance(config, sp.representative) < self.threshold:
                sp.members.append(config)
                return sp
        # No match — create new species
        new_sp = Species(representative=config)
        new_sp.members.append(config)
        self.species.append(new_sp)
        return new_sp
    
    def allocate_offspring(self, total_offspring):
        """
        Allocate reproductive budget proportional to species size,
        but with a minimum allocation so small novel species survive.
        """
        # Remove stagnant species (no fitness improvement for max_stagnation generations)
        self.species = [sp for sp in self.species if sp.stagnation_counter < self.max_stagnation]
        
        # Each species gets at least 1 offspring
        min_per_species = 1
        remaining = total_offspring - len(self.species) * min_per_species
        
        # Distribute remaining proportional to mean fitness (adjusted)
        total_fitness = sum(sp.best_fitness for sp in self.species)
        allocations = {}
        for sp in self.species:
            share = remaining * (sp.best_fitness / (total_fitness + 1e-8))
            allocations[sp] = min_per_species + max(0, int(share))
        
        return allocations
```

---

## Component 10: Evolutionary Search Harness (ENHANCED)

This replaces the basic tournament evolution from the original spec with a novelty-aware quality-diversity search.

### 10.1 Config Genome

Represent each configuration as a dict of booleans and floats. The search mutates these.

```python
SEARCHABLE_PARAMS = {
    # Booleans (toggled by bit flip)
    'use_global_modulation': bool,
    'use_layer_modulation': bool,
    'use_channel_gating': bool,
    'use_iteration_encoding': bool,
    'use_adaptive_modulation': bool,
    'use_attractor_halt': bool,
    'use_learned_halt': bool,
    'use_modulator_halt': bool,
    'use_synaptic_depression': bool,
    'use_oscillatory_gating': bool,
    'use_energy_budget': bool,
    'use_inhibitory_damping': bool,
    
    # Categorical
    'halt_combination': ['any', 'majority', 'learned'],
    
    # Continuous (mutated by gaussian noise)
    'attractor_threshold': (0.001, 0.1),    # min, max
    'depression_rate': (0.01, 0.2),
    'energy_budget': (0.5, 2.0),
    'iteration_cost': (0.001, 0.1),
    'mod_dim': [8, 16, 32],                  # discrete choices
    'max_iterations': [3, 4, 6, 8],
    'num_shared_blocks': [1, 2, 3],
}
```

### 10.2 Composite Fitness (Quality + Novelty + Simplicity)

```python
def compute_composite_fitness(
    val_loss: float,
    avg_iterations: float,
    stability: float,
    novelty_score: float,
    config: NeuroModConfig,
    # Weight parameters (tunable)
    w_quality: float = 1.0,
    w_novelty: float = 0.5,
    w_efficiency: float = 0.1,
    w_simplicity: float = 0.05,
) -> float:
    """
    Composite fitness that rewards quality AND novelty.
    
    The novelty weight (w_novelty) is the critical lever:
    - Set to 0.0 for pure performance optimization (will rediscover known architectures)
    - Set to 0.5 for balanced quality-diversity search (recommended starting point)
    - Set to 1.0+ for aggressive novelty seeking (may sacrifice performance for weirdness)
    """
    quality = -val_loss
    efficiency = -0.1 * avg_iterations
    stability_bonus = -0.05 * stability  # lower variance = better
    
    # Simplicity pressure: penalize number of active mechanisms
    active_mechanisms = sum([
        config.use_global_modulation, config.use_layer_modulation,
        config.use_channel_gating, config.use_iteration_encoding,
        config.use_adaptive_modulation, config.use_attractor_halt,
        config.use_learned_halt, config.use_modulator_halt,
        config.use_synaptic_depression, config.use_oscillatory_gating,
        config.use_energy_budget, config.use_inhibitory_damping,
    ])
    simplicity = -active_mechanisms
    
    return (
        w_quality * quality +
        w_novelty * novelty_score +
        w_efficiency * efficiency +
        w_simplicity * simplicity +
        stability_bonus
    )
```

### 10.3 Search Loop (MAP-Elites + Speciation + Novelty)

```python
def run_evolutionary_search(
    population_size: int = 30,
    num_generations: int = 20,
    training_steps_per_eval: int = 2000,
    novelty_k: int = 15,
    novelty_weight: float = 0.5,
):
    archive = MAPElitesArchive(MAP_ELITES_DIMENSIONS)
    speciation_mgr = SpeciationManager(threshold=4.0)
    
    # --- Initialize population ---
    population = []
    population.append(make_all_on_config())       # all mechanisms on
    population.append(make_minimal_config())       # bare recursive transformer
    population.append(make_modulation_only_config()) # modulation, no fancy halting
    population.append(make_halting_only_config())   # halting, no modulation
    # Rest are random
    while len(population) < population_size:
        population.append(make_random_config())
    
    for gen in range(num_generations):
        results = []
        
        for config in population:
            # Train
            model, val_loss, avg_iters, stability = train_and_evaluate(
                config, num_steps=training_steps_per_eval
            )
            
            # Characterize behavior
            profile = compute_behavioral_profile(model, diagnostic_probes)
            
            # Compute novelty relative to archive
            novelty = compute_novelty(profile, archive.all_profiles, k=novelty_k)
            
            # Composite fitness
            fitness = compute_composite_fitness(
                val_loss, avg_iters, stability, novelty, config,
                w_novelty=novelty_weight
            )
            
            # Update MAP-Elites archive
            archive.add(config, fitness, profile)
            
            # Assign to species
            species = speciation_mgr.assign_species(config)
            
            results.append((config, fitness, profile, species))
        
        # --- Selection and reproduction ---
        offspring_allocation = speciation_mgr.allocate_offspring(population_size - 2)  # reserve 2 for elitism
        
        new_population = []
        
        # Elitism: carry top 2 from archive
        top_configs = sorted(archive.grid.values(), key=lambda x: x[1], reverse=True)[:2]
        for cfg, fit, prof in top_configs:
            new_population.append(cfg)
        
        # Per-species reproduction
        for species, num_offspring in offspring_allocation.items():
            members_ranked = sorted(
                [(cfg, fit) for cfg, fit, prof, sp in results if sp == species],
                key=lambda x: x[1], reverse=True
            )
            
            for _ in range(num_offspring):
                if len(members_ranked) >= 2 and random.random() < 0.7:
                    # Crossover within species
                    p1 = tournament_select(members_ranked, k=3)
                    p2 = tournament_select(members_ranked, k=3)
                    child = crossover(p1, p2)
                else:
                    # Mutation only
                    parent = tournament_select(members_ranked, k=3)
                    child = mutate(parent)
                
                child = mutate(child)  # always mutate
                new_population.append(child)
        
        # Also inject some configs from under-explored MAP-Elites regions
        # This adds pressure to fill the archive
        num_archive_samples = max(2, population_size // 10)
        for _ in range(num_archive_samples):
            parent = archive.sample_parent()
            child = mutate(parent)
            # Aggressively mutate to push into new cells
            child = mutate(child)
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # --- Logging ---
        log_generation(gen, archive, speciation_mgr, results)
        
        # --- Adaptive novelty weight ---
        # If archive coverage is stalling, increase novelty pressure
        if gen > 5:
            recent_coverage = [log[gen-i]['coverage'] for i in range(5)]
            if max(recent_coverage) - min(recent_coverage) < 0.02:
                novelty_weight = min(1.5, novelty_weight * 1.2)
                print(f"  Coverage stalling, increasing novelty_weight to {novelty_weight:.2f}")
    
    return archive
```

### 10.4 Mutation

```python
def mutate(config: NeuroModConfig) -> NeuroModConfig:
    """
    Mutate a config genome. Returns a new NeuroModConfig.
    """
    cfg = copy.deepcopy(config)
    
    # Each boolean has 15% chance of flipping
    for param in BOOLEAN_PARAMS:
        if random.random() < 0.15:
            setattr(cfg, param, not getattr(cfg, param))
    
    # Each continuous param has 20% chance of gaussian perturbation (σ = 10% of range)
    for param, (lo, hi) in CONTINUOUS_PARAMS.items():
        if random.random() < 0.20:
            val = getattr(cfg, param)
            noise = random.gauss(0, 0.1 * (hi - lo))
            setattr(cfg, param, max(lo, min(hi, val + noise)))
    
    # Each categorical/discrete param has 10% chance of random reselection
    for param, choices in CATEGORICAL_PARAMS.items():
        if random.random() < 0.10:
            setattr(cfg, param, random.choice(choices))
    
    return cfg
```

### 10.5 Crossover

```python
def crossover(cfg1: NeuroModConfig, cfg2: NeuroModConfig) -> NeuroModConfig:
    """Uniform crossover: each parameter randomly from one parent."""
    child = NeuroModConfig()
    for field in fields(NeuroModConfig):
        if random.random() < 0.5:
            setattr(child, field.name, getattr(cfg1, field.name))
        else:
            setattr(child, field.name, getattr(cfg2, field.name))
    return child
```

---

## Component 11: Training Task

For the initial small-scale experiments, use a **synthetic task** that rewards variable-depth reasoning. Implement at least Option A, ideally also Option B:

### Option A: Nested Parenthesis Matching
- Input: sequences of open/close parens with varying nesting depth
- Target: predict whether the sequence is balanced, or predict the next token
- Why: deeper nesting naturally requires more recursive passes, so the model is incentivized to learn adaptive computation depth

### Option B: Algorithmic Sequence Prediction
- Input: sequences generated by simple algorithms (reverse, sort, copy, repeat)
- Target: predict the continuation
- A task-type token at the start tells the model what algorithm is active
- Why: different algorithms benefit from different "modes" of processing, incentivizing modulation to specialize

### Option C: Multi-Scale Pattern Completion
- Input: sequences with patterns at multiple scales (e.g., local bigram patterns AND global structure)
- Target: next token prediction
- Why: the model needs to detect and combine information at different depths of recursion

**Recommendation**: Start with Option A (simplest to implement, clearest recursion-depth signal). Add Option B once the framework is validated.

### Data Generation
- Generate training data on-the-fly (no dataset files needed)
- Each batch is freshly generated
- Control difficulty distribution: mix of easy (few iterations needed) and hard (many iterations needed) examples

---

## Component 12: Logging and Visualization

Track and log per training step:
- Loss components (task, ponder, ACT)
- Average iterations used per batch
- Per-mechanism halt trigger rates (how often each mechanism fires)
- Modulation statistics: mean/std of global scale, layer scales, channel gates
- Hidden state convergence rate (delta norm per iteration)

For the evolutionary search, log per generation:
- Best/mean/worst fitness (quality component AND novelty component separately)
- Mechanism frequency in population
- Top config's parameters
- **MAP-Elites archive coverage** (fraction of cells filled — this is the key metric for novelty)
- **Species count and sizes** (diversity of structural niches)
- **Novelty score distribution** (are we still finding novel things?)
- **Behavioral diversity metrics** (spread of behavioral profiles in the archive)

Use **tensorboard** or just save to CSV/JSON for plotting.

### Novelty-Specific Visualizations
- **MAP-Elites heatmap**: 2D slices of the archive grid, colored by fitness. Shows which regions of architecture space have been explored.
- **Behavioral embedding**: UMAP/t-SNE of behavioral profile vectors, colored by fitness. Shows whether novel architectures cluster differently.
- **Mechanism survival curves**: per-mechanism frequency across generations. Shows which mechanisms are evolutionarily stable vs which get selected out.
- **Species phylogeny**: track species birth/death/branching across generations.
- **Novelty vs Quality scatter**: plot novelty score vs val_loss for all evaluated configs. The Pareto frontier here shows the best tradeoffs.

---

## File Structure

```
neuromod_recursive/
├── config.py          # NeuroModConfig dataclass
├── model.py           # Full model: backbone, modulator, halt mechanisms, forward pass
├── modules/
│   ├── backbone.py    # SharedTransformerBlock, Embedding, OutputHead
│   ├── modulator.py   # ModulatorNetwork
│   ├── halting.py     # All 7 halt mechanisms + halt combiner
│   └── oscillator.py  # Oscillatory gating module
├── data.py            # Synthetic data generators (parenthesis, algorithmic, pattern)
├── train.py           # Single-config training loop
├── novelty/
│   ├── behavioral.py  # BehavioralProfile computation + diagnostic probes
│   ├── map_elites.py  # MAPElitesArchive
│   ├── speciation.py  # SpeciationManager + structural_distance
│   └── novelty.py     # Novelty score computation
├── search.py          # Evolutionary search harness (MAP-Elites + speciation + novelty)
├── evaluate.py        # Evaluation: test loss, iteration analysis, mechanism analysis
├── visualize.py       # Plotting: training curves, mechanism frequencies, search results, MAP-Elites heatmaps
├── utils.py           # Param counting, config serialization, seeding
└── run_search.py      # Entry point: launch evolutionary search
```

---

## Implementation Order

Follow this order to build incrementally and test each component:

### Phase 1: Minimal Recursive Transformer
1. Implement `config.py` with all fields
2. Implement `backbone.py`: embedding, one shared transformer block, output head
3. Implement `train.py` with basic next-token-prediction loop on a trivial task
4. **Test**: verify the model trains and loss decreases with recursion=1 (should behave like a normal tiny transformer)
5. Enable recursion (max_iterations=4): verify it still trains (loss should be same or better)

### Phase 2: Add Modulation
6. Implement `modulator.py` with global and layer modulation
7. Wire modulation into the shared blocks' forward pass
8. **Test**: train with modulation on vs off. Modulation should not hurt and might help.
9. Add channel gating, iteration encoding, adaptive modulation one at a time, verifying each

### Phase 3: Add Halting Mechanisms
10. Implement attractor convergence halt (simplest, no learnable params)
11. Implement learned halt (ACT-style)
12. Implement modulator-driven halt
13. Implement synaptic depression, oscillatory gating, energy budget, inhibitory damping
14. Implement halt combiner (all three modes: any, majority, learned)
15. **Test**: verify the model learns to use variable iterations per input (log iteration counts)

### Phase 4: Synthetic Tasks
16. Implement parenthesis matching data generator
17. Train with full mechanism suite on parenthesis task
18. Verify that harder inputs (deeper nesting) use more iterations
19. Implement algorithmic sequence task

### Phase 5: Behavioral Characterization (NEW)
20. Implement diagnostic probe set generator
21. Implement `BehavioralProfile` computation — run a trained model on probes, extract all features
22. Implement novelty score computation (k-nearest in behavioral space)
23. **Test**: train 5 manually-different configs, verify that their behavioral profiles are measurably different and the novelty scores reflect this

### Phase 6: MAP-Elites + Speciation (NEW)
24. Implement `MAPElitesArchive` with the 4 feature dimensions
25. Implement `SpeciationManager` with structural distance
26. **Test**: create 20 random configs, verify they get assigned to different species and MAP-Elites cells

### Phase 7: Evolutionary Search (ENHANCED)
27. Implement composite fitness (quality + novelty + simplicity)
28. Implement the full search loop with MAP-Elites archive, speciation-based reproduction, and adaptive novelty weight
29. Run search for 20 generations with population size 30
30. **Monitor**: archive coverage should increase over generations. If it plateaus, the adaptive novelty weight should kick in.

### Phase 8: Analysis
31. Retrain top 5 configs from the MAP-Elites archive for full duration
32. Generate mechanism frequency report
33. Generate co-occurrence/correlation analysis
34. Generate MAP-Elites heatmaps and novelty vs quality Pareto plot
35. Generate behavioral embedding visualization (UMAP)
36. Write a summary of which architectural strategies emerged and which are genuinely novel

---

## Key Implementation Details & Gotchas

### Differentiability of Halting
- Hard halting (if/break) is not differentiable. Use the ACT trick: compute ALL iterations up to max, but weight their contributions by halting probabilities. Then at inference time, you can hard-halt early.
- During training: always run all iterations, weight outputs. During inference: hard-halt when cumulative probability exceeds threshold.

### Modulation Must Start as Identity
- Initialize all scale parameters near 1.0 (not 0), all shift parameters near 0.0
- Initialize channel gates with positive bias so sigmoid outputs ~0.5-0.7
- If modulation starts at extreme values, it will kill gradients

### Weight Sharing with Modulation
- Do NOT modify weight tensors in-place. Instead, compute `effective_weight = base_weight * modulation_scale` in the forward pass. This keeps autograd clean.
- For synaptic depression: apply as a multiplier in the modulation dict, not on the parameter itself

### Parameter Budget
- Target ~1M total params. Rough allocation:
  - Embeddings: vocab_size * hidden_dim = 512 * 128 = ~65K
  - Per shared block: ~4 * hidden_dim^2 = ~65K (attention + FFN). 2 blocks = ~130K
  - Modulator: ~100-150K
  - Halt mechanisms: ~10-20K total
  - Output head: ~65K
  - Total: ~400-500K. You have room to increase hidden_dim to 192 or 256 if needed.
- Track actual param count and print it at initialization

### Evolutionary Search Practicalities
- Each config evaluation (2000 training steps) should take ~1-5 minutes on a single GPU
- Full search (30 population × 20 generations) = 600 evaluations = ~10-50 GPU-hours
- Save checkpoints after each generation so you can resume
- Use different random seeds for data generation across evaluations to avoid overfitting to a specific data sequence
- The behavioral characterization adds ~30s per evaluation (running the probe set). This is worth it.

### Novelty Search Gotchas
- **Normalize behavioral features**: without normalization, high-magnitude features (like mean_iterations) will dominate the distance metric. Normalize by running standard deviation computed from the archive.
- **Archive growth**: the archive of all profiles grows linearly. For 600 evaluations this is fine. For larger runs, subsample the archive for novelty computation (random 500 from archive).
- **Adaptive novelty weight**: if you see the archive coverage plateau early, increase `w_novelty`. If you see lots of novel but terrible configs, decrease it. The adaptive mechanism in the search loop handles this automatically.
- **Speciation threshold**: too low = every config is its own species (no competition). Too high = one mega-species (no protection). Start at 4.0, adjust if species count is consistently < 3 or > 15.

### Seeding and Reproducibility
- Set random seeds for PyTorch, numpy, and Python's random module
- Each evolutionary run should have a master seed
- Log all seeds

---

## Testing Checklist

Use these checks to verify correctness at each phase:

- [ ] Model with 0 modulation and 1 iteration = vanilla tiny transformer. Loss should match.
- [ ] Increasing max_iterations without modulation should not degrade loss (weight sharing is valid).
- [ ] With learned halt, average iterations should be < max_iterations after training (it learned to halt early).
- [ ] On parenthesis task, deeper nesting examples should use more iterations than shallow ones.
- [ ] Synaptic depression alone should cause natural convergence (attractor halt fires earlier with depression on).
- [ ] All mechanisms toggled off = bare recursive model. All mechanisms toggled on = full model. Both should train without errors.
- [ ] Parameter count stays within budget across all configs.
- [ ] Evolutionary search population diversity doesn't collapse to a single config (check mechanism frequency variance).
- [ ] Modulation statistics (logged during training) should show non-trivial learned patterns, not all-ones or all-zeros.
- [ ] **BehavioralProfile vectors for structurally different configs should have measurably different feature values.** (NEW)
- [ ] **MAP-Elites archive coverage should increase over generations, not plateau immediately.** (NEW)
- [ ] **Species count should be > 3 and < 15 for a population of 30.** (NEW)
- [ ] **Novelty scores should be high early (everything is new) and gradually decrease as the archive fills.** (NEW)
- [ ] **The Pareto frontier of novelty vs quality should contain configs that are BOTH good AND weird.** (NEW)

---

## Extensions (After Initial Search)

Once you have results from the evolutionary search, possible next steps:

1. **Scale up**: increase to 10M params, real text data (TinyStories or similar), longer sequences
2. **Multi-task modulation**: train on multiple tasks simultaneously, see if the modulator learns distinct modes per task
3. **Modulator analysis**: visualize what the modulator learns — do different inputs cluster in modulation space?
4. **Iteration depth analysis**: make heatmaps of which tokens/positions trigger more iterations
5. **Ablation studies**: for the top config, systematically remove one mechanism at a time and measure degradation
6. **Transfer**: take a top config trained on synthetic tasks, fine-tune on real language modeling
7. **Evolve lower-level primitives** (ADVANCED): instead of toggling fixed mechanisms, evolve the computational graph itself — compose mathematical primitives (add, multiply, gate, normalize, convolve) into novel layer types using genetic programming. This is where truly alien architectures come from.
8. **Evolve the learning rule** (ADVANCED): co-evolve the weight update rule alongside the architecture. A novel architecture paired with backprop may underperform, but paired with a co-evolved update rule it could excel.
9. **Cross-archive breeding**: maintain multiple MAP-Elites archives with different feature dimensions. Periodically breed configs from different archives to combine orthogonal innovations.
