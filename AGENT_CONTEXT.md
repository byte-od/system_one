# SCache (`scar-cache`): Agent Context & Engineering Specification

> **Target Audience:** Autonomous AI Agents (Antigravity, Claude Code, Cursor, GitHub Copilot, Codex, etc.).
> **Purpose:** Provide complete structural, algorithmic, and mathematical context to allow AI agents to navigate, modify, extend, and test `scar-cache` safely without breaking its core invariants.

---

## 1. Executive Summary & Core Philosophy

`scar-cache` is an independent, pure PyTorch library that implements an **$O(1)$ Constant-Memory Associative Gradient Scar Cache**.

### Why it exists:
Standard deep reinforcement learning, continual learning, and test-time adaptation struggle with a fundamental dilemma:
1. **Episodic replay buffers (ER/SIL)** grow unboundedly ($O(N)$ with episode steps), consuming gigabytes of RAM, blowing up host memory, and suffering from memory fragmentation and L1/L2 cache misses.
2. **Search trees and lookaheads (MCTS)** require expensive rollouts and world model simulations ($15\text{ ms} - 500\text{ ms}$ per step), stalling high-throughput training and real-time robotics.
3. **Catastrophic forgetting** occurs whenever slow network weights $W$ are aggressively updated to memorize rare, high-reward breakthrough transitions.

### The SCache Mechanism:
SCache solves this by replacing unbounded transition stores with a **constant-size matrix buffer (~4.25 KB)** that lives permanently in **L1 Data Cache**.
When a rare breakthrough occurs, the network calculates the gradient of the breakthrough state with respect to its latent representation:
$$\mathbf{g} = -\nabla_{\mathbf{z}} \mathcal{L}$$
This gradient $\mathbf{g}$ is accumulated with outer product Hebbian fast-weights:
$$W_{scar} \leftarrow (1 - \eta) W_{scar} + \eta (\hat{\mathbf{g}} \hat{\mathbf{z}}^T)$$

During inference, activations $\mathbf{z}$ are modulated in $< 3\ \mu\text{s}$:
$$\mathbf{z}_{guided} = \mathbf{z} + \alpha \cdot \text{Gate}(\mathbf{z}, \mathbf{z}_{target}) \cdot (\mathbf{z} W_{scar}^T)$$

---

## 2. Strict Architectural Invariants (DO NOT VIOLATE)

When modifying or extending this repository, any autonomous agent **MUST** uphold these four invariants:

### Invariant 1: Strict $O(1)$ Constant Space
- **Rule:** The memory footprint in bytes (`cache.memory_bytes`) must **NEVER** depend on:
  - The number of steps elapsed ($t$).
  - The number of episodes played ($E$).
  - The number of scars consolidated (`scars_count`).
  - The training batch size.
- **Forbidden:** Storing historical transitions, appending tensors to Python lists, growing deque buffers, or expanding dimensions dynamically over time.

### Invariant 2: Zero Domain Coupling
- **Rule:** The core library (`scar_cache/core.py`, `scar_cache/modules.py`, `scar_cache/projections.py`) must remain **100% domain-agnostic**.
- **Forbidden:** Introducing any game logic (e.g., Rubik's cube, Atari, Mujoco, robotics coordinates), curriculum assumptions, or reward scale constants into the core package. Domain logic belongs strictly in `examples/` or user client code.

### Invariant 3: Immunological Manifold Gating Safety
- **Rule:** In `hybrid` mode, modulation must always be gated by the cosine affinity with the target breakthrough manifold:
  $$\text{Gate}(\mathbf{z}) = \max(0, \cos(\mathbf{z}, \mathbf{z}_{target}))$$
- **Rationale:** If $\cos(\mathbf{z}, \mathbf{z}_{target}) \le 0$, the state is in a completely different region of the state space. Injecting fast-weights here would cause destructive hallucination / interference on unrelated tasks. The gate must evaluate to exactly $0.0$ in non-aligned regimes.

### Invariant 4: Orthogonal Conflict Resolution (PCGrad)
- **Rule:** Gradients that oppose the consensus compass ($\mathbf{g}_{new} \cdot \mathbf{g}_{target} < 0$) must be orthogonally projected before accumulation:
  $$\mathbf{g}_{proj} = \mathbf{g}_{new} - \frac{\mathbf{g}_{new} \cdot \mathbf{g}_{target}}{\|\mathbf{g}_{target}\|^2} \mathbf{g}_{target}$$
- **Rationale:** Prevents distinct valid paths to a goal from destructively canceling each other out.

---

## 3. Component Architecture & Code Map

```
scar_cache/
├── core.py:
│   └── SCache (nn.Module)
│       ├── Buffers:
│       │   ├── W_scar: [slot_dim, slot_dim] (Hebbian associative matrix)
│       │   ├── g_target: [slot_dim] (Consensus gradient compass)
│       │   ├── z_target: [slot_dim] (Centroid of breakthrough manifold)
│       │   ├── scars_count: [long] (Monotonic scar counter)
│       │   └── total_surprise_accum: [float32] (Cumulative temporal surprise)
│       ├── Methods:
│       │   ├── record_scar(slots, grad_slots, surprise, orthogonal_projection)
│       │   ├── modulate(slots) -> (slots_guided, delta, affinity)
│       │   ├── compute_resonance(delta_slots) -> resonance [-1, 1]
│       │   ├── modulate_logits(logits, resonance) -> sharpened_logits
│       │   ├── modulate_scale(scale, resonance) -> sharpened_scale
│       │   ├── reset_scars() -> resets all buffers to 0 for Tabula Rasa
│       │   └── batch_modulation_scope(scale_vec, beta_vec) -> contextmanager
│
├── multi_slot.py:
│   └── MultiSlotSCache (nn.Module) [34 KB L1 Resident]
│       ├── Buffers:
│       │   ├── W_scar: [num_slots, slot_dim, slot_dim] (Tensorized multi-slot memory)
│       │   ├── g_target: [num_slots, slot_dim] (Independent compass per slot)
│       │   ├── z_target: [num_slots, slot_dim] (Independent centroid per slot)
│       │   ├── decay_rates: [num_slots] (Multi-scale geometric decay)
│       │   └── slot_queries: [num_slots, slot_dim] (Competitive routing parameter)
│       ├── Methods:
│       │   ├── route_tokens(x) -> (routing_weights, projected_slots)
│       │   ├── modulate(x, routing_weights) -> (x_guided, deltas, affinities)
│       │   ├── write_key_value(keys, values, slot_index, learning_rate) [Delta Rule]
│       │   ├── read_key(key, slot_index) -> retrieved_value
│       │   └── record_gradient_scar(slots, grad_slots, surprise) [PCGrad per slot]
│
├── modules.py:
│   ├── ScarLinear: nn.Linear drop-in replacement with 2D SCache
│   ├── SlotScarLinear: nn.Linear drop-in replacement with 3D MultiSlotSCache (34 KB)
│   └── ScarGatedLayer: Universal wrapper for any PyTorch nn.Module
│
├── projections.py:
│   ├── project_conflicting_gradient(g_new, g_target) -> g_proj (PCGrad 2D)
│   ├── project_conflicting_gradients_per_slot(g_news, g_targets) -> g_projs (PCGrad 3D)
│   ├── delta_rule_update(W, keys, values, lr) -> (dW, error) [Widrow-Hoff]
│   └── hebbian_outer_product(norm_g, norm_z, num_slots) -> dW
```

---

## 4. Hardware Sizing & Memory Layout

For standard configuration with `slot_dim = 32`, `float32` (4 bytes per scalar):
- $W_{scar} \in \mathbb{R}^{32 \times 32}$: $1024 \times 4\text{ bytes} = 4096\text{ bytes}$
- $g_{target} \in \mathbb{R}^{32}$: $32 \times 4\text{ bytes} = 128\text{ bytes}$
- $z_{target} \in \mathbb{R}^{32}$: $32 \times 4\text{ bytes} = 128\text{ bytes}$
- **Total Working Footprint:** Exactly **$4352\text{ bytes}$ ($4.25\text{ KB}$)**.

### CPU/GPU Cache Hierarchy Context:
- **L1 Data Cache (per core):**
  - Apple Silicon M-series P-core: $128\text{ KB}$
  - Intel Core / AMD Zen 4: $32\text{ KB} - 64\text{ KB}$
  - NVIDIA H100 / A100 SM: $192\text{ KB} - 256\text{ KB}$ shared L1/shared memory
- **Implication:** The entire memory buffer of `SCache` fits completely within the **L1 Data Cache**, guaranteeing sub-microsecond access latency with zero DRAM page faults.

---

## 5. Potential Future Extensions for AI Agents

If an agent is tasked with extending `scar-cache`, here are high-impact architectural directions:

1. **Multi-Head Fast-Weights (`num_heads > 1`):**
   - Partition the `slot_dim` into $H$ heads of dimension $D_h = D / H$.
   - Maintain independent $W_{scar}^{(h)}$ per head for multi-modal breakthrough representations.
2. **Transformer Key-Value Scar Injection:**
   - Attach `SCache` directly into the $K$ and $V$ projections of Multi-Head Self-Attention in Transformer backbones.
3. **Quantized SCache (FP8 / INT8):**
   - Store $W_{scar}$ in `torch.int8` or `torch.float8_e4m3fn` to reduce footprint from $4.25\text{ KB}$ to **$1.1\text{ KB}$** for ultra-constrained embedded systems and MCU microcontrollers.
4. **Decoupled Read/Write Slots:**
   - Allow cross-attention read keys $\mathbf{k}$ to query fast-weight values $\mathbf{v}$ when `num_slots > 1`.

---

## 6. Development & Verification Guide

### Running Unit Tests:
Ensure you run tests with `pytest`:
```bash
pytest tests/ -v
```
All tests must pass with 100% success before committing changes.

### Running Examples:
Verify that all canonical examples execute without error:
```bash
python examples/01_minimal_dropin.py
python examples/02_reinforcement_learning.py
python examples/03_test_time_adaptation.py
```
