# CLAUDE.md - Project Context

**Last Updated:** February 28, 2026 | **Status:** COMPLETE — Viva defense preparation phase

---

## Project Summary

**Title:** Quantum-Enhanced Simulation Learning for Reinforcement Learning: A Comparative Analysis of World Model Training Approaches

**One-Sentence Summary:** We tested whether quantum-inspired algorithms (running on classical GPUs, NOT quantum computers) can improve world model training in reinforcement learning — finding that interference ensemble averaging improves state-based prediction by 36-47%, while three other quantum-inspired methods fail.

**Student:** Saurabh Jalendra (2023AC05912) | MTech AI/ML, BITS Pilani (WILP)
**Supervisor:** Gaurav Kumar, Deputy Director, IN-SPACe, Ahmedabad
**Examiner:** Rishabh Swami, Orange Business Services India
**Organization:** S K Jalendra Marketing Services Pvt Ltd

---

## Research Question

**"Do quantum-inspired algorithmic approaches improve world model training efficiency compared to classical methods, and under what conditions?"**

**Answer (from our research):** YES, but only under specific conditions. Phase-weighted ensemble averaging (Interference Ensemble) improves prediction accuracy by 36-47% on state-based continuous control tasks. Three other quantum principles (tunneling, superposition, entanglement) produce negligible or harmful effects. The benefits are domain-specific — they fail on visual/pixel-based tasks.

---

## Core Concepts

### World Models
Neural networks that predict how environments behave. Given a state and action, they predict the next state and reward. This lets RL agents "imagine" outcomes before acting — called model-based reinforcement learning. DreamerV3 (Hafner et al., 2023) is the state-of-the-art.

### Quantum-Inspired (NOT Quantum Computing)
We borrow mathematical principles from quantum physics (tunneling, superposition, entanglement, interference) and implement them as classical algorithms on standard GPUs. No quantum hardware, no qubits, no quantum simulators. Like using airplane aerodynamics to design a better car.

---

## The 5 Final Approaches

### Approach Evolution

The project originally proposed 6 approaches. Through evidence-driven iterative refinement, these evolved into 5 final methods. Each change preserved the quantum principle while finding a better classical implementation.

| # | Originally Proposed | Final Implementation | Why Changed |
|---|---|---|---|
| 1 | Classical Baseline | **Classical Baseline** | No change (control group) |
| 2 | QAOA-Enhanced | **Quantum Tunneling** | QAOA's alternating operators caused loss explosions; tunneling is simpler and stable |
| 3 | Superposition Replay | **Superposition Replay** | Kept, reimplemented properly (notebook 04 → 04b) |
| 4 | Gate-Enhanced Layers | **Entanglement Layer** | Quantum gate operations meaningless on continuous activations; entanglement correlations are natural for NNs |
| 5 | Error Correction Ensemble | **Interference Ensemble** | Majority voting had no effect (p=1.0); interference weighting gave principled confidence mechanism |
| 6 | Fully Integrated | **Dropped** | All components combined = 19,870% worse; destructive interference between components |

Legacy experiments for all original approaches are preserved in notebooks (03, 04, 05, 06, 06b, 06c) and `experiments/results/`.

### Final 5 Methods

#### 1. Classical Baseline (Control)
- Standard DreamerV3-style RSSM training with AdamW optimizer
- The reference against which all quantum methods are compared

#### 2. Quantum Tunneling (Optimizer Level)
- **Quantum principle:** Particles tunneling through energy barriers to escape local minima
- **Implementation:** Wraps AdamW, adds periodic adaptive noise every 100 steps (strength=0.001, annealing=0.9999)
- **Result:** Negligible effect everywhere
- **Source:** `src/quantum_inspired/tunneling_optimizer.py` (283 lines)

#### 3. Superposition Replay (Data Level)
- **Quantum principle:** Multiple states existing simultaneously
- **Implementation:** Samples 3 experiences per batch slot, blends observations with phase weights, keeps primary action
- **Result:** Catastrophic failure everywhere (-158% to -630%)
- **Root cause:** Phase range limited to ±π/4; blended observations create invalid "ghost states"
- **Source:** `src/quantum_inspired/superposition_buffer.py` (432 lines)

#### 4. Entanglement Layer (Architecture Level)
- **Quantum principle:** Correlated particles — measuring one determines the other
- **Implementation:** Learnable 512×512 correlation matrix with multiplicative feature interactions, inserted into encoder
- **Result:** Negligible effect everywhere
- **Root cause:** Softmax over 262,144 entries produces near-uniform weights (~3.8×10⁻⁶)
- **Source:** `src/quantum_inspired/entanglement_layer.py` (309 lines)

#### 5. Interference Ensemble (Prediction Level)
- **Quantum principle:** Wave interference — aligned phases amplify, misaligned cancel
- **Implementation:** 5 independent RSSM models with phase-weighted aggregation based on uncertainty
- **Result:** 36-47% improvement on state-based DMControl; 132-414% worse on visual Atari
- **Source:** `src/quantum_inspired/interference_ensemble.py` (454 lines)
- **Known issue:** `compute_ensemble_loss()` omits reward prediction loss

---

## Architecture

All methods share the identical RSSM (Recurrent State-Space Model) backbone:

```
Encoder: obs → [512, ELU, 512, ELU, 512] → embed(512)
GRU: input_proj(stoch+action → 512) → GRUCell(512, 512) → deter(512)
Prior: deter → [512, ELU, 128] → mean(64) + std(64)
Posterior: concat(deter, embed) → [512, ELU, 128] → mean(64) + std(64)
Decoder: state(576) → [512, ELU, 512, ELU] → obs
Reward: state(576) → [512, ELU, 512, ELU] → scalar
Continue: state(576) → [512, ELU, 512, ELU] → scalar
```

### Standard Configuration

```python
stoch_dim = 64           # Stochastic state dimension
deter_dim = 512          # Deterministic state dimension (GRU hidden)
hidden_dim = 512         # All hidden layers
state_dim = 576          # deter + stoch (used for decoding)
batch_size = 32          # (16 for Atari)
seq_len = 20             # Sequence length
num_steps = 10000        # Training steps
learning_rate = 3e-4     # AdamW
kl_weight = 1.0          # KL divergence weight
seeds = [42, 123, 456, 789, 1024]  # 5 required seeds
```

### Parameter Counts
- Single model (state-based): ~4.7M
- +Entanglement layer: ~5.3M (+526K)
- Ensemble 5× (state-based): ~23.7M
- Ensemble 5× (Atari CNN): ~103M

---

## Experimental Results

### 200 experiments: 5 methods × 5 seeds × 8 environments

### Key Results (Test Observation MSE)

| Environment | Baseline | QT | SP | EN | IE | IE vs BL |
|---|---|---|---|---|---|---|
| **CartPole** | 0.109±0.017 | 0.111±0.018 | 0.164±0.030 | 0.112±0.020 | 0.126±0.038 | -15% (NS) |
| **Pendulum** | 0.027±0.003 | 0.026±0.004 | 0.140±0.020 | 0.031±0.008 | 0.031±0.002 | -13% (NS) |
| **Walker** | 1.799±0.060 | 1.798±0.030 | 4.645±0.198 | 1.798±0.032 | **1.022±0.013** | **+43.2%** (p=0.008) |
| **Cheetah** | 0.573±0.009 | 0.578±0.005 | 2.858±0.062 | 0.575±0.007 | **0.367±0.007** | **+35.9%** (p=0.008) |
| **Reacher-easy** | 0.125±0.005 | 0.134±0.006 | 0.915±0.015 | 0.130±0.006 | **0.069±0.004** | **+45.0%** (p=0.008) |
| **Reacher-hard** | 0.127±0.004 | 0.128±0.004 | 0.904±0.021 | 0.129±0.004 | **0.068±0.003** | **+46.7%** (p=0.008) |
| **Pong** | 2.93e-4 | 2.86e-4 | 2.88e-4 | 3.01e-4 | 6.81e-4 | -132% (p=0.008) |
| **Breakout** | 5.39e-4 | 5.39e-4 | 5.31e-4 | 5.57e-4 | 2.77e-3 | -414% (p=0.008) |

**Summary:** 4 significant improvements (IE on DMControl), 8 significant degradations (SP everywhere + IE Atari), 20 non-significant.

### Statistical Methods
- Mann-Whitney U test (non-parametric, n=5)
- Bonferroni correction: α = 0.05/4 = 0.0125
- Cohen's d: 2.42 to 3.21 for significant IE improvements (very large)
- Min achievable p-value with n=5: 0.00794

---

## 8 Test Environments

| Tier | Environment | Obs Dim | Action Dim | Type |
|------|-------------|---------|------------|------|
| Simple | CartPole-v1 | 4 | 2 | Discrete |
| Simple | Pendulum-v1 | 3 | 1 | Continuous |
| DMControl | Walker-walk | 24 | 6 | Continuous |
| DMControl | Cheetah-run | 17 | 6 | Continuous |
| DMControl | Reacher-easy | 6 | 2 | Continuous |
| DMControl | Reacher-hard | 6 | 2 | Continuous |
| Atari | Pong (ALE/Pong-v5) | 84×84×1 | 6 | Discrete |
| Atari | Breakout (ALE/Breakout-v5) | 84×84×1 | 4 | Discrete |

---

## Known Limitations

1. **Training pipeline asymmetry:** Baseline has LR scheduling, weight decay, free_nats, continue_loss, orthogonal init, AMP — quantum notebooks lack these. Makes IE result conservative (IE won despite disadvantages).
2. **IE reward MSE untrained:** `compute_ensemble_loss()` omits reward prediction loss.
3. **IE Atari partial metrics:** Only test_obs_mse available (103M params too large for full eval).
4. **Entanglement softmax issue:** Softmax over 262K entries → near-uniform weights. Per-row softmax would fix this.
5. **n=5 seeds:** Adequate for large effects detected but limited power for subtle differences.

---

## Repository Structure

```
CLAUDE.md                          # This file
README.md                          # Project overview
setup.py                           # Package installation

src/
  config/shared_config.py          # Single source of truth for all params (315 lines)
  quantum_inspired/
    tunneling_optimizer.py         # Quantum Tunneling (283 lines)
    superposition_buffer.py        # Superposition Replay (432 lines)
    entanglement_layer.py          # Entanglement Layer (309 lines)
    interference_ensemble.py       # Interference Ensemble (454 lines)
  utils/__init__.py                # Utilities (set_seed, MetricLogger, COLORS)
  models/                          # Empty — models defined in notebooks
  training/                        # Empty — training loops in notebooks
  evaluation/                      # Empty — evaluation in notebooks

phase1_cartpole_notebooks/         # 17 notebooks (CartPole + Pendulum)
  02_classical_baseline.ipynb      # Active: Baseline
  03b_quantum_tunneling.ipynb      # Active: QT
  04b_superposition_proper.ipynb   # Active: SP
  05b_entanglement_layers.ipynb    # Active: EN
  06d_interference_ensemble.ipynb  # Active: IE
  10_pendulum_experiments.ipynb    # Active: All methods on Pendulum
  03_qaoa_enhanced.ipynb           # Legacy: QAOA (superseded by 03b)
  05_gate_enhanced_layers.ipynb    # Legacy: Gates (superseded by 05b)
  06_error_correction.ipynb        # Legacy: Error Correction (superseded by 06d)
  06b_fully_integrated.ipynb       # Legacy: Fully Integrated (dropped)
  06c_selective_integration.ipynb  # Legacy: Selective Integration (dropped)

phase2_dmcontrol_notebooks/        # 7 notebooks (Walker, Cheetah, Reacher)
phase3_atari_notebooks/            # 4 notebooks (Pong, Breakout)

experiments/
  scripts/                         # 9 experiment runner scripts
    run_phase1_cartpole.py         # CartPole all 5 methods × 5 seeds
    run_phase2_walker.py           # Walker experiments
    run_phase2_cheetah.py          # Cheetah experiments
    run_phase2_reacher.py          # Reacher-easy experiments
    run_phase3_pong.py             # Pong experiments (CNN-based)
    run_phase3_breakout.py         # Breakout experiments (CNN-based)
  results/                         # 222 JSON + 14 CSV + 18 PT checkpoints
    phase1/cartpole/               # 25 seed JSONs + complete_metrics.json
    phase1/pendulum/               # 25 seed JSONs + complete_metrics.json
    phase2/{walker,cheetah,reacher,reacher_hard}/
    phase3/{pong,breakout}/

scripts/                           # Utility scripts
  generate_diagrams.py             # Generates all 8 architecture/results figures
  validate_consistency.py          # Checks notebook param consistency
  aggregate_cartpole_results.py    # Aggregates seed JSONs into complete_metrics

results/figures/                   # 20 PNG figures for dissertation
Reports/
  FINAL_DISSERTATION_2023AC05912.md  # Complete dissertation (1,342 lines)
  VIVA_PREPARATION_GUIDE.md          # Viva prep (1,600+ lines)
  EXAMINER_BRIEFING.md               # 2-page examiner briefing (not in git)

docs/                              # Moved dissertation drafts + reference docs
```

---

## API Contracts

```python
# BaseWorldModel.forward() returns 2-tuple
predictions, states_dict = model(obs_seq, action_seq)
# predictions: (batch, seq_len, obs_dim)
# states_dict keys: 'deter', 'stoch', 'priors', 'posteriors'

# Episodes are dicts
episode = {'obs': array, 'actions': array, 'rewards': array, 'dones': array}

# ReplayBuffer.sample() returns numpy tuple
obs, actions, rewards, dones = buffer.sample(batch_size, seq_len)

# SuperpositionReplayBuffer.sample() returns dict with torch tensors
batch = sp_buffer.sample(batch_size, seq_len)
# keys: 'obs', 'actions', 'rewards', 'indices', 'weights'

# InterferenceEnsemble wraps BaseWorldModel
ensemble = InterferenceEnsemble(model_class=BaseWorldModel, num_models=5, ...)
combined_pred, states_dict = ensemble(obs_seq, action_seq)
```

---

## Technical Stack

```
Python 3.12 | PyTorch 2.10.0+cu128 | Gymnasium 1.2.2
DMControl Suite | ALE (Atari) | NumPy | Pandas | Matplotlib
Hardware: NVIDIA RTX 5090, AMD Ryzen 9 9950X3D, 32GB RAM
Kernel: quantum-rl-venv
```

---

## Key Literature

1. Hafner et al. (2023) — DreamerV3
2. Ha & Schmidhuber (2018) — World Models
3. Schrittwieser et al. (2020) — MuZero
4. Farhi et al. (2014) — QAOA
5. Zhou et al. (2020) — QAOA Performance
6. Wei et al. (2022) — Quantum-Inspired Experience Replay
7. Dong et al. (2012) — Quantum-Inspired Robot Navigation
8. Wu et al. (2025) — RLVR-World
9. Georgiev et al. (2024) — PWM
10. Sutton & Barto (2018) — RL: An Introduction

---

## Key Dates

| Milestone | Date |
|-----------|------|
| Project start | October 30, 2025 |
| All 200 experiments complete | February 2, 2026 |
| Report finalized | February 16, 2026 |
| Viva defense | March 2026 (scheduled) |
