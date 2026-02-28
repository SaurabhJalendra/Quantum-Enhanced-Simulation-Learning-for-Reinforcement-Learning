# Quantum-Enhanced Simulation Learning for Reinforcement Learning

**A Comparative Analysis of World Model Training Approaches**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![BITS Pilani](https://img.shields.io/badge/Institution-BITS%20Pilani-orange.svg)](https://www.bits-pilani.ac.in/)

---

## Overview

M.Tech dissertation (BITS Pilani, WILP) investigating whether **quantum-inspired algorithms** can improve world model training in reinforcement learning. All methods run on classical hardware (GPU) -- no quantum computers involved.

**Research Question:** Do quantum-inspired algorithmic approaches improve world model training efficiency compared to classical methods, and under what conditions?

**Answer:** One method (Interference Ensemble) achieves 36-47% improvement on state-based robotics tasks, while three others fail. Success is conditional on input dimensionality.

---

## Key Results

| Method | Quantum Principle | Outcome |
|--------|------------------|---------|
| **Interference Ensemble** | Wave interference | 36-47% improvement on DMControl (p<0.008) |
| Quantum Tunneling | Barrier crossing | No significant effect |
| Superposition Buffer | State superposition | Catastrophic failure (-158% to -630%) |
| Entanglement Layer | Correlated pairs | No significant effect |

**200 experiments** (5 methods x 5 seeds x 8 environments), Mann-Whitney U tests with Bonferroni correction.

### Interference Ensemble by Domain

| Domain | Environments | Effect |
|--------|-------------|--------|
| State-based robotics | Walker, Cheetah, Reacher-easy, Reacher-hard | +36% to +47% (significant) |
| Simple control | CartPole, Pendulum | No significant effect |
| Visual/Atari | Pong, Breakout | -132% to -414% (5x CNN encoder is wasteful) |

---

## Architecture

All methods share a DreamerV3-style **RSSM world model** (~4.7M parameters):

```
Observation -> Encoder (MLP/CNN) -> GRU (deter=512) -> Prior/Posterior (stoch=64)
                                                    -> Decoder -> Reconstructed obs
                                                    -> Reward predictor
                                                    -> Continue predictor
```

The only difference between approaches is the quantum-inspired enhancement applied.

---

## Repository Structure

```
.
├── CLAUDE.md                          # AI assistant context (completed project)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── src/                               # Source code
│   ├── quantum_inspired/              # 4 quantum-inspired modules
│   │   ├── tunneling_optimizer.py     # Periodic noise for escaping local minima
│   │   ├── superposition_buffer.py    # Phase-weighted experience blending
│   │   ├── entanglement_layer.py      # Pairwise feature correlation layer
│   │   └── interference_ensemble.py   # 5-model phase-weighted ensemble
│   ├── config/shared_config.py        # Shared hyperparameters
│   ├── models/                        # Model definitions (init only)
│   ├── training/                      # Training utilities (init only)
│   ├── evaluation/                    # Evaluation utilities (init only)
│   ├── environments/                  # Environment wrappers (init only)
│   └── utils/                         # Utilities (init only)
│
├── experiments/                       # Experiment scripts & results
│   ├── scripts/                       # Standalone experiment runners
│   │   ├── run_phase1_cartpole.py     # CartPole: all 5 methods
│   │   ├── run_phase2_walker.py       # Walker-walk experiments
│   │   ├── run_phase2_cheetah.py      # Cheetah-run experiments
│   │   ├── run_phase2_reacher.py      # Reacher-easy experiments
│   │   ├── run_phase3_pong.py         # Pong experiments (CNN encoder)
│   │   └── run_phase3_breakout.py     # Breakout experiments (CNN encoder)
│   ├── results/                       # Raw JSON results (per-seed + aggregated)
│   │   ├── phase1/{cartpole,pendulum}/
│   │   ├── phase2/{walker,cheetah,reacher,reacher_hard}/
│   │   └── phase3/{pong,breakout}/
│   └── figures/                       # Generated comparison plots
│
├── phase1_cartpole_notebooks/         # Phase 1: Simple control
│   ├── 01_environment_setup.ipynb     # Environment validation
│   ├── 02_classical_baseline.ipynb    # Baseline implementation
│   ├── 03b_quantum_tunneling.ipynb    # Tunneling optimizer
│   ├── 04b_superposition_proper.ipynb # Superposition buffer
│   ├── 05b_entanglement_layers.ipynb  # Entanglement layer
│   ├── 06d_interference_ensemble.ipynb# Interference ensemble
│   ├── 07_comprehensive_comparison.ipynb
│   ├── 08_ablation_studies.ipynb
│   ├── 09_results_analysis.ipynb
│   ├── 10_pendulum_experiments.ipynb  # Pendulum: all 5 methods
│   └── legacy/                        # Superseded notebooks (03-06c)
│
├── phase2_dmcontrol_notebooks/        # Phase 2: DMControl robotics
│   ├── 02_walker_experiments.ipynb
│   ├── 03_cheetah_experiments.ipynb
│   ├── 04_reacher_experiments.ipynb
│   ├── 05_dmcontrol_comparison.ipynb
│   └── 06_reacher_hard_experiments.ipynb
│
├── phase3_atari_notebooks/            # Phase 3: Visual RL
│   ├── 02_pong_experiments.ipynb
│   ├── 03_breakout_experiments.ipynb
│   └── 04_atari_comparison.ipynb
│
├── scripts/                           # Utility scripts
│   ├── aggregate_cartpole_results.py
│   ├── generate_diagrams.py
│   └── validate_consistency.py
│
├── Reports/                           # Dissertation documents
│   ├── FINAL_DISSERTATION_2023AC05912.md  # Full dissertation (~1342 lines)
│   ├── VIVA_PREPARATION_GUIDE.md
│   └── EXAMINER_BRIEFING.md
│
├── docs/                              # Supporting documentation
│   └── *.md                           # Discussion, limitations, tables, etc.
│
└── results/figures/                   # Publication-quality figures
```

---

## Approach Evolution

The project originally proposed 6 approaches from the literature but evolved during implementation:

| Original Plan | Final Implementation | Why Changed |
|---------------|---------------------|-------------|
| QAOA-Enhanced | **Quantum Tunneling** | Full QAOA was over-engineered; tunneling captures the key "escape local minima" principle |
| Gate-Enhanced | **Entanglement Layer** | Full gate circuits were computationally expensive; entanglement captures pairwise correlation |
| Error Correction | **Interference Ensemble** | Syndrome detection was binary; phase-weighted averaging is more nuanced |
| Fully Integrated | **Dropped** | Individual methods showed insufficient benefit to warrant integration |
| Superposition | **Superposition Buffer** | Kept as designed |
| Baseline | **Classical Baseline** | Kept as designed |

---

## Configuration

All experiments use identical architecture for fair comparison:

```python
# Architecture (RSSM)
stoch_dim = 64, deter_dim = 512, hidden_dim = 512
encoder_hidden = [512, 512], decoder_hidden = [512, 512]

# Training
batch_size = 32 (16 for Atari), seq_len = 20, steps = 10000
learning_rate = 3e-4, kl_weight = 1.0

# Seeds: [42, 123, 456, 789, 1024]
```

---

## Environments

| Tier | Environment | Obs Dim | Action Dim |
|------|-------------|---------|------------|
| Simple | CartPole-v1 | 4 | 2 (discrete) |
| Simple | Pendulum-v1 | 3 | 1 (continuous) |
| DMControl | Walker-walk | 24 | 6 |
| DMControl | Cheetah-run | 17 | 6 |
| DMControl | Reacher-easy | 6 | 2 |
| DMControl | Reacher-hard | 6 | 2 |
| Atari | Pong | 84x84x1 | 6 (discrete) |
| Atari | Breakout | 84x84x1 | 4 (discrete) |

---

## Hardware

- **CPU:** AMD Ryzen 9 9950X3D
- **GPU:** NVIDIA RTX 5090
- **RAM:** 32GB
- **Framework:** PyTorch 2.10.0+cu128

---

## Known Limitations

1. Baseline had LR scheduling + weight decay that quantum variants lacked (makes IE result *stronger*)
2. IE reward prediction was not properly trained (code omission in `compute_ensemble_loss`)
3. Only 5 seeds per config (adequate for observed effect sizes, Cohen's d = 2.4-3.2)
4. IE costs 5x compute due to 5-model ensemble

---

## Citation

```bibtex
@mastersthesis{jalendra2026quantum,
  title={Quantum-Enhanced Simulation Learning for Reinforcement Learning:
         A Comparative Analysis of World Model Training Approaches},
  author={Jalendra, Saurabh},
  year={2026},
  school={Birla Institute of Technology and Science, Pilani},
  type={M.Tech Dissertation},
  note={Supervised by Gaurav Kumar, IN-SPACe}
}
```

---

## Contact

**Student:** Saurabh Jalendra (2023AC05912)
**Email:** 2023ac05912@wilp.bits-pilani.ac.in
**Program:** MTech AI/ML, BITS Pilani (WILP)

**Supervisor:** Gaurav Kumar, Deputy Director, IN-SPACe, Ahmedabad

---

Copyright 2025-2026 Saurabh Jalendra, BITS Pilani. All rights reserved.
