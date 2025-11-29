# Quantum-Enhanced Simulation Learning for Reinforcement Learning

<div align="center">

**A Comparative Analysis of World Model Training Approaches**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![BITS Pilani](https://img.shields.io/badge/Institution-BITS%20Pilani-orange.svg)](https://www.bits-pilani.ac.in/)

</div>

---

## 📚 Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Approaches Implemented](#approaches-implemented)
- [Environments](#environments)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiments](#experiments)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🎯 Overview

This repository contains the complete implementation and analysis for the M.Tech dissertation titled **"Quantum-Enhanced Simulation Learning for Reinforcement Learning: A Comparative Analysis of World Model Training Approaches"**.

### What is This Research About?

**World models** are neural networks that learn to predict how environments behave. They allow reinforcement learning (RL) agents to learn through "imagination" rather than expensive real-world trial-and-error. However, training these models is computationally expensive and time-consuming.

This dissertation investigates whether **quantum-inspired algorithms**—classical algorithms inspired by quantum computing principles—can make world model training more efficient, faster, and more robust.

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                     REINFORCEMENT LEARNING                   │
│                                                              │
│  Agent ←→ Environment                                        │
│    ↑                                                         │
│    │ learns from                                             │
│    ↓                                                         │
│  World Model (Neural Network)                                │
│    - Predicts next states                                    │
│    - Predicts rewards                                        │
│    - Enables "imagination"                                   │
│                                                              │
│  PROBLEM: Training is slow and expensive                     │
│                                                              │
│  OUR SOLUTION: Quantum-inspired training methods             │
│    ✓ QAOA-enhanced optimization                              │
│    ✓ Superposition-based replay                              │
│    ✓ Quantum gate transformations                            │
│    ✓ Error correction ensembles                              │
│                                                              │
│  GOAL: Faster, more efficient world model training           │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovation

**No quantum computers needed!** All implementations run on standard CPUs/GPUs using quantum-*inspired* algorithms—we borrow ideas from quantum computing and adapt them for classical hardware.

---

## ❓ Research Questions

This dissertation seeks to answer:

1. **Do quantum-inspired methods improve world model training efficiency?**
2. **Which quantum principles (QAOA, superposition, gates, error correction) work best?**
3. **Under what conditions do these methods provide advantages?**
4. **What are the computational trade-offs?**
5. **Can quantum-inspired methods improve robustness in stochastic environments?**
6. **How do learned representations differ from classical methods?**

---

## ✨ Key Features

### 🔬 Research Features
- ✅ **Systematic Comparison**: 6 different training approaches
- ✅ **Multiple Environments**: DMControl Suite, Atari, Simple Control
- ✅ **Statistical Rigor**: Multiple seeds, significance testing, effect sizes
- ✅ **Reproducible**: Complete code, configs, and random seeds provided
- ✅ **Practical Focus**: Classical hardware only, no quantum computers needed

### 💻 Implementation Features
- ✅ **Modular Architecture**: Easy to extend and modify
- ✅ **Well-Documented**: Comprehensive docstrings and comments
- ✅ **Configurable**: YAML-based configuration system
- ✅ **Logging**: TensorBoard integration for training visualization
- ✅ **Testing**: Unit tests for all major components

### 📊 Analysis Features
- ✅ **Comprehensive Metrics**: Sample efficiency, training speed, accuracy, robustness
- ✅ **Statistical Analysis**: Hypothesis testing, confidence intervals
- ✅ **Visualizations**: Learning curves, latent space plots, comparison charts
- ✅ **Ablation Studies**: Component-wise analysis

---

## 🏗️ Architecture

### High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        TRAINING SYSTEM                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   RL ENV     │      │  WORLD MODEL │      │   TRAINER    │ │
│  │              │─────→│              │←─────│              │ │
│  │ - DMControl  │      │ - Encoder    │      │ - Classical  │ │
│  │ - Atari      │      │ - Dynamics   │      │ - QAOA       │ │
│  │ - CartPole   │      │ - Reward     │      │ - Quantum-   │ │
│  │ - Pendulum   │      │ - Decoder    │      │   Inspired   │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         ↓                      ↓                      ↓         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    EXPERIENCE BUFFER                      │ │
│  │  - Standard Replay                                        │ │
│  │  - Superposition-Enhanced Replay                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                               ↓                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    METRICS & LOGGING                      │ │
│  │  - TensorBoard  - Weights & Biases  - CSV Logs           │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### World Model Architecture (DreamerV3-Style)

```
Input Observation (o_t)
        ↓
┌───────────────────┐
│     ENCODER       │  → Converts observations to latent representations
│  (CNN/MLP)        │     z_t = encoder(o_t)
└───────────────────┘
        ↓
┌───────────────────┐
│  RECURRENT MODEL  │  → Maintains hidden state over time
│  (GRU/LSTM)       │     h_t = f(h_{t-1}, z_t, a_{t-1})
└───────────────────┘
        ↓
        ├─────────────────┬─────────────────┬─────────────────┐
        ↓                 ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  DYNAMICS    │  │   REWARD     │  │   CONTINUE   │  │   DECODER    │
│  PREDICTOR   │  │  PREDICTOR   │  │  PREDICTOR   │  │              │
│              │  │              │  │              │  │              │
│ ẑ_{t+1}      │  │ r̂_t          │  │ ĉ_t          │  │ ô_t          │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

### Quantum-Inspired Enhancements

Each approach modifies different parts of the training process:

| Approach | Modification Point | Key Component |
|----------|-------------------|---------------|
| **QAOA-Enhanced** | Optimization loop | Alternating operators |
| **Superposition** | Experience replay | Weighted combinations |
| **Gate-Enhanced** | Neural network layers | Hadamard/CNOT-inspired ops |
| **Error Correction** | Ensemble predictions | Syndrome detection |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dissertation-quantum-world-models.git
cd dissertation-quantum-world-models
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n quantum-wm python=3.8
conda activate quantum-wm
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install in development mode (recommended for contributors)
pip install -e .
```

### Step 4: Install RL Environments

```bash
# DMControl Suite
pip install dm_control

# Atari environments
pip install "gymnasium[atari,accept-rom-license]"

# Verify installation
python -c "import gymnasium; import dm_control; print('✓ Environments installed successfully')"
```

### Step 5: (Optional) Install Additional Tools

```bash
# For experiment tracking
pip install wandb
wandb login

# For advanced visualization
pip install plotly
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Run quick demo
python examples/demo_baseline.py
```

---

## 🚀 Quick Start

### 1. Train Classical Baseline

```bash
# Train on CartPole (simple environment for testing)
python experiments/scripts/train_baseline.py \
    --env CartPole-v1 \
    --steps 100000 \
    --seed 42

# Train on DMControl Walker
python experiments/scripts/train_baseline.py \
    --env dm_control:Walker-walk \
    --steps 1000000 \
    --seed 42
```

### 2. Train Quantum-Inspired Approaches

```bash
# QAOA-Enhanced
python experiments/scripts/train_qaoa.py \
    --env dm_control:Walker-walk \
    --p_layers 3 \
    --steps 1000000

# Superposition-Enhanced
python experiments/scripts/train_superposition.py \
    --env dm_control:Walker-walk \
    --parallel_samples 8 \
    --steps 1000000

# Gate-Enhanced
python experiments/scripts/train_gates.py \
    --env dm_control:Walker-walk \
    --gate_layers 4 \
    --steps 1000000

# Error Correction Ensemble
python experiments/scripts/train_error_correction.py \
    --env dm_control:Walker-walk \
    --num_models 3 \
    --steps 1000000
```

### 3. Run All Experiments

```bash
# Run comprehensive comparison with multiple seeds
python experiments/scripts/run_all_experiments.py \
    --config experiments/configs/full_comparison.yaml \
    --num_seeds 5
```

### 4. Evaluate Trained Models

```bash
# Evaluate single model
python src/evaluation/evaluate.py \
    --checkpoint results/checkpoints/baseline_walker_seed42.pt \
    --env dm_control:Walker-walk \
    --episodes 100

# Compare all approaches
python src/evaluation/compare_all.py \
    --results_dir results/ \
    --output results/comparison_report.html
```

### 5. Visualize Results

```bash
# Launch TensorBoard
tensorboard --logdir experiments/logs/

# Generate comparison plots
python analysis/scripts/generate_figures.py \
    --results_dir results/ \
    --output dissertation/figures/

# Interactive analysis notebook
jupyter notebook analysis/notebooks/results_analysis.ipynb
```

---

## 📂 Project Structure

```
dissertation-quantum-world-models/
│
├── README.md                          # This file
├── CLAUDE.md                          # AI assistant context & memory
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── .gitignore                         # Git ignore rules
├── LICENSE                            # License information
│
├── docs/                              # 📚 Documentation
│   ├── abstract-report.pdf            # Approved dissertation abstract
│   ├── literature-review.md           # Literature survey
│   ├── methodology.md                 # Detailed methodology
│   ├── theoretical-background.md      # Quantum computing background
│   ├── implementation-guide.md        # Implementation details
│   ├── experimental-protocol.md       # Experiment procedures
│   ├── results-summary.md             # Results overview
│   ├── progress-log.md                # Weekly progress updates
│   └── api-reference.md               # Code API documentation
│
├── src/                               # 💻 Source Code
│   ├── __init__.py
│   │
│   ├── models/                        # World model implementations
│   │   ├── __init__.py
│   │   ├── base_world_model.py        # Abstract base class
│   │   ├── baseline.py                # Classical DreamerV3-style
│   │   ├── qaoa_enhanced.py           # QAOA-inspired optimization
│   │   ├── superposition.py           # Superposition-based replay
│   │   ├── gate_enhanced.py           # Quantum gate transformations
│   │   ├── error_correction.py        # Error correction ensemble
│   │   ├── integrated.py              # Fully integrated approach
│   │   └── components/                # Shared components
│   │       ├── encoder.py             # Observation encoder
│   │       ├── dynamics.py            # Dynamics predictor
│   │       ├── reward.py              # Reward predictor
│   │       ├── decoder.py             # Reconstruction decoder
│   │       └── rssm.py                # Recurrent state-space model
│   │
│   ├── training/                      # Training procedures
│   │   ├── __init__.py
│   │   ├── base_trainer.py            # Abstract trainer class
│   │   ├── classical_trainer.py       # Standard training loop
│   │   ├── qaoa_trainer.py            # QAOA training loop
│   │   ├── quantum_inspired_trainer.py # Quantum-inspired training
│   │   ├── loss_functions.py          # Loss computations
│   │   ├── optimizers.py              # Custom optimizers
│   │   └── schedulers.py              # Learning rate schedules
│   │
│   ├── replay/                        # Experience replay buffers
│   │   ├── __init__.py
│   │   ├── standard_buffer.py         # Standard replay buffer
│   │   ├── superposition_buffer.py    # Quantum-inspired replay
│   │   ├── prioritized_buffer.py      # Prioritized experience replay
│   │   └── utils.py                   # Buffer utilities
│   │
│   ├── environments/                  # Environment wrappers
│   │   ├── __init__.py
│   │   ├── dm_control_wrapper.py      # DMControl environments
│   │   ├── atari_wrapper.py           # Atari environments
│   │   ├── gymnasium_wrapper.py       # Gymnasium environments
│   │   └── preprocessing.py           # Observation preprocessing
│   │
│   ├── evaluation/                    # Evaluation code
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Performance metrics
│   │   ├── evaluator.py               # Model evaluation
│   │   ├── statistical_tests.py       # Statistical analysis
│   │   ├── visualizations.py          # Plotting functions
│   │   └── comparison.py              # Multi-model comparison
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── logging_utils.py           # Logging helpers
│       ├── reproducibility.py         # Seed setting, determinism
│       ├── checkpointing.py           # Model save/load
│       └── data_processing.py         # Data utilities
│
├── experiments/                       # 🧪 Experiments
│   ├── configs/                       # Configuration files
│   │   ├── baseline/
│   │   │   ├── cartpole.yaml
│   │   │   ├── walker.yaml
│   │   │   └── atari_pong.yaml
│   │   ├── qaoa/
│   │   ├── superposition/
│   │   ├── gates/
│   │   ├── error_correction/
│   │   └── full_comparison.yaml       # All methods comparison
│   │
│   ├── scripts/                       # Experiment scripts
│   │   ├── train_baseline.py          # Train classical baseline
│   │   ├── train_qaoa.py              # Train QAOA-enhanced
│   │   ├── train_superposition.py     # Train superposition
│   │   ├── train_gates.py             # Train gate-enhanced
│   │   ├── train_error_correction.py  # Train error correction
│   │   ├── train_integrated.py        # Train integrated
│   │   ├── run_all_experiments.py     # Run comprehensive suite
│   │   ├── ablation_study.py          # Ablation experiments
│   │   └── hyperparameter_sweep.py    # Hyperparameter tuning
│   │
│   └── logs/                          # Training logs (gitignored)
│       └── .gitkeep
│
├── analysis/                          # 📊 Analysis
│   ├── notebooks/                     # Jupyter notebooks
│   │   ├── 01_exploratory_analysis.ipynb
│   │   ├── 02_learning_curves.ipynb
│   │   ├── 03_statistical_tests.ipynb
│   │   ├── 04_latent_space_viz.ipynb
│   │   ├── 05_ablation_analysis.ipynb
│   │   └── 06_final_comparison.ipynb
│   │
│   └── scripts/                       # Analysis scripts
│       ├── compute_statistics.py      # Statistical computations
│       ├── generate_figures.py        # Create all figures
│       ├── ablation_analysis.py       # Ablation study analysis
│       ├── latent_analysis.py         # Latent space analysis
│       └── export_tables.py           # Generate result tables
│
├── results/                           # 📈 Results (gitignored)
│   ├── checkpoints/                   # Saved model checkpoints
│   ├── data/                          # Raw experimental data
│   ├── figures/                       # Generated plots
│   ├── tables/                        # Result tables
│   └── summary.md                     # Results summary
│
├── dissertation/                      # 📝 Dissertation Document
│   ├── chapters/                      # Chapter source files
│   │   ├── 01_introduction.md
│   │   ├── 02_literature_review.md
│   │   ├── 03_background.md
│   │   ├── 04_methodology.md
│   │   ├── 05_implementation.md
│   │   ├── 06_experiments.md
│   │   ├── 07_results.md
│   │   ├── 08_discussion.md
│   │   ├── 09_conclusion.md
│   │   └── 10_appendices.md
│   │
│   ├── figures/                       # Dissertation figures
│   ├── tables/                        # Dissertation tables
│   ├── references.bib                 # Bibliography
│   ├── main.tex                       # LaTeX main file
│   └── compiled/                      # Compiled PDFs
│       └── dissertation.pdf           # Final dissertation
│
├── tests/                             # ✅ Unit Tests
│   ├── __init__.py
│   ├── test_models.py                 # Model tests
│   ├── test_training.py               # Training tests
│   ├── test_replay.py                 # Replay buffer tests
│   ├── test_evaluation.py             # Evaluation tests
│   ├── test_environments.py           # Environment tests
│   └── test_utils.py                  # Utility tests
│
├── examples/                          # 📖 Examples
│   ├── demo_baseline.py               # Simple baseline demo
│   ├── demo_qaoa.py                   # QAOA approach demo
│   ├── demo_visualization.py          # Visualization demo
│   └── notebooks/
│       └── getting_started.ipynb      # Getting started guide
│
└── scripts/                           # 🛠️ Utility Scripts
    ├── setup_environment.sh           # Environment setup
    ├── download_datasets.sh           # Download pre-trained models
    ├── run_tests.sh                   # Run all tests
    └── generate_documentation.sh      # Generate API docs
```

---

## 🎯 Approaches Implemented

### 1. Classical Baseline

**File:** `src/models/baseline.py`

Standard DreamerV3-style world model training with:
- Recurrent state-space model (RSSM)
- KL-divergence regularization
- Reconstruction loss
- Reward prediction loss
- Continue prediction

**Training:**
```bash
python experiments/scripts/train_baseline.py --env dm_control:Walker-walk
```

**Key Hyperparameters:**
- Learning rate: 3e-4
- Batch size: 50
- Sequence length: 50
- KL weight: 1.0

---

### 2. QAOA-Enhanced Approach

**File:** `src/models/qaoa_enhanced.py`

Quantum Approximate Optimization Algorithm-inspired training:

**Algorithm:**
```python
for p in range(p_layers):
    # Cost layer (problem-specific)
    loss = compute_loss(model, batch)
    grads = compute_gradients(loss)
    
    # Mixing layer (exploration)
    noise = generate_exploration_noise(beta_params)
    perturbed_params = params + noise
    
    # Update
    params = optimizer.step(perturbed_params, grads)
```

**Key Components:**
- Alternating cost and mixing operators
- Parameter-dependent mixing angles
- Adaptive layer depth

**Training:**
```bash
python experiments/scripts/train_qaoa.py \
    --env dm_control:Walker-walk \
    --p_layers 3 \
    --mixing_strength 0.1
```

**Key Hyperparameters:**
- p_layers: 2-4
- mixing_strength: 0.05-0.2
- beta_init: 0.1

---

### 3. Superposition-Enhanced Approach

**File:** `src/models/superposition.py`

Quantum superposition-inspired experience replay:

**Algorithm:**
```python
# Sample multiple trajectories in parallel
trajectories = []
for _ in range(num_parallel):
    traj = replay_buffer.sample()
    weight = compute_quantum_weight(traj)
    trajectories.append((traj, weight))

# Create superposed batch (weighted combination)
batch = weighted_combination(trajectories)

# Train on superposed batch
loss = model.compute_loss(batch)
```

**Key Components:**
- Parallel trajectory sampling
- Quantum-inspired weighting scheme
- Interference-like aggregation

**Training:**
```bash
python experiments/scripts/train_superposition.py \
    --env dm_control:Walker-walk \
    --parallel_samples 8 \
    --weight_decay 0.99
```

**Key Hyperparameters:**
- parallel_samples: 4-16
- weight_decay: 0.95-0.99
- aggregation: 'weighted_mean'

---

### 4. Gate-Enhanced Approach

**File:** `src/models/gate_enhanced.py`

Quantum gate-inspired neural network transformations:

**Architecture:**
```python
class QuantumInspiredLayer(nn.Module):
    def forward(self, x):
        # Hadamard-inspired: uniform mixing
        x = self.hadamard_transform(x)
        
        # CNOT-inspired: feature correlation
        x = self.controlled_transform(x, control_features)
        
        # Phase gate-inspired: learned rotation
        x = x * torch.exp(1j * self.phase_angles)
        
        # Measurement-inspired: projection
        x = self.measurement_projection(x.real)
        
        return x
```

**Key Components:**
- Hadamard-like transformations
- CNOT-like conditional operations
- Phase rotation layers
- Measurement projection

**Training:**
```bash
python experiments/scripts/train_gates.py \
    --env dm_control:Walker-walk \
    --gate_layers 4 \
    --phase_init uniform
```

**Key Hyperparameters:**
- gate_layers: 2-6
- hidden_dim: 256-512
- phase_init: 'uniform', 'normal', 'zeros'

---

### 5. Error Correction Ensemble

**File:** `src/models/error_correction.py`

Quantum error correction-inspired ensemble:

**Algorithm:**
```python
# Multiple redundant predictors
predictors = [WorldModel() for _ in range(num_models)]

# Forward pass through all models
predictions = [m.predict(state, action) for m in predictors]

# Syndrome detection (identify disagreements)
syndrome = detect_disagreement(predictions)

# Error correction (voting/weighting)
if syndrome.any():
    corrected = majority_voting(predictions, syndrome)
else:
    corrected = mean(predictions)
```

**Key Components:**
- Ensemble of 3-5 world models
- Syndrome measurement
- Majority voting / weighted correction
- Redundant encoding

**Training:**
```bash
python experiments/scripts/train_error_correction.py \
    --env dm_control:Walker-walk \
    --num_models 3 \
    --syndrome_threshold 0.1
```

**Key Hyperparameters:**
- num_models: 3-5
- syndrome_threshold: 0.05-0.2
- correction_method: 'majority', 'weighted'

---

### 6. Fully Integrated Approach (Optional)

**File:** `src/models/integrated.py`

Combines multiple quantum-inspired techniques:
- QAOA optimization
- Superposition replay
- Gate-enhanced encoder
- Error correction ensemble

**Training:**
```bash
python experiments/scripts/train_integrated.py \
    --env dm_control:Walker-walk \
    --enable_qaoa \
    --enable_superposition \
    --enable_gates \
    --enable_error_correction
```

---

## 🌍 Environments

### Simple Control (Baseline Validation)

| Environment | Observation | Action | Episodes |
|-------------|-------------|--------|----------|
| CartPole-v1 | 4D | Discrete (2) | 500 |
| Pendulum-v1 | 3D | Continuous (1D) | 500 |

**Purpose:** Quick validation and debugging

### DMControl Suite (Primary Benchmark)

| Environment | Observation | Action | Episode Length |
|-------------|-------------|--------|----------------|
| Walker-walk | 24D | Continuous (6D) | 1000 |
| Cheetah-run | 17D | Continuous (6D) | 1000 |
| Reacher-easy | 6D | Continuous (2D) | 1000 |
| Reacher-hard | 6D | Continuous (2D) | 1000 |

**Purpose:** Complex continuous control, primary evaluation

### Atari (Visual Complexity)

| Environment | Observation | Action | Max Steps |
|-------------|-------------|--------|-----------|
| Pong | 84×84×4 | Discrete (6) | 108,000 |
| Breakout | 84×84×4 | Discrete (4) | 108,000 |

**Purpose:** Visual representation learning, high-dimensional observations

---

## 📏 Evaluation Metrics

### Primary Metrics

#### 1. Sample Efficiency
**Definition:** Number of environment steps to reach target performance

**Computation:**
```python
def compute_sample_efficiency(rewards, target_reward=threshold):
    """Returns number of steps to reach target reward"""
    for step, reward in enumerate(rewards):
        if reward >= target_reward:
            return step
    return None  # Target not reached
```

**Target Thresholds:**
- CartPole: 475
- Walker: 800
- Cheetah: 800
- Pong: 18

#### 2. Training Speed
**Definition:** Wall-clock time to convergence

**Measured:** Time from start to reaching 95% of final performance

#### 3. Prediction Accuracy
**Definition:** Mean Squared Error on held-out test trajectories

**Computation:**
```python
mse = torch.mean((predicted_states - true_states) ** 2)
```

#### 4. Final Performance
**Definition:** Average return over 100 test episodes

### Secondary Metrics

#### 5. Training Stability
**Definition:** Standard deviation across random seeds

**Lower is better** - indicates consistent training

#### 6. Robustness
**Definition:** Performance degradation under noise

**Test:** Add Gaussian noise to observations, measure performance drop

#### 7. Computational Cost
**Metrics:**
- FLOPs per training step
- Memory usage (peak)
- Training time per epoch

### Statistical Analysis

**Methods:**
- Mann-Whitney U test (non-parametric comparison)
- Cohen's d (effect size)
- 95% confidence intervals
- Bonferroni correction (multiple comparisons)

---

## 🧪 Experiments

### Experiment 1: Baseline Performance

**Goal:** Establish classical baseline performance

**Environments:** All (CartPole, Walker, Cheetah, Reacher, Pong, Breakout)

**Seeds:** 5 per environment

**Config:** `experiments/configs/baseline/`

**Run:**
```bash
python experiments/scripts/run_all_experiments.py \
    --experiment baseline \
    --num_seeds 5
```

---

### Experiment 2: QAOA Comparison

**Goal:** Compare QAOA-enhanced vs baseline

**Variables:**
- p_layers: [1, 2, 3, 4]
- mixing_strength: [0.05, 0.1, 0.15, 0.2]

**Environments:** Walker, Cheetah

**Seeds:** 5 per configuration

**Run:**
```bash
python experiments/scripts/ablation_study.py \
    --approach qaoa \
    --ablate p_layers,mixing_strength
```

---

### Experiment 3: Superposition Comparison

**Goal:** Evaluate superposition-enhanced replay

**Variables:**
- parallel_samples: [2, 4, 8, 16]
- weight_decay: [0.95, 0.97, 0.99]

**Environments:** Walker, Cheetah

**Seeds:** 5 per configuration

**Run:**
```bash
python experiments/scripts/ablation_study.py \
    --approach superposition \
    --ablate parallel_samples,weight_decay
```

---

### Experiment 4: Gate Transformations

**Goal:** Test quantum gate-inspired layers

**Variables:**
- gate_layers: [2, 3, 4, 5, 6]
- hidden_dim: [256, 512]

**Environments:** Walker, Pong

**Seeds:** 5 per configuration

---

### Experiment 5: Error Correction

**Goal:** Evaluate ensemble error correction

**Variables:**
- num_models: [3, 4, 5]
- syndrome_threshold: [0.05, 0.1, 0.15, 0.2]

**Environments:** Walker (with added noise)

**Seeds:** 5 per configuration

---

### Experiment 6: Full Comparison

**Goal:** Systematic comparison of all approaches

**Approaches:** All 6 methods

**Environments:** All environments

**Seeds:** 10 per approach per environment

**Config:** `experiments/configs/full_comparison.yaml`

**Run:**
```bash
python experiments/scripts/run_all_experiments.py \
    --experiment full_comparison \
    --num_seeds 10
```

**Estimated Time:** ~2 weeks on single GPU

---

### Experiment 7: Robustness Testing

**Goal:** Test robustness to environmental noise

**Procedure:**
1. Train all models on clean environments
2. Evaluate with Gaussian noise: σ ∈ [0.0, 0.05, 0.1, 0.15, 0.2]
3. Measure performance degradation

**Hypothesis:** Error correction approach should be most robust

---

### Experiment 8: Generalization

**Goal:** Test generalization to unseen environments

**Procedure:**
1. Train on: Walker, Cheetah
2. Test on: Humanoid, Quadruped (zero-shot)
3. Measure transfer performance

---

## 📊 Results

Results will be populated as experiments complete.

### Preliminary Results

*(To be updated)*

### Learning Curves

*(Figures to be added)*

### Statistical Comparisons

*(Tables to be added)*

### Key Findings

*(To be documented)*

---

## 📚 Documentation

### Available Documentation

- **[Abstract Report](docs/abstract-report.pdf)** - Approved dissertation abstract
- **[Literature Review](docs/literature-review.md)** - Comprehensive survey of related work
- **[Methodology](docs/methodology.md)** - Detailed research methodology
- **[Theoretical Background](docs/theoretical-background.md)** - Quantum computing primer
- **[Implementation Guide](docs/implementation-guide.md)** - Code walkthrough
- **[Experimental Protocol](docs/experimental-protocol.md)** - How experiments are run
- **[API Reference](docs/api-reference.md)** - Code documentation
- **[Progress Log](docs/progress-log.md)** - Weekly updates
- **[CLAUDE.md](CLAUDE.md)** - AI assistant context

### Generating Documentation

```bash
# Generate API documentation
python scripts/generate_documentation.sh

# Build HTML docs
cd docs && make html
```

---

## 🤝 Contributing

This is a dissertation project, but feedback and suggestions are welcome!

### Reporting Issues

Found a bug? Please open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)

### Code Style

This project follows:
- PEP 8 style guide
- Type hints for all functions
- Docstrings in NumPy format

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value
    """
    pass
```

---

## 📖 Citation

If you use this code or findings in your research, please cite:

```bibtex
@mastersthesis{jalendra2026quantum,
  title={Quantum-Enhanced Simulation Learning for Reinforcement Learning: A Comparative Analysis of World Model Training Approaches},
  author={Jalendra, Saurabh},
  year={2026},
  school={Birla Institute of Technology and Science, Pilani},
  type={M.Tech Dissertation},
  note={Supervised by Gaurav Kumar}
}
```

---

## 📧 Contact

**Student:** Saurabh Jalendra  
**Email:** 2023ac05912@wilp.bits-pilani.ac.in  
**BITS ID:** 2023AC05912  
**Institution:** BITS Pilani (WILP Division)

**Supervisor:** Gaurav Kumar  
**Position:** Deputy Director, PMA Directorate, IN-SPACe  
**Email:** gaurav.kumar45@inspace.gov.in

**Additional Examiner:** Rishabh Swami  
**Organization:** Orange Business Services India  
**Email:** rishabh.swami@orange.com

---

## 🙏 Acknowledgments

This research would not have been possible without:

- **BITS Pilani** - For the M.Tech program and research support
- **IN-SPACe** - For supervision and guidance (Gaurav Kumar)
- **S K Jalendra Marketing Services Pvt Ltd** - For organizational support
- **PyTorch Team** - For the deep learning framework
- **DeepMind** - For the DreamerV3 architecture and DMControl Suite
- **OpenAI** - For Gymnasium environments
- **Open-source community** - For countless tools and libraries

### Key References

This work builds upon:
- Hafner et al. (2023) - DreamerV3
- Farhi et al. (2014) - QAOA
- Wei et al. (2022) - Quantum-inspired RL
- Sutton & Barto (2018) - RL foundations

---

## 📄 License

**Copyright © 2025 Saurabh Jalendra, BITS Pilani**

This dissertation and associated code are proprietary to:
- Birla Institute of Technology and Science, Pilani
- S K Jalendra Marketing Services Pvt Ltd

**All rights reserved.**

This work is submitted in partial fulfillment of the requirements for the degree of Master of Technology in Artificial Intelligence & Machine Learning.

**For permissions regarding use of this code or findings, please contact the author.**

---

## 📅 Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Literature Review & Baseline | 30 Oct - 20 Nov 2025 | 🟡 In Progress |
| Quantum-Inspired Development | 21 Nov - 24 Dec 2025 | ⚪ Pending |
| Experimental Evaluation | 25 Dec - 14 Jan 2026 | ⚪ Pending |
| Analysis & Documentation | 15 Jan - 22 Jan 2026 | ⚪ Pending |
| Review & Revision | 23 Jan - 28 Jan 2026 | ⚪ Pending |
| Final Submission | 29 Jan - 31 Jan 2026 | ⚪ Pending |

**Last Updated:** November 2025

---

## 🔗 Links

- **Institution:** [BITS Pilani](https://www.bits-pilani.ac.in/)
- **Program:** [WILP Division](https://www.bits-pilani.ac.in/wilp/)
- **Supervisor Organization:** [IN-SPACe](https://www.inspace.gov.in/)
- **PyTorch:** [https://pytorch.org/](https://pytorch.org/)
- **DMControl:** [https://github.com/deepmind/dm_control](https://github.com/deepmind/dm_control)
- **Gymnasium:** [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

---

<div align="center">

**🚀 Built with quantum-inspired algorithms on classical hardware 🚀**

**Made with ❤️ by Saurabh Jalendra**

[Report Issue](https://github.com/yourusername/dissertation-quantum-world-models/issues) • [Request Feature](https://github.com/yourusername/dissertation-quantum-world-models/issues) • [Dissertation Progress](docs/progress-log.md)

</div>
