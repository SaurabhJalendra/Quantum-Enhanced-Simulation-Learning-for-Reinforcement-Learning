# QUANTUM-ENHANCED SIMULATION LEARNING FOR REINFORCEMENT LEARNING: A COMPARATIVE ANALYSIS OF WORLD MODEL TRAINING APPROACHES

**AIMLCZG628T: Dissertation**

by

**Saurabh Jalendra**
**2023AC05912**

Dissertation work carried out at
**S K Jalendra Marketing Services Pvt Ltd**

Submitted in partial fulfilment of Degree Program: **MTech AI/ML**

Under the Supervision of
**Gaurav Kumar**
Deputy Director, IN-SPACe, PMA Directorate, Ahmedabad

---

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE PILANI (RAJASTHAN)**

**(February 2026)**

---

## ACKNOWLEDGEMENTS

I express my sincere gratitude to my supervisor, **Gaurav Kumar**, Deputy Director at IN-SPACe, for his invaluable guidance, constructive feedback, and continuous support throughout this dissertation. His expertise in space technology and AI systems provided crucial insights that shaped this research.

I am thankful to **BITS Pilani** and the WILP Division for providing the academic framework and resources necessary for this work. Special thanks to my examiner, **Rishabh Swami** from Orange Business Services India, for his evaluation and feedback.

I acknowledge **S K Jalendra Marketing Services Pvt Ltd** for providing the computational infrastructure, including access to high-performance GPU systems (NVIDIA RTX 5090) that made the extensive experimental evaluation possible.

Finally, I thank my family for their unwavering support and encouragement throughout this academic journey.

**Saurabh Jalendra**
(2023AC05912)

---

## ABSTRACT

This dissertation presents the first systematic evaluation of quantum-inspired algorithmic approaches for world model training in reinforcement learning. World models—neural networks that learn to predict environment dynamics—are fundamental to model-based reinforcement learning but suffer from slow training, local minima entrapment, and error accumulation over long prediction horizons. We investigate whether mathematical concepts borrowed from quantum computing can address these challenges when adapted to classical neural network training.

We implemented and evaluated four quantum-inspired methods against a classical baseline: (1) Quantum Tunneling optimization for escaping local minima, (2) Superposition-based experience replay for improved sample efficiency, (3) Entanglement-inspired feature layers for richer representations, and (4) Interference Ensemble for uncertainty-weighted prediction aggregation. All methods use identical RSSM (Recurrent State-Space Model) architectures following the DreamerV3 design, ensuring fair comparison.

**Key Findings:**
- **Interference Ensemble achieves statistically significant improvements of 36-47%** on continuous control tasks (Walker-walk: +43.2%, Cheetah-run: +35.9%, Reacher-easy: +45.0%, Reacher-hard: +46.7%; all p < 0.008, the minimum achievable p-value for n=5 per group with Mann-Whitney U)
- **Superposition Replay fails catastrophically** on complex dynamics (-158% to -630% degradation)
- **Critical domain-specific finding:** Interference Ensemble excels on state-based tasks but performs significantly worse on visual tasks (Pong: -132%, Breakout: -414%)
- Quantum Tunneling and Entanglement layers show marginal, non-significant effects

Experiments spanned 8 environments across three tiers: simple control (CartPole, Pendulum), DMControl Suite (Walker-walk, Cheetah-run, Reacher-easy, Reacher-hard), and Atari (Pong, Breakout), with 5 random seeds each, totaling approximately 200 experimental runs. Statistical analysis employed Mann-Whitney U tests with Bonferroni correction (α = 0.05/4 = 0.0125 per environment).

The primary contribution is identifying that quantum-inspired ensemble methods provide substantial benefits for low-dimensional state-based world models but are unsuitable for high-dimensional visual representations—a finding with direct practical implications for practitioners selecting training approaches.

**Repository:** https://github.com/SaurabhJalendra/Quantum-Enhanced-Simulation-Learning-for-Reinforcement-Learning

---

| | |
|---|---|
| **Signature of the Student** | **Signature of the Supervisor** |
| Name: Saurabh Jalendra | Name: Gaurav Kumar |
| Date: February 2026 | Date: February 2026 |
| Place: Jaipur | Place: Ahmedabad |

---

## CONTENTS

1. [Introduction](#1-introduction)
   - 1.1 Context & Problem Statement
   - 1.2 Objectives
   - 1.3 Literature Review
   - 1.4 Limitations of Existing Approaches
   - 1.5 Proposed Solution
   - 1.6 Project Significance & Contributions

2. [System Architecture](#2-system-architecture)
   - 2.1 RSSM World Model Architecture
   - 2.2 Quantum-Inspired Components
   - 2.3 Experimental Framework

3. [Design Considerations](#3-design-considerations)
   - 3.1 Architectural Decisions
   - 3.2 Fair Comparison Strategy
   - 3.3 Statistical Methodology

4. [Dataset & Model Architecture](#4-dataset--model-architecture)
   - 4.1 Environment Specifications
   - 4.2 Data Collection Protocol
   - 4.3 Model Specifications

5. [Implementation Details](#5-implementation-details)
   - 5.1 Quantum Tunneling Optimizer
   - 5.2 Superposition Replay Buffer
   - 5.3 Entanglement Layers
   - 5.4 Interference Ensemble
   - 5.5 Training Pipeline

6. [Performance Evaluation & Results](#6-performance-evaluation--results)
   - 6.1 Experimental Setup
   - 6.2 Phase 1: Simple Control (CartPole, Pendulum)
   - 6.3 Phase 2: DMControl Suite (Walker, Cheetah, Reacher-easy, Reacher-hard)
   - 6.4 Phase 3: Atari (Visual RL)
   - 6.5 Statistical Analysis
   - 6.6 Domain-Specific Findings

7. [Discussion](#7-discussion)
   - 7.1 Why Interference Ensemble Works on State-Based Tasks
   - 7.2 Why Superposition Fails on Complex Dynamics
   - 7.3 Domain Specificity Analysis
   - 7.4 Long-Horizon Prediction Behavior
   - 7.5 Computational Cost Analysis

8. [Conclusion & Future Directions](#8-conclusion--future-directions)
   - 8.1 Summary of Contributions
   - 8.2 Practical Recommendations
   - 8.3 Limitations
   - 8.4 Future Work

9. [References](#9-references)

10. [Appendix: Implementation Code](#10-appendix-implementation-code)

11. [Glossary](#11-glossary)

---

## LIST OF FIGURES

| Figure | Description |
|--------|-------------|
| Figure 2.1 | RSSM World Model Architecture |
| Figure 2.2 | Quantum Tunneling Optimizer Visualization |
| Figure 2.3 | Superposition Replay Buffer (with Warning) |
| Figure 2.4 | Entanglement-Inspired Feature Layer |
| Figure 2.5 | Interference Ensemble Architecture |
| Figure 2.6 | Experimental Pipeline Overview |
| Figure 6.1 | Learning Curves Comparison |
| Figure 6.2 | Box Plot Comparison Across Methods |
| Figure 6.3 | Effect Size Analysis |
| Figure 6.4 | Domain-Specific Performance (Key Finding) |
| Figure 6.5 | Results Summary Across All Environments |
| Figure 6.6 | Long-Horizon Prediction Accuracy |
| Figure 6.7 | Ablation Studies |

---

## LIST OF TABLES

| Table | Description |
|-------|-------------|
| Table 1.1 | Mapping Problems to Quantum-Inspired Solutions |
| Table 4.1 | Environment Specifications |
| Table 4.2 | Model Architecture Parameters |
| Table 4.3 | Training Configuration |
| Table 6.1 | Phase 1 Results: CartPole-v1 |
| Table 6.2 | Phase 1 Results: Pendulum-v1 |
| Table 6.3 | Phase 2 Results: Walker-walk |
| Table 6.4 | Phase 2 Results: Cheetah-run |
| Table 6.5 | Phase 2 Results: Reacher-easy |
| Table 6.6 | Phase 2 Results: Reacher-hard |
| Table 6.7 | Phase 3 Results: Pong |
| Table 6.8 | Phase 3 Results: Breakout |
| Table 6.9 | Statistical Significance Summary |
| Table 6.10 | Long-Horizon Prediction Results |
| Table 6.11 | Generalization Gap Analysis |
| Table 6.12 | Computational Cost Comparison |
| Table 7.1 | Research Question Answers |

---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|--------------|-----------|
| AI | Artificial Intelligence |
| CNN | Convolutional Neural Network |
| CNOT | Controlled-NOT Gate |
| DMControl | DeepMind Control Suite |
| GPU | Graphics Processing Unit |
| GRU | Gated Recurrent Unit |
| IE | Interference Ensemble |
| KL | Kullback-Leibler Divergence |
| ML | Machine Learning |
| MSE | Mean Squared Error |
| QAOA | Quantum Approximate Optimization Algorithm |
| QT | Quantum Tunneling |
| RL | Reinforcement Learning |
| RSSM | Recurrent State-Space Model |
| SP | Superposition |
| EN | Entanglement |

---

## 1. INTRODUCTION

### 1.1 Context & Problem Statement

The central challenge addressed in this dissertation relates to the training of predictive models in reinforcement learning. When an AI agent needs to learn a task—such as controlling a robot arm or playing a video game—it traditionally learns by trial and error, which can be extremely slow and expensive. A more efficient approach is to first build a mental model of how the world works, then practice inside that mental model before acting in reality. This mental model is what researchers call a **world model**.

World models are neural networks that learn to predict what happens next in an environment. The popular DreamerV3 algorithm [1] uses this approach and achieves remarkable results across diverse domains. However, training these predictive models still faces significant challenges:

1. **Local Minima Entrapment:** Training often gets trapped in suboptimal solutions
2. **Sample Inefficiency:** Millions of environment interactions are needed
3. **Slow Convergence:** Days of training even on powerful GPUs
4. **Error Accumulation:** Small prediction mistakes snowball into large errors over long horizons

This research asks: **Can we address these specific pain points by incorporating mathematical structures inspired by quantum mechanics?**

### 1.2 Objectives

The research objectives are:

1. **Build a solid baseline system** following the DreamerV3 architecture and verify correctness on standard test environments

2. **Implement quantum-inspired training methods:**
   - QAOA-style optimization for escaping local minima
   - Superposition-based experience replay for improved sampling
   - Quantum gate-inspired neural network layers
   - Error-correction ensemble for reducing accumulated errors

3. **Conduct rigorous experimental evaluation** across multiple environments (CartPole, DMControl Suite, Atari) with proper statistical analysis

4. **Identify conditions** under which each quantum-inspired method helps or hinders performance

5. **Provide practical guidelines** for practitioners considering these techniques

### 1.3 Literature Review

#### 1.3.1 World Models in Reinforcement Learning

The concept of world models was popularized by Ha and Schmidhuber [3] who demonstrated that agents can learn compact representations of their environment and use them for planning. DreamerV3 [1] represents the current state-of-the-art, achieving human-level performance across diverse domains using a Recurrent State-Space Model (RSSM) architecture.

MuZero [4] showed that world models can work without even knowing environment rules, learning both dynamics and rewards from experience. Recent work like RLVR-World [6] and PWM [7] has demonstrated that training methodology significantly impacts world model quality.

#### 1.3.2 Quantum-Inspired Machine Learning

Quantum-inspired algorithms adapt concepts from quantum computing to classical systems without requiring quantum hardware. Key principles include:

- **Superposition:** Being in multiple states simultaneously
- **Interference:** Combining probability amplitudes
- **Entanglement:** Correlated states across systems
- **Tunneling:** Passing through energy barriers

Wei et al. [8] demonstrated quantum-inspired experience replay for RL, while Dong et al. [9] applied quantum concepts to robot navigation. These works establish that quantum principles can transfer to classical learning.

#### 1.3.3 QAOA and Quantum Optimization

The Quantum Approximate Optimization Algorithm (QAOA) [12] alternates between cost and mixer operators to solve combinatorial problems. Zhou et al. [13] analyzed QAOA performance, showing that the alternating structure helps escape local minima. This principle inspired our QAOA-style training approach.

#### 1.3.4 Quantum Error Correction

Google's Willow chip [16] demonstrated that quantum error correction can achieve below-threshold error rates through redundancy and voting mechanisms. This inspired our ensemble approach where multiple models vote on predictions.

### 1.4 Limitations of Existing Approaches

Current world model training methods face several limitations:

| Limitation | Impact | Existing Mitigation |
|------------|--------|---------------------|
| Local minima | Suboptimal predictions | Learning rate scheduling |
| Sample inefficiency | High data requirements | Prioritized replay |
| Slow convergence | Long training times | Larger models |
| Error accumulation | Poor long-horizon accuracy | Shorter prediction horizons |

No prior work has systematically evaluated quantum-inspired methods for world model training—a gap this dissertation addresses.

### 1.5 Proposed Solution

We propose and evaluate four quantum-inspired training approaches:

**Table 1.1: Mapping Problems to Quantum-Inspired Solutions**

| Training Problem | Quantum Concept | Implementation |
|------------------|-----------------|----------------|
| Local minima | Quantum tunneling | Noise injection optimizer |
| Sample inefficiency | Superposition | Weighted experience replay |
| Slow convergence | Entanglement | Entanglement-inspired feature layers |
| Error accumulation | Interference | Uncertainty-weighted ensemble voting |

### 1.6 Project Significance & Contributions

**Primary Contributions:**

1. **First systematic evaluation** of quantum-inspired methods for world model training
2. **Discovery of domain-specific effectiveness:** Interference Ensemble provides +36-47% improvement on state-based tasks but -132% to -414% on visual tasks
3. **Negative result documentation:** Superposition replay fails catastrophically on complex dynamics (-158% to -630%)
4. **Practical guidelines** for method selection based on task characteristics

**Significance:**
- If methods work: Practitioners can use immediately on classical hardware
- If methods fail: We understand when and why to avoid them
- Both outcomes advance scientific understanding

---

## 2. SYSTEM ARCHITECTURE

### 2.1 RSSM World Model Architecture

All experiments use the Recurrent State-Space Model (RSSM) following DreamerV3 design. Figure 2.1 shows the complete architecture.

**Figure 2.1: RSSM World Model Architecture**

![RSSM World Model Architecture](../results/figures/architecture_rssm.png)

The architecture consists of:
- **Encoder Network:** Transforms observations into latent representations
- **RSSM Core:** Combines deterministic (GRU, 512 units) and stochastic (Gaussian, 64 dimensions) state components
- **Decoder Networks:** Reconstructs observations, predicts rewards and episode continuation

**Standard Configuration (Constant Across All Methods):**

| Parameter | Value |
|-----------|-------|
| Stochastic dimension | 64 |
| Deterministic dimension | 512 |
| Hidden dimension | 512 |
| State dimension | 576 (512 + 64) |
| Encoder hidden | [512, 512] |
| Decoder hidden | [512, 512] |

### 2.2 Quantum-Inspired Components

#### 2.2.1 Quantum Tunneling Optimizer

Adds controlled noise to escape local minima, inspired by quantum particles tunneling through energy barriers.

**Figure 2.2: Quantum Tunneling Optimizer Visualization**

![Quantum Tunneling Optimizer](../results/figures/architecture_quantum_tunneling.png)

The tunneling mechanism adds noise proportional to:
```
noise_scale = tunnel_prob × exp(-barrier_height / temperature)
```

#### 2.2.2 Superposition Replay Buffer

Samples multiple experiences and combines them with learned weights, inspired by quantum superposition.

**Figure 2.3: Superposition Replay Buffer (with Warning)**

![Superposition Replay Buffer](../results/figures/architecture_superposition_buffer.png)

**Warning:** This approach fails catastrophically on complex dynamics as documented in Section 6.

#### 2.2.3 Entanglement Layers

Applies quantum gate-like transformations to create correlated features.

**Figure 2.4: Entanglement-Inspired Feature Layer**

![Entanglement Layer](../results/figures/architecture_entanglement_layer.png)

#### 2.2.4 Interference Ensemble

Multiple models with phase-weighted voting—the most successful quantum-inspired approach.

**Figure 2.5: Interference Ensemble Architecture**

![Interference Ensemble Architecture](../results/figures/architecture_interference_ensemble.png)

The key innovation is phase-weighted aggregation:
```
prediction = Σᵢ wᵢ × modelᵢ(state)
wᵢ = amplitude × cos(phase_difference)
```

### 2.3 Experimental Framework

**Figure 2.6: Experimental Pipeline Overview**

![Experimental Pipeline](../results/figures/experimental_pipeline.png)

The experimental pipeline consists of three phases:
- **Phase 1:** Simple Control (CartPole) - Quick validation
- **Phase 2:** DMControl Suite (Walker, Cheetah, Reacher) - Primary benchmarks
- **Phase 3:** Atari (Pong, Breakout) - Visual RL challenges

---

## 3. DESIGN CONSIDERATIONS

### 3.1 Architectural Decisions

**Decision 1: Fixed Architecture**

All methods use identical neural network architectures. Only the training procedure varies. This ensures fair comparison where differences in results come from the training method, not architecture.

**Decision 2: DreamerV3-Style RSSM**

We chose RSSM because:
- State-of-the-art world model architecture
- Well-documented and reproducible
- Handles both state-based and visual inputs
- Explicit uncertainty modeling through stochastic component

**Decision 3: Classical Hardware Only**

All implementations run on classical CPUs/GPUs without quantum simulators. This ensures practical applicability—results are immediately usable by practitioners.

### 3.2 Fair Comparison Strategy

To ensure fair comparison:

1. **Same seeds:** All experiments use seeds [42, 123, 456, 789, 1024]
2. **Same hyperparameters:** Learning rate (3e-4), batch size (32 for state-based, 16 for Atari due to CNN memory requirements), sequence length (20)
3. **Same architecture:** Identical RSSM structure
4. **Same training steps:** 10,000 steps per configuration
5. **Same evaluation:** Held-out test data, multiple evaluation batches
6. **Environment-specific episodes:** CartPole/Walker (100), DMControl continuous (200), Atari (50) -- adjusted for episode length and computational cost

### 3.3 Statistical Methodology

**Test Selection:** Mann-Whitney U (non-parametric, appropriate for n=5)

**Note on minimum achievable p-value:** With n₁=n₂=5, the smallest possible Mann-Whitney U statistic is U=0, yielding a minimum two-sided p-value of approximately 0.008 (exact: 0.00794). Therefore, reported p-values of "< 0.008" represent the strongest possible significance achievable with this sample size.

**Multiple Comparison Correction:** Bonferroni
- 4 comparisons per environment (4 quantum-inspired methods vs baseline)
- Original α = 0.05
- Corrected α = 0.05/4 = 0.0125

Since the minimum achievable p-value (0.008) is below 0.0125, statistically significant results survive Bonferroni correction.

**Effect Size:** Cohen's d
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- 0.8 ≤ |d| < 1.2: Large
- |d| ≥ 1.2: Very Large

**Note:** Some reported Cohen's d values are extremely large (|d| > 10), which reflects the highly controlled experimental conditions where between-seed variance is small relative to between-method differences. These values are mathematically correct but should be interpreted as indicating clear, unambiguous separation rather than compared to typical behavioral science benchmarks.

---

## 4. DATASET & MODEL ARCHITECTURE

### 4.1 Environment Specifications

**Table 4.1: Environment Specifications**

| Environment | Domain | Obs Dim | Action Dim | Type | Episodes |
|-------------|--------|---------|------------|------|----------|
| CartPole-v1 | Classic | 4 | 2 | Discrete | 100 |
| Pendulum-v1 | Classic | 3 | 1 | Continuous | 100 |
| Walker-walk | DMControl | 24 | 6 | Continuous | 100 |
| Cheetah-run | DMControl | 17 | 6 | Continuous | 200 |
| Reacher-easy | DMControl | 6 | 2 | Continuous | 200 |
| Reacher-hard | DMControl | 6 | 2 | Continuous | 200 |
| Pong | Atari | 84×84×1 | 6 | Discrete | 50 |
| Breakout | Atari | 84×84×1 | 4 | Discrete | 50 |

### 4.2 Data Collection Protocol

1. **Random Policy Collection:** Episodes collected using random actions
2. **Preprocessing:** Observations normalized to [0, 1], actions one-hot encoded
3. **Sequence Sampling:** Random sequences of length 20 from episodes
4. **Train/Test Split:** 80% training, 20% held-out test

### 4.3 Model Specifications

**Table 4.2: Model Architecture Parameters**

| Component | Specification |
|-----------|---------------|
| Encoder (State) | MLP [512, 512] with ELU |
| Encoder (Visual) | CNN: 32→64→64 filters, then MLP |
| RSSM Deterministic | GRU, 512 hidden units |
| RSSM Stochastic | Gaussian, 64 dimensions |
| Prior Network | MLP [512] → (mean, logvar) |
| Posterior Network | MLP [512] → (mean, logvar) |
| Observation Decoder | MLP [512, 512] → obs_dim |
| Reward Predictor | MLP [512, 512] → 1 |

**Table 4.3: Training Configuration**

| Parameter | State-Based | Atari (Visual) |
|-----------|-------------|----------------|
| Optimizer | AdamW | AdamW |
| Learning Rate | 3×10⁻⁴ | 3×10⁻⁴ |
| Batch Size | 32 | 16* |
| Sequence Length | 20 | 20 |
| Training Steps | 10,000 | 10,000 |
| KL Weight | 1.0 | 1.0 |
| Gradient Clipping | 100.0 | 100.0 |
| Seeds | [42, 123, 456, 789, 1024] | [42, 123, 456, 789, 1024] |

*Atari batch size reduced to 16 to accommodate CNN encoder/decoder memory requirements.

---

## 5. IMPLEMENTATION DETAILS

### 5.1 Quantum Tunneling Optimizer

The tunneling optimizer adds controlled noise to parameters during training:

```python
class TunnelingOptimizer:
    def __init__(self, base_optimizer, tunnel_prob=0.1, temperature=1.0):
        self.base_optimizer = base_optimizer
        self.tunnel_prob = tunnel_prob
        self.temperature = temperature

    def step(self, loss):
        self.base_optimizer.step()

        # Estimate barrier height from loss landscape
        barrier_height = self.estimate_barrier(loss)

        # Compute tunneling probability
        tunnel_scale = self.tunnel_prob * torch.exp(
            -barrier_height / self.temperature
        )

        # Add noise to parameters
        for param in self.parameters():
            noise = torch.randn_like(param) * tunnel_scale
            param.data += noise
```

### 5.2 Superposition Replay Buffer

Combines multiple experiences with learned weights:

```python
class SuperpositionBuffer:
    def __init__(self, capacity, num_superposition=4):
        self.buffer = ReplayBuffer(capacity)
        self.num_superposition = num_superposition
        self.weight_network = nn.Linear(obs_dim * num_superposition, num_superposition)

    def sample(self, batch_size):
        # Sample multiple sets of experiences
        experiences = [self.buffer.sample(batch_size)
                      for _ in range(self.num_superposition)]

        # Compute combination weights
        combined_obs = torch.cat([e.obs for e in experiences], dim=-1)
        weights = F.softmax(self.weight_network(combined_obs), dim=-1)

        # Weighted combination (interference pattern)
        final_obs = sum(w * e.obs for w, e in zip(weights.T, experiences))
        final_action = experiences[0].action  # Use primary action
        final_reward = sum(w * e.reward for w, e in zip(weights.T, experiences))

        return Experience(final_obs, final_action, final_reward)
```

### 5.3 Entanglement Layers

Quantum gate-inspired transformations:

```python
class EntanglementLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.hadamard_scale = 1.0 / math.sqrt(2)
        self.control_net = nn.Linear(dim, dim)

    def forward(self, x):
        # Hadamard-like mixing
        x_rotated = torch.roll(x, shifts=1, dims=-1)
        h_out = self.hadamard_scale * (x + x_rotated)

        # CNOT-like controlled operation
        control = torch.sigmoid(self.control_net(h_out))
        target = torch.roll(h_out, shifts=self.dim // 2, dims=-1)
        cnot_out = h_out + control * target

        return cnot_out
```

### 5.4 Interference Ensemble

Multiple models with phase-weighted aggregation:

```python
class InterferenceEnsemble(nn.Module):
    def __init__(self, model_class, num_models=5, **model_kwargs):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(num_models)
        ])
        self.phase_offsets = nn.Parameter(torch.zeros(num_models))
        self.interference_strength = 0.7

    def forward(self, obs, action):
        # Get predictions from all models
        predictions = [model(obs, action) for model in self.models]
        predictions = torch.stack(predictions, dim=0)

        # Compute uncertainties
        mean_pred = predictions.mean(dim=0)
        uncertainties = ((predictions - mean_pred) ** 2).mean(dim=tuple(range(1, predictions.dim())))

        # Compute interference weights
        amplitudes = 1.0 / (uncertainties + 1e-8)
        amplitudes = amplitudes / amplitudes.sum()
        phases = torch.sigmoid(uncertainties) * math.pi + self.phase_offsets

        weights = torch.zeros(self.num_models, device=predictions.device)
        for i in range(self.num_models):
            for j in range(self.num_models):
                phase_diff = phases[i] - phases[j]
                weights[i] += amplitudes[i] * amplitudes[j] * torch.cos(phase_diff)

        # Normalize and combine
        weights = torch.abs(weights)
        weights = weights / weights.sum()
        weights = self.interference_strength * weights + (1 - self.interference_strength) / self.num_models

        # Dynamic dimension expansion for broadcasting
        num_extra_dims = predictions.dim() - 1
        weights_shape = [self.num_models] + [1] * num_extra_dims
        weights_expanded = weights.view(*weights_shape)

        combined = (predictions * weights_expanded).sum(dim=0)
        return combined
```

### 5.5 Training Pipeline

```python
def train_world_model(model, buffer, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for step in range(config.num_steps):
        # Sample batch
        obs, actions, rewards = buffer.sample(config.batch_size, config.seq_len)

        # Forward pass
        pred_obs, info = model(obs, actions)

        # Compute losses
        recon_loss = F.mse_loss(pred_obs, obs)
        kl_loss = compute_kl(info['posteriors'], info['priors'])
        total_loss = recon_loss + config.kl_weight * kl_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Logging
        if step % 1000 == 0:
            print(f"Step {step}: Loss={total_loss:.4f}, Recon={recon_loss:.4f}")

    return model
```

---

## 6. PERFORMANCE EVALUATION & RESULTS

### 6.1 Experimental Setup

**Hardware:** AMD Ryzen 9 9950X3D, NVIDIA RTX 5090, 32GB RAM

**Software:** Python 3.8+, PyTorch 2.0+, Gymnasium, DMControl Suite, ALE

**Evaluation Protocol:**
1. Train each method for 10,000 steps
2. Evaluate on held-out test data (10 batches)
3. Repeat with 5 random seeds
4. Report mean ± standard deviation
5. Perform statistical significance tests

### 6.2 Phase 1: Simple Control (CartPole, Pendulum)

#### 6.2.1 CartPole-v1 Results

CartPole served as a sanity check environment for validating implementations. The simple 4-dimensional state space reveals baseline behavior of each method before scaling to complex environments.

**Table 6.1: CartPole-v1 Results (5 seeds, IE: 2 seeds)**

| Approach | Test Obs MSE | Train Obs MSE | Time (s) | Params |
|----------|-------------|---------------|----------|--------|
| Baseline | 0.109 ± 0.017 | 0.010 ± 0.003 | 953 | 4.7M |
| Quantum Tunneling | 0.111 ± 0.018 | 0.010 ± 0.004 | 1200 | 4.7M |
| Superposition | 0.164 ± 0.030 | 0.167 ± 0.040 | 909 | 4.7M |
| Entanglement | 0.112 ± 0.020 | 0.010 ± 0.003 | 889 | 5.3M |
| Interference Ensemble* | 0.106 ± 0.027 | 0.009 ± 0.002 | 3916 | 23.7M |

*IE CartPole results based on 2 seeds only. †Note on IE reward prediction: The Interference Ensemble's training objective (`compute_ensemble_loss`) optimises only observation reconstruction and diversity—it does not include a reward prediction term. Consequently, the reward heads of the individual ensemble members remain untrained. IE reward MSE values reported throughout this dissertation are therefore unreliable and should not be compared with single-model reward MSE. This is a known architectural limitation; the observation prediction MSE (the primary evaluation metric) is unaffected.

**Key observations:** Baseline, Quantum Tunneling, Entanglement, and IE achieve comparable observation MSE (~0.11). Superposition shows elevated error (0.164, ~50% worse), foreshadowing its catastrophic failure on complex environments. The large generalization gap (train ~0.01 vs test ~0.11) across all methods indicates CartPole's test distribution differs meaningfully from training.

#### 6.2.2 Pendulum-v1 Results

Pendulum-v1 bridges discrete CartPole and continuous DMControl environments with its 3-dimensional state space (cos θ, sin θ, angular velocity) and continuous action (torque in [-2, 2]).

**Table 6.2: Pendulum-v1 Results (5 seeds)**

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 0.027 ± 0.003 | — | — | — |
| Quantum Tunneling | 0.026 ± 0.004 | +5.0% | 0.548 | 0.39 |
| Superposition | **0.140 ± 0.020** | **−411%** | **0.008** | **−8.04** |
| Entanglement | 0.031 ± 0.008 | −14.9% | 0.690 | −0.71 |
| Interference Ensemble | 0.031 ± 0.002 | −12.9% | 0.095 | −1.49 |

**Key observations:** Unlike DMControl environments where IE achieved 36-47% improvements, Pendulum shows **no significant benefit from any quantum-inspired method**. IE is slightly worse (−12.9%, not significant), suggesting that low dimensionality (3D) alone is insufficient—IE improvements also require sufficient dynamical complexity. Superposition continues its catastrophic failure pattern (−411%), consistent across all environments. QT shows marginal improvement (+5.0%) but fails to reach significance.

### 6.3 Phase 2: DMControl Suite

**Figure 6.1: Learning Curves Comparison**

![Learning Curves](../results/figures/comprehensive_comparison_learning.png)

**Figure 6.2: Box Plot Comparison Across Methods**

![Box Plots](../results/figures/comprehensive_comparison_boxplots.png)

#### 6.3.1 Walker-walk Results

**Table 6.3: Walker-walk Results**

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 1.799 ± 0.060 | — | — | — |
| Quantum Tunneling | 1.797 ± 0.030 | +0.1% | 0.841 | 0.03 |
| Superposition | 4.645 ± 0.198 | **−158%** | <0.008 | −17.65 |
| Entanglement | 1.798 ± 0.032 | +0.1% | 0.912 | 0.03 |
| **Interference Ensemble** | **1.022 ± 0.013** | **+43.2%** | **<0.008** | **18.13** |

#### 6.3.2 Cheetah-run Results

**Table 6.4: Cheetah-run Results**

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 0.573 ± 0.009 | — | — | — |
| Quantum Tunneling | 0.578 ± 0.005 | −0.9% | 0.421 | −0.72 |
| Superposition | 2.858 ± 0.062 | **−399%** | <0.008 | −36.71 |
| Entanglement | 0.575 ± 0.007 | −0.3% | 0.754 | −0.22 |
| **Interference Ensemble** | **0.367 ± 0.007** | **+35.9%** | **<0.008** | **26.15** |

#### 6.3.3 Reacher-easy Results

**Table 6.5: Reacher-easy Results**

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 0.125 ± 0.005 | — | — | — |
| Quantum Tunneling | 0.134 ± 0.006 | −7.2% | 0.095 | −1.66 |
| Superposition | 0.915 ± 0.015 | **−630%** | <0.008 | −72.26 |
| Entanglement | 0.130 ± 0.006 | −3.5% | 0.312 | −0.77 |
| **Interference Ensemble** | **0.069 ± 0.004** | **+45.0%** | **<0.008** | **12.76** |

#### 6.3.4 Reacher-hard Results

Reacher-hard uses the same observation and action spaces as Reacher-easy but requires reaching smaller, more distant targets, testing whether quantum-inspired methods scale to harder variants of the same task.

**Table 6.6: Reacher-hard Results (5 seeds)**

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 0.127 ± 0.004 | — | — | — |
| Quantum Tunneling | 0.128 ± 0.004 | −0.8% | 1.000 | −0.24 |
| Superposition | 0.904 ± 0.021 | **−612%** | <0.008 | −51.98 |
| Entanglement | 0.129 ± 0.004 | −1.4% | 0.690 | −0.45 |
| **Interference Ensemble** | **0.068 ± 0.003** | **+46.7%** | **<0.008** | **15.75** |

**Difficulty comparison:** Remarkably, Reacher-hard baseline MSE (0.127) is nearly identical to Reacher-easy (0.125), because both share the same 6-dimensional observation dynamics. The "hard" variant affects reward sparsity (smaller target), not observation complexity. IE improvement is slightly stronger on Reacher-hard (+46.7% vs +45.0%), confirming that the ensemble's observation prediction benefit is robust to reward-structure changes.

### 6.4 Phase 3: Atari (Visual RL)

#### 6.4.1 Pong Results

**Table 6.7: Pong Results**

| Approach | Test MSE (×10⁻⁴) | Δ vs Baseline | p-value | Significant? |
|----------|------------------|---------------|---------|--------------|
| Baseline | 2.93 ± 0.13 | — | — | — |
| Quantum Tunneling | 2.86 ± 0.02 | +2.2% | 0.087 | No |
| Superposition | 2.88 ± 0.12 | +1.7% | 0.234 | No |
| Entanglement | 3.01 ± 0.11 | −2.9% | 0.178 | No |
| **Interference Ensemble** | **6.81 ± 0.16** | **−132%** | **<0.008** | **Yes (worse)** |

#### 6.4.2 Breakout Results

**Table 6.8: Breakout Results**

| Approach | Test MSE (×10⁻⁴) | Δ vs Baseline | p-value | Significant? |
|----------|------------------|---------------|---------|--------------|
| Baseline | 5.39 ± 0.18 | — | — | — |
| Quantum Tunneling | 5.39 ± 0.23 | −0.1% | 0.921 | No |
| Superposition | 5.31 ± 0.13 | +1.4% | 0.312 | No |
| Entanglement | 5.57 ± 0.18 | −3.4% | 0.089 | No |
| **Interference Ensemble** | **27.69 ± 0.08** | **−414%** | **<0.008** | **Yes (worse)** |

### 6.5 Statistical Analysis

**Figure 6.3: Effect Size Analysis**

![Effect Sizes](../results/figures/comprehensive_comparison_effects.png)

**Table 6.9: Summary of Statistically Significant Results (Bonferroni-corrected α = 0.0125)**

| Environment | Significant Improvement | Significant Degradation |
|-------------|------------------------|------------------------|
| CartPole | None | None |
| Pendulum | None | Superposition (−411%, p=0.008) |
| Walker-walk | Interference Ensemble (+43.2%) | Superposition (−158%) |
| Cheetah-run | Interference Ensemble (+35.9%) | Superposition (−399%) |
| Reacher-easy | Interference Ensemble (+45.0%) | Superposition (−630%) |
| Reacher-hard | Interference Ensemble (+46.7%) | Superposition (−612%) |
| Pong | None | Interference Ensemble (−132%) |
| Breakout | None | Interference Ensemble (−414%) |
| **Total** | **4 (all DMControl)** | **8 (6 SP + 2 IE)** |

Note: Bonferroni correction uses 4 comparisons per environment (4 quantum-inspired methods vs baseline), yielding α = 0.05/4 = 0.0125. The minimum achievable p-value for n=5 Mann-Whitney U is 0.008, which survives this correction.

### 6.6 Domain-Specific Findings

**Figure 6.4: Domain-Specific Performance (Key Finding)**

![Domain Comparison](../results/figures/domain_comparison_key_finding.png)

**Critical Discovery:** Interference Ensemble shows opposite effects by domain:

| Domain | Environment | Baseline MSE | IE MSE | Change |
|--------|-------------|-------------|--------|--------|
| Simple | CartPole | 0.112 | 0.106 | +5% (NS) |
| Simple | Pendulum | 0.027 | 0.031 | **−13% (NS)** |
| State-Based | Walker | 1.799 | 1.022 | **+43%** |
| State-Based | Cheetah | 0.573 | 0.367 | **+36%** |
| State-Based | Reacher-easy | 0.125 | 0.069 | **+45%** |
| State-Based | Reacher-hard | 0.127 | 0.068 | **+47%** |
| Visual | Pong | 2.93e-4 | 6.81e-4 | **−132%** |
| Visual | Breakout | 5.39e-4 | 27.69e-4 | **−414%** |

**Figure 6.5: Results Summary Across All Environments**

![Results Summary](../results/figures/results_summary_all.png)

**Figure 6.6: Long-Horizon Prediction Accuracy**

![Long Horizon](../results/figures/error_correction_long_horizon.png)

**Figure 6.7: Ablation Studies**

![Ablation QAOA](../results/figures/ablation_qaoa.png)

![Ablation Superposition](../results/figures/ablation_superposition.png)

![Ablation Ensemble](../results/figures/ablation_ensemble.png)

![Ablation Summary](../results/figures/ablation_summary.png)

The ablation studies reveal the contribution of each quantum-inspired component. For the Interference Ensemble, varying the number of ensemble members (K=1,3,5,7) shows that K=5 provides the best accuracy-cost tradeoff on state-based tasks. The phase-weighting mechanism contributes approximately 8-12% of the total improvement over uniform ensemble averaging. For Quantum Tunneling, the tunneling probability schedule matters more than the noise magnitude, though neither produces statistically significant improvements. Superposition ablation confirms that even small mixing coefficients (α=0.1) degrade performance on complex dynamics, indicating the failure is fundamental rather than a tuning issue.

---

**Table 6.10: Long-Horizon Prediction Results (Mean MSE at Horizon Steps)**

Long-horizon prediction tests the world model's ability to accurately imagine future states over extended rollouts. State-based environments report horizons at steps 5, 10, 15, 20; Atari at steps 5 and 10 only.

*State-Based Environments:*

| Env | Method | H=5 | H=10 | H=15 | H=20 |
|-----|--------|-----|------|------|------|
| CartPole | Baseline | 0.025 | 0.043 | 0.066 | 0.110 |
| CartPole | QT | 0.025 | 0.041 | 0.066 | 0.115 |
| CartPole | SP | 0.046 | 0.078 | 0.124 | 0.166 |
| CartPole | EN | 0.025 | 0.042 | 0.069 | 0.118 |
| CartPole | **IE** | **0.023** | **0.037** | **0.062** | **0.100** |
| Pendulum | Baseline | 0.042 | 0.033 | 0.028 | 0.028 |
| Pendulum | QT | 0.040 | 0.033 | 0.030 | 0.029 |
| Pendulum | SP | 0.196 | 0.145 | 0.131 | 0.138 |
| Pendulum | EN | 0.050 | 0.038 | 0.033 | 0.033 |
| Pendulum | IE | 0.033 | 0.027 | 0.028 | 0.030 |
| Walker | Baseline | 1.833 | 1.760 | 1.740 | 1.757 |
| Walker | QT | 1.832 | 1.754 | 1.740 | 1.767 |
| Walker | SP | 4.538 | 4.513 | 4.560 | 4.575 |
| Walker | EN | 1.859 | 1.765 | 1.739 | 1.768 |
| Walker | **IE** | **0.966** | **0.948** | **0.974** | **1.007** |
| Cheetah | Baseline | 0.921 | 0.796 | 0.690 | 0.609 |
| Cheetah | QT | 0.913 | 0.779 | 0.671 | 0.594 |
| Cheetah | SP | 2.832 | 2.917 | 2.942 | 2.965 |
| Cheetah | EN | 0.956 | 0.804 | 0.695 | 0.612 |
| Cheetah | **IE** | **0.501** | **0.451** | **0.407** | **0.374** |
| Reacher-easy | Baseline | 0.298 | 0.188 | 0.148 | 0.128 |
| Reacher-easy | QT | 0.304 | 0.194 | 0.154 | 0.134 |
| Reacher-easy | SP | 0.940 | 0.951 | 0.945 | 0.951 |
| Reacher-easy | EN | 0.301 | 0.191 | 0.151 | 0.131 |
| Reacher-easy | **IE** | **0.177** | **0.110** | **0.085** | **0.073** |
| Reacher-hard | Baseline | 0.299 | 0.193 | 0.154 | 0.134 |
| Reacher-hard | QT | 0.291 | 0.185 | 0.146 | 0.126 |
| Reacher-hard | SP | 0.908 | 0.910 | 0.920 | 0.920 |
| Reacher-hard | EN | 0.294 | 0.186 | 0.147 | 0.127 |
| Reacher-hard | **IE** | **0.156** | **0.100** | **0.078** | **0.066** |

*Visual (Atari) Environments (×10⁻⁴):*

| Env | Method | H=5 | H=10 |
|-----|--------|-----|------|
| Pong | Baseline | 2.96 | 2.95 |
| Pong | QT | 4.09 | 3.39 |
| Pong | SP | 2.75 | 2.75 |
| Pong | EN | 2.89 | 2.89 |
| Pong | IE | N/A† | N/A† |
| Breakout | Baseline | 6.04 | 6.05 |
| Breakout | QT | 5.02 | 4.96 |
| Breakout | SP | 5.07 | 5.11 |
| Breakout | EN | 5.89 | 5.89 |
| Breakout | IE | N/A† | N/A† |

†IE Atari long-horizon data unavailable due to modified evaluation pipeline required for the 103M parameter ensemble with CNN encoders.

**Key observations:** (1) IE consistently achieves the lowest MSE on all state-based environments at every horizon. (2) Cheetah, Reacher-easy, and Reacher-hard show decreasing MSE with longer horizons (models converge to steady-state dynamics). (3) CartPole shows the conventional increasing-error pattern; Pendulum shows a mild decrease. (4) SP is consistently the worst on state-based tasks. (5) Atari methods show negligible horizon dependence. (6) Reacher-easy and Reacher-hard exhibit nearly identical long-horizon patterns, confirming that the "hard" reward structure does not alter the learned dynamics.

---

**Table 6.11: Generalization Gap Analysis ((Test MSE − Train MSE) / Train MSE × 100)**

This table quantifies overfitting tendency by comparing training and test observation MSE. A larger positive gap indicates greater overfitting; negative values indicate test MSE is lower than train MSE (typically statistical noise).

| Environment | Baseline | QT | SP | EN | IE |
|-------------|----------|-----|-----|-----|-----|
| CartPole | +978% | +988% | −2% | +999% | +1103% |
| Pendulum | +5% | +1% | +6% | +7% | +2% |
| Walker | +17% | +14% | <1% | +17% | +25% |
| Cheetah | +3% | +3% | −2% | +3% | +3% |
| Reacher-easy | +1% | +4% | +1% | <1% | +3% |
| Reacher-hard | −2% | <1% | −1% | −1% | +1% |
| Pong | +2% | −2% | +3% | +3% | N/A† |
| Breakout | <1% | −1% | −1% | +2% | N/A† |

†IE Atari train_obs_mse unavailable.

**Key observations:** (1) **CartPole is a major outlier** with ~1000% generalization gaps across all methods except SP, indicating that the 4D state space leads to severe train/test distribution mismatch. (2) **All other environments show healthy gaps of 0-25%**, suggesting good generalization. (3) **SP on CartPole shows near-zero gap** (−2%), but this is because it fails to fit both train and test data well (high absolute MSE on both). (4) **Walker has the highest non-CartPole gaps** (14-25%), with IE showing the largest relative gap (25%) despite achieving the lowest absolute test MSE—the ensemble fits training data more tightly. (5) **DMControl Reacher and Cheetah environments generalize well** (<5% gap) regardless of method. (6) **Atari environments show negligible gaps** (<3%), consistent with the regularisation provided by high-dimensional CNN representations.

---

**Table 6.12: Computational Cost Comparison**

| Method | Parameters (State / Atari) | Training Time | Memory | Cost-Effective? |
|--------|---------------------------|---------------|--------|-----------------|
| Baseline | 4.7M / 4.7M | 1.0× | 1.0× | Reference |
| Quantum Tunneling | 4.7M / 4.7M | 1.0-1.2× | 1.0× | Neutral |
| Superposition | 4.7M / 4.7M | 0.9-1.0× | 1.2× | **No (harmful)** |
| Entanglement | 5.3M / 5.3M | 1.1-1.2× | 1.1× | No |
| Interference Ensemble | 23.7M / 103M† | 5.0-6.0× | 5.0× | **Yes (state-based only)** |

†IE on Atari requires CNN encoders per ensemble member, increasing parameters from 23.7M (state-based, 5 members × MLP) to 103M (Atari, 5 members × CNN+MLP).

---

## 7. DISCUSSION

### 7.1 Why Interference Ensemble Works on State-Based Tasks

The Interference Ensemble achieved 36-47% improvements on DMControl tasks due to several factors:

**1. Ensemble Diversity Through Phase Initialization**

Each model starts with different "phase" parameters, leading to:
- Different initial configurations
- Diverse gradient trajectories
- Complementary learned representations

**2. Weighted Averaging Mimics Constructive Interference**

The phase-weighted combination allows:
- Agreeing models to reinforce each other (constructive interference)
- Disagreeing models to partially cancel (destructive interference)
- Natural uncertainty-based weighting

**3. Implicit Uncertainty Estimation**

The variance across ensemble members provides:
- Identification of out-of-distribution states
- Confidence-weighted predictions
- Reduced overconfident errors

### 7.2 Why Superposition Fails on Complex Dynamics

Superposition Replay caused catastrophic failure (-158% to -630%) because:

**1. Temporal Coherence Disruption**

World models require consistent sequences:
```
s₀ → a₀ → s₁ → a₁ → s₂ → ...
```

Superposition creates artificial hybrid states:
```
s_hybrid = α·s_trajectory_1 + β·s_trajectory_2
```

These never-occurred states confuse the model about true dynamics.

**2. Complex Dynamics Need Precise Trajectories**

DMControl tasks involve:
- Non-linear multi-body physics
- Chaotic sensitivity to initial conditions
- Precise state-action-next_state relationships

Introducing superposition noise destroys this precision.

### 7.3 Domain Specificity Analysis

The most significant finding is Interference Ensemble's domain specificity:

| Characteristic | Simple (CartPole, Pendulum) | Complex State-Based (DMControl) | Visual (Atari) |
|---------------|---------------------------|-------------------------------|----------------|
| Observation dim | 3-4 | 6-24 | 4096+ (CNN features) |
| Dynamics | Simple/periodic | Multi-body, nonlinear | Visual patterns |
| IE Effect | Neutral (−13% to +5%) | **+36-47%** | **−132-414%** |

**Key insight:** IE benefits require a "sweet spot" of complexity—low-dimensional observations with sufficiently complex dynamics. Simple environments (CartPole, Pendulum) lack the dynamical complexity for ensemble diversity to provide meaningful benefit, while high-dimensional visual environments overwhelm the phase-weighting mechanism.

**Why IE Fails on Visual Tasks:**

1. **Dimensionality Mismatch:** The phase-weighted mechanism is optimized for low-dimensional spaces

2. **Feature Space Characteristics:** CNN features are hierarchically structured and highly correlated, unlike physically meaningful state vectors

3. **Uncertainty Estimation Breakdown:** In 4096+ dimensions, variance becomes less meaningful and potentially misleading

**Why IE Shows No Benefit on Simple Environments:**

1. **Insufficient Dynamical Complexity:** Pendulum (3D) and CartPole (4D) have simple dynamics that a single well-trained model can capture, leaving little room for ensemble improvement

2. **Overhead Without Benefit:** The 5× parameter cost (23.7M vs 4.7M) is not justified when a single model already achieves near-optimal predictions on these simple tasks

### 7.4 Long-Horizon Prediction Behavior

An interesting observation is that long-horizon prediction errors on DMControl environments **decrease** with increasing prediction horizon (e.g., Cheetah baseline mean±std across 5 seeds: h5=0.921, h10=0.796, h15=0.690, h20=0.609), while CartPole errors increase as conventionally expected (h5=0.025, h10=0.043, h15=0.066, h20=0.110).

This occurs because DMControl locomotion tasks converge to steady-state dynamics (periodic gaits, stable postures). The world model learns this attractor: as the imagined rollout progresses, predictions converge toward the steady-state distribution, which is easier to predict than transient dynamics. In contrast, CartPole has inherently unstable dynamics where errors compound, producing the expected increasing-error pattern. Pendulum shows a hybrid pattern: errors decrease from h=5 (0.042) to h=15 (0.028) then plateau at h=20 (0.028), consistent with convergence to a gravity-dominated resting state.

This finding reinforces that world model quality should be evaluated differently for stable vs. unstable environments, and suggests that long-horizon capability is environment-dependent rather than a universal model quality metric.

### 7.5 Computational Cost Analysis

| Method | Time Overhead | Improvement | Recommendation |
|--------|---------------|-------------|----------------|
| Interference Ensemble | 5-6× | +36-47% (state) | **Use for state-based if accuracy critical** |
| Interference Ensemble | 5-6× | −273% (visual) | **Avoid for visual tasks** |
| Quantum Tunneling | 1.0-1.2× | 0-2% | Neutral, minimal benefit |
| Superposition | 0.9-1.0× | Negative | **Avoid entirely** |
| Entanglement | 1.1-1.2× | ~0% | No benefit |

---

## 8. CONCLUSION & FUTURE DIRECTIONS

### 8.1 Summary of Contributions

This dissertation provides the first systematic evaluation of quantum-inspired methods for world model training in reinforcement learning. The key contributions are:

1. **Implementation of four quantum-inspired approaches** with fair comparison methodology

2. **Discovery of significant improvements:** Interference Ensemble achieves +36-47% on state-based continuous control (p < 0.008, the minimum achievable with n=5 per group under Mann-Whitney U)

3. **Documentation of failures:** Superposition Replay fails catastrophically on complex dynamics (−158% to −630%)

4. **Critical domain-specificity finding:** Interference Ensemble excels on state-based tasks but degrades performance on visual tasks

5. **Practical guidelines:** Clear recommendations for when to use (or avoid) each method

### 8.2 Practical Recommendations

**Table 7.1: Research Question Answers**

| Question | Answer | Evidence |
|----------|--------|----------|
| Do quantum-inspired methods improve training? | **Domain-Specific Yes** | IE: +36-47% (state), −273% (visual) |
| Which principles transfer effectively? | Interference (state-based) | Effect sizes |
| What is the cost-benefit tradeoff? | 5× cost for +40% gain (state only) | Time vs accuracy |
| Are improvements consistent? | Within-domain yes | All environments |

**Recommendations:**

- **For state-based continuous control:** Use Interference Ensemble (5× cost, 40% gain)
- **For visual RL:** Use baseline (IE causes degradation)
- **For any task:** Avoid Superposition Replay
- **For quick experiments:** Baseline is sufficient

### 8.3 Limitations

1. **Fixed hyperparameters:** Not tuned per method (conservative estimates)
2. **Single architecture:** Only RSSM tested (may not generalize)
3. **Limited training duration:** 10,000 steps (longer training may differ)
4. **Specific environments:** Results may vary on other tasks
5. **Scope reduction:** The original proposal included a "Fully Integrated" approach combining all four quantum-inspired methods. This was not implemented because the experimental results revealed that the four methods target fundamentally different components (optimizer, buffer, layers, ensemble) and two of them (SP, EN) showed no benefit or harm. A combined approach would inherit SP's catastrophic failure, making integration counterproductive. The four individual methods provide cleaner scientific comparisons.
6. **Small sample size:** With n=5 seeds per configuration, the minimum achievable Mann-Whitney U p-value is 0.00794, limiting our ability to report significance beyond this threshold. Larger seed counts (n=10+) would provide finer statistical resolution.
7. **IE reward prediction untrained:** The Interference Ensemble's training objective combines observation reconstruction, KL divergence, and diversity—but omits reward prediction loss. This means the reward heads of the five ensemble members never receive gradient signal and remain at initialisation values. IE reward MSE figures throughout this dissertation are therefore unreliable. The primary evaluation metric (observation MSE) is unaffected, but reward-dependent downstream tasks (e.g., policy learning) would require adding reward loss to `compute_ensemble_loss`.

### 8.4 Future Work

1. **Hyperparameter optimization** for each quantum-inspired method
2. **Alternative architectures** (Transformers, Neural ODEs)
3. **Hybrid approaches** combining IE with visual-specific methods
4. **Theoretical analysis** of why dimensionality affects IE performance
5. **IE reward prediction training** — adding reward loss to the ensemble objective to enable reward prediction
6. **Real-world validation** on robotics tasks

---

## 9. REFERENCES

**World Models and Model-Based Reinforcement Learning:**

[1] Hafner, D., et al. (2023). "Mastering Diverse Domains through World Models." Nature.

[2] Hafner, D., et al. (2020). "Mastering Atari with Discrete World Models." arXiv:2010.02193.

[3] Ha, D., and Schmidhuber, J. (2018). "World Models." arXiv:1803.10122.

[4] Schrittwieser, J., et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature.

[5] Moerland, T., et al. (2023). "Model-based Reinforcement Learning: A Survey." Foundations and Trends in Machine Learning.

[6] Wu, J., et al. (2025). "RLVR-World: Training World Models with Reinforcement Learning." NeurIPS 2025.

[7] Georgiev, I., et al. (2024). "PWM: Policy Learning with Multi-Task World Models." arXiv:2407.02466.

**Quantum-Inspired Reinforcement Learning:**

[8] Wei, Q., et al. (2022). "Deep Reinforcement Learning with Quantum-Inspired Experience Replay." IEEE Transactions on Cybernetics.

[9] Dong, D., et al. (2012). "Robust Quantum-Inspired Reinforcement Learning for Robot Navigation." IEEE/ASME Transactions on Mechatronics.

[10] Li, J., et al. (2020). "Quantum Reinforcement Learning During Human Decision-Making." Nature Human Behaviour.

[11] Chen, C., and Dong, D. (2008). "Quantum Computation for Action Selection Using Reinforcement Learning." International Journal of Quantum Information.

**QAOA and Quantum Optimization:**

[12] Farhi, E., et al. (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.

[13] Zhou, L., et al. (2020). "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation." Physical Review X.

[14] Shaydulin, R., et al. (2024). "Parameter Setting in Quantum Approximate Optimization of Weighted Problems." Quantum.

**Quantum Machine Learning and Error Correction:**

[15] Dunjko, V., et al. (2016). "Quantum-Enhanced Machine Learning." Physical Review Letters.

[16] Google Quantum AI (2024). "Quantum Error Correction Below the Surface Code Threshold." Nature.

[17] Chen, A., and Heyl, M. (2024). "Empowering Deep Neural Quantum States through Efficient Optimization." Nature Physics.

**Recent Advances (2024-2025):**

[18] Assran, M., et al. (2025). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv preprint.

[19] Zhu, Z., et al. (2024). "Offline Model-Based Reinforcement Learning with Causal Structured World Models." Frontiers of Computer Science.

**Reinforcement Learning Foundations:**

[20] Sutton, R., and Barto, A. (2018). Reinforcement Learning: An Introduction. MIT Press.

[21] Mnih, V., et al. (2015). "Human-Level Control Through Deep Reinforcement Learning." Nature.

[22] Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

---

## 10. APPENDIX: IMPLEMENTATION CODE

### A.1 Repository Structure

The complete implementation is available at:
https://github.com/SaurabhJalendra/Quantum-Enhanced-Simulation-Learning-for-Reinforcement-Learning

```
Quantum-Enhanced-Simulation-Learning-for-Reinforcement-Learning/
├── src/
│   ├── config/
│   │   └── shared_config.py           # Standard architecture & training parameters
│   ├── quantum_inspired/
│   │   ├── tunneling_optimizer.py     # Quantum Tunneling optimizer
│   │   ├── superposition_buffer.py    # Superposition replay buffer
│   │   ├── entanglement_layer.py      # Entanglement-inspired layers
│   │   └── interference_ensemble.py   # Interference Ensemble (5-model)
│   ├── models/                        # Model registry (implementations in notebooks)
│   ├── training/                      # Training utilities
│   └── evaluation/                    # Evaluation utilities
├── phase1_cartpole_notebooks/         # CartPole & Pendulum experiments
│   ├── 02_classical_baseline.ipynb    # Baseline RSSM implementation
│   ├── 03b_quantum_tunneling.ipynb    # QT experiments
│   ├── 04b_superposition_proper.ipynb # SP experiments
│   ├── 05b_entanglement_layers.ipynb  # EN experiments
│   ├── 06d_interference_ensemble.ipynb # IE experiments
│   ├── 07_comprehensive_comparison.ipynb
│   └── 10_pendulum_experiments.ipynb  # All 5 methods on Pendulum
├── phase2_dmcontrol_notebooks/        # DMControl Suite experiments
│   ├── 02_walker_experiments.ipynb    # Walker-walk (5 methods × 5 seeds)
│   ├── 03_cheetah_experiments.ipynb   # Cheetah-run
│   ├── 04_reacher_experiments.ipynb   # Reacher-easy
│   ├── 05_dmcontrol_comparison.ipynb  # Cross-environment comparison
│   └── 06_reacher_hard_experiments.ipynb # Reacher-hard
├── phase3_atari_notebooks/            # Atari experiments
│   ├── 02_pong_experiments.ipynb      # Pong (5 methods × 5 seeds)
│   ├── 03_breakout_experiments.ipynb  # Breakout
│   └── 04_atari_comparison.ipynb      # Cross-game comparison
├── experiments/
│   ├── scripts/                       # Standalone experiment runners
│   │   ├── run_phase1_cartpole.py
│   │   ├── run_phase2_walker.py
│   │   ├── run_phase2_cheetah.py
│   │   ├── run_phase2_reacher.py
│   │   ├── run_phase3_pong.py
│   │   └── run_phase3_breakout.py
│   └── results/
│       ├── phase1/                    # CartPole & Pendulum results
│       ├── phase2/                    # Walker, Cheetah, Reacher results
│       └── phase3/                    # Pong & Breakout results
├── scripts/
│   ├── generate_diagrams.py           # All figures for this report
│   ├── validate_consistency.py        # Cross-notebook parameter checker
│   └── aggregate_cartpole_results.py  # CartPole seed aggregation
├── results/
│   └── figures/                       # All figures used in this report
├── Reports/
│   └── FINAL_DISSERTATION_2023AC05912.md
└── CLAUDE.md                          # Project context & configuration
```

**Note:** The RSSM world model (BaseWorldModel) and all training/evaluation logic are implemented directly within each experiment notebook to maintain self-contained, reproducible experiments. The `src/quantum_inspired/` directory contains the four reusable quantum-inspired modules imported by the notebooks.

### A.2 Key Files

| File | Purpose |
|------|---------|
| `src/quantum_inspired/interference_ensemble.py` | Interference Ensemble (5-model weighted averaging) |
| `src/quantum_inspired/tunneling_optimizer.py` | Quantum Tunneling optimizer wrapper |
| `src/quantum_inspired/superposition_buffer.py` | Superposition-based replay buffer |
| `src/quantum_inspired/entanglement_layer.py` | Entanglement-inspired feature layers |
| `src/config/shared_config.py` | Standard architecture & training configuration |
| `phase2_dmcontrol_notebooks/02_walker_experiments.ipynb` | Representative experiment notebook (contains full RSSM implementation) |
| `scripts/generate_diagrams.py` | Figure generation for all 8 environments |
| `experiments/scripts/run_phase2_walker.py` | Standalone experiment script example |

### A.3 Reproduction Instructions

```bash
# Clone repository
git clone https://github.com/SaurabhJalendra/Quantum-Enhanced-Simulation-Learning-for-Reinforcement-Learning.git
cd Quantum-Enhanced-Simulation-Learning-for-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt

# Option 1: Run via Jupyter notebooks (recommended)
jupyter notebook phase2_dmcontrol_notebooks/02_walker_experiments.ipynb

# Option 2: Run via standalone scripts
python experiments/scripts/run_phase2_walker.py
python experiments/scripts/run_phase2_cheetah.py
python experiments/scripts/run_phase3_pong.py
```

### A.4 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Multi-core | AMD Ryzen 9 |
| GPU | GTX 1060 | RTX 3070+ |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |

---

## 11. GLOSSARY

| Term | Definition |
|------|------------|
| **World Model** | Neural network that predicts environment dynamics |
| **RSSM** | Recurrent State-Space Model with deterministic and stochastic components |
| **Quantum-Inspired** | Classical algorithms borrowing mathematical concepts from quantum computing |
| **Interference** | Quantum concept where probability amplitudes combine constructively or destructively |
| **Superposition** | Quantum concept of existing in multiple states simultaneously |
| **Tunneling** | Quantum concept of passing through energy barriers |
| **Entanglement** | Quantum concept of correlated states |
| **QAOA** | Quantum Approximate Optimization Algorithm |
| **DMControl** | DeepMind Control Suite benchmark |
| **Cohen's d** | Effect size measure for comparing group means |
| **Mann-Whitney U** | Non-parametric statistical test for comparing distributions |
| **Bonferroni Correction** | Multiple comparison correction method |

---

## PARTICULARS OF SUPERVISOR AND EXAMINER

### Supervisor

| Field | Details |
|-------|---------|
| Name | Gaurav Kumar |
| Position | Deputy Director |
| Organization | IN-SPACe, PMA Directorate |
| Location | Ahmedabad |
| Email | gaurav.kumar45@inspace.gov.in |

### Examiner

| Field | Details |
|-------|---------|
| Name | Rishabh Swami |
| Organization | Orange Business Services India |
| Email | rishabh.swami@orange.com |

---

## REMARKS OF THE SUPERVISOR

*(To be filled by supervisor)*

---

## DECLARATION

I, Saurabh Jalendra (2023AC05912), hereby declare that this dissertation titled "Quantum-Enhanced Simulation Learning for Reinforcement Learning: A Comparative Analysis of World Model Training Approaches" is my original work and has been carried out under the supervision of Gaurav Kumar, Deputy Director, IN-SPACe.

All sources of information used have been duly acknowledged through references. This work has not been submitted elsewhere for any other degree or diploma.

**Date:** February 2026
**Place:** Jaipur, Rajasthan

**Signature:** _____________________
**Name:** Saurabh Jalendra

---

*End of Dissertation*
