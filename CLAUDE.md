# CLAUDE.md - AI Assistant Context & Memory

**Purpose:** This file provides complete context about the dissertation project for AI assistants (like Claude, GPT, etc.) to understand the project quickly and provide accurate, contextual assistance.

**Last Updated:** December 17, 2025

---

## 📋 Quick Project Summary

### What Is This Project?

This is a **Master's dissertation** comparing quantum-inspired training methods against classical approaches for world model learning in reinforcement learning.

**One-Sentence Summary:** We're testing if quantum-inspired algorithms can make training world models (neural networks that predict environment dynamics) faster and more efficient on regular computers.

**Key Point:** We're using **quantum-INSPIRED** algorithms (classical implementations borrowing quantum ideas), NOT actual quantum computers.

---

## 👤 Student Information

| Field | Value |
|-------|-------|
| **Name** | Saurabh Jalendra |
| **BITS ID** | 2023AC05912 |
| **Email** | 2023ac05912@wilp.bits-pilani.ac.in |
| **Program** | MTech AI/ML |
| **Institution** | BITS Pilani (WILP Division) |
| **Organization** | S K Jalendra Marketing Services Pvt Ltd |
| **Location** | Sri Ganganagar, Rajasthan, India |

### Supervisor

| Field | Value |
|-------|-------|
| **Name** | Gaurav Kumar |
| **Position** | Deputy Director, PMA Directorate |
| **Organization** | IN-SPACe, Ahmedabad |
| **Email** | gaurav.kumar45@inspace.gov.in |

### Additional Examiner

| Field | Value |
|-------|-------|
| **Name** | Rishabh Swami |
| **Organization** | Orange Business Services India |
| **Email** | rishabh.swami@orange.com |

---

## 🎯 Research Overview

### Title

**"Quantum-Enhanced Simulation Learning for Reinforcement Learning: A Comparative Analysis of World Model Training Approaches"**

### Research Area

**Artificial Intelligence** - Specifically at the intersection of:
1. Reinforcement Learning
2. Model-Based RL (World Models)
3. Quantum-Inspired Algorithms
4. Deep Learning Optimization

### Primary Research Question

**"Do quantum-inspired algorithmic approaches improve world model training efficiency compared to classical methods, and under what conditions?"**

### Expected Timeline

- **Start Date:** October 30, 2025
- **End Date:** January 31, 2026
- **Duration:** 13 weeks (approximately 3 months)

---

## 🧠 Core Concepts Explained

### 1. World Models

**What are they?**
Neural networks that learn to predict how environments work.

**Why are they useful?**
- Allow RL agents to plan through "imagination"
- More sample-efficient than model-free RL
- Can simulate future consequences of actions

**The Problem:**
Training world models is slow, expensive, and often gets stuck in local minima.

**Example:**
```
Real Environment:
  Agent takes action "move forward"
  → Environment returns: new_position, reward

World Model:
  Agent imagines action "move forward"
  → World model predicts: new_position, reward
  → Agent can test many actions mentally before acting
```

### 2. Quantum-Inspired Algorithms

**Important Distinction:**

| Quantum Computing | Quantum-INSPIRED (This Project) |
|-------------------|----------------------------------|
| Needs quantum hardware | Runs on regular CPUs/GPUs |
| Uses qubits | Uses classical bits |
| Very limited access | Anyone can use |
| Expensive | Practical |
| ❌ NOT this project | ✅ This is what we're doing |

**What does "quantum-inspired" mean?**
Taking concepts from quantum computing (like superposition, entanglement, QAOA) and adapting them to classical algorithms.

**Analogy:**
- Quantum computing = Flying in an airplane
- Quantum-inspired = Designing a car with aerodynamics inspired by airplanes

### 3. Reinforcement Learning (RL)

**Basic RL Loop:**
```
1. Agent observes state
2. Agent takes action
3. Environment gives reward
4. Agent updates policy
5. Repeat
```

**Model-Based RL:**
```
1. Agent learns world model (how environment works)
2. Agent uses world model to plan
3. Agent acts in real environment
4. Agent updates both policy and world model
```

---

## 🏗️ Implementation Approaches

### Six Approaches Being Compared

#### 1. Classical Baseline (Control)
- **What:** Standard DreamerV3-style training
- **How:** Gradient descent with Adam optimizer
- **Purpose:** Benchmark for comparison
- **File:** `src/models/baseline.py`

#### 2. QAOA-Enhanced
- **Inspiration:** Quantum Approximate Optimization Algorithm
- **What:** Alternating cost and mixing operators
- **Why:** Helps escape local minima
- **File:** `src/models/qaoa_enhanced.py`

#### 3. Superposition-Enhanced
- **Inspiration:** Quantum superposition
- **What:** Parallel exploration of multiple training paths
- **Why:** Better sample efficiency
- **File:** `src/models/superposition.py`

#### 4. Gate-Enhanced
- **Inspiration:** Quantum gates (Hadamard, CNOT)
- **What:** Special neural network layers with quantum-like operations
- **Why:** Richer feature representations
- **File:** `src/models/gate_enhanced.py`

#### 5. Error Correction
- **Inspiration:** Quantum error correction (Google Willow)
- **What:** Ensemble of models with majority voting
- **Why:** Robust predictions under uncertainty
- **File:** `src/models/error_correction.py`

#### 6. Fully Integrated (REQUIRED)
- **What:** Combines ALL quantum-inspired components into one enhanced DreamerV3
- **Components:**
  - QAOA-inspired optimizer (for training)
  - Superposition-enhanced replay buffer (for sampling)
  - Gate-enhanced neural layers (for representations)
  - Error correction ensemble (for predictions)
- **Why:** DreamerV3 is a collective system - quantum methods should enhance each component they're suited for
- **Comparison:** Integrated DreamerV3 vs Classical DreamerV3
- **File:** `src/models/integrated.py`

---

## 📊 Evaluation Strategy

### Primary Metrics

1. **Sample Efficiency**
   - How many environment steps to reach target performance?
   - Lower is better

2. **Training Speed**
   - Wall-clock time to convergence
   - Faster is better

3. **Prediction Accuracy**
   - Mean Squared Error on test trajectories
   - Lower is better

4. **Final Performance**
   - Average return over 100 episodes
   - Higher is better

### Secondary Metrics

5. **Stability** - Variance across random seeds
6. **Robustness** - Performance under noise
7. **Computational Cost** - FLOPs, memory, time

### Statistical Analysis

- **Multiple seeds:** 5-10 per configuration
- **Tests:** Mann-Whitney U, Cohen's d
- **Confidence:** 95% intervals
- **Correction:** Bonferroni for multiple comparisons

---

## 🌍 Test Environments (ALL REQUIRED)

All environments below are **required** for comprehensive evaluation. Each tests different aspects of world model learning.

### Tier 1: Simple Control (Development & Debugging)
| Environment | Type | Obs Dim | Action Dim | Purpose |
|-------------|------|---------|------------|---------|
| **CartPole-v1** | Discrete | 4 | 2 | Quick iteration, debugging |
| **Pendulum-v1** | Continuous | 3 | 1 | Basic continuous control |

### Tier 2: DMControl Suite (Primary Benchmarks)
| Environment | Type | Obs Dim | Action Dim | Purpose |
|-------------|------|---------|------------|---------|
| **Walker-walk** | Continuous | 24 | 6 | Locomotion, balance |
| **Cheetah-run** | Continuous | 17 | 6 | Fast locomotion |
| **Reacher-easy** | Continuous | 6 | 2 | Manipulation, precision |
| **Reacher-hard** | Continuous | 6 | 2 | Harder manipulation |

### Tier 3: Atari (Visual/Pixel-Based)
| Environment | Type | Obs Dim | Action Dim | Purpose |
|-------------|------|---------|------------|---------|
| **Pong** | Discrete | 84×84×3 | 6 | Visual RL, temporal |
| **Breakout** | Discrete | 84×84×3 | 4 | Visual RL, planning |

### Environment Progression Strategy
```
Phase 1 (Development):   CartPole, Pendulum     → Fast debugging
Phase 2 (Validation):    Reacher-easy           → Verify continuous control
Phase 3 (Full Testing):  Walker, Cheetah        → Primary results
Phase 4 (Comprehensive): Reacher-hard, Atari    → Complete evaluation
```

### Minimum Seeds Per Environment
| Environment Type | Seeds Required |
|-----------------|----------------|
| Simple Control | 5 seeds |
| DMControl Suite | 10 seeds |
| Atari | 5 seeds |

---

## 🛠️ Technical Stack

### Core Technologies

```yaml
Language: Python 3.8+
Framework: PyTorch 2.0+
RL Library: Gymnasium
Environments: DMControl Suite, Atari
Utilities: NumPy, Pandas, Matplotlib
```

### Hardware Requirements

**Minimum:**
- CPU: Multi-core
- RAM: 8GB+
- Storage: 50GB

**Recommended:**
- GPU: NVIDIA GTX 1060+
- RAM: 16GB+
- Storage: 100GB

**Ideal:**
- GPU: NVIDIA RTX 3070+
- RAM: 32GB
- Storage: 200GB+

**Development Machine (Actual):**
- CPU: AMD Ryzen 9 9950X3D
- GPU: ASUS ROG Astral RTX 5090
- RAM: 32GB

**Critical:** No quantum hardware needed!

---

## 🔧 Standard Configuration (CRITICAL)

### Architecture Parameters (MUST BE CONSISTENT)

All notebooks must use these **exact** parameters for fair comparison:

```python
# RSSM World Model Architecture
stoch_dim = 64          # Stochastic state dimension
deter_dim = 512         # Deterministic state dimension
hidden_dim = 512        # Hidden layer dimension
encoder_hidden = [512, 512]    # Encoder MLP layers
decoder_hidden = [512, 512]    # Decoder MLP layers
predictor_hidden = [512, 512]  # Reward/Continue predictor layers

# Combined state dimension
state_dim = deter_dim + stoch_dim  # = 576
```

### Training Parameters (MUST BE CONSISTENT)

```python
# Training Configuration
batch_size = 32         # Batch size for training
seq_len = 20            # Sequence length
num_steps = 10000       # Training steps (CartPole)
learning_rate = 3e-4    # Learning rate (AdamW)
kl_weight = 1.0         # KL divergence weight
num_episodes = 100      # Episodes for data collection
```

### Standard Seeds (REQUIRED)

For statistical validity, ALL notebooks must run experiments with these 5 seeds:

```python
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]
```

### Required Notebook Components

Every experiment notebook (02-08) **MUST** include these components in the Experiments section:

| Component | Purpose |
|-----------|---------|
| **Multi-Seed Experiments** | Run training with all 5 seeds, aggregate with mean ± std |
| **Test Set Evaluation** | Evaluate on held-out data (different seeds), compute generalization gap |
| **Long-Horizon Prediction** | Test imagination accuracy at horizons [5, 10, 15, 20, 30, 40, 50] |

### Why This Matters

- **Fair Comparison:** All approaches must use identical architecture to isolate the effect of quantum-inspired methods
- **Reproducibility:** Standard seeds ensure results can be replicated
- **Statistical Validity:** Multi-seed experiments provide confidence intervals
- **Generalization:** Test set evaluation reveals overfitting
- **Model Quality:** Long-horizon prediction tests true dynamics learning

---

## 📚 Key Literature

### World Models
1. **DreamerV3** (Hafner et al., 2023) - State-of-the-art world model
2. **World Models** (Ha & Schmidhuber, 2018) - Original concept
3. **MuZero** (Schrittwieser et al., 2020) - Game-playing with learned models

### Recent Advances (2024-2025)
4. **RLVR-World** (Wu et al., 2025) - RL-based world model training
5. **PWM** (Georgiev et al., 2024) - Multi-task world models

### Quantum-Inspired RL
6. **Wei et al. (2022)** - Quantum-inspired experience replay
7. **Dong et al. (2012)** - Quantum-inspired robot navigation

### QAOA
8. **Farhi et al. (2014)** - Original QAOA paper
9. **Zhou et al. (2020)** - QAOA performance analysis

### Foundations
10. **Sutton & Barto (2018)** - RL textbook

**Total References:** 22 papers

---

## 🗓️ Project Timeline

| Phase | Dates | Duration | Status |
|-------|-------|----------|--------|
| **Literature Review & Baseline** | Oct 30 - Nov 20, 2025 | 3 weeks | 🟡 In Progress |
| **Quantum-Inspired Development** | Nov 21 - Dec 24, 2025 | 5 weeks | ⚪ Pending |
| **Experimental Evaluation** | Dec 25 - Jan 14, 2026 | 3 weeks | ⚪ Pending |
| **Analysis & Documentation** | Jan 15 - Jan 22, 2026 | 1 week | ⚪ Pending |
| **Review & Revision** | Jan 23 - Jan 28, 2026 | 6 days | ⚪ Pending |
| **Final Submission** | Jan 29 - Jan 31, 2026 | 3 days | ⚪ Pending |

**Total Duration:** 13 weeks

---

## ✅ Success Criteria

### This dissertation is successful if:

1. ✅ All 5-6 approaches implemented correctly
2. ✅ Comprehensive experiments completed (multiple environments, seeds)
3. ✅ Fair statistical comparisons provided
4. ✅ Conditions identified (when each approach works)
5. ✅ Complete documentation delivered
6. ✅ Submitted on time (Jan 31, 2026)

### Important Note

**Negative results are still valuable!**

The research question is: "Under what conditions do quantum-inspired methods help?"

ANY answer to this question—positive, negative, or conditional—is a contribution.

---

## 🎓 Academic Context

### Institution Details

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI**
- Campus: Pilani, Rajasthan - 333031
- Division: Work Integrated Learning Programmes (WILP)
- Program: M.Tech in AI/ML

### Course Details

- **Course Code:** AIMLCZG628T
- **Course Name:** Dissertation
- **Credits:** As per BITS norms
- **Semester:** Second Semester 2021-2022 (actual execution 2025-2026)

### Approval Status

- ✅ **Abstract Report:** Approved by supervisor
- ✅ **Supervisor:** Gaurav Kumar (approved)
- ✅ **Examiner:** Rishabh Swami (assigned)
- ✅ **Ready to Start:** Implementation phase

---

## 💡 Key Insights for AI Assistants

### When Helping with This Project

#### Things to Remember

1. **Hardware:** Classical only, no quantum computers
2. **Scope:** Focus on world model training, NOT policy learning
3. **Timeline:** 13 weeks total, currently in Phase 1
4. **Comparison:** Must compare 5-6 approaches fairly
5. **Results:** Even negative results are valuable

#### Common Questions & Answers

**Q: Do we need quantum hardware?**
A: NO! All quantum-INSPIRED, runs on regular CPUs/GPUs.

**Q: What's the main contribution?**
A: Systematic comparison of quantum-inspired methods for world model training—no prior work exists in this specific area.

**Q: What if quantum methods don't help?**
A: Still valuable! We're answering "under what conditions do they help?" Any answer contributes knowledge.

**Q: Can we reduce scope if needed?**
A: Yes! Minimum viable: Baseline + 2-3 quantum approaches. Quality over quantity.

**Q: What's the difference between model-based and model-free RL?**
A: Model-based learns a world model to predict outcomes. Model-free learns directly from experience without modeling the environment.

#### Code Style Preferences

```python
# Preferred style
def compute_loss(model: WorldModel, batch: dict) -> torch.Tensor:
    """
    Compute world model training loss.

    Parameters
    ----------
    model : WorldModel
        The world model to train
    batch : dict
        Batch of experiences with keys: 'obs', 'actions', 'rewards'

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    predictions = model.predict(batch['obs'], batch['actions'])
    loss = F.mse_loss(predictions, batch['next_obs'])
    return loss
```

**Style Guidelines:**
- Type hints always
- NumPy-style docstrings
- PEP 8 compliant
- Meaningful variable names

#### File Organization

```
Always put:
- Model implementations → src/models/
- Training code → src/training/
- Evaluation code → src/evaluation/
- Experiments → experiments/scripts/
- Analysis → analysis/scripts/ or analysis/notebooks/
- Documentation → docs/
```

---

## 🚨 Important Constraints & Considerations

### Technical Constraints

1. **No Quantum Hardware**
   - All implementations must run on classical hardware
   - No Qiskit, Cirq, or quantum simulators
   - Quantum-INSPIRED only

2. **Classical Implementation**
   - Python 3.8+
   - PyTorch (not TensorFlow)
   - Gymnasium (not old Gym)
   - DMControl Suite

3. **Computational Resources**
   - Must be feasible on single GPU
   - Training time < 2 weeks per approach
   - No distributed training assumptions

### Scope Constraints

**IN SCOPE:**
- ✅ World model training optimization
- ✅ Multiple quantum-inspired approaches
- ✅ Standard RL environments
- ✅ Statistical comparison
- ✅ Ablation studies

**OUT OF SCOPE:**
- ❌ Policy learning optimization
- ❌ Real quantum hardware
- ❌ Entirely new architectures from scratch
- ❌ Real-world robotics deployment
- ❌ Production system development

### Timeline Constraints

- **Start:** October 30, 2025
- **End:** January 31, 2026
- **Total:** 13 weeks
- **Non-negotiable deadline:** Cannot extend beyond Jan 31

### Academic Constraints

- Must meet BITS dissertation standards
- Supervisor approval required
- Defense presentation required
- Proper citation and references mandatory

---

## 📂 Repository Structure Guide

### Critical Directories

```
src/models/          → All world model implementations go here
src/training/        → Training loops and procedures
src/evaluation/      → Metrics and evaluation code
experiments/         → Experiment configs and scripts
analysis/            → Jupyter notebooks and analysis scripts
docs/                → All documentation
dissertation/        → Dissertation document files
```

### File Naming Conventions

```
Models: {approach_name}.py
  ✓ baseline.py
  ✓ qaoa_enhanced.py
  ✓ superposition.py

Scripts: {action}_{target}.py
  ✓ train_baseline.py
  ✓ evaluate_model.py
  ✓ compare_approaches.py

Configs: {approach}_{environment}.yaml
  ✓ baseline_walker.yaml
  ✓ qaoa_cheetah.yaml
```

---

## 🔍 Research Novelty

### What Makes This Novel?

1. **Unexplored Intersection**
   - No prior work on quantum-inspired world model training
   - Bridges quantum computing and practical RL

2. **Systematic Comparison**
   - Multiple quantum principles tested
   - Fair evaluation with statistical rigor

3. **Practical Focus**
   - Classical hardware (immediately accessible)
   - Deployable in real applications TODAY

4. **Timely**
   - Builds on 2024-2025 advances
   - Positioned at intersection of advancing fields

### Research Gap

**Current State:**
- Quantum-inspired methods → used for policy optimization
- World model research → uses only classical training

**This Dissertation:**
- Quantum-inspired methods → applied to world model training ✨
- Systematic evaluation of which principles transfer

### Why This Matters

1. **Practical Impact:** If quantum-inspired methods help, practitioners can use them immediately
2. **Theoretical Understanding:** Reveals which quantum principles transfer to classical ML
3. **Negative Results Valuable:** Even if methods don't help, we learn when/why they fail

---

## 🤝 How AI Assistants Can Help

### Effective Ways to Assist

#### Code Development
- Generate boilerplate for model implementations
- Write unit tests
- Suggest optimizations
- Debug errors

#### Experimentation
- Create experiment configs
- Design ablation studies
- Suggest hyperparameters
- Write evaluation scripts

#### Analysis
- Generate plotting code
- Perform statistical tests
- Create comparison tables
- Interpret results

#### Documentation
- Write docstrings
- Create README sections
- Generate API documentation
- Explain complex concepts

#### Literature
- Summarize papers
- Identify related work
- Suggest relevant citations
- Compare approaches

### What AI Should NOT Do

❌ **Don't:**
- Suggest using actual quantum hardware
- Recommend out-of-scope extensions
- Propose unrealistic timelines
- Generate fake results or data
- Plagiarize existing code without attribution

✅ **Do:**
- Provide realistic, feasible suggestions
- Respect the 13-week timeline
- Focus on classical implementations
- Emphasize reproducibility
- Cite sources appropriately

---

## 📝 Progress Tracking

### Current Status (As of Nov 11, 2025)

**Phase:** Literature Review & Baseline Implementation  
**Progress:** 🟡 In Progress (Week 2 of 13)

**Completed:**
- ✅ Abstract report approved
- ✅ Repository structure planned
- ✅ README and CLAUDE.md created
- ✅ Literature survey initiated

**In Progress:**
- 🟡 Literature review
- 🟡 Development environment setup
- 🟡 Baseline implementation

**Upcoming:**
- ⚪ Complete baseline validation
- ⚪ Begin QAOA implementation
- ⚪ Set up experiment framework

### Weekly Progress Log

Progress should be logged in `/docs/progress-log.md` with:
- Date
- Accomplishments
- Challenges
- Next steps
- Timeline adjustments (if any)

---

## 🎯 Dissertation Chapters

### Planned Structure

1. **Introduction**
   - Motivation
   - Problem statement
   - Research questions
   - Contributions
   - Dissertation organization

2. **Literature Review**
   - World models in RL
   - Quantum-inspired algorithms
   - QAOA and optimization
   - Related work

3. **Background**
   - Reinforcement learning basics
   - Model-based RL
   - Quantum computing principles
   - Neural network optimization

4. **Methodology**
   - Research design
   - Implementation approaches
   - Evaluation strategy
   - Statistical methods

5. **Implementation**
   - System architecture
   - World model design
   - Quantum-inspired adaptations
   - Technical details

6. **Experiments**
   - Experimental setup
   - Environments
   - Hyperparameters
   - Ablation studies

7. **Results**
   - Performance comparisons
   - Statistical analysis
   - Ablation findings
   - Computational costs

8. **Discussion**
   - Interpretation of results
   - When quantum methods help
   - Limitations
   - Implications

9. **Conclusion**
   - Summary
   - Contributions
   - Future work
   - Final remarks

10. **Appendices**
    - Hyperparameter tables
    - Additional figures
    - Code snippets
    - Detailed results

---

## 🧪 Experiment Tracking

### Experiments to Run

#### Phase 1: Individual Method Validation (CartPole + Pendulum)
| ID | Name | Approaches | Environments | Seeds | Status |
|----|------|-----------|--------------|-------|--------|
| E1.1 | Baseline Validation | Baseline | CartPole, Pendulum | 5 | ⚪ Pending |
| E1.2 | QAOA Validation | QAOA | CartPole, Pendulum | 5 | ⚪ Pending |
| E1.3 | Superposition Validation | Superposition | CartPole, Pendulum | 5 | ⚪ Pending |
| E1.4 | Gates Validation | Gates | CartPole, Pendulum | 5 | ⚪ Pending |
| E1.5 | Error Correction Validation | Error Correction | CartPole, Pendulum | 5 | ⚪ Pending |
| E1.6 | Integrated Validation | Integrated | CartPole, Pendulum | 5 | ⚪ Pending |

#### Phase 2: DMControl Suite (Primary Benchmarks)
| ID | Name | Approaches | Environments | Seeds | Status |
|----|------|-----------|--------------|-------|--------|
| E2.1 | Walker Comparison | All 6 | Walker-walk | 10 | ⚪ Pending |
| E2.2 | Cheetah Comparison | All 6 | Cheetah-run | 10 | ⚪ Pending |
| E2.3 | Reacher Easy Comparison | All 6 | Reacher-easy | 10 | ⚪ Pending |
| E2.4 | Reacher Hard Comparison | All 6 | Reacher-hard | 10 | ⚪ Pending |

#### Phase 3: Atari (Visual Benchmarks)
| ID | Name | Approaches | Environments | Seeds | Status |
|----|------|-----------|--------------|-------|--------|
| E3.1 | Pong Comparison | All 6 | Pong | 5 | ⚪ Pending |
| E3.2 | Breakout Comparison | All 6 | Breakout | 5 | ⚪ Pending |

#### Phase 4: Analysis & Ablation
| ID | Name | Approaches | Environments | Seeds | Status |
|----|------|-----------|--------------|-------|--------|
| E4.1 | Ablation Study | Integrated variants | Walker, Cheetah | 5 | ⚪ Pending |
| E4.2 | Robustness Testing | All 6 | Walker (noisy) | 5 | ⚪ Pending |
| E4.3 | Computational Cost | All 6 | Walker | 3 | ⚪ Pending |

### Total Experiment Summary
| Category | Environments | Approaches | Seeds | Total Runs |
|----------|--------------|------------|-------|------------|
| Validation | 2 | 6 | 5 | 60 |
| DMControl | 4 | 6 | 10 | 240 |
| Atari | 2 | 6 | 5 | 60 |
| Analysis | 2 | 6 | 5 | 60 |
| **TOTAL** | **10** | **6** | - | **~420 runs** |

### Experiment Checklist

For each experiment:
- [ ] Configuration file created
- [ ] Script written and tested
- [ ] Baseline run completed
- [ ] All seeds executed
- [ ] Results logged
- [ ] Figures generated
- [ ] Statistical tests performed
- [ ] Findings documented

---

## 🔗 Useful Links

### Documentation
- [PyTorch Docs](https://pytorch.org/docs/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [DMControl Docs](https://github.com/deepmind/dm_control)
- [NumPy Docs](https://numpy.org/doc/)

### Papers (Primary)
- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
- [QAOA Paper](https://arxiv.org/abs/1411.4028)
- [Quantum-Inspired RL](https://arxiv.org/abs/2101.02034)

### Tools
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Weights & Biases](https://wandb.ai/)
- [Matplotlib](https://matplotlib.org/)

### Institution
- [BITS Pilani](https://www.bits-pilani.ac.in/)
- [WILP Division](https://www.bits-pilani.ac.in/wilp/)
- [IN-SPACe](https://www.inspace.gov.in/)

---

## 💭 Common Pitfalls to Avoid

### Technical Pitfalls

1. **Forgetting Seeds**
   - Always set random seeds for reproducibility
   - Run multiple seeds, not just one

2. **Ignoring Hyperparameters**
   - Document all hyperparameters
   - Use consistent settings for fair comparison

3. **Not Testing Early**
   - Test on simple environments first (CartPole)
   - Then scale to complex ones

4. **Poor Logging**
   - Log all metrics, not just final performance
   - Save checkpoints regularly

5. **No Validation Set**
   - Always have train/validation/test split
   - Don't overfit to test environments

### Research Pitfalls

1. **Cherry-Picking Results**
   - Report all results, not just good ones
   - Negative results are valuable

2. **Insufficient Statistical Testing**
   - Use proper significance tests
   - Report confidence intervals

3. **Scope Creep**
   - Stick to original plan
   - Resist adding too many extensions

4. **Poor Time Management**
   - Start experiments early
   - Budget time for failed experiments

5. **Weak Documentation**
   - Document as you go, not at the end
   - Explain all design decisions

### Writing Pitfalls

1. **Vague Claims**
   - Be specific with numbers
   - Quantify improvements

2. **Missing Related Work**
   - Cite relevant papers
   - Compare to baselines

3. **No Limitations Section**
   - Acknowledge limitations
   - Discuss threats to validity

4. **Poor Figure Quality**
   - High-resolution figures
   - Clear labels and legends

5. **Last-Minute Writing**
   - Write continuously
   - Don't wait until the end

---

## 🎓 Dissertation Defense Tips

### Anticipated Questions

1. **"Why quantum-inspired instead of quantum?"**
   - Answer: Practical, accessible, implementable today

2. **"What if results are negative?"**
   - Answer: Still valuable—tells us when/why methods don't work

3. **"How do you ensure fair comparison?"**
   - Answer: Same architecture, hyperparameters, multiple seeds, statistical tests

4. **"What's the practical impact?"**
   - Answer: If methods work, practitioners can use immediately; if not, we know when to avoid them

5. **"What about other quantum principles?"**
   - Answer: Focused on most promising ones; future work can explore others

### Defense Preparation

**Create:**
- 20-minute presentation
- Backup slides with extra details
- Demo videos of trained agents
- Poster summarizing key findings

**Practice:**
- Explain to non-experts
- Handle tough questions
- Time management (stay within limits)

---

## 📊 Key Metrics Summary

### For Quick Reference

| Metric | Definition | Target | Importance |
|--------|-----------|--------|------------|
| **Sample Efficiency** | Steps to reach 95% performance | Lower is better | PRIMARY |
| **Training Speed** | Wall-clock time to convergence | Faster is better | PRIMARY |
| **Prediction Accuracy** | MSE on test trajectories | Lower is better | PRIMARY |
| **Final Performance** | Average return (100 episodes) | Higher is better | PRIMARY |
| **Stability** | Std dev across seeds | Lower is better | SECONDARY |
| **Robustness** | Performance under noise | Less degradation is better | SECONDARY |
| **FLOPs** | Computational operations | Lower is better (if same performance) | SECONDARY |

---

## 🎯 Final Checklist (Before Submission)

### Code & Experiments

- [ ] All 5-6 approaches implemented
- [ ] Unit tests passing
- [ ] Experiments completed (all seeds)
- [ ] Results logged and backed up
- [ ] Code well-documented
- [ ] Repository clean and organized

### Analysis

- [ ] Statistical tests performed
- [ ] All figures generated
- [ ] All tables created
- [ ] Results interpreted
- [ ] Ablation studies completed

### Documentation

- [ ] All chapters written
- [ ] Figures included with captions
- [ ] Tables formatted properly
- [ ] References cited correctly
- [ ] Appendices complete
- [ ] Abstract written
- [ ] Proofread thoroughly

### Submission

- [ ] Supervisor approval received
- [ ] Examiner feedback incorporated
- [ ] Format per BITS guidelines
- [ ] PDF generated
- [ ] Plagiarism check passed
- [ ] Defense presentation ready

---

## 🔄 Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Nov 11, 2025 | Initial CLAUDE.md creation | Saurabh Jalendra |
| 1.1 | Dec 01, 2025 | Updated: All environments now REQUIRED (Reacher, Atari included), Integrated approach now REQUIRED, Comprehensive experiment plan with ~420 runs | Saurabh Jalendra |
| 1.2 | Dec 17, 2025 | Added: Standard Configuration section with architecture params (stoch=64, deter=512, hidden=512), training params (batch=32, seq=20, lr=3e-4), standard seeds [42,123,456,789,1024], required notebook components (multi-seed, test set, long-horizon) | Saurabh Jalendra |

---

## 🙋 FAQ for AI Assistants

### General Questions

**Q: What's the student's background?**
A: MTech AI/ML student at BITS Pilani, working at S K Jalendra Marketing Services.

**Q: What's the main goal?**
A: Compare quantum-inspired vs classical methods for world model training.

**Q: What's the timeline?**
A: 13 weeks (Oct 30, 2025 - Jan 31, 2026).

### Technical Questions

**Q: What framework to use?**
A: PyTorch (not TensorFlow).

**Q: What RL library?**
A: Gymnasium (not old Gym).

**Q: Can we use TensorFlow?**
A: No, stick to PyTorch for consistency.

**Q: Should we implement DreamerV3 from scratch?**
A: No, can use existing implementations and modify training procedures.

### Research Questions

**Q: What if quantum methods don't help?**
A: Still valuable! Negative results are publishable.

**Q: How many environments minimum?**
A: ALL 10 environments are required:
- Simple: CartPole, Pendulum (2)
- DMControl: Walker, Cheetah, Reacher-easy, Reacher-hard (4)
- Atari: Pong, Breakout (2)

**Q: How many seeds?**
A: Minimum 5, ideally 10 for final comparison.

**Q: What if we run out of time?**
A: Reduce to 3-4 approaches instead of 6.

### Configuration Questions

**Q: What architecture dimensions should I use?**
A: ALWAYS use stoch_dim=64, deter_dim=512, hidden_dim=512, all hidden layers [512, 512]. See "Standard Configuration" section.

**Q: What training parameters are standard?**
A: batch_size=32, seq_len=20, num_steps=10000, lr=3e-4, kl_weight=1.0. See "Standard Configuration" section.

**Q: What seeds should experiments use?**
A: [42, 123, 456, 789, 1024] - these 5 seeds are REQUIRED for all multi-seed experiments.

**Q: What components must every notebook have?**
A: The Experiments section must include: Multi-Seed Experiments, Test Set Evaluation, and Long-Horizon Prediction Test.

**Q: Why is consistency so important?**
A: Fair comparison requires identical architecture. The ONLY difference between approaches should be the quantum-inspired enhancement, not architecture variations.

---

## 📬 Contact for Clarifications

If you (AI assistant) encounter ambiguity:

1. **Check this file first** - Most answers here
2. **Check README.md** - Additional details
3. **Check docs/** - Specific documentation
4. **Ask the student** - Last resort

---

## 🔐 Confidentiality Note

**This is a proprietary dissertation project.**

- All code and findings belong to BITS Pilani and S K Jalendra Marketing Services
- Do not share code or results publicly without permission
- Respect academic integrity

---

## ✨ Final Note

**This project is feasible, novel, and valuable.**

Even if quantum-inspired methods don't outperform classical approaches, understanding WHEN and WHY they work (or don't) is a significant contribution to the field.

**Success is NOT about beating state-of-the-art. Success is about rigorous comparison and clear insights.**

---

**Last Updated:** December 17, 2025
**Status:** Active Development - Phase 2 (Quantum-Inspired Development)
**Next Review:** December 15, 2025

---

<div align="center">

**🤖 Built for AI assistants to provide better context-aware help 🤖**

**Questions? Check README.md or ask Saurabh Jalendra**

</div>
