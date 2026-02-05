# DISSERTATION TABLES - READY FOR INSERTION
## Quantum-Enhanced Simulation Learning for Reinforcement Learning

**Author:** Saurabh Jalendra (BITS ID: 2023AC05912)
**Generated:** February 5, 2026
**Format:** Markdown (easily convertible to LaTeX or Word)

---

## TABLE 1: Experimental Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| stoch_dim | 64 | Stochastic state dimension |
| deter_dim | 512 | Deterministic state dimension |
| hidden_dim | 512 | Hidden layer dimension |
| batch_size | 32 | Training batch size |
| seq_len | 20 | Sequence length for BPTT |
| learning_rate | 3×10⁻⁴ | AdamW learning rate |
| kl_weight | 1.0 | KL divergence loss weight |
| grad_clip | 100.0 | Gradient clipping norm |
| num_steps | 10,000 | Training iterations |
| num_seeds | 5 | Seeds: [42, 123, 456, 789, 1024] |

---

## TABLE 2: Environments Evaluated

| Environment | Domain | Obs Dim | Act Dim | Type | Episodes |
|-------------|--------|---------|---------|------|----------|
| CartPole-v1 | Classic | 4 | 2 | Discrete | 100 |
| Walker-walk | DMControl | 24 | 6 | Continuous | 100 |
| Cheetah-run | DMControl | 17 | 6 | Continuous | 200 |
| Reacher-easy | DMControl | 6 | 2 | Continuous | 200 |
| Pong | Atari | 84×84 | 6 | Visual/Discrete | 50 |
| Breakout | Atari | 84×84 | 4 | Visual/Discrete | 50 |

---

## TABLE 3: Methods Evaluated

| Method | Description | Parameters | Training Overhead |
|--------|-------------|------------|-------------------|
| Baseline | Standard RSSM world model | 4.7M | 1.0× |
| Quantum Tunneling | Noise injection for escaping local minima | 4.7M | 1.0-1.2× |
| Superposition | Interference-based experience replay | 4.7M | 0.9-1.0× |
| Entanglement | Correlated feature learning layers | 5.3M | 1.1-1.2× |
| Interference Ensemble | 5-model phase-weighted ensemble | 23.7M | 5.0-6.0× |

---

## TABLE 4: Phase 2 Results - Walker-walk

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 1.799 ± 0.060 | — | — | — |
| Quantum Tunneling | 1.797 ± 0.030 | +0.1% | 0.841 | 0.03 |
| Superposition | 4.645 ± 0.198 | **−158%** | <0.008 | −17.65 |
| Entanglement | 1.798 ± 0.032 | +0.1% | 0.912 | 0.03 |
| **Interference Ensemble** | **1.022 ± 0.013** | **+43.2%** | **<0.008** | **18.13** |

---

## TABLE 5: Phase 2 Results - Cheetah-run

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 0.573 ± 0.009 | — | — | — |
| Quantum Tunneling | 0.578 ± 0.005 | −0.9% | 0.421 | −0.72 |
| Superposition | 2.858 ± 0.062 | **−399%** | <0.008 | −36.71 |
| Entanglement | 0.575 ± 0.007 | −0.3% | 0.754 | −0.22 |
| **Interference Ensemble** | **0.367 ± 0.007** | **+35.9%** | **<0.008** | **26.15** |

---

## TABLE 6: Phase 2 Results - Reacher-easy

| Approach | Test MSE (mean ± std) | Δ vs Baseline | p-value | Cohen's d |
|----------|----------------------|---------------|---------|-----------|
| Baseline | 0.125 ± 0.005 | — | — | — |
| Quantum Tunneling | 0.134 ± 0.006 | −7.2% | 0.095 | −1.66 |
| Superposition | 0.915 ± 0.015 | **−630%** | <0.008 | −72.26 |
| Entanglement | 0.130 ± 0.006 | −3.5% | 0.312 | −0.77 |
| **Interference Ensemble** | **0.069 ± 0.004** | **+45.0%** | **<0.008** | **12.76** |

---

## TABLE 7: Phase 3 Results - Pong

| Approach | Test MSE (×10⁻⁴) | Δ vs Baseline | p-value | Significant? |
|----------|------------------|---------------|---------|--------------|
| Baseline | 2.93 ± 0.13 | — | — | — |
| Quantum Tunneling | 2.86 ± 0.02 | +2.2% | 0.087 | No |
| Superposition | 2.88 ± 0.12 | +1.7% | 0.234 | No |
| Entanglement | 3.01 ± 0.11 | −2.9% | 0.178 | No |
| Interference Ensemble | — | FAILED | — | — |

---

## TABLE 8: Phase 3 Results - Breakout

| Approach | Test MSE (×10⁻⁴) | Δ vs Baseline | p-value | Significant? |
|----------|------------------|---------------|---------|--------------|
| Baseline | 5.39 ± 0.18 | — | — | — |
| Quantum Tunneling | 5.39 ± 0.23 | −0.1% | 0.921 | No |
| Superposition | 5.31 ± 0.13 | +1.4% | 0.312 | No |
| Entanglement | 5.57 ± 0.18 | −3.4% | 0.089 | No |
| Interference Ensemble | — | FAILED | — | — |

---

## TABLE 9: Summary of Statistically Significant Results

| Environment | Significant Improvement | Significant Degradation |
|-------------|------------------------|------------------------|
| Walker-walk | Interference Ensemble (+43.2%) | Superposition (−158%) |
| Cheetah-run | Interference Ensemble (+35.9%) | Superposition (−399%) |
| Reacher-easy | Interference Ensemble (+45.0%) | Superposition (−630%) |
| Pong | None | None |
| Breakout | None | None |
| **Total** | **3 significant improvements** | **3 significant degradations** |

---

## TABLE 10: Method Ranking Summary

| Rank | Method | DMControl Avg | Atari Avg | Overall Verdict |
|------|--------|---------------|-----------|-----------------|
| 1 | **Interference Ensemble** | **+41.4%** | FAILED | **Best (DMControl)** |
| 2 | Baseline | 0% (ref) | 0% (ref) | Reference |
| 3 | Quantum Tunneling | −2.7% | +0.9% | Marginal |
| 4 | Entanglement | −1.2% | −3.2% | No benefit |
| 5 | Superposition | **−396%** | +1.5% | **Avoid on DMControl** |

---

## TABLE 11: Computational Cost Comparison

| Method | Parameters | Training Time | Memory | Cost-Effective? |
|--------|------------|---------------|--------|-----------------|
| Baseline | 4.7M (1.0×) | 1.0× | 1.0× | Reference |
| Quantum Tunneling | 4.7M (1.0×) | 1.0-1.2× | 1.0× | Neutral |
| Superposition | 4.7M (1.0×) | 0.9-1.0× | 1.2× | **No (harmful)** |
| Entanglement | 5.3M (1.1×) | 1.1-1.2× | 1.1× | No |
| Interference Ensemble | 23.7M (5.0×) | 5.0-6.0× | 5.0× | **Yes (if accuracy critical)** |

---

## TABLE 12: Research Question Answers

| Question | Answer | Evidence |
|----------|--------|----------|
| Do QI methods improve world model training? | **Conditionally Yes** | IE: +35-45% on DMControl |
| Which QI principles transfer effectively? | Interference > Tunneling > Superposition ≈ Entanglement | Effect sizes |
| What is the cost-benefit tradeoff? | IE: 5× cost for +40% gain | Time vs accuracy |
| Are improvements consistent? | IE: Yes (DMControl). SP: Consistently bad. | All 3 environments |

---

## TABLE 13: Long-Horizon Prediction Accuracy (Walker-walk, Horizon=20)

| Approach | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1024 | Mean |
|----------|---------|----------|----------|----------|-----------|------|
| Baseline | 1.701 | 1.736 | 1.709 | 1.829 | 1.809 | 1.757 |
| Quantum Tunneling | 1.833 | 1.691 | 1.762 | 1.710 | 1.837 | 1.767 |
| Superposition | 4.276 | 4.210 | 5.106 | 4.573 | 4.709 | 4.575 |
| Entanglement | 1.847 | 1.775 | 1.662 | 1.797 | 1.760 | 1.768 |
| **Interference Ensemble** | **1.039** | **1.013** | **0.984** | **0.994** | **1.008** | **1.008** |

---

## TABLE 14: Confidence Intervals (95%)

| Method | Environment | Mean | 95% CI Lower | 95% CI Upper |
|--------|-------------|------|--------------|--------------|
| Baseline | Walker | 1.799 | 1.739 | 1.859 |
| Interference Ensemble | Walker | 1.022 | 1.009 | 1.036 |
| Baseline | Cheetah | 0.573 | 0.565 | 0.582 |
| Interference Ensemble | Cheetah | 0.367 | 0.360 | 0.374 |
| Baseline | Reacher | 0.125 | 0.121 | 0.130 |
| Interference Ensemble | Reacher | 0.069 | 0.065 | 0.073 |

---

## TABLE 15: Effect Size Interpretation Guide

| |d| Range | Interpretation | Examples from This Study |
|-----------|----------------|------------------------------|
| 0.0 - 0.2 | Negligible | QT on Walker (d=0.03) |
| 0.2 - 0.5 | Small | EN on Cheetah (d=−0.22) |
| 0.5 - 0.8 | Medium | QT on Cheetah (d=−0.72) |
| 0.8 - 1.2 | Large | EN on Breakout (d=−0.99) |
| > 1.2 | Very Large | **IE on Walker (d=18.13)** |

---

## LATEX VERSIONS

### Table 4 in LaTeX:

```latex
\begin{table}[htbp]
\centering
\caption{Walker-walk Results (Test Observation MSE)}
\label{tab:walker_results}
\begin{tabular}{lcccc}
\toprule
Approach & Test MSE & $\Delta$ vs Baseline & p-value & Cohen's d \\
\midrule
Baseline & $1.799 \pm 0.060$ & -- & -- & -- \\
Quantum Tunneling & $1.797 \pm 0.030$ & +0.1\% & 0.841 & 0.03 \\
Superposition & $4.645 \pm 0.198$ & \textbf{-158\%} & <0.008 & -17.65 \\
Entanglement & $1.798 \pm 0.032$ & +0.1\% & 0.912 & 0.03 \\
\textbf{Interference Ensemble} & $\mathbf{1.022 \pm 0.013}$ & \textbf{+43.2\%} & \textbf{<0.008} & \textbf{18.13} \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 9 in LaTeX:

```latex
\begin{table}[htbp]
\centering
\caption{Summary of Statistically Significant Results}
\label{tab:significant_summary}
\begin{tabular}{lcc}
\toprule
Environment & Significant Improvement & Significant Degradation \\
\midrule
Walker-walk & Interference Ensemble (+43.2\%) & Superposition (-158\%) \\
Cheetah-run & Interference Ensemble (+35.9\%) & Superposition (-399\%) \\
Reacher-easy & Interference Ensemble (+45.0\%) & Superposition (-630\%) \\
Pong & None & None \\
Breakout & None & None \\
\midrule
\textbf{Total} & \textbf{3 improvements} & \textbf{3 degradations} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## FIGURE CAPTIONS

### Figure 1: Learning Curves Comparison
**Caption:** Training loss curves for all five approaches on Walker-walk environment. Interference Ensemble (green) achieves the lowest final loss, while Superposition (red) fails to converge to competitive levels. Shaded regions indicate standard error across 5 seeds.

### Figure 2: Test MSE Comparison (Bar Chart)
**Caption:** Test observation MSE comparison across all environments. Error bars indicate 95% confidence intervals. Interference Ensemble significantly outperforms baseline on all DMControl tasks (p < 0.008), while Superposition shows significantly worse performance.

### Figure 3: Long-Horizon Prediction Accuracy
**Caption:** Prediction error at increasing horizon lengths (5, 10, 15, 20 steps). Interference Ensemble maintains lowest error across all horizons, demonstrating superior long-term prediction capability.

### Figure 4: Method Ranking Summary
**Caption:** Radar chart showing relative performance of each method across different metrics (prediction accuracy, training stability, computational efficiency). Interference Ensemble excels in accuracy but requires more resources.

---

*Tables document completed: February 5, 2026*
*All values verified against raw experimental data*
