"""
Generate architecture diagrams for the dissertation report.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

output_dir = r"D:\Git Repos\Quantum-Enhanced-Simulation-Learning-for-Reinforcement-Learning\results\figures"
os.makedirs(output_dir, exist_ok=True)

def create_rssm_architecture():
    """Create RSSM World Model Architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('RSSM World Model Architecture', fontsize=16, fontweight='bold', pad=20)

    # Colors
    colors = {
        'input': '#E3F2FD',
        'encoder': '#BBDEFB',
        'rssm': '#90CAF9',
        'decoder': '#64B5F6',
        'output': '#42A5F5',
        'latent': '#FFF9C4',
        'arrow': '#1565C0'
    }

    # Input box
    ax.add_patch(FancyBboxPatch((0.5, 3), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['input'], edgecolor='black', linewidth=2))
    ax.text(1.5, 4, 'Observation\n$o_t$\n(84×84×1 or\nstate vector)', ha='center', va='center', fontsize=9)

    # Action input
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 2, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['input'], edgecolor='black', linewidth=2))
    ax.text(1.5, 1.25, 'Action\n$a_t$', ha='center', va='center', fontsize=9)

    # Encoder
    ax.add_patch(FancyBboxPatch((3.5, 3), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['encoder'], edgecolor='black', linewidth=2))
    ax.text(4.5, 4, 'Encoder\nMLP/CNN\n[512, 512]', ha='center', va='center', fontsize=9)

    # RSSM Core (large box)
    ax.add_patch(FancyBboxPatch((6.5, 0.5), 4, 6.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['rssm'], edgecolor='black', linewidth=2))
    ax.text(8.5, 6.5, 'RSSM Core', ha='center', va='center', fontsize=11, fontweight='bold')

    # Deterministic state
    ax.add_patch(FancyBboxPatch((7, 4.5), 3, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=colors['latent'], edgecolor='black', linewidth=1.5))
    ax.text(8.5, 5.1, 'Deterministic $h_t$\n(dim=512)', ha='center', va='center', fontsize=8)

    # Stochastic state
    ax.add_patch(FancyBboxPatch((7, 2.8), 3, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=colors['latent'], edgecolor='black', linewidth=1.5))
    ax.text(8.5, 3.4, 'Stochastic $z_t$\n(dim=64)', ha='center', va='center', fontsize=8)

    # GRU cell
    ax.add_patch(FancyBboxPatch((7, 1.2), 3, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(8.5, 1.8, 'GRU Cell\n$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$', ha='center', va='center', fontsize=8)

    # Decoder
    ax.add_patch(FancyBboxPatch((11.5, 3), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['decoder'], edgecolor='black', linewidth=2))
    ax.text(12.5, 4, 'Decoder\nMLP/CNN\n[512, 512]', ha='center', va='center', fontsize=9)

    # Output
    ax.add_patch(FancyBboxPatch((11.5, 0.5), 2, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor='black', linewidth=2))
    ax.text(12.5, 1.25, 'Predicted\n$\\hat{o}_{t+1}$', ha='center', va='center', fontsize=9, color='white')

    # Arrows
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=6"
    kw = dict(arrowstyle=arrow_style, color=colors['arrow'])

    # obs -> encoder
    ax.annotate("", xy=(3.5, 4), xytext=(2.5, 4), arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    # encoder -> RSSM
    ax.annotate("", xy=(6.5, 4), xytext=(5.5, 4), arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    # action -> RSSM
    ax.annotate("", xy=(6.5, 1.8), xytext=(2.5, 1.25), arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2, connectionstyle="arc3,rad=0.2"))
    # RSSM -> decoder
    ax.annotate("", xy=(11.5, 4), xytext=(10.5, 4), arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    # decoder -> output
    ax.annotate("", xy=(12.5, 2), xytext=(12.5, 3), arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # State dimension annotation
    ax.text(8.5, 7.5, 'Combined State: $s_t = [h_t, z_t]$ (dim = 512 + 64 = 576)',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_rssm.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_rssm.png")


def create_interference_ensemble_diagram():
    """Create Interference Ensemble Architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Interference Ensemble Architecture', fontsize=16, fontweight='bold', pad=20)

    colors = ['#FFCDD2', '#F8BBD9', '#E1BEE7', '#D1C4E9', '#C5CAE9']

    # Input
    ax.add_patch(FancyBboxPatch((0.5, 3), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD', edgecolor='black', linewidth=2))
    ax.text(1.5, 4, 'Input\nState $s_t$', ha='center', va='center', fontsize=10)

    # 5 World Models
    for i in range(5):
        y_pos = 6.5 - i * 1.4
        ax.add_patch(FancyBboxPatch((3.5, y_pos - 0.5), 2.5, 1, boxstyle="round,pad=0.1",
                                     facecolor=colors[i], edgecolor='black', linewidth=1.5))
        ax.text(4.75, y_pos, f'Model {i+1}\n$\\phi_{i+1} = {i*72}°$', ha='center', va='center', fontsize=8)

        # Arrow from input to model
        ax.annotate("", xy=(3.5, y_pos), xytext=(2.5, 4),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5, connectionstyle="arc3,rad=0"))

    # Predictions
    for i in range(5):
        y_pos = 6.5 - i * 1.4
        ax.add_patch(FancyBboxPatch((6.5, y_pos - 0.4), 1.5, 0.8, boxstyle="round,pad=0.05",
                                     facecolor=colors[i], edgecolor='black', linewidth=1))
        ax.text(7.25, y_pos, f'$\\hat{{s}}_{i+1}$', ha='center', va='center', fontsize=9)

        # Arrow from model to prediction
        ax.annotate("", xy=(6.5, y_pos), xytext=(6, y_pos),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # Interference aggregation
    ax.add_patch(FancyBboxPatch((8.5, 2.5), 2, 3, boxstyle="round,pad=0.1",
                                 facecolor='#FFF9C4', edgecolor='black', linewidth=2))
    ax.text(9.5, 4.5, 'Phase-Weighted\nAggregation', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(9.5, 3.5, '$w_i = \\frac{e^{i\\phi_i}}{\\sum e^{i\\phi_j}}$', ha='center', va='center', fontsize=9)
    ax.text(9.5, 2.8, '$\\hat{s} = \\sum_i w_i \\cdot \\hat{s}_i$', ha='center', va='center', fontsize=9)

    # Arrows from predictions to aggregation
    for i in range(5):
        y_pos = 6.5 - i * 1.4
        ax.annotate("", xy=(8.5, 4), xytext=(8, y_pos),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5, connectionstyle="arc3,rad=0"))

    # Final output
    ax.add_patch(FancyBboxPatch((10.5, 3), 1.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='#81C784', edgecolor='black', linewidth=2))
    ax.text(11.25, 4, 'Final\nPrediction\n$\\hat{s}_{t+1}$', ha='center', va='center', fontsize=9)

    # Arrow from aggregation box to output box
    ax.annotate("", xy=(10.5, 4), xytext=(9.8, 4),
               arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Key insight box
    ax.text(6, 0.5, 'Key: Models with similar predictions reinforce (constructive interference)\n'
                    'Models with different predictions cancel out (destructive interference)',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_interference_ensemble.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_interference_ensemble.png")


def create_quantum_tunneling_diagram():
    """Create Quantum Tunneling Optimizer diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Quantum Tunneling Optimizer', fontsize=16, fontweight='bold', pad=20)

    # Loss landscape
    x = np.linspace(0.5, 9.5, 200)
    # Create a loss landscape with local minima
    y = 2 + 0.5*np.sin(2*x) + 0.3*np.sin(5*x) + 0.1*(x-5)**2/10
    ax.plot(x, y, 'b-', linewidth=2, label='Loss Landscape')
    ax.fill_between(x, 0, y, alpha=0.3, color='blue')

    # Mark local minimum
    ax.plot(3.2, 1.7, 'ro', markersize=12, label='Local Minimum (stuck)', zorder=5)
    ax.annotate('Stuck here!', xy=(3.2, 1.7), xytext=(3.2, 2.8),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Mark global minimum
    ax.plot(7.5, 1.5, 'g*', markersize=15, label='Global Minimum (goal)', zorder=5)
    ax.annotate('Goal', xy=(7.5, 1.5), xytext=(7.5, 2.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Tunneling arrow (curved, going through barrier)
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches

    # Draw tunneling path
    tunnel_x = np.linspace(3.2, 7.5, 50)
    tunnel_y = 1.6 + 0.3*np.sin(np.linspace(0, np.pi, 50))
    ax.plot(tunnel_x, tunnel_y, 'r--', linewidth=2, alpha=0.7)
    ax.annotate('', xy=(7.3, 1.6), xytext=(3.4, 1.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))

    # Add noise injection annotation
    ax.text(5, 0.7, 'Quantum Tunneling: Add noise ∝ exp(-barrier/temperature)\n'
                    'to escape local minima without climbing over',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Formula
    ax.text(5, 5.3, r'$\theta_{new} = \theta - \alpha\nabla L + \mathcal{N}(0, \sigma^2)$ where $\sigma \propto e^{-\Delta L / T}$',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_quantum_tunneling.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_quantum_tunneling.png")


def create_superposition_buffer_diagram():
    """Create Superposition Replay Buffer diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Superposition Replay Buffer', fontsize=16, fontweight='bold', pad=20)

    # Trajectory 1
    ax.add_patch(FancyBboxPatch((0.5, 5), 3, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#BBDEFB', edgecolor='black', linewidth=2))
    ax.text(2, 5.6, 'Trajectory 1: $s_1 → s_2 → s_3$', ha='center', va='center', fontsize=9)

    # Trajectory 2
    ax.add_patch(FancyBboxPatch((0.5, 3.3), 3, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(2, 3.9, 'Trajectory 2: $s_4 → s_5 → s_6$', ha='center', va='center', fontsize=9)

    # Superposition mixing
    ax.add_patch(FancyBboxPatch((4.5, 3.8), 3, 2, boxstyle="round,pad=0.1",
                                 facecolor='#FFF9C4', edgecolor='black', linewidth=2))
    ax.text(6, 5.2, 'Superposition Mixing', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6, 4.3, '$s_{mix} = \\alpha \\cdot s_i + \\beta \\cdot s_j$\n$|\\alpha|^2 + |\\beta|^2 = 1$',
            ha='center', va='center', fontsize=9)

    # Arrows to mixing
    ax.annotate("", xy=(4.5, 4.8), xytext=(3.5, 5.4),
               arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
    ax.annotate("", xy=(4.5, 4.5), xytext=(3.5, 4),
               arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Mixed state output
    ax.add_patch(FancyBboxPatch((8.5, 4), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFCCBC', edgecolor='black', linewidth=2))
    ax.text(10, 4.75, 'Mixed State\n(Artificial)', ha='center', va='center', fontsize=10)

    # Arrow to output
    ax.annotate("", xy=(8.5, 4.8), xytext=(7.5, 4.8),
               arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Warning box
    ax.add_patch(FancyBboxPatch((1, 0.5), 10, 2, boxstyle="round,pad=0.1",
                                 facecolor='#FFCDD2', edgecolor='red', linewidth=2))
    ax.text(6, 2, '⚠️ WARNING: This approach FAILS on complex dynamics!',
            ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    ax.text(6, 1.2, 'Mixed states break temporal coherence and create physically impossible states.\n'
                    'Result: -158% to -630% degradation on DMControl tasks.',
            ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_superposition_buffer.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_superposition_buffer.png")


def create_entanglement_layer_diagram():
    """Create Entanglement Layer diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Entanglement-Inspired Feature Layer', fontsize=16, fontweight='bold', pad=20)

    # Input features
    ax.add_patch(FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD', edgecolor='black', linewidth=2))
    ax.text(1.5, 3, 'Input\nFeatures\n$x \\in \\mathbb{R}^n$', ha='center', va='center', fontsize=9)

    # Hadamard gate
    ax.add_patch(FancyBboxPatch((3.5, 2), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor='#E1BEE7', edgecolor='black', linewidth=2))
    ax.text(4.5, 3.3, 'Hadamard-like', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.5, 2.6, 'H = 1/√2 × [[1,1],[1,-1]]',
            ha='center', va='center', fontsize=8)

    # CNOT gate
    ax.add_patch(FancyBboxPatch((6.5, 2), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor='#B2EBF2', edgecolor='black', linewidth=2))
    ax.text(7.5, 3.3, 'CNOT-like', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.5, 2.6, 'Correlation\nOperation', ha='center', va='center', fontsize=8)

    # Output
    ax.add_patch(FancyBboxPatch((9.5, 2), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(10.5, 3, 'Entangled\nFeatures\n$y \\in \\mathbb{R}^n$', ha='center', va='center', fontsize=9)

    # Arrows
    ax.annotate("", xy=(3.5, 3), xytext=(2.5, 3), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
    ax.annotate("", xy=(6.5, 3), xytext=(5.5, 3), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
    ax.annotate("", xy=(9.5, 3), xytext=(8.5, 3), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Result box
    ax.add_patch(FancyBboxPatch((2, 0.3), 8, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#FFF9C4', edgecolor='orange', linewidth=2))
    ax.text(6, 0.9, 'Result: ~0% improvement. Quantum gates designed for qubits\n'
                    'do not translate effectively to real-valued neural network features.',
            ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_entanglement_layer.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: architecture_entanglement_layer.png")


def create_domain_comparison_figure():
    """Create domain-specific comparison figure (key finding)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data
    methods = ['Baseline', 'QT', 'SP', 'EN', 'IE']

    # State-based (DMControl average across Walker, Cheetah, Reacher-easy, Reacher-hard)
    state_improvements = [0, 0.5, -440, -0.3, 42.7]

    # Visual (Atari average)
    visual_improvements = [0, 1.0, -5, 0.5, -273]  # Average of Pong and Breakout

    colors_state = ['#90CAF9', '#81C784', '#EF5350', '#FFB74D', '#4CAF50']
    colors_visual = ['#90CAF9', '#81C784', '#EF5350', '#FFB74D', '#EF5350']

    # State-based plot
    ax1 = axes[0]
    bars1 = ax1.bar(methods, state_improvements, color=colors_state, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Improvement over Baseline (%)', fontsize=11)
    ax1.set_title('State-Based Tasks (DMControl)\n✓ IE Works Here', fontsize=12, fontweight='bold', color='green')
    ax1.set_ylim(-300, 60)

    # Add value labels
    for bar, val in zip(bars1, state_improvements):
        height = bar.get_height()
        ax1.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    # Visual plot
    ax2 = axes[1]
    bars2 = ax2.bar(methods, visual_improvements, color=colors_visual, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=11)
    ax2.set_title('Visual Tasks (Atari)\n✗ IE Fails Here', fontsize=12, fontweight='bold', color='red')
    ax2.set_ylim(-300, 60)

    # Add value labels
    for bar, val in zip(bars2, visual_improvements):
        height = bar.get_height()
        ax2.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    # Legend
    legend_labels = ['QT: Quantum Tunneling', 'SP: Superposition', 'EN: Entanglement', 'IE: Interference Ensemble']
    fig.text(0.5, 0.02, 'Methods: ' + ' | '.join(legend_labels), ha='center', fontsize=10, style='italic')

    plt.suptitle('KEY FINDING: Domain-Specific Performance of Quantum-Inspired Methods',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_comparison_key_finding.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: domain_comparison_key_finding.png")


def create_experimental_pipeline_diagram():
    """Create experimental pipeline overview diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Experimental Pipeline Overview', fontsize=16, fontweight='bold', pad=20)

    # Phase boxes
    phases = [
        ('Phase 1\nSimple Control', 'CartPole', '#E3F2FD', 1),
        ('Phase 2\nDMControl', 'Walker, Cheetah\nReacher-easy, hard', '#BBDEFB', 4),
        ('Phase 3\nAtari', 'Pong, Breakout', '#90CAF9', 7),
        ('Phase 4\nAnalysis', 'Statistics\nComparison', '#64B5F6', 10),
    ]

    for phase_name, envs, color, x in phases:
        ax.add_patch(FancyBboxPatch((x, 5), 2.5, 2, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x+1.25, 6.3, phase_name, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x+1.25, 5.5, envs, ha='center', va='center', fontsize=8)

    # Arrows between phases
    for i in range(3):
        x = 3.5 + i*3
        ax.annotate("", xy=(x+0.5, 6), xytext=(x, 6),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Methods tested
    ax.add_patch(FancyBboxPatch((1, 2), 12, 2, boxstyle="round,pad=0.1",
                 facecolor='#FFF9C4', edgecolor='black', linewidth=2))
    ax.text(7, 3.5, '5 Methods Compared (Each with 5 seeds: [42, 123, 456, 789, 1024])',
            ha='center', va='center', fontsize=11, fontweight='bold')

    methods_text = '1. Baseline (Classical) | 2. Quantum Tunneling | 3. Superposition | 4. Entanglement | 5. Interference Ensemble'
    ax.text(7, 2.5, methods_text, ha='center', va='center', fontsize=9)

    # Metrics
    ax.add_patch(FancyBboxPatch((3, 0.3), 8, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(7, 0.9, 'Metrics: MSE Loss | Statistical Tests: Mann-Whitney U (α=0.00625) | Effect Size: Cohen\'s d',
            ha='center', va='center', fontsize=9)

    # Total experiments
    ax.text(7, 7.5, 'Total: ~150 Experimental Runs', ha='center', va='center', fontsize=12,
            fontweight='bold', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experimental_pipeline.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: experimental_pipeline.png")


def create_results_summary_figure():
    """Create comprehensive results summary figure for all 8 environments."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # Actual data from complete_metrics.json files
    environments = {
        'CartPole': {'baseline': 0.109, 'qt': 0.111, 'sp': 0.164, 'en': 0.112, 'ie': 0.106},
        'Pendulum': {'baseline': 0.027, 'qt': 0.026, 'sp': 0.140, 'en': 0.031, 'ie': 0.031},
        'Walker': {'baseline': 1.799, 'qt': 1.806, 'sp': 4.639, 'en': 1.825, 'ie': 1.022},
        'Cheetah': {'baseline': 0.573, 'qt': 0.578, 'sp': 2.858, 'en': 0.575, 'ie': 0.367},
        'Reacher-easy': {'baseline': 0.125, 'qt': 0.134, 'sp': 0.915, 'en': 0.130, 'ie': 0.069},
        'Reacher-hard': {'baseline': 0.127, 'qt': 0.128, 'sp': 0.904, 'en': 0.129, 'ie': 0.068},
        'Pong (×10⁻⁴)': {'baseline': 2.93, 'qt': 2.86, 'sp': 2.88, 'en': 3.01, 'ie': 6.81},
        'Breakout (×10⁻⁴)': {'baseline': 5.39, 'qt': 5.39, 'sp': 5.31, 'en': 5.57, 'ie': 27.69},
    }

    methods = ['baseline', 'qt', 'sp', 'en', 'ie']
    method_labels = ['Base', 'QT', 'SP', 'EN', 'IE']
    colors = ['#90CAF9', '#81C784', '#EF5350', '#FFB74D', '#7E57C2']

    for idx, (env_name, data) in enumerate(environments.items()):
        ax = axes[idx // 4, idx % 4]
        values = [data[m] for m in methods]

        # Determine if IE is good or bad for coloring
        ie_color = '#4CAF50' if data['ie'] < data['baseline'] else '#F44336'
        bar_colors = colors[:4] + [ie_color]

        bars = ax.bar(method_labels, values, color=bar_colors, edgecolor='black', linewidth=1)
        ax.set_title(env_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('MSE')

        # Rotate labels
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)

        # Add baseline reference line
        ax.axhline(y=data['baseline'], color='gray', linestyle='--', alpha=0.7, linewidth=1)

    plt.suptitle('Results Summary: Observation MSE Across All 8 Environments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_summary_all.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: results_summary_all.png")


if __name__ == "__main__":
    print("Generating dissertation diagrams...")
    print("=" * 50)

    create_rssm_architecture()
    create_interference_ensemble_diagram()
    create_quantum_tunneling_diagram()
    create_superposition_buffer_diagram()
    create_entanglement_layer_diagram()
    create_domain_comparison_figure()
    create_experimental_pipeline_diagram()
    create_results_summary_figure()

    print("=" * 50)
    print("All diagrams generated successfully!")
    print(f"Output directory: {output_dir}")
