"""
Quantum Tunneling-Inspired Optimizer
====================================

Quantum Principle:
    In quantum mechanics, particles can "tunnel" through energy barriers
    that would be insurmountable in classical physics. This allows quantum
    systems to find global minima more efficiently.

Classical Translation:
    We add structured noise to the optimization process that:
    1. Is large early in training (high "tunneling rate") - allows exploration
    2. Decreases over time (annealing) - allows convergence
    3. Depends on the loss landscape - higher barriers get more tunneling attempts

This is related to but distinct from:
    - Simulated Annealing (temperature-based acceptance)
    - Noisy Gradient Descent (constant noise)
    - Learning Rate Schedules (only affects step size, not direction)

Key difference: Tunneling probability depends on being "stuck" (low gradient),
not just time or temperature.

References:
    - Kirkpatrick et al. (1983): Simulated Annealing
    - Finnila et al. (1994): Quantum Annealing
    - Kadowaki & Nishimori (1998): Quantum Annealing for Combinatorial Optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Iterator


class QuantumTunnelingOptimizer:
    """
    Quantum tunneling-inspired optimizer that wraps any base optimizer.

    The optimizer adds "tunneling" behavior: random perturbations that allow
    the optimization to escape local minima, with the perturbation strength
    decreasing over time (quantum annealing).

    Parameters
    ----------
    params : Iterator[nn.Parameter]
        Model parameters to optimize
    lr : float, default=3e-4
        Learning rate (passed to base optimizer)
    tunneling_strength : float, default=0.1
        Initial strength of tunneling perturbations
    annealing_rate : float, default=0.9995
        Rate at which tunneling strength decreases (per step)
    tunneling_frequency : int, default=10
        Apply tunneling every N steps (not every step)
    min_tunneling : float, default=1e-6
        Minimum tunneling strength (never goes below this)
    stuck_threshold : int, default=100
        If loss doesn't improve for this many steps, increase tunneling
    base_optimizer : str, default='adamw'
        Base optimizer to use ('adam', 'adamw', 'sgd')

    Example
    -------
    >>> model = RSSMWorldModel(obs_dim=4, action_dim=2)
    >>> optimizer = QuantumTunnelingOptimizer(model.parameters(), lr=3e-4)
    >>>
    >>> for batch in dataloader:
    ...     optimizer.zero_grad()
    ...     loss = compute_loss(model, batch)
    ...     loss.backward()
    ...     optimizer.step(loss.item())  # Pass loss value for adaptive tunneling
    """

    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 3e-4,
        tunneling_strength: float = 0.1,
        annealing_rate: float = 0.9995,
        tunneling_frequency: int = 10,
        min_tunneling: float = 1e-6,
        stuck_threshold: int = 100,
        base_optimizer: str = 'adamw'
    ):
        self.params = list(params)
        self.lr = lr
        self.tunneling_strength = tunneling_strength
        self.initial_tunneling = tunneling_strength
        self.annealing_rate = annealing_rate
        self.tunneling_frequency = tunneling_frequency
        self.min_tunneling = min_tunneling
        self.stuck_threshold = stuck_threshold

        # Create base optimizer
        if base_optimizer.lower() == 'adam':
            self.base_optimizer = torch.optim.Adam(self.params, lr=lr)
        elif base_optimizer.lower() == 'adamw':
            self.base_optimizer = torch.optim.AdamW(self.params, lr=lr)
        elif base_optimizer.lower() == 'sgd':
            self.base_optimizer = torch.optim.SGD(self.params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {base_optimizer}")

        # Tracking variables
        self.step_count = 0
        self.best_loss = float('inf')
        self.steps_since_improvement = 0
        self.loss_history = []
        self.tunneling_events = []  # Track when tunneling happened

    def zero_grad(self):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad()

    def step(self, current_loss: Optional[float] = None):
        """
        Perform optimization step with potential tunneling.

        Parameters
        ----------
        current_loss : float, optional
            Current loss value. If provided, enables adaptive tunneling
            that increases when optimization is "stuck".
        """
        # Standard gradient step
        self.base_optimizer.step()

        # Track loss for stuck detection
        if current_loss is not None:
            self.loss_history.append(current_loss)

            if current_loss < self.best_loss * 0.999:  # Meaningful improvement
                self.best_loss = current_loss
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1

        # Apply tunneling periodically
        if self.step_count % self.tunneling_frequency == 0:
            self._apply_tunneling(current_loss)

        # Anneal tunneling strength
        self.tunneling_strength = max(
            self.min_tunneling,
            self.tunneling_strength * self.annealing_rate
        )

        self.step_count += 1

    def _apply_tunneling(self, current_loss: Optional[float] = None):
        """
        Apply quantum tunneling perturbation to parameters.

        The tunneling strength is adaptive:
        - Higher when stuck (no improvement for many steps)
        - Lower when making progress
        """
        # Compute adaptive tunneling strength
        effective_tunneling = self.tunneling_strength

        # Increase tunneling if stuck (like increasing temperature in annealing)
        if self.steps_since_improvement > self.stuck_threshold:
            stuck_factor = min(5.0, 1 + self.steps_since_improvement / self.stuck_threshold)
            effective_tunneling *= stuck_factor

        # Apply tunneling to each parameter
        with torch.no_grad():
            for param in self.params:
                if param.requires_grad:
                    # Tunneling perturbation scaled by parameter magnitude
                    # This ensures perturbations are meaningful relative to weights
                    param_scale = param.abs().mean() + 1e-8
                    noise = torch.randn_like(param) * effective_tunneling * param_scale

                    # Apply tunneling
                    param.add_(noise)

        # Record tunneling event
        self.tunneling_events.append({
            'step': self.step_count,
            'strength': effective_tunneling,
            'stuck_steps': self.steps_since_improvement,
            'loss': current_loss
        })

    def get_tunneling_stats(self) -> dict:
        """
        Get statistics about tunneling behavior.

        Returns
        -------
        dict
            Dictionary with tunneling statistics
        """
        return {
            'total_steps': self.step_count,
            'tunneling_events': len(self.tunneling_events),
            'current_tunneling_strength': self.tunneling_strength,
            'best_loss': self.best_loss,
            'steps_since_improvement': self.steps_since_improvement,
            'final_vs_initial_tunneling': self.tunneling_strength / self.initial_tunneling
        }

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'tunneling_strength': self.tunneling_strength,
            'best_loss': self.best_loss,
            'steps_since_improvement': self.steps_since_improvement
        }

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from checkpoint."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        self.tunneling_strength = state_dict['tunneling_strength']
        self.best_loss = state_dict['best_loss']
        self.steps_since_improvement = state_dict['steps_since_improvement']


class QuantumAnnealingScheduler:
    """
    Learning rate scheduler inspired by quantum annealing.

    In quantum annealing, the transverse field (which enables tunneling)
    is gradually reduced. Similarly, this scheduler manages the trade-off
    between exploration (high lr) and exploitation (low lr).

    Parameters
    ----------
    optimizer : QuantumTunnelingOptimizer
        The optimizer to schedule
    initial_lr : float
        Starting learning rate
    final_lr : float
        Ending learning rate
    total_steps : int
        Total number of training steps
    warmup_steps : int, default=100
        Number of warmup steps
    """

    def __init__(
        self,
        optimizer: QuantumTunnelingOptimizer,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 100
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Quantum annealing schedule: exponential decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)

            # Exponential interpolation (like quantum annealing)
            lr = self.final_lr + (self.initial_lr - self.final_lr) * np.exp(-5 * progress)

        # Update learning rate in base optimizer
        for param_group in self.optimizer.base_optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.base_optimizer.param_groups[0]['lr']
