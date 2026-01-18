"""
Superposition-Inspired Experience Replay Buffer
================================================

Quantum Principle:
    In quantum mechanics, a particle can exist in a SUPERPOSITION of
    multiple states simultaneously. When measured, the superposition
    "collapses" to a single state, with probabilities determined by
    the amplitudes of each component.

    Key insight: Before measurement, the system explores ALL possible
    states simultaneously. Interference between states can amplify
    "good" states and cancel "bad" states.

Classical Translation:
    Standard replay buffer: Sample one experience at a time
    Superposition buffer: Sample multiple experiences, combine them
                          using interference-based weighting

    This is similar to but distinct from:
    - Prioritized Experience Replay (PER): Samples based on TD-error
    - Mixup data augmentation: Linear interpolation of samples

    Our approach combines BOTH ideas:
    1. Prioritized sampling (like PER, inspired by amplitude)
    2. Interference-based combination (amplitudes with phases)

Mathematical Formulation:
    Each experience has:
    - Amplitude |α| = f(TD-error)  (like quantum amplitude magnitude)
    - Phase φ = g(value estimate)   (like quantum phase)

    Combination: |ψ⟩ = Σ αᵢ e^(iφᵢ) |expᵢ⟩

    Interference weight for experience i:
    w_i = |Σⱼ αᵢαⱼ cos(φᵢ - φⱼ)|

    Experiences that "agree" (similar phases) reinforce each other.
    Experiences that "disagree" (opposite phases) cancel out.

Configuration:
    Output format matches standard buffer:
    - obs: (batch_size, seq_len, obs_dim) = (32, 20, 4)
    - actions: (batch_size, seq_len, action_dim) = (32, 20, 2)
    - rewards: (batch_size, seq_len) = (32, 20)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class SuperpositionReplayBuffer:
    """
    Experience replay buffer with quantum superposition-inspired sampling.

    Parameters
    ----------
    capacity : int, default=10000
        Maximum number of episodes to store
    alpha : float, default=0.6
        Priority exponent (0 = uniform, 1 = full prioritization)
    beta : float, default=0.4
        Importance sampling exponent
    beta_increment : float, default=0.001
        How much to increase beta per sample (anneals to 1)
    num_superpose : int, default=3
        Number of experiences to superpose per sample
    interference_strength : float, default=0.5
        How strongly interference affects combination (0 = no interference, 1 = full)

    Example
    -------
    >>> buffer = SuperpositionReplayBuffer(capacity=10000)
    >>> buffer.add(episode={'obs': obs, 'actions': actions, 'rewards': rewards})
    >>> batch = buffer.sample(batch_size=32, seq_len=20)
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        num_superpose: int = 3,
        interference_strength: float = 0.5
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.num_superpose = num_superpose
        self.interference_strength = interference_strength

        # Storage
        self.buffer: List[Dict] = []
        self.priorities: List[float] = []
        self.phases: List[float] = []  # Quantum-inspired phases
        self.position = 0

        # Statistics
        self.sample_count = 0
        self.interference_stats = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        episode: Dict[str, np.ndarray],
        td_error: Optional[float] = None,
        value_estimate: Optional[float] = None
    ):
        """
        Add an episode to the buffer.

        Parameters
        ----------
        episode : dict
            Episode data with keys 'obs', 'actions', 'rewards'
            - obs: (episode_len, obs_dim)
            - actions: (episode_len, action_dim)
            - rewards: (episode_len,)
        td_error : float, optional
            TD-error for priority calculation. If None, uses max priority.
        value_estimate : float, optional
            Value estimate for phase calculation. If None, uses reward sum.
        """
        # Compute priority (amplitude magnitude)
        if td_error is not None:
            priority = (np.abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = max(self.priorities) if self.priorities else 1.0

        # Compute phase based on value estimate
        # High value → phase near 0 (positive)
        # Low value → phase near π (negative)
        if value_estimate is not None:
            phase = np.arctan2(value_estimate, np.abs(value_estimate) + 1e-6)
        else:
            # Use reward sum as proxy for value
            reward_sum = episode['rewards'].sum()
            phase = np.arctan2(reward_sum, np.abs(reward_sum) + 1e-6)

        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(episode)
            self.priorities.append(priority)
            self.phases.append(phase)
        else:
            # Overwrite oldest
            self.buffer[self.position] = episode
            self.priorities[self.position] = priority
            self.phases[self.position] = phase

        self.position = (self.position + 1) % self.capacity

    def update_priorities(
        self,
        indices: List[int],
        td_errors: List[float],
        value_estimates: Optional[List[float]] = None
    ):
        """
        Update priorities and phases for sampled experiences.

        Parameters
        ----------
        indices : list of int
            Indices of experiences to update
        td_errors : list of float
            New TD-errors
        value_estimates : list of float, optional
            New value estimates for phase update
        """
        for i, (idx, td_error) in enumerate(zip(indices, td_errors)):
            if idx < len(self.buffer):
                self.priorities[idx] = (np.abs(td_error) + 1e-6) ** self.alpha

                if value_estimates is not None:
                    value = value_estimates[i]
                    self.phases[idx] = np.arctan2(value, np.abs(value) + 1e-6)

    def _compute_interference_weights(
        self,
        indices: List[int]
    ) -> np.ndarray:
        """
        Compute interference weights for a set of experiences.

        Experiences with aligned phases reinforce each other.
        Experiences with opposite phases cancel out.

        Parameters
        ----------
        indices : list of int
            Indices of experiences to combine

        Returns
        -------
        np.ndarray
            Weights for each experience, shape (num_experiences,)
        """
        if len(indices) == 1:
            return np.array([1.0])

        # Get amplitudes and phases
        amplitudes = np.array([np.sqrt(self.priorities[i]) for i in indices])
        phases = np.array([self.phases[i] for i in indices])

        # Compute complex amplitudes
        complex_amplitudes = amplitudes * np.exp(1j * phases)

        # Interference: weight based on how well each experience
        # aligns with the total superposition
        total_amplitude = np.sum(complex_amplitudes)

        # Weight = contribution to total amplitude
        # Aligned experiences contribute positively
        # Misaligned experiences contribute less or negatively
        weights = np.real(complex_amplitudes * np.conj(total_amplitude))
        weights = np.abs(weights)  # Ensure positive

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(indices)) / len(indices)

        # Blend with uniform weights based on interference_strength
        uniform = np.ones(len(indices)) / len(indices)
        weights = self.interference_strength * weights + (1 - self.interference_strength) * uniform

        return weights

    def _extract_sequence(
        self,
        episode: Dict[str, np.ndarray],
        seq_len: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a random sequence from an episode.
        """
        rng = np.random.RandomState(seed)
        episode_len = len(episode['obs'])

        if episode_len <= seq_len:
            # Pad if episode is too short
            pad_len = seq_len - episode_len
            obs = np.pad(episode['obs'], ((0, pad_len), (0, 0)), mode='edge')
            actions = np.pad(episode['actions'], ((0, pad_len), (0, 0)), mode='edge')
            rewards = np.pad(episode['rewards'], (0, pad_len), mode='edge')
        else:
            # Random start
            start = rng.randint(0, episode_len - seq_len)
            obs = episode['obs'][start:start + seq_len]
            actions = episode['actions'][start:start + seq_len]
            rewards = episode['rewards'][start:start + seq_len]

        return obs, actions, rewards

    def sample(
        self,
        batch_size: int = 32,
        seq_len: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch with superposition-inspired combination.

        Parameters
        ----------
        batch_size : int, default=32
            Number of samples in batch
        seq_len : int, default=20
            Sequence length

        Returns
        -------
        dict
            Batch dictionary with keys:
            - 'obs': torch.Tensor of shape (batch_size, seq_len, obs_dim)
            - 'actions': torch.Tensor of shape (batch_size, seq_len, action_dim)
            - 'rewards': torch.Tensor of shape (batch_size, seq_len)
            - 'indices': List of primary indices (for priority updates)
            - 'weights': torch.Tensor of importance sampling weights
        """
        if len(self.buffer) < self.num_superpose:
            raise ValueError(f"Buffer has {len(self.buffer)} episodes, need at least {self.num_superpose}")

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        # Sample batch
        obs_batch = []
        action_batch = []
        reward_batch = []
        primary_indices = []
        is_weights = []

        for _ in range(batch_size):
            # Sample multiple experiences for superposition
            indices = np.random.choice(
                len(self.buffer),
                size=min(self.num_superpose, len(self.buffer)),
                p=probs,
                replace=False
            )

            # Compute interference weights
            weights = self._compute_interference_weights(indices)

            # Extract and combine sequences
            combined_obs = None
            combined_actions = None
            combined_rewards = None

            for idx, weight in zip(indices, weights):
                obs, actions, rewards = self._extract_sequence(
                    self.buffer[idx],
                    seq_len
                )

                if combined_obs is None:
                    combined_obs = weight * obs
                    combined_actions = obs.copy()  # Keep from primary (can't interpolate discrete)
                    combined_rewards = weight * rewards
                else:
                    combined_obs += weight * obs
                    combined_rewards += weight * rewards

            # Primary index (highest priority in superposition)
            primary_idx = indices[np.argmax([self.priorities[i] for i in indices])]
            primary_indices.append(primary_idx)

            # Use actions from primary experience (discrete, can't interpolate)
            _, primary_actions, _ = self._extract_sequence(
                self.buffer[primary_idx],
                seq_len
            )

            obs_batch.append(combined_obs)
            action_batch.append(primary_actions)
            reward_batch.append(combined_rewards)

            # Importance sampling weight
            max_weight = (len(self.buffer) * probs.min()) ** (-self.beta)
            is_weight = (len(self.buffer) * probs[primary_idx]) ** (-self.beta)
            is_weights.append(is_weight / max_weight)

        # Update beta (anneal towards 1)
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.sample_count += batch_size

        # Record interference stats
        self.interference_stats.append({
            'sample_count': self.sample_count,
            'beta': self.beta,
            'mean_priority': priorities.mean(),
            'priority_std': priorities.std()
        })

        return {
            'obs': torch.tensor(np.array(obs_batch), dtype=torch.float32),
            'actions': torch.tensor(np.array(action_batch), dtype=torch.float32),
            'rewards': torch.tensor(np.array(reward_batch), dtype=torch.float32),
            'indices': primary_indices,
            'weights': torch.tensor(is_weights, dtype=torch.float32)
        }

    def sample_standard(
        self,
        batch_size: int = 32,
        seq_len: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Sample WITHOUT superposition (for comparison/ablation).
        """
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        obs_batch = []
        action_batch = []
        reward_batch = []
        indices = []

        sampled_indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            p=probs,
            replace=True
        )

        for idx in sampled_indices:
            obs, actions, rewards = self._extract_sequence(
                self.buffer[idx],
                seq_len
            )
            obs_batch.append(obs)
            action_batch.append(actions)
            reward_batch.append(rewards)
            indices.append(idx)

        return {
            'obs': torch.tensor(np.array(obs_batch), dtype=torch.float32),
            'actions': torch.tensor(np.array(action_batch), dtype=torch.float32),
            'rewards': torch.tensor(np.array(reward_batch), dtype=torch.float32),
            'indices': indices,
            'weights': torch.ones(batch_size, dtype=torch.float32)
        }

    def get_stats(self) -> dict:
        """
        Get buffer statistics.
        """
        priorities = np.array(self.priorities) if self.priorities else np.array([0])
        phases = np.array(self.phases) if self.phases else np.array([0])

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'priority_mean': priorities.mean(),
            'priority_std': priorities.std(),
            'priority_max': priorities.max(),
            'priority_min': priorities.min(),
            'phase_mean': phases.mean(),
            'phase_std': phases.std(),
            'beta': self.beta,
            'sample_count': self.sample_count
        }
