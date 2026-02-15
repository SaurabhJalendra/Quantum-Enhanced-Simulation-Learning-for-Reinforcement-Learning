"""
Interference-Based Ensemble for Error Correction
=================================================

Quantum Principle:
    Quantum interference occurs when multiple quantum paths combine.
    - CONSTRUCTIVE interference: aligned phases amplify
    - DESTRUCTIVE interference: opposite phases cancel

    In quantum error correction, redundant encoding allows detection
    and correction of errors through interference patterns.

Classical Translation:
    Standard ensemble: Average all predictions equally
    Interference ensemble: Weight predictions based on phase alignment

    Models that AGREE (similar predictions, aligned phases) reinforce each other.
    Models that DISAGREE (different predictions, opposite phases) cancel out.

    This implements proper "error correction" because:
    1. ERROR DETECTION: Model disagreement indicates potential error
    2. ERROR CORRECTION: Disagreeing models get downweighted

Mathematical Formulation:
    Each model i has:
    - Prediction p_i
    - Confidence (amplitude) α_i = 1 / uncertainty_i
    - Phase φ_i = f(prediction, context)

    Interference weight:
    w_i = |Σⱼ αᵢαⱼ cos(φᵢ - φⱼ)|

    Final prediction:
    p = Σᵢ w_i * p_i / Σᵢ w_i

Configuration:
    Each model uses standard RSSM architecture:
    - stoch_dim = 64
    - deter_dim = 512
    - hidden_dim = 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable


class InterferenceEnsemble(nn.Module):
    """
    Ensemble with quantum interference-inspired prediction combination.

    Parameters
    ----------
    model_class : type
        Class of the base model (e.g., RSSMWorldModel)
    num_models : int, default=5
        Number of models in ensemble
    interference_strength : float, default=0.7
        How strongly interference affects combination (0 = simple average, 1 = full interference)
    uncertainty_method : str, default='disagreement'
        How to estimate uncertainty: 'disagreement', 'variance', 'entropy'
    base_seed : int, optional
        Base seed for model initialization (each model uses base_seed + i * 1000)
    **model_kwargs
        Keyword arguments passed to model_class

    Example
    -------
    >>> ensemble = InterferenceEnsemble(
    ...     model_class=RSSMWorldModel,
    ...     num_models=5,
    ...     obs_dim=4,
    ...     action_dim=2,
    ...     stoch_dim=64,
    ...     deter_dim=512,
    ...     hidden_dim=512,
    ...     base_seed=42
    ... )
    >>> pred, states = ensemble(obs_seq, action_seq)
    """

    def __init__(
        self,
        model_class: type,
        num_models: int = 5,
        interference_strength: float = 0.7,
        uncertainty_method: str = 'disagreement',
        base_seed: Optional[int] = None,
        **model_kwargs
    ):
        super().__init__()
        self.num_models = num_models
        self.interference_strength = interference_strength
        self.uncertainty_method = uncertainty_method
        self.model_kwargs = model_kwargs

        # Create ensemble members with different initializations
        self.models = nn.ModuleList()
        for i in range(num_models):
            if base_seed is not None:
                torch.manual_seed(base_seed + i * 1000)
            model = model_class(**model_kwargs)
            self.models.append(model)

        # Learnable phase offsets for each model
        # This allows the ensemble to learn which models should agree/disagree
        self.phase_offsets = nn.Parameter(torch.zeros(num_models))

        # Learnable uncertainty scaling
        self.uncertainty_scale = nn.Parameter(torch.ones(1))

        # Statistics tracking (bounded to prevent memory growth)
        from collections import deque
        self.prediction_stats = deque(maxlen=200)

    def compute_uncertainty(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty for each model's prediction.

        Parameters
        ----------
        predictions : torch.Tensor
            Predictions from all models, shape (num_models, batch, ...)

        Returns
        -------
        torch.Tensor
            Uncertainty for each model, shape (num_models,)
        """
        if self.uncertainty_method == 'disagreement':
            # Uncertainty = distance from ensemble mean
            mean_pred = predictions.mean(dim=0)
            uncertainties = ((predictions - mean_pred) ** 2).mean(dim=tuple(range(1, predictions.dim())))

        elif self.uncertainty_method == 'variance':
            # Uncertainty = variance across predictions
            uncertainties = predictions.var(dim=tuple(range(1, predictions.dim())))

        elif self.uncertainty_method == 'entropy':
            # Treat predictions as probabilities (after softmax) and compute entropy
            probs = F.softmax(predictions.view(self.num_models, -1), dim=-1)
            uncertainties = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")

        return uncertainties

    def compute_phases(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase for each model based on prediction characteristics.

        High confidence predictions → phase near 0
        Low confidence predictions → phase near π

        Parameters
        ----------
        predictions : torch.Tensor
            Predictions from all models
        uncertainties : torch.Tensor
            Uncertainty estimates for each model

        Returns
        -------
        torch.Tensor
            Phase for each model, shape (num_models,)
        """
        # Base phase from uncertainty (high uncertainty → phase shifts toward π)
        normalized_uncertainty = uncertainties / (uncertainties.max() + 1e-8)
        base_phase = np.pi * torch.sigmoid(normalized_uncertainty * self.uncertainty_scale)

        # Add learnable offset
        phases = base_phase + self.phase_offsets

        return phases

    def compute_interference_weights(
        self,
        amplitudes: torch.Tensor,
        phases: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interference weights using quantum interference formula.

        Parameters
        ----------
        amplitudes : torch.Tensor
            Amplitude (confidence) for each model, shape (num_models,)
        phases : torch.Tensor
            Phase for each model, shape (num_models,)

        Returns
        -------
        torch.Tensor
            Interference weights for each model, shape (num_models,)
        """
        # Compute complex amplitudes
        complex_amp = amplitudes * torch.exp(1j * phases.to(torch.complex64))

        # Interference: each model's weight depends on how it aligns with others
        # w_i = |Σⱼ αᵢαⱼ cos(φᵢ - φⱼ)|
        weights = torch.zeros(self.num_models, device=amplitudes.device)

        for i in range(self.num_models):
            # Compute interference contribution from all pairs involving model i
            for j in range(self.num_models):
                phase_diff = phases[i] - phases[j]
                interference = amplitudes[i] * amplitudes[j] * torch.cos(phase_diff)
                weights[i] += interference

        # Take absolute value and normalize
        weights = torch.abs(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(self.num_models, device=amplitudes.device) / self.num_models

        # Blend with uniform weights based on interference_strength
        uniform = torch.ones(self.num_models, device=amplitudes.device) / self.num_models
        weights = self.interference_strength * weights + (1 - self.interference_strength) * uniform

        return weights

    def detect_errors(
        self,
        predictions: torch.Tensor,
        threshold_std: float = 2.0
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Detect which models have potential errors (outlier predictions).

        This implements the "syndrome measurement" from quantum error correction.

        Parameters
        ----------
        predictions : torch.Tensor
            Predictions from all models
        threshold_std : float
            Number of standard deviations to consider as outlier

        Returns
        -------
        Tuple[torch.Tensor, List[int]]
            - Boolean mask indicating faulty models
            - List of indices of faulty models
        """
        # Compute deviation from median (robust to outliers)
        median_pred = predictions.median(dim=0).values
        deviations = ((predictions - median_pred) ** 2).mean(dim=tuple(range(1, predictions.dim())))

        # Detect outliers
        mean_dev = deviations.mean()
        std_dev = deviations.std()
        threshold = mean_dev + threshold_std * std_dev

        is_faulty = deviations > threshold
        faulty_indices = torch.where(is_faulty)[0].tolist()

        return is_faulty, faulty_indices

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with interference-based combination.

        Parameters
        ----------
        obs_seq : torch.Tensor
            Observation sequence, shape (batch, seq_len, obs_dim)
        action_seq : torch.Tensor
            Action sequence, shape (batch, seq_len, action_dim)
        return_all : bool, default=False
            If True, return individual model predictions

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Combined prediction
            - Dictionary with states and metadata
        """
        # Get predictions from all models
        all_predictions = []
        all_states = []

        for model in self.models:
            pred, states = model(obs_seq, action_seq)
            all_predictions.append(pred)
            all_states.append(states)

        # Stack predictions: (num_models, batch, seq, obs_dim)
        predictions = torch.stack(all_predictions, dim=0)

        # Compute uncertainties and phases
        uncertainties = self.compute_uncertainty(predictions)
        phases = self.compute_phases(predictions, uncertainties)

        # Compute amplitudes (inverse uncertainty = confidence)
        amplitudes = 1.0 / (uncertainties + 1e-8)
        amplitudes = amplitudes / amplitudes.sum()  # Normalize

        # Detect errors (syndrome measurement)
        is_faulty, faulty_indices = self.detect_errors(predictions)

        # Compute interference weights
        weights = self.compute_interference_weights(amplitudes, phases)

        # Downweight faulty models (error correction)
        if len(faulty_indices) > 0 and len(faulty_indices) < self.num_models - 1:
            for idx in faulty_indices:
                weights[idx] *= 0.1  # Reduce weight of faulty models
            weights = weights / weights.sum()  # Renormalize

        # Combine predictions with interference weights
        # Dynamically expand weights to match prediction dimensions
        # predictions shape: (num_models, batch, seq, *obs_shape)
        num_extra_dims = predictions.dim() - 1  # All dims except num_models
        weights_shape = [self.num_models] + [1] * num_extra_dims
        weights_expanded = weights.view(*weights_shape)
        combined_pred = (predictions * weights_expanded).sum(dim=0)

        # Combine states similarly (only tensor entries, skip distributions)
        combined_states = {}
        for key in all_states[0].keys():
            if key in ('priors', 'posteriors'):
                combined_states[key] = all_states[0][key]  # Keep first model's distributions
                continue
            vals = [s[key] for s in all_states]
            if not isinstance(vals[0], torch.Tensor):
                continue
            stacked = torch.stack(vals, dim=0)
            # Reshape weights to match stacked tensor dimensions
            w_shape = [-1] + [1] * (stacked.dim() - 1)
            combined_states[key] = (stacked * weights.view(*w_shape)).sum(dim=0)

        # Track statistics
        self.prediction_stats.append({
            'weights': weights.detach().cpu().numpy(),
            'phases': phases.detach().cpu().numpy(),
            'amplitudes': amplitudes.detach().cpu().numpy(),
            'faulty_indices': faulty_indices,
            'uncertainties': uncertainties.detach().cpu().numpy()
        })

        # Metadata
        combined_states['ensemble_weights'] = weights
        combined_states['ensemble_phases'] = phases
        combined_states['faulty_models'] = faulty_indices

        if return_all:
            combined_states['all_predictions'] = predictions
            combined_states['all_states'] = all_states

        return combined_pred, combined_states

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        target_obs: torch.Tensor,
        target_rewards: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for the ensemble.

        Each model gets individual loss + combined loss.

        Parameters
        ----------
        obs_seq : torch.Tensor
            Input observation sequence
        action_seq : torch.Tensor
            Input action sequence
        target_obs : torch.Tensor
            Target observation sequence
        target_rewards : torch.Tensor, optional
            Target rewards

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with different loss components
        """
        # Get all predictions
        all_predictions = []
        for model in self.models:
            pred, _ = model(obs_seq, action_seq)
            all_predictions.append(pred)

        predictions = torch.stack(all_predictions, dim=0)

        # Individual losses (each model should be accurate)
        individual_losses = []
        for pred in all_predictions:
            loss = F.mse_loss(pred, target_obs)
            individual_losses.append(loss)

        individual_loss = torch.stack(individual_losses).mean()

        # Combined loss (ensemble prediction should be accurate)
        combined_pred, _ = self.forward(obs_seq, action_seq)
        combined_loss = F.mse_loss(combined_pred, target_obs)

        # Diversity loss (encourage models to be diverse but not too different)
        # We want models to explore different hypotheses
        mean_pred = predictions.mean(dim=0)
        diversity = ((predictions - mean_pred) ** 2).mean()
        diversity_loss = -0.01 * diversity  # Negative because we want diversity

        # Total loss
        total_loss = 0.5 * individual_loss + 0.5 * combined_loss + diversity_loss

        return {
            'total': total_loss,
            'individual': individual_loss,
            'combined': combined_loss,
            'diversity': diversity
        }

    def get_stats(self) -> dict:
        """
        Get ensemble statistics.
        """
        if not self.prediction_stats:
            return {}

        recent_stats = self.prediction_stats[-100:]  # Last 100 predictions

        weights = np.array([s['weights'] for s in recent_stats])
        phases = np.array([s['phases'] for s in recent_stats])

        return {
            'mean_weights': weights.mean(axis=0).tolist(),
            'weight_std': weights.std(axis=0).tolist(),
            'mean_phases': phases.mean(axis=0).tolist(),
            'phase_std': phases.std(axis=0).tolist(),
            'num_predictions': len(self.prediction_stats),
            'faulty_model_frequency': {
                i: sum(1 for s in recent_stats if i in s['faulty_indices'])
                for i in range(self.num_models)
            }
        }
