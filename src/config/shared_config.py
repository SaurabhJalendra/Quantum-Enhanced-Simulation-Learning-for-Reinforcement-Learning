"""
SHARED CONFIGURATION FOR ALL NOTEBOOKS

This is the SINGLE SOURCE OF TRUTH for all model architectures, hyperparameters,
and experimental settings across the dissertation project.

IMPORTANT: All notebooks MUST import from this file to ensure consistency.
DO NOT hardcode these values in individual notebooks.

Author: Saurabh Jalendra
Project: Quantum-Enhanced Simulation Learning for Reinforcement Learning
Last Updated: December 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# WORLD MODEL ARCHITECTURE (MUST BE SAME ACROSS ALL NOTEBOOKS)
# =============================================================================
@dataclass
class WorldModelConfig:
    """
    Standard world model architecture configuration.
    ALL notebooks must use these exact values for fair comparison.
    """
    # Observation and action dimensions (CartPole-v1)
    obs_dim: int = 4
    action_dim: int = 2

    # RSSM dimensions - THESE ARE THE CRITICAL VALUES
    stoch_dim: int = 64      # Stochastic state dimension
    deter_dim: int = 512     # Deterministic state dimension (GRU hidden)
    hidden_dim: int = 512    # Hidden layer dimension

    # Derived
    @property
    def state_dim(self) -> int:
        """Total state dimension = stochastic + deterministic"""
        return self.stoch_dim + self.deter_dim

    # Encoder/Decoder
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Predictor networks
    reward_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    continue_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Other
    min_std: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy passing to model constructors."""
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'stoch_dim': self.stoch_dim,
            'deter_dim': self.deter_dim,
            'hidden_dim': self.hidden_dim,
            'state_dim': self.state_dim,
            'encoder_hidden_dims': self.encoder_hidden_dims,
            'decoder_hidden_dims': self.decoder_hidden_dims,
            'reward_hidden_dims': self.reward_hidden_dims,
            'continue_hidden_dims': self.continue_hidden_dims,
            'min_std': self.min_std,
        }


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
@dataclass
class TrainingConfig:
    """Standard training configuration for all approaches.

    Optimized for RTX 5090 (30GB VRAM).
    """
    # Optimizer
    learning_rate: float = 3e-4  # 0.0003 - standard for Adam
    weight_decay: float = 0.0
    grad_clip: float = 100.0

    # Training loop
    batch_size: int = 32         # Larger batch for GPU utilization
    seq_len: int = 20            # Sequence length for world model
    num_epochs: int = 50         # Training epochs

    # KL balancing (DreamerV3 style)
    kl_balance: float = 0.8
    kl_free: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'num_epochs': self.num_epochs,
            'kl_balance': self.kl_balance,
            'kl_free': self.kl_free,
        }


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
@dataclass
class ExperimentConfig:
    """Standard experiment configuration for reproducibility."""
    # Random seeds for multi-seed experiments
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Data collection
    num_episodes: int = 100      # More episodes for better training data
    max_episode_length: int = 500

    # Environment
    env_name: str = "CartPole-v1"

    # Statistical analysis
    confidence_level: float = 0.95
    bonferroni_tests: int = 2  # Number of comparisons for Bonferroni correction

    @property
    def bonferroni_alpha(self) -> float:
        """Bonferroni-corrected significance level."""
        return (1 - self.confidence_level) / self.bonferroni_tests

    def to_dict(self) -> Dict[str, Any]:
        return {
            'seeds': self.seeds,
            'num_episodes': self.num_episodes,
            'max_episode_length': self.max_episode_length,
            'env_name': self.env_name,
            'confidence_level': self.confidence_level,
            'bonferroni_alpha': self.bonferroni_alpha,
        }


# =============================================================================
# ERROR CORRECTION SPECIFIC CONFIG
# =============================================================================
@dataclass
class ErrorCorrectionConfig:
    """Configuration specific to error correction ensemble."""
    num_ensemble_models: int = 5
    correction_method: str = "weighted"  # "majority", "weighted", "exclusion"
    syndrome_threshold: float = 2.0
    diversity_weight: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_ensemble_models': self.num_ensemble_models,
            'correction_method': self.correction_method,
            'syndrome_threshold': self.syndrome_threshold,
            'diversity_weight': self.diversity_weight,
        }


# =============================================================================
# QAOA SPECIFIC CONFIG
# =============================================================================
@dataclass
class QAOAConfig:
    """Configuration specific to QAOA-enhanced training."""
    num_layers: int = 4
    gamma_init: float = 0.1
    beta_init: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_layers': self.num_layers,
            'gamma_init': self.gamma_init,
            'beta_init': self.beta_init,
        }


# =============================================================================
# SUPERPOSITION SPECIFIC CONFIG
# =============================================================================
@dataclass
class SuperpositionConfig:
    """Configuration specific to superposition replay."""
    num_superposition_states: int = 4
    interference_strength: float = 0.1
    collapse_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_superposition_states': self.num_superposition_states,
            'interference_strength': self.interference_strength,
            'collapse_threshold': self.collapse_threshold,
        }


# =============================================================================
# GATE ENHANCED SPECIFIC CONFIG
# =============================================================================
@dataclass
class GateEnhancedConfig:
    """Configuration specific to gate-enhanced layers."""
    use_hadamard: bool = True
    use_phase: bool = True
    use_entanglement: bool = True
    entanglement_pairs: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_hadamard': self.use_hadamard,
            'use_phase': self.use_phase,
            'use_entanglement': self.use_entanglement,
            'entanglement_pairs': self.entanglement_pairs,
        }


# =============================================================================
# DEFAULT INSTANCES (USE THESE IN NOTEBOOKS)
# =============================================================================
DEFAULT_WORLD_MODEL_CONFIG = WorldModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
DEFAULT_ERROR_CORRECTION_CONFIG = ErrorCorrectionConfig()
DEFAULT_QAOA_CONFIG = QAOAConfig()
DEFAULT_SUPERPOSITION_CONFIG = SuperpositionConfig()
DEFAULT_GATE_ENHANCED_CONFIG = GateEnhancedConfig()


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================
def validate_model_config(config: Dict[str, Any], notebook_name: str) -> bool:
    """
    Validate that a model configuration matches the standard.
    Call this in every notebook to ensure consistency.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration being used in the notebook
    notebook_name : str
        Name of the notebook for error messages

    Returns
    -------
    bool
        True if valid, raises ValueError if not
    """
    expected = DEFAULT_WORLD_MODEL_CONFIG.to_dict()

    critical_params = ['stoch_dim', 'deter_dim', 'hidden_dim', 'obs_dim', 'action_dim']

    errors = []
    for param in critical_params:
        if param in config and config[param] != expected[param]:
            errors.append(
                f"  {param}: got {config[param]}, expected {expected[param]}"
            )

    if errors:
        error_msg = f"\n{'='*60}\n"
        error_msg += f"CONFIGURATION ERROR IN {notebook_name}\n"
        error_msg += f"{'='*60}\n"
        error_msg += "The following parameters do not match the standard config:\n"
        error_msg += "\n".join(errors)
        error_msg += f"\n\nPlease use shared_config.DEFAULT_WORLD_MODEL_CONFIG"
        error_msg += f"\n{'='*60}\n"
        raise ValueError(error_msg)

    return True


# =============================================================================
# PRINT CONFIG SUMMARY
# =============================================================================
def print_config_summary():
    """Print a summary of all configurations for verification."""
    print("=" * 60)
    print("SHARED CONFIGURATION SUMMARY")
    print("=" * 60)

    print("\nWorld Model Architecture:")
    print(f"  obs_dim:    {DEFAULT_WORLD_MODEL_CONFIG.obs_dim}")
    print(f"  action_dim: {DEFAULT_WORLD_MODEL_CONFIG.action_dim}")
    print(f"  stoch_dim:  {DEFAULT_WORLD_MODEL_CONFIG.stoch_dim}")
    print(f"  deter_dim:  {DEFAULT_WORLD_MODEL_CONFIG.deter_dim}")
    print(f"  hidden_dim: {DEFAULT_WORLD_MODEL_CONFIG.hidden_dim}")
    print(f"  state_dim:  {DEFAULT_WORLD_MODEL_CONFIG.state_dim}")

    print("\nTraining Configuration:")
    print(f"  learning_rate: {DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"  batch_size:    {DEFAULT_TRAINING_CONFIG.batch_size}")
    print(f"  seq_len:       {DEFAULT_TRAINING_CONFIG.seq_len}")
    print(f"  num_epochs:    {DEFAULT_TRAINING_CONFIG.num_epochs}")

    print("\nExperiment Configuration:")
    print(f"  seeds:        {DEFAULT_EXPERIMENT_CONFIG.seeds}")
    print(f"  num_episodes: {DEFAULT_EXPERIMENT_CONFIG.num_episodes}")
    print(f"  env_name:     {DEFAULT_EXPERIMENT_CONFIG.env_name}")

    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
