"""
Configuration module for Quantum-Enhanced Simulation Learning project.

This module provides centralized configuration management to ensure
consistency across all notebooks and experiments.
"""

from .shared_config import (
    # Device
    DEVICE,

    # Config classes
    WorldModelConfig,
    TrainingConfig,
    ExperimentConfig,
    ErrorCorrectionConfig,
    QAOAConfig,
    SuperpositionConfig,
    GateEnhancedConfig,

    # Default instances
    DEFAULT_WORLD_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_ERROR_CORRECTION_CONFIG,
    DEFAULT_QAOA_CONFIG,
    DEFAULT_SUPERPOSITION_CONFIG,
    DEFAULT_GATE_ENHANCED_CONFIG,

    # Utilities
    validate_model_config,
    print_config_summary,
)

__all__ = [
    'DEVICE',
    'WorldModelConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'ErrorCorrectionConfig',
    'QAOAConfig',
    'SuperpositionConfig',
    'GateEnhancedConfig',
    'DEFAULT_WORLD_MODEL_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_EXPERIMENT_CONFIG',
    'DEFAULT_ERROR_CORRECTION_CONFIG',
    'DEFAULT_QAOA_CONFIG',
    'DEFAULT_SUPERPOSITION_CONFIG',
    'DEFAULT_GATE_ENHANCED_CONFIG',
    'validate_model_config',
    'print_config_summary',
]
