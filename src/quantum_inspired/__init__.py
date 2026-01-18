"""
Quantum-Inspired Methods for World Model Training
==================================================

This module contains properly quantum-inspired implementations
that translate quantum computing principles to classical neural networks.

Modules:
--------
- tunneling_optimizer: Quantum tunneling-inspired optimization (escapes local minima)
- entanglement_layer: Entanglement-inspired feature correlations
- superposition_buffer: Superposition-inspired experience replay with interference
- interference_ensemble: Interference-based ensemble predictions

All implementations maintain consistency with the standard configuration:
- stoch_dim = 64
- deter_dim = 512
- hidden_dim = 512
- batch_size = 32
- seq_len = 20
- learning_rate = 3e-4

Author: Saurabh Jalendra (BITS ID: 2023AC05912)
"""

from .tunneling_optimizer import QuantumTunnelingOptimizer
from .entanglement_layer import EntanglementLayer
from .superposition_buffer import SuperpositionReplayBuffer
from .interference_ensemble import InterferenceEnsemble

__all__ = [
    'QuantumTunnelingOptimizer',
    'EntanglementLayer',
    'SuperpositionReplayBuffer',
    'InterferenceEnsemble'
]

__version__ = '1.0.0'
