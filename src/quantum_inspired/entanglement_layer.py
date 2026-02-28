"""
Entanglement-Inspired Neural Network Layer
==========================================

Quantum Principle:
    Entangled quantum particles exhibit correlations that cannot exist
    in classical systems. When you measure one entangled particle,
    you instantly know something about its partner, regardless of distance.

    Key insight: Entanglement creates LEARNABLE CORRELATIONS between
    parts of a quantum system that go beyond simple linear relationships.

Classical Translation:
    Standard neural network layers process features somewhat independently.
    An entanglement-inspired layer explicitly learns which features should
    be correlated and enforces those correlations during forward pass.

    This is similar to but distinct from:
    - Attention mechanisms (compute correlations dynamically per input)
    - Graph neural networks (fixed graph structure)
    - Batch normalization (normalizes, doesn't correlate)

    Our approach: Learn a STATIC correlation structure that captures
    the physics of the environment (e.g., angle-velocity relationships in CartPole).

Mathematical Formulation:
    Standard linear: y = Wx + b
    Entanglement:    y = Wx + b + E(x)

    where E(x) is the entanglement term:
    E(x)_i = sum_j C_ij * f(x_i, x_j)

    C_ij is the learned correlation strength between features i and j
    f(x_i, x_j) is a correlation function (e.g., product, sum, etc.)

Configuration:
    This layer maintains the same dimensions as standard layers:
    - Input: (batch, hidden_dim) where hidden_dim = 512
    - Output: (batch, hidden_dim) where hidden_dim = 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class EntanglementLayer(nn.Module):
    """
    Entanglement-inspired layer that learns feature correlations.

    Parameters
    ----------
    dim : int, default=512
        Feature dimension (must match hidden_dim in RSSM)
    num_entangled_pairs : int, default=None
        Number of feature pairs to entangle. If None, uses dim // 4.
    correlation_type : str, default='multiplicative'
        Type of correlation function: 'multiplicative', 'additive', 'bilinear'
    learnable_pairs : bool, default=True
        If True, learn which features to entangle. If False, use adjacent pairs.
    residual : bool, default=True
        If True, add residual connection (helps gradient flow)

    Example
    -------
    >>> layer = EntanglementLayer(dim=512)
    >>> x = torch.randn(32, 512)  # (batch, hidden_dim)
    >>> y = layer(x)  # (batch, 512) - same shape
    """

    def __init__(
        self,
        dim: int = 512,
        num_entangled_pairs: Optional[int] = None,
        correlation_type: str = 'multiplicative',
        learnable_pairs: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_pairs = num_entangled_pairs or (dim // 4)
        self.correlation_type = correlation_type
        self.learnable_pairs = learnable_pairs
        self.residual = residual

        if learnable_pairs:
            # Learn which features should be entangled
            # This is a soft attention over all possible pairs
            self.pair_logits = nn.Parameter(torch.zeros(dim, dim))
            # Initialize to slightly prefer adjacent features
            for i in range(dim - 1):
                self.pair_logits.data[i, i + 1] = 0.5
                self.pair_logits.data[i + 1, i] = 0.5
        else:
            # Fixed adjacent pairs
            self.register_buffer(
                'pair_indices',
                torch.tensor([(i, i + 1) for i in range(0, dim - 1, 2)])
            )

        # Correlation strength parameters
        self.correlation_strength = nn.Parameter(torch.ones(self.num_pairs) * 0.1)

        # For bilinear correlation
        if correlation_type == 'bilinear':
            self.bilinear = nn.Bilinear(1, 1, 1, bias=False)

        # Output projection to ensure same dimension
        self.output_proj = nn.Linear(dim, dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(dim)

    def _get_entanglement_matrix(self) -> torch.Tensor:
        """
        Compute the entanglement matrix C where C_ij represents
        the correlation strength between features i and j.
        """
        if self.learnable_pairs:
            # Soft entanglement: all pairs with learned weights
            # Make symmetric (entanglement is bidirectional)
            C = self.pair_logits + self.pair_logits.T
            # Apply per-row softmax so each feature has meaningful correlation weights
            # (Global softmax over 512x512=262K entries produces near-uniform ~0.000004)
            C = torch.softmax(C, dim=1)
            # Scale by overall correlation strength
            C = C * self.correlation_strength.mean()
        else:
            # Hard entanglement: only specified pairs
            C = torch.zeros(self.dim, self.dim, device=self.correlation_strength.device)
            for idx, (i, j) in enumerate(self.pair_indices):
                strength = torch.sigmoid(self.correlation_strength[idx % len(self.correlation_strength)])
                C[i, j] = strength
                C[j, i] = strength
        return C

    def _compute_entanglement(self, x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute the entanglement contribution E(x).

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, dim)
        C : torch.Tensor
            Correlation matrix of shape (dim, dim)

        Returns
        -------
        torch.Tensor
            Entanglement contribution of shape (batch, dim)
        """
        batch_size = x.shape[0]

        if self.correlation_type == 'multiplicative':
            # E(x)_i = sum_j C_ij * x_i * x_j
            # This captures multiplicative interactions (like x1 * x2)
            # Efficient computation: E = (C @ x.T).T * x
            entangled = torch.matmul(x, C) * x

        elif self.correlation_type == 'additive':
            # E(x)_i = sum_j C_ij * (x_i + x_j)
            # This captures additive interactions
            sum_contrib = torch.matmul(x, C) + x * C.sum(dim=1)
            entangled = sum_contrib

        elif self.correlation_type == 'bilinear':
            # More complex learned correlation
            # Slower but more expressive
            entangled = torch.zeros_like(x)
            for i in range(self.dim):
                for j in range(self.dim):
                    if C[i, j] > 0.01:  # Sparse computation
                        contrib = self.bilinear(
                            x[:, i:i+1],
                            x[:, j:j+1]
                        ).squeeze(-1)
                        entangled[:, i] += C[i, j] * contrib
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")

        return entangled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with entanglement.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, dim) or (batch, seq, dim)

        Returns
        -------
        torch.Tensor
            Output of same shape as input
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, dim = x.shape
            x = x.view(batch * seq, dim)

        # Get entanglement matrix
        C = self._get_entanglement_matrix()

        # Compute entanglement contribution
        entanglement = self._compute_entanglement(x, C)

        # Combine with projection
        output = self.output_proj(x + entanglement)

        # Layer normalization
        output = self.layer_norm(output)

        # Residual connection
        if self.residual:
            output = output + x

        # Restore original shape
        if len(original_shape) == 3:
            output = output.view(original_shape)

        return output

    def get_entanglement_stats(self) -> dict:
        """
        Get statistics about learned entanglement structure.

        Returns
        -------
        dict
            Dictionary with entanglement statistics
        """
        C = self._get_entanglement_matrix()
        C_np = C.detach().cpu().numpy()

        # Find most entangled pairs
        upper_tri = np.triu(C_np, k=1)
        flat_indices = np.argsort(upper_tri.flatten())[::-1][:10]
        top_pairs = [
            (idx // self.dim, idx % self.dim, upper_tri.flatten()[idx])
            for idx in flat_indices
        ]

        return {
            'mean_correlation': C_np.mean(),
            'max_correlation': C_np.max(),
            'sparsity': (C_np < 0.01).mean(),
            'top_entangled_pairs': top_pairs,
            'correlation_strength_mean': self.correlation_strength.mean().item()
        }


class MultiHeadEntanglement(nn.Module):
    """
    Multi-head entanglement layer, similar to multi-head attention.

    Different heads can learn different types of correlations.

    Parameters
    ----------
    dim : int, default=512
        Feature dimension
    num_heads : int, default=4
        Number of entanglement heads
    """

    def __init__(self, dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Create multiple entanglement heads
        self.heads = nn.ModuleList([
            EntanglementLayer(
                dim=self.head_dim,
                correlation_type='multiplicative',
                learnable_pairs=True,
                residual=False
            )
            for _ in range(num_heads)
        ])

        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all heads.
        """
        # Split into heads
        chunks = x.chunk(self.num_heads, dim=-1)

        # Process each head
        outputs = [head(chunk) for head, chunk in zip(self.heads, chunks)]

        # Concatenate and project
        concat = torch.cat(outputs, dim=-1)
        output = self.output_proj(concat)
        output = self.layer_norm(output)

        # Residual
        return output + x
