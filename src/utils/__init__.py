"""
Utility Functions for Quantum-Enhanced World Models.

Author: Saurabh Jalendra
Institution: BITS Pilani (WILP Division)
Date: November 2025
"""

import os
import random
import time
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value
    deterministic : bool
        Whether to enable deterministic algorithms (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                pass

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available compute device.

    Parameters
    ----------
    prefer_gpu : bool
        Whether to prefer GPU over CPU if available

    Returns
    -------
    torch.device
        The selected compute device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    return device


@dataclass
class MetricLogger:
    """
    Simple metric logger for tracking training progress.

    Attributes
    ----------
    name : str
        Name of the logger
    metrics : Dict[str, List[float]]
        Dictionary of metric names to values
    """
    name: str
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def log(self, **kwargs) -> None:
        """Log metric values."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get(self, key: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics[key]

    def get_last(self, key: str, n: int = 1) -> Union[float, List[float]]:
        """Get last n values for a metric."""
        values = self.metrics[key][-n:]
        return values[0] if n == 1 else values

    def get_mean(self, key: str, n: Optional[int] = None) -> float:
        """Get mean of last n values (or all if n is None)."""
        values = self.metrics[key]
        if n is not None:
            values = values[-n:]
        return np.mean(values) if values else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame."""
        return pd.DataFrame(dict(self.metrics))

    def save(self, path: Union[str, Path]) -> None:
        """Save metrics to CSV file."""
        self.to_dataframe().to_csv(path, index=False)

    def __repr__(self) -> str:
        return f"MetricLogger(name='{self.name}', metrics={list(self.metrics.keys())})"


class Timer:
    """
    Reusable timer class for tracking multiple operations.

    Examples
    --------
    >>> t = Timer()
    >>> t.start()
    >>> # ... do something ...
    >>> elapsed = t.stop()
    """

    def __init__(self):
        self._start_time = None
        self._elapsed = 0.0
        self._running = False

    def start(self) -> 'Timer':
        """Start the timer."""
        if not self._running:
            self._start_time = time.perf_counter()
            self._running = True
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self._running:
            self._elapsed = time.perf_counter() - self._start_time
            self._running = False
        return self._elapsed

    def reset(self) -> 'Timer':
        """Reset the timer."""
        self._start_time = None
        self._elapsed = 0.0
        self._running = False
        return self

    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self._running:
            return time.perf_counter() - self._start_time
        return self._elapsed


@contextlib.contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    name : str
        Name of the operation being timed

    Yields
    ------
    Dict with timing info
    """
    start = time.perf_counter()
    timing_info = {"name": name}
    try:
        yield timing_info
    finally:
        elapsed = time.perf_counter() - start
        timing_info["elapsed"] = elapsed
        print(f"{name}: {elapsed:.4f}s")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to YAML configuration file

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    path : Union[str, Path]
        Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Color palette for visualizations
COLORS = {
    "baseline": "#2ecc71",      # Green
    "qaoa": "#3498db",          # Blue
    "superposition": "#9b59b6", # Purple
    "gates": "#e74c3c",         # Red
    "error_correction": "#f39c12",  # Orange
    "integrated": "#1abc9c",    # Teal
}
