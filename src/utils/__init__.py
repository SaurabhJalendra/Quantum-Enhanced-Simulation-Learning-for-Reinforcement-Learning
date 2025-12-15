
"""
Utility Functions for Quantum-Enhanced World Models.

Author: Saurabh Jalendra
Institution: BITS Pilani (WILP Division)
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
        Whether to enable deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available compute device.

    Parameters
    ----------
    prefer_gpu : bool
        Whether to prefer GPU over CPU

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
    """Simple metric logger for tracking training progress."""
    name: str
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def log(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_mean(self, key: str, n: Optional[int] = None) -> float:
        values = self.metrics[key]
        if n is not None:
            values = values[-n:]
        return np.mean(values) if values else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self.metrics))

    def save(self, path: Union[str, Path]) -> None:
        self.to_dataframe().to_csv(path, index=False)


@contextlib.contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    timing_info = {"name": name}
    try:
        yield timing_info
    finally:
        elapsed = time.perf_counter() - start
        timing_info["elapsed"] = elapsed
        print(f"{name}: {elapsed:.4f}s")


class Timer:
    """Simple timer class for measuring elapsed time."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> 'Timer':
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time
            self._start_time = None
        return self._elapsed

    def elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._elapsed


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# Color palette for different approaches (for consistent visualizations)
COLORS = {
    "baseline": "#2ecc71",      # Green
    "qaoa": "#3498db",          # Blue
    "superposition": "#9b59b6", # Purple
    "gates": "#e74c3c",         # Red
    "error_correction": "#f39c12",  # Orange
    "integrated": "#1abc9c",    # Teal
}
