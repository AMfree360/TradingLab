"""Professional-grade Monte Carlo validation suite.

This module provides three independent Monte Carlo engines:
1. Permutation Test - Tests if trade outcome distribution is better than random order
2. Block Bootstrap - Preserves time dependencies while testing robustness
3. Randomized Entry - Tests if strategy is better than random entry baseline

All engines use the same risk model and exit logic to ensure fair comparison.
"""

from .permutation import MonteCarloPermutation, PermutationResult
from .block_bootstrap import MonteCarloBlockBootstrap, BootstrapResult
from .randomized_entry import MonteCarloRandomizedEntry, RandomizedEntryResult
from .runner import MonteCarloSuite, MonteCarloSuiteResult

# Backward compatibility: Import MonteCarloResult from old module
# Since Python treats monte_carlo as a package (due to this __init__.py),
# we need to import the old class from the legacy file directly
import importlib.util
from pathlib import Path

try:
    # Load the old monte_carlo.py file (sibling to this package directory)
    _old_file = Path(__file__).parent.parent / 'monte_carlo.py'
    if _old_file.exists():
        spec = importlib.util.spec_from_file_location(
            "_monte_carlo_legacy",
            str(_old_file)
        )
        if spec and spec.loader:
            _legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_legacy)
            MonteCarloResult = _legacy.MonteCarloResult
        else:
            raise ImportError("Could not load legacy module")
    else:
        raise ImportError("Legacy file not found")
except (ImportError, AttributeError, Exception):
    # Fallback: create a compatibility class with similar structure
    from dataclasses import dataclass
    from typing import Dict
    import numpy as np
    
    @dataclass
    class MonteCarloResult:
        """Legacy MonteCarloResult for backward compatibility."""
        observed_metric: float
        permuted_values: np.ndarray
        p_value: float
        percentile_rank: float
        observed_metrics: Dict
        n_iterations: int
        metric_name: str

__all__ = [
    'MonteCarloPermutation',
    'MonteCarloBlockBootstrap',
    'MonteCarloRandomizedEntry',
    'MonteCarloSuite',
    'MonteCarloResult',  # Backward compatibility
    'PermutationResult',
    'BootstrapResult',
    'RandomizedEntryResult',
    'MonteCarloSuiteResult',
]

