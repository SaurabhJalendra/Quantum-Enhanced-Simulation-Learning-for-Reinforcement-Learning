#!/usr/bin/env python3
"""
COMPREHENSIVE CONSISTENCY VALIDATION SCRIPT

This script checks ALL notebooks for consistency in:
1. Model architecture (stoch_dim, deter_dim, hidden_dim)
2. Training parameters (learning_rate, batch_size, seq_len, num_epochs)
3. Experiment settings (seeds, num_episodes)

RUN THIS SCRIPT BEFORE EVERY COMMIT OR AFTER ANY CHANGES.

Usage:
    python scripts/validate_consistency.py

Author: Saurabh Jalendra
Project: Quantum-Enhanced Simulation Learning for Reinforcement Learning
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# EXPECTED VALUES - SINGLE SOURCE OF TRUTH
# Optimized for RTX 5090 (30GB VRAM)
# =============================================================================
EXPECTED = {
    # Architecture (CRITICAL - must match exactly)
    'stoch_dim': 64,
    'deter_dim': 512,
    'hidden_dim': 512,

    # Training (CRITICAL for fair comparison)
    'batch_size': 32,           # Larger batch for RTX 5090
    'seq_len': 20,
    'num_epochs': 50,
    'learning_rate': 3e-4,      # 0.0003

    # Experiment
    'num_episodes': 100,        # More episodes for better training
    'seeds': [42, 123, 456, 789, 1024],
}

# Tolerance for floating point comparison
LR_TOLERANCE = 1e-5


# =============================================================================
# VALIDATION RESULTS
# =============================================================================
@dataclass
class ValidationResult:
    """Result of validating a single notebook."""
    notebook: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    found_values: Dict[str, Any]


# =============================================================================
# NOTEBOOK PARSER
# =============================================================================
def extract_values_from_notebook(notebook_path: Path) -> Dict[str, List[Any]]:
    """Extract all configuration values from a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = {
        'stoch_dim': [],
        'deter_dim': [],
        'hidden_dim': [],
        'batch_size': [],
        'seq_len': [],
        'num_epochs': [],
        'learning_rate': [],
        'num_episodes': [],
        'uses_shared_config': False,
    }

    full_content = ''
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            full_content += source + '\n'

    # Check if notebook imports shared_config
    if 'from src.config' in full_content or 'shared_config' in full_content:
        found['uses_shared_config'] = True

    # Extract architecture values (multiple patterns)
    arch_patterns = {
        'stoch_dim': [
            r'stoch_dim\s*[=:]\s*(\d+)',
            r"'stoch_dim',\s*(\d+)",
            r'"stoch_dim",\s*(\d+)',
        ],
        'deter_dim': [
            r'deter_dim\s*[=:]\s*(\d+)',
            r"'deter_dim',\s*(\d+)",
            r'"deter_dim",\s*(\d+)',
        ],
        'hidden_dim': [
            r'hidden_dim\s*[=:]\s*(\d+)',
            r"'hidden_dim',\s*(\d+)",
            r'"hidden_dim",\s*(\d+)',
        ],
    }

    for key, patterns in arch_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, full_content)
            found[key].extend([int(m) for m in matches])
        found[key] = list(set(found[key]))

    # Extract training values
    training_patterns = {
        'batch_size': r'batch_size\s*[=:]\s*(\d+)',
        'seq_len': r'seq_len\s*[=:]\s*(\d+)',
        'num_episodes': r'num_episodes\s*[=:]\s*(\d+)',
    }

    for key, pattern in training_patterns.items():
        matches = re.findall(pattern, full_content)
        found[key] = list(set([int(m) for m in matches]))

    # Extract num_epochs (multiple patterns)
    epoch_patterns = [
        r'num_epochs?\s*[=:]\s*(\d+)',
        r'for\s+epoch\s+in\s+range\((\d+)\)',
        r'epochs?\s*=\s*(\d+)',
    ]
    for pattern in epoch_patterns:
        matches = re.findall(pattern, full_content)
        found['num_epochs'].extend([int(m) for m in matches])
    found['num_epochs'] = list(set(found['num_epochs']))

    # Extract learning rate
    lr_patterns = [
        r'learning_rate\s*[=:]\s*([\d.e-]+)',
        r'lr\s*[=:]\s*([\d.e-]+)',
    ]
    for pattern in lr_patterns:
        matches = re.findall(pattern, full_content)
        for m in matches:
            try:
                found['learning_rate'].append(float(m))
            except:
                pass
    found['learning_rate'] = list(set(found['learning_rate']))

    return found


def validate_notebook(notebook_path: Path) -> ValidationResult:
    """Validate a single notebook for consistency."""
    notebook_name = notebook_path.name
    errors = []
    warnings = []

    # Skip non-experiment notebooks
    skip_notebooks = ['01_environment_setup.ipynb', '09_results_analysis.ipynb']
    if notebook_name in skip_notebooks:
        return ValidationResult(
            notebook=notebook_name,
            passed=True,
            errors=[],
            warnings=["Skipped (setup/analysis notebook)"],
            found_values={}
        )

    try:
        found = extract_values_from_notebook(notebook_path)
    except Exception as e:
        return ValidationResult(
            notebook=notebook_name,
            passed=False,
            errors=[f"Failed to parse: {e}"],
            warnings=[],
            found_values={}
        )

    # Check if using shared_config
    if not found['uses_shared_config']:
        warnings.append("Does not import from src.config.shared_config")

    # ==========================================================================
    # ARCHITECTURE VALIDATION (CRITICAL)
    # ==========================================================================
    arch_params = ['stoch_dim', 'deter_dim', 'hidden_dim']
    for param in arch_params:
        values = found[param]
        expected = EXPECTED[param]

        if not values:
            warnings.append(f"No {param} found (may use shared_config)")
        else:
            wrong_values = [v for v in values if v != expected]
            if wrong_values:
                errors.append(f"ARCHITECTURE: {param}={wrong_values} (expected {expected})")

    # ==========================================================================
    # TRAINING PARAMETER VALIDATION
    # ==========================================================================

    # Batch size
    if found['batch_size']:
        wrong = [v for v in found['batch_size'] if v != EXPECTED['batch_size']]
        if wrong:
            errors.append(f"TRAINING: batch_size={wrong} (expected {EXPECTED['batch_size']})")

    # Sequence length
    if found['seq_len']:
        wrong = [v for v in found['seq_len'] if v != EXPECTED['seq_len']]
        if wrong:
            errors.append(f"TRAINING: seq_len={wrong} (expected {EXPECTED['seq_len']})")

    # Number of epochs
    if found['num_epochs']:
        wrong = [v for v in found['num_epochs'] if v != EXPECTED['num_epochs']]
        if wrong:
            errors.append(f"TRAINING: num_epochs={wrong} (expected {EXPECTED['num_epochs']})")

    # Learning rate (with tolerance)
    if found['learning_rate']:
        expected_lr = EXPECTED['learning_rate']
        wrong = [v for v in found['learning_rate'] if abs(v - expected_lr) > LR_TOLERANCE]
        if wrong:
            errors.append(f"TRAINING: learning_rate={wrong} (expected {expected_lr})")

    # Number of episodes
    if found['num_episodes']:
        wrong = [v for v in found['num_episodes'] if v != EXPECTED['num_episodes']]
        if wrong:
            errors.append(f"EXPERIMENT: num_episodes={wrong} (expected {EXPECTED['num_episodes']})")

    passed = len(errors) == 0

    return ValidationResult(
        notebook=notebook_name,
        passed=passed,
        errors=errors,
        warnings=warnings,
        found_values=found
    )


def validate_all_notebooks() -> Tuple[List[ValidationResult], bool]:
    """Validate all notebooks in the notebooks directory."""
    notebooks_dir = PROJECT_ROOT / 'notebooks'

    if not notebooks_dir.exists():
        print(f"ERROR: Notebooks directory not found: {notebooks_dir}")
        return [], False

    notebook_files = sorted(notebooks_dir.glob('*.ipynb'))

    if not notebook_files:
        print(f"ERROR: No notebooks found in {notebooks_dir}")
        return [], False

    results = []
    for nb_path in notebook_files:
        result = validate_notebook(nb_path)
        results.append(result)

    all_passed = all(r.passed for r in results)
    return results, all_passed


def print_report(results: List[ValidationResult], all_passed: bool):
    """Print a formatted validation report."""
    print("=" * 80)
    print("COMPREHENSIVE NOTEBOOK CONSISTENCY VALIDATION REPORT")
    print("=" * 80)
    print()

    # Print expected values
    print("EXPECTED VALUES (from shared_config):")
    print("  Architecture:")
    print(f"    stoch_dim:     {EXPECTED['stoch_dim']}")
    print(f"    deter_dim:     {EXPECTED['deter_dim']}")
    print(f"    hidden_dim:    {EXPECTED['hidden_dim']}")
    print("  Training:")
    print(f"    batch_size:    {EXPECTED['batch_size']}")
    print(f"    seq_len:       {EXPECTED['seq_len']}")
    print(f"    num_epochs:    {EXPECTED['num_epochs']}")
    print(f"    learning_rate: {EXPECTED['learning_rate']}")
    print("  Experiment:")
    print(f"    num_episodes:  {EXPECTED['num_episodes']}")
    print()

    # Print results for each notebook
    print("-" * 80)
    print("NOTEBOOK RESULTS:")
    print("-" * 80)

    for result in results:
        status_symbol = "[OK]" if result.passed else "[XX]"

        print(f"\n{status_symbol} {result.notebook}")

        if result.errors:
            for error in result.errors:
                print(f"    ERROR: {error}")

        if result.warnings and not result.errors:
            for warning in result.warnings:
                print(f"    WARNING: {warning}")

    # Print summary
    print()
    print("=" * 80)
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    if all_passed:
        print(f"RESULT: ALL {total_count} NOTEBOOKS PASSED")
        print("All notebooks are consistent with shared configuration.")
    else:
        print(f"RESULT: {passed_count}/{total_count} NOTEBOOKS PASSED")
        failed = [r.notebook for r in results if not r.passed]
        print(f"FAILED NOTEBOOKS: {', '.join(failed)}")
        print()
        print("ACTION REQUIRED:")
        print("  1. Fix the failed notebooks to use correct values")
        print("  2. Import from src.config.shared_config")
        print("  3. Re-run this validation script")

    print("=" * 80)


def main():
    """Main entry point."""
    print()
    print("Running comprehensive notebook consistency validation...")
    print()

    results, all_passed = validate_all_notebooks()

    if not results:
        print("No results to report.")
        sys.exit(1)

    print_report(results, all_passed)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
