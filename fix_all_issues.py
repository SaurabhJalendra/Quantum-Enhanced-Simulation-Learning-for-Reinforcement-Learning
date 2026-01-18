"""
FIX ALL ISSUES SCRIPT
=====================
Comprehensive fixes for all issues identified in the project audit.

Issues to fix:
1. Error correction hardcoded seeds in notebook 06 and 07
2. Notebook 08 collect_data seed parameter bug
3. Bonferroni alpha correction (0.025 -> 0.00625)
4. Action one-hot encoding verification

Author: Claude Code Audit
Date: January 18, 2026
"""

import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path("phase1_cartpole_notebooks")


def fix_notebook_07_ensemble_seeds():
    """
    Fix the hardcoded seeds in EnsembleModel in notebook 07.

    The bug: torch.manual_seed(42 + i * 1000) ignores experiment seed
    The fix: Pass base_seed parameter to __init__ and use it
    """
    notebook_path = NOTEBOOKS_DIR / "07_comprehensive_comparison.ipynb"

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_made = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Fix 1: Update EnsembleModel class to accept base_seed parameter
            if 'class EnsembleModel(nn.Module):' in source and 'torch.manual_seed(42 + i * 1000)' in source:
                # Replace the __init__ to accept base_seed
                old_init = '''def __init__(self, obs_dim, action_dim, num_models=5, hidden_dim=512,
                 deter_dim=512, stoch_dim=64):'''
                new_init = '''def __init__(self, obs_dim, action_dim, num_models=5, hidden_dim=512,
                 deter_dim=512, stoch_dim=64, base_seed=None):'''

                # Replace the hardcoded seed logic
                old_seed_logic = '''        # Initialize with different seeds
        for i, model in enumerate(self.models):
            torch.manual_seed(42 + i * 1000)
            for m in model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()'''

                new_seed_logic = '''        # Initialize with different seeds based on experiment seed
        # If base_seed is provided, use it; otherwise rely on global random state
        if base_seed is not None:
            for i, model in enumerate(self.models):
                torch.manual_seed(base_seed + i * 1000)
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
        # If no base_seed, models use current random state (set by set_seed())'''

                new_source = source.replace(old_init, new_init)
                new_source = new_source.replace(old_seed_logic, new_seed_logic)

                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    changes_made = True
                    print("  [FIXED] EnsembleModel class - now accepts base_seed parameter")

            # Fix 2: Update run_experiment to pass seed to EnsembleModel
            if "elif approach == 'error_correction':" in source and 'EnsembleModel(obs_dim, action_dim, num_models=5).to(device)' in source:
                old_ensemble_create = "model = EnsembleModel(obs_dim, action_dim, num_models=5).to(device)"
                new_ensemble_create = "model = EnsembleModel(obs_dim, action_dim, num_models=5, base_seed=seed).to(device)"

                new_source = source.replace(old_ensemble_create, new_ensemble_create)

                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    changes_made = True
                    print("  [FIXED] run_experiment - now passes seed to EnsembleModel")

            # Fix 3: Correct Bonferroni alpha
            if 'bonferroni_alpha = 0.025' in source:
                old_bonferroni = '''# Bonferroni correction: alpha = 0.05 / number of comparisons
# With 4 approaches vs baseline, bonferroni_alpha = 0.05 / 4 = 0.0125
# Using 0.025 to account for testing 2 metrics (loss and error)
bonferroni_alpha = 0.025'''

                new_bonferroni = '''# Bonferroni correction: alpha = 0.05 / number of comparisons
# With 4 approaches vs baseline and 2 metrics = 8 total comparisons
# bonferroni_alpha = 0.05 / 8 = 0.00625
bonferroni_alpha = 0.00625'''

                new_source = source.replace(old_bonferroni, new_bonferroni)

                # Also handle simpler version
                if new_source == source:
                    new_source = source.replace('bonferroni_alpha = 0.025', 'bonferroni_alpha = 0.00625  # Corrected: 0.05/8')

                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    changes_made = True
                    print("  [FIXED] Bonferroni alpha corrected to 0.00625")

    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  Saved: {notebook_path}")
    else:
        print(f"  No changes needed in {notebook_path}")

    return changes_made


def fix_notebook_06_ensemble_seeds():
    """
    Fix the hardcoded seeds in ErrorCorrectionEnsemble in notebook 06.

    The bug: self._reset_parameters(model, seed=42 + i) ignores experiment seed
    The fix: Accept base_seed in __init__ and use it
    """
    notebook_path = NOTEBOOKS_DIR / "06_error_correction.ipynb"

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_made = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Fix the ErrorCorrectionEnsemble class
            if 'class ErrorCorrectionEnsemble(nn.Module):' in source and 'self._reset_parameters(model, seed=42 + i)' in source:
                # Replace hardcoded seed
                old_seed = 'self._reset_parameters(model, seed=42 + i)'
                new_seed = '# Seed is set by set_seed() before model creation - no override needed\n            pass  # Models initialized with current random state'

                new_source = source.replace(old_seed, new_seed)

                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    changes_made = True
                    print("  [FIXED] ErrorCorrectionEnsemble - removed hardcoded seeds")

    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  Saved: {notebook_path}")
    else:
        print(f"  No changes needed in {notebook_path}")

    return changes_made


def fix_notebook_08_collect_data():
    """
    Fix the collect_data seed parameter bug in notebook 08.

    The bug: collect_data() function doesn't accept seed parameter but is called with seed=TEST_SEED
    The fix: Add seed parameter to collect_data function
    """
    notebook_path = NOTEBOOKS_DIR / "08_ablation_studies.ipynb"

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_made = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Fix 1: Update collect_data function to accept seed parameter
            if "def collect_data(env_name='CartPole-v1', num_episodes=100, max_steps=200):" in source:
                old_def = "def collect_data(env_name='CartPole-v1', num_episodes=100, max_steps=200):"
                new_def = "def collect_data(env_name='CartPole-v1', num_episodes=100, max_steps=200, seed=None):"

                # Also need to use the seed in env.reset()
                old_reset = "obs, _ = env.reset()"
                new_reset = "obs, _ = env.reset(seed=seed if ep == 0 else None)"

                new_source = source.replace(old_def, new_def)

                # Find and replace env.reset() pattern - need to handle the loop variable
                if 'for ep in range(num_episodes):' in new_source:
                    # More careful replacement for the reset inside the loop
                    new_source = new_source.replace(
                        'for ep in range(num_episodes):\n        obs, _ = env.reset()',
                        'for ep in range(num_episodes):\n        reset_seed = seed + ep if seed is not None else None\n        obs, _ = env.reset(seed=reset_seed)'
                    )

                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    changes_made = True
                    print("  [FIXED] collect_data function - now accepts seed parameter")

    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  Saved: {notebook_path}")
    else:
        print(f"  No changes needed or manual review required for {notebook_path}")

    return changes_made


def add_action_onehot_encoding_check():
    """
    Add a verification function to check action encoding.
    This creates a utility function that can be added to notebooks.
    """
    check_code = '''
def verify_action_encoding(actions, expected_dim=2):
    """
    Verify that actions are properly one-hot encoded.

    Parameters
    ----------
    actions : np.ndarray or torch.Tensor
        Action array to verify
    expected_dim : int
        Expected action dimension (2 for CartPole)

    Returns
    -------
    bool
        True if encoding is correct
    """
    import numpy as np
    import torch

    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()

    # Check shape
    if actions.ndim == 1:
        print(f"WARNING: Actions are 1D (shape {actions.shape})")
        print("Actions should be one-hot encoded to shape (batch, action_dim)")
        return False

    if actions.shape[-1] != expected_dim:
        print(f"WARNING: Action dim is {actions.shape[-1]}, expected {expected_dim}")
        return False

    # Check if one-hot
    row_sums = actions.sum(axis=-1)
    if not np.allclose(row_sums, 1.0):
        print(f"WARNING: Actions don't sum to 1.0 (one-hot requirement)")
        print(f"Row sums: min={row_sums.min():.3f}, max={row_sums.max():.3f}")
        return False

    print(f"Actions correctly one-hot encoded: shape={actions.shape}")
    return True


def one_hot_encode_actions(actions, num_actions=2):
    """
    Convert scalar actions to one-hot encoding.

    Parameters
    ----------
    actions : np.ndarray
        Scalar actions of shape (batch,) or (batch, seq_len)
    num_actions : int
        Number of possible actions

    Returns
    -------
    np.ndarray
        One-hot encoded actions
    """
    import numpy as np

    actions = np.asarray(actions, dtype=np.int64)
    original_shape = actions.shape
    actions_flat = actions.flatten()

    one_hot = np.zeros((len(actions_flat), num_actions), dtype=np.float32)
    one_hot[np.arange(len(actions_flat)), actions_flat] = 1.0

    # Reshape back
    new_shape = original_shape + (num_actions,)
    return one_hot.reshape(new_shape)
'''

    print("\n" + "="*60)
    print("ACTION ENCODING UTILITIES")
    print("="*60)
    print("Add the following functions to notebooks that need action encoding:")
    print(check_code)

    return check_code


def create_training_metrics_documentation():
    """
    Create documentation about training metrics equivalence.
    """
    doc = """
## Training Metrics Equivalence

### Steps vs Epochs

Some notebooks use `num_steps` (10,000) while others use `num_epochs` (50).

**Equivalence calculation:**
- Data: 100 episodes × ~21 avg steps = ~2,100 samples
- Batch size: 32
- Sequences per epoch: 2,100 / 32 ≈ 65 batches
- Steps in 50 epochs: 50 × 65 = 3,250 steps

**Conclusion:**
- 10,000 steps ≈ 154 epochs (more training)
- 50 epochs ≈ 3,250 steps (less training)

**These are NOT equivalent!** Notebooks using 10,000 steps train ~3x longer.

### Standardization Recommendation

For fair comparison, all notebooks should use the same training budget:
- Option A: All use `num_epochs=50` (3,250 steps equivalent)
- Option B: All use `num_steps=10000` (154 epochs equivalent)

The current inconsistency may explain some result discrepancies.
"""

    doc_path = Path("docs/training_metrics_equivalence.md")
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(doc_path, 'w') as f:
        f.write(doc)

    print(f"\nCreated documentation: {doc_path}")
    return doc


def main():
    """Run all fixes."""
    print("="*60)
    print("COMPREHENSIVE FIX SCRIPT")
    print("="*60)
    print("\nThis script fixes all issues identified in the project audit.\n")

    # Change to project root
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("\n1. FIXING NOTEBOOK 07 (Comprehensive Comparison)")
    print("-"*50)
    fix_notebook_07_ensemble_seeds()

    print("\n2. FIXING NOTEBOOK 06 (Error Correction)")
    print("-"*50)
    fix_notebook_06_ensemble_seeds()

    print("\n3. FIXING NOTEBOOK 08 (Ablation Studies)")
    print("-"*50)
    fix_notebook_08_collect_data()

    print("\n4. ACTION ENCODING UTILITIES")
    print("-"*50)
    add_action_onehot_encoding_check()

    print("\n5. TRAINING METRICS DOCUMENTATION")
    print("-"*50)
    create_training_metrics_documentation()

    print("\n" + "="*60)
    print("FIX SCRIPT COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the changes in each notebook")
    print("2. Re-run notebooks 06, 07, 08 to verify fixes")
    print("3. Commit changes to git")
    print("4. Update PROGRESS_LOG.md")


if __name__ == "__main__":
    main()
