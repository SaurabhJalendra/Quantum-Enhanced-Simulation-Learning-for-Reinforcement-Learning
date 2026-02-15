"""
Aggregate CartPole Phase 1 individual seed results into complete_metrics.json.

Reads individual seed JSON files from experiments/results/phase1/cartpole/
and produces a complete_metrics.json in the same format as Phase 2/3.
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_seed_results(results_dir: Path) -> dict:
    """Load all individual seed JSON files grouped by approach."""
    approaches = defaultdict(list)

    for f in sorted(results_dir.glob("*_seed_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        approach = data["approach"]
        approaches[approach].append(data)

    return dict(approaches)


def compute_summary(results: list) -> dict:
    """Compute summary statistics for a list of seed results."""
    test_obs = [r["test_obs_mse"] for r in results]
    train_obs = [r["train_obs_mse"] for r in results]
    test_reward = [r["test_reward_mse"] for r in results]
    times = [r["time_seconds"] for r in results]

    summary = {
        "test_obs_mse_mean": float(np.mean(test_obs)),
        "test_obs_mse_std": float(np.std(test_obs)),
        "train_obs_mse_mean": float(np.mean(train_obs)),
        "train_obs_mse_std": float(np.std(train_obs)),
        "test_reward_mse_mean": float(np.mean(test_reward)),
        "test_reward_mse_std": float(np.std(test_reward)),
        "time_mean": float(np.mean(times)),
        "num_params": results[0]["num_params"],
        "num_seeds": len(results),
    }
    return summary


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "experiments" / "results" / "phase1" / "cartpole"

    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        return

    approaches = load_seed_results(results_dir)

    print(f"Found {len(approaches)} approaches:")
    for name, results in approaches.items():
        print(f"  {name}: {len(results)} seeds")

    # Build complete_metrics in Phase 2/3 format
    complete_metrics = {
        "experiment": "phase1_cartpole",
        "environment": {
            "name": "CartPole-v1",
            "obs_dim": 4,
            "action_dim": 2,
            "action_type": "discrete",
        },
        "config": {
            "stoch_dim": 64,
            "deter_dim": 512,
            "hidden_dim": 512,
            "batch_size": 32,
            "seq_len": 20,
            "num_steps": 10000,
            "learning_rate": 0.0003,
            "kl_weight": 1.0,
            "grad_clip": 100.0,
            "num_episodes": 100,
            "seeds": [42, 123, 456, 789, 1024],
        },
        "summary": {},
        "raw_results": [],
    }

    for approach_name, results in approaches.items():
        complete_metrics["summary"][approach_name] = compute_summary(results)
        complete_metrics["raw_results"].extend(results)

    # Sort raw results by approach then seed
    complete_metrics["raw_results"].sort(
        key=lambda r: (r["approach"], r["seed"])
    )

    # Save
    output_path = results_dir / "complete_metrics.json"
    with open(output_path, "w") as f:
        json.dump(complete_metrics, f, indent=2)

    print(f"\nSaved complete_metrics.json to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CARTPOLE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Approach':<25} {'Test Obs MSE':>20} {'Test Reward MSE':>20} {'Time (s)':>12}")
    print("-" * 80)
    for name, summary in complete_metrics["summary"].items():
        obs = f"{summary['test_obs_mse_mean']:.4f} ± {summary['test_obs_mse_std']:.4f}"
        rew = f"{summary['test_reward_mse_mean']:.6f} ± {summary['test_reward_mse_std']:.6f}"
        time_str = f"{summary['time_mean']:.0f}"
        print(f"{name:<25} {obs:>20} {rew:>20} {time_str:>12}")


if __name__ == "__main__":
    main()
