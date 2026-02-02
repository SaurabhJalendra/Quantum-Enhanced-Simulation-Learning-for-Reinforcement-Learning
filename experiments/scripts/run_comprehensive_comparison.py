#!/usr/bin/env python3
"""
Comprehensive comparison of all quantum-inspired approaches vs baseline.

Loads results from Phase 1 (CartPole), Phase 2 (DMControl), and Phase 3 (Atari),
generates summary tables, statistical tests, and figures.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "experiments" / "figures"

# Phase 1 approach directories and display names
PHASE1_APPROACHES = {
    "baseline": "Baseline",
    "quantum_tunneling": "Quantum Tunneling",
    "superposition_proper": "Superposition",
    "entanglement_proper": "Entanglement",
    "interference_ensemble": "Interference Ensemble",
}

# Phase 2 approach keys (inside summary dict) and display names
PHASE2_APPROACHES = {
    "baseline": "Baseline",
    "quantum_tunneling": "Quantum Tunneling",
    "superposition": "Superposition",
    "entanglement": "Entanglement",
    "interference_ensemble": "Interference Ensemble",
}

PHASE2_ENVS = ["walker", "cheetah", "reacher"]
PHASE3_GAMES = ["pong", "breakout"]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
COLORS = {
    "Baseline": "#4C72B0",
    "Quantum Tunneling": "#DD8452",
    "Superposition": "#55A868",
    "Entanglement": "#C44E52",
    "Interference Ensemble": "#8172B3",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _extract_phase1_test_mses(data: Dict[str, Any]) -> Optional[List[float]]:
    """Extract per-seed test MSEs from a Phase 1 complete_metrics.json.

    Different notebooks stored results under slightly different keys.
    This function tries all known layouts.
    """
    # Layout A (baseline): raw_results.test_mses
    raw = data.get("raw_results", {})
    if isinstance(raw, dict) and "test_mses" in raw:
        return raw["test_mses"]

    # Layout B (entanglement_proper): results.test_mses
    res = data.get("results", {})
    if isinstance(res, dict) and "test_mses" in res:
        return res["test_mses"]

    # Layout C (superposition_proper / interference_ensemble): raw_results.test_mses
    # already covered by Layout A

    return None


def _extract_phase1_summary(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (test_mse_mean, test_mse_std) from Phase 1 data."""
    # Try prediction_accuracy (baseline style)
    pa = data.get("prediction_accuracy", {})
    if "test_mse_mean" in pa:
        return pa["test_mse_mean"], pa.get("test_mse_std")

    # Try final_performance (quantum_tunneling / superposition_proper style)
    fp = data.get("final_performance", {})
    if "test_mse_mean" in fp:
        return fp["test_mse_mean"], fp.get("test_mse_std")

    # Try results (entanglement_proper style)
    res = data.get("results", {})
    if "test_mse_mean" in res:
        return res["test_mse_mean"], res.get("test_mse_std")

    return None, None


# ---------------------------------------------------------------------------
# Phase 1 loading
# ---------------------------------------------------------------------------

def load_phase1() -> Dict[str, Dict[str, Any]]:
    """Load Phase 1 CartPole results.

    Returns dict mapping display_name -> {mean, std, per_seed}.
    """
    results = {}
    for key, name in PHASE1_APPROACHES.items():
        path = RESULTS_DIR / key / "complete_metrics.json"
        data = load_json(path)
        if data is None:
            continue
        mean, std = _extract_phase1_summary(data)
        per_seed = _extract_phase1_test_mses(data)
        if mean is not None:
            results[name] = {"mean": mean, "std": std, "per_seed": per_seed}
    return results


# ---------------------------------------------------------------------------
# Phase 2 loading
# ---------------------------------------------------------------------------

def load_phase2() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load Phase 2 DMControl results.

    Returns dict mapping env -> {display_name -> {mean, std, per_seed}}.
    """
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for env in PHASE2_ENVS:
        path = RESULTS_DIR / "phase2" / env / "complete_metrics.json"
        data = load_json(path)
        if data is None:
            continue
        summary = data.get("summary", {})
        raw_list = data.get("raw_results", [])

        env_results: Dict[str, Dict[str, Any]] = {}
        for key, name in PHASE2_APPROACHES.items():
            s = summary.get(key)
            if s is None:
                continue
            per_seed = [
                r["test_obs_mse"]
                for r in raw_list
                if r.get("approach") == key and "test_obs_mse" in r
            ]
            env_results[name] = {
                "mean": s["test_obs_mse_mean"],
                "std": s["test_obs_mse_std"],
                "per_seed": per_seed if per_seed else None,
            }
        all_results[env] = env_results
    return all_results


# ---------------------------------------------------------------------------
# Phase 3 loading
# ---------------------------------------------------------------------------

def load_phase3() -> Dict[str, Optional[Dict[str, Dict[str, Any]]]]:
    """Load Phase 3 Atari results (may not exist)."""
    all_results: Dict[str, Optional[Dict[str, Dict[str, Any]]]] = {}
    for game in PHASE3_GAMES:
        path = RESULTS_DIR / "phase3" / game / "complete_metrics.json"
        data = load_json(path)
        if data is None:
            all_results[game] = None
            continue
        # Assume same format as Phase 2
        summary = data.get("summary", {})
        raw_list = data.get("raw_results", [])
        game_results: Dict[str, Dict[str, Any]] = {}
        for key, name in PHASE2_APPROACHES.items():
            s = summary.get(key)
            if s is None:
                continue
            per_seed = [
                r["test_obs_mse"]
                for r in raw_list
                if r.get("approach") == key and "test_obs_mse" in r
            ]
            game_results[name] = {
                "mean": s.get("test_obs_mse_mean"),
                "std": s.get("test_obs_mse_std"),
                "per_seed": per_seed if per_seed else None,
            }
        all_results[game] = game_results if game_results else None
    return all_results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mann_whitney_vs_baseline(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Run Mann-Whitney U tests for each approach vs Baseline.

    Returns dict mapping approach_name -> {U, p, significant, effect_direction}.
    """
    baseline = results.get("Baseline")
    if baseline is None or baseline.get("per_seed") is None:
        return {}
    base_vals = np.array(baseline["per_seed"])
    tests: Dict[str, Dict[str, Any]] = {}
    for name, info in results.items():
        if name == "Baseline":
            continue
        vals = info.get("per_seed")
        if vals is None or len(vals) < 2:
            continue
        arr = np.array(vals)
        try:
            u_stat, p_val = stats.mannwhitneyu(arr, base_vals, alternative="two-sided")
        except ValueError:
            continue
        direction = "better" if np.mean(arr) < np.mean(base_vals) else "worse"
        tests[name] = {
            "U": u_stat,
            "p": p_val,
            "significant": p_val < 0.05,
            "direction": direction,
        }
    return tests


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

HEADER_WIDTH = 90


def print_header(title: str) -> None:
    print()
    print("=" * HEADER_WIDTH)
    print(f"  {title}")
    print("=" * HEADER_WIDTH)


def print_table(
    rows: List[Tuple[str, str]],
    col_headers: Tuple[str, str] = ("Approach", "Test MSE (mean +/- std)"),
) -> None:
    """Print a simple two-column table."""
    w0 = max(len(col_headers[0]), max(len(r[0]) for r in rows)) + 2
    w1 = max(len(col_headers[1]), max(len(r[1]) for r in rows)) + 2
    fmt = f"  {{:<{w0}}} {{:<{w1}}}"
    print(fmt.format(*col_headers))
    print(fmt.format("-" * w0, "-" * w1))
    for r in rows:
        print(fmt.format(*r))


def format_mse(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "N/A"
    if std is not None:
        return f"{mean:.6f} +/- {std:.6f}"
    return f"{mean:.6f}"


def print_stat_tests(tests: Dict[str, Dict[str, Any]], label: str) -> None:
    if not tests:
        return
    print(f"\n  Mann-Whitney U tests vs Baseline ({label}):")
    print(f"  {'Approach':<25} {'U':>8} {'p-value':>12} {'Sig.':>6} {'Direction':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*6} {'-'*10}")
    for name, t in tests.items():
        sig = "YES" if t["significant"] else "no"
        print(
            f"  {name:<25} {t['U']:>8.1f} {t['p']:>12.6f} {sig:>6} {t['direction']:>10}"
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_bar_chart(
    results: Dict[str, Dict[str, Any]],
    title: str,
    filename: str,
    ylabel: str = "Test MSE",
) -> None:
    """Save a grouped bar chart comparing approaches."""
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n].get("std") or 0 for n in names]
    colors = [COLORS.get(n, "#999999") for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / filename}")


def save_phase2_grouped(
    phase2: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Save a grouped bar chart for all Phase 2 environments."""
    if not phase2:
        return
    approach_names = list(PHASE2_APPROACHES.values())
    envs = list(phase2.keys())

    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        env_data = phase2[env]
        names = [n for n in approach_names if n in env_data]
        means = [env_data[n]["mean"] for n in names]
        stds = [env_data[n].get("std") or 0 for n in names]
        colors = [COLORS.get(n, "#999999") for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{env.capitalize()}")
        ax.set_ylabel("Test Obs MSE")

    fig.suptitle("Phase 2: DMControl Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'phase2_comparison.png'}")


def save_cross_phase_ranking(
    phase1: Dict[str, Dict[str, Any]],
    phase2: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Rank approaches by how often they beat baseline, save figure."""
    approach_names = [
        "Quantum Tunneling",
        "Superposition",
        "Entanglement",
        "Interference Ensemble",
    ]

    # Count wins (lower MSE than baseline) across environments
    wins: Dict[str, int] = {n: 0 for n in approach_names}
    total_envs = 0

    # Phase 1
    if "Baseline" in phase1:
        base_mean = phase1["Baseline"]["mean"]
        total_envs += 1
        for n in approach_names:
            if n in phase1 and phase1[n]["mean"] < base_mean:
                wins[n] += 1

    # Phase 2
    for env, env_data in phase2.items():
        if "Baseline" not in env_data:
            continue
        base_mean = env_data["Baseline"]["mean"]
        total_envs += 1
        for n in approach_names:
            if n in env_data and env_data[n]["mean"] < base_mean:
                wins[n] += 1

    if total_envs == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(approach_names))
    colors = [COLORS.get(n, "#999999") for n in approach_names]
    bars = ax.bar(x, [wins[n] for n in approach_names], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(approach_names, rotation=20, ha="right")
    ax.set_ylabel("Environments where approach beats baseline")
    ax.set_title(f"Cross-Phase Ranking (out of {total_envs} environments)")
    ax.set_ylim(0, total_envs + 0.5)
    for bar, n in zip(bars, approach_names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(wins[n]), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_phase_ranking.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'cross_phase_ranking.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: CartPole
    # ------------------------------------------------------------------
    print_header("PHASE 1: CartPole-v1 Results")
    phase1 = load_phase1()
    if phase1:
        rows = [
            (name, format_mse(info["mean"], info.get("std")))
            for name, info in phase1.items()
        ]
        print_table(rows)
        stat_tests_p1 = mann_whitney_vs_baseline(phase1)
        print_stat_tests(stat_tests_p1, "CartPole")
        save_bar_chart(phase1, "Phase 1: CartPole Test MSE", "phase1_comparison.png")
    else:
        print("  No Phase 1 results found.")

    # ------------------------------------------------------------------
    # Phase 2: DMControl
    # ------------------------------------------------------------------
    print_header("PHASE 2: DMControl Suite Results")
    phase2 = load_phase2()
    all_p2_tests: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if phase2:
        for env, env_data in phase2.items():
            print(f"\n  --- {env.upper()} ---")
            rows = [
                (name, format_mse(info["mean"], info.get("std")))
                for name, info in env_data.items()
            ]
            print_table(rows, ("Approach", "Test Obs MSE (mean +/- std)"))
            t = mann_whitney_vs_baseline(env_data)
            all_p2_tests[env] = t
            print_stat_tests(t, env.capitalize())
        save_phase2_grouped(phase2)
    else:
        print("  No Phase 2 results found.")

    # ------------------------------------------------------------------
    # Phase 3: Atari
    # ------------------------------------------------------------------
    print_header("PHASE 3: Atari Results")
    phase3 = load_phase3()
    any_phase3 = False
    for game in PHASE3_GAMES:
        game_data = phase3.get(game)
        if game_data:
            any_phase3 = True
            print(f"\n  --- {game.upper()} ---")
            rows = [
                (name, format_mse(info["mean"], info.get("std")))
                for name, info in game_data.items()
            ]
            print_table(rows, ("Approach", "Test Obs MSE (mean +/- std)"))
            t = mann_whitney_vs_baseline(game_data)
            print_stat_tests(t, game.capitalize())
        else:
            print(f"\n  {game.capitalize()}: pending (no complete_metrics.json)")
    if not any_phase3:
        print("  Phase 3: pending")

    # ------------------------------------------------------------------
    # Cross-Phase Ranking
    # ------------------------------------------------------------------
    print_header("CROSS-PHASE RANKING")
    save_cross_phase_ranking(phase1, phase2)

    # Build ranking table: for each approach, show mean MSE relative to baseline
    approach_names = [
        "Quantum Tunneling",
        "Superposition",
        "Entanglement",
        "Interference Ensemble",
    ]
    ratios: Dict[str, List[float]] = {n: [] for n in approach_names}

    if "Baseline" in phase1:
        bm = phase1["Baseline"]["mean"]
        for n in approach_names:
            if n in phase1:
                ratios[n].append(phase1[n]["mean"] / bm)

    for env, env_data in phase2.items():
        if "Baseline" not in env_data:
            continue
        bm = env_data["Baseline"]["mean"]
        for n in approach_names:
            if n in env_data:
                ratios[n].append(env_data[n]["mean"] / bm)

    print(f"\n  {'Approach':<25} {'Avg MSE Ratio vs Baseline':>28} {'Verdict':>12}")
    print(f"  {'-'*25} {'-'*28} {'-'*12}")
    for n in approach_names:
        if ratios[n]:
            avg_ratio = np.mean(ratios[n])
            verdict = "BETTER" if avg_ratio < 1.0 else "WORSE"
            print(f"  {n:<25} {avg_ratio:>28.4f} {verdict:>12}")
        else:
            print(f"  {n:<25} {'N/A':>28} {'N/A':>12}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_header("KEY FINDINGS")

    # Determine best approach per environment
    all_envs: Dict[str, Dict[str, float]] = {}
    if phase1:
        all_envs["CartPole"] = {n: info["mean"] for n, info in phase1.items()}
    for env, env_data in phase2.items():
        all_envs[env.capitalize()] = {n: info["mean"] for n, info in env_data.items()}

    print()
    for env_name, scores in all_envs.items():
        best_name = min(scores, key=scores.get)  # type: ignore[arg-type]
        best_val = scores[best_name]
        print(f"  {env_name:<12}: Best = {best_name} (MSE = {best_val:.6f})")

    # Count statistically significant improvements
    sig_improvements = 0
    sig_degradations = 0
    all_tests = {}
    if phase1:
        all_tests["CartPole"] = mann_whitney_vs_baseline(phase1)
    for env, t in all_p2_tests.items():
        all_tests[env] = t
    for env_label, tests in all_tests.items():
        for name, t in tests.items():
            if t["significant"]:
                if t["direction"] == "better":
                    sig_improvements += 1
                else:
                    sig_degradations += 1

    print()
    print(f"  Statistically significant improvements over baseline: {sig_improvements}")
    print(f"  Statistically significant degradations vs baseline:   {sig_degradations}")
    print()

    # Interference ensemble special note
    if phase2:
        ie_wins = sum(
            1
            for env_data in phase2.values()
            if "Interference Ensemble" in env_data
            and "Baseline" in env_data
            and env_data["Interference Ensemble"]["mean"] < env_data["Baseline"]["mean"]
        )
        print(
            f"  Interference Ensemble beats baseline in {ie_wins}/{len(phase2)} "
            f"DMControl environments (obs MSE)."
        )
        print(
            "  Note: Interference Ensemble uses ~5x more parameters and ~2-3x more time."
        )

    print()
    print("  Done. Figures saved to experiments/figures/")
    print()


if __name__ == "__main__":
    main()
