"""
Script to create the Reacher-hard experiment notebook from the Pendulum template.
"""
import json
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
pendulum_path = PROJECT_ROOT / "phase1_cartpole_notebooks" / "10_pendulum_experiments.ipynb"
output_path = PROJECT_ROOT / "phase2_dmcontrol_notebooks" / "06_reacher_hard_experiments.ipynb"

with open(pendulum_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

rh = copy.deepcopy(nb)

# ============================================================
# Cell 0: Title markdown
# ============================================================
rh["cells"][0]["source"] = [
    "# Phase 2: Reacher-hard Experiments\n",
    "\n",
    "**Notebook:** `06_reacher_hard_experiments.ipynb`  \n",
    "**Phase:** 2 - DMControl Suite  \n",
    "**Author:** Saurabh Jalendra  \n",
    "\n",
    "## Objectives\n",
    "1. Run all 5 quantum-inspired approaches on Reacher-hard (DMControl)\n",
    "2. Collect multi-seed results (5 seeds each)\n",
    "3. Evaluate test set performance and long-horizon prediction\n",
    "4. Save results to experiments/results/phase2/reacher_hard/\n",
]

# ============================================================
# Cell 1: Imports & Config
# ============================================================
rh["cells"][1]["source"] = [
    '"""\n',
    "Cell: Imports and Configuration\n",
    "Purpose: Set up environment, imports, and experiment configuration\n",
    '"""\n',
    "import sys\n",
    "import os\n",
    "import json\n",
    "import time\n",
    "import traceback\n",
    "from pathlib import Path\n",
    "from typing import Dict, List, Tuple, Optional, Any, NamedTuple\n",
    "from dataclasses import dataclass, field\n",
    "from collections import defaultdict\n",
    "import warnings\n",
    'warnings.filterwarnings("ignore")\n',
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm.notebook import tqdm\n",
    "\n",
    "# DMControl Suite\n",
    "from dm_control import suite\n",
    "\n",
    "# Add project root to path\n",
    "PROJECT_ROOT = Path.cwd().parent\n",
    'sys.path.insert(0, str(PROJECT_ROOT / "src"))\n',
    "\n",
    "# Import quantum-inspired components\n",
    "from quantum_inspired.quantum_tunneling import QuantumTunnelingOptimizer\n",
    "from quantum_inspired.superposition_buffer import SuperpositionReplayBuffer\n",
    "from quantum_inspired.entanglement_layer import EntanglementLayer\n",
    "from quantum_inspired.interference_ensemble import InterferenceEnsemble\n",
    "\n",
    "# Device setup\n",
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
    'print(f"Device: {DEVICE}")\n',
    'if DEVICE.type == "cuda":\n',
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
    "\n",
    "# Configuration\n",
    "OBS_DIM = 6\n",
    "ACTION_DIM = 2\n",
    "STOCH_DIM = 64\n",
    "DETER_DIM = 512\n",
    "HIDDEN_DIM = 512\n",
    "STATE_DIM = DETER_DIM + STOCH_DIM\n",
    "\n",
    "NUM_STEPS = 10000\n",
    "BATCH_SIZE = 32\n",
    "SEQ_LEN = 20\n",
    "LEARNING_RATE = 3e-4\n",
    "KL_WEIGHT = 1.0\n",
    "GRAD_CLIP = 100.0\n",
    "NUM_EPISODES = 200  # More episodes for DMControl\n",
    "EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]\n",
    "\n",
    'APPROACHES = ["baseline", "quantum_tunneling", "superposition", "entanglement", "interference_ensemble"]\n',
    "\n",
    'RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "phase2" / "reacher_hard"\n',
    "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    'print(f"Configuration: obs_dim={OBS_DIM}, action_dim={ACTION_DIM}")\n',
    'print(f"Architecture: stoch={STOCH_DIM}, deter={DETER_DIM}, hidden={HIDDEN_DIM}")\n',
    'print(f"Training: steps={NUM_STEPS}, batch={BATCH_SIZE}, seq_len={SEQ_LEN}, lr={LEARNING_RATE}")\n',
    'print(f"Episodes: {NUM_EPISODES} (DMControl collects more for better coverage)")\n',
    'print(f"Seeds: {EXPERIMENT_SEEDS}")\n',
    'print(f"Results directory: {RESULTS_DIR}")\n',
]

# ============================================================
# Cell 2: Data Collection markdown
# ============================================================
rh["cells"][2]["source"] = [
    "---\n",
    "## Data Collection\n",
    "\n",
    "Collect Reacher-hard episodes from DMControl Suite using a random policy.\n",
    "Reacher-hard has obs_dim=6 (position, to_target, velocity) and action_dim=2.\n",
]

# ============================================================
# Cell 3: Data Collection code
# ============================================================
rh["cells"][3]["source"] = [
    '"""\n',
    "Cell: Data Collection and Replay Buffer\n",
    "Purpose: Collect Reacher-hard episodes using DMControl Suite\n",
    '"""\n',
    "\n",
    "def set_seed(seed):\n",
    '    """Set all random seeds for reproducibility."""\n',
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "\n",
    "@dataclass\n",
    "class Episode:\n",
    "    observations: np.ndarray\n",
    "    actions: np.ndarray\n",
    "    rewards: np.ndarray\n",
    "    dones: np.ndarray\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.observations)\n",
    "\n",
    "    @property\n",
    "    def total_reward(self):\n",
    "        return self.rewards.sum()\n",
    "\n",
    "\n",
    "def flatten_obs(obs):\n",
    '    """Flatten DMControl observation dictionary to array."""\n',
    "    parts = []\n",
    "    for key in sorted(obs.keys()):\n",
    "        parts.append(np.array(obs[key], dtype=np.float32).flatten())\n",
    "    return np.concatenate(parts)\n",
    "\n",
    "\n",
    "def collect_episodes(domain, task, num_episodes, seed):\n",
    '    """Collect episodes from DMControl environment using random policy."""\n',
    "    rng = np.random.RandomState(seed)\n",
    "    episodes = []\n",
    "\n",
    "    for ep_idx in range(num_episodes):\n",
    '        env = suite.load(domain, task, task_kwargs={"random": seed + ep_idx})\n',
    "        time_step = env.reset()\n",
    "\n",
    "        obs_list, act_list, rew_list, done_list = [], [], [], []\n",
    "        step_count = 0\n",
    "        max_steps = 1000\n",
    "\n",
    "        while not time_step.last() and step_count < max_steps:\n",
    "            obs = flatten_obs(time_step.observation)\n",
    "            obs_list.append(obs)\n",
    "\n",
    "            action = rng.uniform(-1, 1, size=env.action_spec().shape).astype(np.float32)\n",
    "            act_list.append(action)\n",
    "\n",
    "            time_step = env.step(action)\n",
    "            reward = time_step.reward if time_step.reward is not None else 0.0\n",
    "            rew_list.append(float(reward))\n",
    "            done_list.append(float(time_step.last()))\n",
    "            step_count += 1\n",
    "\n",
    "        if len(obs_list) > SEQ_LEN:\n",
    "            episodes.append(Episode(\n",
    "                observations=np.array(obs_list, dtype=np.float32),\n",
    "                actions=np.array(act_list, dtype=np.float32),\n",
    "                rewards=np.array(rew_list, dtype=np.float32),\n",
    "                dones=np.array(done_list, dtype=np.float32),\n",
    "            ))\n",
    "    return episodes\n",
    "\n",
    "\n",
    "class ReplayBuffer:\n",
    '    """Simple replay buffer for sequence sampling."""\n',
    "\n",
    "    def __init__(self):\n",
    "        self.episodes = []\n",
    "\n",
    "    def add(self, episode):\n",
    "        self.episodes.append(episode)\n",
    "\n",
    "    def sample(self, batch_size, seq_len):\n",
    '        """Sample random sequences from stored episodes."""\n',
    "        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []\n",
    "\n",
    "        valid_episodes = [ep for ep in self.episodes if len(ep) >= seq_len + 1]\n",
    "        if not valid_episodes:\n",
    '            raise ValueError(f"No episodes long enough for seq_len={seq_len}")\n',
    "\n",
    "        for _ in range(batch_size):\n",
    "            ep = valid_episodes[np.random.randint(len(valid_episodes))]\n",
    "            start = np.random.randint(0, len(ep) - seq_len)\n",
    "            obs_batch.append(ep.observations[start:start + seq_len])\n",
    "            act_batch.append(ep.actions[start:start + seq_len])\n",
    "            rew_batch.append(ep.rewards[start:start + seq_len])\n",
    "            done_batch.append(ep.dones[start:start + seq_len])\n",
    "\n",
    "        return {\n",
    '            "observations": np.array(obs_batch, dtype=np.float32),\n',
    '            "actions": np.array(act_batch, dtype=np.float32),\n',
    '            "rewards": np.array(rew_batch, dtype=np.float32),\n',
    '            "dones": np.array(done_batch, dtype=np.float32),\n',
    "        }\n",
    "\n",
    "\n",
    'print("Data collection utilities defined.")\n',
    'print("Testing DMControl Reacher-hard...")\n',
    'test_eps = collect_episodes("reacher", "hard", 3, seed=42)\n',
    'print(f"  Collected {len(test_eps)} test episodes")\n',
    "if test_eps:\n",
    '    print(f"  Obs shape: {test_eps[0].observations.shape}")\n',
    '    print(f"  Action shape: {test_eps[0].actions.shape}")\n',
    '    print(f"  Mean episode length: {np.mean([len(ep) for ep in test_eps]):.0f}")\n',
]

# ============================================================
# Cell 14: Run Experiments markdown
# ============================================================
rh["cells"][14]["source"] = [
    "---\n",
    "## Run All Experiments\n",
    "\n",
    "Running all 5 approaches x 5 seeds = 25 experiments on Reacher-hard.\n",
    "Each experiment: collect data -> train -> evaluate (train + test + long-horizon)\n",
]

# ============================================================
# Cell 15: Run Experiments code - fix data collection calls
# ============================================================
src15 = "".join(rh["cells"][15]["source"])
# Replace any Pendulum-specific collection calls
src15 = src15.replace(
    "collect_reacher_hard_episodes(NUM_EPISODES, seed)",
    'collect_episodes("reacher", "hard", NUM_EPISODES, seed)',
)
src15 = src15.replace(
    "collect_reacher_hard_episodes(50, seed + 10000)",
    'collect_episodes("reacher", "hard", 50, seed + 10000)',
)
# Also catch the case where simple replacement left pendulum pattern
import re
src15 = re.sub(
    r"collect_episodes\(NUM_EPISODES, seed\)",
    'collect_episodes("reacher", "hard", NUM_EPISODES, seed)',
    src15,
)
src15 = re.sub(
    r"collect_episodes\(50, seed \+ 10000\)",
    'collect_episodes("reacher", "hard", 50, seed + 10000)',
    src15,
)
# Fix print messages
src15 = src15.replace(
    "Collecting {NUM_EPISODES} training episodes",
    "Collecting {NUM_EPISODES} Reacher-hard episodes",
)

rh["cells"][15]["source"] = src15.split("\n")
# Re-add newlines
rh["cells"][15]["source"] = [
    line + "\n" for line in rh["cells"][15]["source"][:-1]
] + [rh["cells"][15]["source"][-1]]

# ============================================================
# Cell 16: Save Results markdown
# ============================================================
rh["cells"][16]["source"] = [
    "---\n",
    "## Save Complete Results\n",
    "\n",
    "Save aggregated metrics as `complete_metrics.json` and print final summary.\n",
]

# ============================================================
# Cell 17: Save Results code - fix environment metadata
# ============================================================
rh["cells"][17]["source"] = [
    '"""\n',
    "Cell: Save Complete Results\n",
    "Purpose: Save aggregated results as complete_metrics.json\n",
    '"""\n',
    "\n",
    "complete_metrics = {\n",
    '    "experiment": "reacher_hard_quantum_comparison",\n',
    '    "environment": {\n',
    '        "name": "Reacher-hard",\n',
    '        "domain": "reacher",\n',
    '        "task": "hard",\n',
    '        "obs_dim": OBS_DIM,\n',
    '        "action_dim": ACTION_DIM,\n',
    '        "type": "continuous",\n',
    "    },\n",
    '    "config": {\n',
    '        "stoch_dim": STOCH_DIM,\n',
    '        "deter_dim": DETER_DIM,\n',
    '        "hidden_dim": HIDDEN_DIM,\n',
    '        "batch_size": BATCH_SIZE,\n',
    '        "seq_len": SEQ_LEN,\n',
    '        "num_steps": NUM_STEPS,\n',
    '        "learning_rate": LEARNING_RATE,\n',
    '        "kl_weight": KL_WEIGHT,\n',
    '        "grad_clip": GRAD_CLIP,\n',
    '        "num_episodes": NUM_EPISODES,\n',
    '        "seeds": EXPERIMENT_SEEDS,\n',
    "    },\n",
    '    "summary": summaries,\n',
    '    "raw_results": all_results,\n',
    "}\n",
    "\n",
    'metrics_file = RESULTS_DIR / "complete_metrics.json"\n',
    'with open(metrics_file, "w") as f:\n',
    "    json.dump(complete_metrics, f, indent=2)\n",
    'print(f"Saved complete metrics: {metrics_file}")\n',
    "\n",
    "# Print final summary table\n",
    "print(f\"\\n{'='*70}\")\n",
    'print("REACHER-HARD RESULTS SUMMARY")\n',
    "print(f\"{'='*70}\")\n",
    "print(f\"{'Approach':<25} {'Test MSE':>15} {'Train MSE':>15} {'Time (s)':>12} {'Params':>10}\")\n",
    'print("-" * 80)\n',
    "for approach in APPROACHES:\n",
    "    if approach in summaries:\n",
    "        s = summaries[approach]\n",
    "        print(f\"{approach:<25} {s['test_obs_mse_mean']:.6f}\\u00b1{s['test_obs_mse_std']:.4f} \"\n",
    "              f\"{s['train_obs_mse_mean']:.6f}\\u00b1{s['train_obs_mse_std']:.4f} \"\n",
    "              f\"{s['time_mean']:>8.1f}\\u00b1{s['time_std']:.1f} {s['num_params']:>10,}\")\n",
    "    else:\n",
    "        print(f\"{approach:<25} {'FAILED':>15}\")\n",
    "\n",
    "# Identify best\n",
    "if summaries:\n",
    '    best = min(summaries.items(), key=lambda x: x[1]["test_obs_mse_mean"])\n',
    '    baseline_mse = summaries.get("baseline", {}).get("test_obs_mse_mean", float("nan"))\n',
    "    if baseline_mse and not np.isnan(baseline_mse):\n",
    '        improvement = (baseline_mse - best[1]["test_obs_mse_mean"]) / baseline_mse * 100\n',
    '        print(f"\\nBest approach: {best[0]} ({improvement:+.1f}% vs baseline)")\n',
    "\n",
    'print(f"\\nResults saved to: {RESULTS_DIR}")\n',
    'print("Next: Update 04_reacher_experiments.ipynb to load and analyze these results")\n',
]

# ============================================================
# Save
# ============================================================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(rh, f, indent=1)

print(f"Created: {output_path}")
print(f"Cells: {len(rh['cells'])}")
