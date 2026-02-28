"""
Phase 1: CartPole-v1 Experiment
===============================

Runs ALL 5 approaches on CartPole-v1 with 5 seeds each.

Approaches:
    1. Baseline (standard RSSM + AdamW)
    2. Quantum Tunneling (RSSM + QuantumTunnelingOptimizer)
    3. Superposition (RSSM + SuperpositionReplayBuffer)
    4. Entanglement (RSSM with EntanglementLayer in encoder)
    5. Interference Ensemble (5 BaseWorldModels with InterferenceEnsemble)

All approaches use the same RSSM architecture (stoch=64, deter=512, hidden=512)
to ensure fair comparison. Only the training/sampling/architecture enhancement differs.

Author: Saurabh Jalendra (BITS ID: 2023AC05912)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from quantum_inspired import (
    QuantumTunnelingOptimizer,
    SuperpositionReplayBuffer,
    EntanglementLayer,
    InterferenceEnsemble,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]
OBS_DIM = 4       # CartPole: cart_pos, cart_vel, pole_angle, pole_vel
ACTION_DIM = 1    # Discrete action represented as single float (0.0 or 1.0)
STOCH_DIM = 64
DETER_DIM = 512
HIDDEN_DIM = 512
STATE_DIM = STOCH_DIM + DETER_DIM  # 576

NUM_EPISODES = 100
NUM_STEPS = 10000
BATCH_SIZE = 32
SEQ_LEN = 20
LEARNING_RATE = 3e-4
KL_WEIGHT = 1.0
GRAD_CLIP = 100.0

NUM_ENSEMBLE_MODELS = 5
INTERFERENCE_STRENGTH = 0.7

APPROACHES = [
    "baseline",
    "quantum_tunneling",
    "superposition",
    "entanglement",
    "interference_ensemble",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------
def collect_cartpole_episodes(num_episodes: int, seed: int) -> List[Dict]:
    """Collect episodes from CartPole-v1 using random policy.

    Actions are discrete (0 or 1) but stored as float for the world model.
    """
    env = gym.make("CartPole-v1")
    episodes = []
    rng = np.random.RandomState(seed)

    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        obs_list, act_list, rew_list, done_list = [], [], [], []

        terminated = False
        truncated = False
        while not terminated and not truncated:
            obs_list.append(np.array(obs, dtype=np.float32))
            action = rng.randint(0, 2)
            obs, reward, terminated, truncated, info = env.step(action)
            # Store discrete action as float with shape (1,)
            act_list.append(np.array([float(action)], dtype=np.float32))
            rew_list.append(float(reward))
            done_list.append(float(terminated or truncated))

        if len(obs_list) > 0:
            episodes.append({
                "obs": np.array(obs_list, dtype=np.float32),
                "actions": np.array(act_list, dtype=np.float32),
                "rewards": np.array(rew_list, dtype=np.float32),
                "dones": np.array(done_list, dtype=np.float32),
            })

    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Simple episode replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.episodes: List[Dict] = []
        self.capacity = capacity

    def add(self, episode: Dict) -> None:
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample(self, batch_size: int, seq_len: int):
        obs_b, act_b, rew_b, done_b = [], [], [], []
        for _ in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            L = len(ep["obs"])
            if L <= seq_len:
                pad = seq_len - L
                obs = np.pad(ep["obs"], ((0, pad), (0, 0)), mode="edge")
                act = np.pad(ep["actions"], ((0, pad), (0, 0)), mode="edge")
                rew = np.pad(ep["rewards"], (0, pad), mode="edge")
                done = np.pad(ep["dones"], (0, pad), mode="edge")
            else:
                s = np.random.randint(0, L - seq_len)
                obs = ep["obs"][s : s + seq_len]
                act = ep["actions"][s : s + seq_len]
                rew = ep["rewards"][s : s + seq_len]
                done = ep["dones"][s : s + seq_len]
            obs_b.append(obs)
            act_b.append(act)
            rew_b.append(rew)
            done_b.append(done)
        return np.array(obs_b), np.array(act_b), np.array(rew_b), np.array(done_b)

    def __len__(self) -> int:
        return len(self.episodes)


# ---------------------------------------------------------------------------
# RSSM State
# ---------------------------------------------------------------------------
class RSSMState(NamedTuple):
    deter: torch.Tensor
    stoch: torch.Tensor

    @property
    def combined(self):
        return torch.cat([self.deter, self.stoch], dim=-1)


# ---------------------------------------------------------------------------
# Base World Model (used by all approaches)
# ---------------------------------------------------------------------------
class BaseWorldModel(nn.Module):
    """Standard RSSM world model matching the project architecture spec."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        stoch_dim: int = STOCH_DIM,
        deter_dim: int = DETER_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = stoch_dim + deter_dim

        # Encoder: obs_dim -> 512 -> 512 -> 512
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: stoch + action -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ELU()
        )

        # GRU dynamics
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior: deter -> stoch*2
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

        # Posterior: deter + embed -> stoch*2
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

        # Decoder: state -> obs
        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor
        self.reward_pred = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue predictor
        self.continue_pred = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def _get_dist(self, stats: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return torch.distributions.Normal(mean, std)

    def observe(self, obs: torch.Tensor, action: torch.Tensor, state: RSSMState):
        embed = self.encoder(obs)
        x = self.input_proj(torch.cat([state.stoch, action], dim=-1))
        deter = self.gru(x, state.deter)
        posterior_stats = self.posterior_net(torch.cat([deter, embed], dim=-1))
        posterior = self._get_dist(posterior_stats)
        stoch = posterior.rsample()
        prior_stats = self.prior_net(deter)
        prior = self._get_dist(prior_stats)
        return RSSMState(deter, stoch), prior, posterior

    def decode(self, state: RSSMState) -> torch.Tensor:
        return self.decoder(state.combined)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device
        state = self.initial_state(batch_size, device)

        recon_obs, all_deter, all_stoch = [], [], []
        priors, posteriors = [], []

        for t in range(seq_len):
            state, prior, posterior = self.observe(
                obs_seq[:, t], action_seq[:, t], state
            )
            recon_obs.append(self.decode(state))
            all_deter.append(state.deter)
            all_stoch.append(state.stoch)
            priors.append(prior)
            posteriors.append(posterior)

        predictions = torch.stack(recon_obs, dim=1)
        states_dict = {
            "deter": torch.stack(all_deter, dim=1),
            "stoch": torch.stack(all_stoch, dim=1),
            "priors": priors,
            "posteriors": posteriors,
        }
        return predictions, states_dict


# ---------------------------------------------------------------------------
# Entanglement World Model (approach 4)
# ---------------------------------------------------------------------------
class EntanglementWorldModel(BaseWorldModel):
    """RSSM with EntanglementLayer inserted into the encoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace encoder with entanglement-enhanced version
        hidden_dim = kwargs.get("hidden_dim", HIDDEN_DIM)
        obs_dim = kwargs.get("obs_dim", OBS_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            EntanglementLayer(dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def compute_loss(
    model: BaseWorldModel,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    kl_weight: float = KL_WEIGHT,
) -> Dict[str, torch.Tensor]:
    """Standard RSSM loss for single-model approaches."""
    predictions, states = model(obs_seq, action_seq)
    recon_loss = F.mse_loss(predictions, obs_seq)

    kl_losses = []
    for prior, posterior in zip(states["priors"], states["posteriors"]):
        kl = torch.distributions.kl_divergence(posterior, prior).sum(-1).mean()
        kl_losses.append(kl)
    kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=obs_seq.device)

    combined_states = torch.cat([states["deter"], states["stoch"]], dim=-1)
    reward_pred = model.reward_pred(combined_states)
    reward_loss = F.mse_loss(reward_pred.squeeze(-1), reward_seq)

    total = recon_loss + kl_weight * kl_loss + reward_loss
    return {"total": total, "recon": recon_loss, "kl": kl_loss, "reward": reward_loss}


def compute_ensemble_loss(
    ensemble_model,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    kl_weight: float = KL_WEIGHT,
) -> Dict[str, torch.Tensor]:
    """Loss for InterferenceEnsemble approach."""
    combined_pred, states = ensemble_model(obs_seq, action_seq)
    all_predictions = states.get("all_predictions", None)

    combined_recon_loss = F.mse_loss(combined_pred, obs_seq)

    individual_loss = torch.tensor(0.0, device=obs_seq.device)
    if all_predictions is not None:
        individual_losses = [F.mse_loss(pred, obs_seq) for pred in all_predictions]
        individual_loss = torch.stack(individual_losses).mean()

    kl_losses = []
    if "all_states" in states:
        for model_states in states["all_states"]:
            if "priors" in model_states and "posteriors" in model_states:
                for prior, posterior in zip(model_states["priors"], model_states["posteriors"]):
                    kl = torch.distributions.kl_divergence(posterior, prior).sum(-1).mean()
                    kl_losses.append(kl)
    kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=obs_seq.device)

    diversity = torch.tensor(0.0, device=obs_seq.device)
    if all_predictions is not None:
        mean_pred = all_predictions.mean(dim=0)
        diversity = ((all_predictions - mean_pred) ** 2).mean()

    # Reward prediction loss (average across ensemble members)
    reward_loss = torch.tensor(0.0, device=obs_seq.device)
    if "all_states" in states:
        reward_losses = []
        for i, model_states in enumerate(states["all_states"]):
            combined = torch.cat([model_states["deter"], model_states["stoch"]], dim=-1)
            rpred = ensemble_model.ensemble.models[i].reward_pred(combined).squeeze(-1)
            reward_losses.append(F.mse_loss(rpred, reward_seq))
        reward_loss = torch.stack(reward_losses).mean()

    total = 0.5 * combined_recon_loss + 0.5 * individual_loss + kl_weight * kl_loss + reward_loss - 0.01 * diversity
    return {
        "total": total,
        "combined_recon": combined_recon_loss,
        "individual_recon": individual_loss,
        "kl": kl_loss,
        "reward": reward_loss,
        "diversity": diversity,
    }


# ---------------------------------------------------------------------------
# Interference Ensemble Wrapper
# ---------------------------------------------------------------------------
class InterferenceWorldModel(nn.Module):
    """Wrapper around InterferenceEnsemble for approach 5."""

    def __init__(self, base_seed: Optional[int] = None):
        super().__init__()
        self.ensemble = InterferenceEnsemble(
            model_class=BaseWorldModel,
            num_models=NUM_ENSEMBLE_MODELS,
            interference_strength=INTERFERENCE_STRENGTH,
            uncertainty_method="disagreement",
            base_seed=base_seed,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            stoch_dim=STOCH_DIM,
            deter_dim=DETER_DIM,
            hidden_dim=HIDDEN_DIM,
        )

    def forward(self, obs_seq, action_seq):
        return self.ensemble(obs_seq, action_seq, return_all=True)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_single_model(
    model: nn.Module,
    buffer,
    approach: str,
    seed: int,
    num_steps: int = NUM_STEPS,
) -> pd.DataFrame:
    """Train a single-model approach (baseline, tunneling, superposition, entanglement)."""
    # Select optimizer
    if approach == "quantum_tunneling":
        optimizer = QuantumTunnelingOptimizer(
            model.parameters(),
            lr=LEARNING_RATE,
            tunneling_strength=0.001,
            annealing_rate=0.9999,
            tunneling_frequency=100,
            min_tunneling=1e-8,
            stuck_threshold=500,
            base_optimizer="adamw",
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    use_superposition_buffer = approach == "superposition"
    history = []

    for step in range(num_steps):
        model.train()

        if use_superposition_buffer:
            batch = buffer.sample(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
            obs = batch["obs"].to(DEVICE)
            actions = batch["actions"].to(DEVICE)
            rewards = batch["rewards"].to(DEVICE)
        else:
            obs_np, act_np, rew_np, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(rew_np, dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        losses = compute_loss(model, obs, actions, rewards)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        if approach == "quantum_tunneling":
            optimizer.step(losses["total"].item())
        else:
            optimizer.step()

        history.append({
            "step": step,
            "total": losses["total"].item(),
            "recon": losses["recon"].item(),
            "kl": losses["kl"].item(),
            "reward": losses["reward"].item(),
        })

        if step % 1000 == 0:
            print(
                f"    Step {step}/{num_steps}: total={losses['total'].item():.4f} "
                f"recon={losses['recon'].item():.4f} kl={losses['kl'].item():.4f} "
                f"reward={losses['reward'].item():.4f}"
            )

    return pd.DataFrame(history)


def train_interference_ensemble(
    model: InterferenceWorldModel,
    buffer: ReplayBuffer,
    num_steps: int = NUM_STEPS,
) -> pd.DataFrame:
    """Train interference ensemble approach."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for step in range(num_steps):
        model.train()
        obs_np, act_np, rew_np, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rew_np, dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        losses = compute_ensemble_loss(model, obs, actions, rewards)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        history.append({
            "step": step,
            "total": losses["total"].item(),
            "combined_recon": losses["combined_recon"].item(),
            "individual_recon": losses["individual_recon"].item(),
            "kl": losses["kl"].item(),
            "diversity": losses["diversity"].item(),
        })

        if step % 1000 == 0:
            print(
                f"    Step {step}/{num_steps}: total={losses['total'].item():.4f} "
                f"combined={losses['combined_recon'].item():.4f} "
                f"kl={losses['kl'].item():.4f}"
            )

    return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    buffer: ReplayBuffer,
    approach: str,
    num_eval_batches: int = 10,
) -> Dict[str, float]:
    """Evaluate a model on data from the buffer."""
    total_mse = 0.0
    total_reward_mse = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(num_eval_batches):
            obs_np, act_np, rew_np, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(rew_np, dtype=torch.float32, device=DEVICE)

            if approach == "interference_ensemble":
                pred, states = model(obs, actions)
            else:
                model.eval()
                pred, states = model(obs, actions)

            mse = F.mse_loss(pred, obs).item()
            total_mse += mse

            # Reward MSE
            if approach == "interference_ensemble":
                first_states = states["all_states"][0] if "all_states" in states else states
                combined = torch.cat([first_states["deter"], first_states["stoch"]], dim=-1)
                rpred = model.ensemble.models[0].reward_pred(combined).squeeze(-1)
            else:
                combined = torch.cat([states["deter"], states["stoch"]], dim=-1)
                rpred = model.reward_pred(combined).squeeze(-1)

            reward_mse = F.mse_loss(rpred, rewards).item()
            total_reward_mse += reward_mse
            count += 1

    return {
        "obs_mse": total_mse / count,
        "reward_mse": total_reward_mse / count,
    }


def evaluate_long_horizon(
    model,
    buffer: ReplayBuffer,
    approach: str,
    horizons: List[int] = None,
) -> Dict[int, float]:
    """Evaluate prediction quality at different horizons."""
    if horizons is None:
        horizons = [5, 10, 15, 20]
    max_horizon = max(horizons)

    results = {}
    with torch.no_grad():
        obs_np, act_np, _, _ = buffer.sample(BATCH_SIZE, max_horizon)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)

        if approach == "interference_ensemble":
            pred, _ = model(obs, actions)
        else:
            model.eval()
            pred, _ = model(obs, actions)

        for h in horizons:
            mse = F.mse_loss(pred[:, :h], obs[:, :h]).item()
            results[h] = mse

    return results


# ---------------------------------------------------------------------------
# Per-Approach Runner
# ---------------------------------------------------------------------------
def create_model_for_approach(approach: str, seed: int):
    """Create the appropriate model for the given approach."""
    if approach == "entanglement":
        return EntanglementWorldModel(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM,
        ).to(DEVICE)

    if approach == "interference_ensemble":
        return InterferenceWorldModel(base_seed=seed).to(DEVICE)

    # baseline, quantum_tunneling, superposition all use BaseWorldModel
    return BaseWorldModel(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)


def run_single(approach: str, seed: int, train_episodes: List[Dict]) -> Dict:
    """Run a single approach + seed combination."""
    print(f"\n  [{approach}] seed={seed}")
    set_seed(seed)
    start = time.time()

    # Build buffer
    if approach == "superposition":
        buffer = SuperpositionReplayBuffer(capacity=10000)
        for ep in train_episodes:
            buffer.add(ep)
    else:
        buffer = ReplayBuffer(capacity=10000)
        for ep in train_episodes:
            buffer.add(ep)

    # Create model
    model = create_model_for_approach(approach, seed)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {num_params:,}")

    # Train
    if approach == "interference_ensemble":
        history = train_interference_ensemble(model, buffer)
    else:
        history = train_single_model(model, buffer, approach, seed)

    elapsed = time.time() - start

    # Evaluate on training data
    eval_buffer = ReplayBuffer()
    for ep in train_episodes[:50]:
        eval_buffer.add(ep)
    train_metrics = evaluate_model(model, eval_buffer, approach)

    # Evaluate on test data (different seed offset)
    print(f"    Collecting test episodes...")
    test_episodes = collect_cartpole_episodes(50, seed + 10000)
    test_buffer = ReplayBuffer()
    for ep in test_episodes:
        test_buffer.add(ep)
    test_metrics = evaluate_model(model, test_buffer, approach)

    # Long horizon
    horizon_results = evaluate_long_horizon(model, test_buffer, approach)

    print(
        f"    Train MSE: {train_metrics['obs_mse']:.6f} | "
        f"Test MSE: {test_metrics['obs_mse']:.6f} | "
        f"Time: {elapsed:.1f}s"
    )

    return {
        "approach": approach,
        "seed": seed,
        "train_obs_mse": train_metrics["obs_mse"],
        "train_reward_mse": train_metrics["reward_mse"],
        "test_obs_mse": test_metrics["obs_mse"],
        "test_reward_mse": test_metrics["reward_mse"],
        "long_horizon": {str(k): v for k, v in horizon_results.items()},
        "final_train_loss": float(history["total"].iloc[-1]),
        "time_seconds": elapsed,
        "num_params": num_params,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 1: CARTPOLE-v1 - ALL 5 APPROACHES x 5 SEEDS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Approaches: {APPROACHES}")
    print(f"Seeds: {EXPERIMENT_SEEDS}")
    print(f"NUM_EPISODES={NUM_EPISODES}, NUM_STEPS={NUM_STEPS}")
    print(f"Architecture: obs={OBS_DIM}, act={ACTION_DIM}, stoch={STOCH_DIM}, deter={DETER_DIM}")
    print()

    results_dir = PROJECT_ROOT / "experiments" / "results" / "phase1" / "cartpole"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for approach in APPROACHES:
        print(f"\n{'=' * 70}")
        print(f"APPROACH: {approach}")
        print(f"{'=' * 70}")

        for seed in EXPERIMENT_SEEDS:
            # Collect training data (same data per seed, shared across approaches)
            print(f"\n  Collecting training data (seed={seed}, {NUM_EPISODES} episodes)...")
            train_episodes = collect_cartpole_episodes(NUM_EPISODES, seed)
            print(f"  Collected {len(train_episodes)} episodes")

            try:
                result = run_single(approach, seed, train_episodes)
                all_results.append(result)

                # Save per-seed result
                seed_file = results_dir / f"{approach}_seed_{seed}.json"
                with open(seed_file, "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "approach": approach,
                    "seed": seed,
                    "error": str(e),
                })

    # -----------------------------------------------------------------------
    # Aggregate and print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS - CARTPOLE-v1")
    print("=" * 70)

    summary = {}
    for approach in APPROACHES:
        approach_results = [r for r in all_results if r.get("approach") == approach and "error" not in r]
        if not approach_results:
            print(f"\n{approach}: ALL FAILED")
            continue

        test_mses = [r["test_obs_mse"] for r in approach_results]
        train_mses = [r["train_obs_mse"] for r in approach_results]
        reward_mses = [r["test_reward_mse"] for r in approach_results]
        times = [r["time_seconds"] for r in approach_results]

        summary[approach] = {
            "test_obs_mse_mean": float(np.mean(test_mses)),
            "test_obs_mse_std": float(np.std(test_mses)),
            "train_obs_mse_mean": float(np.mean(train_mses)),
            "train_obs_mse_std": float(np.std(train_mses)),
            "test_reward_mse_mean": float(np.mean(reward_mses)),
            "test_reward_mse_std": float(np.std(reward_mses)),
            "time_mean": float(np.mean(times)),
            "num_params": approach_results[0]["num_params"],
            "num_seeds": len(approach_results),
        }

        print(f"\n{approach} ({len(approach_results)}/{len(EXPERIMENT_SEEDS)} seeds):")
        print(f"  Train Obs MSE: {np.mean(train_mses):.6f} +/- {np.std(train_mses):.6f}")
        print(f"  Test  Obs MSE: {np.mean(test_mses):.6f} +/- {np.std(test_mses):.6f}")
        print(f"  Test  Rew MSE: {np.mean(reward_mses):.6f} +/- {np.std(reward_mses):.6f}")
        print(f"  Avg Time:      {np.mean(times):.1f}s")
        print(f"  Parameters:    {approach_results[0]['num_params']:,}")

    # Save complete metrics
    complete_metrics = {
        "experiment": "phase1_cartpole_v1",
        "environment": {"name": "CartPole-v1", "obs_dim": OBS_DIM, "action_dim": ACTION_DIM},
        "config": {
            "stoch_dim": STOCH_DIM,
            "deter_dim": DETER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "num_steps": NUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "kl_weight": KL_WEIGHT,
            "grad_clip": GRAD_CLIP,
            "num_episodes": NUM_EPISODES,
            "seeds": EXPERIMENT_SEEDS,
        },
        "summary": summary,
        "raw_results": all_results,
    }

    metrics_path = results_dir / "complete_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(complete_metrics, f, indent=2, default=str)

    print(f"\nResults saved to {results_dir}")
    print(f"Complete metrics: {metrics_path}")
    print("DONE")


if __name__ == "__main__":
    main()
