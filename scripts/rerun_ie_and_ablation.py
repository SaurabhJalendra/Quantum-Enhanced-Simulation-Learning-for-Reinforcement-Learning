"""
Re-run IE experiments (with fixed reward loss) + Uniform Ensemble Ablation
=========================================================================

This script:
1. Re-runs interference_ensemble on all 8 environments with 5 seeds each
   (now with reward prediction loss properly trained)
2. Runs a new "uniform_ensemble" ablation (5 models, equal weights, no phase)
   to isolate ensemble effect vs interference effect
3. Merges results back into each environment's complete_metrics.json

Usage:
    python scripts/rerun_ie_and_ablation.py                    # Run all
    python scripts/rerun_ie_and_ablation.py --env cartpole     # Run one env
    python scripts/rerun_ie_and_ablation.py --only-ablation    # Skip IE re-run

Author: Saurabh Jalendra (2023AC05912)
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from quantum_inspired import (
    QuantumTunnelingOptimizer,
    SuperpositionReplayBuffer,
    EntanglementLayer,
    InterferenceEnsemble,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Standard Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]
STOCH_DIM = 64
DETER_DIM = 512
HIDDEN_DIM = 512
STATE_DIM = STOCH_DIM + DETER_DIM
NUM_STEPS = 10000
BATCH_SIZE = 32
SEQ_LEN = 20
LEARNING_RATE = 3e-4
KL_WEIGHT = 1.0
GRAD_CLIP = 100.0
NUM_ENSEMBLE_MODELS = 5
INTERFERENCE_STRENGTH = 0.7


# ---------------------------------------------------------------------------
# Replay Buffer (standard, from experiment scripts)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Simple replay buffer for sequences."""

    def __init__(self, capacity: int = 10000):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.capacity = capacity

    def add(self, episode: Dict):
        self.obs.append(np.array(episode["obs"], dtype=np.float32))
        self.actions.append(np.array(episode["actions"], dtype=np.float32))
        self.rewards.append(np.array(episode["rewards"], dtype=np.float32))
        self.dones.append(np.array(episode["dones"], dtype=np.float32))

    def __len__(self):
        return len(self.obs)

    def sample(self, batch_size: int, seq_len: int):
        indices = np.random.randint(0, len(self.obs), size=batch_size)
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        for idx in indices:
            ep_len = len(self.obs[idx])
            start = np.random.randint(0, max(1, ep_len - seq_len))
            end = start + seq_len
            obs_batch.append(self.obs[idx][start:end])
            act_batch.append(self.actions[idx][start:end])
            rew_batch.append(self.rewards[idx][start:end])
            done_batch.append(self.dones[idx][start:end])
        return (
            np.array(obs_batch),
            np.array(act_batch),
            np.array(rew_batch),
            np.array(done_batch),
        )


# ---------------------------------------------------------------------------
# Seed Utility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Base World Model (MLP-based, for state-based environments)
# ---------------------------------------------------------------------------
class BaseWorldModel(nn.Module):
    """RSSM world model with MLP encoder/decoder.

    Architecture matches the original experiment scripts exactly:
    - input_proj(stoch + action) -> GRU (autoregressive on stochastic state)
    - Encoder output used only for posterior computation
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 stoch_dim: int = STOCH_DIM, deter_dim: int = DETER_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = stoch_dim + deter_dim

        # Encoder: obs -> embed
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: stoch + action -> hidden (autoregressive)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ELU()
        )

        # GRU dynamics: input is projected(stoch+action), hidden is deter
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

        # Decoder: state(deter+stoch) -> obs
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

    def _get_dist(self, stats):
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return torch.distributions.Normal(mean, std)

    def forward(self, obs_seq, action_seq):
        B, T, _ = obs_seq.shape
        device = obs_seq.device
        # Initial state: zeros for both deter and stoch
        h = torch.zeros(B, self.deter_dim, device=device)
        z = torch.zeros(B, self.stoch_dim, device=device)
        predictions, priors, posteriors = [], [], []
        deter_states, stoch_states = [], []

        for t in range(T):
            embed = self.encoder(obs_seq[:, t])
            act = action_seq[:, t] if action_seq.dim() == 3 else action_seq[:, t:t+1]

            # Autoregressive: previous stoch + current action -> projection -> GRU
            x = self.input_proj(torch.cat([z, act], dim=-1))
            h = self.gru(x, h)

            # Prior from deterministic state
            prior_dist = self._get_dist(self.prior_net(h))
            # Posterior from deterministic state + encoder embedding
            posterior_dist = self._get_dist(self.posterior_net(torch.cat([h, embed], dim=-1)))
            z = posterior_dist.rsample()

            state = torch.cat([h, z], dim=-1)
            pred = self.decoder(state)

            predictions.append(pred)
            priors.append(prior_dist)
            posteriors.append(posterior_dist)
            deter_states.append(h)
            stoch_states.append(z)

        predictions = torch.stack(predictions, dim=1)
        deter_states = torch.stack(deter_states, dim=1)
        stoch_states = torch.stack(stoch_states, dim=1)

        states = {
            "deter": deter_states,
            "stoch": stoch_states,
            "priors": priors,
            "posteriors": posteriors,
        }
        return predictions, states


# ---------------------------------------------------------------------------
# CNN World Model (for Atari)
# ---------------------------------------------------------------------------
class BaseCNNWorldModel(nn.Module):
    """RSSM world model with CNN encoder/decoder for visual tasks.

    Architecture matches the original experiment scripts exactly:
    - CNN encoder: Conv2d layers (no padding) -> flatten -> Linear
    - input_proj(stoch + action) -> GRU (autoregressive)
    - CNN decoder: Linear -> reshape -> ConvTranspose2d layers
    """

    CNN_FLATTEN_DIM = 256 * 4 * 4  # 4096

    def __init__(self, obs_channels: int = 1, action_dim: int = 6,
                 stoch_dim: int = STOCH_DIM, deter_dim: int = DETER_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = stoch_dim + deter_dim

        # CNN Encoder: (B, 1, 84, 84) -> (B, hidden_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 4, stride=2),  nn.ELU(),   # -> (32, 41, 41)
            nn.Conv2d(32, 64, 4, stride=2),             nn.ELU(),   # -> (64, 19, 19)
            nn.Conv2d(64, 128, 3, stride=2),            nn.ELU(),   # -> (128, 9, 9)
            nn.Conv2d(128, 256, 3, stride=2),           nn.ELU(),   # -> (256, 4, 4)
        )
        self.fc_encoder = nn.Linear(self.CNN_FLATTEN_DIM, hidden_dim)

        # Input projection: stoch + action -> hidden (autoregressive)
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

        # CNN Decoder: (B, state_dim) -> (B, 1, 84, 84)
        self.fc_decoder = nn.Linear(self.state_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2), nn.ELU(),   # -> (128, 9, 9)
            nn.ConvTranspose2d(128, 64, 3, stride=2),  nn.ELU(),   # -> (64, 19, 19)
            nn.ConvTranspose2d(64, 32, 4, stride=2),   nn.ELU(),   # -> (32, 40, 40)
            nn.ConvTranspose2d(32, obs_channels, 6, stride=2),     # -> (1, 84, 84)
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

    def _get_dist(self, stats):
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return torch.distributions.Normal(mean, std)

    def _encode(self, obs):
        """Encode a single frame: (B, C, H, W) -> (B, hidden_dim)"""
        features = self.cnn(obs)
        flat = features.view(features.size(0), -1)
        return self.fc_encoder(flat)

    def _decode(self, state):
        """Decode state to image: (B, state_dim) -> (B, C, H, W)"""
        x = self.fc_decoder(state)
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)

    def forward(self, obs_seq, action_seq):
        B, T = obs_seq.shape[0], obs_seq.shape[1]
        device = obs_seq.device
        # Initial state: zeros
        h = torch.zeros(B, self.deter_dim, device=device)
        z = torch.zeros(B, self.stoch_dim, device=device)

        # Encode all timesteps at once for efficiency
        C, H, W = obs_seq.shape[2], obs_seq.shape[3], obs_seq.shape[4]
        obs_flat = obs_seq.view(B * T, C, H, W)
        embeds = self._encode(obs_flat)     # (B*T, hidden_dim)
        embeds = embeds.view(B, T, -1)      # (B, T, hidden_dim)

        predictions, priors, posteriors = [], [], []
        deter_states, stoch_states = [], []

        for t in range(T):
            act = action_seq[:, t]

            # Autoregressive: previous stoch + current action -> projection -> GRU
            x = self.input_proj(torch.cat([z, act], dim=-1))
            h = self.gru(x, h)

            # Prior from deterministic state
            prior_dist = self._get_dist(self.prior_net(h))
            # Posterior from deterministic state + encoder embedding
            posterior_dist = self._get_dist(self.posterior_net(torch.cat([h, embeds[:, t]], dim=-1)))
            z = posterior_dist.rsample()

            state = torch.cat([h, z], dim=-1)
            pred = self._decode(state)

            predictions.append(pred)
            priors.append(prior_dist)
            posteriors.append(posterior_dist)
            deter_states.append(h)
            stoch_states.append(z)

        predictions = torch.stack(predictions, dim=1)
        deter_states = torch.stack(deter_states, dim=1)
        stoch_states = torch.stack(stoch_states, dim=1)

        states = {
            "deter": deter_states,
            "stoch": stoch_states,
            "priors": priors,
            "posteriors": posteriors,
        }
        return predictions, states


# ---------------------------------------------------------------------------
# Interference Ensemble Wrapper
# ---------------------------------------------------------------------------
class InterferenceWorldModel(nn.Module):
    """IE wrapper for state-based envs."""

    def __init__(self, obs_dim, action_dim, base_seed=None):
        super().__init__()
        self.ensemble = InterferenceEnsemble(
            model_class=BaseWorldModel,
            num_models=NUM_ENSEMBLE_MODELS,
            interference_strength=INTERFERENCE_STRENGTH,
            uncertainty_method="disagreement",
            base_seed=base_seed,
            obs_dim=obs_dim,
            action_dim=action_dim,
            stoch_dim=STOCH_DIM,
            deter_dim=DETER_DIM,
            hidden_dim=HIDDEN_DIM,
        )

    def forward(self, obs_seq, action_seq):
        return self.ensemble(obs_seq, action_seq, return_all=True)


class InterferenceCNNWorldModel(nn.Module):
    """IE wrapper for Atari envs."""

    def __init__(self, obs_channels, action_dim, base_seed=None):
        super().__init__()
        self.ensemble = InterferenceEnsemble(
            model_class=BaseCNNWorldModel,
            num_models=NUM_ENSEMBLE_MODELS,
            interference_strength=INTERFERENCE_STRENGTH,
            uncertainty_method="disagreement",
            base_seed=base_seed,
            obs_channels=obs_channels,
            action_dim=action_dim,
            stoch_dim=STOCH_DIM,
            deter_dim=DETER_DIM,
            hidden_dim=HIDDEN_DIM,
        )

    def forward(self, obs_seq, action_seq):
        return self.ensemble(obs_seq, action_seq, return_all=True)


# ---------------------------------------------------------------------------
# Uniform Ensemble (ABLATION - no phase interference)
# ---------------------------------------------------------------------------
class UniformEnsembleWorldModel(nn.Module):
    """5-model ensemble with uniform (equal) weights.

    No phase interference - pure model averaging.
    Used to isolate whether IE's benefit comes from
    the interference mechanism or simply from ensembling.
    """

    def __init__(self, obs_dim, action_dim, num_models=5, base_seed=None):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList()
        for i in range(num_models):
            seed_i = (base_seed + i * 1000) if base_seed else None
            if seed_i:
                set_seed(seed_i)
            self.models.append(BaseWorldModel(
                obs_dim=obs_dim, action_dim=action_dim,
                stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM,
            ))

    def forward(self, obs_seq, action_seq):
        all_predictions = []
        all_states = []
        for model in self.models:
            pred, states = model(obs_seq, action_seq)
            all_predictions.append(pred)
            all_states.append(states)

        # Uniform average (no phase weighting)
        stacked = torch.stack(all_predictions, dim=0)
        combined_pred = stacked.mean(dim=0)

        combined_states = {
            "deter": all_states[0]["deter"],
            "stoch": all_states[0]["stoch"],
            "priors": all_states[0]["priors"],
            "posteriors": all_states[0]["posteriors"],
            "all_predictions": stacked,
            "all_states": all_states,
        }
        return combined_pred, combined_states


class UniformCNNEnsembleWorldModel(nn.Module):
    """5-model CNN ensemble with uniform weights for Atari."""

    def __init__(self, obs_channels, action_dim, num_models=5, base_seed=None):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList()
        for i in range(num_models):
            seed_i = (base_seed + i * 1000) if base_seed else None
            if seed_i:
                set_seed(seed_i)
            self.models.append(BaseCNNWorldModel(
                obs_channels=obs_channels, action_dim=action_dim,
                stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM,
            ))

    def forward(self, obs_seq, action_seq):
        all_predictions = []
        all_states = []
        for model in self.models:
            pred, states = model(obs_seq, action_seq)
            all_predictions.append(pred)
            all_states.append(states)

        stacked = torch.stack(all_predictions, dim=0)
        combined_pred = stacked.mean(dim=0)

        combined_states = {
            "deter": all_states[0]["deter"],
            "stoch": all_states[0]["stoch"],
            "priors": all_states[0]["priors"],
            "posteriors": all_states[0]["posteriors"],
            "all_predictions": stacked,
            "all_states": all_states,
        }
        return combined_pred, combined_states


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def compute_ensemble_loss(model, obs_seq, action_seq, reward_seq, kl_weight=KL_WEIGHT):
    """Loss for ensemble approaches (IE and Uniform).

    Includes reward prediction loss (fix for IE reward bug).
    """
    combined_pred, states = model(obs_seq, action_seq)
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

    # Reward prediction loss (averaged across ensemble members)
    reward_loss = torch.tensor(0.0, device=obs_seq.device)
    if "all_states" in states:
        reward_losses = []
        models = _get_models(model)
        for i, model_states in enumerate(states["all_states"]):
            combined = torch.cat([model_states["deter"], model_states["stoch"]], dim=-1)
            rpred = models[i].reward_pred(combined).squeeze(-1)
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


def _get_models(model):
    """Extract list of sub-models from any ensemble wrapper."""
    if hasattr(model, "ensemble"):
        return model.ensemble.models
    elif hasattr(model, "models"):
        return model.models
    else:
        raise ValueError(f"Cannot extract models from {type(model)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_ensemble(model, buffer, num_steps=NUM_STEPS, batch_size=BATCH_SIZE,
                   approach_name="ensemble"):
    """Train any ensemble model (IE or Uniform)."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for step in range(num_steps):
        model.train()
        obs_np, act_np, rew_np, _ = buffer.sample(batch_size, SEQ_LEN)
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
            "reward": losses["reward"].item(),
            "diversity": losses["diversity"].item(),
        })

        if step % 1000 == 0:
            print(
                f"    Step {step}/{num_steps}: total={losses['total'].item():.4f} "
                f"combined={losses['combined_recon'].item():.4f} "
                f"reward={losses['reward'].item():.6f} "
                f"kl={losses['kl'].item():.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, buffer, approach, num_eval_batches=10,
                   batch_size=BATCH_SIZE):
    """Evaluate model on held-out data."""
    total_mse = 0.0
    total_reward_mse = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for _ in range(num_eval_batches):
            obs_np, act_np, rew_np, _ = buffer.sample(batch_size, SEQ_LEN)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(rew_np, dtype=torch.float32, device=DEVICE)

            pred, states = model(obs, actions)
            mse = F.mse_loss(pred, obs).item()
            total_mse += mse

            # Reward MSE - use first model's reward predictor
            models = _get_models(model)
            if "all_states" in states:
                first_states = states["all_states"][0]
            else:
                first_states = states
            combined = torch.cat([first_states["deter"], first_states["stoch"]], dim=-1)
            rpred = models[0].reward_pred(combined).squeeze(-1)
            reward_mse = F.mse_loss(rpred, rewards).item()
            total_reward_mse += reward_mse
            count += 1

    return {
        "obs_mse": total_mse / count,
        "reward_mse": total_reward_mse / count,
    }


def evaluate_long_horizon(model, buffer, approach, horizons=None,
                          batch_size=BATCH_SIZE):
    """Evaluate at different prediction horizons."""
    if horizons is None:
        horizons = [5, 10, 15, 20]
    max_horizon = max(horizons)

    results = {}
    model.eval()
    with torch.no_grad():
        obs_np, act_np, _, _ = buffer.sample(batch_size, max_horizon)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
        pred, _ = model(obs, actions)

        for h in horizons:
            mse = F.mse_loss(pred[:, :h], obs[:, :h]).item()
            results[h] = mse

    return results


# ---------------------------------------------------------------------------
# Data Collection Functions
# ---------------------------------------------------------------------------
def collect_cartpole_episodes(num_episodes, seed):
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    episodes = []
    rng = np.random.RandomState(seed)
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        episode = {"obs": [obs], "actions": [], "rewards": [], "dones": []}
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode["actions"].append([float(action)])
            episode["rewards"].append(float(reward))
            episode["dones"].append(float(done))
            episode["obs"].append(next_obs)
            if done:
                break
        episode["obs"] = episode["obs"][:-1]  # Remove last (next_obs without action)
        min_len = min(len(episode["obs"]), len(episode["actions"]))
        episode = {k: v[:min_len] for k, v in episode.items()}
        if min_len >= SEQ_LEN:
            episodes.append(episode)
    env.close()
    return episodes


def collect_pendulum_episodes(num_episodes, seed):
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    episodes = []
    rng = np.random.RandomState(seed)
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        episode = {"obs": [obs], "actions": [], "rewards": [], "dones": []}
        for step in range(200):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode["actions"].append(action.tolist() if hasattr(action, 'tolist') else [float(action)])
            episode["rewards"].append(float(reward))
            episode["dones"].append(float(done))
            episode["obs"].append(next_obs)
            if done:
                break
        episode["obs"] = episode["obs"][:-1]
        min_len = min(len(episode["obs"]), len(episode["actions"]))
        episode = {k: v[:min_len] for k, v in episode.items()}
        if min_len >= SEQ_LEN:
            episodes.append(episode)
    env.close()
    return episodes


def collect_dmcontrol_episodes(domain, task, num_episodes, seed):
    from dm_control import suite
    env = suite.load(domain, task, task_kwargs={"random": seed})
    episodes = []
    rng = np.random.RandomState(seed)
    spec = env.action_spec()
    for _ in range(num_episodes):
        timestep = env.reset()
        obs = np.concatenate([v.flatten() for v in timestep.observation.values()])
        episode = {"obs": [obs], "actions": [], "rewards": [], "dones": []}
        for step in range(1000):
            action = rng.uniform(spec.minimum, spec.maximum, spec.shape)
            timestep = env.step(action)
            next_obs = np.concatenate([v.flatten() for v in timestep.observation.values()])
            episode["actions"].append(action.tolist())
            episode["rewards"].append(float(timestep.reward or 0.0))
            episode["dones"].append(float(timestep.last()))
            episode["obs"].append(next_obs)
            if timestep.last():
                break
        episode["obs"] = episode["obs"][:-1]
        min_len = min(len(episode["obs"]), len(episode["actions"]))
        episode = {k: v[:min_len] for k, v in episode.items()}
        if min_len >= SEQ_LEN:
            episodes.append(episode)
    return episodes


def collect_atari_episodes(game, num_episodes, seed):
    import gymnasium as gym
    env = gym.make(f"ALE/{game}-v5", render_mode=None, frameskip=4)
    episodes = []
    rng = np.random.RandomState(seed)
    action_dim = env.action_space.n

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        # Preprocess: grayscale, resize to 84x84, normalize
        obs = _preprocess_atari(obs)
        episode = {"obs": [obs], "actions": [], "rewards": [], "dones": []}
        for step in range(2000):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = _preprocess_atari(next_obs)
            # One-hot action
            action_onehot = np.zeros(action_dim, dtype=np.float32)
            action_onehot[action] = 1.0
            episode["actions"].append(action_onehot.tolist())
            episode["rewards"].append(float(reward))
            episode["dones"].append(float(done))
            episode["obs"].append(next_obs)
            if done:
                break
        episode["obs"] = episode["obs"][:-1]
        min_len = min(len(episode["obs"]), len(episode["actions"]))
        episode = {k: v[:min_len] for k, v in episode.items()}
        if min_len >= SEQ_LEN:
            episodes.append(episode)
    env.close()
    return episodes


def _preprocess_atari(obs):
    """Convert Atari frame to (1, 84, 84) grayscale float."""
    import cv2
    if obs.ndim == 3 and obs.shape[2] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        gray = obs
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[np.newaxis]  # (1, 84, 84)


# ---------------------------------------------------------------------------
# Environment Configurations
# ---------------------------------------------------------------------------
ENV_CONFIGS = {
    "cartpole": {
        "phase": "phase1", "name": "CartPole-v1",
        "obs_dim": 4, "action_dim": 1, "type": "state",
        "collect_fn": lambda n, s: collect_cartpole_episodes(n, s),
        "num_episodes": 100, "batch_size": 32,
    },
    "pendulum": {
        "phase": "phase1", "name": "Pendulum-v1",
        "obs_dim": 3, "action_dim": 1, "type": "state",
        "collect_fn": lambda n, s: collect_pendulum_episodes(n, s),
        "num_episodes": 100, "batch_size": 32,
    },
    "walker": {
        "phase": "phase2", "name": "Walker-walk",
        "obs_dim": 24, "action_dim": 6, "type": "dmcontrol",
        "domain": "walker", "task": "walk",
        "collect_fn": lambda n, s, d="walker", t="walk": collect_dmcontrol_episodes(d, t, n, s),
        "num_episodes": 100, "batch_size": 32,
    },
    "cheetah": {
        "phase": "phase2", "name": "Cheetah-run",
        "obs_dim": 17, "action_dim": 6, "type": "dmcontrol",
        "domain": "cheetah", "task": "run",
        "collect_fn": lambda n, s, d="cheetah", t="run": collect_dmcontrol_episodes(d, t, n, s),
        "num_episodes": 100, "batch_size": 32,
    },
    "reacher": {
        "phase": "phase2", "name": "Reacher-easy",
        "obs_dim": 6, "action_dim": 2, "type": "dmcontrol",
        "domain": "reacher", "task": "easy",
        "collect_fn": lambda n, s, d="reacher", t="easy": collect_dmcontrol_episodes(d, t, n, s),
        "num_episodes": 200, "batch_size": 32,
    },
    "reacher_hard": {
        "phase": "phase2", "name": "Reacher-hard",
        "obs_dim": 6, "action_dim": 2, "type": "dmcontrol",
        "domain": "reacher", "task": "hard",
        "collect_fn": lambda n, s, d="reacher", t="hard": collect_dmcontrol_episodes(d, t, n, s),
        "num_episodes": 200, "batch_size": 32,
    },
    "pong": {
        "phase": "phase3", "name": "Pong",
        "obs_channels": 1, "action_dim": 6, "type": "atari",
        "collect_fn": lambda n, s: collect_atari_episodes("Pong", n, s),
        "num_episodes": 50, "batch_size": 16,
    },
    "breakout": {
        "phase": "phase3", "name": "Breakout",
        "obs_channels": 1, "action_dim": 4, "type": "atari",
        "collect_fn": lambda n, s: collect_atari_episodes("Breakout", n, s),
        "num_episodes": 50, "batch_size": 16,
    },
}


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def create_model(approach, env_config, seed):
    """Create the right model for the approach and environment."""
    if env_config["type"] == "atari":
        obs_ch = env_config["obs_channels"]
        act_dim = env_config["action_dim"]
        if approach == "interference_ensemble":
            return InterferenceCNNWorldModel(obs_ch, act_dim, base_seed=seed).to(DEVICE)
        elif approach == "uniform_ensemble":
            return UniformCNNEnsembleWorldModel(obs_ch, act_dim, base_seed=seed).to(DEVICE)
    else:
        obs_dim = env_config["obs_dim"]
        act_dim = env_config["action_dim"]
        if approach == "interference_ensemble":
            return InterferenceWorldModel(obs_dim, act_dim, base_seed=seed).to(DEVICE)
        elif approach == "uniform_ensemble":
            return UniformEnsembleWorldModel(obs_dim, act_dim, base_seed=seed).to(DEVICE)

    raise ValueError(f"Unknown approach: {approach}")


# ---------------------------------------------------------------------------
# Single Run
# ---------------------------------------------------------------------------
def run_single(approach, seed, train_episodes, env_config):
    """Run a single approach+seed combination."""
    print(f"\n  [{approach}] seed={seed}")
    set_seed(seed)
    start = time.time()

    # Build buffer
    buffer = ReplayBuffer(capacity=10000)
    for ep in train_episodes:
        buffer.add(ep)

    # Create model
    model = create_model(approach, env_config, seed)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {num_params:,}")

    # Train
    batch_size = env_config["batch_size"]
    history = train_ensemble(model, buffer, NUM_STEPS, batch_size, approach)
    elapsed = time.time() - start

    # Evaluate on training data
    eval_buffer = ReplayBuffer()
    for ep in train_episodes[:50]:
        eval_buffer.add(ep)
    train_metrics = evaluate_model(model, eval_buffer, approach, batch_size=batch_size)

    # Evaluate on test data
    print(f"    Collecting test episodes...")
    test_episodes = env_config["collect_fn"](50, seed + 10000)
    test_buffer = ReplayBuffer()
    for ep in test_episodes:
        test_buffer.add(ep)
    test_metrics = evaluate_model(model, test_buffer, approach, batch_size=batch_size)

    # Long horizon
    horizon_results = evaluate_long_horizon(model, test_buffer, approach,
                                            batch_size=batch_size)

    print(
        f"    Train MSE: {train_metrics['obs_mse']:.6f} | "
        f"Test MSE: {test_metrics['obs_mse']:.6f} | "
        f"Reward MSE: {test_metrics['reward_mse']:.6f} | "
        f"Time: {elapsed:.1f}s"
    )

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        "approach": approach,
        "seed": seed,
        "train_obs_mse": train_metrics["obs_mse"],
        "train_reward_mse": train_metrics["reward_mse"],
        "test_obs_mse": test_metrics["obs_mse"],
        "test_reward_mse": test_metrics["reward_mse"],
        "long_horizon": {str(k): v for k, v in horizon_results.items()},
        "final_train_loss": float(history[-1]["total"]),
        "time_seconds": elapsed,
        "num_params": num_params,
    }


# ---------------------------------------------------------------------------
# Environment Runner
# ---------------------------------------------------------------------------
def run_environment(env_key, approaches, skip_ie=False):
    """Run specified approaches on one environment."""
    env_config = ENV_CONFIGS[env_key]
    phase = env_config["phase"]
    env_name = env_key if env_key != "reacher" else "reacher"

    # Results directory
    results_dir = PROJECT_ROOT / "experiments" / "results" / phase / env_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"ENVIRONMENT: {env_config['name']} ({env_key})")
    print(f"{'=' * 70}")

    all_results = []

    for approach in approaches:
        if skip_ie and approach == "interference_ensemble":
            print(f"\n  Skipping {approach} (--only-ablation)")
            continue

        print(f"\n  --- {approach} ---")

        for seed in EXPERIMENT_SEEDS:
            # Collect training data
            print(f"\n  Collecting training data (seed={seed})...")
            train_episodes = env_config["collect_fn"](env_config["num_episodes"], seed)
            print(f"  Collected {len(train_episodes)} episodes")

            try:
                result = run_single(approach, seed, train_episodes, env_config)
                all_results.append(result)

                # Save per-seed result
                seed_file = results_dir / f"{approach}_seed_{seed}.json"
                with open(seed_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    Saved: {seed_file.name}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "approach": approach,
                    "seed": seed,
                    "error": str(e),
                })

    # Merge into complete_metrics.json
    _merge_results(results_dir, env_key, all_results)

    return all_results


def _merge_results(results_dir, env_key, new_results):
    """Merge new results into existing complete_metrics.json."""
    metrics_path = results_dir / "complete_metrics.json"

    # Load existing
    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    # Get existing raw results (if any)
    existing_raw = existing.get("raw_results", [])

    # Remove old results for approaches we're updating
    new_approaches = {r["approach"] for r in new_results if "error" not in r}
    filtered_raw = [r for r in existing_raw if r.get("approach") not in new_approaches]
    filtered_raw.extend([r for r in new_results if "error" not in r])

    # Recompute summary
    all_approaches = set(r["approach"] for r in filtered_raw if "error" not in r)
    summary = {}
    for approach in sorted(all_approaches):
        approach_results = [r for r in filtered_raw if r.get("approach") == approach and "error" not in r]
        if not approach_results:
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

    env_config = ENV_CONFIGS[env_key]
    complete_metrics = {
        "experiment": f"{env_config['phase']}_{env_key}",
        "environment": {"name": env_config["name"]},
        "config": {
            "stoch_dim": STOCH_DIM, "deter_dim": DETER_DIM, "hidden_dim": HIDDEN_DIM,
            "batch_size": env_config["batch_size"], "seq_len": SEQ_LEN,
            "num_steps": NUM_STEPS, "learning_rate": LEARNING_RATE,
            "kl_weight": KL_WEIGHT, "grad_clip": GRAD_CLIP,
            "num_episodes": env_config["num_episodes"],
            "seeds": EXPERIMENT_SEEDS,
        },
        "summary": summary,
        "raw_results": filtered_raw,
    }

    with open(metrics_path, "w") as f:
        json.dump(complete_metrics, f, indent=2, default=str)

    print(f"\n  Updated: {metrics_path}")
    print(f"  Approaches in file: {list(summary.keys())}")

    # Print summary
    for approach, stats in summary.items():
        if approach in new_approaches:
            print(f"  {approach}: test_obs={stats['test_obs_mse_mean']:.6f}±{stats['test_obs_mse_std']:.6f} "
                  f"reward={stats['test_reward_mse_mean']:.6f}±{stats['test_reward_mse_std']:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Re-run IE + Uniform Ensemble Ablation")
    parser.add_argument("--env", type=str, default=None,
                        help="Single environment to run (e.g., cartpole, walker)")
    parser.add_argument("--only-ablation", action="store_true",
                        help="Only run uniform_ensemble, skip IE re-run")
    parser.add_argument("--only-ie", action="store_true",
                        help="Only re-run IE, skip ablation")
    args = parser.parse_args()

    # Determine approaches
    approaches = []
    if not args.only_ablation:
        approaches.append("interference_ensemble")
    if not args.only_ie:
        approaches.append("uniform_ensemble")

    # Determine environments
    if args.env:
        env_keys = [args.env]
    else:
        env_keys = list(ENV_CONFIGS.keys())

    print("=" * 70)
    print("IE RE-RUN + UNIFORM ENSEMBLE ABLATION")
    print("=" * 70)
    print(f"Approaches: {approaches}")
    print(f"Environments: {env_keys}")
    print(f"Seeds: {EXPERIMENT_SEEDS}")
    print(f"Total runs: {len(approaches) * len(env_keys) * len(EXPERIMENT_SEEDS)}")
    print()

    total_start = time.time()

    for env_key in env_keys:
        run_environment(env_key, approaches, skip_ie=args.only_ablation)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {total_elapsed / 3600:.1f} hours")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
