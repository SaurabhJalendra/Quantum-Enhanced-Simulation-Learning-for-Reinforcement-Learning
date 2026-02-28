"""
Phase 3: Atari Breakout Experiment
===============================

Runs ALL 5 approaches on Atari Breakout with 5 seeds each.

Key differences from Phase 2 (DMControl):
    - Pixel observations (84x84 grayscale) require CNN encoder/decoder
    - Discrete actions (4) with one-hot encoding
    - Smaller batch size (16) due to image memory overhead
    - Shorter sequences (10) for efficiency

Approaches:
    1. Baseline (CNN-RSSM + AdamW)
    2. Quantum Tunneling (CNN-RSSM + QuantumTunnelingOptimizer)
    3. Superposition (CNN-RSSM + SuperpositionReplayBuffer)
    4. Entanglement (CNN-RSSM with EntanglementLayer after CNN flatten)
    5. Interference Ensemble (5 CNN-RSSMs with InterferenceEnsemble)

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

# Check for Atari dependencies before proceeding
try:
    import gymnasium as gym
except ImportError:
    print("ERROR: gymnasium is not installed. Install with: pip install gymnasium")
    sys.exit(1)

try:
    import ale_py  # noqa: F401
    gym.register_envs(ale_py)
    # Verify Breakout ROM is available
    _test_env = gym.make("ALE/Breakout-v5", render_mode=None)
    _test_env.close()
    del _test_env
except Exception as e:
    print(f"ERROR: Atari/Breakout ROM not available: {e}")
    print("Install with: pip install gymnasium[atari] ale-py")
    print("Then install ROMs: ale-import-roms (or AutoROM)")
    sys.exit(1)

# Preprocessing: try cv2 first, fall back to PIL
try:
    import cv2

    def preprocess_obs(obs: np.ndarray) -> np.ndarray:
        """Convert RGB frame to 84x84 grayscale float in [0,1]."""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

except ImportError:
    from PIL import Image

    def preprocess_obs(obs: np.ndarray) -> np.ndarray:
        """Convert RGB frame to 84x84 grayscale float in [0,1] (PIL fallback)."""
        gray = np.mean(obs, axis=-1).astype(np.uint8)
        img = Image.fromarray(gray)
        img = img.resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

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
OBS_SHAPE = (1, 84, 84)  # (C, H, W) grayscale
ACTION_DIM = 4  # Breakout discrete actions
STOCH_DIM = 64
DETER_DIM = 512
HIDDEN_DIM = 512
STATE_DIM = STOCH_DIM + DETER_DIM  # 576
EMBED_DIM = 512  # CNN encoder output

# Spatial dim after conv layers on 84x84:
#   84 -> 41 (k=4,s=2) -> 19 (k=4,s=2) -> 9 (k=3,s=2) -> 4 (k=3,s=2)
CNN_FLATTEN_DIM = 256 * 4 * 4  # 4096

NUM_EPISODES = 50  # fewer since visual episodes are longer
NUM_STEPS = 10000
BATCH_SIZE = 16  # smaller due to image memory
SEQ_LEN = 20  # standard sequence length
LEARNING_RATE = 3e-4
KL_WEIGHT = 1.0
GRAD_CLIP = 100.0

NUM_ENSEMBLE_MODELS = 5
INTERFERENCE_STRENGTH = 0.7

ENV_NAME = "ALE/Breakout-v5"

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
def collect_episodes(env_name: str, num_episodes: int, seed: int) -> List[Dict]:
    """Collect episodes from Atari Breakout using random policy."""
    env = gym.make(env_name, render_mode=None)
    episodes = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        observations, actions, rewards, dones = [], [], [], []
        done = False
        while not done:
            processed = preprocess_obs(obs)
            observations.append(processed)
            action = env.action_space.sample()
            action_oh = np.zeros(ACTION_DIM, dtype=np.float32)
            action_oh[action] = 1.0
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            actions.append(action_oh)
            rewards.append(float(r))
            dones.append(float(done))
        if len(observations) > 0:
            episodes.append({
                "obs": np.array(observations, dtype=np.float32),  # (T, 84, 84)
                "actions": np.array(actions, dtype=np.float32),   # (T, 4)
                "rewards": np.array(rewards, dtype=np.float32),   # (T,)
                "dones": np.array(dones, dtype=np.float32),       # (T,)
            })
        if (i + 1) % 20 == 0:
            print(f"    Collected {i + 1}/{num_episodes} episodes")
    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Replay Buffer (Image-aware)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Episode replay buffer handling image observations (T, 84, 84)."""

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
                obs = np.pad(ep["obs"], ((0, pad), (0, 0), (0, 0)), mode="edge")
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
# CNN Encoder / Decoder
# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """CNN encoder: (B, 1, 84, 84) -> (B, 512)."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2),    nn.ELU(),  # -> (32, 41, 41)
            nn.Conv2d(32, 64, 4, stride=2),   nn.ELU(),  # -> (64, 19, 19)
            nn.Conv2d(64, 128, 3, stride=2),  nn.ELU(),  # -> (128, 9, 9)
            nn.Conv2d(128, 256, 3, stride=2), nn.ELU(),  # -> (256, 4, 4)
        )
        self.fc = nn.Linear(CNN_FLATTEN_DIM, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class CNNDecoder(nn.Module):
    """CNN decoder: (B, state_dim) -> (B, 1, 84, 84)."""

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.fc = nn.Linear(state_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2), nn.ELU(),  # -> (128, 9, 9)
            nn.ConvTranspose2d(128, 64, 3, stride=2),  nn.ELU(),  # -> (64, 19, 19)
            nn.ConvTranspose2d(64, 32, 4, stride=2),   nn.ELU(),  # -> (32, 40, 40)
            nn.ConvTranspose2d(32, 1, 6, stride=2),               # -> (1, 84, 84)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.fc(state)
        h = h.view(h.size(0), 256, 4, 4)
        return self.deconv(h)


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
# Base CNN World Model
# ---------------------------------------------------------------------------
class BaseCNNWorldModel(nn.Module):
    """RSSM world model with CNN encoder/decoder for pixel observations."""

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        stoch_dim: int = STOCH_DIM,
        deter_dim: int = DETER_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = stoch_dim + deter_dim

        # CNN encoder/decoder
        self.encoder = CNNEncoder(embed_dim=hidden_dim)
        self.decoder = CNNDecoder(state_dim=self.state_dim)

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

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a single timestep of observations. obs: (B, 1, 84, 84)."""
        return self.encoder(obs)

    def observe(self, obs_embed: torch.Tensor, action: torch.Tensor, state: RSSMState):
        """Single-step observe using pre-computed embedding."""
        x = self.input_proj(torch.cat([state.stoch, action], dim=-1))
        deter = self.gru(x, state.deter)
        posterior_stats = self.posterior_net(torch.cat([deter, obs_embed], dim=-1))
        posterior = self._get_dist(posterior_stats)
        stoch = posterior.rsample()
        prior_stats = self.prior_net(deter)
        prior = self._get_dist(prior_stats)
        return RSSMState(deter, stoch), prior, posterior

    def decode(self, state: RSSMState) -> torch.Tensor:
        """Decode state to image: (B, 1, 84, 84)."""
        return self.decoder(state.combined)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        """
        Forward pass through sequence.

        Parameters
        ----------
        obs_seq : torch.Tensor
            Shape (B, T, 1, 84, 84)
        action_seq : torch.Tensor
            Shape (B, T, ACTION_DIM)
        """
        B, T, C, H, W = obs_seq.shape
        device = obs_seq.device
        state = self.initial_state(B, device)

        # Encode all timesteps at once
        obs_flat = obs_seq.view(B * T, C, H, W)
        embeds = self.encoder(obs_flat)  # (B*T, hidden_dim)
        embeds = embeds.view(B, T, -1)   # (B, T, hidden_dim)

        recon_obs, all_deter, all_stoch = [], [], []
        priors, posteriors = [], []

        for t in range(T):
            state, prior, posterior = self.observe(embeds[:, t], action_seq[:, t], state)
            recon_obs.append(self.decode(state))
            all_deter.append(state.deter)
            all_stoch.append(state.stoch)
            priors.append(prior)
            posteriors.append(posterior)

        predictions = torch.stack(recon_obs, dim=1)  # (B, T, 1, 84, 84)
        states_dict = {
            "deter": torch.stack(all_deter, dim=1),
            "stoch": torch.stack(all_stoch, dim=1),
            "priors": priors,
            "posteriors": posteriors,
        }
        return predictions, states_dict


# ---------------------------------------------------------------------------
# Entanglement CNN World Model (approach 4)
# ---------------------------------------------------------------------------
class EntanglementCNNWorldModel(BaseCNNWorldModel):
    """CNN-RSSM with EntanglementLayer after CNN flatten."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", HIDDEN_DIM)
        # Replace encoder with entanglement-enhanced version
        self.encoder = nn.Sequential(
            CNNEncoder(embed_dim=hidden_dim).cnn,
            nn.Flatten(),
            nn.Linear(CNN_FLATTEN_DIM, hidden_dim),
            nn.ELU(),
            EntanglementLayer(dim=hidden_dim),
        )


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def compute_loss(
    model: BaseCNNWorldModel,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    kl_weight: float = KL_WEIGHT,
) -> Dict[str, torch.Tensor]:
    """Standard RSSM loss for single-model approaches (image-based)."""
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
    """Loss for InterferenceEnsemble approach (image-based)."""
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
class InterferenceCNNWorldModel(nn.Module):
    """Wrapper around InterferenceEnsemble for approach 5 (CNN-based)."""

    def __init__(self, base_seed: Optional[int] = None):
        super().__init__()
        self.ensemble = InterferenceEnsemble(
            model_class=BaseCNNWorldModel,
            num_models=NUM_ENSEMBLE_MODELS,
            interference_strength=INTERFERENCE_STRENGTH,
            uncertainty_method="disagreement",
            base_seed=base_seed,
            action_dim=ACTION_DIM,
            stoch_dim=STOCH_DIM,
            deter_dim=DETER_DIM,
            hidden_dim=HIDDEN_DIM,
        )

    def forward(self, obs_seq, action_seq):
        return self.ensemble(obs_seq, action_seq, return_all=True)


# ---------------------------------------------------------------------------
# Helper: convert buffer sample to tensors with channel dim
# ---------------------------------------------------------------------------
def buffer_to_tensors(obs_np, act_np, rew_np):
    """Convert numpy buffer samples to GPU tensors, adding channel dim to obs."""
    # obs_np: (B, T, 84, 84) -> (B, T, 1, 84, 84)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(2)
    actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
    rewards = torch.tensor(rew_np, dtype=torch.float32, device=DEVICE)
    return obs, actions, rewards


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
            # SuperpositionReplayBuffer returns dict with torch tensors
            obs = batch["obs"].to(DEVICE)
            # Add channel dim if needed: (B, T, 84, 84) -> (B, T, 1, 84, 84)
            if obs.dim() == 4:
                obs = obs.unsqueeze(2)
            actions = batch["actions"].to(DEVICE)
            rewards = batch["rewards"].to(DEVICE)
        else:
            obs_np, act_np, rew_np, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
            obs, actions, rewards = buffer_to_tensors(obs_np, act_np, rew_np)

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
    model: InterferenceCNNWorldModel,
    buffer: ReplayBuffer,
    num_steps: int = NUM_STEPS,
) -> pd.DataFrame:
    """Train interference ensemble approach."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for step in range(num_steps):
        model.train()
        obs_np, act_np, rew_np, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
        obs, actions, rewards = buffer_to_tensors(obs_np, act_np, rew_np)

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
            obs, actions, rewards = buffer_to_tensors(obs_np, act_np, rew_np)

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
        horizons = [5, 10]
    max_horizon = max(horizons)

    results = {}
    with torch.no_grad():
        obs_np, act_np, _, _ = buffer.sample(BATCH_SIZE, max_horizon)
        obs, actions, _ = buffer_to_tensors(obs_np, act_np, np.zeros_like(obs_np[:, :, 0, 0]))

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
    """Create the appropriate CNN model for the given approach."""
    kwargs = dict(
        action_dim=ACTION_DIM,
        stoch_dim=STOCH_DIM,
        deter_dim=DETER_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    if approach == "entanglement":
        return EntanglementCNNWorldModel(**kwargs).to(DEVICE)

    if approach == "interference_ensemble":
        return InterferenceCNNWorldModel(base_seed=seed).to(DEVICE)

    # baseline, quantum_tunneling, superposition
    return BaseCNNWorldModel(**kwargs).to(DEVICE)


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
    test_episodes = collect_episodes(ENV_NAME, 50, seed + 10000)
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
    print("PHASE 3: ATARI BREAKOUT - ALL 5 APPROACHES x 5 SEEDS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Approaches: {APPROACHES}")
    print(f"Seeds: {EXPERIMENT_SEEDS}")
    print(f"NUM_EPISODES={NUM_EPISODES}, NUM_STEPS={NUM_STEPS}")
    print(f"BATCH_SIZE={BATCH_SIZE}, SEQ_LEN={SEQ_LEN}")
    print(f"Architecture: CNN encoder -> RSSM (stoch={STOCH_DIM}, deter={DETER_DIM})")
    print(f"OBS_SHAPE={OBS_SHAPE}, ACTION_DIM={ACTION_DIM}")
    print()

    results_dir = PROJECT_ROOT / "experiments" / "results" / "phase3" / "breakout"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for approach in APPROACHES:
        print(f"\n{'=' * 70}")
        print(f"APPROACH: {approach}")
        print(f"{'=' * 70}")

        for seed in EXPERIMENT_SEEDS:
            print(f"\n  Collecting training data (seed={seed}, {NUM_EPISODES} episodes)...")
            train_episodes = collect_episodes(ENV_NAME, NUM_EPISODES, seed)
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
    print("AGGREGATED RESULTS - ATARI BREAKOUT")
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
        "experiment": "phase3_atari_breakout",
        "environment": {
            "name": ENV_NAME,
            "obs_shape": list(OBS_SHAPE),
            "action_dim": ACTION_DIM,
            "action_type": "discrete",
        },
        "config": {
            "stoch_dim": STOCH_DIM,
            "deter_dim": DETER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "cnn_flatten_dim": CNN_FLATTEN_DIM,
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
