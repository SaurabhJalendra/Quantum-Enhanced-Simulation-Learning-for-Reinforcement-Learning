"""
Run CartPole Interference Ensemble for missing seeds (456, 789, 1024).

Seeds 42 and 123 already exist in experiments/results/phase1/cartpole/.
After running this, re-run aggregate_cartpole_results.py to update complete_metrics.json.

Expected runtime: ~45 min per seed (3 seeds = ~2.25 hours) on RTX 5090.
"""

import sys
import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_inspired.interference_ensemble import InterferenceEnsemble

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Standard config (matches all notebooks)
OBS_DIM = 4
ACTION_DIM = 2
STOCH_DIM = 64
DETER_DIM = 512
HIDDEN_DIM = 512
NUM_EPISODES = 100
NUM_STEPS = 10000
BATCH_SIZE = 32
SEQ_LEN = 20
LEARNING_RATE = 3e-4
KL_WEIGHT = 1.0
NUM_MODELS = 5
INTERFERENCE_STRENGTH = 0.7

MISSING_SEEDS = [456, 789, 1024]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "phase1" / "cartpole"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- RSSM World Model (identical to all notebooks) ----

class RSSMState(NamedTuple):
    deter: torch.Tensor
    stoch: torch.Tensor

    @property
    def combined(self):
        return torch.cat([self.deter, self.stoch], dim=-1)


class BaseWorldModel(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = stoch_dim + deter_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ELU()
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        self.reward_pred = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.continue_pred = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def initial_state(self, batch_size, device):
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device)
        )

    def _get_dist(self, stats):
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return torch.distributions.Normal(mean, std)

    def observe(self, obs, action, state):
        embed = self.encoder(obs)
        x = self.input_proj(torch.cat([state.stoch, action], dim=-1))
        deter = self.gru(x, state.deter)
        posterior_stats = self.posterior_net(torch.cat([deter, embed], dim=-1))
        posterior = self._get_dist(posterior_stats)
        stoch = posterior.rsample()
        prior_stats = self.prior_net(deter)
        prior = self._get_dist(prior_stats)
        return RSSMState(deter, stoch), prior, posterior

    def decode(self, state):
        return self.decoder(state.combined)

    def forward(self, obs_seq, action_seq):
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device
        state = self.initial_state(batch_size, device)

        recon_obs, all_deter, all_stoch, priors, posteriors = [], [], [], [], []
        for t in range(seq_len):
            state, prior, posterior = self.observe(obs_seq[:, t], action_seq[:, t], state)
            recon_obs.append(self.decode(state))
            all_deter.append(state.deter)
            all_stoch.append(state.stoch)
            priors.append(prior)
            posteriors.append(posterior)

        return torch.stack(recon_obs, dim=1), {
            'deter': torch.stack(all_deter, dim=1),
            'stoch': torch.stack(all_stoch, dim=1),
            'priors': priors,
            'posteriors': posteriors
        }


# ---- Interference World Model ----

class InterferenceWorldModel(nn.Module):
    def __init__(self, base_seed=None):
        super().__init__()
        self.ensemble = InterferenceEnsemble(
            model_class=BaseWorldModel,
            num_models=NUM_MODELS,
            interference_strength=INTERFERENCE_STRENGTH,
            uncertainty_method='disagreement',
            base_seed=base_seed,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM
        )
        self.stoch_dim = STOCH_DIM
        self.deter_dim = DETER_DIM
        self.state_dim = STOCH_DIM + DETER_DIM

    def forward(self, obs_seq, action_seq):
        return self.ensemble(obs_seq, action_seq, return_all=True)


# ---- Data Collection & Buffer ----

def collect_episodes(env_name, num_episodes, seed):
    env = gym.make(env_name)
    episodes = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        observations, actions, rewards, dones = [obs], [], [], []
        done = False
        while not done:
            action = env.action_space.sample()
            action_oh = np.zeros(ACTION_DIM)
            action_oh[action] = 1.0
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            observations.append(obs)
            actions.append(action_oh)
            rewards.append(r)
            dones.append(float(done))
        observations = observations[:-1]
        if len(observations) > 0:
            episodes.append({
                'obs': np.array(observations, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'rewards': np.array(rewards, dtype=np.float32),
                'dones': np.array(dones, dtype=np.float32)
            })
    env.close()
    return episodes


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.episodes = []
        self.capacity = capacity

    def add(self, ep):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(ep)

    def sample(self, batch_size, seq_len):
        obs_b, act_b, rew_b, done_b = [], [], [], []
        for _ in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            L = len(ep['obs'])
            if L <= seq_len:
                pad = seq_len - L
                obs = np.pad(ep['obs'], ((0, pad), (0, 0)), mode='edge')
                act = np.pad(ep['actions'], ((0, pad), (0, 0)), mode='edge')
                rew = np.pad(ep['rewards'], (0, pad), mode='edge')
                done = np.pad(ep['dones'], (0, pad), mode='edge')
            else:
                s = np.random.randint(0, L - seq_len)
                obs = ep['obs'][s:s + seq_len]
                act = ep['actions'][s:s + seq_len]
                rew = ep['rewards'][s:s + seq_len]
                done = ep['dones'][s:s + seq_len]
            obs_b.append(obs)
            act_b.append(act)
            rew_b.append(rew)
            done_b.append(done)
        return np.array(obs_b), np.array(act_b), np.array(rew_b), np.array(done_b)

    def __len__(self):
        return len(self.episodes)


# ---- Training ----

def compute_ensemble_loss(model, obs_seq, action_seq, reward_seq, kl_weight=KL_WEIGHT):
    combined_pred, states = model(obs_seq, action_seq)
    all_predictions = states.get('all_predictions', None)

    combined_recon_loss = F.mse_loss(combined_pred, obs_seq)

    individual_losses = []
    if all_predictions is not None:
        for pred in all_predictions:
            individual_losses.append(F.mse_loss(pred, obs_seq))
        individual_loss = torch.stack(individual_losses).mean()
    else:
        individual_loss = torch.tensor(0.0, device=obs_seq.device)

    kl_losses = []
    if 'all_states' in states:
        for model_states in states['all_states']:
            if 'priors' in model_states and 'posteriors' in model_states:
                for prior, posterior in zip(model_states['priors'], model_states['posteriors']):
                    kl = torch.distributions.kl_divergence(posterior, prior).sum(-1).mean()
                    kl_losses.append(kl)
    kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=obs_seq.device)

    if all_predictions is not None:
        mean_pred = all_predictions.mean(dim=0)
        diversity = ((all_predictions - mean_pred) ** 2).mean()
        diversity_bonus = -0.01 * diversity
    else:
        diversity_bonus = torch.tensor(0.0, device=obs_seq.device)

    total = 0.5 * combined_recon_loss + 0.5 * individual_loss + kl_weight * kl_loss + diversity_bonus
    return total


def train_model(model, buffer, num_steps=NUM_STEPS):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    final_loss = 0.0
    for step in range(num_steps):
        model.train()
        obs, actions, rewards, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        loss = compute_ensemble_loss(model, obs, actions, rewards)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if step % 1000 == 0:
            print(f"    Step {step}/{num_steps}: loss={final_loss:.4f}")

    return final_loss


# ---- Evaluation ----

def evaluate_model(model, buffer, num_batches=10):
    model.eval()
    obs_mses, reward_mses = [], []
    with torch.no_grad():
        for _ in range(num_batches):
            obs, actions, rewards, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            act_t = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
            rew_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

            pred, states = model(obs_t, act_t)
            obs_mses.append(F.mse_loss(pred, obs_t).item())

            # Reward prediction from first ensemble member
            first_states = states["all_states"][0]
            combined = torch.cat([first_states["deter"], first_states["stoch"]], dim=-1)
            rpred = model.ensemble.models[0].reward_pred(combined).squeeze(-1)
            reward_mses.append(F.mse_loss(rpred, rew_t).item())

    return float(np.mean(obs_mses)), float(np.mean(reward_mses))


def evaluate_long_horizon(model, buffer, horizons=[5, 10, 15, 20]):
    model.eval()
    results = {}
    with torch.no_grad():
        for h in horizons:
            obs, actions, _, _ = buffer.sample(BATCH_SIZE, max(h + 5, SEQ_LEN))
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            act_t = torch.tensor(actions, dtype=torch.float32, device=DEVICE)

            pred, _ = model(obs_t[:, :h], act_t[:, :h])
            target = obs_t[:, :h]
            results[str(h)] = float(F.mse_loss(pred, target).item())

    return results


# ---- Main ----

def run_seed(seed):
    print(f"\n{'=' * 60}")
    print(f"  CartPole IE - Seed {seed}")
    print(f"{'=' * 60}")

    set_seed(seed)
    start = time.time()

    # Collect training data
    print(f"  Collecting {NUM_EPISODES} training episodes...")
    episodes = collect_episodes('CartPole-v1', NUM_EPISODES, seed)
    buffer = ReplayBuffer()
    for ep in episodes:
        buffer.add(ep)
    print(f"  Collected {len(episodes)} episodes")

    # Create and train model
    model = InterferenceWorldModel(base_seed=seed).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    print(f"  Training for {NUM_STEPS} steps...")
    final_loss = train_model(model, buffer)

    elapsed = time.time() - start
    print(f"  Training completed in {elapsed:.1f}s")

    # Evaluate on training data
    train_obs_mse, train_reward_mse = evaluate_model(model, buffer)

    # Evaluate on test data
    print(f"  Collecting test episodes...")
    test_eps = collect_episodes('CartPole-v1', 50, seed + 10000)
    test_buffer = ReplayBuffer()
    for ep in test_eps:
        test_buffer.add(ep)
    test_obs_mse, test_reward_mse = evaluate_model(model, test_buffer)

    # Long-horizon prediction
    long_horizon = evaluate_long_horizon(model, test_buffer)

    total_time = time.time() - start

    result = {
        "approach": "interference_ensemble",
        "seed": seed,
        "train_obs_mse": train_obs_mse,
        "train_reward_mse": train_reward_mse,
        "test_obs_mse": test_obs_mse,
        "test_reward_mse": test_reward_mse,
        "long_horizon": long_horizon,
        "final_train_loss": final_loss,
        "time_seconds": total_time,
        "num_params": num_params
    }

    # Save individual seed file
    save_path = RESULTS_DIR / f"interference_ensemble_seed_{seed}.json"
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {save_path}")
    print(f"  Test MSE: {test_obs_mse:.6f}, Train MSE: {train_obs_mse:.6f}")
    print(f"  Total time: {total_time:.1f}s")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("CartPole IE - Missing Seeds (456, 789, 1024)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check which seeds already exist
    for seed in MISSING_SEEDS:
        path = RESULTS_DIR / f"interference_ensemble_seed_{seed}.json"
        if path.exists():
            print(f"  WARNING: {path} already exists, will be overwritten")

    results = []
    for seed in MISSING_SEEDS:
        result = run_seed(seed)
        results.append(result)

    # Summary
    test_mses = [r['test_obs_mse'] for r in results]
    print(f"\n{'=' * 60}")
    print(f"DONE - 3 seeds completed")
    print(f"Test MSEs: {[f'{m:.6f}' for m in test_mses]}")
    print(f"Mean: {np.mean(test_mses):.6f} +/- {np.std(test_mses):.6f}")
    print(f"\nNow run: python scripts/aggregate_cartpole_results.py")
    print(f"{'=' * 60}")
