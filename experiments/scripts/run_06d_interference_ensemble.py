"""
Run 06d Interference Ensemble Experiment
Runs all 5 seeds, saves results to experiments/results/interference_ensemble/
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, NamedTuple

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

from quantum_inspired import InterferenceEnsemble

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Configuration
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]
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
NUM_MODELS = 5
INTERFERENCE_STRENGTH = 0.7


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Data collection
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
            ep = np.random.choice(self.episodes)
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


# RSSM State
class RSSMState(NamedTuple):
    deter: torch.Tensor
    stoch: torch.Tensor

    @property
    def combined(self):
        return torch.cat([self.deter, self.stoch], dim=-1)


class BaseWorldModel(nn.Module):
    """Standard RSSM world model for ensemble - matches baseline architecture."""
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = stoch_dim + deter_dim

        # Encoder [512, 512, 512] - matches baseline
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)  # Latent projection
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

        recon_obs, all_deter, all_stoch = [], [], []
        priors, posteriors = [], []

        for t in range(seq_len):
            state, prior, posterior = self.observe(obs_seq[:, t], action_seq[:, t], state)
            recon_obs.append(self.decode(state))
            all_deter.append(state.deter)
            all_stoch.append(state.stoch)
            priors.append(prior)
            posteriors.append(posterior)

        predictions = torch.stack(recon_obs, dim=1)
        states_dict = {
            'deter': torch.stack(all_deter, dim=1),
            'stoch': torch.stack(all_stoch, dim=1),
            'priors': priors,
            'posteriors': posteriors
        }
        return predictions, states_dict


class InterferenceWorldModel(nn.Module):
    """Ensemble with interference-based combination."""
    def __init__(self, num_models=NUM_MODELS, interference_strength=INTERFERENCE_STRENGTH, base_seed=None):
        super().__init__()
        self.num_models = num_models
        self.interference_strength = interference_strength
        self.ensemble = InterferenceEnsemble(
            model_class=BaseWorldModel,
            num_models=num_models,
            interference_strength=interference_strength,
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

    def get_ensemble_stats(self):
        return self.ensemble.get_stats()


def compute_ensemble_loss(ensemble, obs_seq, action_seq, reward_seq, kl_weight=1.0):
    combined_pred, states = ensemble(obs_seq, action_seq)
    all_predictions = states.get('all_predictions', None)

    combined_recon_loss = F.mse_loss(combined_pred, obs_seq)

    individual_losses = []
    if all_predictions is not None:
        for pred in all_predictions:
            loss = F.mse_loss(pred, obs_seq)
            individual_losses.append(loss)
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
        diversity = torch.tensor(0.0, device=obs_seq.device)

    total = 0.5 * combined_recon_loss + 0.5 * individual_loss + kl_weight * kl_loss + diversity_bonus

    return {
        'total': total,
        'combined_recon': combined_recon_loss,
        'individual_recon': individual_loss,
        'kl': kl_loss,
        'diversity': diversity
    }


def train_ensemble(model, buffer, num_steps=NUM_STEPS, verbose=True):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for step in range(num_steps):
        model.train()
        obs, actions, rewards, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        losses = compute_ensemble_loss(model, obs, actions, rewards)
        losses['total'].backward()
        optimizer.step()

        history.append({
            'step': step,
            'total': losses['total'].item(),
            'combined_recon': losses['combined_recon'].item(),
            'individual_recon': losses['individual_recon'].item(),
            'kl': losses['kl'].item(),
            'diversity': losses['diversity'].item()
        })

        if verbose and step % 1000 == 0:
            print(f"  Step {step}: total={losses['total'].item():.4f}, "
                  f"combined={losses['combined_recon'].item():.4f}, "
                  f"diversity={losses['diversity'].item():.6f}")

    return pd.DataFrame(history)


def evaluate(model, buffer):
    model.eval()
    obs, actions, _, _ = buffer.sample(100, SEQ_LEN)
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred, states = model(obs, actions)
        mse = F.mse_loss(pred, obs).item()
        faulty_models = states.get('faulty_models', [])
        ensemble_weights = states.get('ensemble_weights', None)

    return mse, len(faulty_models), ensemble_weights


def run_experiment(seed):
    print(f"\n{'=' * 50}")
    print(f"Seed {seed}")
    print(f"{'=' * 50}")

    set_seed(seed)
    start = time.time()

    episodes = collect_episodes('CartPole-v1', NUM_EPISODES, seed)
    buffer = ReplayBuffer()
    for ep in episodes:
        buffer.add(ep)

    model = InterferenceWorldModel(
        num_models=NUM_MODELS,
        interference_strength=INTERFERENCE_STRENGTH,
        base_seed=seed
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    history = train_ensemble(model, buffer)

    train_mse, train_faulty, train_weights = evaluate(model, buffer)

    test_eps = collect_episodes('CartPole-v1', 50, seed + 10000)
    test_buffer = ReplayBuffer()
    for ep in test_eps:
        test_buffer.add(ep)
    test_mse, test_faulty, test_weights = evaluate(model, test_buffer)

    print(f"  Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    print(f"  Faulty models (train): {train_faulty}, (test): {test_faulty}")

    return {
        'seed': seed,
        'history': history,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_faulty': train_faulty,
        'test_faulty': test_faulty,
        'time': time.time() - start,
        'num_params': num_params,
        'final_weights': train_weights.cpu().numpy().tolist() if train_weights is not None else None
    }


if __name__ == '__main__':
    print("=" * 60)
    print("INTERFERENCE ENSEMBLE EXPERIMENTS")
    print("=" * 60)

    results = []
    for seed in EXPERIMENT_SEEDS:
        result = run_experiment(seed)
        results.append(result)

    # Aggregate
    test_mses = [r['test_mse'] for r in results]
    train_mses = [r['train_mse'] for r in results]

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS (Interference Ensemble)")
    print("=" * 60)
    print(f"Train MSE: {np.mean(train_mses):.6f} +/- {np.std(train_mses):.6f}")
    print(f"Test MSE:  {np.mean(test_mses):.6f} +/- {np.std(test_mses):.6f}")
    print(f"Model params: {results[0]['num_params']:,}")

    # Save results
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / 'interference_ensemble'
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'approach': 'interference_ensemble',
        'description': 'Interference-based ensemble with error correction',
        'config': {
            'stoch_dim': STOCH_DIM,
            'deter_dim': DETER_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_models': NUM_MODELS,
            'interference_strength': INTERFERENCE_STRENGTH,
            'seeds': EXPERIMENT_SEEDS
        },
        'final_performance': {
            'test_mse_mean': float(np.mean(test_mses)),
            'test_mse_std': float(np.std(test_mses)),
            'train_mse_mean': float(np.mean(train_mses)),
            'train_mse_std': float(np.std(train_mses)),
            'model_params': results[0]['num_params']
        },
        'raw_results': {
            'test_mses': [float(x) for x in test_mses],
            'train_mses': [float(x) for x in train_mses],
            'seeds': EXPERIMENT_SEEDS
        }
    }

    with open(results_dir / 'complete_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    print("DONE")
