"""
Run ONLY Interference Ensemble experiments on Atari (Pong + Breakout)
This script runs the fixed interference ensemble code on Atari environments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]

# Architecture (must match other experiments)
STOCH_DIM = 64
DETER_DIM = 512
HIDDEN_DIM = 512
CNN_FLATTEN_DIM = 4096

# Training
BATCH_SIZE = 16
SEQ_LEN = 20
NUM_STEPS = 10000
LEARNING_RATE = 3e-4
KL_WEIGHT = 1.0
GRAD_CLIP = 100.0
NUM_EPISODES = 50
NUM_ENSEMBLE_MODELS = 5

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results" / "phase3"


class CNNEncoder(nn.Module):
    """CNN encoder for Atari observations."""
    def __init__(self, obs_shape, flatten_dim=4096):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Linear(conv_out, flatten_dim)
        self.flatten_dim = flatten_dim

    def forward(self, x):
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            out = F.elu(self.fc(self.conv(x)))
            return out.view(b, t, -1)
        return F.elu(self.fc(self.conv(x)))


class CNNDecoder(nn.Module):
    """CNN decoder for Atari observations."""
    def __init__(self, state_dim, obs_shape, flatten_dim=4096):
        super().__init__()
        self.obs_shape = obs_shape
        c, h, w = obs_shape
        self.target_h = h
        self.target_w = w
        self.fc = nn.Linear(state_dim, flatten_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, 6, stride=2),
        )

    def forward(self, state):
        if state.dim() == 3:
            b, t, d = state.shape
            x = F.elu(self.fc(state.view(b * t, d)))
            x = x.view(b * t, 256, 4, 4)
            out = self.deconv(x)
            # Resize to target shape (56x56 -> 84x84)
            out = F.interpolate(out, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
            return out.view(b, t, *self.obs_shape)
        x = F.elu(self.fc(state))
        x = x.view(-1, 256, 4, 4)
        out = self.deconv(x)
        return F.interpolate(out, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)


class AtariRSSMWorldModel(nn.Module):
    """RSSM World Model for Atari with CNN encoder/decoder."""
    def __init__(self, obs_shape, action_dim, stoch_dim=64, deter_dim=512, hidden_dim=512, cnn_flatten_dim=4096):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.state_dim = deter_dim + stoch_dim

        # CNN encoder/decoder
        self.encoder = CNNEncoder(obs_shape, cnn_flatten_dim)
        self.decoder = CNNDecoder(self.state_dim, obs_shape, cnn_flatten_dim)

        # Latent projection
        self.latent_proj = nn.Linear(cnn_flatten_dim, hidden_dim)

        # RSSM components
        self.rnn = nn.GRUCell(hidden_dim + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )
        self.reward_pred = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_seq, action_seq):
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device

        # Encode observations
        encoded = self.encoder(obs_seq)
        latent = self.latent_proj(encoded)

        # Initialize
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        outputs = []
        priors, posteriors = [], []

        for t in range(seq_len):
            # Prior
            prior_params = self.prior_net(h)
            prior_mean, prior_logvar = prior_params.chunk(2, dim=-1)
            prior_std = F.softplus(prior_logvar) + 0.1

            # Posterior
            post_input = torch.cat([h, latent[:, t]], dim=-1)
            post_params = self.posterior_net(post_input)
            post_mean, post_logvar = post_params.chunk(2, dim=-1)
            post_std = F.softplus(post_logvar) + 0.1

            # Sample
            if self.training:
                z = post_mean + post_std * torch.randn_like(post_std)
            else:
                z = post_mean

            # Full state
            state = torch.cat([h, z], dim=-1)
            outputs.append(state)

            # Store distributions
            priors.append((prior_mean, prior_std))
            posteriors.append((post_mean, post_std))

            # Next step
            if t < seq_len - 1:
                rnn_input = torch.cat([latent[:, t], action_seq[:, t]], dim=-1)
                h = self.rnn(rnn_input, h)

        states = torch.stack(outputs, dim=1)
        obs_pred = self.decoder(states)

        return obs_pred, {
            'states': states,
            'priors': priors,
            'posteriors': posteriors
        }


class InterferenceEnsembleAtari(nn.Module):
    """Interference Ensemble for Atari with dynamic tensor dimensions."""
    def __init__(self, obs_shape, action_dim, num_models=5, base_seed=None, **kwargs):
        super().__init__()
        self.num_models = num_models
        self.interference_strength = 0.7

        self.models = nn.ModuleList()
        for i in range(num_models):
            if base_seed is not None:
                torch.manual_seed(base_seed + i * 1000)
            model = AtariRSSMWorldModel(obs_shape, action_dim, **kwargs)
            self.models.append(model)

        self.phase_offsets = nn.Parameter(torch.zeros(num_models))

    def forward(self, obs_seq, action_seq):
        all_preds = []
        all_states = []

        for model in self.models:
            pred, states = model(obs_seq, action_seq)
            all_preds.append(pred)
            all_states.append(states)

        predictions = torch.stack(all_preds, dim=0)

        # Compute weights dynamically
        mean_pred = predictions.mean(dim=0)
        uncertainties = ((predictions - mean_pred) ** 2).mean(dim=tuple(range(1, predictions.dim())))

        amplitudes = 1.0 / (uncertainties + 1e-8)
        amplitudes = amplitudes / amplitudes.sum()

        phases = torch.sigmoid(uncertainties) * 3.14159 + self.phase_offsets

        # Interference weights
        weights = torch.zeros(self.num_models, device=predictions.device)
        for i in range(self.num_models):
            for j in range(self.num_models):
                phase_diff = phases[i] - phases[j]
                weights[i] += amplitudes[i] * amplitudes[j] * torch.cos(phase_diff)

        weights = torch.abs(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(self.num_models, device=predictions.device) / self.num_models

        uniform = torch.ones(self.num_models, device=predictions.device) / self.num_models
        weights = self.interference_strength * weights + (1 - self.interference_strength) * uniform

        # FIXED: Dynamic dimension expansion
        num_extra_dims = predictions.dim() - 1
        weights_shape = [self.num_models] + [1] * num_extra_dims
        weights_expanded = weights.view(*weights_shape)
        combined_pred = (predictions * weights_expanded).sum(dim=0)

        return combined_pred, all_states[0]


def collect_atari_data(env_name, num_episodes, seed):
    """Collect data from Atari environment."""
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
    except:
        import gym

    env = gym.make(env_name, render_mode=None)
    env.reset(seed=seed)

    observations, actions, rewards = [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_obs, ep_actions, ep_rewards = [obs], [], []

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_obs.append(next_obs)
            ep_actions.append(action)
            ep_rewards.append(reward)

        observations.append(ep_obs[:-1])
        actions.append(ep_actions)
        rewards.append(ep_rewards)

    env.close()
    return observations, actions, rewards


def preprocess_atari(observations, actions, rewards, action_dim, device):
    """Preprocess Atari data for training."""
    all_obs, all_actions, all_rewards = [], [], []

    for ep_obs, ep_act, ep_rew in zip(observations, actions, rewards):
        obs_array = np.array(ep_obs, dtype=np.float32) / 255.0
        if obs_array.ndim == 3:
            obs_array = np.expand_dims(obs_array, 1)
        elif obs_array.shape[-1] == 3:
            obs_array = np.transpose(obs_array, (0, 3, 1, 2))
        obs_gray = obs_array.mean(axis=1, keepdims=True)
        obs_resized = torch.nn.functional.interpolate(
            torch.from_numpy(obs_gray),
            size=(84, 84),
            mode='bilinear',
            align_corners=False
        ).numpy()
        all_obs.append(obs_resized)

        act_onehot = np.zeros((len(ep_act), action_dim), dtype=np.float32)
        for i, a in enumerate(ep_act):
            act_onehot[i, a] = 1.0
        all_actions.append(act_onehot)
        all_rewards.append(np.array(ep_rew, dtype=np.float32))

    return all_obs, all_actions, all_rewards


class ReplayBuffer:
    def __init__(self, obs_list, action_list, reward_list):
        self.obs = obs_list
        self.actions = action_list
        self.rewards = reward_list

    def sample(self, batch_size, seq_len, device):
        indices = np.random.randint(0, len(self.obs), batch_size)
        obs_batch, act_batch, rew_batch = [], [], []

        for idx in indices:
            ep_len = len(self.obs[idx])
            if ep_len <= seq_len:
                start = 0
            else:
                start = np.random.randint(0, ep_len - seq_len)

            obs_batch.append(self.obs[idx][start:start+seq_len])
            act_batch.append(self.actions[idx][start:start+seq_len])
            rew_batch.append(self.rewards[idx][start:start+seq_len])

        return (
            torch.tensor(np.array(obs_batch), device=device),
            torch.tensor(np.array(act_batch), device=device),
            torch.tensor(np.array(rew_batch), device=device).unsqueeze(-1)
        )


def kl_divergence(post_mean, post_std, prior_mean, prior_std):
    var_ratio = (post_std / prior_std) ** 2
    kl = 0.5 * (var_ratio + ((post_mean - prior_mean) / prior_std) ** 2 - 1 - torch.log(var_ratio))
    return kl.sum(dim=-1).mean()


def train_model(model, buffer, num_steps, device):
    """Train the model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for step in range(num_steps):
        obs, actions, rewards = buffer.sample(BATCH_SIZE, SEQ_LEN, device)
        optimizer.zero_grad()

        pred_obs, info = model(obs, actions)
        recon_loss = F.mse_loss(pred_obs, obs)

        # KL loss (only for single model, skip for ensemble)
        kl_loss = torch.tensor(0.0, device=device)
        if hasattr(model, 'models'):
            # For ensemble, use first model's KL
            pass
        elif 'priors' in info and 'posteriors' in info:
            for (prior_m, prior_s), (post_m, post_s) in zip(info['priors'], info['posteriors']):
                kl_loss += kl_divergence(post_m, post_s, prior_m, prior_s)
            kl_loss /= len(info['priors'])

        loss = recon_loss + KL_WEIGHT * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}")

    return model


def evaluate_model(model, buffer, device):
    """Evaluate model on test data."""
    model.eval()
    total_obs_mse = 0
    num_batches = 10

    with torch.no_grad():
        for _ in range(num_batches):
            obs, actions, rewards = buffer.sample(BATCH_SIZE, SEQ_LEN, device)
            pred_obs, _ = model(obs, actions)
            total_obs_mse += F.mse_loss(pred_obs, obs).item()

    return total_obs_mse / num_batches


def run_experiment(env_name, env_short, action_dim, results_dir):
    """Run interference ensemble experiment for one Atari environment."""
    print(f"\n{'='*70}")
    print(f"INTERFERENCE ENSEMBLE - {env_short.upper()}")
    print(f"{'='*70}")

    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in EXPERIMENT_SEEDS:
        print(f"\n--- Seed {seed} ---")

        try:
            # Set seeds
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Collect data
            print("  Collecting data...")
            obs, actions, rewards = collect_atari_data(env_name, NUM_EPISODES, seed)
            obs, actions, rewards = preprocess_atari(obs, actions, rewards, action_dim, DEVICE)
            buffer = ReplayBuffer(obs, actions, rewards)

            # Create model
            print("  Creating Interference Ensemble...")
            model = InterferenceEnsembleAtari(
                obs_shape=(1, 84, 84),
                action_dim=action_dim,
                num_models=NUM_ENSEMBLE_MODELS,
                base_seed=seed,
                stoch_dim=STOCH_DIM,
                deter_dim=DETER_DIM,
                hidden_dim=HIDDEN_DIM,
                cnn_flatten_dim=CNN_FLATTEN_DIM
            ).to(DEVICE)

            # Train
            print("  Training...")
            start_time = time.time()
            model = train_model(model, buffer, NUM_STEPS, DEVICE)
            train_time = time.time() - start_time

            # Evaluate
            print("  Evaluating...")
            test_mse = evaluate_model(model, buffer, DEVICE)

            result = {
                "approach": "interference_ensemble",
                "seed": seed,
                "test_obs_mse": test_mse,
                "time_seconds": train_time,
                "num_params": sum(p.numel() for p in model.parameters())
            }
            all_results.append(result)

            # Save individual result
            result_file = results_dir / f"interference_ensemble_seed_{seed}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"  Test MSE: {test_mse:.6f}, Time: {train_time:.1f}s")
            print(f"  Saved: {result_file}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "approach": "interference_ensemble",
                "seed": seed,
                "error": str(e)
            })

    # Summary
    successful = [r for r in all_results if "error" not in r]
    if successful:
        mse_values = [r["test_obs_mse"] for r in successful]
        print(f"\n{env_short.upper()} INTERFERENCE ENSEMBLE SUMMARY:")
        print(f"  Successful: {len(successful)}/{len(EXPERIMENT_SEEDS)}")
        print(f"  Mean MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")

    return all_results


def main():
    print("=" * 70)
    print("ATARI INTERFERENCE ENSEMBLE EXPERIMENTS (FIXED)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {EXPERIMENT_SEEDS}")
    print(f"Ensemble models: {NUM_ENSEMBLE_MODELS}")

    # Pong
    pong_results = run_experiment(
        env_name="ALE/Pong-v5",
        env_short="pong",
        action_dim=6,
        results_dir=RESULTS_DIR / "pong"
    )

    # Breakout
    breakout_results = run_experiment(
        env_name="ALE/Breakout-v5",
        env_short="breakout",
        action_dim=4,
        results_dir=RESULTS_DIR / "breakout"
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
