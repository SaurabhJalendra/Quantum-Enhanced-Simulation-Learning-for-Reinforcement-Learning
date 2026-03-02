"""Run remaining CartPole experiments: IE seed=1024 + all uniform_ensemble seeds."""
import sys
import os
import logging

# Setup logging to file AND stderr for reliable output
LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'cartpole_remaining.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stderr),
    ]
)
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from quantum_inspired import InterferenceEnsemble

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Config
SEEDS = [42, 123, 456, 789, 1024]
STOCH_DIM = 64
DETER_DIM = 512
HIDDEN_DIM = 512
NUM_STEPS = 10000
BATCH_SIZE = 32
SEQ_LEN = 20
LR = 3e-4
KL_WEIGHT = 1.0
GRAD_CLIP = 100.0
OBS_DIM = 4
ACTION_DIM = 1

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "phase1", "cartpole")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.episodes = []
        self.capacity = capacity

    def add(self, episode):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample(self, batch_size, seq_len):
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
                obs = ep["obs"][s:s+seq_len]
                act = ep["actions"][s:s+seq_len]
                rew = ep["rewards"][s:s+seq_len]
                done = ep["dones"][s:s+seq_len]
            obs_b.append(obs); act_b.append(act)
            rew_b.append(rew); done_b.append(done)
        return np.array(obs_b), np.array(act_b), np.array(rew_b), np.array(done_b)

    def __len__(self):
        return len(self.episodes)


class BaseWorldModel(nn.Module):
    """RSSM matching original experiment scripts: input_proj(stoch+action)->GRU."""
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
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ELU()
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_pred = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
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
        h = torch.zeros(B, self.deter_dim, device=device)
        z = torch.zeros(B, self.stoch_dim, device=device)
        predictions, priors, posteriors = [], [], []
        deter_states, stoch_states = [], []
        for t in range(T):
            embed = self.encoder(obs_seq[:, t])
            act = action_seq[:, t] if action_seq.dim() == 3 else action_seq[:, t:t+1]
            x = self.input_proj(torch.cat([z, act], dim=-1))
            h = self.gru(x, h)
            prior_dist = self._get_dist(self.prior_net(h))
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
        return predictions, {
            "deter": deter_states, "stoch": stoch_states,
            "priors": priors, "posteriors": posteriors,
        }


class UniformEnsembleWorldModel(nn.Module):
    """5-model ensemble with uniform (equal) weights. No phase interference."""
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, num_models=5, base_seed=None):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList()
        for i in range(num_models):
            seed_i = (base_seed + i * 1000) if base_seed else None
            if seed_i:
                set_seed(seed_i)
            self.models.append(BaseWorldModel(obs_dim=obs_dim, action_dim=action_dim))

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


class InterferenceWorldModel(nn.Module):
    """IE wrapper for state-based envs."""
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, base_seed=None):
        super().__init__()
        self.ensemble = InterferenceEnsemble(
            model_class=BaseWorldModel, num_models=5,
            interference_strength=0.7, uncertainty_method="disagreement",
            base_seed=base_seed, obs_dim=obs_dim, action_dim=action_dim,
            stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM,
        )

    def forward(self, obs_seq, action_seq):
        return self.ensemble(obs_seq, action_seq, return_all=True)


def _get_models(model):
    if hasattr(model, "ensemble"):
        return model.ensemble.models
    elif hasattr(model, "models"):
        return model.models
    raise ValueError(f"Cannot extract models from {type(model)}")


def compute_ensemble_loss(model, obs_seq, action_seq, reward_seq, kl_weight=KL_WEIGHT):
    combined_pred, states = model(obs_seq, action_seq)
    all_predictions = states.get("all_predictions", None)

    combined_recon = F.mse_loss(combined_pred, obs_seq)
    individual_loss = torch.tensor(0.0, device=obs_seq.device)
    if all_predictions is not None:
        individual_loss = torch.stack([F.mse_loss(p, obs_seq) for p in all_predictions]).mean()

    kl_losses = []
    if "all_states" in states:
        for ms in states["all_states"]:
            if "priors" in ms and "posteriors" in ms:
                for prior, posterior in zip(ms["priors"], ms["posteriors"]):
                    kl = torch.distributions.kl_divergence(posterior, prior).sum(-1).mean()
                    kl_losses.append(kl)
    kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=obs_seq.device)

    diversity = torch.tensor(0.0, device=obs_seq.device)
    if all_predictions is not None:
        mean_pred = all_predictions.mean(dim=0)
        diversity = ((all_predictions - mean_pred) ** 2).mean()

    # Reward prediction loss
    reward_loss = torch.tensor(0.0, device=obs_seq.device)
    if "all_states" in states:
        reward_losses = []
        models = _get_models(model)
        for i, ms in enumerate(states["all_states"]):
            combined = torch.cat([ms["deter"], ms["stoch"]], dim=-1)
            rpred = models[i].reward_pred(combined).squeeze(-1)
            reward_losses.append(F.mse_loss(rpred, reward_seq))
        reward_loss = torch.stack(reward_losses).mean()

    total = 0.5 * combined_recon + 0.5 * individual_loss + kl_weight * kl_loss + reward_loss - 0.01 * diversity
    return {"total": total, "combined_recon": combined_recon, "individual_recon": individual_loss,
            "kl": kl_loss, "reward": reward_loss, "diversity": diversity}


def collect_episodes(num_episodes, seed):
    env = gym.make("CartPole-v1")
    episodes = []
    rng = np.random.RandomState(seed)
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        ep = {"obs": [], "actions": [], "rewards": [], "dones": []}
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep["obs"].append(obs.copy())
            ep["actions"].append([float(action)])
            ep["rewards"].append(float(reward))
            ep["dones"].append(float(done))
            obs = next_obs
        for k in ep:
            ep[k] = np.array(ep[k], dtype=np.float32)
        if len(ep["obs"]) >= SEQ_LEN:
            episodes.append(ep)
    env.close()
    return episodes


def train_and_eval(model, approach, seed):
    log.info(f"  [{approach}] seed={seed}")
    set_seed(seed)
    start = time.time()

    # Collect train data
    log.info(f"    Collecting train episodes...")
    train_eps = collect_episodes(100, seed)
    log.info(f"    Collected {len(train_eps)} episodes")

    buffer = ReplayBuffer()
    for ep in train_eps:
        buffer.add(ep)

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"    Parameters: {num_params:,}")

    # Train
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    for step in range(NUM_STEPS):
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

        if step % 1000 == 0:
            log.info(f"    Step {step}/{NUM_STEPS}: total={losses['total'].item():.4f} "
                     f"combined={losses['combined_recon'].item():.4f} "
                     f"reward={losses['reward'].item():.6f} kl={losses['kl'].item():.4f}")

    elapsed = time.time() - start

    # Eval on train data
    eval_buffer = ReplayBuffer()
    for ep in train_eps[:50]:
        eval_buffer.add(ep)
    train_metrics = _evaluate(model, eval_buffer)

    # Eval on test data
    log.info(f"    Collecting test episodes...")
    test_eps = collect_episodes(50, seed + 10000)
    test_buffer = ReplayBuffer()
    for ep in test_eps:
        test_buffer.add(ep)
    test_metrics = _evaluate(model, test_buffer)

    # Long horizon
    horizon_results = _eval_horizon(model, test_buffer)

    log.info(f"    Train MSE: {train_metrics['obs_mse']:.6f} | "
             f"Test MSE: {test_metrics['obs_mse']:.6f} | "
             f"Reward MSE: {test_metrics['reward_mse']:.6f} | "
             f"Time: {elapsed:.1f}s")

    result = {
        "approach": approach, "seed": seed,
        "train_obs_mse": train_metrics["obs_mse"],
        "train_reward_mse": train_metrics["reward_mse"],
        "test_obs_mse": test_metrics["obs_mse"],
        "test_reward_mse": test_metrics["reward_mse"],
        "long_horizon": {str(k): v for k, v in horizon_results.items()},
        "final_train_loss": losses["total"].item(),
        "time_seconds": elapsed,
        "num_params": num_params,
    }

    # Save
    outpath = os.path.join(RESULTS_DIR, f"{approach}_seed_{seed}.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"    Saved: {outpath}")

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()
    return result


def _evaluate(model, buffer, num_batches=10):
    total_mse = 0.0
    total_rmse = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            obs_np, act_np, rew_np, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(rew_np, dtype=torch.float32, device=DEVICE)
            pred, states = model(obs, actions)
            total_mse += F.mse_loss(pred, obs).item()
            models = _get_models(model)
            if "all_states" in states:
                fs = states["all_states"][0]
            else:
                fs = states
            combined = torch.cat([fs["deter"], fs["stoch"]], dim=-1)
            rpred = models[0].reward_pred(combined).squeeze(-1)
            total_rmse += F.mse_loss(rpred, rewards).item()
    return {"obs_mse": total_mse / num_batches, "reward_mse": total_rmse / num_batches}


def _eval_horizon(model, buffer, horizons=None):
    if horizons is None:
        horizons = [5, 10, 15, 20]
    results = {}
    model.eval()
    with torch.no_grad():
        obs_np, act_np, _, _ = buffer.sample(BATCH_SIZE, max(horizons))
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(act_np, dtype=torch.float32, device=DEVICE)
        pred, _ = model(obs, actions)
        for h in horizons:
            results[h] = F.mse_loss(pred[:, :h], obs[:, :h]).item()
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. IE seed=1024 (remaining)
    log.info("=" * 60)
    log.info("PART 1: IE seed=1024")
    log.info("=" * 60)
    try:
        model = InterferenceWorldModel(base_seed=1024).to(DEVICE)
        train_and_eval(model, "interference_ensemble", 1024)
    except Exception as e:
        log.error(f"IE seed=1024 FAILED: {e}")
        traceback.print_exc()

    # 2. Uniform ensemble - all 5 seeds
    log.info("=" * 60)
    log.info("PART 2: Uniform Ensemble (5 seeds)")
    log.info("=" * 60)
    for seed in SEEDS:
        try:
            model = UniformEnsembleWorldModel(base_seed=seed).to(DEVICE)
            train_and_eval(model, "uniform_ensemble", seed)
        except Exception as e:
            log.error(f"uniform_ensemble seed={seed} FAILED: {e}")
            traceback.print_exc()

    log.info("ALL DONE")


if __name__ == "__main__":
    main()
