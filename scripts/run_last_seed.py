"""Run ONLY uniform_ensemble seed=1024 on CartPole."""
import sys, os, json, time, gc, traceback, logging
import numpy as np

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'last_seed.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stderr)])
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import gymnasium as gym

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'N/A'}")

SEED = 1024
STOCH_DIM, DETER_DIM, HIDDEN_DIM = 64, 512, 512
NUM_STEPS, BATCH_SIZE, SEQ_LEN, LR, KL_WEIGHT, GRAD_CLIP = 10000, 32, 20, 3e-4, 1.0, 100.0
OBS_DIM, ACTION_DIM = 4, 1
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "phase1", "cartpole")

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class ReplayBuffer:
    def __init__(self, cap=10000): self.episodes, self.capacity = [], cap
    def add(self, ep):
        if len(self.episodes) >= self.capacity: self.episodes.pop(0)
        self.episodes.append(ep)
    def sample(self, bs, sl):
        obs_b, act_b, rew_b, done_b = [], [], [], []
        for _ in range(bs):
            ep = self.episodes[np.random.randint(len(self.episodes))]; L = len(ep["obs"])
            if L <= sl:
                pad = sl - L
                obs_b.append(np.pad(ep["obs"],((0,pad),(0,0)),mode="edge"))
                act_b.append(np.pad(ep["actions"],((0,pad),(0,0)),mode="edge"))
                rew_b.append(np.pad(ep["rewards"],(0,pad),mode="edge"))
                done_b.append(np.pad(ep["dones"],(0,pad),mode="edge"))
            else:
                s = np.random.randint(0, L - sl)
                obs_b.append(ep["obs"][s:s+sl]); act_b.append(ep["actions"][s:s+sl])
                rew_b.append(ep["rewards"][s:s+sl]); done_b.append(ep["dones"][s:s+sl])
        return np.array(obs_b), np.array(act_b), np.array(rew_b), np.array(done_b)
    def __len__(self): return len(self.episodes)

class BaseWorldModel(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, stoch_dim=STOCH_DIM, deter_dim=DETER_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.stoch_dim, self.deter_dim, self.hidden_dim = stoch_dim, deter_dim, hidden_dim
        self.state_dim = stoch_dim + deter_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Sequential(nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ELU())
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = nn.Sequential(nn.Linear(deter_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, stoch_dim * 2))
        self.posterior_net = nn.Sequential(nn.Linear(deter_dim + hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, stoch_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, obs_dim))
        self.reward_pred = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.continue_pred = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
    def _get_dist(self, stats):
        m, ls = stats.chunk(2, dim=-1); return torch.distributions.Normal(m, F.softplus(ls) + 0.1)
    def forward(self, obs_seq, action_seq):
        B, T, _ = obs_seq.shape; device = obs_seq.device
        h = torch.zeros(B, self.deter_dim, device=device); z = torch.zeros(B, self.stoch_dim, device=device)
        preds, priors, posts, ds, ss = [], [], [], [], []
        for t in range(T):
            embed = self.encoder(obs_seq[:, t]); act = action_seq[:, t] if action_seq.dim() == 3 else action_seq[:, t:t+1]
            h = self.gru(self.input_proj(torch.cat([z, act], -1)), h)
            prior = self._get_dist(self.prior_net(h)); post = self._get_dist(self.posterior_net(torch.cat([h, embed], -1)))
            z = post.rsample(); state = torch.cat([h, z], -1)
            preds.append(self.decoder(state)); priors.append(prior); posts.append(post); ds.append(h); ss.append(z)
        return torch.stack(preds, 1), {"deter": torch.stack(ds, 1), "stoch": torch.stack(ss, 1), "priors": priors, "posteriors": posts}

class UniformEnsemble(nn.Module):
    def __init__(self, base_seed=None, num_models=5):
        super().__init__()
        self.models = nn.ModuleList()
        for i in range(num_models):
            if base_seed: set_seed(base_seed + i * 1000)
            self.models.append(BaseWorldModel())
    def forward(self, obs_seq, action_seq):
        all_p, all_s = [], []
        for m in self.models:
            p, s = m(obs_seq, action_seq); all_p.append(p); all_s.append(s)
        stacked = torch.stack(all_p, 0)
        return stacked.mean(0), {"deter": all_s[0]["deter"], "stoch": all_s[0]["stoch"],
            "priors": all_s[0]["priors"], "posteriors": all_s[0]["posteriors"],
            "all_predictions": stacked, "all_states": all_s}

def collect_episodes(n, seed):
    env = gym.make("CartPole-v1"); eps = []; rng = np.random.RandomState(seed)
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        ep = {"obs": [], "actions": [], "rewards": [], "dones": []}; done = False
        while not done:
            a = env.action_space.sample(); nobs, r, term, trunc, _ = env.step(a); done = term or trunc
            ep["obs"].append(obs.copy()); ep["actions"].append([float(a)]); ep["rewards"].append(float(r)); ep["dones"].append(float(done)); obs = nobs
        for k in ep: ep[k] = np.array(ep[k], dtype=np.float32)
        if len(ep["obs"]) >= SEQ_LEN: eps.append(ep)
    env.close(); return eps

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED); start = time.time()
    log.info(f"Training uniform_ensemble seed={SEED}")
    train_eps = collect_episodes(100, SEED)
    log.info(f"Collected {len(train_eps)} train episodes")
    buf = ReplayBuffer()
    for ep in train_eps: buf.add(ep)
    model = UniformEnsemble(base_seed=SEED).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {num_params:,}")
    opt = optim.AdamW(model.parameters(), lr=LR)
    for step in range(NUM_STEPS):
        model.train()
        o, a, r, _ = buf.sample(BATCH_SIZE, SEQ_LEN)
        obs = torch.tensor(o, dtype=torch.float32, device=DEVICE)
        acts = torch.tensor(a, dtype=torch.float32, device=DEVICE)
        rews = torch.tensor(r, dtype=torch.float32, device=DEVICE)
        opt.zero_grad()
        pred, states = model(obs, acts)
        ap = states["all_predictions"]; cr = F.mse_loss(pred, obs)
        il = torch.stack([F.mse_loss(p, obs) for p in ap]).mean()
        kl_l = []
        for ms in states["all_states"]:
            for pr, po in zip(ms["priors"], ms["posteriors"]): kl_l.append(torch.distributions.kl_divergence(po, pr).sum(-1).mean())
        kl = torch.stack(kl_l).mean()
        div = ((ap - ap.mean(0)) ** 2).mean()
        rl = []
        for i, ms in enumerate(states["all_states"]):
            c = torch.cat([ms["deter"], ms["stoch"]], -1); rl.append(F.mse_loss(model.models[i].reward_pred(c).squeeze(-1), rews))
        rew_loss = torch.stack(rl).mean()
        total = 0.5*cr + 0.5*il + KL_WEIGHT*kl + rew_loss - 0.01*div
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        if step % 1000 == 0:
            log.info(f"Step {step}/{NUM_STEPS}: total={total.item():.4f} combined={cr.item():.4f} reward={rew_loss.item():.6f} kl={kl.item():.4f}")
    elapsed = time.time() - start
    # Eval
    model.eval()
    def _eval(buffer):
        tm, trm = 0, 0
        with torch.no_grad():
            for _ in range(10):
                o2, a2, r2, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)
                obs2 = torch.tensor(o2, dtype=torch.float32, device=DEVICE)
                acts2 = torch.tensor(a2, dtype=torch.float32, device=DEVICE)
                rews2 = torch.tensor(r2, dtype=torch.float32, device=DEVICE)
                p2, s2 = model(obs2, acts2); tm += F.mse_loss(p2, obs2).item()
                fs = s2["all_states"][0]; c = torch.cat([fs["deter"], fs["stoch"]], -1)
                trm += F.mse_loss(model.models[0].reward_pred(c).squeeze(-1), rews2).item()
        return {"obs_mse": tm/10, "reward_mse": trm/10}
    eb = ReplayBuffer()
    for ep in train_eps[:50]: eb.add(ep)
    train_m = _eval(eb)
    log.info(f"Collecting test episodes...")
    test_eps = collect_episodes(50, SEED + 10000)
    tb = ReplayBuffer()
    for ep in test_eps: tb.add(ep)
    test_m = _eval(tb)
    # Horizon
    with torch.no_grad():
        o3, a3, _, _ = tb.sample(BATCH_SIZE, 20)
        obs3 = torch.tensor(o3, dtype=torch.float32, device=DEVICE); acts3 = torch.tensor(a3, dtype=torch.float32, device=DEVICE)
        p3, _ = model(obs3, acts3)
        hz = {str(h): F.mse_loss(p3[:,:h], obs3[:,:h]).item() for h in [5,10,15,20]}
    log.info(f"Train MSE: {train_m['obs_mse']:.6f} | Test MSE: {test_m['obs_mse']:.6f} | Reward MSE: {test_m['reward_mse']:.6f} | Time: {elapsed:.1f}s")
    result = {"approach": "uniform_ensemble", "seed": SEED, "train_obs_mse": train_m["obs_mse"],
        "train_reward_mse": train_m["reward_mse"], "test_obs_mse": test_m["obs_mse"],
        "test_reward_mse": test_m["reward_mse"], "long_horizon": hz,
        "final_train_loss": total.item(), "time_seconds": elapsed, "num_params": num_params}
    outpath = os.path.join(RESULTS_DIR, f"uniform_ensemble_seed_{SEED}.json")
    with open(outpath, "w") as f: json.dump(result, f, indent=2)
    log.info(f"Saved: {outpath}")
    log.info("DONE")

if __name__ == "__main__":
    try: main()
    except Exception as e: log.error(f"FAILED: {e}\n{traceback.format_exc()}")
