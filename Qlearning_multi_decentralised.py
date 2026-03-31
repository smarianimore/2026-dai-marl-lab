"""
Independent Q-Learning (IQL) for VMAS "Navigation"
====================================================
Each agent owns a **separate, sparse Q-table** mapping discretised
(observation, action) pairs to Q-values.

Because the navigation environment has continuous observations and 2-D
continuous actions, both must be discretised:

  observations : every one of the obs_dim dimensions is split into ``obs_bins``
                 equally-spaced bins after clipping to
                 [obs_clip_low, obs_clip_high].

  actions      : each of the 2 action dimensions is split into ``act_bins``
                 bins within the bounds provided by the environment spec, giving
                 act_bins² discrete actions per agent.

The Q-table is implemented as a ``defaultdict`` (sparse), so only visited
state entries consume memory – which is essential because the nominal state
space (obs_bins^obs_dim) is astronomically large.

NOTE – tabular RL does NOT scale to high-dimensional continuous spaces
(the "curse of dimensionality").  This script is kept intentionally simple as a
*pedagogical contrast* to the function-approximation approach found in
``pytorch_multi_decentralised.py`` (IPPO).  Do not expect competitive
navigation performance from this tabular baseline.
"""

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class Params(NamedTuple):
    # ── Environment ──────────────────────────────────────────────────────── #
    max_steps: int    # maximum steps per episode (avoid infinite loops)
    n_agents:  int    # number of agents in the navigation scenario

    # ── Discretisation ───────────────────────────────────────────────────── #
    obs_bins:     int    # bins per observation dimension
    act_bins:     int    # bins per action dimension (2-D → act_bins² total actions)
    obs_clip_low:  float # clip each obs dim below this before discretising
    obs_clip_high: float # clip each obs dim above this before discretising

    # ── Q-learning ───────────────────────────────────────────────────────── #
    episodes: int    # training episodes per run
    lr:       float  # Q-table learning rate  (α)
    gamma:    float  # discount factor        (γ)
    e_init:   float  # initial exploration rate  (ε at episode 0)
    e_min:    float  # minimum exploration rate  (ε floor after decay)
    runs:     int    # independent runs for result averaging

    # ── Plot / output ────────────────────────────────────────────────────── #
    rolling_window_frac: int  # window = episodes // rolling_window_frac
    phase_frac:          int  # first/last slice = episodes // phase_frac
    savefig_folder: Path      # directory where plots are saved


params = Params(
    max_steps      = 100,
    n_agents       = 3,
    # ---- Discretisation ----
    # Keeping obs_bins small is intentional: with a typical VMAS navigation
    # observation of ~18 dimensions, obs_bins=5 → 5^18 ≈ 3.8 billion nominal
    # states.  The sparse Q-table only stores visited states, but convergence
    # is still very slow.  Reduce obs_bins further to speed up exploration.
    obs_bins       = 5,
    act_bins       = 2,
    obs_clip_low   = -2.0,
    obs_clip_high  =  2.0,
    # ---- Q-learning ----
    episodes       = 1_000,
    lr             = 0.5,
    gamma          = 0.9,
    e_init         = 0.9,
    e_min          = 0.05,
    runs           = 1,
    # ---- Plots ----
    rolling_window_frac = 100,   # 4 % of episodes per rolling window
    phase_frac          = 10,   # compare first vs last 10 % of training
    savefig_folder = Path("res/iql_navigation/"),
)

# ε reaches e_min at exactly 2/3 of training:
#   e_min = e_init * e_decay ^ (2/3 * episodes)
#   ⟹  e_decay = (e_min / e_init) ^ (3 / (2 * episodes))
E_DECAY: float = (params.e_min / params.e_init) ** (3.0 / (2.0 * params.episodes))

# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env() -> TransformedEnv:
    """Create a single-instance VMAS navigation environment.

    The ``RewardSum`` transform accumulates per-step rewards into an episode
    reward stored under ``("agents", "episode_reward")``.
    """
    raw = VmasEnv(
        scenario          = "navigation",
        num_envs          = 1,               # one env for tabular learning
        continuous_actions = True,
        max_steps         = params.max_steps,
        device            = "cuda",
        n_agents          = params.n_agents,
    )
    env = TransformedEnv(
        raw,
        RewardSum(
            in_keys  = [raw.reward_key],
            out_keys = [("agents", "episode_reward")],
        ),
    )
    return env

# ─────────────────────────────────────────────────────────────────────────────
# Discretiser
# ─────────────────────────────────────────────────────────────────────────────

class Discretiser:
    """Converts a continuous vector to a tuple of integer bin indices and back.

    Parameters
    ----------
    low, high : array-like
        Per-dimension clip bounds.
    n_bins : int
        Number of equally-spaced bins for every dimension.
    """

    def __init__(self, low, high, n_bins: int):
        self.low    = np.asarray(low,  dtype=np.float64)
        self.high   = np.asarray(high, dtype=np.float64)
        self.n_bins = n_bins
        # n_bins intervals → n_bins-1 internal edges
        self.edges  = [
            np.linspace(lo, hi, n_bins + 1)[1:-1]
            for lo, hi in zip(self.low, self.high)
        ]

    def encode(self, values: np.ndarray) -> tuple:
        """Continuous vector → tuple of integer bin indices (0 … n_bins-1)."""
        values = np.clip(values, self.low, self.high)
        return tuple(int(np.digitize(v, e)) for v, e in zip(values, self.edges))

    def decode(self, indices: tuple) -> np.ndarray:
        """Tuple of bin indices → bin-centre continuous vector."""
        step = (self.high - self.low) / self.n_bins
        return self.low + (np.array(indices, dtype=np.float64) + 0.5) * step

# ─────────────────────────────────────────────────────────────────────────────
# Epsilon-greedy exploration
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonGreedy:
    """Shared ε-greedy explorer with exponential decay.

    The same ε schedule is applied to every agent during a given run; each
    agent simply queries ``choose_action`` with its own Q-value array.

    Decay formula: ε(ep) = max(e_min, e_init · e_decay^ep)
    The decay factor is chosen so that ε = e_min at 2/3 of total episodes.
    """

    def __init__(self, e_init: float, e_min: float, e_decay: float):
        self.e_init  = e_init
        self.e_min   = e_min
        self.e_decay = e_decay
        self.e       = e_init   # current exploration rate

    def reset(self):
        """Reset ε to e_init (call at the start of every run)."""
        self.e = self.e_init

    def decay(self):
        """Apply one multiplicative decay step; floor at e_min."""
        self.e = max(self.e_min, self.e * self.e_decay)

    def choose_action(self, q_values: np.ndarray) -> int:
        """ε-greedy selection over a flat array of Q-values.

        Ties among the maximum Q-values are broken uniformly at random so that
        the agent does not always favour the first action in the table.
        """
        if np.random.random() < self.e:
            return int(np.random.randint(len(q_values)))
        max_q = q_values.max()
        best  = np.where(np.isclose(q_values, max_q))[0]
        return int(np.random.choice(best))

# ─────────────────────────────────────────────────────────────────────────────
# Independent Q-Learning agent
# ─────────────────────────────────────────────────────────────────────────────

class IQLAgent:
    """One Q-learning agent with its own sparse Q-table.

    Sparsity is achieved via ``defaultdict``: only (state, action) pairs that
    have been visited are stored.  Unvisited states default to all-zero
    Q-values, which equals a neutral (unexplored) prior.
    """

    def __init__(self,
                 obs_disc: Discretiser,
                 act_disc: Discretiser,
                 act_dim:  int,
                 lr:       float,
                 gamma:    float):
        self.obs_disc  = obs_disc
        self.act_disc  = act_disc
        self.lr        = lr
        self.gamma     = gamma

        # All discrete action tuples in a fixed canonical order.
        # act_dim=2, act_bins=3 → 9 actions: (0,0),(0,1),...,(2,2)
        self.actions:   list[tuple] = list(product(range(act_disc.n_bins), repeat=act_dim))
        self.n_actions: int         = len(self.actions)

        # Sparse Q-table: state_tuple → np.ndarray of shape (n_actions,)
        self.q_table: dict = defaultdict(lambda: np.zeros(self.n_actions))

    # ── helpers ──────────────────────────────────────────────────────────── #

    def encode_obs(self, obs: np.ndarray) -> tuple:
        """Discretise a continuous observation to a hashable state tuple."""
        return self.obs_disc.encode(obs)

    def to_continuous_action(self, action_idx: int) -> np.ndarray:
        """Map a discrete action index to its continuous bin-centre vector."""
        return self.act_disc.decode(self.actions[action_idx])

    # ── Q-learning Bellman update ─────────────────────────────────────────── #

    def update(self,
               state:      tuple,
               action_idx: int,
               reward:     float,
               next_state: tuple,
               done:       bool) -> None:
        """Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]"""
        q_old  = self.q_table[state][action_idx]
        q_next = 0.0 if done else float(self.q_table[next_state].max())
        td_err = reward + self.gamma * q_next - q_old
        self.q_table[state][action_idx] += self.lr * td_err

# ─────────────────────────────────────────────────────────────────────────────
# Helpers – inspect env specs & build agents
# ─────────────────────────────────────────────────────────────────────────────

def get_env_specs() -> tuple:
    """Open a temporary env to read observation / action dimensions and bounds.

    Returns
    -------
    obs_dim : int
    act_dim : int
    act_low : np.ndarray  shape (act_dim,)
    act_high: np.ndarray  shape (act_dim,)
    """
    env      = make_env()
    obs_spec = env.observation_spec["agents", "observation"]
    act_spec = env.unbatched_action_spec[env.action_key]
    obs_dim  = int(obs_spec.shape[-1])
    act_dim  = int(act_spec.shape[-1])
    # act_spec may have shape (n_agents, act_dim); we only need the per-agent
    # bounds (all agents share the same action space in VMAS navigation).
    act_low  = act_spec.space.low.cpu().numpy().flatten()[:act_dim]
    act_high = act_spec.space.high.cpu().numpy().flatten()[:act_dim]
    try:
        env.close()
    except Exception:
        pass
    return obs_dim, act_dim, act_low, act_high


def build_agents(obs_dim: int, act_dim: int,
                 act_low: np.ndarray, act_high: np.ndarray) -> list:
    """Create params.n_agents fresh IQL agents from environment dimensions.

    All agents share the *same* Discretiser objects (they are stateless), but
    each has its own independent Q-table.
    """
    obs_low  = np.full(obs_dim, params.obs_clip_low)
    obs_high = np.full(obs_dim, params.obs_clip_high)
    obs_disc = Discretiser(obs_low,  obs_high,  params.obs_bins)
    act_disc = Discretiser(act_low,  act_high,  params.act_bins)
    return [
        IQLAgent(
            obs_disc = obs_disc,
            act_disc = act_disc,
            act_dim  = act_dim,
            lr       = params.lr,
            gamma    = params.gamma,
        )
        for _ in range(params.n_agents)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def visualise_random_policy(n_steps: int = 100) -> None:
    """Render ``n_steps`` environment steps driven by a *random* policy.

    Called before training to give an intuitive baseline of how the agents
    behave without any learned knowledge.  VMAS opens an interactive window
    that updates on every step.
    """
    print(f"\n{'─'*62}")
    print(f"  Pre-training visualisation  –  {n_steps} steps, random policy")
    print(f"{'─'*62}")
    env = make_env()
    with torch.no_grad():
        env.rollout(
            max_steps           = n_steps,
            callback            = lambda env, _: env.render(),
            auto_cast_to_device = True,
            break_when_any_done = False,
        )
    try:
        env.close()
    except Exception:
        pass
    print("  Pre-training visualisation complete.\n")


def visualise_trained_policy(agents: list, n_episodes: int = 3) -> None:
    """Render ``n_episodes`` full episodes using the *trained* (greedy) policy.

    Each agent acts greedily (ε = 0): it always picks ``argmax Q(s, ·)`` for
    the current discretised state.  Unvisited states fall back to all-zero
    Q-values, so the agent picks action index 0 (a centre-area action after
    decoding) – essentially still random in uncharted territory.

    Parameters
    ----------
    agents     : list of IQLAgent returned by ``train()``.
    n_episodes : number of episodes to render.
    """
    print(f"\n{'─'*62}")
    print(f"  Post-training visualisation  –  {n_episodes} episodes, greedy policy")
    print(f"{'─'*62}")
    act_dim = len(agents[0].actions[0])
    env     = make_env()

    for ep in range(n_episodes):
        td   = env.reset()
        step = 0
        print(f"  Episode {ep + 1}/{n_episodes} … ", end="", flush=True)

        for step in range(params.max_steps):
            # Move to CPU before converting to numpy (env device may be GPU)
            obs = td["agents"]["observation"].squeeze(0).cpu().numpy()

            continuous = np.zeros((params.n_agents, act_dim))
            for i, agent in enumerate(agents):
                state          = agent.encode_obs(obs[i])
                act_idx        = int(np.argmax(agent.q_table[state]))
                continuous[i]  = agent.to_continuous_action(act_idx)

            td["agents"]["action"] = torch.tensor(
                continuous, dtype=torch.float32
            ).unsqueeze(0)

            td   = env.step(td)
            env.render()

            done = bool(td["next"]["done"].reshape(-1).any())
            if done:
                break
            td = td["next"]

        print(f"finished in {step + 1} step(s).")

    try:
        env.close()
    except Exception:
        pass
    print("  Post-training visualisation complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop – single run
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(
    agents:   list,
    explorer: EpsilonGreedy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train ``agents`` for ``params.episodes`` episodes.

    Parameters
    ----------
    agents   : list of IQLAgent – pre-built with fresh Q-tables
    explorer : EpsilonGreedy    – will be reset before training starts

    Returns
    -------
    episode_rewards : (episodes,)  total reward per episode (sum over agents & steps)
    episode_steps   : (episodes,)  number of environment steps per episode
    epsilon_history : (episodes,)  ε used at the *start* of each episode
    """
    env = make_env()
    explorer.reset()

    act_dim          = len(agents[0].actions[0])   # e.g. 2 for navigation
    episode_rewards  = np.zeros(params.episodes)
    episode_steps    = np.zeros(params.episodes, dtype=int)
    epsilon_history  = np.zeros(params.episodes)

    for ep in tqdm(range(params.episodes), desc="Episodes", leave=False):
        epsilon_history[ep] = explorer.e

        td        = env.reset()
        ep_reward = 0.0
        step      = 0

        for step in range(params.max_steps):

            # ── observe ────────────────────────────────────────────────── #
            # td["agents"]["observation"] shape: (num_envs=1, n_agents, obs_dim)
            obs = td["agents"]["observation"].squeeze(0).cpu().numpy()   # (n_agents, obs_dim)

            # ── each agent independently selects an action ────────────── #
            states         = []
            action_indices = []
            continuous     = np.zeros((params.n_agents, act_dim))

            for i, agent in enumerate(agents):
                state   = agent.encode_obs(obs[i])
                act_idx = explorer.choose_action(agent.q_table[state])
                states.append(state)
                action_indices.append(act_idx)
                continuous[i] = agent.to_continuous_action(act_idx)

            # ── environment step ─────────────────────────────────────── #
            # Action tensor shape required by VMAS: (num_envs, n_agents, act_dim)
            td["agents"]["action"] = torch.tensor(
                continuous, dtype=torch.float32
            ).unsqueeze(0)

            td = env.step(td)

            # ── collect next observations, rewards, done flag ─────────── #
            next_obs = td["next"]["agents"]["observation"].squeeze(0).cpu().numpy()
            rewards  = td["next"]["agents"]["reward"].squeeze(0).squeeze(-1).cpu().numpy()
            done     = bool(td["next"]["done"].reshape(-1).any())

            ep_reward += float(rewards.sum())

            # ── independent Q-table updates (one per agent) ───────────── #
            for i, agent in enumerate(agents):
                next_state = agent.encode_obs(next_obs[i])
                agent.update(
                    state      = states[i],
                    action_idx = action_indices[i],
                    reward     = float(rewards[i]),
                    next_state = next_state,
                    done       = done,
                )

            if done:
                break

            # Advance tensordict: "next" becomes current for the next step
            td = td["next"]

        episode_rewards[ep] = ep_reward
        episode_steps[ep]   = step + 1
        explorer.decay()    # ε decayed exactly once per episode

    try:
        env.close()
    except Exception:
        pass

    return episode_rewards, episode_steps, epsilon_history

# ─────────────────────────────────────────────────────────────────────────────
# Multi-run training
# ─────────────────────────────────────────────────────────────────────────────

def train() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Execute ``params.runs`` independent training runs.

    Returns
    -------
    all_rewards  : (runs, episodes)
    all_steps    : (runs, episodes)
    eps_hist     : (episodes,)  ε schedule (identical across runs)
    last_agents  : list of IQLAgent from the final run (used for visualisation)
    """
    obs_dim, act_dim, act_low, act_high = get_env_specs()

    print("=" * 62)
    print("  Independent Q-Learning  –  VMAS Navigation")
    print("=" * 62)
    print(f"  Observation dim  : {obs_dim}")
    print(f"  Action dim       : {act_dim}")
    print(f"  Discrete states  : {params.obs_bins}^{obs_dim} nominal per agent")
    print(f"                     (sparse table – only visited states stored)")
    print(f"  Discrete actions : {params.act_bins}^{act_dim} = "
          f"{params.act_bins ** act_dim} per agent")
    print(f"  ε decay factor   : {E_DECAY:.6f}  "
          f"(reaches e_min={params.e_min} at ep "
          f"{int(2/3*params.episodes):,})")
    print(f"  Runs × Episodes  : {params.runs} × {params.episodes:,}")
    print("=" * 62)

    all_rewards = np.zeros((params.runs, params.episodes))
    all_steps   = np.zeros((params.runs, params.episodes))
    eps_hist    = None
    last_agents = None

    explorer = EpsilonGreedy(params.e_init, params.e_min, E_DECAY)

    for run in range(params.runs):
        print(f"\n─── Run {run + 1}/{params.runs} ───")
        agents  = build_agents(obs_dim, act_dim, act_low, act_high)
        rewards, steps, eh = train_one_run(agents, explorer)
        all_rewards[run] = rewards
        all_steps[run]   = steps
        if run == 0:
            eps_hist = eh   # ε schedule is deterministic and identical each run
        last_agents = agents  # keep agents from the final run for visualisation

    return all_rewards, all_steps, eps_hist, last_agents

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    """1-D rolling mean via convolution (mode='valid')."""
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _save(fig: plt.Figure, name: str) -> None:
    params.savefig_folder.mkdir(parents=True, exist_ok=True)
    path = params.savefig_folder / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


def plot_epsilon_decay(eps_hist: np.ndarray) -> None:
    """Plot the ε schedule used during training.

    Vertical marker shows the episode at which ε reaches e_min (2/3 point).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps_hist, color="steelblue", lw=1.5, label="ε (current)")
    ax.axhline(params.e_min,  color="red",    ls="--", lw=1,
               label=f"e_min  = {params.e_min}")
    ax.axhline(params.e_init, color="green",  ls="--", lw=1,
               label=f"e_init = {params.e_init}")
    two_thirds = int(2 / 3 * params.episodes)
    ax.axvline(two_thirds, color="orange", ls=":", lw=1,
               label=f"2/3 of training (ep {two_thirds:,})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε  (exploration rate)")
    ax.set_title("Exploration-rate schedule (exponential ε decay)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "iql_epsilon_decay.png")
    plt.show()


def plot_episodic_reward(all_rewards: np.ndarray) -> None:
    """Rolling mean ± rolling std of the episodic reward (averaged over runs)."""
    window = max(1, params.episodes // params.rolling_window_frac)
    mean_r = all_rewards.mean(axis=0)
    std_r  = all_rewards.std(axis=0)
    rm     = _rolling(mean_r, window)
    rs     = _rolling(std_r,  window)
    x      = np.arange(len(rm))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, rm, color="steelblue", lw=1.5, label="mean over runs")
    ax.fill_between(x, rm - rs, rm + rs,
                    alpha=0.25, color="steelblue", label="±1 std (rolling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward  (sum over agents & steps)")
    ax.set_title(
        f"Episodic reward  (rolling mean ± std, window={window}, "
        f"averaged over {params.runs} runs)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "iql_episodic_reward.png")
    plt.show()


def plot_episodic_steps(all_steps: np.ndarray) -> None:
    """Rolling mean ± rolling std of steps per episode (averaged over runs)."""
    window  = max(1, params.episodes // params.rolling_window_frac)
    mean_s  = all_steps.mean(axis=0).astype(float)
    std_s   = all_steps.std(axis=0)
    rm      = _rolling(mean_s, window)
    rs      = _rolling(std_s,  window)
    x       = np.arange(len(rm))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, rm, color="darkorange", lw=1.5, label="mean over runs")
    ax.fill_between(x, rm - rs, rm + rs,
                    alpha=0.25, color="darkorange", label="±1 std (rolling)")
    ax.axhline(params.max_steps, color="red", ls="--", lw=1,
               label=f"max_steps = {params.max_steps}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode")
    ax.set_title(
        f"Episodic steps  (rolling mean ± std, window={window}, "
        f"averaged over {params.runs} runs)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "iql_episodic_steps.png")
    plt.show()


def plot_reward_distribution(all_rewards: np.ndarray) -> None:
    """Box-plot comparing episodic reward distribution at the start vs end of training.

    Uses the run-averaged reward, so each box shows variability *across episodes*
    (not across runs) within the chosen phase.
    """
    tenth  = max(1, params.episodes // params.phase_frac)
    mean_r = all_rewards.mean(axis=0)
    first  = mean_r[:tenth]
    last   = mean_r[-tenth:]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [first, last],
        labels=[
            f"First 1/{params.phase_frac}\n(ep 1–{tenth:,})",
            f"Last 1/{params.phase_frac}\n(ep "
            f"{params.episodes - tenth + 1:,}–{params.episodes:,})",
        ],
        patch_artist=True,
        widths=0.4,
    )
    for patch, colour in zip(bp["boxes"], ["salmon", "lightgreen"]):
        patch.set_facecolor(colour)
    ax.set_ylabel("Total reward  (mean over runs)")
    ax.set_title(
        f"Reward distribution: start vs end of training\n"
        f"(first / last 1/{params.phase_frac} of {params.episodes:,} episodes)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "iql_reward_distribution.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Visualise agents BEFORE training (random policy) ──────────────── #
    visualise_random_policy(n_steps=100)

    # ── 2. Train ──────────────────────────────────────────────────────────── #
    all_rewards, all_steps, eps_hist, trained_agents = train()

    mean_r = all_rewards.mean(axis=0)
    tenth  = max(1, params.episodes // params.phase_frac)
    print(f"\nFirst {tenth} eps – avg reward : {mean_r[:tenth].mean():.4f}")
    print(f"Last  {tenth} eps – avg reward : {mean_r[-tenth:].mean():.4f}")

    # ── 3. Visualise agents AFTER training (greedy / learnt policy) ───────── #
    visualise_trained_policy(trained_agents, n_episodes=3)

    # ── 4. Plot training metrics ──────────────────────────────────────────── #
    print("\nPlotting results …")
    plot_epsilon_decay(eps_hist)
    plot_episodic_reward(all_rewards)
    plot_episodic_steps(all_steps)
    plot_reward_distribution(all_rewards)


