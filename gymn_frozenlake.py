from pathlib import Path
from typing import NamedTuple
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm  # smart progress meter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()


class Params(NamedTuple):  # https://docs.python.org/3/library/collections.html#collections.namedtuple
    episodes: int  # Episode = "1 game run (e.g. until win or lose or maximum "play time" reached)
    lr: float  # Learning rate ("how fast" the agent adapts to changes---learns)
    gamma: float  # Discounting rate ("how much value" to give to future vs immediate rewards)
    e_init: float  # Initial exploration rate (e.g. do random action instead of arg_max(Q_values))
    e_min: float  # Minimum exploration rate (floor after exponential decay)
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    runs: int  # 1 experiment (run) is a series of episodes, we need more runs to get a good estimate of the performance (e.g. averaging random effects)
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    prob_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved
    rolling_window_frac: int  # Denominator for the rolling-mean window: window = episodes // rolling_window_frac
    phase_frac: int  # Denominator for the first/last phase slice in reward distribution: slice = episodes // phase_frac


params = Params(
    episodes=10_000,
    lr=0.1,  # learn "slowly" (=do not change Q-values too much at once)
    gamma=0.9,  # give value to future rewards ('cause we need many steps to reach the destination!)
    e_init=0.5,  # start exploring 90% of the time
    e_min=0.05,  # never go below 5% exploration
    map_size=7,
    seed=123,
    is_slippery=True,
    runs=3,
    action_size=None,
    state_size=None,
    prob_frozen=0.9,  # probability that a tile is NOT frozen (I know, it's confusing but it is not my fault :/)
    savefig_folder=Path("res/frozenlake/"),
    rolling_window_frac=50,  # rolling window = episodes // 50  (i.e. 2% of episodes)
    phase_frac=10,           # first/last phase = episodes // 10  (i.e. 1/10 of episodes)
)

rng = np.random.default_rng(params.seed)
params.savefig_folder.mkdir(parents=True, exist_ok=True)  # Create the figure folder if it doesn't exist


class QLearningAgent:
    def __init__(self,
                 lr,
                 gamma,
                 state_s,
                 action_s):
        self.lr = lr
        self.gamma = gamma
        self.state_s = state_s
        self.action_s = action_s
        self.qtable = self.init_qtable()

    def update(self, state, action, reward, next_state):
        """Q(s,a):= Q(s,a) + lr [r + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
                reward
                + self.gamma * np.max(self.qtable[next_state, :])
                - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.lr * delta
        return q_update

    def init_qtable(self):
        return np.zeros((self.state_s, self.action_s))


class EpsilonGreedy:
    """Exploration strategy with exponential epsilon decay.

    e_decay is computed so that at 2/3 of total training episodes the current
    exploration rate reaches e_min:
        e_min = e_init * e_decay ^ (2/3 * episodes)
        => e_decay = (e_min / e_init) ^ (3 / (2 * episodes))
    """
    def __init__(self, e_init, e_min, e_decay):
        self.e_init  = e_init   # initial exploration rate
        self.e_min   = e_min    # minimum exploration rate (floor)
        self.e_decay = e_decay  # multiplicative decay factor applied after each episode
        self.e = e_init         # current exploration rate

    def reset(self):
        """Reset to the initial exploration rate (call at the start of each run)."""
        self.e = self.e_init

    def decay(self):
        """Apply one exponential decay step; never go below e_min."""
        self.e = max(self.e_min, self.e * self.e_decay)

    def choose_action(self, action_s, state, qtable, test: bool=False):
        rnd = rng.uniform(0, 1)
        if rnd < self.e and not test:  # If we are testing we don't want to explore
            action = action_s.sample()  # random action (e.g. e=0.1 means 10% of the time)
        else:
            if np.all(qtable[state, :] == qtable[state, 0]):  # If all actions are the same for this state we choose a random one (otherwise `np.argmax()` would always take the first one)
                action = action_s.sample()
            else:
                action = np.argmax(qtable[state, :])  # "greedy" action (= the "best" as far as we currently know)
        return action


def run_env():
    """Run 1 experiment (= 1 whole series of episodes)"""
    # data tracking for measuring performance
    rewards = np.zeros((params.episodes, params.runs))
    steps = np.zeros((params.episodes, params.runs))
    episodes = np.arange(params.episodes)
    qtables = np.zeros((params.runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []
    e_history = []  # exploration rate used at the start of each episode (recorded from run 0)

    for run in range(params.runs):  # 1 run is one experiment (series of episodes)
        learner.init_qtable()
        explorer.reset()  # reset e to e_init at the start of every run

        for episode in tqdm(
            episodes, desc=f"Run {run + 1}/{params.runs} - Episodes", leave=False
        ):
            if run == 0:
                e_history.append(explorer.e)  # record e used for this episode

            state, _ = env.reset(seed=params.seed)
            step = 0
            done = False
            total_rewards = 0

            while not done:  # the Gymnasium environment itself tells us when it's done (episode ended)
                action = explorer.choose_action(
                    action_s=env.action_space, state=state, qtable=learner.qtable, test=False
                )

                all_states.append(state)
                all_actions.append(action)

                next_s, rew, term, trunc, _ = env.step(action)  # see Gymnasium API
                done = term or trunc  # either the episode ended or we reached the maximum number of steps

                learner.qtable[state, action] = learner.update(
                    state, action, rew, next_s
                )

                total_rewards += rew
                step += 1  # keep track of steps needed to reach the goal
                state = next_s
            rewards[episode, run] = total_rewards
            # For failed episodes (reward=0) the actual step count is misleadingly
            # small (e.g. 1–2 steps into a hole).  Use map_size² as a penalty value
            # so failures are clearly visible as a high step count in the plot.
            steps[episode, run] = step if total_rewards > 0 else params.state_size

            explorer.decay()  # apply epsilon decay once per episode, after it ends

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions, e_history

#############################################
# CODE BELOW IS JUST FOR NICE VISUALIZATION #
#############################################

def to_pandas(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in Pandas dataframes.

    Returns
    -------
    res : DataFrame
        One row per (episode, run) — used for the reward-distribution box plot.
        Episodes are aligned with np.repeat so that episode i pairs with all
        run values for that episode (fixes the np.tile misalignment for runs>1).
    st : DataFrame
        One row per episode — rewards and steps averaged over runs.
        Used for the rolling line-plots so the smoothing is applied to the
        already-averaged signal, not to the interleaved per-run rows.
    """
    # Per-run data: np.repeat gives [0,0,1,1,...] which aligns with the
    # C-order flatten [r[0,run0], r[0,run1], r[1,run0], r[1,run1], ...]
    res = pd.DataFrame(
        data={
            "Episodes": np.repeat(episodes, repeats=params.runs),
            "Rewards": rewards.flatten(),   # C-order matches np.repeat
            "Steps": steps.flatten()
        }
    )
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    # Per-episode averaged data: mean over the runs axis for both metrics
    st = pd.DataFrame(data={
        "Episodes": episodes,
        "Rewards": rewards.mean(axis=1),   # run-averaged episodic reward
        "Steps":   steps.mean(axis=1),     # run-averaged episodic steps
    })
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


def plot_steps_and_rewards(steps_df):
    """Plot rolling mean ± rolling std for the success rate and average steps.

    Both metrics come from *steps_df*, whose Rewards and Steps columns are
    already averaged over runs for each episode.  Using run-averaged values
    here is important: applying the rolling window to interleaved per-run
    rows would mix episodes from different runs at every window boundary.

    The shaded band is ± one rolling standard deviation computed over the
    same window as the mean, so it reflects how consistent the agent's
    performance is within that recent stretch of episodes.
    """
    window = max(1, params.episodes // params.rolling_window_frac)
    df = steps_df.copy()

    for col in ("Rewards", "Steps"):
        df[f"{col}_mean"] = df.groupby("map_size")[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f"{col}_std"] = df.groupby("map_size")[col].transform(
            # std is NaN for the very first point (only 1 sample); fill with 0
            # so fill_between draws no band there rather than leaving a gap.
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    for map_s, group in df.groupby("map_size"):
        # --- success rate ---
        (line,) = ax[0].plot(group["Episodes"], group["Rewards_mean"], label=map_s)
        ax[0].fill_between(
            group["Episodes"],
            group["Rewards_mean"] - group["Rewards_std"],
            group["Rewards_mean"] + group["Rewards_std"],
            alpha=0.25, color=line.get_color(),
        )
        # --- steps ---
        (line,) = ax[1].plot(group["Episodes"], group["Steps_mean"], label=map_s)
        ax[1].fill_between(
            group["Episodes"],
            group["Steps_mean"] - group["Steps_std"],
            group["Steps_mean"] + group["Steps_std"],
            alpha=0.25, color=line.get_color(),
        )

    ax[0].set(
        xlabel="Episodes",
        ylabel=f"Success rate\n(rolling mean ± std, window={window})",
    )
    ax[1].set(
        xlabel="Episodes",
        ylabel=f"Avg. steps\n(rolling mean ± std, window={window})\nfailures penalised at map_size²",
    )
    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    # extract the best Q-values from the Q-table for each state
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    # get the corresponding best action for those Q-values
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    # map each action to an arrow so we can visualize it
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:  # Assign an arrow only if a minimal Q-value has been learned as best action otherwise since 0 is a direction, it also gets mapped on the tiles where it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """With the following function, we’ll plot on the left the last frame of the simulation.
    If the agent learned a good policy to solve the task,
    we expect to see it on the tile of the treasure in the last frame of the video.
    On the right we’ll plot the policy the agent has learned.
    Each arrow will represent the best action to choose for each tile/state."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"←": 0, "↓": 1, "→": 2, "↑": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_reward_distribution(rewards_df, map_size):
    """Compare the episodic reward distribution between the first and last 1/10 of training
    using box plots. For binary 0/1 rewards the median shift from 0 to 1 directly shows
    that the agent has learned to reach the goal reliably."""
    tenth = params.episodes // params.phase_frac

    first = rewards_df[rewards_df["Episodes"] < tenth].copy()
    first["Phase"] = f"First 1/{params.phase_frac}\n(ep. 1–{tenth:,})"

    last = rewards_df[rewards_df["Episodes"] >= params.episodes - tenth].copy()
    last["Phase"] = f"Last 1/{params.phase_frac}\n(ep. {params.episodes - tenth + 1:,}–{params.episodes:,})"

    combined = pd.concat([first, last])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=combined, x="Phase", y="Rewards", ax=ax,
        palette=["salmon", "steelblue"], width=0.4,
        order=[first["Phase"].iloc[0], last["Phase"].iloc[0]],
    )
    ax.set_title(
        f"Episodic reward: first vs last 1/{params.phase_frac} of training\nmap {map_size}x{map_size}"
    )
    ax.set_ylabel("Episodic reward  (0 = fail, 1 = goal reached)")
    ax.set_xlabel("")
    ax.set_yticks([0, 1])
    fig.tight_layout()
    img_title = f"frozenlake_reward_distribution_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_visits_map(all_states, map_size):
    """Plot a heatmap of how often each tile (state) was visited during training,
    normalised by the total number of state visits."""
    state_counts = np.zeros(map_size * map_size)
    for s in all_states:
        state_counts[s] += 1
    total_visits = state_counts.sum()
    state_freqs = (state_counts / total_visits).reshape(map_size, map_size)

    # Annotate each cell with the normalised frequency as a percentage
    annot = np.array(
        [[f"{state_freqs[r, c]:.1%}" for c in range(map_size)] for r in range(map_size)]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        state_freqs,
        annot=annot,
        fmt="",
        ax=ax,
        cmap=sns.color_palette("Greens", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "large"},
    ).set(title="State visit frequency\n(normalised over total visits)")
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    fig.tight_layout()
    img_title = f"frozenlake_state_visits_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_epsilon_decay(e_history, map_size):
    """Plot the exploration rate (epsilon) used at each episode during training."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(e_history, color="steelblue", linewidth=1.5, label="ε (current)")
    ax.axhline(y=params.e_min, color="red", linestyle="--", linewidth=1,
               label=f"e_min = {params.e_min}")
    ax.axhline(y=params.e_init, color="green", linestyle="--", linewidth=1,
               label=f"e_init = {params.e_init}")
    ax.axvline(x=int(2 / 3 * params.episodes), color="orange", linestyle=":",
               linewidth=1, label=f"2/3 of training (ep. {int(2 / 3 * params.episodes):,})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration rate (ε)")
    ax.set_title(f"Epsilon decay over training — map {map_size}x{map_size}")
    ax.legend()
    fig.tight_layout()
    img_title = f"frozenlake_epsilon_decay_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


#############################
# MAIN FUNCTION TO RUN CODE #
#############################

if __name__ == "__main__":

    map_sizes = [7]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    current_map = None
    for map_size in map_sizes:
        current_map = generate_random_map(
                size=map_size, p=params.prob_frozen, seed=params.seed
            )
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            #render_mode="rgb_array",
            render_mode="human",  # to see the AI playing!
            desc=current_map
        )
        params = params._replace(action_size=env.action_space.n)
        params = params._replace(state_size=env.observation_space.n)
        env.action_space.seed(
            params.seed
        )

        learner = QLearningAgent(
            lr=params.lr,
            gamma=params.gamma,
            state_s=params.state_size,
            action_s=params.action_size,
        )
        # e_decay so that at 2/3 of training episodes, e reaches e_min:
        #   e_min = e_init * e_decay^(2/3 * episodes)  =>  e_decay = (e_min/e_init)^(3/(2*episodes))
        e_decay = (params.e_min / params.e_init) ** (3 / (2 * params.episodes))
        explorer = EpsilonGreedy(
            e_init=params.e_init,
            e_min=params.e_min,
            e_decay=e_decay,
        )

        print("Let the agent play before training to see initial behavior")
        for ep in tqdm(
                range(1), desc=f"Episodes", leave=False
        ):
            state, _ = env.reset(seed=params.seed)
            done = False
            while not done:
                action = explorer.choose_action(
                    action_s=env.action_space, state=state, qtable=learner.qtable, test=False
                )
                next_s, _, term, trunc, _ = env.step(action)
                done = term or trunc
                state = next_s

        env.close()

        env = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            render_mode="rgb_array",
            #render_mode="human",  # to see the AI playing!
            desc=current_map
        )
        env.action_space.seed(
            params.seed
        )

        print("Now training the agent...")
        print(f"\nMap size: {map_size}x{map_size}")
        rewards, steps, episodes, qtables, all_states, all_actions, e_history = run_env()  # multiple episodes, as per config

        # Save the results in dataframes
        res, st = to_pandas(episodes, params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])
        qtable = qtables.mean(axis=0)  # Average the Q-table between runs

        #plot_states_actions_distribution(
        #    states=all_states, actions=all_actions, map_size=map_size
        #)  # Sanity check

        print("Plotting epsilon decay...")
        plot_epsilon_decay(e_history, map_size)

        print("Plotting state visit frequencies...")
        plot_states_visits_map(all_states, map_size)

        print("Plotting steps and rewards...")
        plot_steps_and_rewards(st_all)

        print("Plotting reward distribution (early vs late stages of training)...")
        plot_reward_distribution(res, map_size)

        print(f"Plotting learned Q-values (averaged over {params.runs})...")
        plot_q_values_map(qtable, env, map_size)

        env.close()

    ################################
    #### SEE WHAT AGENT LEARNT #####
    ################################

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        #render_mode="rgb_array",
        render_mode="human",  # to see the AI playing!
        desc=current_map
    )
    env.action_space.seed(
        params.seed
    )

    print("Let the trained agent play episodes to see learned behavior")
    for ep in tqdm(
            range(10), desc=f"Episodes", leave=False
        ):
        state, _ = env.reset(seed=params.seed)
        done = False
        while not done:
            action = explorer.choose_action(
                action_s=env.action_space, state=state, qtable=learner.qtable, test=True
            )
            next_s, _, term, trunc, _ = env.step(action)
            done = term or trunc
            state = next_s

    env.close()
