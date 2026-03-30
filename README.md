# 🤖 Reinforcement Learning & Multi-Agent RL — Lab

A self-contained, well-commented Python codebase for learning the fundamentals of **Reinforcement Learning (RL)** and **Multi-Agent Reinforcement Learning (MARL)**, progressing from a classic tabular Q-learning example to modern deep MARL with PPO.

---

## 📚 Overview

The repository is structured as a series of progressively more complex examples:

| Step | Topic | Script |
|------|-------|--------|
| 1 | **Tabular Q-learning** — single agent, discrete actions | `gymn_frozenlake.py` |
| 2 | **Decentralised MARL** (IPPO) — each agent learns independently | `pytorch_multi_decentralised.py` |
| 3 | **Centralised Training, Decentralised Execution** ✅ (MAPPO) | `pytorch_multi_centralised_correct.py` |
| 4 | **Wrong CTDE** ❌ — common mistake illustrated & explained | `pytorch_multi_centralised_wrong.py` |

### Key concepts covered

- **Markov Decision Processes (MDPs)**: states, actions, rewards, transitions
- **Q-learning**: tabular value-function learning, Bellman update
- **ε-greedy exploration** with exponential decay
- **Multi-agent environments**: cooperative navigation with continuous actions
- **IPPO** (Independent Proximal Policy Optimisation): fully decentralised agents
- **MAPPO** (Multi-Agent PPO): centralised critic, decentralised actors — the correct CTDE pattern
- **CTDE pitfalls**: what happens when the *actor* (not only the critic) is centralised during execution

---

## 🗂️ Repository Structure

```
.
├── gymn_frozenlake.py                  # Step 1 – Tabular Q-learning on FrozenLake (Gymnasium)
├── pytorch_multi_decentralised.py      # Step 2 – Fully decentralised MARL (IPPO, TorchRL + VMAS)
├── pytorch_multi_centralised_correct.py# Step 3 – CTDE done right: MAPPO (TorchRL + VMAS)
├── pytorch_multi_centralised_wrong.py  # Step 4 – CTDE done wrong: centralised actor (antipattern)
├── ctde_wrong.png                      # Diagram illustrating the wrong CTDE architecture
├── requirements.txt                    # Python dependencies
└── res/
    └── frozenlake/                     # Auto-generated plots from gymn_frozenlake.py
        ├── frozenlake_epsilon_decay_7x7.png
        ├── frozenlake_q_values_7x7.png
        ├── frozenlake_reward_distribution_7x7.png
        ├── frozenlake_state_visits_7x7.png
        └── frozenlake_steps_and_rewards.png
```

---

## 🧊 Step 1 — Tabular Q-learning on FrozenLake (`gymn_frozenlake.py`)

The agent must navigate a randomly generated frozen-lake grid from the **start tile** to the **goal tile** without falling into any holes.

### Algorithm: Q-learning

The Q-table is updated after every step using the Bellman equation:

```
Q(s,a) ← Q(s,a) + α · [R(s,a) + γ · max_a' Q(s',a') − Q(s,a)]
```

### Exploration: ε-greedy with exponential decay

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Initial exploration rate | `e_init` | `0.9` | Probability of taking a random action at the start of training |
| Minimum exploration rate | `e_min` | `0.05` | Floor value — exploration never drops below this |
| Decay factor | `e_decay` | *computed* | Multiplicative factor applied after each episode; chosen so that `e_min` is reached at **2/3 of total training** |

The decay factor is derived analytically:

```
e_min = e_init × e_decay^(⅔ × episodes)
  ⟹  e_decay = (e_min / e_init)^(3 / (2 × episodes))
```

### Visualisations produced

| Plot | Description |
|------|-------------|
| **Epsilon decay** | ε at each episode; marks `e_init`, `e_min`, and the 2/3 training milestone |
| **Q-values heatmap** | Best Q-value and greedy action (arrow) for every tile |
| **State visit frequency** | Heatmap (greens) of how often each tile was visited, normalised to relative frequency |
| **Episodic reward distribution** | Box plots comparing the first vs. last `1/phase_frac` of episodes — shows the agent shifting from failing to reaching the goal reliably |
| **Steps & success rate** | Rolling-mean lines for episodic reward and steps-to-goal; failed episodes are penalised with `map_size²` steps instead of the misleadingly small raw step count |

### Key hyper-parameters

```python
episodes          = 10_000   # total training episodes
lr                = 0.1      # Q-learning rate α
gamma             = 0.9      # discount factor γ
map_size          = 7        # 7×7 grid
is_slippery       = False    # deterministic transitions
rolling_window_frac = 50     # rolling window = episodes // 50
phase_frac          = 10     # first/last phase = episodes // 10
```

---

## 🤝 Steps 2–4 — Multi-Agent RL with PPO (TorchRL + VMAS)

All three MARL scripts share the same **Navigation** scenario from [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator): `n_agents = 3` agents must navigate to individual goals in a 2-D continuous world while avoiding collisions detected via LIDAR sensors.

The learning algorithm is **PPO** (Proximal Policy Optimisation) throughout, implemented with [TorchRL](https://github.com/pytorch/rl).

### Architecture comparison

| Feature | Decentralised (IPPO) | CTDE correct (MAPPO) | CTDE wrong ❌ |
|---------|---------------------|---------------------|--------------|
| Actor observes | own obs only | own obs only | **all agents' obs** ← wrong |
| Critic observes | own obs only | all agents' obs | all agents' obs |
| Parameter sharing (policy) | ✗ | ✓ | ✓ |
| Parameter sharing (critic) | ✗ | ✗ | **✓** ← wrong |
| Algorithm | IPPO | MAPPO | — |

### Why does the "wrong" CTDE matter?

**Centralised Training, Decentralised Execution (CTDE)** means that during *training* we may give agents extra information (e.g. global state), but at *execution* each agent must act using only its own local observation.

`pytorch_multi_centralised_wrong.py` (together with `ctde_wrong.png`) illustrates the most common CTDE mistake: making the **actor** centralised. A centralised actor cannot be deployed in a real decentralised setting — it would need access to other agents' observations at runtime, which is typically unavailable.

`pytorch_multi_centralised_correct.py` fixes this: only the **critic** is centralised (it sees all observations during training), while each **actor** still acts from its own local observation — exactly the MAPPO recipe.

---

## ⚙️ Setup

### Prerequisites

- Python ≥ 3.9
- (Optional) a CUDA-capable GPU — all scripts fall back to CPU automatically

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd 2023-dai-marl-lab

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the examples

```bash
# Step 1 – Q-learning on FrozenLake
python gymn_frozenlake.py

# Step 2 – Decentralised MARL (IPPO)
python pytorch_multi_decentralised.py

# Step 3 – CTDE correct (MAPPO)
python pytorch_multi_centralised_correct.py

# Step 4 – CTDE wrong (for illustration)
python pytorch_multi_centralised_wrong.py
```

> **Note:** The FrozenLake script opens a live Gymnasium render window before and after training (requires a display). The VMAS scripts render the Navigation environment using PyOpenGL.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `gymnasium` | Single-agent RL environment (FrozenLake) |
| `numpy`, `pandas` | Numerical computing and data wrangling |
| `matplotlib`, `seaborn` | Plotting and visualisation |
| `tqdm` | Progress bars |
| `torch` | Deep learning backend |
| `torchrl` | High-level RL library (PPO, collectors, replay buffers) |
| `vmas` | Vectorised Multi-Agent Simulator (Navigation scenario) |
| `PyOpenGL`, `PyOpenGL_accelerate` | VMAS rendering |
| `pygame` | Additional rendering support |

---

## 📄 License

This project is intended for educational purposes.

