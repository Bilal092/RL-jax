# RL-jax

Self-contained Jupyter notebooks that implement deep reinforcement learning, imitation learning, and Schrödinger-bridge algorithms **from scratch in JAX**.

Every notebook follows the same pattern:

1. **Derive** the algorithm (the losses, targets, and estimators are written out in LaTeX).
2. **Implement** it in pure JAX/Flax/Optax with JIT-compiled, vectorized rollouts.
3. **Test** the tricky parts (GAE recursions, terminal masking, replay sampling, importance weights) with small unit tests inside the notebook.
4. **Train, evaluate, and ablate**, with paired evaluation, confidence intervals, and optional multi-seed runs.

The aim is code that stays mathematically transparent: each piece of math maps to a short, readable function.

---

## Contents

- [Notebooks at a glance](#notebooks-at-a-glance)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running the notebooks](#running-the-notebooks)
- [Notebook details](#notebook-details)
  - [01 — Flow-matching imitation learning on Pendulum](#01--flow-matching-imitation-learning-on-pendulum)
  - [02 — REINFORCE](#02--reinforce)
  - [03 — Actor–critic and GAE](#03--actorcritic-and-gae)
  - [04 — PPO](#04--ppo)
  - [05 — DQN](#05--dqn)
  - [06 — Double, Dueling, and Prioritized DQN](#06--double-dueling-and-prioritized-dqn)
  - [Diffusion Schrödinger Bridge](#diffusion-schrödinger-bridge-de-bortoli-et-al-2021)
- [Shared design conventions](#shared-design-conventions)
- [Results snapshot](#results-snapshot)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Notebooks at a glance

| # | Notebook | Environment | Family | Key ideas |
|---|----------|-------------|--------|-----------|
| 01 | [`01_pendulum_flow_imitation.ipynb`](01_pendulum_flow_imitation.ipynb) | Pendulum-v1 (continuous) | Imitation learning | PPO expert → behavioral cloning → conditional flow matching → action chunks, sampled with Diffrax |
| 02 | [`02_cartpole_reinforce_jax.ipynb`](02_cartpole_reinforce_jax.ipynb) | CartPole-v1 | Policy gradient | Four REINFORCE estimators, baselines, gradient-variance diagnostics |
| 03 | [`03_cartpole_actor_critic_gae_jax.ipynb`](03_cartpole_actor_critic_gae_jax.ipynb) | CartPole-v1 | Actor–critic | Bootstrapped critic, TD(0) vs. GAE(λ), bias–variance trade-off |
| 04 | [`04_cartpole_ppo_jax.ipynb`](04_cartpole_ppo_jax.ipynb) | CartPole-v1 | Actor–critic | Clipped PPO, minibatch epochs, value clipping, approximate KL |
| 05 | [`05_cartpole_dqn_jax.ipynb`](05_cartpole_dqn_jax.ipynb) | CartPole-v1 | Value-based | DQN, replay buffer, target network, ε-greedy |
| 06 | [`06_double_dueling_per_dqn_jax.ipynb`](06_double_dueling_per_dqn_jax.ipynb) | CartPole-v1 | Value-based | Double DQN, dueling heads, prioritized replay, factorial ablation |
| — | [`de_bortoli_dsb_jax.ipynb`](de_bortoli_dsb_jax.ipynb) | 2-D point clouds | Generative modeling | Diffusion Schrödinger Bridge via IPF and drift matching |
| — | [`schrodinger_bridges.ipynb`](schrodinger_bridges.ipynb) | — | — | Placeholder (work in progress) |

**Suggested reading order:** 02 → 03 → 04 → 05 → 06 for the core RL track, where each notebook explicitly builds on the previous one. Notebook 01 and the DSB notebook stand alone and connect RL to generative modeling.

---

## Repository layout

```
RL-jax/
├── 01_pendulum_flow_imitation.ipynb      # Flow-matching imitation learning (Pendulum)
├── 02_cartpole_reinforce_jax.ipynb       # REINFORCE variants
├── 03_cartpole_actor_critic_gae_jax.ipynb# Actor–critic + GAE
├── 04_cartpole_ppo_jax.ipynb             # PPO
├── 05_cartpole_dqn_jax.ipynb             # DQN
├── 06_double_dueling_per_dqn_jax.ipynb   # Double / Dueling / PER DQN
├── de_bortoli_dsb_jax.ipynb              # Diffusion Schrödinger Bridge
├── schrodinger_bridges.ipynb             # Placeholder
└── gymnax-blines/                        # External repo: pretrained PPO checkpoints (used by notebook 01)
```

---

## Installation

Python 3.10+ is recommended. All notebooks were developed against JAX 0.11.x with Flax linen.

```bash
git clone <this-repo-url> RL-jax
cd RL-jax

python -m venv .venv
source .venv/bin/activate

pip install -U "jax>=0.4.30" "flax>=0.10" "optax>=0.2.3" "gymnax>=0.0.8" \
               diffrax dotmap matplotlib pandas numpy jupyterlab
```

**GPU:** install the CUDA build of JAX instead of the CPU build:

```bash
pip install -U "jax[cuda12]"
```

Check which device JAX is using:

```bash
python -c "import jax; print(jax.__version__, jax.devices())"
```

### Per-notebook dependencies

| Package | 01 | 02–06 | DSB |
|---------|:--:|:-----:|:---:|
| `jax`, `flax`, `optax` | ✓ | ✓ | ✓ |
| `gymnax` | ✓ | ✓ | |
| `diffrax` | ✓ | | |
| `dotmap` (to unpickle the checkpoint) | ✓ | | |
| `matplotlib`, `numpy` | ✓ | ✓ | ✓ |
| `pandas` | | ✓ | |

Each notebook also has a commented-out `%pip install` cell at the top.

### Pretrained expert for notebook 01

Notebook 01 imitates a pretrained PPO agent from [`gymnax-blines`](https://github.com/RobertTLange/gymnax-blines) and loads it from:

```
./gymnax-blines/agents/Pendulum-v1/ppo.pkl
```

`gymnax-blines` is committed as a nested Git repository but has no `.gitmodules` entry, so **a fresh clone does not include it**. Fetch it manually from the repository root:

```bash
git clone https://github.com/RobertTLange/gymnax-blines.git
```

To use a checkpoint stored somewhere else, edit `GYMNAX_BLINES` in section 6 of the notebook.

---

## Running the notebooks

```bash
jupyter lab
```

Open a notebook and choose **Run → Run All Cells**. Every notebook is designed to run top to bottom.

- **The defaults are CPU-friendly.** The default configuration in each notebook trains in minutes on CPU, and faster on GPU.
- **Expensive experiments are opt-in.** Multi-seed runs, sweeps, and ablations are controlled by boolean flags in the relevant cells:

| Notebook | Flag | Default | What it enables |
|----------|------|:-------:|-----------------|
| 02 | `RUN_MULTISEED` | `False` | Repeat all four estimators over seeds 0–4 |
| 03 | `RUN_MULTISEED` | `False` | Repeat the λ comparison over seeds 0–4 |
| 04 | `RUN_ABLATION` | `True` | PPO vs. unclipped surrogate |
| 04 | `RUN_CLIP_SWEEP` | `False` | Sweep the clip threshold ε |
| 04 | `RUN_MULTISEED` | `False` | Multi-seed PPO evaluation |
| 05 | `RUN_ABLATIONS` | `False` | Replay vs. no replay; target-sync frequency |
| 05 | `RUN_MULTISEED` | `False` | Seeds 0–4 |
| 06 | `RUN_FACTORIAL` | `False` | Full 2×2×2 grid of {Double, Dueling, PER} |
| 06 | `RUN_MULTISEED` | `False` | Seeds 0–4 |
| DSB | `PAPER_SETTINGS` | `True` | The paper's full hyperparameters (slow); set to `False` for a quick development run |

Hyperparameters live in a frozen `Config` dataclass near the top of each notebook (notebooks 02–06). To create variants, use `dataclasses.replace(CFG, ...)`.

---

## Notebook details

### 01 — Flow-matching imitation learning on Pendulum

**Goal:** learn a continuous-control policy purely from expert demonstrations. The policies progress from deterministic regression, to a generative single-action policy, to a generative policy over *action trajectories*.

The environment's observation is $s_t = (\cos\theta_t, \sin\theta_t, \dot\theta_t)$ and its action is a scalar torque.

**Structure**

| Part | Sections | Content |
|------|----------|---------|
| Setup | 0–5 | Create the environment and inspect transitions. Write a readable Python-loop rollout, then a `lax.scan` + `vmap` version, and benchmark the two (~600× faster after JIT). |
| Expert | 6–12 | Load the gymnax-blines PPO checkpoint, rebuild its Flax network, check that the parameters match, compare the mean and sampled policies, collect demonstrations, and split the data into train/validation sets with observation normalization. |
| I — BC | 13–16 | Deterministic behavioral cloning (MSE regression), evaluated in closed loop. |
| II — Flow matching | 17–21 | Conditional flow matching $v_\theta(x_\tau, \tau, s)$ over one action. Actions are sampled by integrating the learned vector field with **Diffrax**. |
| III — Action chunks | 22–25, 27 | Generate $H$-step action chunks $[a_t,\dots,a_{t+H-1}]$ and execute them with receding-horizon control: replan every $K$ steps. |
| IV — Comparison | 26 | Training curves and a sampled expert-action MSE validation metric. |
| V — $(H, K)$ study | 28–38 | Train a separate flow model for each $H \in \{1,\dots,16\}$, evaluate every valid $(H, K)$ pair, and plot return histograms and a summary table. |

**Key configuration:** `CHUNK_H = 8`, `REPLAN_K = 1`. The sweep uses `SWEEP_EPOCHS = 50`, `SWEEP_BATCH_SIZE = 256`, `SWEEP_LR = 3e-4`, `SWEEP_EVAL_EPISODES = 100`, `SWEEP_MAX_STEPS = 200`.

**Ideas for follow-up experiments** (listed at the end of the notebook): compare Euler, Heun, and Tsit5 solvers; vary the number of demonstrations; add Gaussian BC; add DAgger; move to a multi-dimensional task such as Reacher.

---

### 02 — REINFORCE

**Goal:** implement and compare four Monte Carlo policy-gradient estimators for the discounted objective $J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t]$.

| Method key | Estimator |
|------------|-----------|
| `full` | $\sum_t \nabla\log\pi(a_t\mid s_t)\,R_\gamma(\tau)$ (whole-trajectory return) |
| `rtg` | Causal reward-to-go $\sum_t \gamma^t \nabla\log\pi(a_t\mid s_t)\,G_t$ |
| `constant` | Reward-to-go minus a leave-one-trajectory-out mean baseline |
| `value` | Reward-to-go minus a learned $V_\phi(s_t)$, regressed onto Monte Carlo returns |

The learned value baseline is **not** bootstrapped, so this notebook is still pure REINFORCE rather than actor–critic. Episodes run in a fixed-horizon `lax.scan`, and steps after termination are masked out.

**Sections:** derivation · config/networks · batched episodic rollout · estimators and updates · training · learning curves · paired evaluation · gradient-noise diagnostic · multi-seed (optional) · conclusions.

**Config:** `gamma=0.99`, `batch_episodes=16`, `updates=350`, `max_steps=500`, `policy_lr=1e-3`, `value_lr=2e-3`, `value_epochs=4`, `hidden=64`, `eval_episodes=100`.

---

### 03 — Actor–critic and GAE

**Goal:** bridge from REINFORCE to PPO with an on-policy actor–critic whose critic **bootstraps**, and study how the GAE parameter λ affects learning.

- TD error: $\delta_t = r_t + \gamma(1-d_t)V_\phi(s_{t+1}) - V_\phi(s_t)$
- GAE recursion: $\hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1}$
- Critic target: the λ-return $\operatorname{stopgrad}(\hat A_t + V_\phi(s_t))$
- Actor loss: policy gradient plus an entropy bonus. Updates are **unclipped**; clipping is introduced in notebook 04.

The notebook compares **λ ∈ {0, 0.5, 0.95, 1.0}**. Rollouts use a continuing stream of auto-resetting environments. A segment that is truncated while still running bootstraps from its last state; a true termination never bootstraps.

**Sections:** derivation · config · auto-reset rollouts · GAE plus math checks · actor–critic update · training · evaluation with common random keys · λ comparison · final evaluation · advantage-variance diagnostic · multi-seed (optional) · interpretation and limitations.

**Config:** `num_envs=32`, `rollout_steps=64`, `updates=500`, `gamma=0.99`, `gae_lambda=0.95`, `lr=3e-4`, `entropy_coef=0.01`, `value_coef=0.5`, `max_grad_norm=0.5`.

---

### 04 — PPO

**Goal:** implement clipped PPO from scratch, getting every detail right.

$$
L^{\text{CLIP}}(\theta)=\mathbb E\big[\min(r_t\hat A_t,\ \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)\big],\qquad
\mathcal L=-L^{\text{CLIP}}+c_vL_V-c_e\mathcal H
$$

Implemented components:

- Vectorized collection that stores the **old** log-probabilities and values, plus the value of the successor state *before* reset.
- GAE that never crosses episode terminations, with targets held fixed across PPO epochs, plus unit tests.
- Shuffled minibatches over several epochs, optional value clipping, an entropy bonus, and global gradient clipping.
- Diagnostics: approximate KL `mean((ratio-1) - logratio)`, clip fraction, and early stopping at `target_kl`.

**Ablation:** clipped PPO vs. an *unclipped surrogate* $\mathbb E[r_t\hat A_t]$, run with the **same** multi-epoch optimization. This isolates the effect of clipping. An optional sweep varies ε.

**Config:** `num_envs=32`, `rollout_steps=64`, `updates=300`, `ppo_epochs=4`, `num_minibatches=4`, `clip_eps=0.2`, `clip_value_loss=True`, `learning_rate=3e-4`, `gae_lambda=0.95`, `target_kl=0.03`.

---

### 05 — DQN

**Goal:** a faithful **standard** DQN (Double DQN is deliberately left for notebook 06).

- Target: $y = r + \gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')$ with a Huber loss.
- Fixed-capacity circular replay buffer stored as device arrays, sampled uniformly from the filled portion.
- A frozen target network, hard-copied every `target_update_frequency` steps.
- Linearly decayed ε-greedy exploration.

**Ablations** (`RUN_ABLATIONS`): replay vs. no replay (`use_replay=False` trains on the latest transition only), and different target-sync frequencies. The evaluation reports both greedy and ε-greedy returns.

**Config:** `total_steps=120_000`, `buffer_capacity=30_000`, `batch_size=64`, `learning_starts=1_000`, `train_frequency=4`, `target_update_frequency=500`, ε from 1.0 to 0.05 over 35k steps, `lr=1e-3`, `max_grad_norm=10`.

---

### 06 — Double, Dueling, and Prioritized DQN

**Goal:** extend notebook 05 with three improvements that can each be switched on or off independently.

| Switch | Idea |
|--------|------|
| `double` | The online network selects $\arg\max_{a'}$ and the target network evaluates it, which reduces overestimation. |
| `dueling` | $Q(s,a) = V(s) + A(s,a) - \frac{1}{\lvert\mathcal A\rvert}\sum_b A(s,b)$ |
| `prioritized` | Sample transitions with $P(i) \propto (\lvert\delta_i\rvert+\epsilon)^\alpha$ and correct with importance weights $w_i \propto (N P(i))^{-\beta}$, annealing β → 1 |

The notebook includes unit tests for replay and importance weights, for Double DQN and terminal targets, and an end-to-end smoke test. By default it trains a **baseline** (all switches off) and an **improved** agent (all on). It then compares them in a paired evaluation with confidence intervals. `RUN_FACTORIAL` runs all 8 combinations.

**Config:** `total_steps=100_000`, `buffer_capacity=20_000`, `per_alpha=0.6`, `per_beta_start=0.4`, `per_beta_steps=100_000`, `per_eps=1e-5`; other settings match notebook 05.

> **Caveat (from the notebook):** with β = 1, importance weights correct PER's *sampling* distribution in expectation, but not the broader off-policy bootstrapping bias.

---

### Diffusion Schrödinger Bridge (De Bortoli et al., 2021)

**Goal:** reproduce the DSB algorithm from *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling* (NeurIPS 2021) on 2-D data. The bridge maps an **8-mode Gaussian ring** to **two moons**.

- Reference process: Brownian motion $dX_t = \sqrt{2}\,dW_t$, discretized with step size γ.
- Alternating **IPF** half-steps that learn the forward drift $f^n$ and the backward drift $b^n$.
- The **drift-matching** regression from Appendix E.2.2, e.g.
  $b^n_{k+1} = \arg\min_b \mathbb E\big\|b(X_{k+1}) - \tfrac{F^n_k(X_k) - F^n_k(X_{k+1})}{\gamma}\big\|^2$ with $F^n_k(x) = x + \gamma f^n_k(x)$,
  in place of a naive local-velocity regression.
- The 2-D architecture from Appendix J.1: separate state and sinusoidal-time branches (2→16→32 and 1→16→32), concatenated and passed through an MLP 64→128→128→2 that predicts the drift.
- A trajectory cache that is refreshed periodically during training.

**Paper settings** (`PAPER_SETTINGS=True`): Adam with learning rate 1e-4, batch 512, γ = 1e-2, 50 diffusion steps, cache size 1e4 refreshed every 1e3 steps, 1e4 steps per half-step, and `N_IPF=100`. Set the flag to `False` for a much faster development run.

**Outputs:** training diagnostics, learned bridge marginals over time, comparisons at the endpoints, and individual stochastic bridge trajectories. Section 14 lists which parts are faithful to the paper and which are simplified.

---

## Shared design conventions

- **JAX-native environment loops.** Rollouts use `jax.lax.scan` over time and `jax.vmap` over environments or episodes, and are JIT-compiled end to end. Gymnax environments run on the accelerator, so there are no Python-level step loops on the training path.
- **Explicit randomness.** A single PRNG key is threaded through each notebook and split wherever randomness is needed, so every run is reproducible from `seed`.
- **Termination vs. truncation.** GAE and TD targets never bootstrap across a true terminal transition. Rollout segments that end mid-episode *do* bootstrap. Gymnax's CartPole time limit is treated as terminal in these episodic experiments, and each notebook states this explicitly.
- **Fixed targets.** Advantages, value targets, and old log-probabilities are computed once per batch and wrapped in `stop_gradient`.
- **Careful evaluation.** Evaluation is kept separate from training. Both **sampled** and **greedy** policies are reported, with means, medians, standard deviations, and bootstrap confidence intervals. Paired comparisons use common random keys.
- **Tests inside the notebooks.** Small `assert`-based checks cover the parts that are easy to get wrong.

---

## Results snapshot

These are the results stored in the committed notebook outputs. They are **single-seed (seed 0) runs** with default configurations, so they illustrate typical behavior rather than serve as benchmarks. Use the multi-seed flags for proper comparisons. CartPole-v1 has a maximum return of 500.

**02 — REINFORCE** (100 evaluation episodes)

| Estimator | Sampled return | Greedy return | Gradient trace variance |
|-----------|---------------:|--------------:|------------------------:|
| full | 282.6 | 362.4 | 2.7 × 10⁶ |
| rtg | 429.2 | 500.0 | 3.8 × 10⁵ |
| constant | 500.0 | 500.0 | 6.3 × 10¹ |
| value | 500.0 | 500.0 | 1.2 × 10⁴ |

**03 — Actor–critic, λ comparison** (greedy evaluation)

| λ | Greedy return | Gradient covariance trace |
|---|--------------:|--------------------------:|
| 0.0 | 21.2 | 0.045 |
| 0.5 | 89.2 | 0.065 |
| 0.95 | 219.0 | 0.367 |
| 1.0 | 246.2 | 0.642 |

These results show the bias–variance trade-off: larger λ gives a less-biased advantage estimate but noisier gradients.

**04 — PPO:** 500.0 ± 0.0 (sampled and greedy). In the ablation, clipped PPO scored 498.5 and the unclipped surrogate scored 496.1 (sampled).

**05 — DQN:** greedy 90.8, ε-greedy 67.5 after 120k steps. Standard DQN on CartPole is notoriously unstable, which motivates notebook 06.

**01 — Pendulum imitation** (episode return; higher is better)

| Policy | Return |
|--------|-------:|
| Random | −1248.4 ± 321.2 |
| PPO expert (mean action) | −464.8 ± 464.4 |
| PPO expert (sampled, used as demonstrator) | −253.2 ± 208.9 |
| Behavioral cloning | −211.9 ± 193.1 |
| One-step flow matching | −308.3 ± 236.9 |
| Chunked flow matching (H = 8, K = 1) | −272.4 ± 166.8 |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: ./gymnax-blines/agents/Pendulum-v1/ppo.pkl` | Clone `gymnax-blines` into the repo root (see [Installation](#pretrained-expert-for-notebook-01)). |
| `ModuleNotFoundError: dotmap` when loading the checkpoint | `pip install dotmap`. The pickle stores a `DotMap` config. |
| The first call to a training function is slow | That is JIT compilation. Later calls are fast. Benchmarks warm up first. |
| JAX only sees the CPU | Install `jax[cuda12]` and check with `jax.devices()`. |
| Gymnax `step` returns 4 vs. 5 values | Notebook 01 includes a helper that handles both the old and new Gymnax step APIs. |
| Results differ from the snapshot | Seeds, JAX versions, and hardware (CPU vs. GPU) change the numerics. Use the multi-seed flags. |

---

## References

**Reinforcement learning**
- R. J. Williams. *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 1992.
- J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* 2017.
- V. Mnih et al. *Human-level control through deep reinforcement learning.* Nature, 2015.
- H. van Hasselt, A. Guez, D. Silver. *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016.
- Z. Wang et al. *Dueling Network Architectures for Deep Reinforcement Learning.* ICML 2016.
- T. Schaul, J. Quan, I. Antonoglou, D. Silver. *Prioritized Experience Replay.* ICLR 2016.

**Generative modeling and imitation**
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR 2023.
- V. De Bortoli, J. Thornton, J. Heng, A. Doucet. *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling.* NeurIPS 2021.

**Software**
- R. T. Lange. *gymnax: A JAX-based Reinforcement Learning Environment Library.* 2022. · [gymnax-blines](https://github.com/RobertTLange/gymnax-blines)
- [JAX](https://github.com/jax-ml/jax) · [Flax](https://github.com/google/flax) · [Optax](https://github.com/google-deepmind/optax) · [Diffrax](https://github.com/patrick-kidger/diffrax)
