# RPS_RL

Cheap testbed for the A-parameter hypothesis using Rock-Paper-Scissors.

## The A Parameter

A = probability of sampling an opponent from the historical zoo (vs. playing the latest opponent):
- **A=0**: Self-play — always play latest opponent, no zoo. Use `train_selfplay.py`.
- **A in (0, 1)**: Mix of latest opponent + zoo sampling.
- **A near 1**: Almost always sample from zoo (approaches SSP).

A=0 is self-play. A >= 1 is invalid. Arms race is a separate concept (sequential iteration with discarding), not a value of A.

## Hypothesis

A* (optimal zoo sampling ratio for convergence to Nash equilibrium) is inversely proportional to the algorithm's effective memory capacity. PPO (memoryless) should need more zoo sampling (higher A*) than algorithms with replay buffers.

## The Game

Rock-Paper-Scissors is a zero-sum game with a known **Nash equilibrium**: play each action with probability 1/3.

```
Payoff matrix (row player):
         Rock  Paper  Scissors
Rock     [ 0    -1      +1   ]
Paper    [+1     0      -1   ]
Scissors [-1    +1       0   ]
```

Any deviation from (1/3, 1/3, 1/3) can be exploited. **Exploitability** measures how far a strategy is from Nash — it's the best-response payoff an omniscient opponent could achieve:

```
exploitability = max(p_S - p_P, p_R - p_S, p_P - p_R)
```

At Nash, exploitability = 0. At a pure strategy, exploitability = 1.

## Results

We ran three experiment sets: a standard sweep (150 experiments, 200k timesteps, 10 seeds), an aggressive hyperparameter sweep, and an entropy/hyperparameter sweep (252 experiments, 500k timesteps, 3 seeds).

### The Problem: Self-Play Cycles

Without a zoo, PPO agents oscillate through strategies and never converge to Nash. Each agent over-adapts to the other's current strategy, creating Rock → Paper → Scissors cycling.

**Self-play (A=0)** — strategies wander far from Nash:

![Self-play cycling](experiments/results/selfplay_simplex.gif)

Each point in the simplex represents a mixed strategy over (Rock, Paper, Scissors). The center (+) is Nash equilibrium (1/3, 1/3, 1/3). Solid lines/circles are the agent; dashed lines/diamonds are the opponent. Colors go from light (early) to dark (late).

With aggressive hyperparameters (no entropy regularization, high learning rate, small network, no PPO clipping), the cycling becomes dramatic:

```bash
python train_selfplay.py --entropy-coef 0.0 --lr 0.05 --hidden 4 --clip-ratio 100.0 --train-iters 5 --seed 5 --timesteps 500000
```

![Aggressive self-play cycling](experiments/results/aggressive_selfplay_fixed.gif)

### The Solution: Zoo Sampling

Zoo sampling mixes in historical opponents, preventing the co-evolutionary spiral. At 200k timesteps with standard hyperparameters, more zoo = lower exploitability:

![Exploitability vs A curve](experiments/results/a_curve.png)

**PPO (memoryless, on-policy) — 200k timesteps, 10 seeds:**

| Condition | Exploitability (mean +/- std) | Entropy |
|-----------|-------------------------------|---------|
| Self-play (A=0) | 0.0724 +/- 0.0370 | 1.0905 |
| A=0.05 | 0.0380 +/- 0.0176 | 1.0957 |
| A=0.10 | 0.0374 +/- 0.0171 | 1.0958 |
| A=0.20 | 0.0355 +/- 0.0167 | 1.0960 |
| A=0.30 | 0.0321 +/- 0.0155 | 1.0965 |
| A=0.50 | 0.0258 +/- 0.0125 | 1.0972 |
| A=0.70 | 0.0170 +/- 0.0093 | 1.0980 |
| A=0.90 | 0.0075 +/- 0.0033 | 1.0985 |

**Buffered (replay buffer, off-policy) — 200k timesteps, 10 seeds:**

| Condition | Exploitability (mean +/- std) | Entropy |
|-----------|-------------------------------|---------|
| A=0.05 | 0.0380 +/- 0.0178 | 1.0957 |
| A=0.10 | 0.0372 +/- 0.0172 | 1.0957 |
| A=0.20 | 0.0366 +/- 0.0157 | 1.0958 |
| A=0.30 | 0.0338 +/- 0.0155 | 1.0962 |
| A=0.50 | 0.0317 +/- 0.0152 | 1.0964 |
| A=0.70 | 0.0286 +/- 0.0145 | 1.0968 |
| A=0.90 | 0.0236 +/- 0.0121 | 1.0973 |

PPO's A curve drops more steeply than Buffered's — PPO reaches 0.0075 at A=0.9 while Buffered sits at 0.0236. This matches the hypothesis that memoryless algorithms are more sensitive to zoo sampling.

**PPO Zoo A=0.10** — some wandering, but pulled back toward Nash:

![PPO Zoo A=0.10](experiments/results/A0.10_simplex.gif)

**PPO Zoo A=0.90** — converges tightly to Nash:

![PPO Zoo A=0.90](experiments/results/A0.90_simplex.gif)

**Simplex comparisons** across all A values and seeds:

![PPO simplex comparison](experiments/results/simplex_comparison.png)

![Buffered simplex comparison](experiments/results/simplex_comparison_buffered.png)

### But Heavy Zoo Degrades Over Time

The 200k results tell an incomplete story. At 500k timesteps, the A curve **inverts** — agents trained with heavy zoo sampling have degraded significantly:

![Training length effect](experiments/results/training_length_effect.png)

| A | Exploitability at 200k | Exploitability at 500k |
|---|----------------------|----------------------|
| 0.0 (self-play) | 0.0724 | 0.0753 |
| 0.05 | 0.0380 | 0.0556 |
| 0.10 | 0.0374 | 0.0590 |
| 0.50 | 0.0258 | 0.1349 |
| 0.90 | 0.0075 | 0.1054 |

A=0.9 produces the lowest exploitability at 200k (0.0075) but rises 14x to 0.105 by 500k. Meanwhile, A=0.05 slowly improves from 0.038 to 0.056.

**Interpretation:** Heavy zoo sampling accelerates early convergence by providing diverse opponents. But after the agent nears Nash, the continued diversity becomes destabilizing — the agent chases each zoo opponent's particular weakness, slowly pulling away from the uniform equilibrium. Light zoo sampling (A=0.05) is slower but more stable long-term. This suggests A should be **annealed during training** — high early, low late.

### Aggressive Hyperparameters Invert the A Curve

With aggressive hyperparameters (ent=0.0, lr=0.05, hidden=4, clip=100), the A curve inverts completely — more zoo = worse, even at short timescales:

![Aggressive vs Standard A curve](experiments/results/aggressive_a_curve.png)

| Condition | Exploitability (aggressive) | Exploitability (standard) |
|-----------|----------------------------|--------------------------|
| Self-play (A=0) | 0.3170 | 0.0724 +/- 0.0370 |
| A=0.05 | 0.2081 +/- 0.0626 | 0.0380 +/- 0.0176 |
| A=0.1 | 0.2412 +/- 0.0575 | 0.0374 +/- 0.0171 |
| A=0.3 | 0.3002 +/- 0.0817 | 0.0321 +/- 0.0155 |
| A=0.5 | 0.3467 +/- 0.0226 | 0.0258 +/- 0.0125 |
| A=0.9 | 0.9258 +/- 0.0744 | 0.0075 +/- 0.0033 |

The agent over-fits to beating specific zoo members instead of generalizing toward Nash. The zoo's diversity becomes a liability — each historical opponent pulls the agent toward a different counter-strategy, and the agent collapses to whichever pure strategy beats the most recent zoo sample.

**Aggressive A=0.05** — slight improvement over self-play, but still noisy:

![Aggressive A=0.05](experiments/results/aggressive_fixed_A0.05.gif)

**Aggressive A=0.1** — mild zoo, cycling with moderate amplitude:

![Aggressive A=0.1](experiments/results/aggressive_fixed_A0.1.gif)

**Aggressive A=0.3** — more zoo pulls the agent further off Nash:

![Aggressive A=0.3](experiments/results/aggressive_fixed_A0.3.gif)

**Aggressive A=0.5** — agent drifts toward corners, large exploitability:

![Aggressive A=0.5](experiments/results/aggressive_fixed_A0.5.gif)

**Aggressive A=0.9** — near-pure zoo sampling, agent collapses to a corner:

![Aggressive A=0.9](experiments/results/aggressive_fixed_A0.9.gif)

### Entropy Regularization Doesn't Shift A*

We swept 7 entropy levels (0.0 to 0.02) across 8 A values for PPO and 4 entropy levels across 7 A values for Buffered, all at 500k timesteps. Result: **A*=0.05 for all entropy levels tested, for both algorithms.**

![A* comparison](experiments/results/a_star_comparison.png)

Entropy regularization slightly reduces overall exploitability (all curves shift down with higher entropy) but does not change the optimal zoo sampling ratio. The A curve shape — minimum at A=0.05, rising through A=0.5–0.7, then partially recovering at A=0.9 — is consistent across all conditions.

### Training Dynamics

![Exploitability over training](experiments/results/timeseries.png)

## Key Findings

1. **Zoo sampling helps — in the right amount.** A small amount of zoo sampling (A=0.05) consistently reduces exploitability vs self-play. At 200k timesteps, heavier zoo sampling helps more; at 500k, only light zoo remains beneficial.
2. **A* depends on training length, not entropy.** At 200k, A*=0.9. At 500k, A*=0.05. The entropy coefficient (0.0–0.02) does not shift A*. This suggests the zoo's benefit is about convergence speed, not a fixed equilibrium property.
3. **Heavy zoo sampling degrades over time.** A=0.9 produces the lowest exploitability at 200k (0.0075) but rises 14x by 500k (0.105). The continued diversity of zoo opponents destabilizes the agent after initial convergence.
4. **PPO benefits more from zoo sampling than the buffered agent.** PPO's A curve drops more steeply than Buffered's, matching the hypothesis that memoryless algorithms are more sensitive to zoo sampling.
5. **Aggressive hyperparameters invert the A curve.** With high LR, small network, and no clipping, more zoo = worse performance. The agent needs enough capacity and stability to generalize from diverse opponents.

## Implications for the A-Parameter Hypothesis

**Confirmed:**
- Zoo sampling helps memoryless PPO converge (self-play alone cycles)
- PPO (memoryless) has a steeper A curve than the buffered agent — the "two curve shapes" prediction holds
- Interior A* exists — too much zoo hurts, even with proper regularization (at 500k)

**New insights from RPS:**
- **A* is dynamic, not static.** The optimal zoo ratio changes over training. Early on, heavy zoo accelerates convergence. Later, it destabilizes. This suggests A should be annealed during training — high early, low late.
- **Entropy regularization doesn't shift A*.** Within the range tested (0.0–0.02), all entropy levels produce the same A*=0.05 at 500k. Entropy reduces overall exploitability but doesn't change the optimal zoo ratio.
- **Aggressive hyperparameters break the zoo.** When the learning rate is too high or the network too small, zoo diversity becomes harmful at any level. The agent needs enough capacity and stability to generalize from diverse opponents.

**Not testable in RPS:**
- Whether the zoo *staleness* mechanism produces a U-shape. RPS Nash is fixed, so old checkpoints never become misleading. The degradation at high A we observe comes from over-diversity, not staleness. Testing the staleness mechanism requires a non-stationary environment like Tag or WindGym.

## Quick Start

```bash
pip install -r requirements.txt

# Self-play baseline (no zoo)
python train_selfplay.py --timesteps 200000

# Aggressive self-play (dramatic cycling — no entropy, high LR)
python train_selfplay.py --timesteps 500000 --entropy-coef 0.0 --lr 0.05 --hidden 4 --clip-ratio 100.0 --train-iters 5

# Single zoo run (PPO)
python train_zoo.py -A 0.1 --timesteps 200000

# Single zoo run (buffered)
python train_zoo_buffered.py -A 0.1 --timesteps 200000

# Full A sweep — both algorithms (150 experiments, ~minutes on CPU)
python run_sweep.py --timesteps 200000

# Analyze results
python analyze.py experiments/results/

# Generate simplex animations and visualizations
python visualize.py experiments/results/
```
