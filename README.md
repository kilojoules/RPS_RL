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

Any deviation from (1/3, 1/3, 1/3) can be exploited. This makes RPS ideal for testing convergence: we know exactly what the optimal strategy is.

## Exploitability: How We Measure Convergence

**Exploitability** measures how far a strategy is from Nash equilibrium. It answers: *if an opponent knew my strategy exactly, how much could they win per round?*

Given an agent's mixed strategy (p_R, p_P, p_S), the exploitability is the best-response payoff:

```
exploitability = max(
    p_S - p_P,    # payoff from always playing Rock
    p_R - p_S,    # payoff from always playing Paper
    p_P - p_R     # payoff from always playing Scissors
)
```

**Examples:**

| Strategy | Exploitability | Best response | Meaning |
|----------|---------------|---------------|---------|
| (0.33, 0.33, 0.33) | 0.000 | None (Nash) | Unexploitable |
| (0.50, 0.25, 0.25) | 0.250 | Always Paper | Wins +0.25/round |
| (0.70, 0.15, 0.15) | 0.550 | Always Paper | Wins +0.55/round |
| (1.00, 0.00, 0.00) | 1.000 | Always Paper | Wins every round |

At Nash, exploitability = 0. At a pure strategy, exploitability = 1.

## Example Episodes

What does play actually look like at different exploitability levels?

### Near-Nash agent (exploitability = 0.01)

Agent: R=0.34, P=0.33, S=0.33 — nearly uniform, almost unexploitable.

```
Round   Agent  Opponent  Result  Payoff  Cumulative
-------------------------------------------------------
    1   Paper      Rock       W      +1          +1
    2  Scissors    Rock       L      -1          +0
    3  Scissors   Paper       W      +1          +1
    4   Paper     Paper       D      +0          +1
    5    Rock      Rock       D      +0          +1
```

Even the best response (always Paper) only wins +0.01/round on average. Essentially unbeatable.

### Rock-biased agent (exploitability = 0.55)

Agent: R=0.70, P=0.15, S=0.15 — heavily exploitable. This is what self-play cycling produces: the agent over-commits to one action.

```
Round   Agent  Opponent  Result  Payoff  Cumulative
-------------------------------------------------------
    1    Rock     Paper       L      -1          -1
    2    Rock     Paper       L      -1          -2
    3    Rock     Paper       L      -1          -3
    4  Scissors   Paper       W      +1          -2
    5    Rock     Paper       L      -1          -3
```

An opponent who knows this strategy just plays Paper every time and wins +0.55/round.

### Self-play cycling failure (exploitability = 0.70)

Agent locked into Rock, opponent locked into Paper — both have drifted from Nash. This is the co-evolutionary failure mode: each agent over-adapts to the other's current strategy.

```
Round   Agent  Opponent  Result  Payoff  Cumulative
-------------------------------------------------------
    1    Rock     Paper       L      -1          -1
    2    Rock     Paper       L      -1          -2
    3   Paper      Rock       W      +1          -1
    4    Rock     Paper       L      -1          -2
    5    Rock     Paper       L      -1          -3
```

Both agents are highly exploitable by a third party playing the right counter-strategy, even though they may be "winning" against each other in alternating cycles.

## Results

Standard sweep: 150 experiments (7 A values x 10 seeds x 2 algorithms + 10 self-play seeds) at 200k timesteps each. Entropy/hyperparameter sweep: 252 experiments at 500k timesteps each.

### PPO vs Buffered: Exploitability vs A

![Exploitability vs A curve](experiments/results/a_curve.png)

**PPO (memoryless, on-policy):**

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

**Buffered (replay buffer, off-policy):**

| Condition | Exploitability (mean +/- std) | Entropy |
|-----------|-------------------------------|---------|
| A=0.05 | 0.0380 +/- 0.0178 | 1.0957 |
| A=0.10 | 0.0372 +/- 0.0172 | 1.0957 |
| A=0.20 | 0.0366 +/- 0.0157 | 1.0958 |
| A=0.30 | 0.0338 +/- 0.0155 | 1.0962 |
| A=0.50 | 0.0317 +/- 0.0152 | 1.0964 |
| A=0.70 | 0.0286 +/- 0.0145 | 1.0968 |
| A=0.90 | 0.0236 +/- 0.0121 | 1.0973 |

### Strategy Trajectories on the Simplex

Each point in the triangle represents a mixed strategy over (Rock, Paper, Scissors). The center (+) is Nash equilibrium (1/3, 1/3, 1/3). Colors go from **light (early)** to **dark (late)** iterations. Each color is a different random seed.

**PPO (memoryless):**

![PPO simplex comparison](experiments/results/simplex_comparison.png)

**Buffered (replay buffer):**

![Buffered simplex comparison](experiments/results/simplex_comparison_buffered.png)

### Animated Trajectories

**Self-play (A=0)** — strategies wander far from Nash, cycling through Rock → Paper → Scissors:

![Self-play cycling](experiments/results/selfplay_simplex.gif)

**PPO Zoo A=0.10** — some wandering, but pulled back toward Nash:

![PPO Zoo A=0.10](experiments/results/A0.10_simplex.gif)

**PPO Zoo A=0.90** — converges tightly to Nash:

![PPO Zoo A=0.90](experiments/results/A0.90_simplex.gif)

### Aggressive Self-Play: Dramatic Cycling

With default hyperparameters, self-play cycling is mild (exploitability 0.03–0.16). With aggressive hyperparameters — no entropy regularization, high learning rate, small network, and no PPO clipping — the agent over-commits much harder. Solid lines/circles are the agent; dashed lines/diamonds are the opponent.

```bash
python train_selfplay.py --entropy-coef 0.0 --lr 0.05 --hidden 4 --clip-ratio 100.0 --train-iters 5 --seed 5 --timesteps 500000
```

![Aggressive self-play cycling](experiments/results/aggressive_selfplay_fixed.gif)

Without entropy regularization the policy has no incentive to stay mixed, and the high learning rate makes each correction overshoot, creating the classic Rock → Paper → Scissors cycling failure mode. The small hidden layer (4 vs 32) reduces the network's capacity to represent smooth mixed strategies, and disabling PPO clipping (clip_ratio=100) removes the safety net that prevents large policy updates.

### Aggressive Config: Zoo Makes Things Worse

With the aggressive hyperparameters, the A curve **inverts** — more zoo sampling *increases* exploitability:

![Aggressive vs Standard A curve](experiments/results/aggressive_a_curve.png)

| Condition | Exploitability (aggressive) | Exploitability (standard) |
|-----------|----------------------------|--------------------------|
| Self-play (A=0) | 0.3170 | 0.0724 +/- 0.0370 |
| A=0.1 | 0.2412 +/- 0.0575 | 0.0374 +/- 0.0171 |
| A=0.3 | 0.3002 +/- 0.0817 | 0.0321 +/- 0.0155 |
| A=0.5 | 0.3467 +/- 0.0226 | 0.0258 +/- 0.0125 |
| A=0.9 | 0.9258 +/- 0.0744 | 0.0075 +/- 0.0033 |

With aggressive hyperparameters, the agent over-fits to beating specific zoo members instead of generalizing toward Nash. The zoo's diversity becomes a liability — each historical opponent pulls the agent toward a different counter-strategy, and the agent collapses to whichever pure strategy beats the most recent zoo sample.

**Aggressive A=0.1** — mild zoo, cycling with moderate amplitude:

![Aggressive A=0.1](experiments/results/aggressive_fixed_A0.1.gif)

**Aggressive A=0.3** — more zoo pulls the agent further off Nash:

![Aggressive A=0.3](experiments/results/aggressive_fixed_A0.3.gif)

**Aggressive A=0.5** — agent drifts toward corners, large exploitability:

![Aggressive A=0.5](experiments/results/aggressive_fixed_A0.5.gif)

**Aggressive A=0.9** — near-pure zoo sampling, agent collapses to a corner:

![Aggressive A=0.9](experiments/results/aggressive_fixed_A0.9.gif)

### The A Curve Evolves with Training Length

A surprising finding: the optimal A* depends on how long you train. At 200k timesteps, A*=0.9 — heavy zoo sampling produces the lowest exploitability. But by 500k timesteps, the picture inverts: A*=0.05, and high-A agents have degraded significantly.

![Training length effect](experiments/results/training_length_effect.png)

| A | Exploitability at 200k | Exploitability at 500k |
|---|----------------------|----------------------|
| 0.0 (self-play) | 0.0724 | 0.0753 |
| 0.05 | 0.0380 | 0.0556 |
| 0.10 | 0.0374 | 0.0590 |
| 0.50 | 0.0258 | 0.1349 |
| 0.90 | 0.0075 | 0.1054 |

At 200k, heavy zoo sampling (A=0.9) rapidly pushes the agent toward Nash — exploitability is only 0.0075. But this convergence doesn't persist. By 500k, the A=0.9 agent's exploitability has risen 14x to 0.105. Meanwhile, the A=0.05 agent slowly improves from 0.038 to 0.056.

**Interpretation:** Heavy zoo sampling accelerates early convergence by providing diverse opponents. But after the agent nears Nash, the continued diversity becomes destabilizing — the agent chases each zoo opponent's particular weakness, slowly pulling away from the uniform equilibrium. Light zoo sampling (A=0.05) is slower but more stable long-term.

### Does Entropy Shift A*?

After fixing a bug where the entropy coefficient wasn't being applied correctly (see Bug Fix note below), we ran a comprehensive sweep: 7 entropy levels × 8 A values × 3 seeds for PPO, and 4 entropy levels × 7 A values × 3 seeds for Buffered, all at 500k timesteps.

![A* comparison](experiments/results/a_star_comparison.png)

**Result: A*=0.05 for all entropy levels tested (0.0 to 0.02), for both PPO and Buffered.** Entropy regularization slightly reduces overall exploitability (all curves shift down with higher entropy) but does not change the optimal zoo sampling ratio. The A curve shape — minimum at A=0.05, rising through A=0.5–0.7, then partially recovering at A=0.9 — is consistent across all conditions.

**Bug fix note:** An earlier version of this code had a bug where `SoftmaxPolicy._entropy_coef` was a property that always returned 0.01, ignoring the configured value. This meant all experiments claiming to vary entropy coefficient were actually running with ent=0.01. The standard config results (which used ent=0.01 as the default) were unaffected. The aggressive experiments have been re-run with the fix.

### Training Dynamics

![Exploitability over training](experiments/results/timeseries.png)

### Key Findings

1. **Zoo sampling helps — in the right amount.** A small amount of zoo sampling (A=0.05) consistently reduces exploitability vs self-play. At 200k timesteps, heavier zoo sampling helps more; at 500k, only light zoo remains beneficial.
2. **A* depends on training length, not entropy.** At 200k, A*=0.9. At 500k, A*=0.05. The entropy coefficient (0.0–0.02) does not shift A*. This suggests the zoo's benefit is about convergence speed, not a fixed equilibrium property.
3. **Self-play cycles.** Without a zoo, PPO agents oscillate through strategies and never converge to Nash (1/3, 1/3, 1/3). Exploitability swings between 0.03 and 0.16.
4. **Heavy zoo sampling degrades over time.** A=0.9 produces the lowest exploitability at 200k (0.0075) but rises 14x by 500k (0.105). The continued diversity of zoo opponents destabilizes the agent after initial convergence.
5. **PPO benefits more from zoo sampling than the buffered agent.** At 200k, PPO's A curve drops more steeply than Buffered's — PPO reaches exploitability 0.0075 at A=0.9 while Buffered sits at 0.0236. This matches the hypothesis prediction that memoryless algorithms are more sensitive to zoo sampling.
6. **Aggressive hyperparameters invert the A curve.** With high LR, small network, and no clipping, more zoo = worse performance. The agent over-fits to beating specific zoo members rather than generalizing.

### Implications for the A-Parameter Hypothesis

RPS confirms the core predictions and adds nuances:

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
python train_selfplay.py --timesteps 200000 --entropy-coef 0.0 --lr 0.05 --hidden 4 --clip-ratio 100.0 --train-iters 5

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
