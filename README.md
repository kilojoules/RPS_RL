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

Full sweep: 7 A values x 10 seeds + 10 self-play seeds = 80 experiments at 200k timesteps each.

### Exploitability vs A

![Exploitability vs A curve](experiments/results/a_curve.png)

| Condition | Exploitability (mean +/- std) | Entropy |
|-----------|-------------------------------|---------|
| Self-play (A=0) | 0.0724 +/- 0.0370 | 1.0905 |
| A=0.05 | 0.0380 +/- 0.0176 | 1.0957 |
| A=0.10 | 0.0374 +/- 0.0171 | 1.0958 |
| A=0.20 | 0.0354 +/- 0.0166 | 1.0960 |
| A=0.30 | 0.0322 +/- 0.0155 | 1.0965 |
| A=0.50 | 0.0258 +/- 0.0124 | 1.0972 |
| A=0.70 | 0.0169 +/- 0.0092 | 1.0980 |
| A=0.90 | 0.0075 +/- 0.0032 | 1.0985 |

### Strategy Trajectories on the Simplex

Each point in the triangle represents a mixed strategy over (Rock, Paper, Scissors). The center (+) is Nash equilibrium (1/3, 1/3, 1/3). Trajectories show how the agent's strategy evolves over 200k timesteps.

![Simplex comparison](experiments/results/simplex_comparison.png)

**Self-play (A=0)** — strategies wander far from Nash, scattered across seeds:

![Self-play cycling](experiments/results/selfplay_simplex.gif)

**Zoo A=0.10** — some wandering, but pulled back toward Nash:

![Zoo A=0.10](experiments/results/zoo_A0.10_simplex.gif)

**Zoo A=0.50** — tighter clustering around Nash:

![Zoo A=0.50](experiments/results/zoo_A0.50_simplex.gif)

**Zoo A=0.90** — converges tightly to Nash:

![Zoo A=0.90](experiments/results/zoo_A0.90_simplex.gif)

### Training Dynamics

![Exploitability over training](experiments/results/timeseries.png)

### Key Findings

1. **Zoo sampling monotonically improves convergence to Nash.** More zoo = lower exploitability. No interior optimum (A*) observed.
2. **Self-play cycles.** Without a zoo, PPO agents oscillate through strategies and never converge to Nash (1/3, 1/3, 1/3). Exploitability swings between 0.03 and 0.16.
3. **Even small zoo mixing helps.** A=0.05 (5% zoo) cuts exploitability roughly in half vs self-play.
4. **No U-shape.** The hypothesis predicted an optimal interior A* — too much zoo should hurt because historical opponents become stale. In RPS this doesn't happen because the Nash equilibrium is fixed. The zoo never goes stale.

### Implications for the A-Parameter Hypothesis

RPS confirms that zoo sampling helps a memoryless learner (PPO) converge, and that self-play alone cycles. But RPS is **too stationary** to test the core prediction: that there exists an optimal A* in the interior where too much zoo hurts performance.

The U-shape prediction requires a **non-stationary** environment where opponent strategies genuinely evolve over training, making old zoo checkpoints misleading. Tag and WindGym should exhibit this — opponents develop new strategies over time that make historical checkpoints poor training partners.

**RPS validates**: Zoo sampling > self-play for memoryless PPO.
**RPS cannot test**: Whether there's a point of diminishing (or negative) returns from too much zoo sampling.

## Quick Start

```bash
pip install -r requirements.txt

# Self-play baseline (no zoo)
python train_selfplay.py --timesteps 200000

# Single zoo run
python train_zoo.py -A 0.1 --timesteps 200000

# Full A sweep (80 experiments, ~minutes on CPU)
python run_sweep.py --timesteps 200000

# Analyze results
python analyze.py experiments/results/

# Generate simplex animations and episode examples
python visualize.py experiments/results/
```
