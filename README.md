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

## Metrics

- **Exploitability**: Max best-response payoff against the agent's mixed strategy. 0 at Nash (1/3, 1/3, 1/3), 1 at any pure strategy.
- **Action entropy**: Shannon entropy of the policy. Max = log(3) ~ 1.099 at Nash.

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
```
