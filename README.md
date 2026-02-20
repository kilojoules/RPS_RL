# RPS_RL

Cheap testbed for the A-parameter hypothesis using Rock-Paper-Scissors.

## The A Parameter

In adversarial self-play, A (`--latest-prob`) controls opponent sampling:
- **A=0**: Always sample from historical zoo (SSP)
- **0 < A < 1**: Mix of latest opponent + zoo (Hybrid)
- **A=1**: Always play latest checkpoint (Arms Race)

A separate **self-play baseline** (`train_selfplay.py`) runs pure co-evolution with no zoo.

## Hypothesis

A* (optimal mixing ratio for convergence to Nash equilibrium) is inversely proportional to the algorithm's effective memory capacity.

- **PPO (memoryless)**: Sharp, unforgiving A curve. High exploitability at A=0 and A=1. High A*.
- **Tabular Q / DQN with replay buffer**: Smoother, more forgiving A curve. Lower A*, but still > 0.

## Metrics

- **Exploitability**: Distance from Nash equilibrium (1/3, 1/3, 1/3)
- **Action entropy**: Shannon entropy of the policy (max = log(3) at Nash)
- **Cycle amplitude**: Magnitude of strategy oscillations over time

## Quick Start

```bash
pip install -r requirements.txt

# Self-play baseline (no zoo)
python train_selfplay.py --timesteps 100000

# Single zoo run
python train_zoo.py --latest-prob 0.1 --timesteps 100000

# Full A sweep
python run_sweep.py

# Analyze results
python analyze.py experiments/results/
```
