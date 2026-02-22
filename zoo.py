"""
Opponent zoo: stores historical policy checkpoints and samples from them.

Mirrors the OpponentZoo from AI-Plays-Tag/trainer/train_zoo.py.
Supports uniform random sampling or Thompson Sampling (Beta-Bernoulli).
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ppo import PPOAgent, PPOConfig


class OpponentZoo:
    """Manages a zoo of opponent checkpoints."""

    def __init__(self, cfg: PPOConfig, max_size: int = 50,
                 sampling_strategy: str = "uniform",
                 competitiveness_threshold: float = 0.3):
        self.cfg = cfg
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self.competitiveness_threshold = competitiveness_threshold
        self.checkpoints: List[Dict[str, Any]] = []
        # Thompson Sampling posteriors: Beta(alpha, beta) per checkpoint
        self.alphas: List[float] = []
        self.betas: List[float] = []
        # Coverage-based sampling: counts per mode (R, P, S)
        self.coverage_counts = [0, 0, 0]

    def add(self, agent: PPOAgent, update: int):
        """Snapshot current agent params into the zoo."""
        probs = agent.action_probs(np.zeros((1, 3), dtype=np.float32))[0]
        self.checkpoints.append({
            "params": agent.get_state(),
            "update": update,
            "dominant_action": int(np.argmax(probs)),
        })
        self.alphas.append(1.0)
        self.betas.append(1.0)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)
            self.alphas.pop(0)
            self.betas.pop(0)

    def sample(self) -> Tuple[PPOAgent, int]:
        """Return (agent, index) from the zoo.

        With Thompson Sampling, samples theta_i ~ Beta(alpha_i, beta_i)
        for each checkpoint and picks argmax. With uniform, picks randomly.
        """
        if not self.checkpoints:
            raise ValueError("Zoo is empty")

        if self.sampling_strategy == "thompson" and len(self.checkpoints) > 1:
            thetas = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
            idx = int(np.argmax(thetas))
        elif self.sampling_strategy == "coverage" and len(self.checkpoints) > 1:
            # Group checkpoints by dominant action (cached at add time)
            groups: Dict[int, List[int]] = {0: [], 1: [], 2: []}
            for i, ckpt in enumerate(self.checkpoints):
                groups[ckpt["dominant_action"]].append(i)
            # Pick from least-covered mode that has checkpoints
            order = sorted(range(3), key=lambda m: self.coverage_counts[m])
            idx = None
            for mode in order:
                if groups[mode]:
                    idx = random.choice(groups[mode])
                    self.coverage_counts[mode] += 1
                    break
            if idx is None:
                idx = random.randrange(len(self.checkpoints))
        else:
            idx = random.randrange(len(self.checkpoints))

        ckpt = self.checkpoints[idx]
        agent = PPOAgent(self.cfg)
        agent.load_state(ckpt["params"])
        return agent, idx

    def update_outcome(self, idx: int, mean_reward: float):
        """Update Beta posterior for checkpoint idx based on match competitiveness.

        A match is 'competitive' (success) if |mean_reward| < threshold,
        meaning neither side dominated.
        """
        if idx < 0 or idx >= len(self.checkpoints):
            return
        if abs(mean_reward) < self.competitiveness_threshold:
            self.alphas[idx] += 1.0
        else:
            self.betas[idx] += 1.0

    def ts_diagnostics(self) -> Dict[str, float]:
        """Return Thompson Sampling diagnostic metrics."""
        if not self.alphas:
            return {}
        return {
            "ts_alpha_mean": float(np.mean(self.alphas)),
            "ts_beta_mean": float(np.mean(self.betas)),
            "ts_success_rate": float(
                np.mean([a / (a + b) for a, b in zip(self.alphas, self.betas)])
            ),
        }

    def coverage_diagnostics(self) -> Dict[str, float]:
        """Return coverage-based sampling diagnostic metrics."""
        total = sum(self.coverage_counts)
        if total == 0:
            return {"coverage_r": 0, "coverage_p": 0, "coverage_s": 0,
                    "coverage_entropy": 0.0}
        probs = [c / total for c in self.coverage_counts]
        entropy = -sum(p * math.log(p) if p > 0 else 0.0 for p in probs)
        return {
            "coverage_r": self.coverage_counts[0],
            "coverage_p": self.coverage_counts[1],
            "coverage_s": self.coverage_counts[2],
            "coverage_entropy": float(entropy),
        }

    def __len__(self):
        return len(self.checkpoints)


def a_schedule(t, timesteps, schedule="exponential", halflife=0.25):
    """Map training progress to A value in [0, 1].

    Increasing schedules (exponential, linear, sigmoid): start near 0, reach 1.
    Decreasing schedules (*_down): start near 1, reach 0.
    All pass through A=0.5 at t = halflife * timesteps.
    """
    # Decreasing schedules: mirror of the increasing variant
    if schedule.endswith("_down"):
        base = schedule[:-5]  # strip "_down"
        return 1.0 - a_schedule(t, timesteps, base, halflife)

    frac = t / timesteps
    h = halflife
    if schedule == "exponential":
        return 1.0 - math.exp(-math.log(2) * frac / h)
    elif schedule == "linear":
        return min(frac / (2 * h), 1.0)
    elif schedule == "sigmoid":
        k = math.log(99) / h
        return 1.0 / (1.0 + math.exp(-k * (frac - h)))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
