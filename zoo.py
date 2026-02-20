"""
Opponent zoo: stores historical policy checkpoints and samples from them.

Mirrors the OpponentZoo from AI-Plays-Tag/trainer/train_zoo.py.
"""
import random
from typing import Any, Dict, List
from ppo import PPOAgent, PPOConfig


class OpponentZoo:
    """Manages a zoo of opponent checkpoints."""

    def __init__(self, cfg: PPOConfig, max_size: int = 50):
        self.cfg = cfg
        self.max_size = max_size
        self.checkpoints: List[Dict[str, Any]] = []

    def add(self, agent: PPOAgent, update: int):
        """Snapshot current agent params into the zoo."""
        self.checkpoints.append({
            "params": agent.get_state(),
            "update": update,
        })
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample(self) -> PPOAgent:
        """Return a new agent loaded with a random checkpoint from the zoo."""
        if not self.checkpoints:
            raise ValueError("Zoo is empty")
        ckpt = random.choice(self.checkpoints)
        agent = PPOAgent(self.cfg)
        agent.load_state(ckpt["params"])
        return agent

    def __len__(self):
        return len(self.checkpoints)
