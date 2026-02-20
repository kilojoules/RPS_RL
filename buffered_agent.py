"""
Replay-buffer agent for RPS.

Uses the same softmax policy as PPO but trains off-policy from a
FIFO replay buffer. This is the "buffered" counterpart to PPO's
memoryless on-policy learning — the key distinction for the
A-parameter hypothesis.

Not a full SAC implementation (RPS doesn't need continuous actions
or Q-networks), but captures the essential property: the agent has
memory of past experience via the replay buffer.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ppo import SoftmaxPolicy


@dataclass
class BufferedConfig:
    obs_dim: int = 3
    act_dim: int = 3
    hidden: int = 32
    lr: float = 3e-3
    buffer_size: int = 10000
    batch_size: int = 256
    train_iters: int = 4
    entropy_coef: float = 0.01


class ReplayBuffer:
    """Simple FIFO replay buffer."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.obs = []
        self.actions = []
        self.rewards = []

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Add a batch of transitions."""
        for i in range(len(obs)):
            self.obs.append(obs[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])

        # FIFO eviction
        while len(self.obs) > self.max_size:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)

    def sample(self, batch_size: int):
        """Sample a random batch."""
        n = len(self.obs)
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        return (
            np.array([self.obs[i] for i in idx]),
            np.array([self.actions[i] for i in idx]),
            np.array([self.rewards[i] for i in idx]),
        )

    def __len__(self):
        return len(self.obs)


class BufferedAgent:
    """Off-policy agent with replay buffer.

    Trains via policy gradient on replayed transitions. The replay
    buffer provides the "memory" that PPO lacks — old experiences
    persist and influence learning even after the opponent changes.
    """

    def __init__(self, cfg: BufferedConfig):
        self.cfg = cfg
        self.policy = SoftmaxPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden)
        self.buffer = ReplayBuffer(max_size=cfg.buffer_size)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.policy.sample(obs)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.action_probs(obs)

    def store(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Store transitions in the replay buffer."""
        self.buffer.add(obs, actions, rewards)

    def update(self):
        """Train on a batch sampled from the replay buffer."""
        if len(self.buffer) < self.cfg.batch_size:
            return

        cfg = self.cfg
        for _ in range(cfg.train_iters):
            obs, actions, rewards = self.buffer.sample(cfg.batch_size)

            # Normalize rewards
            advantages = rewards.copy()
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute gradients via backward pass
            probs, cache = self.policy.forward(obs)
            grads = self.policy.backward(cache, actions, advantages)

            # Gradient ascent
            self.policy.W1 += cfg.lr * grads["W1"]
            self.policy.b1 += cfg.lr * grads["b1"]
            self.policy.W2 += cfg.lr * grads["W2"]
            self.policy.b2 += cfg.lr * grads["b2"]

    def get_state(self) -> List[np.ndarray]:
        return [p.copy() for p in self.policy.get_params()]

    def load_state(self, state: List[np.ndarray]):
        self.policy.set_params(state)
