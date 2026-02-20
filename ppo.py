"""
PPO agent for discrete-action RPS with analytic policy gradients.

Uses a two-layer softmax policy with exact backprop (no autograd framework needed).
On-policy / memoryless — the key property for the A-parameter hypothesis.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PPOConfig:
    obs_dim: int = 3
    act_dim: int = 3
    hidden: int = 32
    lr: float = 3e-3
    gamma: float = 0.0  # single-step game
    clip_ratio: float = 0.2
    train_iters: int = 4
    entropy_coef: float = 0.01


class SoftmaxPolicy:
    """Two-layer ReLU network with softmax output and analytic gradients.

    Forward: h = ReLU(W1 @ x + b1), logits = W2 @ h + b2, pi = softmax(logits)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32, entropy_coef: float = 0.01):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.entropy_coef = entropy_coef

        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = np.random.randn(hidden, obs_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(act_dim, hidden).astype(np.float32) * scale2
        self.b2 = np.zeros(act_dim, dtype=np.float32)

    def forward(self, obs: np.ndarray):
        """Full forward pass returning intermediates for backprop.

        Args:
            obs: (batch, obs_dim)

        Returns:
            probs: (batch, act_dim)
            cache: dict of intermediates
        """
        pre_h = obs @ self.W1.T + self.b1  # (batch, hidden)
        h = np.maximum(0, pre_h)  # ReLU
        logits = h @ self.W2.T + self.b2  # (batch, act_dim)

        # Stable softmax
        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        cache = {"obs": obs, "pre_h": pre_h, "h": h, "probs": probs}
        return probs, cache

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(obs)
        return probs

    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs, _ = self.forward(obs)
        actions = np.array([np.random.choice(len(p), p=p) for p in probs])
        log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-10)
        return actions, log_probs

    def backward(self, cache: dict, actions: np.ndarray, weights: np.ndarray):
        """Compute parameter gradients for policy gradient loss.

        Computes gradients of: mean_i [ weights[i] * log pi(actions[i] | obs[i]) ]
        plus entropy bonus.

        Args:
            cache: from forward()
            actions: (batch,) int
            weights: (batch,) float — e.g. clipped advantages

        Returns:
            grads: dict with keys W1, b1, W2, b2
        """
        batch = len(actions)
        obs = cache["obs"]
        pre_h = cache["pre_h"]
        h = cache["h"]
        probs = cache["probs"]

        # d(loss)/d(logits): for log pi(a|s), gradient is (one_hot(a) - pi)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch), actions] = 1.0
        d_logits = (one_hot - probs) * weights[:, None] / batch  # (batch, act_dim)

        # Entropy bonus gradient: d/d(logits) [ -sum pi log pi ] = -sum (1 + log pi)(one_hot(a) - pi)
        # Simplified: for softmax, d(entropy)/d(logits_j) = -sum_k pi_k (delta_jk - pi_j)(1 + log pi_k)
        # But easier: entropy gradient w.r.t. logits = -pi * (log pi + 1) + pi * sum(pi * (log pi + 1))
        # = pi * (H + 1 - log pi - 1) = pi * (H - log pi)
        # Actually for policy gradient, the entropy bonus just adds to d_logits:
        log_probs = np.log(probs + 1e-10)
        entropy_grad = -(log_probs + 1) * probs + probs * np.sum((log_probs + 1) * probs, axis=-1, keepdims=True)
        d_logits += self.entropy_coef * entropy_grad / batch

        # Backprop through W2, b2
        dW2 = d_logits.T @ h  # (act_dim, hidden)
        db2 = d_logits.sum(axis=0)  # (act_dim,)

        # Backprop through ReLU
        d_h = d_logits @ self.W2  # (batch, hidden)
        d_pre_h = d_h * (pre_h > 0).astype(np.float32)  # ReLU mask

        # Backprop through W1, b1
        dW1 = d_pre_h.T @ obs  # (hidden, obs_dim)
        db1 = d_pre_h.sum(axis=0)  # (hidden,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: List[np.ndarray]):
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]

    def copy(self) -> "SoftmaxPolicy":
        new = SoftmaxPolicy.__new__(SoftmaxPolicy)
        new.obs_dim = self.obs_dim
        new.act_dim = self.act_dim
        new.hidden = self.hidden
        new.entropy_coef = self.entropy_coef
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new


class PPOAgent:
    """PPO with clipped surrogate and analytic gradients.

    For a single-step game: reward = advantage = return (no discounting, no baseline needed).
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.policy = SoftmaxPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.entropy_coef)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.policy.sample(obs)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.action_probs(obs)

    def update(self, obs: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray, old_log_probs: np.ndarray):
        """PPO clipped update with analytic gradients."""
        cfg = self.cfg

        # Normalize advantages
        advantages = rewards.copy()
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(cfg.train_iters):
            probs, cache = self.policy.forward(obs)
            new_log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-10)

            # Importance sampling ratio
            ratio = np.exp(new_log_probs - old_log_probs)

            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages

            # Use the min for each sample, but we need the gradient
            # When clipped, gradient is zero (clip stops gradient).
            # So: weight = advantage if not clipped, else 0
            clipped = (ratio < 1 - cfg.clip_ratio) | (ratio > 1 + cfg.clip_ratio)
            weights = np.where(clipped, 0.0, advantages)

            grads = self.policy.backward(cache, actions, weights)

            # Gradient ascent
            self.policy.W1 += cfg.lr * grads["W1"]
            self.policy.b1 += cfg.lr * grads["b1"]
            self.policy.W2 += cfg.lr * grads["W2"]
            self.policy.b2 += cfg.lr * grads["b2"]

    def get_state(self) -> List[np.ndarray]:
        return [p.copy() for p in self.policy.get_params()]

    def load_state(self, state: List[np.ndarray]):
        self.policy.set_params(state)
