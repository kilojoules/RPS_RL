"""
Rock-Paper-Scissors environment for adversarial self-play experiments.

Payoff matrix (row = agent, col = opponent):
         R    P    S
    R  [ 0,  -1,  +1]
    P  [+1,   0,  -1]
    S  [-1,  +1,   0]

Actions: 0=Rock, 1=Paper, 2=Scissors
Observation: opponent's last action (one-hot), or zeros on first step.
"""
import numpy as np


ROCK, PAPER, SCISSORS = 0, 1, 2
NUM_ACTIONS = 3

# Payoff for row player
PAYOFF = np.array([
    [ 0, -1,  1],  # Rock vs {R, P, S}
    [ 1,  0, -1],  # Paper vs {R, P, S}
    [-1,  1,  0],  # Scissors vs {R, P, S}
], dtype=np.float32)


class RPSEnv:
    """Single-step RPS environment for vectorized play.

    Each call to step() is one round. Observation is the opponent's
    previous action as a one-hot vector (3,). On reset, obs is zeros.
    """

    def __init__(self, num_envs: int = 64):
        self.num_envs = num_envs
        self.obs_dim = NUM_ACTIONS  # one-hot of opponent's last action
        self.act_dim = NUM_ACTIONS
        self.last_opponent_actions = None

    def reset(self) -> np.ndarray:
        """Returns initial observation: zeros (no history yet)."""
        self.last_opponent_actions = None
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

    def step(self, agent_actions: np.ndarray, opponent_actions: np.ndarray):
        """Play one round.

        Args:
            agent_actions: (num_envs,) int array of agent choices
            opponent_actions: (num_envs,) int array of opponent choices

        Returns:
            obs: (num_envs, 3) one-hot of opponent's action this round
            rewards: (num_envs,) agent's payoff
            opp_rewards: (num_envs,) opponent's payoff
        """
        rewards = PAYOFF[agent_actions, opponent_actions]
        opp_rewards = PAYOFF[opponent_actions, agent_actions]

        # Next observation: one-hot of what the opponent just played
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        obs[np.arange(self.num_envs), opponent_actions] = 1.0

        self.last_opponent_actions = opponent_actions.copy()
        return obs, rewards, opp_rewards


def exploitability(action_probs: np.ndarray) -> float:
    """Compute exploitability: max best-response payoff against this mixed strategy.

    At Nash (1/3, 1/3, 1/3), exploitability = 0.
    At a pure strategy, exploitability = 1.

    Args:
        action_probs: (3,) probability distribution over {R, P, S}

    Returns:
        Exploitability (non-negative float). 0 = Nash equilibrium.
    """
    # Best response payoff = max_a E[payoff(a, opponent)] where opponent plays action_probs
    br_payoffs = PAYOFF @ action_probs  # (3,) expected payoff for each pure strategy
    return float(np.max(br_payoffs))


def action_entropy(action_probs: np.ndarray) -> float:
    """Shannon entropy of the action distribution. Max = log(3) ~ 1.099 at Nash."""
    p = np.clip(action_probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))
