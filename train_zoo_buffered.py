#!/usr/bin/env python3
"""
Zoo-based training for RPS using the replay-buffer agent.

Same structure as train_zoo.py but uses BufferedAgent (off-policy with
replay buffer) instead of PPO (on-policy, memoryless). This tests
whether the replay buffer produces the predicted smoother A curve.
"""
import argparse
import json
import numpy as np
from pathlib import Path

from rps_env import RPSEnv, exploitability, action_entropy
from buffered_agent import BufferedAgent, BufferedConfig
from zoo import OpponentZoo
from ppo import PPOAgent, PPOConfig


class BufferedZoo:
    """Zoo that stores BufferedAgent checkpoints."""

    def __init__(self, cfg: BufferedConfig, max_size: int = 50):
        self.cfg = cfg
        self.max_size = max_size
        self.checkpoints = []

    def add(self, agent: BufferedAgent, update: int):
        self.checkpoints.append({
            "params": agent.get_state(),
            "update": update,
        })
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample(self) -> BufferedAgent:
        if not self.checkpoints:
            raise ValueError("Zoo is empty")
        import random
        ckpt = random.choice(self.checkpoints)
        agent = BufferedAgent(self.cfg)
        agent.load_state(ckpt["params"])
        return agent

    def __len__(self):
        return len(self.checkpoints)


def train_zoo_buffered(
    latest_prob: float = 0.1,
    timesteps: int = 100_000,
    num_envs: int = 256,
    buffer_size: int = 10000,
    update_every: int = 2,
    zoo_update_interval: int = 10,
    zoo_max_size: int = 50,
    log_interval: int = 100,
    output_dir: str = "experiments/results/zoo_buffered",
    seed: int = 0,
):
    np.random.seed(seed)

    env = RPSEnv(num_envs=num_envs)
    cfg = BufferedConfig(buffer_size=buffer_size)
    agent = BufferedAgent(cfg)
    latest_opponent = BufferedAgent(cfg)

    opponent_zoo = BufferedZoo(cfg, max_size=zoo_max_size)
    opponent_zoo.add(latest_opponent, update=0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    obs = env.reset()
    total_rounds = 0
    update_step = 0
    step_in_batch = 0

    while total_rounds < timesteps:
        # Decide opponent: zoo sample or latest
        if len(opponent_zoo) > 0 and np.random.random() < latest_prob:
            current_opponent = opponent_zoo.sample()
        else:
            current_opponent = latest_opponent

        # Both act
        actions, _ = agent.act(obs)
        opp_obs = np.zeros((num_envs, 3), dtype=np.float32)
        opp_actions, _ = current_opponent.act(opp_obs)

        obs_next, rewards, opp_rewards = env.step(actions, opp_actions)

        # Store in replay buffers
        agent.store(obs, actions, rewards)
        if current_opponent is latest_opponent:
            latest_opponent.store(opp_obs, opp_actions, opp_rewards)

        obs = obs_next
        total_rounds += num_envs
        step_in_batch += 1

        # Update from replay buffer periodically
        if step_in_batch >= update_every:
            agent.update()
            latest_opponent.update()
            step_in_batch = 0
            update_step += 1

            if update_step % zoo_update_interval == 0:
                opponent_zoo.add(latest_opponent, update=update_step)

            if update_step % log_interval == 0:
                test_obs = np.zeros((1, 3), dtype=np.float32)
                probs = agent.action_probs(test_obs)[0]

                metrics = {
                    "update": update_step,
                    "timesteps": total_rounds,
                    "latest_prob": latest_prob,
                    "zoo_size": len(opponent_zoo),
                    "agent_probs": probs.tolist(),
                    "agent_exploitability": exploitability(probs),
                    "agent_entropy": action_entropy(probs),
                    "mean_reward": float(rewards.mean()),
                    "buffer_size": len(agent.buffer),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                print(
                    f"[{total_rounds:>8d}] A={latest_prob} zoo={len(opponent_zoo):>3d} "
                    f"buf={len(agent.buffer):>5d} "
                    f"agent={probs.round(3)} expl={metrics['agent_exploitability']:.4f}"
                )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="RPS zoo training (buffered agent)")
    parser.add_argument("--latest-prob", "-A", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--zoo-update-interval", type=int, default=10)
    parser.add_argument("--zoo-max-size", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/results/zoo_buffered")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_zoo_buffered(
        latest_prob=args.latest_prob,
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        zoo_update_interval=args.zoo_update_interval,
        zoo_max_size=args.zoo_max_size,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
