#!/usr/bin/env python3
"""
Pure self-play baseline: two agents co-evolve live against each other.
No zoo, no checkpoint sampling. This is the A=0 self-play baseline.
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path

from rps_env import RPSEnv, exploitability, action_entropy
from ppo import PPOAgent, PPOConfig


def train_selfplay(
    timesteps: int = 100_000,
    num_envs: int = 256,
    batch_size: int = 512,
    log_interval: int = 100,
    output_dir: str = "experiments/results/selfplay",
    seed: int = 0,
    cfg: PPOConfig = None,
    checkpoint_interval: int = 0,
):
    np.random.seed(seed)

    if cfg is None:
        cfg = PPOConfig()
    env = RPSEnv(num_envs=num_envs)
    agent = PPOAgent(cfg)
    opponent = PPOAgent(cfg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    if checkpoint_interval > 0:
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

    obs = env.reset()
    opp_obs = env.reset()

    total_rounds = 0
    update_step = 0

    # Collect batches
    all_obs, all_acts, all_rewards, all_logp = [], [], [], []
    opp_all_obs, opp_all_acts, opp_all_rewards, opp_all_logp = [], [], [], []

    while total_rounds < timesteps:
        # Both agents act
        actions, log_probs = agent.act(obs)
        opp_actions, opp_log_probs = opponent.act(opp_obs)

        obs_next, rewards, opp_rewards = env.step(actions, opp_actions)

        all_obs.append(obs)
        all_acts.append(actions)
        all_rewards.append(rewards)
        all_logp.append(log_probs)

        opp_all_obs.append(opp_obs)
        opp_all_acts.append(opp_actions)
        opp_all_rewards.append(opp_rewards)
        opp_all_logp.append(opp_log_probs)

        # Opponent sees agent's action as next obs
        opp_obs_next = np.zeros_like(obs_next)
        opp_obs_next[np.arange(num_envs), actions] = 1.0

        obs = obs_next
        opp_obs = opp_obs_next
        total_rounds += num_envs

        # Update when we have enough data
        if len(all_obs) * num_envs >= batch_size:
            batch_obs = np.concatenate(all_obs)
            batch_acts = np.concatenate(all_acts)
            batch_rew = np.concatenate(all_rewards)
            batch_logp = np.concatenate(all_logp)
            agent.update(batch_obs, batch_acts, batch_rew, batch_logp)

            opp_batch_obs = np.concatenate(opp_all_obs)
            opp_batch_acts = np.concatenate(opp_all_acts)
            opp_batch_rew = np.concatenate(opp_all_rewards)
            opp_batch_logp = np.concatenate(opp_all_logp)
            opponent.update(opp_batch_obs, opp_batch_acts, opp_batch_rew, opp_batch_logp)

            all_obs, all_acts, all_rewards, all_logp = [], [], [], []
            opp_all_obs, opp_all_acts, opp_all_rewards, opp_all_logp = [], [], [], []

            update_step += 1

            if checkpoint_interval > 0 and update_step % checkpoint_interval == 0:
                state = np.array(agent.get_state(), dtype=object)
                np.save(ckpt_dir / f"agent_{total_rounds}.npy", state, allow_pickle=True)
                state = np.array(opponent.get_state(), dtype=object)
                np.save(ckpt_dir / f"opponent_{total_rounds}.npy", state, allow_pickle=True)

            if update_step % log_interval == 0:
                # Evaluate: get marginal action distribution
                test_obs = np.zeros((1, 3), dtype=np.float32)
                probs = agent.action_probs(test_obs)[0]
                opp_probs = opponent.action_probs(test_obs)[0]

                metrics = {
                    "update": update_step,
                    "timesteps": total_rounds,
                    "agent_probs": probs.tolist(),
                    "opponent_probs": opp_probs.tolist(),
                    "agent_exploitability": exploitability(probs),
                    "opponent_exploitability": exploitability(opp_probs),
                    "agent_entropy": action_entropy(probs),
                    "opponent_entropy": action_entropy(opp_probs),
                    "mean_reward": float(batch_rew.mean()),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                print(
                    f"[{total_rounds:>8d}] "
                    f"agent={probs.round(3)} expl={metrics['agent_exploitability']:.4f} "
                    f"opp={opp_probs.round(3)} expl={metrics['opponent_exploitability']:.4f}"
                )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="RPS self-play baseline")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/results/selfplay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--train-iters", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=0,
                        help="Save checkpoints every N updates (0=off)")
    args = parser.parse_args()

    cfg = PPOConfig(
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        hidden=args.hidden,
        clip_ratio=args.clip_ratio,
        train_iters=args.train_iters,
    )

    train_selfplay(
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        cfg=cfg,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
