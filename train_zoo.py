#!/usr/bin/env python3
"""
Zoo-based training for RPS.

Mirrors AI-Plays-Tag/trainer/train_zoo.py structure:
- Agent trains against opponents sampled from a zoo of historical checkpoints
- With probability A: uniformly sample from the zoo
- With probability (1-A): play against latest opponent checkpoint
- A=0 is self-play (use train_selfplay.py). A>=1 is invalid.
"""
import argparse
import json
import numpy as np
from pathlib import Path

from rps_env import RPSEnv, exploitability, action_entropy
from ppo import PPOAgent, PPOConfig
from zoo import OpponentZoo, a_schedule


def train_zoo(
    latest_prob: float = 0.1,
    timesteps: int = 100_000,
    num_envs: int = 256,
    batch_size: int = 512,
    zoo_update_interval: int = 10,
    zoo_max_size: int = 50,
    log_interval: int = 100,
    output_dir: str = "experiments/results/zoo",
    seed: int = 0,
    cfg: PPOConfig = None,
    sampling_strategy: str = "uniform",
    competitiveness_threshold: float = 0.3,
    a_schedule_type: str = "constant",
    a_halflife: float = 0.25,
):
    np.random.seed(seed)

    if cfg is None:
        cfg = PPOConfig()
    env = RPSEnv(num_envs=num_envs)
    agent = PPOAgent(cfg)

    # The "latest" opponent that evolves alongside the agent
    latest_opponent = PPOAgent(cfg)

    # Historical zoo
    opponent_zoo = OpponentZoo(cfg, max_size=zoo_max_size,
                               sampling_strategy=sampling_strategy,
                               competitiveness_threshold=competitiveness_threshold)
    # Seed the zoo with the initial opponent
    opponent_zoo.add(latest_opponent, update=0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    obs = env.reset()
    total_rounds = 0
    update_step = 0

    all_obs, all_acts, all_rewards, all_logp = [], [], [], []
    opp_all_obs, opp_all_acts, opp_all_rewards, opp_all_logp = [], [], [], []

    while total_rounds < timesteps:
        # Compute current A value (scheduled or constant)
        if a_schedule_type != "constant":
            current_a = a_schedule(total_rounds, timesteps, a_schedule_type, a_halflife)
        else:
            current_a = latest_prob

        # Decide opponent for this step: zoo sample or latest
        zoo_idx = None
        if len(opponent_zoo) > 0 and np.random.random() < current_a:
            current_opponent, zoo_idx = opponent_zoo.sample()
        else:
            current_opponent = latest_opponent

        # Both act
        actions, log_probs = agent.act(obs)
        opp_obs = np.zeros((num_envs, 3), dtype=np.float32)  # opponent doesn't condition on history for simplicity
        opp_actions, opp_log_probs = current_opponent.act(opp_obs)

        obs_next, rewards, opp_rewards = env.step(actions, opp_actions)

        # Update Thompson Sampling posterior
        if zoo_idx is not None:
            opponent_zoo.update_outcome(zoo_idx, float(rewards.mean()))

        all_obs.append(obs)
        all_acts.append(actions)
        all_rewards.append(rewards)
        all_logp.append(log_probs)

        # Only collect opponent data when playing against latest (it's the one being trained)
        if current_opponent is latest_opponent:
            opp_all_obs.append(opp_obs)
            opp_all_acts.append(opp_actions)
            opp_all_rewards.append(opp_rewards)
            opp_all_logp.append(opp_log_probs)

        obs = obs_next
        total_rounds += num_envs

        # Update when we have enough data
        if len(all_obs) * num_envs >= batch_size:
            # Update agent
            batch_obs = np.concatenate(all_obs)
            batch_acts = np.concatenate(all_acts)
            batch_rew = np.concatenate(all_rewards)
            batch_logp = np.concatenate(all_logp)
            agent.update(batch_obs, batch_acts, batch_rew, batch_logp)

            # Update latest opponent (if we collected data)
            if opp_all_obs:
                opp_batch_obs = np.concatenate(opp_all_obs)
                opp_batch_acts = np.concatenate(opp_all_acts)
                opp_batch_rew = np.concatenate(opp_all_rewards)
                opp_batch_logp = np.concatenate(opp_all_logp)
                latest_opponent.update(opp_batch_obs, opp_batch_acts, opp_batch_rew, opp_batch_logp)

            all_obs, all_acts, all_rewards, all_logp = [], [], [], []
            opp_all_obs, opp_all_acts, opp_all_rewards, opp_all_logp = [], [], [], []

            update_step += 1

            # Add to zoo periodically
            if update_step % zoo_update_interval == 0:
                opponent_zoo.add(latest_opponent, update=update_step)

            if update_step % log_interval == 0:
                test_obs = np.zeros((1, 3), dtype=np.float32)
                probs = agent.action_probs(test_obs)[0]
                opp_probs = latest_opponent.action_probs(test_obs)[0]

                metrics = {
                    "update": update_step,
                    "timesteps": total_rounds,
                    "latest_prob": current_a,
                    "a_schedule_type": a_schedule_type,
                    "sampling_strategy": sampling_strategy,
                    "zoo_size": len(opponent_zoo),
                    "agent_probs": probs.tolist(),
                    "opponent_probs": opp_probs.tolist(),
                    "agent_exploitability": exploitability(probs),
                    "opponent_exploitability": exploitability(opp_probs),
                    "agent_entropy": action_entropy(probs),
                    "opponent_entropy": action_entropy(opp_probs),
                    "mean_reward": float(batch_rew.mean()),
                }
                if sampling_strategy == "thompson":
                    metrics.update(opponent_zoo.ts_diagnostics())
                with open(log_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                print(
                    f"[{total_rounds:>8d}] A={current_a:.4f} zoo={len(opponent_zoo):>3d} "
                    f"agent={probs.round(3)} expl={metrics['agent_exploitability']:.4f}"
                )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="RPS zoo training")
    parser.add_argument("--latest-prob", "-A", type=float, default=0.1,
                        help="A = probability of sampling from zoo (default: 0.1)")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--zoo-update-interval", type=int, default=10)
    parser.add_argument("--zoo-max-size", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/results/zoo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-strategy", type=str, default="uniform",
                        choices=["uniform", "thompson"],
                        help="Zoo sampling strategy (default: uniform)")
    parser.add_argument("--competitiveness-threshold", type=float, default=0.3,
                        help="Thompson Sampling competitiveness threshold (default: 0.3)")
    parser.add_argument("--a-schedule", type=str, default="constant",
                        choices=["constant", "exponential", "linear", "sigmoid",
                                 "exponential_down", "linear_down", "sigmoid_down"],
                        help="A-parameter schedule type (default: constant)")
    parser.add_argument("--a-halflife", type=float, default=0.25,
                        help="Fraction of training where A reaches 0.5 (default: 0.25)")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--train-iters", type=int, default=4)
    args = parser.parse_args()

    cfg = PPOConfig(
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        hidden=args.hidden,
        clip_ratio=args.clip_ratio,
        train_iters=args.train_iters,
    )

    train_zoo(
        latest_prob=args.latest_prob,
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        zoo_update_interval=args.zoo_update_interval,
        zoo_max_size=args.zoo_max_size,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        cfg=cfg,
        sampling_strategy=args.sampling_strategy,
        competitiveness_threshold=args.competitiveness_threshold,
        a_schedule_type=args.a_schedule,
        a_halflife=args.a_halflife,
    )


if __name__ == "__main__":
    main()
