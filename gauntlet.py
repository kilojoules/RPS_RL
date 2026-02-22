#!/usr/bin/env python3
"""
Gauntlet matrix: round-robin evaluation of self-play checkpoints.

Loads all checkpoint .npy files from a directory, plays every pair against
each other, and produces a heatmap showing win rates. Reveals transitive
vs cyclic structure in the agent's training trajectory.
"""
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rps_env import RPSEnv, exploitability
from ppo import PPOAgent, PPOConfig


def load_checkpoints(ckpt_dir: Path):
    """Load agent checkpoints sorted by timestep.

    Returns:
        list of (timestep, params) tuples, sorted by timestep.
    """
    checkpoints = []
    for f in sorted(ckpt_dir.glob("agent_*.npy")):
        m = re.search(r"agent_(\d+)\.npy", f.name)
        if m:
            ts = int(m.group(1))
            params = np.load(f, allow_pickle=True)
            checkpoints.append((ts, list(params)))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_pair(params_i, params_j, cfg, num_rounds=10_000):
    """Play agent_i vs agent_j for num_rounds and return win rate for i."""
    env = RPSEnv(num_envs=num_rounds)
    agent_i = PPOAgent(cfg)
    agent_j = PPOAgent(cfg)
    agent_i.load_state(params_i)
    agent_j.load_state(params_j)

    obs = np.zeros((num_rounds, 3), dtype=np.float32)
    actions_i, _ = agent_i.act(obs)
    actions_j, _ = agent_j.act(obs)
    _, rewards_i, _ = env.step(actions_i, actions_j)

    wins = (rewards_i > 0).sum()
    total = (rewards_i != 0).sum()  # exclude draws
    return wins / total if total > 0 else 0.5


def build_gauntlet(checkpoints, cfg, num_rounds=10_000):
    """Build the full win-rate matrix."""
    n = len(checkpoints)
    matrix = np.full((n, n), 0.5)

    for i in range(n):
        for j in range(i + 1, n):
            wr = evaluate_pair(checkpoints[i][1], checkpoints[j][1], cfg, num_rounds)
            matrix[i, j] = wr
            matrix[j, i] = 1.0 - wr

    return matrix


def plot_gauntlet(matrix, checkpoints, output_path):
    """Plot the gauntlet heatmap."""
    n = len(checkpoints)
    timesteps = [ts for ts, _ in checkpoints]

    # Compute exploitability for each checkpoint
    cfg = PPOConfig()
    test_obs = np.zeros((1, 3), dtype=np.float32)
    expls = []
    for _, params in checkpoints:
        agent = PPOAgent(cfg)
        agent.load_state(params)
        probs = agent.action_probs(test_obs)[0]
        expls.append(exploitability(probs))

    # Build tick labels: timestep (expl)
    labels = [f"t={ts}\n({e:.2f})" for ts, e in zip(timesteps, expls)]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5 + 2), max(6, n * 0.5 + 2)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=1, origin="upper")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=max(5, 10 - n // 10), rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=max(5, 10 - n // 10))

    ax.set_xlabel("Column agent (opponent)")
    ax.set_ylabel("Row agent")
    ax.set_title("Gauntlet Matrix: Self-Play Checkpoint Win Rates")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Win rate (row vs col)")

    # Annotate cells if matrix is small enough
    if n <= 20:
        for i in range(n):
            for j in range(n):
                color = "white" if abs(matrix[i, j] - 0.5) > 0.3 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=max(5, 8 - n // 5), color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gauntlet heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Gauntlet matrix for self-play checkpoints")
    parser.add_argument("results_dir", type=str, help="Directory containing checkpoints/ subfolder")
    parser.add_argument("--num-rounds", type=int, default=10_000,
                        help="Rounds per matchup (default: 10000)")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for heatmap (default: {results_dir}/gauntlet.png)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"Error: {ckpt_dir} not found. Run train_selfplay.py with --checkpoint-interval first.")
        return

    cfg = PPOConfig(hidden=args.hidden)
    checkpoints = load_checkpoints(ckpt_dir)
    print(f"Loaded {len(checkpoints)} checkpoints from {ckpt_dir}")

    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints for a gauntlet matrix.")
        return

    matrix = build_gauntlet(checkpoints, cfg, args.num_rounds)

    output_path = args.output or str(results_dir / "gauntlet.png")
    plot_gauntlet(matrix, checkpoints, output_path)

    # Print summary statistics
    off_diag = matrix[~np.eye(len(checkpoints), dtype=bool)]
    print(f"\nSummary:")
    print(f"  Mean off-diagonal win rate: {off_diag.mean():.3f} (should be ~0.5)")
    print(f"  Std off-diagonal win rate:  {off_diag.std():.3f}")
    print(f"  Max win rate: {off_diag.max():.3f}")
    print(f"  Min win rate: {off_diag.min():.3f}")

    # Check for cycles: if any triple (i,j,k) has i>j, j>k, k>i
    n = len(checkpoints)
    cycles = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (matrix[i, j] > 0.5 and matrix[j, k] > 0.5 and matrix[k, i] > 0.5):
                    cycles += 1
                elif (matrix[j, i] > 0.5 and matrix[k, j] > 0.5 and matrix[i, k] > 0.5):
                    cycles += 1
    total_triples = max(1, n * (n - 1) * (n - 2) // 6)
    print(f"  Cyclic triples: {cycles}/{total_triples} ({100 * cycles / total_triples:.1f}%)")


if __name__ == "__main__":
    main()
