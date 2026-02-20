#!/usr/bin/env python3
"""
Generate visualizations for the RPS A-parameter study.

Produces:
1. Simplex trajectory animations (GIF) showing how the agent's strategy
   evolves over training for different A values
2. Static simplex trajectories for comparison
3. Example episode rollouts showing concrete play
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from collections import defaultdict


# --- Simplex geometry ---
# The 2-simplex maps (p_R, p_P, p_S) to 2D equilateral triangle.
# Corners: Rock=(0,0), Scissors=(1,0), Paper=(0.5, sqrt(3)/2)
TRI_CORNERS = np.array([
    [0.0, 0.0],          # Rock
    [1.0, 0.0],          # Scissors
    [0.5, np.sqrt(3)/2], # Paper
])

NASH_2D = TRI_CORNERS.mean(axis=0)  # Center = (1/3, 1/3, 1/3)


def probs_to_2d(probs):
    """Map (p_R, p_P, p_S) to 2D simplex coordinates.

    Uses barycentric coordinates:
        point = p_R * corner_R + p_P * corner_P + p_S * corner_S
    """
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs @ TRI_CORNERS
    return probs @ TRI_CORNERS


def draw_simplex(ax, labels=True):
    """Draw the equilateral triangle with labels."""
    tri = plt.Polygon(TRI_CORNERS, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(tri)

    if labels:
        offset = 0.06
        ax.text(TRI_CORNERS[0, 0] - offset, TRI_CORNERS[0, 1] - offset,
                'Rock', ha='center', va='top', fontsize=11, fontweight='bold')
        ax.text(TRI_CORNERS[1, 0] + offset, TRI_CORNERS[1, 1] - offset,
                'Scissors', ha='center', va='top', fontsize=11, fontweight='bold')
        ax.text(TRI_CORNERS[2, 0], TRI_CORNERS[2, 1] + offset,
                'Paper', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Nash equilibrium marker
    ax.plot(*NASH_2D, 'k+', markersize=12, markeredgewidth=2, zorder=10)
    ax.text(NASH_2D[0] + 0.04, NASH_2D[1] + 0.02, 'Nash\n(1/3, 1/3, 1/3)',
            fontsize=8, color='gray')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')


def load_metrics(path):
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def collect_all(results_dir):
    """Collect all metrics files grouped by experiment type."""
    results = defaultdict(list)
    results_dir = Path(results_dir)

    for mf in results_dir.rglob("metrics.jsonl"):
        rel = mf.relative_to(results_dir)
        parts = rel.parts
        metrics = load_metrics(mf)
        if not metrics:
            continue

        if "selfplay" in parts[0]:
            seed = parts[1] if len(parts) > 1 else "0"
            results["selfplay"].append(metrics)
        else:
            import re
            match = re.search(r"zoo_A([\d.]+)", parts[0])
            if match:
                A = float(match.group(1))
                results[f"A={A:.2f}"].append(metrics)

    return results


# --- Animation: simplex trajectory ---

def animate_simplex(metrics_list, title, output_path, max_frames=200):
    """Create a GIF of strategy trajectories on the simplex.

    Args:
        metrics_list: list of metrics sequences (one per seed)
        title: plot title
        output_path: where to save the GIF
        max_frames: downsample to this many frames
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    draw_simplex(ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Prepare trajectories
    trajectories = []
    for metrics in metrics_list:
        probs = np.array([m["agent_probs"] for m in metrics])
        pts = probs_to_2d(probs)
        # Downsample
        if len(pts) > max_frames:
            idx = np.linspace(0, len(pts)-1, max_frames, dtype=int)
            pts = pts[idx]
            probs = probs[idx]
        trajectories.append((pts, probs))

    n_frames = max(len(t[0]) for t in trajectories)
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    # Initialize lines and dots
    lines = []
    dots = []
    for i, (pts, _) in enumerate(trajectories):
        line, = ax.plot([], [], '-', color=colors[i], alpha=0.3, linewidth=1)
        dot, = ax.plot([], [], 'o', color=colors[i], markersize=5)
        lines.append(line)
        dots.append(dot)

    # Info text
    info = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame):
        for i, (pts, probs) in enumerate(trajectories):
            idx = min(frame, len(pts)-1)
            lines[i].set_data(pts[:idx+1, 0], pts[:idx+1, 1])
            dots[i].set_data([pts[idx, 0]], [pts[idx, 1]])

        # Show probs from first seed
        pts0, probs0 = trajectories[0]
        idx0 = min(frame, len(probs0)-1)
        p = probs0[idx0]
        info.set_text(f'R={p[0]:.3f}  P={p[1]:.3f}  S={p[2]:.3f}\nStep {frame}/{n_frames}')
        return lines + dots + [info]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    anim.save(str(output_path), writer=PillowWriter(fps=20))
    plt.close()
    print(f"Saved: {output_path}")


# --- Static comparison plot ---

def plot_simplex_comparison(results, output_dir):
    """Plot final strategy positions on simplex for all A values."""
    conditions = ["selfplay"] + [k for k in sorted(results.keys()) if k.startswith("A=")]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, cond in enumerate(conditions):
        if i >= len(axes):
            break
        ax = axes[i]
        draw_simplex(ax, labels=(i == 0))  # Only label first

        for metrics in results[cond]:
            probs = np.array([m["agent_probs"] for m in metrics])
            pts = probs_to_2d(probs)
            ax.plot(pts[:, 0], pts[:, 1], '-', alpha=0.25, linewidth=0.8, color='blue')
            ax.plot(pts[-1, 0], pts[-1, 1], 'o', color='red', markersize=4, zorder=5)

        label = "Self-play (A=0)" if cond == "selfplay" else cond
        ax.set_title(label, fontsize=11)

    # Hide unused axes
    for j in range(len(conditions), len(axes)):
        axes[j].axis('off')

    fig.suptitle("Strategy Trajectories on the Simplex\n(blue=trajectory, red=final position, +=Nash)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / "simplex_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# --- Example episodes ---

def generate_episodes(output_dir):
    """Generate example episode rollouts showing concrete play.

    Simulates short sequences of play to illustrate what happens
    at different strategy distributions.
    """
    from rps_env import RPSEnv, PAYOFF

    np.random.seed(42)
    move_names = ["Rock", "Paper", "Scissors"]
    move_emoji = ["R", "P", "S"]
    outcome_sym = {1: "W", -1: "L", 0: "D"}

    scenarios = [
        ("Near-Nash agent vs Near-Nash opponent",
         np.array([0.34, 0.33, 0.33]),
         np.array([0.33, 0.34, 0.33])),
        ("Rock-biased agent (exploitable) vs Best Response",
         np.array([0.7, 0.15, 0.15]),
         np.array([0.0, 1.0, 0.0])),  # Paper always beats Rock
        ("Cycling agent vs Cycling opponent (self-play failure)",
         np.array([0.8, 0.1, 0.1]),   # Currently Rock-heavy
         np.array([0.1, 0.8, 0.1])),  # Currently Paper-heavy
    ]

    lines = ["## Example Episodes\n"]
    lines.append("What does play look like at different strategy distributions?\n")

    for title, agent_probs, opp_probs in scenarios:
        lines.append(f"### {title}\n")
        lines.append(f"Agent policy: R={agent_probs[0]:.2f}  P={agent_probs[1]:.2f}  S={agent_probs[2]:.2f}")
        lines.append(f"Opponent policy: R={opp_probs[0]:.2f}  P={opp_probs[1]:.2f}  S={opp_probs[2]:.2f}\n")
        lines.append("```")
        lines.append(f"{'Round':>5}  {'Agent':>6}  {'Opponent':>8}  {'Result':>6}  {'Payoff':>6}  {'Cumulative':>10}")
        lines.append("-" * 55)

        cumulative = 0
        n_rounds = 15
        agent_acts = np.random.choice(3, size=n_rounds, p=agent_probs)
        opp_acts = np.random.choice(3, size=n_rounds, p=opp_probs)

        for r in range(n_rounds):
            a, o = agent_acts[r], opp_acts[r]
            payoff = PAYOFF[a, o]
            cumulative += payoff
            result = outcome_sym[int(payoff)]
            lines.append(
                f"{r+1:>5}  {move_names[a]:>6}  {move_names[o]:>8}  "
                f"{result:>6}  {payoff:>+6.0f}  {cumulative:>+10.0f}"
            )

        lines.append("```\n")

        # Exploitability calculation
        br_payoffs = PAYOFF @ agent_probs
        expl = float(np.max(br_payoffs))
        br_action = move_names[int(np.argmax(br_payoffs))]
        lines.append(
            f"**Exploitability = {expl:.3f}** "
            f"(best response: always play {br_action} for expected payoff {expl:+.3f}/round)\n"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate RPS visualizations")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-animations", action="store_true",
                        help="Skip GIF generation (slow)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = collect_all(results_dir)

    # Static simplex comparison
    print("Generating simplex comparison...")
    plot_simplex_comparison(results, output_dir)

    # Animations
    if not args.skip_animations:
        # Self-play animation (3 seeds for clarity)
        if "selfplay" in results:
            print("Animating self-play trajectories...")
            animate_simplex(
                results["selfplay"][:3],
                "Self-Play (A=0): Strategy Cycling",
                output_dir / "selfplay_simplex.gif",
            )

        # Zoo animations for select A values
        for key in ["A=0.10", "A=0.50", "A=0.90"]:
            if key in results:
                print(f"Animating {key} trajectories...")
                animate_simplex(
                    results[key][:3],
                    f"Zoo ({key}): Strategy Trajectory",
                    output_dir / f"zoo_{key.replace('=','')}_simplex.gif",
                )

    # Episode examples
    print("Generating episode examples...")
    episodes_md = generate_episodes(output_dir)
    with open(output_dir / "episodes.md", "w") as f:
        f.write(episodes_md)
    print(f"Saved: {output_dir / 'episodes.md'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
