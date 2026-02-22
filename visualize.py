#!/usr/bin/env python3
"""
Generate visualizations for the RPS A-parameter study.

Produces:
1. Simplex trajectory animations (GIF) with darker colors for later iterations
2. Static simplex trajectories for comparison
3. PPO vs Buffered comparison plots
"""
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from collections import defaultdict


# --- Simplex geometry ---
TRI_CORNERS = np.array([
    [0.0, 0.0],          # Rock
    [1.0, 0.0],          # Scissors
    [0.5, np.sqrt(3)/2], # Paper
])

NASH_2D = TRI_CORNERS.mean(axis=0)


def probs_to_2d(probs):
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs @ TRI_CORNERS
    return probs @ TRI_CORNERS


def draw_simplex(ax, labels=True):
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

    ax.plot(*NASH_2D, 'k+', markersize=12, markeredgewidth=2, zorder=10)
    ax.text(NASH_2D[0] + 0.04, NASH_2D[1] + 0.02, 'Nash\n(1/3, 1/3, 1/3)',
            fontsize=8, color='gray')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_trajectory_colored(ax, pts, cmap_name='Blues', alpha=0.7, linewidth=1.5):
    """Draw a trajectory with color gradient: lighter=early, darker=late."""
    if len(pts) < 2:
        return
    segments = np.array([[pts[i], pts[i+1]] for i in range(len(pts)-1)])
    t = np.linspace(0.25, 1.0, len(segments))  # Start at 0.25 to avoid white
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(t)
    lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)
    # Start and end markers
    ax.plot(pts[0, 0], pts[0, 1], 'o', color=cmap(0.25), markersize=4, zorder=5)
    ax.plot(pts[-1, 0], pts[-1, 1], 'o', color=cmap(1.0), markersize=6, zorder=6,
            markeredgecolor='black', markeredgewidth=0.5)


def load_metrics(path):
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def collect_all(results_dir):
    results = defaultdict(list)
    results_dir = Path(results_dir)

    for mf in results_dir.rglob("metrics.jsonl"):
        rel = mf.relative_to(results_dir)
        parts = rel.parts
        metrics = load_metrics(mf)
        if not metrics:
            continue

        if "selfplay" in parts[0]:
            results["selfplay"].append(metrics)
        else:
            is_thompson = parts[0].startswith("ts_")
            is_coverage = parts[0].startswith("cov_")
            is_buffered = "buffered_" in parts[0]
            strat_prefix = "ts_" if is_thompson else "cov_" if is_coverage else ""
            buf_prefix = "buffered_" if is_buffered else ""
            match = re.search(r"zoo_A([\d.]+)", parts[0])
            if match:
                A = float(match.group(1))
                results[f"{strat_prefix}{buf_prefix}A={A:.2f}"].append(metrics)

    return results


# --- Animation with color gradient ---

def animate_simplex(metrics_list, title, output_path, max_frames=200):
    """Create a GIF with darker colors for later iterations.

    Shows both agent (solid line, circle) and opponent (dashed line, diamond)
    trajectories per seed so you can see them chasing each other.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    draw_simplex(ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    seed_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds']

    # Each seed produces two trajectories: agent and opponent
    trajectories = []  # list of (agent_pts, agent_probs, opp_pts, opp_probs)
    for metrics in metrics_list:
        agent_probs = np.array([m["agent_probs"] for m in metrics])
        agent_pts = probs_to_2d(agent_probs)
        has_opp = "opponent_probs" in metrics[0]
        if has_opp:
            opp_probs = np.array([m["opponent_probs"] for m in metrics])
            opp_pts = probs_to_2d(opp_probs)
        else:
            opp_probs = opp_pts = None

        if len(agent_pts) > max_frames:
            idx = np.linspace(0, len(agent_pts)-1, max_frames, dtype=int)
            agent_pts = agent_pts[idx]
            agent_probs = agent_probs[idx]
            if has_opp:
                opp_pts = opp_pts[idx]
                opp_probs = opp_probs[idx]
        trajectories.append((agent_pts, agent_probs, opp_pts, opp_probs))

    n_frames = max(len(t[0]) for t in trajectories)

    # For animation: draw colored segments incrementally
    collections = []  # (agent_lc, opp_lc, cmap) per seed
    dots = []  # (agent_dot, opp_dot) per seed
    for i in range(len(trajectories)):
        cmap = plt.get_cmap(seed_cmaps[i % len(seed_cmaps)])
        # Agent trajectory (solid)
        agent_lc = LineCollection([], linewidths=1.5, alpha=0.7)
        ax.add_collection(agent_lc)
        agent_dot, = ax.plot([], [], 'o', color=cmap(0.5), markersize=6,
                             markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        # Opponent trajectory (dashed, thinner)
        opp_lc = LineCollection([], linewidths=1.0, alpha=0.5, linestyles='dashed')
        ax.add_collection(opp_lc)
        opp_dot, = ax.plot([], [], 'D', color=cmap(0.5), markersize=5,
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        collections.append((agent_lc, opp_lc, cmap))
        dots.append((agent_dot, opp_dot))

    info = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Legend
    ax.text(0.98, 0.02, 'solid=agent  dashed=opponent\nlight=early  dark=late',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def update(frame):
        artists = []
        for i, (agent_pts, agent_probs, opp_pts, opp_probs) in enumerate(trajectories):
            idx = min(frame, len(agent_pts)-1)
            agent_lc, opp_lc, cmap = collections[i]
            agent_dot, opp_dot = dots[i]

            # Agent trajectory
            if idx >= 1:
                segs = np.array([[agent_pts[j], agent_pts[j+1]] for j in range(idx)])
                t = np.linspace(0.25, 1.0, len(segs))
                colors = cmap(t)
                agent_lc.set_segments(segs)
                agent_lc.set_color(colors)
            else:
                agent_lc.set_segments([])

            color_now = cmap(min(1.0, 0.25 + 0.75 * idx / max(1, len(agent_pts)-1)))
            agent_dot.set_data([agent_pts[idx, 0]], [agent_pts[idx, 1]])
            agent_dot.set_color(color_now)
            artists.extend([agent_lc, agent_dot])

            # Opponent trajectory
            if opp_pts is not None:
                opp_idx = min(frame, len(opp_pts)-1)
                if opp_idx >= 1:
                    opp_segs = np.array([[opp_pts[j], opp_pts[j+1]] for j in range(opp_idx)])
                    opp_t = np.linspace(0.25, 1.0, len(opp_segs))
                    opp_colors = cmap(opp_t)
                    opp_lc.set_segments(opp_segs)
                    opp_lc.set_color(opp_colors)
                else:
                    opp_lc.set_segments([])

                opp_color_now = cmap(min(1.0, 0.25 + 0.75 * opp_idx / max(1, len(opp_pts)-1)))
                opp_dot.set_data([opp_pts[opp_idx, 0]], [opp_pts[opp_idx, 1]])
                opp_dot.set_color(opp_color_now)
                artists.extend([opp_lc, opp_dot])

        agent_pts0, agent_probs0 = trajectories[0][0], trajectories[0][1]
        idx0 = min(frame, len(agent_probs0)-1)
        p = agent_probs0[idx0]
        info.set_text(f'R={p[0]:.3f}  P={p[1]:.3f}  S={p[2]:.3f}\nStep {frame}/{n_frames}')
        artists.append(info)
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    anim.save(str(output_path), writer=PillowWriter(fps=20))
    plt.close()
    print(f"Saved: {output_path}")


def animate_sidebyside(left_metrics, right_metrics, left_title, right_title,
                       output_path, max_frames=200):
    """Create a side-by-side GIF comparing two conditions on the simplex."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.5))

    seed_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds']

    all_ax_data = []
    for ax, metrics_list, title in [(ax_l, left_metrics, left_title),
                                     (ax_r, right_metrics, right_title)]:
        draw_simplex(ax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        trajectories = []
        for metrics in metrics_list:
            agent_probs = np.array([m["agent_probs"] for m in metrics])
            agent_pts = probs_to_2d(agent_probs)
            has_opp = "opponent_probs" in metrics[0]
            opp_pts = probs_to_2d(np.array([m["opponent_probs"] for m in metrics])) if has_opp else None
            opp_probs = np.array([m["opponent_probs"] for m in metrics]) if has_opp else None

            if len(agent_pts) > max_frames:
                idx = np.linspace(0, len(agent_pts)-1, max_frames, dtype=int)
                agent_pts = agent_pts[idx]
                agent_probs = agent_probs[idx]
                if has_opp:
                    opp_pts = opp_pts[idx]
                    opp_probs = opp_probs[idx]
            trajectories.append((agent_pts, agent_probs, opp_pts, opp_probs))

        collections = []
        dots = []
        for i in range(len(trajectories)):
            cmap = plt.get_cmap(seed_cmaps[i % len(seed_cmaps)])
            agent_lc = LineCollection([], linewidths=1.5, alpha=0.7)
            ax.add_collection(agent_lc)
            agent_dot, = ax.plot([], [], 'o', color=cmap(0.5), markersize=6,
                                 markeredgecolor='black', markeredgewidth=0.5, zorder=10)
            opp_lc = LineCollection([], linewidths=1.0, alpha=0.5, linestyles='dashed')
            ax.add_collection(opp_lc)
            opp_dot, = ax.plot([], [], 'D', color=cmap(0.5), markersize=5,
                               markeredgecolor='black', markeredgewidth=0.5, zorder=10)
            collections.append((agent_lc, opp_lc, cmap))
            dots.append((agent_dot, opp_dot))

        info = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=8,
                       verticalalignment='bottom', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        all_ax_data.append((trajectories, collections, dots, info))

    n_frames = max(
        max((len(t[0]) for t in ad[0]), default=1)
        for ad in all_ax_data
    )

    ax_r.text(0.98, 0.02, 'solid=agent  dashed=opponent\nlight=early  dark=late',
              transform=ax_r.transAxes, fontsize=7, ha='right', va='bottom', color='gray',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def update(frame):
        artists = []
        for trajectories, collections, dots, info in all_ax_data:
            for i, (agent_pts, agent_probs, opp_pts, opp_probs) in enumerate(trajectories):
                idx = min(frame, len(agent_pts)-1)
                agent_lc, opp_lc, cmap = collections[i]
                agent_dot, opp_dot = dots[i]

                if idx >= 1:
                    segs = np.array([[agent_pts[j], agent_pts[j+1]] for j in range(idx)])
                    t = np.linspace(0.25, 1.0, len(segs))
                    agent_lc.set_segments(segs)
                    agent_lc.set_color(cmap(t))
                else:
                    agent_lc.set_segments([])

                color_now = cmap(min(1.0, 0.25 + 0.75 * idx / max(1, len(agent_pts)-1)))
                agent_dot.set_data([agent_pts[idx, 0]], [agent_pts[idx, 1]])
                agent_dot.set_color(color_now)
                artists.extend([agent_lc, agent_dot])

                if opp_pts is not None:
                    opp_idx = min(frame, len(opp_pts)-1)
                    if opp_idx >= 1:
                        opp_segs = np.array([[opp_pts[j], opp_pts[j+1]] for j in range(opp_idx)])
                        opp_t = np.linspace(0.25, 1.0, len(opp_segs))
                        opp_lc.set_segments(opp_segs)
                        opp_lc.set_color(cmap(opp_t))
                    else:
                        opp_lc.set_segments([])
                    opp_color_now = cmap(min(1.0, 0.25 + 0.75 * opp_idx / max(1, len(opp_pts)-1)))
                    opp_dot.set_data([opp_pts[opp_idx, 0]], [opp_pts[opp_idx, 1]])
                    opp_dot.set_color(opp_color_now)
                    artists.extend([opp_lc, opp_dot])

            agent_pts0, agent_probs0 = trajectories[0][0], trajectories[0][1]
            idx0 = min(frame, len(agent_probs0)-1)
            p = agent_probs0[idx0]
            info.set_text(f'R={p[0]:.3f} P={p[1]:.3f} S={p[2]:.3f}')
            artists.append(info)
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    anim.save(str(output_path), writer=PillowWriter(fps=20))
    plt.close()
    print(f"Saved: {output_path}")


# --- Static comparison plot with color gradient ---

def plot_simplex_comparison(results, output_dir, prefix="", title_suffix=""):
    """Plot trajectories on simplex for all A values, colored by iteration."""
    conditions = []
    if "selfplay" in results and prefix == "":
        conditions.append("selfplay")
    conditions += [k for k in sorted(results.keys())
                   if k.startswith(f"{prefix}A=") and (prefix != "" or not k.startswith("buffered_"))]

    if not conditions:
        return

    n = len(conditions)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    axes_flat = axes.flatten()

    seed_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds',
                  'YlOrBr', 'BuGn', 'PuRd', 'GnBu', 'OrRd']

    for i, cond in enumerate(conditions):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        draw_simplex(ax, labels=(i == 0))

        for si, metrics in enumerate(results[cond]):
            probs = np.array([m["agent_probs"] for m in metrics])
            pts = probs_to_2d(probs)
            cmap = seed_cmaps[si % len(seed_cmaps)]
            draw_trajectory_colored(ax, pts, cmap_name=cmap, alpha=0.5, linewidth=1.0)

        if cond == "selfplay":
            label = "Self-play (A=0)"
        else:
            label = cond.replace("buffered_", "Buf ")
        ax.set_title(label, fontsize=11)

    for j in range(len(conditions), len(axes_flat)):
        axes_flat[j].axis('off')

    algo_label = title_suffix if title_suffix else "PPO"
    fig.suptitle(f"Strategy Trajectories — {algo_label}\n(light=early, dark=late, +=Nash)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    suffix = f"_{prefix.rstrip('_')}" if prefix else ""
    path = Path(output_dir) / f"simplex_comparison{suffix}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate RPS visualizations")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-animations", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = collect_all(results_dir)

    has_ppo = any(k.startswith("A=") for k in results)
    has_buffered = any(k.startswith("buffered_A=") for k in results)

    # Static simplex comparisons
    if has_ppo or "selfplay" in results:
        print("Generating PPO simplex comparison...")
        plot_simplex_comparison(results, output_dir, prefix="", title_suffix="PPO (memoryless)")

    if has_buffered:
        print("Generating Buffered simplex comparison...")
        plot_simplex_comparison(results, output_dir, prefix="buffered_",
                               title_suffix="Buffered (replay buffer)")

    # Animations
    if not args.skip_animations:
        for prefix, algo_label in [("", "PPO"), ("buffered_", "Buffered")]:
            keys_to_animate = ["selfplay"] if prefix == "" else []
            keys_to_animate += [k for k in sorted(results.keys())
                                if k.startswith(f"{prefix}A=") and
                                (prefix != "" or not k.startswith("buffered_"))]

            for key in keys_to_animate:
                if key not in results or not results[key]:
                    continue
                if key == "selfplay":
                    title = "Self-Play (A=0): Strategy Cycling"
                    fname = "selfplay_simplex.gif"
                else:
                    a_str = key.replace("buffered_", "")
                    title = f"{algo_label} ({a_str}): Strategy Trajectory"
                    fname = f"{prefix}{a_str.replace('=','')}_simplex.gif"

                print(f"Animating {key}...")
                animate_simplex(
                    results[key][:3],  # 3 seeds for clarity
                    title,
                    output_dir / fname,
                )

    # Thompson vs Uniform side-by-side animations
    if not args.skip_animations:
        for prefix, algo_label in [("", "PPO"), ("buffered_", "Buffered")]:
            for A in [0.50, 0.70, 0.90]:
                uni_key = f"{prefix}A={A:.2f}"
                ts_key = f"ts_{prefix}A={A:.2f}"
                if uni_key in results and ts_key in results:
                    print(f"Animating Thompson comparison: {algo_label} A={A:.2f}...")
                    animate_sidebyside(
                        results[uni_key][:3],
                        results[ts_key][:3],
                        f"Uniform (A={A:.2f})",
                        f"Thompson (A={A:.2f})",
                        output_dir / f"ts_vs_uniform_{prefix}A{A:.2f}.gif",
                    )

    # Coverage vs Uniform side-by-side animations
    if not args.skip_animations:
        for prefix, algo_label in [("", "PPO"), ("buffered_", "Buffered")]:
            for A in [0.50, 0.70, 0.90]:
                uni_key = f"{prefix}A={A:.2f}"
                cov_key = f"cov_{prefix}A={A:.2f}"
                if uni_key in results and cov_key in results:
                    print(f"Animating Coverage comparison: {algo_label} A={A:.2f}...")
                    animate_sidebyside(
                        results[uni_key][:3],
                        results[cov_key][:3],
                        f"Uniform (A={A:.2f})",
                        f"Coverage (A={A:.2f})",
                        output_dir / f"cov_vs_uniform_{prefix}A{A:.2f}.gif",
                    )

    print("\nDone!")


if __name__ == "__main__":
    main()
