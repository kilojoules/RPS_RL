#!/usr/bin/env python3
"""
Scatter candidate proxy metrics (from A=0 self-play) against A* (from full sweeps).

Each entropy level in ent_sweep/, hyper_sweep/, and moderate_sweep/ yields one
(metric, A*) data point.  Candidate metrics are computed from the A=0 training
log alone — no checkpoint gauntlets required.

Candidate metrics:
  1. Exploitability variance — var(agent_exploitability) over training
  2. Exploitability mean    — mean(agent_exploitability) over training
  3. Cycling amplitude      — mean range of action probs across actions
  4. Strategy drift rate    — mean |Δprob| between consecutive updates

Usage:
    python plot_metric_vs_astar.py [--results-dir DIR] [--output-dir DIR]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ── helpers ──────────────────────────────────────────────────────────────────

def load_metrics(path: Path):
    """Load a metrics.jsonl file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_final_exploitability(rows, window=10):
    """Mean exploitability over last *window* logged entries."""
    tail = rows[-window:]
    return np.mean([r["agent_exploitability"] for r in tail])


# ── candidate proxy metrics (computed from a single A=0 run) ─────────────

def compute_proxy_metrics(rows):
    """Return dict of candidate proxy metrics for one training run."""
    expls = np.array([r["agent_exploitability"] for r in rows])
    probs = np.array([r["agent_probs"] for r in rows])  # (T, 3)

    # 1. Exploitability variance
    expl_var = float(np.var(expls))

    # 2. Exploitability mean
    expl_mean = float(np.mean(expls))

    # 3. Cycling amplitude: mean across actions of (max - min) over time
    cycling_amp = float(np.mean(np.max(probs, axis=0) - np.min(probs, axis=0)))

    # 4. Strategy drift rate: mean absolute change in probs between updates
    if len(probs) > 1:
        diffs = np.abs(np.diff(probs, axis=0))  # (T-1, 3)
        drift_rate = float(np.mean(diffs))
    else:
        drift_rate = 0.0

    return {
        "expl_var": expl_var,
        "expl_mean": expl_mean,
        "cycling_amp": cycling_amp,
        "drift_rate": drift_rate,
    }


# ── sweep discovery ──────────────────────────────────────────────────────────

def discover_sweep(sweep_dir: Path, prefix: str):
    """Discover (entropy, A) configurations in a sweep directory.

    Returns:
        dict: {entropy_level: {A_value: [list of metrics.jsonl paths]}}
    """
    configs = defaultdict(lambda: defaultdict(list))
    if not sweep_dir.is_dir():
        return configs

    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(prefix):
            continue
        # Strip prefix to get ent{E}_A{A}
        rest = name[len(prefix):]
        m = re.match(r"([\d.]+)_A([\d.]+)", rest)
        if not m:
            continue
        ent = float(m.group(1))
        A = float(m.group(2))
        for seed_dir in sorted(child.iterdir()):
            mf = seed_dir / "metrics.jsonl"
            if mf.is_file():
                configs[ent][A].append(mf)

    return configs


def extract_astar_and_metrics(configs):
    """For each entropy level, extract A* and proxy metrics from A=0.

    Args:
        configs: {entropy: {A: [metrics paths]}}

    Returns:
        tuple: (list of dicts with keys: ent, astar, + proxy metric names,
                bool indicating whether A=0 data is shared across entropy levels)
    """
    # Check if all entropy levels share the same A=0 data by comparing
    # file sizes (fast proxy for content identity)
    a0_sizes = []
    for ent in sorted(configs):
        a0_paths = configs[ent].get(0.0, [])
        if a0_paths:
            sizes = tuple(sorted(p.stat().st_size for p in a0_paths))
            a0_sizes.append(sizes)
    shared_baseline = (len(a0_sizes) > 1
                       and all(s == a0_sizes[0] for s in a0_sizes[1:]))

    results = []
    for ent in sorted(configs):
        a_to_paths = configs[ent]

        # Need A=0 for proxy metrics
        a0_paths = a_to_paths.get(0.0, [])
        if not a0_paths:
            continue

        # Need at least 2 A values to find A*
        if len(a_to_paths) < 2:
            continue

        # A* = A value that minimises mean final exploitability across seeds
        a_vals = []
        mean_expls = []
        for A in sorted(a_to_paths):
            paths = a_to_paths[A]
            expls = [get_final_exploitability(load_metrics(p)) for p in paths]
            a_vals.append(A)
            mean_expls.append(np.mean(expls))

        astar = a_vals[int(np.argmin(mean_expls))]

        # Proxy metrics: average across seeds at A=0
        seed_metrics = [compute_proxy_metrics(load_metrics(p)) for p in a0_paths]
        avg_metrics = {}
        for key in seed_metrics[0]:
            avg_metrics[key] = float(np.mean([sm[key] for sm in seed_metrics]))

        results.append({"ent": ent, "astar": astar, **avg_metrics})

    return results, shared_baseline


# ── plotting ─────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "expl_var": "Exploitability Variance",
    "expl_mean": "Mean Exploitability",
    "cycling_amp": "Cycling Amplitude",
    "drift_rate": "Strategy Drift Rate",
}


def _safe_linregress(xs, ys):
    """Run linregress, returning None if data is degenerate."""
    if len(xs) < 3:
        return None
    if np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return None
    try:
        return stats.linregress(xs, ys)
    except ValueError:
        return None


def plot_scatter(all_points, output_dir: Path):
    """Multi-panel scatter: one subplot per candidate metric, x=metric, y=A*."""
    metric_keys = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, mkey in zip(axes, metric_keys):
        y_offset = 0.05
        for sweep_label, points, marker, color, shared in all_points:
            if not points:
                continue
            xs = [p[mkey] for p in points]
            ys = [p["astar"] for p in points]
            suffix = " *" if shared else ""
            ax.scatter(xs, ys, marker=marker, color=color, s=60,
                       label=sweep_label + suffix, zorder=3,
                       edgecolors="k", linewidths=0.5,
                       alpha=0.4 if shared else 1.0)

            # Linear regression if data has variation in both axes
            result = _safe_linregress(xs, ys)
            if result is not None:
                slope, intercept, r, p_val, _ = result
                x_fit = np.linspace(min(xs), max(xs), 50)
                ax.plot(x_fit, slope * x_fit + intercept, "--", color=color,
                        alpha=0.5, linewidth=1)
                ax.annotate(f"R²={r**2:.2f}", xy=(0.95, y_offset),
                            xycoords="axes fraction", ha="right", fontsize=8,
                            color=color)
            y_offset += 0.08

        ax.set_xlabel(METRIC_LABELS[mkey])
        ax.set_ylabel("A*")
        ax.set_title(METRIC_LABELS[mkey])
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Candidate Proxy Metrics vs Optimal A*\n"
                 "(each point = one entropy level; * = shared A=0 baseline)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = output_dir / "metric_vs_astar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()
    return fig_path


def save_summary(all_points, output_dir: Path):
    """Save a JSON summary of all data points and per-metric R² values."""
    metric_keys = list(METRIC_LABELS.keys())
    summary = {"sweeps": {}, "correlations": {}}

    # Collect all points across sweeps
    combined = []
    for sweep_label, points, _, _, shared in all_points:
        summary["sweeps"][sweep_label] = {
            "shared_baseline": shared,
            "points": points,
        }
        combined.extend(points)

    # Compute correlations on combined data
    for mkey in metric_keys:
        xs = [p[mkey] for p in combined]
        ys = [p["astar"] for p in combined]
        result = _safe_linregress(xs, ys)
        if result is not None:
            slope, intercept, r, p_val, stderr = result
            summary["correlations"][mkey] = {
                "label": METRIC_LABELS[mkey],
                "R2": round(r**2, 4),
                "slope": round(slope, 6),
                "intercept": round(intercept, 4),
                "p_value": round(p_val, 6),
                "n_points": len(xs),
            }
        else:
            reason = "too few points" if len(xs) < 3 else "no variation in x or y"
            summary["correlations"][mkey] = {
                "label": METRIC_LABELS[mkey],
                "n_points": len(xs),
                "note": reason,
            }

    out_path = output_dir / "metric_vs_astar_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")
    return summary


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scatter candidate proxy metrics against A*")
    parser.add_argument("--results-dir", type=str,
                        default="experiments/results",
                        help="Root results directory (default: experiments/results)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results-dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sweeps (buf_ent excluded: no A=0 data in hyper_sweep buffered)
    sweeps = [
        ("ent_sweep", results_dir / "ent_sweep", "ent", "o", "C0"),
        ("hyper_sweep (PPO)", results_dir / "hyper_sweep", "ppo_ent", "s", "C1"),
        ("moderate_sweep", results_dir / "moderate_sweep", "ent", "D", "C3"),
    ]

    all_points = []
    total = 0
    for label, sweep_dir, prefix, marker, color in sweeps:
        configs = discover_sweep(sweep_dir, prefix)
        if not configs:
            print(f"  {label}: no data found in {sweep_dir}")
            continue
        points, shared = extract_astar_and_metrics(configs)
        shared_tag = " [SHARED A=0 BASELINE]" if shared else ""
        print(f"  {label}: {len(points)} entropy levels{shared_tag} "
              f"(ent={[p['ent'] for p in points]})")
        for p in points:
            print(f"    ent={p['ent']:<6}  A*={p['astar']:<5}  "
                  f"expl_var={p['expl_var']:.4f}  expl_mean={p['expl_mean']:.4f}  "
                  f"cycling_amp={p['cycling_amp']:.4f}  drift_rate={p['drift_rate']:.4f}")
        all_points.append((label, points, marker, color, shared))
        total += len(points)

    if total == 0:
        print("No data found! Check --results-dir path.")
        return

    print(f"\nTotal data points: {total}")

    # Plot and save
    plot_scatter(all_points, output_dir)
    summary = save_summary(all_points, output_dir)

    # Print correlation summary
    print("\n" + "=" * 60)
    print("CORRELATION SUMMARY (all sweeps combined)")
    print("=" * 60)
    for mkey, info in summary["correlations"].items():
        if "R2" in info:
            sig = "***" if info["p_value"] < 0.001 else "**" if info["p_value"] < 0.01 else "*" if info["p_value"] < 0.05 else ""
            print(f"  {info['label']:<25}  R²={info['R2']:.4f}  "
                  f"p={info['p_value']:.4f} {sig}  (n={info['n_points']})")
        else:
            print(f"  {info['label']:<25}  {info.get('note', 'N/A')}")


if __name__ == "__main__":
    main()
