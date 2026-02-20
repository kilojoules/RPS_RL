#!/usr/bin/env python3
"""
Analyze A-parameter sweep results.

Reads metrics.jsonl files from sweep output and produces:
1. Exploitability vs A curve (the main result)
2. Entropy vs A curve
3. Time series of exploitability for each A value
4. PPO vs Buffered comparison (if both present)
"""
import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_metrics(path: Path):
    """Load a metrics.jsonl file."""
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def get_final_metrics(metrics, window: int = 10):
    """Get average of last `window` logged entries."""
    if not metrics:
        return None
    tail = metrics[-window:]
    return {
        "exploitability": np.mean([m["agent_exploitability"] for m in tail]),
        "entropy": np.mean([m["agent_entropy"] for m in tail]),
        "probs": np.mean([m["agent_probs"] for m in tail], axis=0).tolist(),
    }


def collect_results(results_dir: Path):
    """Walk results directory and collect all experiments."""
    results = defaultdict(list)  # key -> list of final metrics across seeds

    for metrics_file in results_dir.rglob("metrics.jsonl"):
        rel = metrics_file.relative_to(results_dir)
        parts = rel.parts

        metrics = load_metrics(metrics_file)
        final = get_final_metrics(metrics)
        if final is None:
            continue

        # Determine experiment type
        if "selfplay" in parts[0]:
            results["selfplay"].append({"final": final, "timeseries": metrics})
        else:
            # Check for buffered_ prefix
            is_buffered = "buffered_" in parts[0]
            prefix = "buffered_" if is_buffered else ""

            match = re.search(r"zoo_A([\d.]+)", parts[0])
            if match:
                A = float(match.group(1))
                results[f"{prefix}zoo_A{A:.2f}"].append({
                    "final": final, "timeseries": metrics, "A": A
                })

    return results


def _collect_curve(results, prefix=""):
    """Extract A values and stats for a given algorithm prefix."""
    a_vals, expl_means, expl_stds = [], [], []
    ent_means, ent_stds = [], []

    for key, runs in sorted(results.items()):
        if not key.startswith(f"{prefix}zoo_A"):
            continue
        # Skip keys from the other algorithm
        if prefix == "" and key.startswith("buffered_"):
            continue
        A = runs[0]["A"]
        expls = [r["final"]["exploitability"] for r in runs]
        ents = [r["final"]["entropy"] for r in runs]
        a_vals.append(A)
        expl_means.append(np.mean(expls))
        expl_stds.append(np.std(expls))
        ent_means.append(np.mean(ents))
        ent_stds.append(np.std(ents))

    return (np.array(a_vals), np.array(expl_means), np.array(expl_stds),
            np.array(ent_means), np.array(ent_stds))


def plot_a_curve(results, output_dir: Path):
    """Plot exploitability vs A (the main result)."""
    has_ppo = any(k.startswith("zoo_A") and not k.startswith("buffered_") for k in results)
    has_buffered = any(k.startswith("buffered_zoo_A") for k in results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PPO curve
    if has_ppo:
        a_vals, expl_means, expl_stds, ent_means, ent_stds = _collect_curve(results, "")
        ax1.errorbar(a_vals, expl_means, yerr=expl_stds, fmt="o-", capsize=4,
                     color="C0", label="PPO (memoryless)")
        ax2.errorbar(a_vals, ent_means, yerr=ent_stds, fmt="s-", capsize=4,
                     color="C0", label="PPO (memoryless)")

    # Buffered curve
    if has_buffered:
        a_vals_b, expl_means_b, expl_stds_b, ent_means_b, ent_stds_b = _collect_curve(results, "buffered_")
        ax1.errorbar(a_vals_b, expl_means_b, yerr=expl_stds_b, fmt="^--", capsize=4,
                     color="C1", label="Buffered (replay buffer)")
        ax2.errorbar(a_vals_b, ent_means_b, yerr=ent_stds_b, fmt="^--", capsize=4,
                     color="C1", label="Buffered (replay buffer)")

    # Self-play baseline
    if "selfplay" in results:
        sp_expls = [r["final"]["exploitability"] for r in results["selfplay"]]
        sp_mean = np.mean(sp_expls)
        sp_std = np.std(sp_expls)
        ax1.axhline(sp_mean, color="red", linestyle="--", label="Self-play baseline (A=0)")
        ax1.axhspan(sp_mean - sp_std, sp_mean + sp_std, alpha=0.15, color="red")

    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.5, label="Nash (exploitability=0)")
    ax1.set_xlabel("A (zoo sampling probability)")
    ax1.set_ylabel("Exploitability")
    ax1.set_title("Exploitability vs A")
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)

    ax2.axhline(np.log(3), color="gray", linestyle=":", alpha=0.5, label="Max entropy (Nash)")
    ax2.set_xlabel("A (zoo sampling probability)")
    ax2.set_ylabel("Action entropy")
    ax2.set_title("Entropy vs A")
    ax2.legend()
    ax2.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    fig_path = output_dir / "a_curve.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    plt.close()


def plot_timeseries(results, output_dir: Path):
    """Plot exploitability over training for each A value."""
    has_buffered = any(k.startswith("buffered_") for k in results)
    n_plots = 2 if has_buffered else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 6), squeeze=False)

    for plot_idx, (prefix, title) in enumerate(
        [("", "PPO (memoryless)"), ("buffered_", "Buffered (replay buffer)")][:n_plots]
    ):
        ax = axes[0, plot_idx]
        zoo_keys = [k for k in sorted(results.keys())
                    if k.startswith(f"{prefix}zoo_A") and
                    (prefix != "" or not k.startswith("buffered_"))]
        colors = plt.cm.viridis(np.linspace(0, 1, len(zoo_keys)))

        # Self-play
        if "selfplay" in results and prefix == "":
            for run in results["selfplay"]:
                ts = run["timeseries"]
                steps = [m["timesteps"] for m in ts]
                expls = [m["agent_exploitability"] for m in ts]
                ax.plot(steps, expls, color="red", alpha=0.2, linewidth=0.5)
            ax.plot([], [], color="red", label="Self-play (A=0)")

        for ci, key in enumerate(zoo_keys):
            A = results[key][0]["A"]
            for run in results[key]:
                ts = run["timeseries"]
                steps = [m["timesteps"] for m in ts]
                expls = [m["agent_exploitability"] for m in ts]
                ax.plot(steps, expls, color=colors[ci], alpha=0.3, linewidth=0.5)
            ax.plot([], [], color=colors[ci], label=f"A={A:.2f}")

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Exploitability")
        ax.set_title(f"Exploitability over training — {title}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    fig_path = output_dir / "timeseries.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze RPS A-parameter sweep")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save plots (default: results_dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    results = collect_results(results_dir)

    print(f"Found {len(results)} experiment groups:")
    for key in sorted(results.keys()):
        n = len(results[key])
        if key == "selfplay":
            print(f"  {key}: {n} seeds")
        else:
            A = results[key][0]["A"]
            expls = [r["final"]["exploitability"] for r in results[key]]
            print(f"  {key}: {n} seeds, mean_expl={np.mean(expls):.4f} +/- {np.std(expls):.4f}")

    if not results:
        print("No results found!")
        return

    plot_a_curve(results, output_dir)
    plot_timeseries(results, output_dir)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for prefix, label in [("", "PPO (memoryless)"), ("buffered_", "Buffered (replay buffer)")]:
        keys = [k for k in sorted(results.keys())
                if k.startswith(f"{prefix}zoo_A") and
                (prefix != "" or not k.startswith("buffered_"))]
        if not keys:
            continue

        print(f"\n--- {label} ---")
        print(f"{'Condition':<25} {'Exploitability':>15} {'Entropy':>15}")
        print("-" * 55)

        if prefix == "" and "selfplay" in results:
            expls = [r["final"]["exploitability"] for r in results["selfplay"]]
            ents = [r["final"]["entropy"] for r in results["selfplay"]]
            print(f"{'Self-play (A=0)':<25} {np.mean(expls):>8.4f} +/- {np.std(expls):.4f} {np.mean(ents):>8.4f}")

        for key in keys:
            A = results[key][0]["A"]
            expls = [r["final"]["exploitability"] for r in results[key]]
            ents = [r["final"]["entropy"] for r in results[key]]
            print(f"{'A=' + f'{A:.2f}':<25} {np.mean(expls):>8.4f} +/- {np.std(expls):.4f} {np.mean(ents):>8.4f}")


if __name__ == "__main__":
    main()
