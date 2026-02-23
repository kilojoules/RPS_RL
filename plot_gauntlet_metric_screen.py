#!/usr/bin/env python3
"""
Screen gauntlet-derived metrics against A* using synthetic gauntlet matrices.

In RPS, win rates between two policies can be computed analytically from their
action probability vectors (no simulation needed):

    W(i,j) = p_R^i * p_S^j + p_S^i * p_P^j + p_P^i * p_R^j

Every A=0 training log contains ~97 snapshots of agent_probs, so we can
construct a synthetic 97×97 gauntlet matrix per run — giving ~16 data points
across entropy sweeps instead of just the 2 real cached gauntlets.

15 candidate metrics are computed from each gauntlet and correlated with A*.

Usage:
    python plot_gauntlet_metric_screen.py [--results-dir DIR] [--output-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from calibrate_forgetting import (
    analyze_gap_structure,
    analyze_strategy_drift,
    fit_exponential_decay,
    fit_cosine_decay,
)
from plot_metric_vs_astar import (
    discover_sweep,
    load_metrics,
    get_final_exploitability,
)


# ── Synthetic gauntlet construction ──────────────────────────────────────────

def analytic_win_rate(probs_i, probs_j):
    """Compute P(i beats j | non-draw) from action probability vectors.

    probs = [P(Rock), P(Paper), P(Scissors)]
    Actions: 0=Rock, 1=Paper, 2=Scissors.
    Rock > Scissors, Paper > Rock, Scissors > Paper.
    """
    p_win = (probs_i[0] * probs_j[2]    # Rock beats Scissors
           + probs_i[1] * probs_j[0]    # Paper beats Rock
           + probs_i[2] * probs_j[1])   # Scissors beats Paper
    p_lose = (probs_i[0] * probs_j[1]   # Rock loses to Paper
            + probs_i[1] * probs_j[2]   # Paper loses to Scissors
            + probs_i[2] * probs_j[0])  # Scissors loses to Rock
    denom = p_win + p_lose
    return p_win / denom if denom > 1e-10 else 0.5


def build_synthetic_gauntlet(probs_sequence):
    """Build n×n analytic gauntlet from action probability snapshots.

    Args:
        probs_sequence: (n, 3) array — [P(R), P(P), P(S)] per snapshot.

    Returns:
        (n, n) win-rate matrix conditioned on non-draws.
    """
    n = len(probs_sequence)
    matrix = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            wr = analytic_win_rate(probs_sequence[i], probs_sequence[j])
            matrix[i, j] = wr
            matrix[j, i] = 1.0 - wr
    return matrix


# ── Metric extraction (15 metrics) ──────────────────────────────────────────

METRIC_NAMES = [
    "wr_range", "wr_var", "mean_abs_dev", "frac_decisive",
    "comp_at_1", "comp_at_half", "comp_drop",
    "comp_lambda", "comp_r2", "h_forget",
    "l2_rate", "cos_rate", "max_l2", "mean_l2_half",
    "transitivity",
]

METRIC_LABELS = {
    "wr_range": "WR Range",
    "wr_var": "WR Variance",
    "mean_abs_dev": "Mean |WR − 0.5|",
    "frac_decisive": "Frac Decisive (>0.1)",
    "comp_at_1": "Comp at Δ=1",
    "comp_at_half": "Comp at Δ=n/2",
    "comp_drop": "Comp Drop",
    "comp_lambda": "Comp λ",
    "comp_r2": "Comp Fit R²",
    "h_forget": "H_forget",
    "l2_rate": "L2 Growth Rate λ",
    "cos_rate": "Cosine Decay Rate λ",
    "max_l2": "Max L2 Distance",
    "mean_l2_half": "Mean L2 at Δ=n/2",
    "transitivity": "Transitivity",
}


def compute_gauntlet_metrics(matrix, probs_sequence=None):
    """Compute all 15 gauntlet-derived metrics.

    Args:
        matrix: (n, n) gauntlet win-rate matrix.
        probs_sequence: optional (n, 3) action probability array for
                        strategy-distance metrics. If None, those 4 metrics
                        are set to NaN.

    Returns:
        dict: metric_name -> float
    """
    n = matrix.shape[0]
    metrics = {}

    # Off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]

    # ── Win-rate matrix statistics ──
    metrics["wr_range"] = float(off_diag.max() - off_diag.min())
    metrics["wr_var"] = float(np.var(off_diag))
    metrics["mean_abs_dev"] = float(np.mean(np.abs(off_diag - 0.5)))
    metrics["frac_decisive"] = float(np.mean(np.abs(off_diag - 0.5) > 0.1))

    # ── Gap-dependent (competitiveness curve) ──
    gap_data = analyze_gap_structure(matrix)
    gaps = np.array(sorted(gap_data.keys()))
    comps = np.array([gap_data[d]["competitiveness"] for d in gaps])

    metrics["comp_at_1"] = float(comps[0]) if len(comps) > 0 else 1.0
    half_idx = len(gaps) // 2
    metrics["comp_at_half"] = float(comps[half_idx]) if half_idx < len(comps) else float(comps[-1])
    metrics["comp_drop"] = metrics["comp_at_1"] - metrics["comp_at_half"]

    # ── Exponential fit to competitiveness ──
    lam, A_fit, r_sq = fit_exponential_decay(gaps, comps)
    h_forget = np.log(2) / lam if lam > 1e-8 else 1e4
    h_forget = min(h_forget, 1e4)

    metrics["comp_lambda"] = float(lam)
    metrics["comp_r2"] = float(r_sq)
    metrics["h_forget"] = float(h_forget)

    # ── Strategy distance metrics ──
    if probs_sequence is not None and len(probs_sequence) >= 2:
        drift = analyze_strategy_drift(probs_sequence)
        drift_gaps = np.array(sorted(drift.keys()))
        l2_means = np.array([drift[d]["l2_mean"] for d in drift_gaps])
        cos_means = np.array([drift[d]["cosine_mean"] for d in drift_gaps])

        # L2 growth rate (invert to decay and fit)
        l2_max = max(float(l2_means.max()), 1e-6)
        l2_remaining = np.clip(1.0 - l2_means / (l2_max * 1.1), 1e-6, None)
        lam_l2, _, _ = fit_exponential_decay(drift_gaps, l2_remaining)
        metrics["l2_rate"] = float(lam_l2)

        # Cosine decay rate
        lam_cos, _, _, _ = fit_cosine_decay(drift_gaps, cos_means)
        metrics["cos_rate"] = float(lam_cos)

        # Max L2 and mean L2 at half
        metrics["max_l2"] = float(l2_means.max())
        half = len(drift_gaps) // 2
        window = slice(max(0, half - 2), min(len(drift_gaps), half + 3))
        metrics["mean_l2_half"] = float(np.mean(l2_means[window]))
    else:
        for k in ["l2_rate", "cos_rate", "max_l2", "mean_l2_half"]:
            metrics[k] = float("nan")

    # ── Transitivity (vectorized) ──
    B = (matrix > 0.5).astype(float)
    np.fill_diagonal(B, 0)
    B2 = B @ B  # B2[i,k] = count of j where i>j>k
    offdiag = ~np.eye(n, dtype=bool)
    n_total = float(B2[offdiag].sum())
    n_trans = float((B2 * B)[offdiag].sum())
    metrics["transitivity"] = n_trans / n_total if n_total > 0 else 1.0

    return metrics


# ── A* extraction ────────────────────────────────────────────────────────────

def extract_astar_for_configs(configs):
    """Extract A* and A=0 paths per entropy level from sweep configs.

    Args:
        configs: {entropy: {A: [metrics.jsonl paths]}}

    Returns:
        list of (entropy, astar, [A=0 seed paths])
    """
    results = []
    for ent in sorted(configs):
        a_to_paths = configs[ent]
        a0_paths = a_to_paths.get(0.0, [])
        if not a0_paths or len(a_to_paths) < 2:
            continue

        a_vals, mean_expls = [], []
        for A in sorted(a_to_paths):
            expls = [get_final_exploitability(load_metrics(p))
                     for p in a_to_paths[A]]
            a_vals.append(A)
            mean_expls.append(np.mean(expls))

        astar = a_vals[int(np.argmin(mean_expls))]
        results.append((ent, astar, a0_paths))

    return results


# ── Reference gauntlets ─────────────────────────────────────────────────────

def load_reference_gauntlets(results_dir):
    """Load real cached gauntlet matrices and compute metrics.

    Returns list of dicts with label, astar, and metric values.
    """
    refs = []
    candidates = [
        ("Standard (h=32)", results_dir / "selfplay_standard" / "gauntlet_matrix.npy", 0.9),
        ("Aggressive (h=4)", results_dir / "selfplay" / "gauntlet_matrix.npy", 0.05),
    ]
    for label, path, astar in candidates:
        if not path.exists():
            print(f"  Reference gauntlet not found: {path}")
            continue
        matrix = np.load(path)
        print(f"  Loaded {label}: {matrix.shape[0]}x{matrix.shape[1]}")
        mets = compute_gauntlet_metrics(matrix, probs_sequence=None)
        refs.append({"label": label, "astar": astar, **mets})

    return refs


# ── Plotting ─────────────────────────────────────────────────────────────────

def _safe_linregress(xs, ys):
    """Linear regression, returning None if degenerate."""
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    if len(xs) < 3 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return None
    try:
        return stats.linregress(xs, ys)
    except ValueError:
        return None


def plot_metric_screen(all_points, ref_points, output_dir):
    """4x4 grid: one subplot per metric (15 metrics + ranked summary)."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes_flat = axes.ravel()

    # Collect all data points across sweeps for combined regression
    combined = []
    for _, points, _, _, _ in all_points:
        combined.extend(points)

    ref_colors = ["red", "darkred"]

    for idx, mkey in enumerate(METRIC_NAMES):
        ax = axes_flat[idx]

        # Scatter sweep data
        for sweep_label, points, marker, color, shared in all_points:
            if not points:
                continue
            xs = [p[mkey] for p in points]
            ys = [p["astar"] for p in points]
            suffix = " *" if shared else ""
            ax.scatter(xs, ys, marker=marker, color=color, s=50,
                       label=sweep_label + suffix, zorder=3,
                       edgecolors="k", linewidths=0.4,
                       alpha=0.5 if shared else 0.9)

        # Reference gauntlets
        for ri, ref in enumerate(ref_points):
            val = ref.get(mkey)
            if val is not None and np.isfinite(val):
                lbl = ref["label"] if idx == 0 else None
                ax.scatter(val, ref["astar"], marker="*", s=200,
                           color=ref_colors[ri % len(ref_colors)], zorder=5,
                           edgecolors="k", linewidths=0.5, label=lbl)

        # Combined linear regression
        pairs = [(p[mkey], p["astar"]) for p in combined
                 if np.isfinite(p.get(mkey, float("nan")))]
        result = None
        if len(pairs) >= 3:
            xs_all, ys_all = zip(*pairs)
            result = _safe_linregress(list(xs_all), list(ys_all))

        if result is not None:
            slope, intercept, r, p_val, _ = result
            x_min, x_max = min(xs_all), max(xs_all)
            x_fit = np.linspace(x_min, x_max, 50)
            ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray",
                    alpha=0.6, linewidth=1.5)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.set_title(f"{METRIC_LABELS[mkey]}\nR\u00b2={r**2:.3f} {sig}",
                         fontsize=9)
        else:
            ax.set_title(METRIC_LABELS[mkey], fontsize=9)

        ax.set_xlabel(METRIC_LABELS[mkey], fontsize=7)
        ax.set_ylabel("A*", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    # 16th subplot: ranked correlation table
    ax = axes_flat[15]
    ax.axis("off")

    rows = []
    for mkey in METRIC_NAMES:
        pairs = [(p[mkey], p["astar"]) for p in combined
                 if np.isfinite(p.get(mkey, float("nan")))]
        if len(pairs) >= 3:
            xs_t, ys_t = zip(*pairs)
            result = _safe_linregress(list(xs_t), list(ys_t))
            if result is not None:
                _, _, r, p_val, _ = result
                rows.append((mkey, r**2, p_val))
                continue
        rows.append((mkey, 0.0, 1.0))

    rows.sort(key=lambda x: -x[1])

    sup2 = "\u00b2"
    hline = "\u2500"
    table_text = f"Ranked by R{sup2}:\n\n"
    table_text += f"{'Metric':<22} {'R' + sup2:>6} {'p':>8}\n"
    table_text += hline * 38 + "\n"
    for mkey, r2, pval in rows:
        sig = "***" if pval < 0.001 else "** " if pval < 0.01 else "*  " if pval < 0.05 else "   "
        table_text += f"{mkey:<22} {r2:>6.3f} {pval:>7.4f} {sig}\n"

    ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=7, verticalalignment="top")

    # Legend on first subplot
    axes_flat[0].legend(fontsize=6, loc="best")

    plt.suptitle(
        "Gauntlet-Derived Metrics vs A*\n"
        "(synthetic 97\u00d797 gauntlets from A=0 agent_probs; "
        "\u2605 = real gauntlets; * = shared A=0 baseline)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = output_dir / "gauntlet_metric_screen.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()
    return fig_path


# ── JSON summary ─────────────────────────────────────────────────────────────

def save_summary(all_points, ref_points, output_dir):
    """Save JSON summary with data points and ranked correlations."""
    combined = []
    sweeps_data = {}
    for label, points, _, _, shared in all_points:
        sweeps_data[label] = {"shared_baseline": shared, "points": points}
        combined.extend(points)

    correlations = {}
    for mkey in METRIC_NAMES:
        pairs = [(p[mkey], p["astar"]) for p in combined
                 if np.isfinite(p.get(mkey, float("nan")))]
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            result = _safe_linregress(list(xs), list(ys))
            if result is not None:
                slope, intercept, r, p_val, _ = result
                correlations[mkey] = {
                    "label": METRIC_LABELS[mkey],
                    "R2": round(r**2, 4),
                    "slope": round(float(slope), 6),
                    "p_value": round(float(p_val), 6),
                    "n_points": len(pairs),
                }
                continue
        correlations[mkey] = {
            "label": METRIC_LABELS[mkey],
            "n_points": len(pairs) if pairs else 0,
            "note": "degenerate",
        }

    ranked = sorted(
        ((k, v.get("R2", 0.0)) for k, v in correlations.items()),
        key=lambda x: -x[1],
    )

    summary = {
        "sweeps": sweeps_data,
        "reference_gauntlets": ref_points,
        "correlations": correlations,
        "ranked_metrics": [{"metric": k, "R2": r2} for k, r2 in ranked],
    }

    def json_default(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return None

    out_path = output_dir / "gauntlet_metric_screen_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)
    print(f"Saved: {out_path}")
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Screen gauntlet-derived metrics against A*")
    parser.add_argument("--results-dir", type=str,
                        default="experiments/results",
                        help="Root results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results-dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAUNTLET-DERIVED METRIC SCREEN")
    print("=" * 70)

    # ── Discover sweeps and build synthetic gauntlets ──
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
            print(f"  {label}: no data in {sweep_dir}")
            continue

        ent_data = extract_astar_for_configs(configs)
        if not ent_data:
            print(f"  {label}: no valid entropy levels")
            continue

        # Check for shared baseline (same A=0 files across entropy levels)
        a0_sizes = []
        for _, _, paths in ent_data:
            sizes = tuple(sorted(p.stat().st_size for p in paths))
            a0_sizes.append(sizes)
        shared = (len(a0_sizes) > 1
                  and all(s == a0_sizes[0] for s in a0_sizes[1:]))

        points = []
        shared_tag = " [SHARED BASELINE]" if shared else ""
        print(f"\n  {label}: {len(ent_data)} entropy levels{shared_tag}")

        for ent, astar, a0_paths in ent_data:
            # Compute metrics per seed, then average
            seed_metrics = []
            for path in a0_paths:
                rows = load_metrics(path)
                probs = np.array([r["agent_probs"] for r in rows])
                matrix = build_synthetic_gauntlet(probs)
                mets = compute_gauntlet_metrics(matrix, probs_sequence=probs)
                seed_metrics.append(mets)

            # Average across seeds
            avg = {}
            for k in METRIC_NAMES:
                vals = [sm[k] for sm in seed_metrics if np.isfinite(sm[k])]
                avg[k] = float(np.mean(vals)) if vals else float("nan")

            points.append({"ent": ent, "astar": astar, **avg})
            print(f"    ent={ent:<8}  A*={astar:<5}  "
                  f"wr_range={avg['wr_range']:.4f}  "
                  f"comp_drop={avg['comp_drop']:.4f}  "
                  f"transitivity={avg['transitivity']:.4f}")

        all_points.append((label, points, marker, color, shared))
        total += len(points)

    if total == 0:
        print("No data found! Check --results-dir path.")
        return

    print(f"\nTotal synthetic gauntlet data points: {total}")

    # ── Reference gauntlets ──
    print("\nReference gauntlets:")
    ref_points = load_reference_gauntlets(results_dir)

    # ── Plot and save ──
    fig_path = plot_metric_screen(all_points, ref_points, output_dir)
    summary = save_summary(all_points, ref_points, output_dir)

    # ── Print ranked correlations ──
    print("\n" + "=" * 60)
    print("RANKED CORRELATIONS (all sweeps combined)")
    print("=" * 60)
    for item in summary["ranked_metrics"]:
        mkey = item["metric"]
        info = summary["correlations"][mkey]
        if "R2" in info:
            pval = info["p_value"]
            sig = ("***" if pval < 0.001
                   else "**" if pval < 0.01
                   else "*" if pval < 0.05
                   else "")
            print(f"  {METRIC_LABELS[mkey]:<25}  R\u00b2={info['R2']:.4f}  "
                  f"p={pval:.4f} {sig}  (n={info['n_points']})")
        else:
            print(f"  {METRIC_LABELS[mkey]:<25}  {info.get('note', 'N/A')}")


if __name__ == "__main__":
    main()
