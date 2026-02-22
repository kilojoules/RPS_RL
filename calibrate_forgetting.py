#!/usr/bin/env python3
"""
Phase 1 Calibration: Extract forgetting rate λ and coefficient C from RPS.

Part of the expansion plan: 'Formalizing the A-Parameter and Predicting A*'

Fits an exponential decay curve to the A=0 self-play cross-evaluation matrix
to measure how rapidly agents forget past strategies. Then uses the known A*
from completed sweeps to derive the universal coefficient C.

Three metrics are analyzed:
  1. Competitiveness: 1 - 2*mean(|WR - 0.5|) per gap Δ (from gauntlet matrix)
  2. Strategy distance: L2 distance between action probability vectors per gap
  3. Strategy autocorrelation: cosine similarity between action prob vectors

Usage:
    python calibrate_forgetting.py
    python calibrate_forgetting.py --a-star-ppo 0.9 --a-star-buffered 0.9
    python calibrate_forgetting.py --ckpt-dir experiments/results/selfplay/checkpoints --hidden 4
"""
import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from rps_env import RPSEnv, exploitability
from ppo import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# Gauntlet matrix construction
# ---------------------------------------------------------------------------

def load_checkpoints(ckpt_dir: Path, role: str = "agent"):
    """Load checkpoints sorted by timestep."""
    checkpoints = []
    for f in sorted(ckpt_dir.glob(f"{role}_*.npy")):
        m = re.search(rf"{role}_(\d+)\.npy", f.name)
        if m:
            ts = int(m.group(1))
            params = np.load(f, allow_pickle=True)
            checkpoints.append((ts, list(params)))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_pair(params_i, params_j, cfg, num_rounds=10_000):
    """Play agent_i vs agent_j and return win rate for i."""
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
    total = (rewards_i != 0).sum()
    return wins / total if total > 0 else 0.5


def build_gauntlet(row_ckpts, col_ckpts, cfg, num_rounds=10_000):
    """Build win-rate matrix: matrix[i,j] = P(row_i beats col_j)."""
    nr = len(row_ckpts)
    nc = len(col_ckpts)
    symmetric = row_ckpts is col_ckpts
    matrix = np.full((nr, nc), 0.5)

    total = (nr * (nr - 1) // 2) if symmetric else nr * nc
    done = 0
    for i in range(nr):
        jrange = range(i + 1, nc) if symmetric else range(nc)
        for j in jrange:
            if symmetric and i == j:
                continue
            wr = evaluate_pair(
                row_ckpts[i][1], col_ckpts[j][1], cfg, num_rounds
            )
            matrix[i, j] = wr
            if symmetric:
                matrix[j, i] = 1.0 - wr
            done += 1
            if done % 200 == 0:
                print(f"  Evaluated {done}/{total} matchups...")

    return matrix


def build_or_load_matrix(ckpt_dir, hidden, num_rounds, cache, mode):
    """Build gauntlet matrix or load from cache."""
    suffix = "" if mode == "agent_vs_agent" else "_avo"
    cache_path = ckpt_dir.parent / f"gauntlet_matrix{suffix}.npy"
    ts_cache = ckpt_dir.parent / f"gauntlet_timesteps{suffix}.npy"

    if cache and cache_path.exists() and ts_cache.exists():
        print(f"Loading cached matrix from {cache_path}")
        return np.load(cache_path), np.load(ts_cache)

    cfg = PPOConfig(hidden=hidden)
    agent_ckpts = load_checkpoints(ckpt_dir, "agent")
    print(f"Loaded {len(agent_ckpts)} agent checkpoints")

    if mode == "agent_vs_opponent":
        opp_ckpts = load_checkpoints(ckpt_dir, "opponent")
        print(f"Loaded {len(opp_ckpts)} opponent checkpoints")
        agent_ts = {ts for ts, _ in agent_ckpts}
        opp_ts = {ts for ts, _ in opp_ckpts}
        common = sorted(agent_ts & opp_ts)
        agent_map = {ts: p for ts, p in agent_ckpts}
        opp_map = {ts: p for ts, p in opp_ckpts}
        agent_ckpts = [(ts, agent_map[ts]) for ts in common]
        opp_ckpts = [(ts, opp_map[ts]) for ts in common]
        print(f"Common timesteps: {len(common)}")
        matrix = build_gauntlet(agent_ckpts, opp_ckpts, cfg, num_rounds)
        timesteps = np.array(common)
    else:
        matrix = build_gauntlet(agent_ckpts, agent_ckpts, cfg, num_rounds)
        timesteps = np.array([ts for ts, _ in agent_ckpts])

    if cache:
        np.save(cache_path, matrix)
        np.save(ts_cache, timesteps)
        print(f"Cached to {cache_path}")

    return matrix, timesteps


# ---------------------------------------------------------------------------
# Strategy distance analysis (action-probability drift)
# ---------------------------------------------------------------------------

def extract_action_probs(ckpt_dir, hidden):
    """Compute action probability vector for each checkpoint."""
    cfg = PPOConfig(hidden=hidden)
    agent_ckpts = load_checkpoints(ckpt_dir, "agent")
    test_obs = np.zeros((1, 3), dtype=np.float32)

    probs_list = []
    timesteps = []
    for ts, params in agent_ckpts:
        agent = PPOAgent(cfg)
        agent.load_state(params)
        probs = agent.action_probs(test_obs)[0]
        probs_list.append(probs)
        timesteps.append(ts)

    return np.array(probs_list), np.array(timesteps)


def analyze_strategy_drift(probs_matrix):
    """Compute strategy distance metrics per generation gap.

    Returns dict: gap -> {l2_mean, l2_std, cosine_mean, cosine_std}
    """
    n = len(probs_matrix)
    gap_data = {}

    for delta in range(1, n):
        l2_dists = []
        cosine_sims = []
        for i in range(delta, n):
            j = i - delta
            pi = probs_matrix[i]
            pj = probs_matrix[j]

            l2 = np.linalg.norm(pi - pj)
            cos = np.dot(pi, pj) / (np.linalg.norm(pi) * np.linalg.norm(pj) + 1e-10)
            l2_dists.append(l2)
            cosine_sims.append(cos)

        gap_data[delta] = {
            "l2_mean": float(np.mean(l2_dists)),
            "l2_std": float(np.std(l2_dists)),
            "cosine_mean": float(np.mean(cosine_sims)),
            "cosine_std": float(np.std(cosine_sims)),
        }

    return gap_data


# ---------------------------------------------------------------------------
# Gap analysis (gauntlet matrix)
# ---------------------------------------------------------------------------

def analyze_gap_structure(matrix):
    """Compute metrics per generation gap Δ from the lower triangle."""
    n = matrix.shape[0]
    gap_data = {}

    for delta in range(1, n):
        win_rates = []
        for i in range(delta, n):
            j = i - delta
            win_rates.append(matrix[i, j])

        win_rates = np.array(win_rates)
        abs_dev = np.mean(np.abs(win_rates - 0.5))

        gap_data[delta] = {
            "win_rates": win_rates,
            "n_pairs": len(win_rates),
            "mean_wr": float(np.mean(win_rates)),
            "std_wr": float(np.std(win_rates)),
            "abs_dev": float(abs_dev),
            "competitiveness": float(1.0 - 2.0 * abs_dev),
        }

    return gap_data


# ---------------------------------------------------------------------------
# Exponential fitting
# ---------------------------------------------------------------------------

def fit_exponential_decay(x, y, min_val=1e-6):
    """Fit y = A * exp(-λ * x) via log-linear regression.

    Returns (lambda, A, r_squared). λ > 0 means decay.
    """
    mask = y > min_val
    if mask.sum() < 2:
        return 0.0, 1.0, 0.0

    x_fit = x[mask].astype(float)
    y_fit = y[mask].astype(float)
    log_y = np.log(y_fit)

    coeffs = np.polyfit(x_fit, log_y, 1)
    lam = -coeffs[0]
    A_fit = np.exp(coeffs[1])

    y_pred = A_fit * np.exp(-lam * x_fit)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return float(lam), float(A_fit), float(r_sq)


def fit_cosine_decay(x, y):
    """Fit cosine similarity decay: y = A * exp(-λ * x) + baseline.

    Uses iterative approach: estimate baseline from tail, then fit residual.
    """
    baseline = float(np.mean(y[-max(1, len(y) // 5):]))
    shifted = y - baseline
    mask = shifted > 1e-6
    if mask.sum() < 2:
        return 0.0, 1.0, baseline, 0.0

    lam, A_fit, _ = fit_exponential_decay(x[mask], shifted[mask])

    y_pred = A_fit * np.exp(-lam * x) + baseline
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return float(lam), float(A_fit), float(baseline), float(r_sq)


# ---------------------------------------------------------------------------
# A* extraction
# ---------------------------------------------------------------------------

def find_a_star(results_dir: Path):
    """Find A* for PPO and Buffered from sweep results.

    Only considers seed_*/metrics.jsonl files to avoid mixing in
    non-sweep results (e.g. aggressive selfplay).
    """
    results = defaultdict(list)
    window = 10

    for metrics_file in results_dir.rglob("metrics.jsonl"):
        rel = metrics_file.relative_to(results_dir)
        parts = rel.parts

        # Only consider seed directories to avoid mixing run types
        if len(parts) < 2 or not parts[-2].startswith("seed_"):
            continue

        lines = []
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
        if not lines:
            continue

        tail = lines[-window:]
        final_expl = float(np.mean([m["agent_exploitability"] for m in tail]))

        exp_name = parts[0]
        if "selfplay" in exp_name:
            results["selfplay"].append(final_expl)
        else:
            is_buffered = "buffered" in exp_name
            match = re.search(r"zoo_A([\d.]+)", exp_name)
            if match:
                A = float(match.group(1))
                prefix = "buffered_" if is_buffered else ""
                results[f"{prefix}A{A:.2f}"].append(final_expl)

    if not results:
        return {}

    output = {}
    for prefix, label in [("", "PPO"), ("buffered_", "Buffered")]:
        a_vals, expl_means, expl_stds = [], [], []

        for key in sorted(results.keys()):
            if not key.startswith(f"{prefix}A"):
                continue
            if prefix == "" and key.startswith("buffered_"):
                continue
            m = re.search(r"A([\d.]+)", key)
            if m:
                expls = results[key]
                a_vals.append(float(m.group(1)))
                expl_means.append(np.mean(expls))
                expl_stds.append(np.std(expls))

        if not a_vals:
            continue

        if prefix == "" and "selfplay" in results:
            sp = results["selfplay"]
            a_vals.insert(0, 0.0)
            expl_means.insert(0, np.mean(sp))
            expl_stds.insert(0, np.std(sp))

        a_vals = np.array(a_vals)
        expl_means = np.array(expl_means)
        expl_stds = np.array(expl_stds)

        best_idx = int(np.argmin(expl_means))
        output[label] = {
            "a_star": float(a_vals[best_idx]),
            "min_expl": float(expl_means[best_idx]),
            "a_vals": a_vals.tolist(),
            "expl_means": expl_means.tolist(),
            "expl_stds": expl_stds.tolist(),
        }

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_full_analysis(gap_data, strategy_drift, fit_results, a_star_data,
                       output_dir, label=""):
    """Generate the comprehensive diagnostic figure."""
    gaps = np.array(sorted(gap_data.keys()))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # -- Mean win rate vs gap --
    ax = axes[0, 0]
    mean_wrs = np.array([gap_data[d]["mean_wr"] for d in gaps])
    ax.plot(gaps, mean_wrs, "o-", markersize=2, alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation gap (Δ)")
    ax.set_ylabel("Mean win rate")
    ax.set_title("Win rate vs gap (lower triangle)")

    # -- Competitiveness with exponential fit --
    ax = axes[0, 1]
    comps = np.array([gap_data[d]["competitiveness"] for d in gaps])
    ax.plot(gaps, comps, "o-", markersize=2, alpha=0.7, color="C2", label="Data")
    if "competitiveness" in fit_results:
        fr = fit_results["competitiveness"]
        fit_x = np.linspace(1, gaps[-1], 200)
        fit_y = fr["A"] * np.exp(-fr["lambda"] * fit_x)
        ax.plot(fit_x, fit_y, "--", color="red", linewidth=2,
                label=f"λ={fr['lambda']:.4f}, R²={fr['r_sq']:.3f}")
    ax.set_xlabel("Generation gap (Δ)")
    ax.set_ylabel("Competitiveness")
    ax.set_title("Competitiveness decay")
    ax.legend(fontsize=8)

    # -- Win rate distributions at selected gaps --
    ax = axes[0, 2]
    n_ckpts = int(gaps[-1])
    selected = sorted(set([1, max(1, n_ckpts // 4), max(1, n_ckpts // 2), n_ckpts])
                       & set(gap_data.keys()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected)))
    for d, c in zip(selected, colors):
        wrs = gap_data[d]["win_rates"]
        ax.hist(wrs, bins=20, alpha=0.4, color=c, label=f"Δ={d}")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Win rate")
    ax.set_ylabel("Count")
    ax.set_title("Win rate distributions")
    ax.legend(fontsize=8)

    # -- Strategy L2 distance --
    if strategy_drift:
        drift_gaps = np.array(sorted(strategy_drift.keys()))

        ax = axes[1, 0]
        l2_means = np.array([strategy_drift[d]["l2_mean"] for d in drift_gaps])
        l2_stds = np.array([strategy_drift[d]["l2_std"] for d in drift_gaps])
        ax.plot(drift_gaps, l2_means, "o-", markersize=2, alpha=0.7, color="C3")
        ax.fill_between(drift_gaps, l2_means - l2_stds, l2_means + l2_stds,
                         alpha=0.2, color="C3")
        ax.set_xlabel("Generation gap (Δ)")
        ax.set_ylabel("L2 distance")
        ax.set_title("Strategy drift (action prob L2 distance)")

        # -- Cosine similarity decay --
        ax = axes[1, 1]
        cos_means = np.array([strategy_drift[d]["cosine_mean"] for d in drift_gaps])
        cos_stds = np.array([strategy_drift[d]["cosine_std"] for d in drift_gaps])
        ax.plot(drift_gaps, cos_means, "o-", markersize=2, alpha=0.7, color="C4",
                label="Data")
        ax.fill_between(drift_gaps, cos_means - cos_stds, cos_means + cos_stds,
                         alpha=0.2, color="C4")
        if "cosine_similarity" in fit_results:
            fr = fit_results["cosine_similarity"]
            fit_x = np.linspace(1, drift_gaps[-1], 200)
            fit_y = fr["A"] * np.exp(-fr["lambda"] * fit_x) + fr["baseline"]
            ax.plot(fit_x, fit_y, "--", color="red", linewidth=2,
                    label=f"λ={fr['lambda']:.4f}, R²={fr['r_sq']:.3f}")
        ax.set_xlabel("Generation gap (Δ)")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Strategy autocorrelation")
        ax.legend(fontsize=8)

    # -- A-sweep curve --
    ax = axes[1, 2]
    if a_star_data:
        for algo, data in a_star_data.items():
            fmt = "o-" if algo == "PPO" else "^--"
            color = "C0" if algo == "PPO" else "C1"
            ax.errorbar(data["a_vals"], data["expl_means"],
                         yerr=data["expl_stds"], fmt=fmt, capsize=4,
                         color=color,
                         label=f"{algo} (A*={data['a_star']:.2f})")
        ax.set_xlabel("A (zoo sampling probability)")
        ax.set_ylabel("Exploitability")
        ax.set_title("A-sweep: known A*")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, "No sweep data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("A-sweep (no data)")

    title = "RPS Phase 1 Calibration"
    if label:
        title += f" — {label}"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    path = output_dir / f"calibration_{safe_label}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(ckpt_dir, hidden, num_rounds, cache, mode, a_star_data,
                 output_dir):
    """Run full analysis for one gauntlet mode. Returns results dict."""

    print(f"\n--- Gauntlet matrix ({mode}) ---")
    matrix, timesteps = build_or_load_matrix(
        ckpt_dir, hidden, num_rounds, cache, mode,
    )
    n = len(matrix)
    off_diag = matrix[~np.eye(n, dtype=bool)]
    ckpt_gap = int(timesteps[1] - timesteps[0]) if n > 1 else 0

    print(f"  Size: {n}x{n}  |  Timesteps: {timesteps[0]:,}–{timesteps[-1]:,}")
    print(f"  Win rate range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")

    # Gap analysis
    print(f"\n--- Gap analysis ---")
    gap_data = analyze_gap_structure(matrix)
    gaps = np.array(sorted(gap_data.keys()))

    print(f"  {'Δ':>5} {'N':>5} {'Mean WR':>9} {'|WR−0.5|':>9} {'Comp':>8}")
    for d in gaps:
        if d <= 3 or d % 10 == 0 or d == gaps[-1]:
            gd = gap_data[d]
            print(f"  {d:>5} {gd['n_pairs']:>5} {gd['mean_wr']:>9.4f} "
                  f"{gd['abs_dev']:>9.4f} {gd['competitiveness']:>8.4f}")

    # Strategy distance
    print(f"\n--- Strategy distance ---")
    probs_matrix, _ = extract_action_probs(ckpt_dir, hidden)
    strategy_drift = analyze_strategy_drift(probs_matrix)

    # Print strategy drift summary
    for d in [1, max(1, len(gaps) // 4), max(1, len(gaps) // 2), gaps[-1]]:
        if d in strategy_drift:
            sd = strategy_drift[d]
            print(f"  Δ={d:>3}: L2={sd['l2_mean']:.4f}±{sd['l2_std']:.4f}  "
                  f"cos={sd['cosine_mean']:.4f}±{sd['cosine_std']:.4f}")

    # Exponential fits
    print(f"\n--- Fits ---")
    fit_results = {}

    # Competitiveness decay
    comps = np.array([gap_data[d]["competitiveness"] for d in gaps])
    lam, A_fit, r_sq = fit_exponential_decay(gaps, comps)
    h_forget = np.log(2) / lam if lam > 1e-8 else float("inf")
    fit_results["competitiveness"] = {
        "lambda": lam, "A": A_fit, "r_sq": r_sq, "h_forget": h_forget,
    }
    print(f"  Competitiveness: λ={lam:.6f}, R²={r_sq:.4f}, "
          f"H_forget={'%.1f' % h_forget if h_forget < 1e6 else '∞'} gen")

    # Cosine similarity decay
    drift_gaps = np.array(sorted(strategy_drift.keys()))
    cos_means = np.array([strategy_drift[d]["cosine_mean"] for d in drift_gaps])
    lam_cos, A_cos, baseline_cos, r_sq_cos = fit_cosine_decay(drift_gaps, cos_means)
    h_forget_cos = np.log(2) / lam_cos if lam_cos > 1e-8 else float("inf")
    fit_results["cosine_similarity"] = {
        "lambda": lam_cos, "A": A_cos, "baseline": baseline_cos,
        "r_sq": r_sq_cos, "h_forget": h_forget_cos,
    }
    print(f"  Cosine sim:      λ={lam_cos:.6f}, R²={r_sq_cos:.4f}, "
          f"H_forget={'%.1f' % h_forget_cos if h_forget_cos < 1e6 else '∞'} gen")

    # L2 growth (fit 1 - norm(L2) to get decay)
    l2_means = np.array([strategy_drift[d]["l2_mean"] for d in drift_gaps])
    l2_max = max(l2_means.max(), 1e-6)
    l2_remaining = np.clip(1.0 - l2_means / (l2_max * 1.1), 1e-6, None)
    lam_l2, A_l2, r_sq_l2 = fit_exponential_decay(drift_gaps, l2_remaining)
    h_forget_l2 = np.log(2) / lam_l2 if lam_l2 > 1e-8 else float("inf")
    fit_results["l2_growth"] = {
        "lambda": lam_l2, "A": A_l2, "r_sq": r_sq_l2, "h_forget": h_forget_l2,
    }
    print(f"  L2 growth:       λ={lam_l2:.6f}, R²={r_sq_l2:.4f}, "
          f"H_forget={'%.1f' % h_forget_l2 if h_forget_l2 < 1e6 else '∞'} gen")

    # Pick best fit (highest R² with positive λ)
    best_metric = max(
        ((k, v) for k, v in fit_results.items() if v["lambda"] > 1e-8),
        key=lambda x: x[1]["r_sq"],
        default=(None, None),
    )
    best_name, best_fit = best_metric
    if best_fit:
        h_forget_best = best_fit["h_forget"]
        print(f"\n  Best fit: {best_name} (R²={best_fit['r_sq']:.4f})")
    else:
        h_forget_best = float("inf")
        print(f"\n  No positive-λ fit found. H_forget = ∞")

    # Coefficient C
    print(f"\n--- Coefficient C ---")
    calibration = {}
    if h_forget_best < float("inf"):
        for algo, data in a_star_data.items():
            a_star = data["a_star"]
            C = a_star * h_forget_best
            calibration[algo] = {
                "a_star": a_star, "h_forget": h_forget_best,
                "lambda": best_fit["lambda"], "best_metric": best_name,
                "C": C,
            }
            print(f"  {algo}: C = {a_star:.2f} × {h_forget_best:.2f} = {C:.4f}")
    else:
        print(f"  H_forget = ∞ → C is undefined.")
        print(f"  This means the gauntlet shows NO gap-dependent forgetting.")
        print(f"  (Expected for RPS standard hyperparams — see README.)")
        for algo, data in a_star_data.items():
            calibration[algo] = {
                "a_star": data["a_star"], "h_forget": float("inf"),
                "lambda": 0.0, "best_metric": None,
                "C": float("inf"),
            }

    # Plots
    label = mode.replace("_", " ")
    plot_full_analysis(gap_data, strategy_drift, fit_results, a_star_data,
                       output_dir, label)

    return {
        "mode": mode,
        "n_checkpoints": n,
        "timestep_range": [int(timesteps[0]), int(timesteps[-1])],
        "checkpoint_gap_timesteps": ckpt_gap,
        "win_rate_range": [float(off_diag.min()), float(off_diag.max())],
        "fit": {k: {kk: vv for kk, vv in v.items()}
                for k, v in fit_results.items()},
        "best_fit_metric": best_name,
        "calibration": calibration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Calibrate forgetting rate from RPS self-play gauntlet"
    )
    parser.add_argument(
        "--ckpt-dir", type=str,
        default="experiments/results/selfplay_standard/checkpoints",
        help="Checkpoint directory (default: selfplay_standard)")
    parser.add_argument(
        "--sweep-dir", type=str, default="experiments/results",
        help="Sweep results directory for A* extraction")
    parser.add_argument(
        "--hidden", type=int, default=32,
        help="Hidden size (32=standard, 4=aggressive)")
    parser.add_argument(
        "--num-rounds", type=int, default=10_000,
        help="Rounds per gauntlet matchup")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Rebuild gauntlet matrix from scratch")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: parent of ckpt-dir)")
    parser.add_argument(
        "--mode", type=str, default="agent_vs_agent",
        choices=["agent_vs_agent", "agent_vs_opponent", "both"],
        help="Gauntlet mode")
    parser.add_argument(
        "--a-star-ppo", type=float, default=None,
        help="Override A* for PPO (default: extract from sweep)")
    parser.add_argument(
        "--a-star-buffered", type=float, default=None,
        help="Override A* for Buffered (default: extract from sweep)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.exists():
        print(f"Error: {ckpt_dir} not found.")
        print("Run: python train_selfplay.py --checkpoint-interval 20 "
              "--output-dir experiments/results/selfplay_standard")
        return

    print("=" * 70)
    print("PHASE 1: RPS Forgetting Calibration")
    print("=" * 70)

    # A* data
    a_star_data = {}
    if sweep_dir.exists():
        a_star_data = find_a_star(sweep_dir)

    # Apply overrides
    if args.a_star_ppo is not None:
        if "PPO" not in a_star_data:
            a_star_data["PPO"] = {"a_vals": [], "expl_means": [], "expl_stds": []}
        a_star_data["PPO"]["a_star"] = args.a_star_ppo
        a_star_data["PPO"]["min_expl"] = 0.0
    if args.a_star_buffered is not None:
        if "Buffered" not in a_star_data:
            a_star_data["Buffered"] = {"a_vals": [], "expl_means": [], "expl_stds": []}
        a_star_data["Buffered"]["a_star"] = args.a_star_buffered
        a_star_data["Buffered"]["min_expl"] = 0.0

    print(f"\nA* reference values:")
    for algo, data in a_star_data.items():
        src = "override" if (
            (algo == "PPO" and args.a_star_ppo is not None) or
            (algo == "Buffered" and args.a_star_buffered is not None)
        ) else "sweep"
        print(f"  {algo}: A* = {data['a_star']:.2f} ({src})")

    if not a_star_data:
        print("  WARNING: No A* data. Use --a-star-ppo / --a-star-buffered.")

    # Run analysis
    modes = (["agent_vs_agent", "agent_vs_opponent"]
             if args.mode == "both" else [args.mode])

    all_results = {}
    for mode in modes:
        all_results[mode] = run_analysis(
            ckpt_dir, args.hidden, args.num_rounds,
            cache=not args.no_cache, mode=mode,
            a_star_data=a_star_data, output_dir=output_dir,
        )

    # Save
    combined = {
        "ckpt_dir": str(ckpt_dir),
        "hidden": args.hidden,
        "a_star": a_star_data,
        "modes": all_results,
    }
    results_path = output_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2,
                  default=lambda x: (float(x)
                                     if isinstance(x, (np.floating, np.integer))
                                     else x))
    print(f"\nSaved: {results_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    for mode, mr in all_results.items():
        print(f"\n  [{mode}]")
        bf = mr.get("best_fit_metric")
        if bf:
            fr = mr["fit"][bf]
            print(f"    Best metric: {bf}")
            print(f"    λ        = {fr['lambda']:.6f}")
            h = fr['h_forget']
            print(f"    H_forget = {'%.1f' % h if h < 1e6 else '∞'} generations")
            print(f"    R²       = {fr['r_sq']:.4f}")
        else:
            print(f"    No decay signal (λ ≤ 0 for all metrics)")
            print(f"    H_forget = ∞")

        for algo, cal in mr.get("calibration", {}).items():
            c = cal['C']
            c_str = f"{c:.4f}" if c < 1e6 else "∞"
            print(f"    {algo}: A*={cal['a_star']:.2f}, C={c_str}")

    # Interpretation
    best_r_sq = max(
        (mr["fit"].get(mr.get("best_fit_metric", ""), {}).get("r_sq", 0.0)
         for mr in all_results.values()),
        default=0.0,
    )
    any_inf = any(
        mr.get("calibration", {}).get("PPO", {}).get("C", float("inf")) == float("inf")
        for mr in all_results.values()
    )

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("=" * 70)

    if any_inf or best_r_sq < 0.2:
        print("""
  The gauntlet shows NO reliable gap-dependent forgetting signal.
  Best fit R² = %.4f (threshold for meaningful signal: > 0.2).

  This is expected for RPS with standard hyperparameters and confirms
  the README finding:

    'Zoo sampling in RPS solves a cycling problem, not a forgetting problem.'

  Win rates between checkpoints depend on their CYCLING PHASE (which
  strategy each happens to play), not their GENERATION GAP. A checkpoint
  from t=10k is no easier/harder to beat than one from t=400k.

  Implications for the expansion plan:
  1. The exponential decay model R(i,i-Δ) = R(i,i)·e^{-λΔ} assumes
     MONOTONIC forgetting, which RPS standard hyperparams do not exhibit.
  2. RPS serves as a NEGATIVE CONTROL: the model correctly identifies
     "no forgetting" (R² ≈ 0, λ ≈ 0) when it's absent.
  3. Coefficient C must be calibrated on a domain WITH monotonic
     forgetting (Tag or Chaos-1B in Phase 2/3).
  4. For comparison, aggressive hyperparams (hidden=4, no entropy) DO
     show a clear signal (R² = 0.74, λ = 0.004) because the dramatic
     cycling produces gap-dependent competitiveness decay.
""" % best_r_sq)
    else:
        print(f"""
  Decay signal detected. Best fit R² = {best_r_sq:.4f}.

  The exponential decay model captures gap-dependent structure in the
  cross-evaluation matrix. The derived coefficient C can be used to
  predict A* in other domains via: A* = C / H_forget.
""")

    print(f"Next: Validate on AI-Plays-Tag (Phase 2)")


if __name__ == "__main__":
    main()
