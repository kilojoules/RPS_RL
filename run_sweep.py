#!/usr/bin/env python3
"""
A-parameter sweep for RPS.

Mirrors AI-Plays-Tag/experiments/run_zoo_sweep.py structure.
Runs zoo training across a grid of A values + the self-play baseline.
"""
import argparse
import subprocess
import sys
import json
import time
from pathlib import Path


A_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
NUM_SEEDS = 10


def main():
    parser = argparse.ArgumentParser(description="RPS A-parameter sweep")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--a-values", type=float, nargs="+", default=A_VALUES)
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = []

    # Self-play baseline across seeds
    for seed in range(args.seeds):
        experiments.append({
            "name": f"selfplay_s{seed}",
            "script": "train_selfplay.py",
            "args": [
                "--timesteps", str(args.timesteps),
                "--seed", str(seed),
                "--output-dir", f"{args.output_dir}/selfplay/seed_{seed}",
            ],
        })

    # Zoo sweep: A × seed
    for A in args.a_values:
        for seed in range(args.seeds):
            experiments.append({
                "name": f"zoo_A{A:.2f}_s{seed}",
                "script": "train_zoo.py",
                "args": [
                    "--latest-prob", str(A),
                    "--timesteps", str(args.timesteps),
                    "--seed", str(seed),
                    "--output-dir", f"{args.output_dir}/zoo_A{A:.2f}/seed_{seed}",
                ],
            })

    print(f"Total experiments: {len(experiments)}")
    print(f"  Self-play: {args.seeds}")
    print(f"  Zoo: {len(args.a_values)} A values x {args.seeds} seeds = {len(args.a_values) * args.seeds}")
    print(f"  A values: {args.a_values}")
    print(f"  Max parallel: {args.max_parallel}")

    if args.dry_run:
        for exp in experiments:
            print(f"  {exp['name']}: python {exp['script']} {' '.join(exp['args'])}")
        return

    running = []
    completed = []
    queue = list(experiments)

    while queue or running:
        # Launch
        while len(running) < args.max_parallel and queue:
            exp = queue.pop(0)
            cmd = [sys.executable, exp["script"]] + exp["args"]

            # Ensure output dir exists for log
            out_dir = None
            for i, a in enumerate(exp["args"]):
                if a == "--output-dir" and i + 1 < len(exp["args"]):
                    out_dir = exp["args"][i + 1]
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                log_file = open(Path(out_dir) / "train.log", "w")
            else:
                log_file = subprocess.DEVNULL

            proc = subprocess.Popen(
                cmd,
                stdout=log_file if isinstance(log_file, int) else log_file,
                stderr=subprocess.STDOUT,
            )
            running.append((exp, proc, log_file))

        # Check
        still_running = []
        for exp, proc, log_file in running:
            ret = proc.poll()
            if ret is not None:
                if hasattr(log_file, "close"):
                    log_file.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                completed.append((exp["name"], ret))
                print(f"  [{len(completed)}/{len(experiments)}] {exp['name']}: {status}")
            else:
                still_running.append((exp, proc, log_file))
        running = still_running

        if running:
            time.sleep(1)

    # Summary
    successes = sum(1 for _, r in completed if r == 0)
    failures = len(completed) - successes
    print(f"\nDone: {successes} succeeded, {failures} failed")

    summary = {"experiments": [{"name": n, "return_code": r} for n, r in completed]}
    summary_path = Path(args.output_dir) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
