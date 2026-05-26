"""Re-optimize the worst-performing segments with multi-restart CEM.

Strategy: identify the top-N highest-cost segments from the existing ILC
results, then re-optimise each with several independent restarts (different
init seeds, larger search) and keep the best.

Even halving the cost of the worst 100 segments would meaningfully lower
the mean (each one contributes 5-20x more than a median segment).
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from cem import _segment_fingerprint, cem_optimize_segment
from ilc import ilc_optimize_segment


def _worker(args):
    (csv_path, n_restarts, cem_iters, cem_pop, cem_workers,
     ilc_iters, ilc_lr, out_dir) = args
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    rng = np.random.default_rng(hash(seg_id) % (2**32))

    # Always include the previous best (ILC) as one candidate
    best_path = Path(out_dir) / f"{seg_id}.npz"
    if best_path.exists():
        d = np.load(best_path)
        best_cost = float(d["best_cost"])
        best_actions = d["actions"].astype(np.float32)
        baseline = float(d["baseline_cost"])
    else:
        best_cost = float("inf")
        best_actions = None
        baseline = float("inf")

    # Restart 1: ILC with much higher iterations (50 vs 30)
    try:
        a1, c1, b1, _ = ilc_optimize_segment(
            csv_path, "./models/tinyphysics.onnx",
            n_iters=ilc_iters, lr=ilc_lr,
            verbose=False,
        )
        if c1 < best_cost:
            best_cost, best_actions = c1, a1
        if baseline == float("inf"):
            baseline = b1
    except Exception as e:
        pass

    # Restarts 2..n_restarts: CEM with random init perturbations
    for r in range(n_restarts):
        try:
            actions, cost, _, _ = cem_optimize_segment(
                csv_path, "./models/tinyphysics.onnx",
                iters=cem_iters, pop_size=cem_pop,
                workers=cem_workers, seed=int(rng.integers(0, 1_000_000)),
                verbose=False,
            )
            if cost < best_cost:
                best_cost, best_actions = cost, actions
        except Exception as e:
            pass

    fp = _segment_fingerprint(csv_path)
    np.savez(best_path, actions=best_actions, best_cost=best_cost,
             baseline_cost=baseline, fingerprint=fp)
    return {"seg_id": seg_id, "init": baseline, "best": float(best_cost)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="ilc_5000_summary.json")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--top_n", type=int, default=100, help="worst N to re-optimize")
    parser.add_argument("--cost_threshold", type=float, default=None,
                        help="alternative to top_n: re-optimize all segments above this cost")
    parser.add_argument("--n_restarts", type=int, default=2)
    parser.add_argument("--cem_iters", type=int, default=30)
    parser.add_argument("--cem_pop", type=int, default=32)
    parser.add_argument("--ilc_iters", type=int, default=60)
    parser.add_argument("--ilc_lr", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cem_workers", type=int, default=2)
    args = parser.parse_args()

    # Pick worst segments
    s = json.load(open(args.summary))
    s = [r for r in s if "best" in r]
    s.sort(key=lambda r: r["best"], reverse=True)
    if args.cost_threshold is not None:
        targets = [r for r in s if r["best"] > args.cost_threshold]
    else:
        targets = s[:args.top_n]

    print(f"Targeting {len(targets)} segments (worst by current ILC cost)")
    print(f"Their cost range: {targets[-1]['best']:.2f}..{targets[0]['best']:.2f}")
    print(f"  contribution to total cost: {sum(r['best'] for r in targets)/sum(r['best'] for r in s)*100:.1f}%")

    work = [
        (str(Path(args.data_path) / f"{r['seg_id']}.csv"),
         args.n_restarts, args.cem_iters, args.cem_pop, args.cem_workers,
         args.ilc_iters, args.ilc_lr, args.out_dir)
        for r in targets
    ]

    t0 = time.time()
    results = process_map(_worker, work, max_workers=args.workers, chunksize=1)
    dt = time.time() - t0

    init = np.array([r["init"] for r in results])
    best = np.array([r["best"] for r in results])
    improved = (best < init - 1e-3).sum()
    print()
    print("=" * 60)
    print(f"Re-optimization done in {dt/60:.1f} min")
    print(f"  Before mean: {init.mean():7.2f}")
    print(f"  After  mean: {best.mean():7.2f}")
    print(f"  Improved:    {improved}/{len(results)}")


if __name__ == "__main__":
    main()
