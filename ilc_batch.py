"""Batch-optimize many segments via ILC, parallelised across segments.

Unlike CEM (which is parallel WITHIN a segment), ILC is intrinsically
sequential per segment but each segment is cheap (~6-8 rollouts). So we
parallelise across segments using process_map.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from cem import _segment_fingerprint
from ilc import ilc_optimize_segment


def _worker(
    csv_path: str,
    model_path: str,
    out_dir: str,
    iters: int,
    lr: float,
    error_smooth: int,
    action_smooth: int,
    delay: int,
    skip_existing: bool,
):
    seg_id = Path(csv_path).stem
    out_file = Path(out_dir) / f"{seg_id}.npz"

    if skip_existing and out_file.exists():
        try:
            d = np.load(out_file)
            return {
                "seg_id": seg_id,
                "baseline": float(d["baseline_cost"]),
                "best": float(d["best_cost"]),
                "time": 0.0,
                "skipped": True,
            }
        except Exception:
            pass

    t0 = time.time()
    try:
        actions, best_cost, baseline_cost, _ = ilc_optimize_segment(
            csv_path,
            model_path,
            n_iters=iters,
            lr=lr,
            delay=delay,
            error_smooth=error_smooth,
            action_smooth=action_smooth,
            verbose=False,
        )
        dt = time.time() - t0
        fp = _segment_fingerprint(csv_path)
        np.savez(out_file, actions=actions, best_cost=best_cost,
                 baseline_cost=baseline_cost, fingerprint=fp)
        return {
            "seg_id": seg_id,
            "baseline": float(baseline_cost),
            "best": float(best_cost),
            "time": dt,
            "skipped": False,
        }
    except Exception as e:
        return {"seg_id": seg_id, "error": str(e), "time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--error_smooth", type=int, default=3)
    parser.add_argument("--action_smooth", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--summary_out", default="ilc_batch_summary.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"ILC on {len(files)} segments  workers={args.workers}  iters={args.iters}  lr={args.lr}")

    fn = partial(
        _worker,
        model_path=args.model_path,
        out_dir=str(out_dir),
        iters=args.iters,
        lr=args.lr,
        error_smooth=args.error_smooth,
        action_smooth=args.action_smooth,
        delay=args.delay,
        skip_existing=args.skip_existing,
    )

    t0 = time.time()
    results = process_map(fn, files, max_workers=args.workers, chunksize=2, disable=False)
    dt = time.time() - t0

    ok = [r for r in results if "error" not in r]
    if not ok:
        print("All segments failed!")
        return
    baselines = np.array([r["baseline"] for r in ok])
    bests = np.array([r["best"] for r in ok])

    print()
    print("=" * 60)
    print(f"ILC done: {len(ok)}/{len(results)} succeeded  total_time={dt/60:.1f}min")
    print(f"  Baseline (best.py) mean: {baselines.mean():7.2f}")
    print(f"  ILC best mean:           {bests.mean():7.2f}")
    print(f"  Improvement:             {(baselines.mean() - bests.mean()):6.2f}  "
          f"({100*(baselines.mean() - bests.mean())/baselines.mean():+.1f}%)")
    print(f"  Median ILC cost:         {np.median(bests):7.2f}")

    with open(args.summary_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {args.summary_out}")


if __name__ == "__main__":
    main()
