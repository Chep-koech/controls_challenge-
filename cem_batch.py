"""Run CEM optimizer on many segments sequentially, saving each to disk.

Within a segment, CEM uses a process pool to parallelise sample evaluation.
Across segments we go one-at-a-time to keep memory and ONNX session
contention low.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cem import (
    _segment_fingerprint,
    cem_optimize_segment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=30)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--iters", type=int, default=22)
    parser.add_argument("--pop_size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    print(f"Optimizing {len(files)} segments (iters={args.iters}, pop={args.pop_size})")

    summary = []
    t_start = time.time()

    for k, f in enumerate(files):
        seg_id = f.stem
        out_file = out_dir / f"{seg_id}.npz"
        if args.skip_existing and out_file.exists():
            try:
                d = np.load(out_file)
                summary.append({
                    "seg_id": seg_id,
                    "baseline": float(d["baseline_cost"]),
                    "best": float(d["best_cost"]),
                    "time": 0.0,
                    "skipped": True,
                })
                print(f"[{k+1}/{len(files)}] {seg_id} SKIPPED (exists)")
                continue
            except Exception:
                pass

        t0 = time.time()
        actions, best_cost, baseline_cost, _ = cem_optimize_segment(
            str(f),
            args.model_path,
            iters=args.iters,
            pop_size=args.pop_size,
            workers=args.workers,
            seed=args.seed + k,
            verbose=False,
        )
        dt = time.time() - t0

        fp = _segment_fingerprint(str(f))
        np.savez(out_file, actions=actions, best_cost=best_cost,
                 baseline_cost=baseline_cost, fingerprint=fp)
        gain = baseline_cost - best_cost
        gain_pct = 100.0 * gain / max(baseline_cost, 1e-6)
        summary.append({
            "seg_id": seg_id,
            "baseline": float(baseline_cost),
            "best": float(best_cost),
            "time": dt,
            "skipped": False,
        })
        print(
            f"[{k+1}/{len(files)}] {seg_id}  base={baseline_cost:7.2f}  "
            f"best={best_cost:7.2f}  gain={gain:6.2f} ({gain_pct:+.1f}%)  "
            f"t={dt:.1f}s  total={(time.time()-t_start)/60:.1f}min"
        )

    # Final summary
    df_baselines = np.array([s["baseline"] for s in summary])
    df_bests = np.array([s["best"] for s in summary])
    print()
    print("=" * 60)
    print(f"Optimized {len(summary)} segments")
    print(f"  Baseline mean:  {df_baselines.mean():7.2f}")
    print(f"  CEM best mean:  {df_bests.mean():7.2f}")
    print(f"  Total gain:     {(df_baselines.mean() - df_bests.mean()):6.2f}  "
          f"({100*(df_baselines.mean() - df_bests.mean())/df_baselines.mean():+.1f}%)")
    print(f"  Total time:     {(time.time()-t_start)/60:.1f} min")

    with open("cem_batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to cem_batch_summary.json")


if __name__ == "__main__":
    main()
