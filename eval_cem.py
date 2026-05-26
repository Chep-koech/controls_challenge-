"""Evaluate cem_playback vs pid on a specific set of segments.

Useful for confirming CEM-optimized segments score better than baseline,
without running the full 5000-segment official eval.
"""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from tinyphysics import run_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--optimized_dir", default="optimized_actions")
    parser.add_argument("--num_segs", type=int, default=0,
                        help="If >0, restrict to first N optimized segments; else all")
    args = parser.parse_args()

    # Take ONLY segments we have optimized actions for (others fall back to best.py)
    opt_dir = Path(args.optimized_dir)
    seg_ids = sorted(p.stem for p in opt_dir.glob("*.npz"))
    if args.num_segs > 0:
        seg_ids = seg_ids[: args.num_segs]
    data_dir = Path(args.data_path)
    files = [data_dir / f"{s}.csv" for s in seg_ids]
    files = [f for f in files if f.exists()]

    print(f"Evaluating {len(files)} optimized segments...")

    rp_test = partial(run_rollout, controller_type="cem_playback",
                      model_path=args.model_path, debug=False)
    rp_base = partial(run_rollout, controller_type="pid",
                      model_path=args.model_path, debug=False)
    rp_best = partial(run_rollout, controller_type="best",
                      model_path=args.model_path, debug=False)

    for name, fn in [("pid", rp_base), ("best", rp_best), ("cem_playback", rp_test)]:
        results = process_map(fn, files, max_workers=8, chunksize=2, disable=True)
        costs = [r[0] for r in results]
        df = pd.DataFrame(costs)
        print(f"\nController: {name}")
        print(f"  lataccel_cost:  {df['lataccel_cost'].mean():.4f}")
        print(f"  jerk_cost:      {df['jerk_cost'].mean():.4f}")
        print(f"  total_cost:     {df['total_cost'].mean():.4f}")
        print(f"  median total:   {df['total_cost'].median():.4f}")


if __name__ == "__main__":
    main()
