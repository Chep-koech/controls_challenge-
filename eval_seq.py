"""Sequential evaluation, no multiprocessing.

Slower (~40 min for 5000 segs single controller) but reliable on this
Windows system where multiprocessing has been crashing.
"""
import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    print(f"Evaluating {args.controller} on {len(files)} segments (sequential)")

    # Load ONNX model once
    sim_model = TinyPhysicsModel(args.model_path, debug=False)
    rows = []
    t0 = time.time()
    for i, csv in enumerate(files):
        ctrl = importlib.import_module(f"controllers.{args.controller}").Controller()
        sim = TinyPhysicsSimulator(sim_model, str(csv), controller=ctrl, debug=False)
        cost = sim.rollout()
        rows.append({
            "seg_id": csv.stem,
            "lataccel_cost": float(cost["lataccel_cost"]),
            "jerk_cost": float(cost["jerk_cost"]),
            "total_cost": float(cost["total_cost"]),
        })
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (len(files) - i - 1)
            df = pd.DataFrame(rows)
            print(f"  [{i+1}/{len(files)}]  "
                  f"mean={df['total_cost'].mean():.3f}  "
                  f"median={df['total_cost'].median():.3f}  "
                  f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
    print()
    print("=" * 50)
    print(f"Controller: {args.controller}")
    print(f"  n_segs:        {len(df)}")
    print(f"  lataccel_cost: {df['lataccel_cost'].mean():.4f}")
    print(f"  jerk_cost:     {df['jerk_cost'].mean():.4f}")
    print(f"  total_cost:    {df['total_cost'].mean():.4f}")
    print(f"  median total:  {df['total_cost'].median():.4f}")


if __name__ == "__main__":
    main()
