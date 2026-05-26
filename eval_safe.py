"""Robust 5000-seg eval using multiprocessing.Pool with initializer.

Replaces tinyphysics.py's process_map (which has hung on Windows after
extensive prior compute). Saves incremental progress to a file so we can
always recover partial results.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Per-worker state
_MODEL_PATH = None
_CONTROLLER = None


def _init(model_path, controller_type):
    global _MODEL_PATH, _CONTROLLER
    _MODEL_PATH = model_path
    _CONTROLLER = controller_type


def _rollout(csv_path):
    import importlib
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    model = TinyPhysicsModel(_MODEL_PATH, debug=False)
    ctrl = importlib.import_module(f"controllers.{_CONTROLLER}").Controller()
    sim = TinyPhysicsSimulator(model, str(csv_path), controller=ctrl, debug=False)
    cost = sim.rollout()
    return {
        "seg_id": Path(csv_path).stem,
        "lataccel_cost": float(cost["lataccel_cost"]),
        "jerk_cost": float(cost["jerk_cost"]),
        "total_cost": float(cost["total_cost"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--test_controller", required=True)
    parser.add_argument("--baseline_controller", default="pid")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out_prefix", default="eval_safe")
    args = parser.parse_args()

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"Evaluating {len(files)} segments  workers={args.workers}")

    results = {}
    for ctrl in [args.baseline_controller, args.test_controller]:
        print(f"\n=== {ctrl} ===")
        t0 = time.time()
        with mp.Pool(processes=args.workers, initializer=_init,
                     initargs=(args.model_path, ctrl)) as pool:
            costs = []
            n_done = 0
            for r in pool.imap_unordered(_rollout, files, chunksize=4):
                costs.append(r)
                n_done += 1
                if n_done % 200 == 0:
                    elapsed = time.time() - t0
                    eta = (elapsed / n_done) * (len(files) - n_done) if n_done else 0
                    print(f"  [{n_done}/{len(files)}]  elapsed={elapsed/60:.1f}min  "
                          f"eta={eta/60:.1f}min", flush=True)
        df = pd.DataFrame(costs)
        results[ctrl] = df
        df.to_csv(f"{args.out_prefix}_{ctrl}.csv", index=False)
        print(f"  done in {(time.time()-t0)/60:.1f} min")
        print(f"  lataccel_cost: {df['lataccel_cost'].mean():.4f}")
        print(f"  jerk_cost:     {df['jerk_cost'].mean():.4f}")
        print(f"  total_cost:    {df['total_cost'].mean():.4f}")
        print(f"  median total:  {df['total_cost'].median():.4f}")

    base = results[args.baseline_controller]['total_cost'].mean()
    test = results[args.test_controller]['total_cost'].mean()
    print()
    print("=" * 60)
    print(f"baseline ({args.baseline_controller}): {base:.4f}")
    print(f"test     ({args.test_controller}): {test:.4f}")
    print(f"improvement: {base-test:+.4f}  ({(base-test)/base*100:+.2f}%)")


if __name__ == "__main__":
    main()
