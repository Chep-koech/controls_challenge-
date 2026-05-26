"""Tune controller parameters via random search."""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from tinyphysics import run_rollout


CONTROLLER_TEMPLATE = """from . import BaseController
import numpy as np


class Controller(BaseController):
    def __init__(self):
        self.p = {p}
        self.i = {i}
        self.d = {d}
        self.kff_rate = {kff_rate}
        self.kff_preview = {kff_preview}
        self.lookahead = {lookahead}
        self.alpha = {alpha}
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        ff_rate = 0.0
        ff_preview = 0.0
        n = min(self.lookahead, len(future_plan.lataccel))
        if n > 0:
            fut_lat = np.asarray(future_plan.lataccel[:n], dtype=np.float64)
            fut_roll = np.asarray(future_plan.roll_lataccel[:n], dtype=np.float64)
            future_avg = float(np.mean(fut_lat - fut_roll))
            compensated_now = target_lataccel - state.roll_lataccel
            ff_rate = self.kff_rate * (future_avg - compensated_now)
            ff_preview = self.kff_preview * future_avg

        raw = (
            self.p * error
            + self.i * self.error_integral
            + self.d * error_diff
            + ff_rate
            + ff_preview
        )

        filtered = self.alpha * raw + (1.0 - self.alpha) * self.prev_filtered
        self.prev_filtered = filtered
        return float(filtered)
"""


def eval_params(params, files, model_path):
    code = CONTROLLER_TEMPLATE.format(**params)
    with open("controllers/_tune_tmp.py", "w") as f:
        f.write(code)
    rp = partial(run_rollout, controller_type="_tune_tmp", model_path=model_path, debug=False)
    results = process_map(rp, files, max_workers=8, chunksize=4, disable=True)
    costs = [r[0]["total_cost"] for r in results]
    return float(np.mean(costs)), float(np.median(costs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_segs", type=int, default=200)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="tune_result.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    files = sorted(Path("data").iterdir())[: args.num_segs]
    model_path = "./models/tinyphysics.onnx"

    base = {
        "p": 0.397,
        "i": 0.080,
        "d": -0.007,
        "kff_rate": 0.198,
        "kff_preview": 0.255,
        "lookahead": 6,
        "alpha": 0.394,
    }
    ranges = {
        "p": (0.25, 0.55),
        "i": (0.04, 0.16),
        "d": (-0.10, 0.05),
        "kff_rate": (0.05, 0.40),
        "kff_preview": (0.10, 0.40),
        "lookahead": (3, 10),
        "alpha": (0.25, 0.65),
    }

    best = None
    best_cost = float("inf")
    history = []

    print(f"Tuning across {args.num_segs} segs, {args.iters} iters")
    for it in range(args.iters):
        if it == 0:
            params = dict(base)
        else:
            params = {}
            for k, (lo, hi) in ranges.items():
                if best is not None and np.random.random() < 0.75:
                    span = (hi - lo) * 0.15
                    val = best[k] + np.random.normal(0, span)
                else:
                    val = np.random.uniform(lo, hi)
                val = float(np.clip(val, lo, hi))
                if k == "lookahead":
                    val = int(round(val))
                params[k] = val

        t0 = time.time()
        mean_cost, med_cost = eval_params(params, files, model_path)
        dt = time.time() - t0
        history.append({"params": params, "mean": mean_cost, "median": med_cost})

        marker = ""
        if mean_cost < best_cost:
            best_cost = mean_cost
            best = params
            marker = "  <-- NEW BEST"
        print(f"[{it+1:02d}/{args.iters}] mean={mean_cost:7.2f}  med={med_cost:7.2f}  ({dt:.1f}s){marker}")
        sys.stdout.flush()

    print(f"\nBest mean cost: {best_cost:.3f}")
    print(f"Best params: {json.dumps(best, indent=2)}")
    with open(args.out, "w") as f:
        json.dump({"best_cost": best_cost, "best_params": best, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
