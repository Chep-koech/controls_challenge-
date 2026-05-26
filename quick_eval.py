"""Quick controller evaluation without matplotlib popups."""
import argparse
import sys
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
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    files = sorted(data_path.iterdir())[args.start:args.start + args.num_segs]
    rp = partial(run_rollout, controller_type=args.controller, model_path=args.model_path, debug=False)
    results = process_map(rp, files, max_workers=8, chunksize=4, disable=False)
    costs = [r[0] for r in results]
    df = pd.DataFrame(costs)
    print(f"Controller: {args.controller}")
    print(f"  num_segs: {len(files)}")
    print(f"  lataccel_cost: {df['lataccel_cost'].mean():.4f}")
    print(f"  jerk_cost:     {df['jerk_cost'].mean():.4f}")
    print(f"  total_cost:    {df['total_cost'].mean():.4f}")
    print(f"  median total:  {df['total_cost'].median():.4f}")


if __name__ == "__main__":
    main()
