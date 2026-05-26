"""Fit a linear ARX model from existing surrogate training data.

Model:
    a_y[t+1] = α₁·a_y[t] + α₂·a_y[t-1] + β·action[t] + γ·roll[t] + δ·v_ego[t] + bias

Least-squares fit on the surrogate_data_v2.npz dataset which has 17500
(action_seq, lat_traj) pairs from real-sim rollouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="surrogate_data_v2.npz")
    parser.add_argument("--out", default="arx_model.json")
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=True)
    actions = d["actions"].astype(np.float64)        # (N, 400)
    lat_traj = d["lat_traj"].astype(np.float64)      # (N, TRAJ_LEN)
    states = d["states"].astype(np.float64)          # (S, TRAJ_LEN, 4)  -- roll, vego, aego, target
    seg_idx = d["seg_idx"].astype(np.int64)          # (N,)

    CONTROL_START = 100
    COST_END = 500
    HORIZON = COST_END - CONTROL_START  # 400

    # Build feature matrix X and target y
    # For each sample n, for each step i in [0, 399], we have:
    #   a_y[t] = lat_traj[n, CONTROL_START + i]
    #   a_y[t-1] = lat_traj[n, CONTROL_START + i - 1]
    #   action[t] = actions[n, i]
    #   roll[t] = states[seg_idx[n], CONTROL_START + i, 0]
    #   v[t] = states[seg_idx[n], CONTROL_START + i, 1]
    # Target: a_y[t+1] = lat_traj[n, CONTROL_START + i + 1]
    rows = []
    targets = []
    for n in range(len(actions)):
        sidx = seg_idx[n]
        lat = lat_traj[n]
        st = states[sidx]
        for i in range(HORIZON - 1):
            t = CONTROL_START + i
            rows.append([
                lat[t],              # a_y[t]
                lat[t - 1],          # a_y[t-1]
                actions[n, i],       # action[t]
                st[t, 0],            # roll[t]
                st[t, 1],            # v_ego[t]
                1.0,                 # bias
            ])
            targets.append(lat[t + 1])
        if n % 2000 == 0:
            print(f"  building rows: {n}/{len(actions)}")

    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Least squares: y = X @ θ
    theta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ theta
    err = y - pred
    rmse = float(np.sqrt((err ** 2).mean()))
    r2 = float(1.0 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum())

    print(f"ARX model coefficients:")
    names = ["a_y[t]", "a_y[t-1]", "action[t]", "roll[t]", "v_ego[t]", "bias"]
    for nm, v in zip(names, theta):
        print(f"  {nm:>12}:  {v:+.6f}")
    print(f"RMSE: {rmse:.5f}  R²: {r2:.5f}")

    with open(args.out, "w") as f:
        json.dump({
            "coefficients": dict(zip(names, theta.tolist())),
            "rmse": rmse,
            "r2": r2,
        }, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
