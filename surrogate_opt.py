"""Optimize per-segment action sequence via the trained surrogate.

The surrogate is a non-autoregressive TCN that maps (per-step features,
including action) -> lat_accel trajectory. Because it has no chained
softmax sampling, gradients through it are smooth and Adam works.

Per-segment workflow:
  1. Build per-step features from segment data + initial ILC actions.
  2. Mark `feats[:, action_channel, control_window]` as requires_grad.
  3. Adam-step the action slice to minimise the surrogate cost.
  4. Validate the resulting actions on the REAL simulator.
  5. Take whichever of (initial ILC, surrogate-optimized) gives the
     lower REAL cost. Save.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from train_surrogate import (
    ACTION_HORIZON,
    TRAJ_LEN,
    N_FEATURES,
    LAT_COST_MULT,
    DEL_T,
    TrajSurrogate,
    compute_cost,
)


def load_surrogate(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TrajSurrogate(hidden=ckpt["hidden"], n_blocks=ckpt["n_blocks"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    state_mean = torch.from_numpy(ckpt["state_mean"])
    state_std = torch.from_numpy(ckpt["state_std"])
    return model, state_mean, state_std


def build_features(csv_path, action_seq, state_mean, state_std):
    df = pd.read_csv(csv_path)
    T = min(len(df), TRAJ_LEN)
    roll_lat = (np.sin(df["roll"].values[:T]) * ACC_G).astype(np.float32)
    v_ego = df["vEgo"].values[:T].astype(np.float32)
    a_ego = df["aEgo"].values[:T].astype(np.float32)
    target = df["targetLateralAcceleration"].values[:T].astype(np.float32)
    pre_steer = (-df["steerCommand"].values[:CONTROL_START_IDX]).astype(np.float32)
    pre_steer = np.nan_to_num(pre_steer, nan=0.0)
    pre_steer = np.clip(pre_steer, -2.0, 2.0)

    feats = torch.zeros((TRAJ_LEN, N_FEATURES), dtype=torch.float32)
    feats[:CONTROL_START_IDX, 0] = torch.from_numpy(pre_steer)
    # action window will be set externally
    action_t = torch.tensor(action_seq, dtype=torch.float32)
    feats[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON, 0] = action_t
    state_block = np.stack([roll_lat, v_ego, a_ego, target], axis=-1)
    if state_block.shape[0] < TRAJ_LEN:
        state_block = np.pad(state_block, ((0, TRAJ_LEN - state_block.shape[0]), (0, 0)))
    s_norm = (state_block - state_mean.numpy()) / state_std.numpy()
    feats[:, 1:] = torch.from_numpy(s_norm.astype(np.float32))
    return feats, torch.from_numpy(target[:TRAJ_LEN]) if T == TRAJ_LEN else torch.from_numpy(
        np.pad(target, (0, TRAJ_LEN - T)).astype(np.float32)
    )


def real_cost(sim_model, csv_path, actions):
    controller = PlaybackController(action_seq=actions)
    sim = TinyPhysicsSimulator(sim_model, csv_path, controller=controller, debug=False)
    return float(sim.rollout()["total_cost"])


def optimize_one(args):
    csv_path, opt_actions_dir, model_path, surrogate_state, n_steps, lr = args
    surrogate, state_mean, state_std = surrogate_state

    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    init_actions = np.load(str(Path(opt_actions_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
    if len(init_actions) < ACTION_HORIZON:
        init_actions = np.pad(init_actions, (0, ACTION_HORIZON - len(init_actions)))

    sim_model = TinyPhysicsModel(model_path, debug=False)
    init_real = real_cost(sim_model, csv_path, init_actions)

    # Build static features (everything except the action window)
    feats_init, target_seq = build_features(csv_path, init_actions, state_mean, state_std)
    target_cost_window = target_seq[CONTROL_START_IDX:COST_END_IDX]

    # Optimisable action vector
    a = torch.tensor(init_actions, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)

    best_surrogate_cost = float("inf")
    best_actions = init_actions.copy()
    best_real_cost = init_real
    history = []

    for it in range(n_steps):
        opt.zero_grad(set_to_none=True)
        # Inject current actions into features
        feats = feats_init.clone()
        feats[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON, 0] = a
        feats_b = feats.unsqueeze(0)  # (1, TRAJ_LEN, 5)
        pred = surrogate(feats_b)  # (1, ACTION_HORIZON)
        cost = compute_cost(pred, target_cost_window.unsqueeze(0)).squeeze()
        cost.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-2.0, 2.0)
        history.append(float(cost.detach()))

        # Periodic real-sim validation
        if (it + 1) % 5 == 0 or it == n_steps - 1:
            candidate = a.detach().cpu().numpy().astype(np.float32)
            rc = real_cost(sim_model, csv_path, candidate)
            if rc < best_real_cost - 1e-3:
                best_real_cost = rc
                best_actions = candidate.copy()

    return {
        "seg_id": seg_id,
        "init_cost": float(init_real),
        "best_cost": float(best_real_cost),
        "best_actions": best_actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--surrogate", default="surrogate.pt")
    parser.add_argument("--out_dir", default="optimized_actions_v2")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load surrogate once (each worker also loads it independently)
    print("Loading surrogate...")
    surrogate_state = load_surrogate(args.surrogate)
    print("Loaded")

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"Optimising {len(files)} segments  workers={args.workers}  steps={args.n_steps}  lr={args.lr}")

    work = [(f, args.opt_actions, args.model_path, surrogate_state, args.n_steps, args.lr) for f in files]

    t0 = time.time()
    results = process_map(optimize_one, work, max_workers=args.workers, chunksize=2, disable=False)
    dt = time.time() - t0

    # Save per-segment results
    from cem import _segment_fingerprint
    for r in results:
        if "best_actions" not in r:
            continue
        seg_id = r["seg_id"]
        csv = str(Path(args.data_path) / f"{seg_id}.csv")
        fp = _segment_fingerprint(csv)
        np.savez(
            out_dir / f"{seg_id}.npz",
            actions=r["best_actions"],
            best_cost=r["best_cost"],
            baseline_cost=r["init_cost"],
            fingerprint=fp,
        )

    init_costs = np.array([r["init_cost"] for r in results])
    best_costs = np.array([r["best_cost"] for r in results])
    print()
    print("=" * 60)
    print(f"Surrogate opt done in {dt/60:.1f} min")
    print(f"  init (ILC) mean:        {init_costs.mean():7.2f}")
    print(f"  surrogate-opt mean:     {best_costs.mean():7.2f}")
    print(f"  improvement:            {(init_costs.mean()-best_costs.mean()):+.2f}  "
          f"({100*(init_costs.mean()-best_costs.mean())/init_costs.mean():+.1f}%)")
    print(f"  segments improved:      {(best_costs < init_costs - 1e-3).sum()}/{len(results)}")


if __name__ == "__main__":
    main()
