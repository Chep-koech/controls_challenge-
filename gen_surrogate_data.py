"""Generate training data for the trajectory surrogate.

For each chosen training segment, run the real ONNX simulator on the ILC
action sequence plus several Gaussian perturbations, recording
(state_history, action_seq) -> lat_accel_trajectory tuples. The
perturbation set is intentionally diverse so the surrogate generalises
to the action regions where the optimizer wants to explore.
"""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
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

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX  # 400
TRAJ_LEN = 580  # we keep state for [0, TRAJ_LEN) - segments are ~580 long


def _smooth_noise(n_samples, n_action, noise_std, rng):
    """Generate smooth Gaussian noise via low-pass filter (so perturbations
    aren't pure white noise, closer to actions that real optimizers would
    propose)."""
    raw = rng.normal(0, 1, size=(n_samples, n_action))
    # 5-tap moving average for smoothness
    kern = np.ones(5) / 5.0
    smoothed = np.array([np.convolve(raw[i], kern, mode="same") for i in range(n_samples)])
    return smoothed * noise_std


def _gen_one_segment(args):
    csv_path, n_variants, noise_levels, model_path, opt_actions_path, seed = args
    rng = np.random.default_rng(seed)
    seg_id = Path(csv_path).stem

    # Load ILC actions for warm-start
    ilc = np.load(str(Path(opt_actions_path) / f"{seg_id}.npz"))["actions"].astype(np.float32)
    if len(ilc) < ACTION_HORIZON:
        ilc = np.pad(ilc, (0, ACTION_HORIZON - len(ilc)))

    # Load segment data once for state features
    df = pd.read_csv(csv_path)
    roll_lat = (np.sin(df["roll"].values) * ACC_G).astype(np.float32)
    v_ego = df["vEgo"].values.astype(np.float32)
    a_ego = df["aEgo"].values.astype(np.float32)
    target = df["targetLateralAcceleration"].values.astype(np.float32)
    steer = (-df["steerCommand"].values).astype(np.float32)  # sign flipped like sim does
    steer = np.nan_to_num(steer, nan=0.0)  # NaN for control window, irrelevant (we overwrite)
    T = len(df)

    # Set up sim model once per process (avoid per-call ONNX init)
    sim_model = TinyPhysicsModel(model_path, debug=False)

    # Generate variants
    all_actions = [ilc.copy()]  # variant 0 = ILC unperturbed
    for ns in noise_levels:
        if ns == 0:
            continue
        noise = _smooth_noise(n_variants, ACTION_HORIZON, ns, rng)
        for n in noise:
            all_actions.append(np.clip(ilc + n.astype(np.float32), -2.0, 2.0))
    all_actions = np.array(all_actions)  # (n_total, ACTION_HORIZON)

    # Run sim with each
    results = []
    for a in all_actions:
        controller = PlaybackController(action_seq=a)
        sim = TinyPhysicsSimulator(sim_model, csv_path, controller=controller, debug=False)
        cost = sim.rollout()
        lat_traj = np.array(sim.current_lataccel_history, dtype=np.float32)
        if len(lat_traj) < TRAJ_LEN:
            lat_traj = np.pad(lat_traj, (0, TRAJ_LEN - len(lat_traj)))
        results.append({
            "actions": a,
            "lat_traj": lat_traj[:TRAJ_LEN],
            "cost": cost["total_cost"],
        })

    return {
        "seg_id": seg_id,
        "roll_lataccel": roll_lat[:TRAJ_LEN],
        "v_ego": v_ego[:TRAJ_LEN],
        "a_ego": a_ego[:TRAJ_LEN],
        "target": target[:TRAJ_LEN],
        "pre_steer": steer[:CONTROL_START_IDX],
        "variants": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_per_level", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default="surrogate_data.npz")
    args = parser.parse_args()

    # Bias toward small-noise regions where the optimizer actually moves.
    # Heavy concentration around 0.01-0.05 noise; less at large noise.
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.15]

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    work = [(str(f), args.n_per_level, noise_levels, args.model_path, args.opt_actions, 1000 + i)
            for i, f in enumerate(files)]
    print(f"Generating data for {len(files)} segments, "
          f"{1 + (len(noise_levels)-1)*args.n_per_level} variants each "
          f"= {len(files) * (1 + (len(noise_levels)-1)*args.n_per_level)} samples")

    t0 = time.time()
    results = process_map(_gen_one_segment, work, max_workers=args.workers, chunksize=1)
    dt = time.time() - t0
    print(f"\nData gen done in {dt/60:.1f} min")

    # Flatten into arrays for compact storage
    seg_meta = []
    actions_buf = []
    lat_traj_buf = []
    state_buf = []  # (roll, vego, aego, target, pre_steer_pad) -> per timestep features
    cost_buf = []
    seg_idx = []  # which segment each sample belongs to

    for s_i, seg in enumerate(results):
        roll, vego, aego, target = seg["roll_lataccel"], seg["v_ego"], seg["a_ego"], seg["target"]
        pre = seg["pre_steer"]
        # Build per-segment state feature matrix (TRAJ_LEN, 4)
        state_buf.append(np.stack([roll, vego, aego, target], axis=-1))
        seg_meta.append({"seg_id": seg["seg_id"], "pre_steer": pre.tolist()})
        for v in seg["variants"]:
            actions_buf.append(v["actions"])
            lat_traj_buf.append(v["lat_traj"])
            cost_buf.append(v["cost"])
            seg_idx.append(s_i)

    actions_buf = np.array(actions_buf, dtype=np.float32)         # (N, 400)
    lat_traj_buf = np.array(lat_traj_buf, dtype=np.float32)       # (N, TRAJ_LEN)
    state_buf = np.array(state_buf, dtype=np.float32)             # (S, TRAJ_LEN, 4)
    cost_buf = np.array(cost_buf, dtype=np.float32)               # (N,)
    seg_idx = np.array(seg_idx, dtype=np.int32)                   # (N,)

    np.savez(
        args.out,
        actions=actions_buf,
        lat_traj=lat_traj_buf,
        states=state_buf,
        costs=cost_buf,
        seg_idx=seg_idx,
        pre_steer=np.array([m["pre_steer"] for m in seg_meta], dtype=np.float32),
        seg_ids=np.array([m["seg_id"] for m in seg_meta]),
    )

    print(f"Saved to {args.out}")
    print(f"  samples: {len(actions_buf)}")
    print(f"  costs:   mean={cost_buf.mean():.2f}  min={cost_buf.min():.2f}  max={cost_buf.max():.2f}")


if __name__ == "__main__":
    main()
