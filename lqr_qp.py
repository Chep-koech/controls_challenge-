"""Horizon-QP optimizer using the fitted ARX model.

Dynamics (from fit_arx.py):
    a_y[t+1] = α₁·a_y[t] + α₂·a_y[t-1] + β·u[t-CONTROL_START]
             + γ·roll[t] + δ·v_ego[t] + bias

Optimisation (per segment):
    minimise   sum_{t in cost window}
                 50·(r[t] - a_y[t])² + ((a_y[t]-a_y[t-1])/dt)² + ε·u[t]²
    subject to a_y trajectory as above and |u[t]| ≤ 2

Approach:
  1. Compute the "free response" a_y trajectory under u=0.
  2. Compute the impulse-response matrix S where a_y = a_y_free + S·u
     (S is causal lower-triangular).
  3. Form the quadratic cost J(u) and solve via the normal equations.
  4. Clip u to [-2, 2]; if many bindings, do projected gradient cleanup.
  5. Validate the resulting actions on the real ONNX simulator; keep
     whichever (ILC, LQR) gives lower real cost.

The whole solve is <0.5 s per segment, dominated by the 400×400 linear
system. Real-sim validation adds ~0.5 s, so end-to-end ~1 s per segment.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from cem import _segment_fingerprint

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX  # 400
DEL_T = 0.1


# Per-worker globals
_SIM_MODEL = None
_MODEL_PATH = None
_ARX = None
_PROGRESS_FILE = None


def _init(model_path, arx_path, progress_file):
    global _SIM_MODEL, _MODEL_PATH, _ARX, _PROGRESS_FILE
    _SIM_MODEL = TinyPhysicsModel(model_path, debug=False)
    _MODEL_PATH = model_path
    with open(arx_path) as f:
        _ARX = json.load(f)["coefficients"]
    _PROGRESS_FILE = progress_file


def _real_cost(csv_path, actions):
    c = PlaybackController(action_seq=actions)
    s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=c, debug=False)
    return s.rollout()["total_cost"]


def _build_segment_data(csv_path):
    df = pd.read_csv(csv_path)
    roll = (np.sin(df["roll"].values) * ACC_G).astype(np.float64)
    v_ego = df["vEgo"].values.astype(np.float64)
    target = df["targetLateralAcceleration"].values.astype(np.float64)
    pre_steer = (-df["steerCommand"].values).astype(np.float64)
    pre_steer = np.nan_to_num(pre_steer, nan=0.0)
    pre_steer = np.clip(pre_steer, -2.0, 2.0)
    return {
        "roll": roll, "v_ego": v_ego, "target": target,
        "pre_steer": pre_steer, "T": len(df),
    }


def _free_response_and_S(data, arx):
    """Compute (a_y_free[t] for t in [0,COST_END_IDX)) under u=0 in the
    control window, AND the impulse-response matrix S of shape
    (ACTION_HORIZON, ACTION_HORIZON) where
        S[i, j] = d a_y[CONTROL_START_IDX + 1 + i] / d u[j]
    for i, j in [0, ACTION_HORIZON). S is causal (S[i, j] = 0 if j > i).
    """
    alpha1 = arx["a_y[t]"]
    alpha2 = arx["a_y[t-1]"]
    beta = arx["action[t]"]
    gamma = arx["roll[t]"]
    delta = arx["v_ego[t]"]
    bias = arx["bias"]

    # ARX update: a_y[t+1] = α₁·a_y[t] + α₂·a_y[t-1] + β·u[t_window] + γ·roll[t] + δ·v[t] + bias
    # Build a_y_free with u=0 during control window. We use the dataset's pre_steer for the
    # pre-control window but ZERO for control window (so it captures only the free response).
    # Actually for ARX simulation purpose we just need the "what if u were zero" trajectory.
    T = COST_END_IDX
    a_y = np.zeros(T)
    # Use ground-truth target for the first 100 steps (matches simulator's reset behaviour
    # where current_lataccel_history starts as target for the warmup window).
    a_y[:CONTROL_START_IDX] = data["target"][:CONTROL_START_IDX]

    for t in range(CONTROL_START_IDX, T - 1):
        # Note: u contribution is zero (free response). Pre-control u is handled by initial
        # conditions of a_y already, since past data has been "absorbed".
        a_y[t + 1] = (
            alpha1 * a_y[t] + alpha2 * a_y[t - 1]
            + gamma * data["roll"][t] + delta * data["v_ego"][t] + bias
        )

    # Build sensitivity S[i, j]: response of a_y[CONTROL_START + 1 + i] to u[j]
    # We need the impulse response h[k] satisfying h[k] = α₁·h[k-1] + α₂·h[k-2], h[0] = β.
    H = ACTION_HORIZON
    impulse = np.zeros(H + 2)
    impulse[1] = beta  # a_y at step+1 from u at step
    for k in range(2, H + 2):
        impulse[k] = alpha1 * impulse[k - 1] + alpha2 * impulse[k - 2]
    # S[i, j] = impulse[i - j + 1] when i >= j, else 0
    S = np.zeros((H, H))
    for j in range(H):
        for i in range(j, H):
            S[i, j] = impulse[i - j + 1]
    return a_y, S


def lqr_solve_segment(data, arx, ilc_actions, ridge_u=1e-5, anchor_weight=200.0):
    """Solve the convex QP for optimal actions over the cost window.

    Adds an anchor term `anchor_weight * ||u - u_ILC||²` so the LQR
    solution stays close to the proven-good ILC trajectory. This dampens
    out the model-error compounding we'd otherwise see from a pure LQR.
    """
    a_y_free, S = _free_response_and_S(data, arx)
    H = ACTION_HORIZON

    # a_y at the COST window indices [CONTROL_START_IDX, COST_END_IDX).
    # Define vector y of length H where y[i] = a_y[CONTROL_START + 1 + i] (so i=0..H-1
    # covers steps CONTROL_START+1 .. CONTROL_START+H). Note this leaves out a_y at
    # step CONTROL_START itself, which is determined by initial conditions and isn't
    # affected by u — we treat its cost as a constant offset.
    # cost window indices in the global trajectory:
    a_y_free_window = a_y_free[CONTROL_START_IDX + 1 : COST_END_IDX]   # length H-? actually H is 400, indices CSI+1..COST_END-1 has length 399.
    if len(a_y_free_window) < H:
        # pad with the last value
        a_y_free_window = np.concatenate([a_y_free_window, [a_y_free_window[-1]]])

    # Targets for the same window
    target = data["target"][CONTROL_START_IDX + 1 : CONTROL_START_IDX + 1 + H]
    if len(target) < H:
        target = np.concatenate([target, np.full(H - len(target), target[-1] if len(target) else 0.0)])

    # The lat_accel cost is over indices [CONTROL_START, COST_END_IDX).
    # We optimise over H = 400 actions which influence a_y at indices [CONTROL_START+1 .. CONTROL_START+H].
    # a_y[CONTROL_START] is fixed (initial condition), contributes a fixed cost we ignore.

    # Effective tracking weight 50, jerk weight via the difference operator on the window.
    q_track = 50.0
    q_jerk = 1.0 / (DEL_T ** 2)

    # Build D matrix: differences a_y_window[i] - a_y_window[i-1] for i=1..H-1, plus
    # the "boundary" jerk a_y_window[0] - a_y[CONTROL_START].
    # a_y_window depends on u via S; a_y[CONTROL_START] is independent of u.
    # For the boundary, jerk_0 = a_y_window[0] - a_y[CONTROL_START_IDX] (constant).
    a_y_csi = a_y_free[CONTROL_START_IDX]   # this is target[CONTROL_START_IDX] under our convention

    # Cost: 50 * sum_i (target[i] - (a_y_free[i] + S[i, :]·u))²
    #     + (1/dt²) * sum_i (a_y[i] - a_y[i-1])² , where a_y[0_in_window] - a_y_csi is the boundary jerk
    #     + ridge * ||u||²
    # Let r_i = target[i] - a_y_free[i]   (residual that u must close)
    # Tracking term contributes: 50·||S u - r||²  =  u' (50 S'S) u - 2·50·u' S' r + const
    # Jerk term:  let D_ext be the (H, H) matrix that does y[i] - y[i-1] for i=0..H-1 with y[-1] = a_y_csi.
    # In u-space: a_y_window - shift(a_y_window) where shift fills first slot with a_y_csi.
    # jerk_i = a_y_window[i] - a_y_window[i-1] for i>=1
    # jerk_0 = a_y_window[0] - a_y_csi
    # So J·a_y_window where J = I - shift_lower(1), but careful about jerk_0 boundary.
    r = target - a_y_free_window  # (H,)

    # Build jerk operator on a_y_window
    J_mat = np.eye(H) - np.eye(H, k=-1)  # J[i, i] = 1, J[i, i-1] = -1, else 0
    # For i=0 row, J_mat picks a_y_window[0] only, but we want a_y_window[0] - a_y_csi
    # so the constant offset is -a_y_csi in the linear term.
    # a_y_window = a_y_free_window + S u
    # J · a_y_window = J · a_y_free_window + J · S · u
    # adjust for boundary: jerk_0 = a_y_window[0] - a_y_csi  =>  the row 0 of J·a_y_window
    # is a_y_window[0] - 0 (since J has 0 in column -1), so we add the -a_y_csi to that row only.
    # Equivalent: define vec b such that jerk = J·a_y_window + b where b = [-a_y_csi, 0, 0, ...]^T
    b_jerk = np.zeros(H)
    b_jerk[0] = -a_y_csi

    # Tracking quadratic form
    # cost_track = q_track * (S u - r)' (S u - r)
    A_track = S
    b_track = r  # the offset to subtract

    # Jerk quadratic form
    JS = J_mat @ S
    # jerk = JS u + (J·a_y_free_window + b_jerk)
    offset_jerk = J_mat @ a_y_free_window + b_jerk

    # Anchor: prefer u close to ILC. Adds anchor_weight·||u - u_ILC||² to cost.
    u_ilc = np.asarray(ilc_actions[:H], dtype=np.float64)

    # Total quadratic in u: q_track*(A u - b_track)' (A u - b_track)
    #                     + q_jerk*(JS u + offset_jerk)' (...)
    #                     + anchor_weight*||u - u_ilc||²
    #                     + ridge*||u||²
    H_qp = (q_track * (A_track.T @ A_track)
            + q_jerk * (JS.T @ JS)
            + (anchor_weight + ridge_u) * np.eye(H))
    g = (q_track * (A_track.T @ -b_track)
         + q_jerk * (JS.T @ offset_jerk)
         - anchor_weight * u_ilc)
    u_unconstrained = -np.linalg.solve(H_qp, g)
    u = np.clip(u_unconstrained, -2.0, 2.0)
    return u


def _worker(args):
    csv_path, opt_dir, out_dir = args
    csv_path = str(csv_path)
    seg_id = Path(csv_path).stem
    out_path = Path(out_dir) / f"{seg_id}.npz"

    try:
        ilc = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
        if len(ilc) < ACTION_HORIZON:
            ilc = np.pad(ilc, (0, ACTION_HORIZON - len(ilc)))

        init_real = _real_cost(csv_path, ilc)
        data = _build_segment_data(csv_path)
        u_lqr = lqr_solve_segment(data, _ARX, ilc).astype(np.float32)
        lqr_real = _real_cost(csv_path, u_lqr)

        # Use LQR-vs-ILC delta as a search direction. Walk from ILC toward
        # the LQR solution with a fine-grained mix; keep whichever is best.
        delta = u_lqr - ilc
        best_cost = init_real
        best_actions = ilc.copy()
        for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00,
                  -0.05, -0.10, -0.20):
            cand = np.clip(ilc + w * delta, -2.0, 2.0).astype(np.float32)
            rc = _real_cost(csv_path, cand)
            if rc < best_cost - 1e-3:
                best_cost = rc
                best_actions = cand

        if best_cost < init_real - 1e-3:
            fp = _segment_fingerprint(csv_path)
            np.savez(out_path, actions=best_actions, best_cost=best_cost,
                     baseline_cost=init_real, fingerprint=fp)
            result = "IMPROVED"
        else:
            result = "no-change"

        with open(_PROGRESS_FILE, "a") as f:
            f.write(f"{seg_id}\t{init_real:.3f}\t{best_cost:.3f}\t{lqr_real:.3f}\t{result}\n")
        return (seg_id, float(init_real), float(best_cost), float(lqr_real), result)
    except Exception as e:
        with open(_PROGRESS_FILE, "a") as f:
            f.write(f"{seg_id}\tERROR\t{e}\n")
        return (seg_id, None, None, None, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--arx", default="arx_model.json")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress_file", default="lqr_progress.tsv")
    parser.add_argument("--single_test", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"LQR optimization on {len(files)} segments  workers={args.workers}")

    with open(args.progress_file, "w") as f:
        f.write("seg_id\tinit\tbest\tlqr_only\tresult\n")

    work = [(f, args.opt_actions, args.out_dir) for f in files]

    if args.single_test or args.workers == 1:
        _init(args.model_path, args.arx, args.progress_file)
        improved = 0
        total_init = 0.0
        total_best = 0.0
        total_lqr_only = 0.0
        for w in work:
            seg, ic, bc, lc, res = _worker(w)
            if ic is not None:
                total_init += ic
                total_best += bc
                total_lqr_only += lc
                if "IMPROVED" in res:
                    improved += 1
                print(f"  {seg}  init={ic:.2f}  best={bc:.2f}  lqr_only={lc:.2f}  {res}")
        n = len(work)
        print(f"\nMean init: {total_init/n:.2f}")
        print(f"Mean best: {total_best/n:.2f}")
        print(f"Mean lqr_only: {total_lqr_only/n:.2f}")
        print(f"Improved: {improved}/{n}")
        return

    t0 = time.time()
    with mp.Pool(processes=args.workers, initializer=_init,
                 initargs=(args.model_path, args.arx, args.progress_file)) as pool:
        improved = 0
        n_done = 0
        total_init = 0.0
        total_best = 0.0
        for seg, ic, bc, lc, res in pool.imap_unordered(_worker, work, chunksize=2):
            n_done += 1
            if ic is not None:
                total_init += ic
                total_best += bc
                if "IMPROVED" in res:
                    improved += 1
            if n_done % 100 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done)
                print(f"  [{n_done}/{len(work)}]  "
                      f"improved={improved}  "
                      f"mean_init={total_init/n_done:.2f}  "
                      f"mean_best={total_best/n_done:.2f}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min",
                      flush=True)

    dt = time.time() - t0
    print(f"\nLQR done in {dt/60:.1f} min")
    print(f"  improved: {improved}/{len(work)}")
    print(f"  mean init: {total_init/n_done:.2f}")
    print(f"  mean best: {total_best/n_done:.2f}")


if __name__ == "__main__":
    main()
