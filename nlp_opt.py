"""L-BFGS-B multi-start trajectory optimization on the ARX surrogate.

Differences from our earlier `lqr_qp.py` (which solves the same cost as
a closed-form normal-equations problem):

  - Uses scipy's L-BFGS-B, a true bound-constrained quasi-Newton method.
    Handles |u| <= 2 explicitly during optimization (vs. clip-after).
  - Multi-start: tries N random perturbations of the ILC init. Helps
    escape the single local minimum that the LQR closed-form lands on.
  - Anchor regularizer to ILC keeps each optimization centered near a
    good solution while still allowing local refinement.
  - Real-sim validation after each restart; keep whichever (ILC or one
    of the optimized variants) has the lowest real cost.

The ARX model has R² = 0.987 on training data but error compounds during
rollout, so we don't expect a big jump from this. The added value over
lqr_qp is in the multi-start exploration + true bound constraints.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    STEER_RANGE,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from cem import _segment_fingerprint

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX  # 400


# Per-worker state
_SIM_MODEL = None
_ARX = None


def _init(model_path, arx_path):
    global _SIM_MODEL, _ARX
    _SIM_MODEL = TinyPhysicsModel(model_path, debug=False)
    with open(arx_path) as f:
        _ARX = json.load(f)["coefficients"]


def _build_segment(csv_path):
    df = pd.read_csv(csv_path)
    roll = (np.sin(df["roll"].values) * ACC_G).astype(np.float64)
    v_ego = df["vEgo"].values.astype(np.float64)
    target = df["targetLateralAcceleration"].values.astype(np.float64)
    return roll, v_ego, target, len(df)


def _rollout_arx(u, roll, v_ego, target, arx):
    """Compute ARX-predicted lat_accel trajectory under actions u.
    Returns the trajectory vector y[0..COST_END_IDX-1].
    """
    a1 = arx["a_y[t]"]; a2 = arx["a_y[t-1]"]; b = arx["action[t]"]
    g = arx["roll[t]"]; d = arx["v_ego[t]"]; bias = arx["bias"]
    T = COST_END_IDX
    y = np.zeros(T)
    y[:CONTROL_START_IDX] = target[:CONTROL_START_IDX]
    for t in range(CONTROL_START_IDX, T - 1):
        i = t - CONTROL_START_IDX
        y[t + 1] = a1 * y[t] + a2 * y[t - 1] + b * u[i] + g * roll[t] + d * v_ego[t] + bias
    return y


def _cost_and_grad(u, roll, v_ego, target, arx, u_anchor, anchor_w):
    """Return ARX-cost J(u) and its analytical gradient w.r.t. u."""
    a1 = arx["a_y[t]"]; a2 = arx["a_y[t-1]"]; b = arx["action[t]"]
    g = arx["roll[t]"]; d = arx["v_ego[t]"]; bias = arx["bias"]
    T = COST_END_IDX
    q_track = 50.0
    q_jerk = 1.0 / (DEL_T ** 2)

    # Forward
    y = np.zeros(T)
    y[:CONTROL_START_IDX] = target[:CONTROL_START_IDX]
    for t in range(CONTROL_START_IDX, T - 1):
        i = t - CONTROL_START_IDX
        y[t + 1] = a1 * y[t] + a2 * y[t - 1] + b * u[i] + g * roll[t] + d * v_ego[t] + bias

    # Cost: sum over cost window
    err = target[CONTROL_START_IDX:T] - y[CONTROL_START_IDX:T]
    track_cost = q_track * (err ** 2).sum()
    dy = y[CONTROL_START_IDX:T] - y[CONTROL_START_IDX - 1: T - 1]
    jerk_cost = q_jerk * (dy ** 2).sum()
    anchor_cost = anchor_w * ((u - u_anchor) ** 2).sum()
    J = track_cost + jerk_cost + anchor_cost

    # Adjoint backward for gradient
    # dJ/dy[t]:
    #   from tracking:  -2*q_track*(target[t] - y[t])  for t in cost window
    #   from jerk:      +2*q_jerk*(y[t] - y[t-1]) - 2*q_jerk*(y[t+1] - y[t])
    dJ_dy = np.zeros(T)
    for t in range(CONTROL_START_IDX, T):
        dJ_dy[t] += -2.0 * q_track * (target[t] - y[t])
        # jerk contribution: this y[t] appears in dy[t] = y[t] - y[t-1] (positive)
        # and in dy[t+1] = y[t+1] - y[t] (negative) if t+1 in window
        dJ_dy[t] += 2.0 * q_jerk * (y[t] - y[t - 1])
        if t + 1 < T:
            dJ_dy[t] += -2.0 * q_jerk * (y[t + 1] - y[t])

    # Recursive: y[t+1] = a1*y[t] + a2*y[t-1] + b*u[i] + ...
    # so dy[t+1]/du[i] = b; dy[t+1]/dy[t] = a1; dy[t+1]/dy[t-1] = a2
    # Backward pass: define lambda[t] = dJ/dy[t] accumulated from future
    lam = np.zeros(T + 2)
    for t in range(T - 1, CONTROL_START_IDX - 1, -1):
        # this y[t] affects y[t+1] (coeff a1) and y[t+2] (coeff a2)
        lam[t] = dJ_dy[t] + a1 * lam[t + 1] + a2 * lam[t + 2]

    # gradient w.r.t. u[i]: only affects y[t+1] where t = CONTROL_START_IDX + i
    grad = np.zeros(ACTION_HORIZON)
    for i in range(ACTION_HORIZON):
        t = CONTROL_START_IDX + i
        if t + 1 < T:
            grad[i] = b * lam[t + 1]
    grad += 2.0 * anchor_w * (u - u_anchor)
    return J, grad


def _real_cost(csv_path, actions):
    c = PlaybackController(action_seq=actions)
    s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=c, debug=False)
    return s.rollout()["total_cost"]


def optimize_segment_nlp(csv_path, ilc_actions, n_restarts=4, anchor=10.0,
                         maxiter=150, ftol=1e-7):
    csv_path = str(Path(csv_path))
    roll, v_ego, target, T = _build_segment(csv_path)
    if len(ilc_actions) < ACTION_HORIZON:
        ilc_actions = np.pad(ilc_actions, (0, ACTION_HORIZON - len(ilc_actions)))
    ilc_actions = ilc_actions[:ACTION_HORIZON].astype(np.float64)

    init_real = _real_cost(csv_path, ilc_actions.astype(np.float32))
    best_real = init_real
    best_actions = ilc_actions.copy().astype(np.float32)

    bounds = [(STEER_RANGE[0], STEER_RANGE[1])] * ACTION_HORIZON
    rng = np.random.default_rng(hash(csv_path) % (2**32))

    for r in range(n_restarts):
        if r == 0:
            x0 = ilc_actions.copy()
        else:
            x0 = np.clip(ilc_actions + rng.normal(0, 0.03, ACTION_HORIZON), -2.0, 2.0)

        try:
            res = minimize(
                _cost_and_grad,
                x0,
                args=(roll, v_ego, target, _ARX, ilc_actions, anchor),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": 1e-6},
            )
            cand = res.x.astype(np.float32)
            rc = _real_cost(csv_path, cand)
            if rc < best_real - 1e-3:
                best_real = rc
                best_actions = cand
        except Exception:
            continue

    return best_actions, best_real, init_real


def _worker(args):
    csv_path, opt_dir, out_dir, n_restarts, anchor = args
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    try:
        ilc = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
        new_actions, best_real, init_real = optimize_segment_nlp(
            csv_path, ilc, n_restarts=n_restarts, anchor=anchor,
        )
        if best_real < init_real - 1e-3:
            fp = _segment_fingerprint(csv_path)
            np.savez(
                str(Path(out_dir) / f"{seg_id}.npz"),
                actions=new_actions,
                best_cost=best_real,
                baseline_cost=init_real,
                fingerprint=fp,
            )
            result = "IMPROVED"
        else:
            result = "no-change"
        return (seg_id, float(init_real), float(best_real), result)
    except Exception as e:
        return (seg_id, None, None, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Single-segment test path")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--arx", default="arx_model.json")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_restarts", type=int, default=4)
    parser.add_argument("--anchor", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.csv:
        # Single-segment test
        _init(args.model_path, args.arx)
        ilc = np.load(str(Path(args.opt_actions) / f"{Path(args.csv).stem}.npz"))["actions"].astype(np.float32)
        t0 = time.time()
        new_actions, best_real, init_real = optimize_segment_nlp(args.csv, ilc, n_restarts=args.n_restarts, anchor=args.anchor)
        dt = time.time() - t0
        print(f"init real:   {init_real:.3f}")
        print(f"NLP best:    {best_real:.3f}")
        print(f"gain:        {init_real - best_real:+.3f}  ({100*(init_real-best_real)/init_real:+.2f}%)")
        print(f"time: {dt:.1f}s")
        return

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"NLP opt on {len(files)} segments  workers={args.workers}  n_restarts={args.n_restarts}  anchor={args.anchor}")

    work = [(f, args.opt_actions, args.out_dir, args.n_restarts, args.anchor) for f in files]

    t0 = time.time()
    with mp.Pool(processes=args.workers, initializer=_init,
                 initargs=(args.model_path, args.arx)) as pool:
        improved = 0
        n_done = 0
        total_init = 0.0
        total_best = 0.0
        for sg, ic, bc, res in pool.imap_unordered(_worker, work, chunksize=2):
            n_done += 1
            if ic is not None:
                total_init += ic
                total_best += bc
                if "IMPROVED" in res:
                    improved += 1
            if n_done % 100 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done)
                print(f"  [{n_done}/{len(work)}]  improved={improved}  "
                      f"mean_init={total_init/n_done:.2f}  "
                      f"mean_best={total_best/n_done:.2f}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min",
                      flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
