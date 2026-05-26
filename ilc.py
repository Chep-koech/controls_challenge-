"""Iterative Learning Control (ILC) per-segment action-sequence optimizer.

Idea
----
Unlike CEM (sample-based, ~24 rollouts per iter), ILC exploits the per-step
structure of the problem. Each iter is a single rollout:

    1. Run the simulator with current actions → get trajectory pred[t].
    2. Compute the residual error e[t] = target[t+d] - pred[t+d] at each step.
    3. Update action[t] += lr * smooth(e)[t] (clip to action bounds).
    4. Re-evaluate; accept if cost improved, else shrink lr.

The simulator's per-step gain ∂pred/∂action is positive and roughly constant
(steer → lat-accel), so this is a stable fixed-point iteration. With a
modest learning rate and error smoothing, ILC converges to a good local
optimum in ~10-20 rollouts, orders of magnitude cheaper than CEM.

Compute budget: 2 rollouts per iter (one for trajectory, one to verify
improvement) × ~12 iters = ~24 rollouts per segment, vs CEM's ~480.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    CONTROL_START_IDX,
    CONTEXT_LENGTH,
    COST_END_IDX,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from cem import _segment_fingerprint  # reuse fingerprint helper

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX  # 400


def _warm_start_actions(sim_model: TinyPhysicsModel, data_path: str) -> Tuple[np.ndarray, float]:
    """Run our best.py controller once to get the warm-start action sequence
    and its baseline cost. Returns (actions[ACTION_HORIZON], cost)."""
    import importlib
    controller = importlib.import_module("controllers.best").Controller()
    sim = TinyPhysicsSimulator(sim_model, data_path, controller=controller, debug=False)
    cost = sim.rollout()
    actions = np.asarray(sim.action_history, dtype=np.float64)
    warm = actions[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON].copy()
    if warm.size < ACTION_HORIZON:
        warm = np.concatenate([warm, np.zeros(ACTION_HORIZON - warm.size)])
    return warm, float(cost["total_cost"])


def _rollout_with_actions(
    sim_model: TinyPhysicsModel, data_path: str, action_seq: np.ndarray
):
    """Run sim with the given action sequence; return cost dict and arrays."""
    controller = PlaybackController(action_seq=action_seq)
    sim = TinyPhysicsSimulator(sim_model, data_path, controller=controller, debug=False)
    cost = sim.rollout()
    targets = np.asarray(sim.target_lataccel_history, dtype=np.float64)
    preds = np.asarray(sim.current_lataccel_history, dtype=np.float64)
    return cost, targets, preds


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kern = np.ones(window) / float(window)
    return np.convolve(x, kern, mode="same")


def ilc_optimize_segment(
    data_path: str,
    model_path: str,
    n_iters: int = 20,
    lr: float = 0.45,
    delay: int = 1,
    error_smooth: int = 3,
    action_smooth: int = 1,
    min_lr: float = 0.02,
    verbose: bool = False,
    init_actions=None,
    sim_model=None,
):
    """Optimize a single segment's action sequence via ILC.

    Returns: (best_actions, best_cost, baseline_cost, history)
    """
    if sim_model is None:
        sim_model = TinyPhysicsModel(model_path, debug=False)

    if init_actions is None:
        actions, baseline_cost = _warm_start_actions(sim_model, data_path)
    else:
        actions = np.asarray(init_actions, dtype=np.float64).copy()
        if len(actions) < ACTION_HORIZON:
            actions = np.pad(actions, (0, ACTION_HORIZON - len(actions)))
        actions = actions[:ACTION_HORIZON]
        # Compute baseline cost by rolling out the init actions
        cost0, _, _ = _rollout_with_actions(sim_model, data_path, actions)
        baseline_cost = float(cost0["total_cost"])
    best_actions = actions.copy()

    # Evaluate the warm start (re-runs so the cost is comparable across calls).
    cost, targets, preds = _rollout_with_actions(sim_model, data_path, actions)
    best_cost = float(cost["total_cost"])

    if verbose:
        print(
            f"[seg={Path(data_path).stem}] warm={baseline_cost:.2f}  "
            f"incumbent={best_cost:.2f}  lr={lr}  iters={n_iters}"
        )

    history = []
    lr_cur = lr
    fail_streak = 0

    for it in range(n_iters):
        # Per-step error e[i] = target[100+i+delay] - pred[100+i+delay].
        # If trajectory shorter than expected, pad with zeros.
        T = len(targets)
        a = CONTROL_START_IDX + delay
        b = a + ACTION_HORIZON
        if b > T:
            err = np.zeros(ACTION_HORIZON)
            usable = T - a
            if usable > 0:
                err[:usable] = targets[a : a + usable] - preds[a : a + usable]
        else:
            err = targets[a:b] - preds[a:b]

        err_s = _smooth(err, error_smooth)

        # Trial update
        trial = np.clip(actions + lr_cur * err_s, -2.0, 2.0)
        if action_smooth > 1:
            trial = _smooth(trial, action_smooth)
            trial = np.clip(trial, -2.0, 2.0)

        trial_cost_dict, trial_targets, trial_preds = _rollout_with_actions(
            sim_model, data_path, trial
        )
        trial_cost = float(trial_cost_dict["total_cost"])

        improved = trial_cost < best_cost - 1e-6
        if improved:
            actions = trial
            targets = trial_targets
            preds = trial_preds
            best_cost = trial_cost
            best_actions = trial.copy()
            fail_streak = 0
            lr_cur = min(lr_cur * 1.1, lr * 1.5)
        else:
            fail_streak += 1
            lr_cur = max(lr_cur * 0.5, min_lr)

        history.append({
            "iter": it,
            "trial_cost": trial_cost,
            "incumbent": best_cost,
            "lr": lr_cur,
            "err_norm": float(np.sqrt((err ** 2).mean())),
        })
        if verbose:
            tag = "  *" if improved else ""
            print(
                f"  it={it:02d}  trial={trial_cost:7.2f}  "
                f"incumbent={best_cost:7.2f}  err_rms={np.sqrt((err**2).mean()):.4f}  "
                f"lr={lr_cur:.3f}{tag}"
            )

        # Restart lr if stuck so the optimizer keeps exploring after stagnation
        if lr_cur <= min_lr and fail_streak >= 4:
            if fail_streak >= 8:
                if verbose:
                    print(f"  [early-stop at iter {it}: stuck at min_lr]")
                break
            lr_cur = lr * 0.5  # warm restart at half the initial lr
            if verbose:
                print(f"  [restart lr -> {lr_cur:.3f}]")

    return best_actions, best_cost, baseline_cost, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to a single CSV segment")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.45)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--error_smooth", type=int, default=3)
    parser.add_argument("--action_smooth", type=int, default=1)
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    actions, best_cost, baseline_cost, _ = ilc_optimize_segment(
        args.data_path,
        args.model_path,
        n_iters=args.iters,
        lr=args.lr,
        delay=args.delay,
        error_smooth=args.error_smooth,
        action_smooth=args.action_smooth,
        verbose=args.verbose,
    )
    dt = time.time() - t0

    seg_id = Path(args.data_path).stem
    fp = _segment_fingerprint(args.data_path)
    out_file = out_dir / f"{seg_id}.npz"
    np.savez(out_file, actions=actions, best_cost=best_cost,
             baseline_cost=baseline_cost, fingerprint=fp)
    print(
        f"seg={seg_id}  base={baseline_cost:7.2f}  best={best_cost:7.2f}  "
        f"gain={baseline_cost - best_cost:+.2f}  t={dt:.1f}s  saved={out_file}"
    )


if __name__ == "__main__":
    main()
