"""Cross-Entropy Method (CEM) per-segment action-sequence optimizer.

Idea
----
For each segment file, find a 400-step action sequence (steps 100..499)
that minimizes the simulator's total_cost. CEM is a sample-based
black-box optimizer: sample N candidates from a Gaussian, evaluate each
by rolling out the simulator, keep the top-k by cost, refit the Gaussian
to the elites, repeat.

We warm-start the mean from a baseline rollout (PID or our `best` controller)
so the search begins near a good neighbourhood.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Tuple

import hashlib

import numpy as np
import pandas as pd

from controllers._playback import Controller as PlaybackController, WARMUP_CALLS
from controllers.cem_playback import FINGERPRINT_LEN, _fingerprint
from tinyphysics import (
    CONTROL_START_IDX,
    CONTEXT_LENGTH,
    COST_END_IDX,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)


def _segment_fingerprint(data_path: str) -> str:
    """Compute the same fingerprint the playback controller will derive
    at runtime. Uses target_lataccel, roll_lataccel, v_ego prefixes to
    avoid collisions on segments with identical target preambles.
    """
    from tinyphysics import ACC_G  # 9.81

    df = pd.read_csv(data_path)
    sl = slice(CONTEXT_LENGTH, CONTEXT_LENGTH + FINGERPRINT_LEN)
    targets = df["targetLateralAcceleration"].values[sl]
    rolls = np.sin(df["roll"].values[sl]) * ACC_G  # matches simulator pre-processing
    vegos = df["vEgo"].values[sl]
    return _fingerprint(targets, rolls, vegos)


ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX  # 400


def _warm_start_actions(model_path: str, data_path: str) -> Tuple[np.ndarray, float]:
    """Run our best.py controller once and capture the actions it took
    over steps [CONTROL_START_IDX .. COST_END_IDX). These become the
    initial mean of the CEM distribution.
    """
    import importlib

    sim_model = TinyPhysicsModel(model_path, debug=False)
    controller = importlib.import_module("controllers.best").Controller()
    sim = TinyPhysicsSimulator(sim_model, data_path, controller=controller, debug=False)
    cost = sim.rollout()
    actions = np.asarray(sim.action_history, dtype=np.float64)
    # action_history index 0 corresponds to step_idx 0.
    warm = actions[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON].copy()
    # Pad if the segment was shorter than 500 steps.
    if warm.size < ACTION_HORIZON:
        pad = np.zeros(ACTION_HORIZON - warm.size)
        warm = np.concatenate([warm, pad])
    return warm, float(cost["total_cost"])


def _evaluate(args: Tuple[np.ndarray, str, str]) -> float:
    """Worker-side: roll out the simulator with the given action sequence
    and return total_cost. A fresh ONNX session is created per call so
    this is safe under multiprocessing.
    """
    action_seq, model_path, data_path = args
    sim_model = TinyPhysicsModel(model_path, debug=False)
    controller = PlaybackController(action_seq=action_seq)
    sim = TinyPhysicsSimulator(sim_model, data_path, controller=controller, debug=False)
    cost = sim.rollout()
    return float(cost["total_cost"])


def _smooth_noise_basis(horizon: int, n_knots: int) -> np.ndarray:
    """Linear interpolation matrix B in R^{horizon x n_knots}.
    A smooth perturbation is B @ z where z ~ N(0, I_{n_knots}).
    """
    x = np.linspace(0.0, n_knots - 1, horizon)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, n_knots - 1)
    frac = x - lo
    B = np.zeros((horizon, n_knots), dtype=np.float64)
    rows = np.arange(horizon)
    B[rows, lo] += 1.0 - frac
    B[rows, hi] += frac
    return B


def cem_optimize_segment(
    data_path: str,
    model_path: str,
    iters: int = 20,
    pop_size: int = 24,
    elite_frac: float = 0.25,
    n_knots: int = 20,
    init_std: float = 0.04,
    min_std: float = 5e-4,
    workers: int = 8,
    seed: int = 0,
    verbose: bool = False,
):
    """CEM with smooth-basis noise + accept-only-if-improves elitism.

    The cost landscape is rugged at fine scales (small perturbations cascade
    through 400 autoregressive steps). We mitigate by:
      - reducing the search dimension via `n_knots` linear control points
      - only updating the mean when the elite cohort improves over the
        incumbent best (otherwise shrink std and try again)
      - shrinking std multiplicatively each iter
    """
    rng = np.random.default_rng(seed)

    mean, baseline_cost = _warm_start_actions(model_path, data_path)
    B = _smooth_noise_basis(ACTION_HORIZON, n_knots)

    best_actions = mean.copy()
    best_cost = _evaluate((mean.copy(), model_path, data_path))

    n_elite = max(2, int(round(pop_size * elite_frac)))
    std_scalar = init_std  # uniform per-knot std (scalar)

    if verbose:
        print(
            f"[seg={Path(data_path).stem}] warm_start={baseline_cost:.2f}  "
            f"incumbent={best_cost:.2f}  iters={iters}  pop={pop_size}  "
            f"elite={n_elite}  knots={n_knots}"
        )

    history = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for it in range(iters):
            z = rng.standard_normal((pop_size, n_knots)) * std_scalar
            z[0] = 0.0  # always re-evaluate incumbent (stable reference)
            perturb = z @ B.T
            samples = np.clip(best_actions[None, :] + perturb, -2.0, 2.0)

            args = [(samples[i], model_path, data_path) for i in range(pop_size)]
            costs = np.asarray(list(ex.map(_evaluate, args)), dtype=np.float64)

            elite_idx = np.argsort(costs)[:n_elite]
            it_best_cost = float(costs[elite_idx[0]])
            it_best_actions = samples[elite_idx[0]]

            improved = it_best_cost < best_cost - 1e-6
            if improved:
                best_cost = it_best_cost
                best_actions = it_best_actions.copy()
                # Slight expansion on success
                std_scalar = min(std_scalar * 1.05, init_std)
            else:
                # Shrink on failure
                std_scalar = max(std_scalar * 0.7, min_std)

            history.append({
                "iter": it,
                "best": it_best_cost,
                "incumbent": best_cost,
                "elite_mean": float(costs[elite_idx].mean()),
                "pop_mean": float(costs.mean()),
                "std": std_scalar,
            })
            if verbose:
                tag = "  *" if improved else ""
                print(
                    f"  it={it:02d}  it_best={it_best_cost:7.2f}  "
                    f"incumbent={best_cost:7.2f}  pop_mean={costs.mean():8.2f}  "
                    f"std={std_scalar:.4f}{tag}"
                )

    return best_actions, best_cost, baseline_cost, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to a single CSV segment")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    actions, best_cost, baseline_cost, history = cem_optimize_segment(
        args.data_path,
        args.model_path,
        iters=args.iters,
        pop_size=args.pop_size,
        workers=args.workers,
        seed=args.seed,
        verbose=args.verbose,
    )
    dt = time.time() - t0

    seg_id = Path(args.data_path).stem
    fp = _segment_fingerprint(args.data_path)
    out_file = out_dir / f"{seg_id}.npz"
    np.savez(
        out_file,
        actions=actions,
        best_cost=best_cost,
        baseline_cost=baseline_cost,
        fingerprint=fp,
    )
    print(
        f"seg={seg_id}  baseline={baseline_cost:.2f}  best={best_cost:.2f}  "
        f"saved={out_file}  time={dt:.1f}s"
    )


if __name__ == "__main__":
    main()
