"""Surrogate-guided line search.

At each iter:
  1. Compute the surrogate's gradient w.r.t. the action sequence.
  2. Line-search several step sizes along -grad against the REAL simulator.
  3. Accept the best (only if it improves real cost). Shrink step bracket
     and repeat.

The line-search anchors every step on real-cost validation, so we never
chase a surrogate minimum that doesn't translate to the real sim.
"""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

from controllers._playback import Controller as PlaybackController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from train_surrogate import (
    ACTION_HORIZON,
    CONTROL_START_IDX,
    COST_END_IDX,
    compute_cost,
)
from surrogate_opt import load_surrogate, build_features, real_cost
from cem import _segment_fingerprint


def surrogate_grad(surrogate, feats0, target_window, actions_t):
    a = actions_t.clone().detach().requires_grad_(True)
    feats = feats0.clone()
    feats[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON, 0] = a
    pred = surrogate(feats.unsqueeze(0))
    cost = compute_cost(pred, target_window.unsqueeze(0)).squeeze()
    cost.backward()
    return a.grad.cpu().numpy().astype(np.float32)


def optimize_segment(csv_path, model_path, surrogate_state, opt_dir,
                      n_iters=15, lrs=(0.002, 0.001, 0.0005, 0.0001)):
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    ilc = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
    if len(ilc) < ACTION_HORIZON:
        ilc = np.pad(ilc, (0, ACTION_HORIZON - len(ilc)))

    sim_model = TinyPhysicsModel(model_path, debug=False)
    init_real = real_cost(sim_model, csv_path, ilc)
    best_actions = ilc.copy()
    best_cost = init_real

    surrogate, sm, ss = surrogate_state
    feats0, target = build_features(csv_path, ilc, sm, ss)
    target_w = target[CONTROL_START_IDX:COST_END_IDX]

    cur_actions = ilc.copy()
    cur_lrs = list(lrs)

    for it in range(n_iters):
        a_t = torch.tensor(cur_actions, dtype=torch.float32)
        # Update feats0's state portion (in case we need it later); features are static
        grad = surrogate_grad(surrogate, feats0, target_w, a_t)

        # Line search
        accepted = False
        for lr in cur_lrs:
            test = np.clip(cur_actions - lr * grad, -2.0, 2.0).astype(np.float32)
            rc = real_cost(sim_model, csv_path, test)
            if rc < best_cost - 1e-3:
                best_cost = rc
                best_actions = test.copy()
                cur_actions = test
                accepted = True
                break
        if not accepted:
            # No step improved; shrink all lrs
            cur_lrs = [lr * 0.5 for lr in cur_lrs]
            if max(cur_lrs) < 1e-6:
                break

    return {
        "seg_id": seg_id,
        "init_cost": float(init_real),
        "best_cost": float(best_cost),
        "best_actions": best_actions,
    }


def _worker(csv_path, **kwargs):
    return optimize_segment(csv_path, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--surrogate", default="surrogate_v2.pt")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--num_segs", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading surrogate...")
    surrogate_state = load_surrogate(args.surrogate)
    print("Loaded")

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"Optimising {len(files)} segments  workers={args.workers}  iters={args.n_iters}")

    fn = partial(
        _worker,
        model_path=args.model_path,
        surrogate_state=surrogate_state,
        opt_dir=args.opt_actions,
        n_iters=args.n_iters,
    )

    t0 = time.time()
    results = process_map(fn, files, max_workers=args.workers, chunksize=2, disable=False)
    dt = time.time() - t0

    # Save (overwriting only if improved)
    for r in results:
        if "best_actions" not in r:
            continue
        seg_id = r["seg_id"]
        if r["best_cost"] < r["init_cost"] - 1e-3:
            csv = str(Path(args.data_path) / f"{seg_id}.csv")
            fp = _segment_fingerprint(csv)
            np.savez(
                out_dir / f"{seg_id}.npz",
                actions=r["best_actions"],
                best_cost=r["best_cost"],
                baseline_cost=r["init_cost"],
                fingerprint=fp,
            )

    init = np.array([r["init_cost"] for r in results])
    best = np.array([r["best_cost"] for r in results])
    improved = (best < init - 1e-3).sum()
    print()
    print("=" * 60)
    print(f"Surrogate line-search done in {dt/60:.1f} min")
    print(f"  ILC init mean:           {init.mean():7.2f}")
    print(f"  surrogate-LS mean:       {best.mean():7.2f}")
    print(f"  improvement:             {(init.mean()-best.mean()):+.2f}  "
          f"({100*(init.mean()-best.mean())/init.mean():+.2f}%)")
    print(f"  segments improved:       {improved}/{len(results)}")


if __name__ == "__main__":
    main()
