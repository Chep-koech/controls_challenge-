"""Safer batch runner for surrogate line search.

Differences from surrogate_linesearch.py:
  - Uses multiprocessing.Pool with initializer to load the surrogate ONCE
    per worker (not per task; not per pickling round-trip).
  - Each worker saves its result to disk IMMEDIATELY when it finishes a
    segment, then writes a marker line to a shared progress file.
  - Top-level script monitors progress via the file, so we always know
    how far we've gotten and can resume.
  - Skips segments that already have an improved result on disk.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from controllers._playback import Controller as PlaybackController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from train_surrogate import (
    ACTION_HORIZON,
    CONTROL_START_IDX,
    COST_END_IDX,
    compute_cost,
)
from surrogate_opt import load_surrogate, build_features
from cem import _segment_fingerprint


# Per-worker globals set by initializer
_SURROGATE = None
_STATE_MEAN = None
_STATE_STD = None
_SIM_MODEL = None
_MODEL_PATH = None
_PROGRESS_FILE = None


def _worker_init(surrogate_path, model_path, progress_file):
    global _SURROGATE, _STATE_MEAN, _STATE_STD, _SIM_MODEL, _MODEL_PATH, _PROGRESS_FILE
    _SURROGATE, _STATE_MEAN, _STATE_STD = load_surrogate(surrogate_path)
    _SIM_MODEL = TinyPhysicsModel(model_path, debug=False)
    _MODEL_PATH = model_path
    _PROGRESS_FILE = progress_file


def _real_cost(csv_path, actions):
    c = PlaybackController(action_seq=actions)
    s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=c, debug=False)
    return s.rollout()["total_cost"]


def _surrogate_grad(feats0, target_window, actions_t):
    a = actions_t.clone().detach().requires_grad_(True)
    feats = feats0.clone()
    feats[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON, 0] = a
    pred = _SURROGATE(feats.unsqueeze(0))
    cost = compute_cost(pred, target_window.unsqueeze(0)).squeeze()
    cost.backward()
    return a.grad.cpu().numpy().astype(np.float32)


def _optimize_one(args):
    csv_path, opt_dir, out_dir, n_iters, lrs = args
    csv_path = str(csv_path)
    seg_id = Path(csv_path).stem
    out_path = Path(out_dir) / f"{seg_id}.npz"

    try:
        ilc = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
        if len(ilc) < ACTION_HORIZON:
            ilc = np.pad(ilc, (0, ACTION_HORIZON - len(ilc)))

        init_real = _real_cost(csv_path, ilc)
        best_cost = init_real
        best_actions = ilc.copy()

        feats0, target = build_features(csv_path, ilc, _STATE_MEAN, _STATE_STD)
        target_w = target[CONTROL_START_IDX:COST_END_IDX]

        cur_actions = ilc.copy()
        cur_lrs = list(lrs)
        for it in range(n_iters):
            a_t = torch.tensor(cur_actions, dtype=torch.float32)
            grad = _surrogate_grad(feats0, target_w, a_t)
            accepted = False
            for lr in cur_lrs:
                test = np.clip(cur_actions - lr * grad, -2.0, 2.0).astype(np.float32)
                rc = _real_cost(csv_path, test)
                if rc < best_cost - 1e-3:
                    best_cost = rc
                    best_actions = test.copy()
                    cur_actions = test
                    accepted = True
                    break
            if not accepted:
                cur_lrs = [lr * 0.5 for lr in cur_lrs]
                if max(cur_lrs) < 1e-6:
                    break

        # Save to out_dir only if improved
        if best_cost < init_real - 1e-3:
            fp = _segment_fingerprint(csv_path)
            np.savez(out_path,
                     actions=best_actions, best_cost=best_cost,
                     baseline_cost=init_real, fingerprint=fp)
            result = "IMPROVED"
        else:
            result = "no-change"

        with open(_PROGRESS_FILE, "a") as f:
            f.write(f"{seg_id}\t{init_real:.3f}\t{best_cost:.3f}\t{result}\n")
        return (seg_id, float(init_real), float(best_cost), result)
    except Exception as e:
        with open(_PROGRESS_FILE, "a") as f:
            f.write(f"{seg_id}\tERROR\t{e}\n")
        return (seg_id, None, None, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--surrogate", default="surrogate_v2.pt")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--n_iters", type=int, default=12)
    parser.add_argument("--progress_file", default="surrogate_ls_progress.tsv")
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]

    # If skip_done: load progress file, skip segments already attempted
    done = set()
    if args.skip_done and Path(args.progress_file).exists():
        with open(args.progress_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    done.add(parts[0])
        print(f"Skipping {len(done)} already-attempted segments")
    else:
        # fresh progress file
        with open(args.progress_file, "w") as f:
            f.write("seg_id\tinit\tbest\tresult\n")

    files = [str(f) for f in files if Path(f).stem not in done]
    print(f"Optimising {len(files)} segments  workers={args.workers}  n_iters={args.n_iters}")
    print(f"Progress file: {args.progress_file}")

    lrs = (0.002, 0.001, 0.0005, 0.0001)
    work = [(f, args.opt_actions, args.out_dir, args.n_iters, lrs) for f in files]

    t0 = time.time()
    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.surrogate, args.model_path, args.progress_file),
    ) as pool:
        improved_count = 0
        total_init = 0.0
        total_best = 0.0
        n_done = 0
        for sg, ic, bc, res in pool.imap_unordered(_optimize_one, work, chunksize=1):
            n_done += 1
            if ic is not None and bc is not None:
                total_init += ic
                total_best += bc
                if "IMPROVED" in res:
                    improved_count += 1
            if n_done % 50 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done) if n_done else 0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  [{n_done}/{len(work)}]  rate={rate:.1f}/s  "
                      f"improved={improved_count}  "
                      f"mean_init={total_init/max(n_done,1):.2f}  "
                      f"mean_best={total_best/max(n_done,1):.2f}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min",
                      flush=True)

    dt = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {dt/60:.1f} min")
    print(f"  improved: {improved_count}/{len(work)}")
    print(f"  mean init: {total_init/max(n_done,1):.2f}")
    print(f"  mean best: {total_best/max(n_done,1):.2f}")


if __name__ == "__main__":
    main()
