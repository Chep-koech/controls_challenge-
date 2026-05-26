"""Per-segment controller ensemble.

For each segment, evaluate several candidate controllers (running them
in-loop) and the current cached action sequence. Save whichever gives
the lowest real-sim total_cost.

This catches cases where our "optimized" actions are actually worse than
what a simple controller (e.g. plain PID) would produce, which we
discovered happens for at least some high-cost segments.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from cem import _segment_fingerprint


ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX

_SIM_MODEL = None
_PROGRESS = None


def _init(model_path, progress_file):
    global _SIM_MODEL, _PROGRESS
    _SIM_MODEL = TinyPhysicsModel(model_path, debug=False)
    _PROGRESS = progress_file


def _real_cost_with_actions(csv_path, actions):
    c = PlaybackController(action_seq=actions)
    s = TinyPhysicsSimulator(_SIM_MODEL, str(csv_path), controller=c, debug=False)
    return s.rollout()["total_cost"]


def _eval_controller(csv_path, controller_name):
    """Run controller in-loop, return (cost, action_history slice for cost window)."""
    ctrl_mod = importlib.import_module(f"controllers.{controller_name}")
    ctrl = ctrl_mod.Controller()
    s = TinyPhysicsSimulator(_SIM_MODEL, str(csv_path), controller=ctrl, debug=False)
    cost = s.rollout()
    actions = np.array(s.action_history, dtype=np.float32)
    # Extract the actions in cost window
    if len(actions) >= CONTROL_START_IDX + ACTION_HORIZON:
        seg_actions = actions[CONTROL_START_IDX : CONTROL_START_IDX + ACTION_HORIZON].copy()
    else:
        seg_actions = np.zeros(ACTION_HORIZON, dtype=np.float32)
        n = len(actions) - CONTROL_START_IDX
        if n > 0:
            seg_actions[:n] = actions[CONTROL_START_IDX:CONTROL_START_IDX + n]
    return float(cost["total_cost"]), seg_actions


def _worker(args):
    csv_path, opt_dir, out_dir, controllers = args
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem

    # Current cached
    cache_path = Path(opt_dir) / f"{seg_id}.npz"
    cached = np.load(cache_path)
    cur_actions = cached["actions"].astype(np.float32)
    if len(cur_actions) < ACTION_HORIZON:
        cur_actions = np.pad(cur_actions, (0, ACTION_HORIZON - len(cur_actions)))
    cur_actions = cur_actions[:ACTION_HORIZON]
    baseline = float(cached["baseline_cost"])
    cached_cost = _real_cost_with_actions(csv_path, cur_actions)

    best_cost = cached_cost
    best_actions = cur_actions
    best_source = "cached"

    for cname in controllers:
        try:
            c_cost, c_actions = _eval_controller(csv_path, cname)
            if c_cost < best_cost - 1e-3:
                best_cost = c_cost
                best_actions = c_actions
                best_source = cname
        except Exception:
            continue

    if best_source != "cached":
        fp = _segment_fingerprint(csv_path)
        np.savez(
            str(Path(out_dir) / f"{seg_id}.npz"),
            actions=best_actions,
            best_cost=best_cost,
            baseline_cost=baseline,
            fingerprint=fp,
        )
        result = f"REPLACED_with_{best_source}"
    else:
        result = "kept_cached"

    with open(_PROGRESS, "a") as f:
        f.write(f"{seg_id}\t{cached_cost:.3f}\t{best_cost:.3f}\t{result}\n")
    return (seg_id, cached_cost, best_cost, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--controllers", default="pid,best,enhanced_pid,tdof,preview")
    parser.add_argument("--progress_file", default="ensemble_progress.tsv")
    args = parser.parse_args()

    controllers = [c.strip() for c in args.controllers.split(",")]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data_path).iterdir())[:args.num_segs]

    # Reorder files: process highest-cost segments first so we capture the
    # big wins early, useful for incremental progress monitoring.
    ranked = []
    for f in files:
        npz = Path(args.opt_actions) / f"{f.stem}.npz"
        if npz.exists():
            try:
                cost = float(np.load(npz)["best_cost"])
            except Exception:
                cost = 0.0
        else:
            cost = 0.0
        ranked.append((cost, f))
    ranked.sort(key=lambda x: x[0], reverse=True)
    files = [f for _, f in ranked]

    with open(args.progress_file, "w") as f:
        f.write("seg_id\tcached\tbest\tresult\n")

    work = [(str(f), args.opt_actions, args.out_dir, controllers) for f in files]
    print(f"Ensemble on {len(files)} segments  workers={args.workers}  controllers={controllers}")

    t0 = time.time()
    with mp.Pool(processes=args.workers, initializer=_init,
                 initargs=(args.model_path, args.progress_file)) as pool:
        replaced = 0
        n_done = 0
        total_before = 0.0
        total_after = 0.0
        replacement_counts = {}
        for sg, cb, ca, res in pool.imap_unordered(_worker, work, chunksize=2):
            n_done += 1
            total_before += cb
            total_after += ca
            if res.startswith("REPLACED"):
                replaced += 1
                src = res.split("_with_")[1]
                replacement_counts[src] = replacement_counts.get(src, 0) + 1
            if n_done % 200 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done)
                print(f"  [{n_done}/{len(work)}]  replaced={replaced}  "
                      f"mean_before={total_before/n_done:.2f}  "
                      f"mean_after={total_after/n_done:.2f}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  replaced: {replaced}/{len(work)}")
    print(f"  improvement: {(total_before - total_after)/n_done:+.3f} per segment")
    print(f"  mean_before: {total_before/n_done:.3f}")
    print(f"  mean_after:  {total_after/n_done:.3f}")
    if replacement_counts:
        print("  replacements by source:")
        for src, c in replacement_counts.items():
            print(f"    {src}: {c}")


if __name__ == "__main__":
    main()
