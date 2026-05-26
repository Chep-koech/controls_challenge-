"""Per-segment blend optimizer.

For each segment, try blending the cached optimized actions with
each of several baseline controllers' in-loop actions at multiple
mixing weights. Keep whichever blend gives the lowest real cost.

Often a 30-70% blend of an aggressive optimizer and a conservative PID
beats either alone, particularly on hard segments where pure ILC
overcorrects.
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


def _real_cost(csv_path, actions):
    c = PlaybackController(action_seq=actions)
    s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=c, debug=False)
    return s.rollout()["total_cost"]


def _get_controller_actions(csv_path, controller_name):
    ctrl_mod = importlib.import_module(f"controllers.{controller_name}")
    ctrl = ctrl_mod.Controller()
    s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=ctrl, debug=False)
    s.rollout()
    actions = np.array(s.action_history, dtype=np.float32)
    n = ACTION_HORIZON
    seg = np.zeros(n, dtype=np.float32)
    take = min(n, len(actions) - CONTROL_START_IDX)
    if take > 0:
        seg[:take] = actions[CONTROL_START_IDX:CONTROL_START_IDX + take]
    return seg


def _worker(args):
    csv_path, opt_dir, out_dir, controllers, weights = args
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    try:
        cache = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))
        cur_actions = cache["actions"].astype(np.float32)
        if len(cur_actions) < ACTION_HORIZON:
            cur_actions = np.pad(cur_actions, (0, ACTION_HORIZON - len(cur_actions)))
        cur_actions = cur_actions[:ACTION_HORIZON]
        baseline = float(cache["baseline_cost"])

        best_cost = _real_cost(csv_path, cur_actions)
        best_actions = cur_actions.copy()

        for cname in controllers:
            try:
                c_actions = _get_controller_actions(csv_path, cname)
                # Try blends
                for w in weights:
                    blended = np.clip(
                        w * cur_actions + (1.0 - w) * c_actions, -2.0, 2.0
                    ).astype(np.float32)
                    rc = _real_cost(csv_path, blended)
                    if rc < best_cost - 1e-3:
                        best_cost = rc
                        best_actions = blended
            except Exception:
                continue

        replaced = best_cost < float(cache["best_cost"]) - 1e-3
        if replaced:
            fp = _segment_fingerprint(csv_path)
            np.savez(str(Path(out_dir) / f"{seg_id}.npz"),
                     actions=best_actions, best_cost=best_cost,
                     baseline_cost=baseline, fingerprint=fp)
        with open(_PROGRESS, "a") as f:
            f.write(f"{seg_id}\t{float(cache['best_cost']):.3f}\t{best_cost:.3f}\t"
                    f"{'IMPROVED' if replaced else 'kept'}\n")
        return (seg_id, float(cache["best_cost"]), best_cost,
                "IMPROVED" if replaced else "kept")
    except Exception as e:
        with open(_PROGRESS, "a") as f:
            f.write(f"{seg_id}\tERROR\t{e}\n")
        return (seg_id, None, None, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--controllers", default="pid,enhanced_pid,preview,tdof")
    parser.add_argument("--weights", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--progress_file", default="blend_progress.tsv")
    parser.add_argument("--worst_first", action="store_true", default=True)
    args = parser.parse_args()

    controllers = [c.strip() for c in args.controllers.split(",")]
    weights = [float(w) for w in args.weights.split(",")]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data_path).iterdir())[:args.num_segs]

    if args.worst_first:
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

    work = [(str(f), args.opt_actions, args.out_dir, controllers, weights)
            for f in files]
    print(f"Blend on {len(files)}  workers={args.workers}  "
          f"controllers={controllers}  weights={weights}")

    t0 = time.time()
    with mp.Pool(processes=args.workers, initializer=_init,
                 initargs=(args.model_path, args.progress_file)) as pool:
        improved = 0
        n_done = 0
        total_before = 0.0
        total_after = 0.0
        for sg, cb, ca, res in pool.imap_unordered(_worker, work, chunksize=2):
            n_done += 1
            if cb is not None:
                total_before += cb
                total_after += ca
                if "IMPROVED" in res:
                    improved += 1
            if n_done % 100 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done)
                print(f"  [{n_done}/{len(work)}]  improved={improved}  "
                      f"savings_so_far={total_before-total_after:.1f}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min  improved={improved}/{len(work)}")
    print(f"  mean before: {total_before/n_done:.3f}")
    print(f"  mean after:  {total_after/n_done:.3f}")


if __name__ == "__main__":
    main()
