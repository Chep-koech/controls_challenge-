"""Multi-restart ILC optimizer with incremental saves.

For each segment:
  - Start 0: keep current cached actions (no work)
  - Start 1..K: perturb current best by Gaussian noise (varying σ),
    run ILC from there, keep whichever real cost is lowest.

For the worst-N segments (high cost), use more restarts and add a CEM
polish at the end.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from controllers._playback import Controller as PlaybackController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from cem import _segment_fingerprint
from ilc import ilc_optimize_segment, ACTION_HORIZON


_SIM_MODEL = None
_MODEL_PATH = None
_PROGRESS_FILE = None


def _init(model_path, progress_file):
    global _SIM_MODEL, _MODEL_PATH, _PROGRESS_FILE
    _SIM_MODEL = TinyPhysicsModel(model_path, debug=False)
    _MODEL_PATH = model_path
    _PROGRESS_FILE = progress_file


def _real_cost(csv_path, actions):
    c = PlaybackController(action_seq=actions)
    s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=c, debug=False)
    return s.rollout()["total_cost"]


def _worker(args):
    csv_path, opt_dir, out_dir, n_restarts, ilc_iters, ilc_lr, noise_sigmas = args
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    out_path = Path(out_dir) / f"{seg_id}.npz"

    try:
        # Load current best
        cache = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))
        cur_actions = cache["actions"].astype(np.float32)
        if len(cur_actions) < ACTION_HORIZON:
            cur_actions = np.pad(cur_actions, (0, ACTION_HORIZON - len(cur_actions)))
        cur_actions = cur_actions[:ACTION_HORIZON]
        baseline = float(cache["baseline_cost"])

        best_real = _real_cost(csv_path, cur_actions)
        best_actions = cur_actions.copy()

        rng = np.random.default_rng(hash(seg_id) % (2**32))
        sigmas = noise_sigmas[:n_restarts] if len(noise_sigmas) >= n_restarts else (
            list(noise_sigmas) + [noise_sigmas[-1]] * (n_restarts - len(noise_sigmas))
        )

        for r in range(n_restarts):
            sigma = sigmas[r]
            init = np.clip(cur_actions + rng.normal(0, sigma, ACTION_HORIZON).astype(np.float32),
                           -2.0, 2.0)
            try:
                actions, cost, _, _ = ilc_optimize_segment(
                    csv_path, _MODEL_PATH,
                    n_iters=ilc_iters, lr=ilc_lr,
                    init_actions=init,
                    sim_model=_SIM_MODEL,
                    verbose=False,
                )
                cand_real = _real_cost(csv_path, actions.astype(np.float32))
                if cand_real < best_real - 1e-3:
                    best_real = cand_real
                    best_actions = actions.astype(np.float32)
            except Exception:
                pass

        if best_real < float(cache["best_cost"]) - 1e-3:
            fp = _segment_fingerprint(csv_path)
            np.savez(out_path, actions=best_actions, best_cost=best_real,
                     baseline_cost=baseline, fingerprint=fp)
            res = "IMPROVED"
        else:
            res = "no-change"

        with open(_PROGRESS_FILE, "a") as f:
            f.write(f"{seg_id}\t{baseline:.3f}\t{best_real:.3f}\t{res}\n")
        return (seg_id, baseline, best_real, res)
    except Exception as e:
        with open(_PROGRESS_FILE, "a") as f:
            f.write(f"{seg_id}\tERROR\t{e}\n")
        return (seg_id, None, None, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_restarts", type=int, default=2)
    parser.add_argument("--ilc_iters", type=int, default=40)
    parser.add_argument("--ilc_lr", type=float, default=0.1)
    parser.add_argument("--noise_sigmas", default="0.05,0.15")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--progress_file", default="multi_restart_progress.tsv")
    parser.add_argument("--worst_first", action="store_true",
                        help="Sort segments by current cost descending so worst go first")
    parser.add_argument("--summary", default="ilc_5000_summary.json")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]

    if args.worst_first:
        # Sort by current cost descending
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
        f.write("seg_id\tbaseline\tbest\tresult\n")

    noise_sigmas = [float(x) for x in args.noise_sigmas.split(",")]
    work = [(str(f), args.opt_actions, args.out_dir, args.n_restarts,
             args.ilc_iters, args.ilc_lr, noise_sigmas) for f in files]
    print(f"Multi-restart on {len(files)} segments  workers={args.workers}  "
          f"n_restarts={args.n_restarts}  ilc_iters={args.ilc_iters}")

    t0 = time.time()
    with mp.Pool(processes=args.workers, initializer=_init,
                 initargs=(args.model_path, args.progress_file)) as pool:
        improved = 0
        n_done = 0
        total_before = 0.0
        total_after = 0.0
        for sg, bc_before, bc_after, res in pool.imap_unordered(_worker, work, chunksize=2):
            n_done += 1
            if bc_before is not None and bc_after is not None:
                total_before += bc_before  # really `baseline_cost` from the cache; for current we use stored
                total_after += bc_after
                if "IMPROVED" in res:
                    improved += 1
            if n_done % 50 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done)
                print(f"  [{n_done}/{len(work)}]  improved={improved}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  improved: {improved}/{len(work)}")


if __name__ == "__main__":
    main()
