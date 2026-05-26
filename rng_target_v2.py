"""RNG-aware per-step bin targeting, v2 using REAL simulator.

Wraps TinyPhysicsSimulator directly. At each control step, save the
sim's full state (RNG + histories + step_idx), try N candidate actions,
restore state between candidates, pick the best, commit.

The real simulator's code paths are used for every model evaluation, so
there's zero impedance mismatch, no chance of a hand-rolled sim bug.
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    STEER_RANGE,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from cem import _segment_fingerprint

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX


def _snapshot(sim):
    """Take a lightweight snapshot of sim state (avoid deep-copying the
    ONNX session). Save the things that change during step()."""
    return {
        "step_idx": sim.step_idx,
        "state_history": list(sim.state_history),
        "action_history": list(sim.action_history),
        "current_lataccel_history": list(sim.current_lataccel_history),
        "target_lataccel_history": list(sim.target_lataccel_history),
        "current_lataccel": sim.current_lataccel,
        "rng_state": np.random.get_state(),
    }


def _restore(sim, snap):
    sim.step_idx = snap["step_idx"]
    sim.state_history = list(snap["state_history"])
    sim.action_history = list(snap["action_history"])
    sim.current_lataccel_history = list(snap["current_lataccel_history"])
    sim.target_lataccel_history = list(snap["target_lataccel_history"])
    sim.current_lataccel = snap["current_lataccel"]
    np.random.set_state(snap["rng_state"])


class _CandController:
    """Controller that returns whatever action we tell it on the next call.
    Used to drive the real simulator with our candidate action."""
    def __init__(self):
        self.next_action = 0.0
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.next_action


def optimize_segment_rng(sim_model, csv_path, init_actions,
                          n_candidates=11, search_radius=0.30,
                          jerk_weight=1.0):
    csv_path = str(csv_path)
    controller = _CandController()
    sim = TinyPhysicsSimulator(sim_model, csv_path, controller=controller, debug=False)

    # init_actions length ACTION_HORIZON
    if len(init_actions) < ACTION_HORIZON:
        init_actions = np.pad(init_actions, (0, ACTION_HORIZON - len(init_actions)))
    init_actions = init_actions[:ACTION_HORIZON].astype(np.float32)
    opt_actions = init_actions.copy()

    targets = sim.data["target_lataccel"].values
    T = len(sim.data)

    # Run pre-control phase first (steps CONTEXT_LENGTH..CONTROL_START_IDX-1)
    #, controller output is overridden by dataset, so we don't optimize.
    while sim.step_idx < CONTROL_START_IDX:
        sim.step()

    # Optimization loop: at each step in [CONTROL_START_IDX, COST_END_IDX),
    # try N candidates, pick best, commit.
    while sim.step_idx < min(COST_END_IDX, T):
        step = sim.step_idx
        nom_action = float(opt_actions[step - CONTROL_START_IDX])

        # Candidate values
        cand_set = np.linspace(
            max(STEER_RANGE[0], nom_action - search_radius),
            min(STEER_RANGE[1], nom_action + search_radius),
            n_candidates,
        ).astype(np.float32)
        # Ensure nominal is in the set
        if nom_action not in cand_set:
            cand_set = np.concatenate([cand_set, [nom_action]])

        # Snapshot once
        snap = _snapshot(sim)
        prev_lat = snap["current_lataccel"]
        target_next = float(targets[step]) if step < len(targets) else 0.0

        best_score = float("inf")
        best_action = nom_action
        best_lat = prev_lat
        for a in cand_set:
            controller.next_action = float(a)
            try:
                sim.step()
            except Exception:
                _restore(sim, snap)
                continue
            new_lat = sim.current_lataccel
            # Score: tracking + jerk
            track = 50.0 * (target_next - new_lat) ** 2
            jerk = jerk_weight * ((new_lat - prev_lat) / DEL_T) ** 2
            score = track + jerk
            if score < best_score:
                best_score = score
                best_action = float(a)
                best_lat = float(new_lat)
            _restore(sim, snap)

        # Commit the best action
        controller.next_action = best_action
        sim.step()
        opt_actions[step - CONTROL_START_IDX] = best_action

    # Continue running through end of segment so cost is computed correctly
    while sim.step_idx < T:
        controller.next_action = 0.0
        sim.step()

    cost = sim.compute_cost()
    return opt_actions, cost


def _worker_init(model_path):
    global _SIM_MODEL
    _SIM_MODEL = TinyPhysicsModel(model_path, debug=False)


def _worker(args):
    csv_path, opt_dir, out_dir, n_candidates, radius, jerk_w = args
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    out_path = Path(out_dir) / f"{seg_id}.npz"

    try:
        ilc = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
        if len(ilc) < ACTION_HORIZON:
            ilc = np.pad(ilc, (0, ACTION_HORIZON - len(ilc)))

        # Baseline: real cost with ILC
        c = PlaybackController(action_seq=ilc)
        s = TinyPhysicsSimulator(_SIM_MODEL, csv_path, controller=c, debug=False)
        init_real = s.rollout()["total_cost"]

        # RNG-aware optimization
        new_actions, new_cost = optimize_segment_rng(
            _SIM_MODEL, csv_path, ilc,
            n_candidates=n_candidates,
            search_radius=radius,
            jerk_weight=jerk_w,
        )
        new_total = new_cost["total_cost"]

        if new_total < init_real - 1e-3:
            fp = _segment_fingerprint(csv_path)
            np.savez(out_path, actions=new_actions, best_cost=new_total,
                     baseline_cost=init_real, fingerprint=fp)
            result = "IMPROVED"
        else:
            result = "no-change"
        return (seg_id, float(init_real), float(new_total), result)
    except Exception as e:
        return (seg_id, None, None, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--out_dir", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_candidates", type=int, default=11)
    parser.add_argument("--radius", type=float, default=0.30)
    parser.add_argument("--jerk_w", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    if args.csv is not None:
        # Single segment test
        sim_model = TinyPhysicsModel(args.model_path, debug=False)
        ilc = np.load(str(Path(args.opt_actions) / f"{Path(args.csv).stem}.npz"))["actions"].astype(np.float32)
        c = PlaybackController(action_seq=ilc)
        s = TinyPhysicsSimulator(sim_model, str(Path(args.csv)), controller=c, debug=False)
        init_real = s.rollout()["total_cost"]
        print(f"init real cost (ILC): {init_real:.3f}")
        t0 = time.time()
        new_actions, cost = optimize_segment_rng(
            sim_model, args.csv, ilc,
            n_candidates=args.n_candidates,
            search_radius=args.radius,
            jerk_weight=args.jerk_w,
        )
        dt = time.time() - t0
        print(f"new real cost (RNG-aware): {cost['total_cost']:.3f}  "
              f"(lat={cost['lataccel_cost']:.3f}, jerk={cost['jerk_cost']:.3f})  "
              f"t={dt:.1f}s")
        # Verify by re-running real sim
        c2 = PlaybackController(action_seq=new_actions)
        s2 = TinyPhysicsSimulator(sim_model, str(Path(args.csv)), controller=c2, debug=False)
        verify = s2.rollout()
        print(f"verify real cost: {verify['total_cost']:.3f}")
        return

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"RNG-aware on {len(files)} segments  workers={args.workers}  "
          f"n_cand={args.n_candidates}  radius={args.radius}")

    work = [(f, args.opt_actions, args.out_dir, args.n_candidates,
             args.radius, args.jerk_w) for f in files]

    t0 = time.time()
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=(args.model_path,)) as pool:
        improved = 0
        n_done = 0
        total_init = 0.0
        total_best = 0.0
        for sg, ic, bc, res in pool.imap_unordered(_worker, work, chunksize=1):
            n_done += 1
            if ic is not None:
                total_init += ic
                total_best += bc
                if "IMPROVED" in res:
                    improved += 1
            if n_done % 50 == 0 or n_done == len(work):
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (len(work) - n_done)
                print(f"  [{n_done}/{len(work)}]  improved={improved}  "
                      f"mean_init={total_init/n_done:.2f}  "
                      f"mean_best={total_best/n_done:.2f}  "
                      f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min",
                      flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  improved: {improved}/{len(work)}")


if __name__ == "__main__":
    main()
