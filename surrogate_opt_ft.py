"""Per-segment fine-tuned surrogate optimization.

The universal surrogate trained on 200 segments is too generic for any
individual segment. Strategy: for EACH segment, do a few real-sim queries
near the ILC actions to gather local samples, fine-tune the surrogate on
those samples (~50 steps), then Adam-optimize the actions through the
fine-tuned surrogate, periodically validating on the real sim.
"""
from __future__ import annotations

import argparse
import copy
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map

from controllers._playback import Controller as PlaybackController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from train_surrogate import (
    ACTION_HORIZON,
    TRAJ_LEN,
    N_FEATURES,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    LAT_COST_MULT,
    TrajSurrogate,
    compute_cost,
)
from surrogate_opt import load_surrogate, build_features, real_cost
from cem import _segment_fingerprint


def collect_local_samples(sim_model, csv_path, ilc_actions, n_samples=8, noise_std=0.05, rng=None):
    """Run real sim with ILC + small perturbations; return (actions, lat_traj) pairs."""
    if rng is None:
        rng = np.random.default_rng(123)
    samples = []
    # Always include the unperturbed ILC
    actions_list = [ilc_actions.copy()]
    for _ in range(n_samples - 1):
        noise = rng.normal(0, noise_std, size=ACTION_HORIZON).astype(np.float32)
        # Smooth slightly
        kern = np.ones(5) / 5.0
        noise = np.convolve(noise, kern, mode="same")
        a = np.clip(ilc_actions + noise, -2.0, 2.0).astype(np.float32)
        actions_list.append(a)

    for a in actions_list:
        controller = PlaybackController(action_seq=a)
        sim = TinyPhysicsSimulator(sim_model, csv_path, controller=controller, debug=False)
        sim.rollout()
        lat = np.asarray(sim.current_lataccel_history, dtype=np.float32)
        if len(lat) < TRAJ_LEN:
            lat = np.pad(lat, (0, TRAJ_LEN - len(lat)))
        samples.append((a, lat[:TRAJ_LEN]))
    return samples


def finetune_surrogate(surrogate, state_mean, state_std, csv_path, samples, n_steps=80, lr=1e-3):
    """Fine-tune a copy of the surrogate on local samples for this segment."""
    fts = copy.deepcopy(surrogate)
    fts.train()
    opt = torch.optim.Adam(fts.parameters(), lr=lr)

    # Build all feature/target tensors
    feats_list = []
    target_list = []
    target_seq = None
    for a, lat in samples:
        feats, target_seq = build_features(csv_path, a, state_mean, state_std)
        feats_list.append(feats)
        target_list.append(torch.from_numpy(lat[CONTROL_START_IDX:COST_END_IDX]))
    feats_b = torch.stack(feats_list)         # (n, TRAJ_LEN, 5)
    target_b = torch.stack(target_list)       # (n, ACTION_HORIZON)

    for it in range(n_steps):
        opt.zero_grad(set_to_none=True)
        pred = fts(feats_b)
        loss = F.mse_loss(pred, target_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fts.parameters(), 1.0)
        opt.step()
    fts.eval()
    return fts, float(loss.detach())


def optimize_segment(csv_path, model_path, surrogate_state, opt_dir,
                      n_local=8, ft_steps=80, ft_lr=1e-3,
                      opt_steps=40, opt_lr=0.02, validate_every=5):
    csv_path = str(Path(csv_path))
    seg_id = Path(csv_path).stem
    ilc = np.load(str(Path(opt_dir) / f"{seg_id}.npz"))["actions"].astype(np.float32)
    if len(ilc) < ACTION_HORIZON:
        ilc = np.pad(ilc, (0, ACTION_HORIZON - len(ilc)))
    sim_model = TinyPhysicsModel(model_path, debug=False)
    init_real = real_cost(sim_model, csv_path, ilc)

    # 1. Gather local samples
    samples = collect_local_samples(sim_model, csv_path, ilc, n_samples=n_local, noise_std=0.05)

    # 2. Fine-tune surrogate
    surrogate, state_mean, state_std = surrogate_state
    fts, ft_final_loss = finetune_surrogate(
        surrogate, state_mean, state_std, csv_path, samples, n_steps=ft_steps, lr=ft_lr,
    )

    # 3. Build static features + target window
    feats0, target_seq = build_features(csv_path, ilc, state_mean, state_std)
    target_window = target_seq[CONTROL_START_IDX:COST_END_IDX]

    # 4. Adam-optimize via fine-tuned surrogate, validate on real sim periodically
    a = torch.tensor(ilc, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a], lr=opt_lr)
    best_real = init_real
    best_actions = ilc.copy()

    for it in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        feats = feats0.clone()
        feats[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON, 0] = a
        pred = fts(feats.unsqueeze(0))
        cost = compute_cost(pred, target_window.unsqueeze(0)).squeeze()
        cost.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-2.0, 2.0)

        if (it + 1) % validate_every == 0 or it == opt_steps - 1:
            cand = a.detach().cpu().numpy().astype(np.float32)
            rc = real_cost(sim_model, csv_path, cand)
            if rc < best_real - 1e-3:
                best_real = rc
                best_actions = cand.copy()

    return {
        "seg_id": seg_id,
        "init_cost": float(init_real),
        "best_cost": float(best_real),
        "best_actions": best_actions,
        "ft_loss": ft_final_loss,
    }


def _worker(csv_path, **kwargs):
    return optimize_segment(csv_path, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--opt_actions", default="optimized_actions")
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--surrogate", default="surrogate.pt")
    parser.add_argument("--out_dir", default="optimized_actions_v2")
    parser.add_argument("--num_segs", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n_local", type=int, default=8)
    parser.add_argument("--ft_steps", type=int, default=80)
    parser.add_argument("--ft_lr", type=float, default=1e-3)
    parser.add_argument("--opt_steps", type=int, default=40)
    parser.add_argument("--opt_lr", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading surrogate...")
    surrogate_state = load_surrogate(args.surrogate)
    print("Loaded")

    files = sorted(Path(args.data_path).iterdir())[args.start : args.start + args.num_segs]
    files = [str(f) for f in files]
    print(f"Optimising {len(files)} segments  workers={args.workers}")

    fn = partial(
        _worker,
        model_path=args.model_path,
        surrogate_state=surrogate_state,
        opt_dir=args.opt_actions,
        n_local=args.n_local,
        ft_steps=args.ft_steps,
        ft_lr=args.ft_lr,
        opt_steps=args.opt_steps,
        opt_lr=args.opt_lr,
    )

    t0 = time.time()
    results = process_map(fn, files, max_workers=args.workers, chunksize=1, disable=False)
    dt = time.time() - t0

    # Save
    for r in results:
        if "best_actions" not in r:
            continue
        seg_id = r["seg_id"]
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
    print(f"Surrogate-FT opt done in {dt/60:.1f} min")
    print(f"  ILC init mean:           {init.mean():7.2f}")
    print(f"  surrogate-ft mean:       {best.mean():7.2f}")
    print(f"  improvement:             {(init.mean()-best.mean()):+.2f}  "
          f"({100*(init.mean()-best.mean())/init.mean():+.1f}%)")
    print(f"  segments improved:       {improved}/{len(results)}")


if __name__ == "__main__":
    main()
