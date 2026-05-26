"""Differentiable rollout of the TinyPhysics simulator in PyTorch.

The ONNX simulator emits a softmax over 1024 lat-accel bins. The original
simulator samples from this distribution (stochastic). For optimisation we
use the *expected* lat-accel under the softmax, this is fully
differentiable in the action sequence and removes sampling noise, while
matching the stochastic simulator in expectation.

Public entry points:
    load_torch_sim(onnx_path)               -> torch.nn.Module
    rollout(model, data, actions)           -> (lataccel_traj, cost_dict)
    optimize_segment(model, csv_path, ...)  -> (best_actions, best_cost)
"""
from __future__ import annotations

import math
from hashlib import md5
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnx2torch
import pandas as pd
import torch
import torch.nn.functional as F


def sim_seed_from_path(data_path: str) -> int:
    """Replicates TinyPhysicsSimulator.reset()'s seed."""
    return int(md5(data_path.encode()).hexdigest(), 16) % 10**4

# Mirror constants from tinyphysics.py
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULT = 50.0


def load_torch_sim(onnx_path: str) -> torch.nn.Module:
    m = onnx.load(onnx_path)
    m = onnx.shape_inference.infer_shapes(m)
    model = onnx2torch.convert(m)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_segment(csv_path: str):
    """Match TinyPhysicsSimulator.get_data preprocessing."""
    df = pd.read_csv(csv_path)
    roll_lataccel = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values
    a_ego = df["aEgo"].values
    target_lataccel = df["targetLateralAcceleration"].values
    # The simulator flips steer sign: right-positive
    steer_command = -df["steerCommand"].values
    return {
        "roll_lataccel": roll_lataccel.astype(np.float32),
        "v_ego": v_ego.astype(np.float32),
        "a_ego": a_ego.astype(np.float32),
        "target_lataccel": target_lataccel.astype(np.float32),
        "steer_command": steer_command.astype(np.float32),
        "T": int(len(df)),
    }


def _bins(device, dtype) -> torch.Tensor:
    return torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE,
                          device=device, dtype=dtype)


def _encode_tokens(lataccels: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Mirror LataccelTokenizer.encode: digitize with right=True after clip."""
    x = torch.clamp(lataccels, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
    # np.digitize(x, bins, right=True): returns index i such that
    # bins[i-1] < x <= bins[i].  bucketize(right=True) returns smallest i with
    # bins[i] >= x, which matches when x lies between bin edges.
    return torch.bucketize(x, bins, right=False)


def differentiable_rollout(
    model: torch.nn.Module,
    data: dict,
    actions: torch.Tensor,
    temperature: float = 0.8,
    seed: int | None = None,
):
    """Run a differentiable rollout. `actions` is shape (ACTION_HORIZON,)
    representing the controller's output for step indices
    [CONTROL_START_IDX, COST_END_IDX). Steps before CONTROL_START_IDX use the
    dataset's recorded steer_command; steps at and beyond CONTROL_START_IDX
    use `actions` (clamped to STEER_RANGE).

    Returns: (lataccel_traj, lataccel_cost, jerk_cost, total_cost).
    The cost dict mirrors TinyPhysicsSimulator.compute_cost (over
    [CONTROL_START_IDX, COST_END_IDX]).
    """
    device = actions.device
    dtype = actions.dtype
    bins = _bins(device, dtype)

    if seed is not None:
        np.random.seed(seed)

    T = data["T"]
    n_action = COST_END_IDX - CONTROL_START_IDX
    assert actions.shape == (n_action,), f"actions shape {actions.shape}"

    roll_t = torch.as_tensor(data["roll_lataccel"], device=device, dtype=dtype)
    vego_t = torch.as_tensor(data["v_ego"], device=device, dtype=dtype)
    aego_t = torch.as_tensor(data["a_ego"], device=device, dtype=dtype)
    target_t = torch.as_tensor(data["target_lataccel"], device=device, dtype=dtype)
    steercmd_t = torch.as_tensor(data["steer_command"], device=device, dtype=dtype)

    # Pre-build full action and current_lataccel histories.
    # action[t] for t < CONTROL_START_IDX: dataset steer (clamped)
    # action[t] for t >= CONTROL_START_IDX: optimisable (clamped)
    pre_actions = torch.clamp(steercmd_t[:CONTROL_START_IDX], *STEER_RANGE)
    opt_actions = torch.clamp(actions, *STEER_RANGE)

    # For steps past COST_END_IDX we pad with zeros (doesn't affect cost).
    pad_len = max(0, T - CONTROL_START_IDX - n_action)
    if pad_len > 0:
        all_actions = torch.cat([pre_actions, opt_actions, torch.zeros(pad_len, device=device, dtype=dtype)])
    else:
        all_actions = torch.cat([pre_actions, opt_actions])[:T]

    # Allocate trajectory: current_lataccel_history.
    # For step_idx < CONTROL_START_IDX, simulator uses target_lataccel as
    # the "current" (see TinyPhysicsSimulator.sim_step). For step_idx >=
    # CONTROL_START_IDX it uses the model's prediction (clipped).
    lataccel_hist = torch.zeros(T, device=device, dtype=dtype)
    # Fill in pre-control values: same as target up to CONTROL_START_IDX.
    # (Simulator reset() also sets early lataccel = target.)
    lataccel_hist[:CONTROL_START_IDX] = target_t[:CONTROL_START_IDX]

    # Roll the model from step_idx = CONTEXT_LENGTH..T-1, but we only
    # *need* trajectory inside the cost window [CONTROL_START_IDX..COST_END_IDX].
    # For step_idx < CONTROL_START_IDX, we just copy the target (per the
    # simulator's behaviour).
    # So we start meaningful integration at step_idx = CONTROL_START_IDX.

    # We do need to call the model from step_idx = CONTROL_START_IDX onward.
    # At each call, inputs use the last CONTEXT_LENGTH history values.
    end = min(COST_END_IDX, T)
    # Loop from CONTEXT_LENGTH to mirror real sim's call pattern, which
    # queries the model every step from CONTEXT_LENGTH onward. For
    # step_idx < CONTROL_START_IDX the prediction is computed but
    # discarded (current_lataccel = target). The RNG state still advances
    # though, so we must mirror those calls to keep sample sequences aligned.
    for step in range(CONTEXT_LENGTH, end):
        a_window = all_actions[step - CONTEXT_LENGTH + 1 : step + 1]
        r_window = roll_t[step - CONTEXT_LENGTH + 1 : step + 1]
        v_window = vego_t[step - CONTEXT_LENGTH + 1 : step + 1]
        e_window = aego_t[step - CONTEXT_LENGTH + 1 : step + 1]
        states = torch.stack([a_window, r_window, v_window, e_window], dim=-1)
        states = states.unsqueeze(0)

        past_lat = lataccel_hist[step - CONTEXT_LENGTH : step]
        tokens = _encode_tokens(past_lat, bins).unsqueeze(0)

        in_control = step >= CONTROL_START_IDX
        if in_control:
            logits = model(states, tokens)
        else:
            with torch.no_grad():
                logits = model(states, tokens)
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits / temperature, dim=-1)
        soft_pred = (probs * bins).sum()

        # Advance RNG state the same way the real sim does, so that if we
        # ever need to compare against the real sim, sample indices line up.
        with torch.no_grad():
            probs_np = probs.detach().cpu().numpy()
            _ = int(np.random.choice(VOCAB_SIZE, p=probs_np))

        if in_control:
            # Use the soft (expected) value in forward AND gradient, this
            # makes Adam steps actually change the trajectory.
            prev = lataccel_hist[step - 1]
            pred = torch.clamp(soft_pred, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA)
            lataccel_hist[step] = pred
        # Pre-control: lataccel_hist[step] already set to target_t[step]

    # Costs over [CONTROL_START_IDX, COST_END_IDX]
    tgt = target_t[CONTROL_START_IDX:COST_END_IDX]
    prd = lataccel_hist[CONTROL_START_IDX:COST_END_IDX]
    lataccel_cost = ((tgt - prd) ** 2).mean() * 100.0
    jerk = (prd[1:] - prd[:-1]) / DEL_T
    jerk_cost = (jerk ** 2).mean() * 100.0
    total_cost = LAT_ACCEL_COST_MULT * lataccel_cost + jerk_cost
    return lataccel_hist, lataccel_cost, jerk_cost, total_cost


def optimize_segment(
    model: torch.nn.Module,
    csv_path: str,
    init_actions: np.ndarray | None = None,
    n_steps: int = 60,
    lr: float = 0.05,
    temperature: float = 0.8,
    verbose: bool = False,
    use_sim_seed: bool = True,
):
    """Optimize a single segment by gradient descent on its action sequence.

    Returns: (best_actions [np.ndarray], best_cost [float], init_cost [float], history [list])
    """
    data = load_segment(csv_path)
    n_action = COST_END_IDX - CONTROL_START_IDX

    if init_actions is None:
        init_actions = np.zeros(n_action, dtype=np.float32)
    elif len(init_actions) < n_action:
        init_actions = np.pad(init_actions, (0, n_action - len(init_actions)))
    else:
        init_actions = init_actions[:n_action].astype(np.float32)

    a = torch.tensor(init_actions, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)

    seed = sim_seed_from_path(csv_path) if use_sim_seed else None

    with torch.no_grad():
        _, _, _, init_cost = differentiable_rollout(model, data, a, temperature, seed=seed)
        best_cost = float(init_cost)
    best_actions = init_actions.copy()

    history = []
    for it in range(n_steps):
        opt.zero_grad(set_to_none=True)
        _, lc, jc, tc = differentiable_rollout(model, data, a, temperature, seed=seed)
        tc.backward()
        opt.step()
        cur = float(tc.detach())
        improved = cur < best_cost - 1e-6
        if improved:
            best_cost = cur
            best_actions = a.detach().cpu().numpy().copy()
        history.append({"iter": it, "cost": cur, "lat": float(lc.detach()), "jerk": float(jc.detach())})
        if verbose:
            tag = "  *" if improved else ""
            print(f"  it={it:03d}  cost={cur:7.2f}  lat={float(lc):.3f}  jerk={float(jc):.2f}{tag}")
    return best_actions, best_cost, float(init_cost), history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--onnx", default="./models/tinyphysics.onnx")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm_start", default=None,
                        help="Path to a .npz with cached actions to warm-start from")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model = load_torch_sim(args.onnx)
    print("Model ready")

    init = None
    if args.warm_start:
        init = np.load(args.warm_start)["actions"]

    actions, best_cost, init_cost, _ = optimize_segment(
        model, args.csv,
        init_actions=init,
        n_steps=args.steps,
        lr=args.lr,
        verbose=args.verbose,
    )
    print(f"init_cost={init_cost:.2f}  best_cost={best_cost:.2f}  gain={init_cost - best_cost:+.2f}")
