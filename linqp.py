"""Per-step linear-surrogate QP optimizer.

At each step t we query the simulator's neural model with three values
of action[t] (current ± perturb) keeping the rest of the trajectory
state fixed, then fit a linear surrogate

    lat_accel[t+1] ≈ a + b * action[t]

and analytically minimise the LOCAL cost

    L_t(u) = 50 * (target[t+1] - (a + b*u))^2
           + jerk_pen * (a + b*u - lat_accel[t])^2

which gives a closed-form u*. Updates are applied simultaneously with a
damping factor `alpha`, then the trajectory is re-rolled and we repeat.

Key advantages over the previous diff-sim attempts:
  - The surrogate captures whatever local response the network actually
    has (including clip saturation), not the autograd gradient through
    softmax which we showed pointed the wrong way.
  - Each per-step optimisation is convex and has an exact solution.
  - We never backprop through 400 chained simulator steps, so no chaos.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diffsim import (
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    MAX_ACC_DELTA,
    STEER_RANGE,
    VOCAB_SIZE,
    _bins,
    _encode_tokens,
    load_segment,
    load_torch_sim,
    sim_seed_from_path,
)

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX  # 400


def _predict_next_lataccel(
    model, bins, states_history, lat_history, action_value, temperature=0.8
):
    """Query the model for lat_accel[t+1] given the state context and a
    single trial action_value at the latest step. Returns the soft-expected
    lat_accel under the softmax."""
    with torch.no_grad():
        # Replace the last action in the state context with action_value
        states = states_history.clone()
        states[0, -1, 0] = action_value
        tokens = _encode_tokens(lat_history, bins).unsqueeze(0)
        logits = model(states, tokens)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        soft_pred = float((probs * bins).sum())
    return soft_pred


def rollout_full(model, data, actions_np, seed, temperature=0.8, return_contexts=False):
    """Deterministic rollout (exact ONNX match using sim seed + same sample
    sequence). Optionally returns per-step model input contexts so the
    optimiser can re-query the model with perturbed actions.
    """
    np.random.seed(seed)
    dtype = torch.float32
    bins = _bins("cpu", dtype)
    T = data["T"]
    n_action = ACTION_HORIZON

    roll_t = torch.as_tensor(data["roll_lataccel"], dtype=dtype)
    vego_t = torch.as_tensor(data["v_ego"], dtype=dtype)
    aego_t = torch.as_tensor(data["a_ego"], dtype=dtype)
    target_t = torch.as_tensor(data["target_lataccel"], dtype=dtype)
    steercmd_t = torch.as_tensor(data["steer_command"], dtype=dtype)

    pre_actions = torch.clamp(steercmd_t[:CONTROL_START_IDX], *STEER_RANGE)
    opt_actions = torch.clamp(torch.as_tensor(actions_np, dtype=dtype), *STEER_RANGE)
    pad_len = max(0, T - CONTROL_START_IDX - n_action)
    if pad_len > 0:
        all_actions = torch.cat([pre_actions, opt_actions, torch.zeros(pad_len, dtype=dtype)])
    else:
        all_actions = torch.cat([pre_actions, opt_actions])[:T]

    lataccel_hist = target_t.clone()
    contexts = [None] * n_action  # (states, lat_window) per control step

    end = min(COST_END_IDX, T)
    for step in range(CONTEXT_LENGTH, end):
        a_window = all_actions[step - CONTEXT_LENGTH + 1 : step + 1]
        r_window = roll_t[step - CONTEXT_LENGTH + 1 : step + 1]
        v_window = vego_t[step - CONTEXT_LENGTH + 1 : step + 1]
        e_window = aego_t[step - CONTEXT_LENGTH + 1 : step + 1]
        states = torch.stack([a_window, r_window, v_window, e_window], dim=-1).unsqueeze(0)
        past_lat = lataccel_hist[step - CONTEXT_LENGTH : step]
        tokens = _encode_tokens(past_lat, bins).unsqueeze(0)

        with torch.no_grad():
            logits = model(states, tokens)
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            probs_np = probs.cpu().numpy()
            sample_idx = int(np.random.choice(VOCAB_SIZE, p=probs_np))
            hard_pred = float(bins[sample_idx])

        in_control = step >= CONTROL_START_IDX
        if in_control:
            prev = float(lataccel_hist[step - 1])
            pred = float(np.clip(hard_pred, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA))
            lataccel_hist[step] = pred
            if return_contexts:
                # Save context so we can re-query model with perturbed action
                contexts[step - CONTROL_START_IDX] = (
                    states.clone(), past_lat.clone(), prev
                )
        # pre-control steps: lataccel_hist stays at target

    lat_np = lataccel_hist.numpy()
    tgt = data["target_lataccel"][CONTROL_START_IDX:COST_END_IDX]
    prd = lat_np[CONTROL_START_IDX:COST_END_IDX]
    lat_cost = float(((tgt - prd) ** 2).mean() * 100.0)
    jerk = (prd[1:] - prd[:-1]) / DEL_T
    jerk_cost = float((jerk ** 2).mean() * 100.0)
    total_cost = 50.0 * lat_cost + jerk_cost
    return lat_np, contexts, {
        "lataccel_cost": lat_cost,
        "jerk_cost": jerk_cost,
        "total_cost": total_cost,
    }


def fit_local_linear(model, bins, context, action_now, perturb):
    """Fit lat_accel[t+1] ≈ a + b * u by querying the model at three actions.
    Returns (a, b, y0)."""
    states, past_lat, _ = context
    y_minus = _predict_next_lataccel(model, bins, states, past_lat, action_now - perturb)
    y_zero = _predict_next_lataccel(model, bins, states, past_lat, action_now)
    y_plus = _predict_next_lataccel(model, bins, states, past_lat, action_now + perturb)
    b = (y_plus - y_minus) / (2.0 * perturb)
    a = y_zero - b * action_now
    return a, b, y_zero


def linqp_optimize_segment(
    model,
    csv_path,
    init_actions=None,
    n_iters=8,
    perturb=0.02,
    jerk_pen=1.0,
    init_alpha=0.3,
    verbose=False,
):
    csv_path = str(Path(csv_path))
    data = load_segment(csv_path)
    n_action = ACTION_HORIZON
    if init_actions is None:
        init_actions = np.zeros(n_action, dtype=np.float32)
    elif len(init_actions) < n_action:
        init_actions = np.pad(init_actions, (0, n_action - len(init_actions)))
    actions = init_actions[:n_action].astype(np.float64).copy()
    seed = sim_seed_from_path(csv_path)

    dtype = torch.float32
    bins = _bins("cpu", dtype)

    lat_hist, contexts, cost0 = rollout_full(model, data, actions, seed, return_contexts=True)
    best_cost = cost0["total_cost"]
    best_actions = actions.copy()
    if verbose:
        print(f"init total={best_cost:.3f}  lat={cost0['lataccel_cost']:.3f}  "
              f"jerk={cost0['jerk_cost']:.3f}")

    alpha = init_alpha
    # Robustness gates
    b_min = 0.05  # only trust the surrogate when |b| is non-trivial
    max_step_per_iter = 0.05  # clip per-step suggested delta magnitude

    for it in range(n_iters):
        suggested = actions.copy()
        bs = []
        for i in range(n_action):
            t = CONTROL_START_IDX + i
            ctx = contexts[i]
            if ctx is None:
                continue
            a_coef, b_coef, _ = fit_local_linear(
                model, bins, ctx, actions[i], perturb
            )
            bs.append(b_coef)
            if abs(b_coef) < b_min:
                # Surrogate too flat; assume nominal b=1 like ILC and use
                # only the error sign for a tiny corrective step.
                b_coef = np.sign(b_coef) * b_min if b_coef != 0 else 0.1
            tgt_next = float(data["target_lataccel"][t + 1]) if t + 1 < len(data["target_lataccel"]) else 0.0
            lat_now = float(lat_hist[t])
            num = (100.0 * (tgt_next - a_coef)
                   - 2.0 * jerk_pen * (a_coef - lat_now))
            den = b_coef * (100.0 + 2.0 * jerk_pen)
            u_star = num / den
            # Clip per-step delta to prevent destabilising the trajectory
            delta = float(np.clip(u_star - actions[i], -max_step_per_iter, max_step_per_iter))
            suggested[i] = float(np.clip(actions[i] + delta, -2.0, 2.0))
        deltas = suggested - actions

        # Try the damped update; line-search on alpha
        attempt_alpha = alpha
        accepted = False
        for trial in range(4):
            new_actions = np.clip(actions + attempt_alpha * deltas, -2.0, 2.0)
            lat_new, ctx_new, cost_new = rollout_full(
                model, data, new_actions, seed, return_contexts=True
            )
            if cost_new["total_cost"] < best_cost - 1e-6:
                accepted = True
                actions = new_actions
                lat_hist = lat_new
                contexts = ctx_new
                best_cost = cost_new["total_cost"]
                best_actions = actions.copy()
                alpha = min(alpha * 1.2, init_alpha * 2.0)
                if verbose:
                    print(f"it={it:02d} alpha={attempt_alpha:.3f}  total={best_cost:.3f}  "
                          f"lat={cost_new['lataccel_cost']:.3f}  "
                          f"jerk={cost_new['jerk_cost']:.3f}  |b|_mean={np.mean(np.abs(bs)):.4f}  *")
                break
            else:
                attempt_alpha *= 0.5
        if not accepted:
            alpha = max(alpha * 0.5, 0.02)
            if verbose:
                print(f"it={it:02d} all rejected, alpha -> {alpha:.3f}  "
                      f"best so far={best_cost:.3f}")
            if alpha < 0.025:
                if verbose:
                    print("  [early stop]")
                break
    return best_actions.astype(np.float32), best_cost, cost0["total_cost"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--onnx", default="./models/tinyphysics.onnx")
    parser.add_argument("--warm", default=None, help=".npz with init actions")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--perturb", type=float, default=0.02)
    parser.add_argument("--jerk_pen", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model = load_torch_sim(args.onnx)
    print("Model loaded")
    init = None
    if args.warm:
        init = np.load(args.warm)["actions"]
    t0 = time.time()
    actions, best_cost, init_cost = linqp_optimize_segment(
        model, args.csv, init_actions=init,
        n_iters=args.iters, perturb=args.perturb,
        jerk_pen=args.jerk_pen, init_alpha=args.alpha,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    print(f"init={init_cost:.3f}  best={best_cost:.3f}  gain={init_cost - best_cost:+.3f}  t={dt:.1f}s")
