"""Per-step iLQR-style optimizer.

Avoids backpropagating through 400 autoregressive sim steps (chaos /
trajectory divergence). Instead:

  1. Run a full deterministic rollout (using sim seed + ST sample = exact
     match with the real ONNX simulator).
  2. At each step t we *separately* compute the local sensitivity
     g_t = ∂lat_accel[t+1]/∂action[t] from a fresh tiny autograd pass
     through that one model call (cheap: ~10 ms each).
  3. Treat the local cost as a quadratic in action[t]:
        L_t(a) ≈ 50 * (target[t+1] - (la[t+1] + g_t*(a - a_t)))^2
              +       ((la[t+1] + g_t*(a - a_t) - la[t]) / dt)^2
        d L_t / d a = 0  =>  closed-form optimal a*
  4. Damped update: a_t ← (1-α) * a_t + α * a*. Then re-roll.

Convergence is much steadier than full-rollout gradient descent because
each step's update only relies on the LOCAL linearization, not on
gradients chained through chaos.
"""
from __future__ import annotations

import argparse
import time
from hashlib import md5
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diffsim import (
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    LATACCEL_RANGE,
    LAT_ACCEL_COST_MULT,
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


def _build_inputs(all_actions, roll_t, vego_t, aego_t, lataccel_hist, step):
    a_window = all_actions[step - CONTEXT_LENGTH + 1 : step + 1]
    r_window = roll_t[step - CONTEXT_LENGTH + 1 : step + 1]
    v_window = vego_t[step - CONTEXT_LENGTH + 1 : step + 1]
    e_window = aego_t[step - CONTEXT_LENGTH + 1 : step + 1]
    states = torch.stack([a_window, r_window, v_window, e_window], dim=-1).unsqueeze(0)
    past_lat = lataccel_hist[step - CONTEXT_LENGTH : step]
    return states, a_window, past_lat


def rollout_with_grad(model, data, actions_np, seed, temperature=0.8):
    """Run a deterministic rollout (exact ONNX match) AND collect per-step
    local sensitivities g_t = ∂soft_pred[t+1]/∂action[t].

    Returns:
        lataccel_hist: np.ndarray (T,)
        sensitivities: np.ndarray (ACTION_HORIZON,), g_t for each t
        cost dict
    """
    np.random.seed(seed)
    device = "cpu"
    dtype = torch.float32
    bins = _bins(device, dtype)

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

    lataccel_hist = target_t.clone().detach().numpy()
    sensitivities = np.zeros(n_action, dtype=np.float32)
    lataccel_hist_t = torch.from_numpy(lataccel_hist)

    end = min(COST_END_IDX, T)
    for step in range(CONTEXT_LENGTH, end):
        in_control = step >= CONTROL_START_IDX

        if in_control:
            # Build inputs with the last action as the *only* tensor with grad.
            # This restricts autograd to a tiny subgraph (one model call).
            a_window_static = all_actions[step - CONTEXT_LENGTH + 1 : step].detach()
            a_last = all_actions[step:step + 1].detach().clone().requires_grad_(True)
            a_window = torch.cat([a_window_static, a_last])
            r_window = roll_t[step - CONTEXT_LENGTH + 1 : step + 1]
            v_window = vego_t[step - CONTEXT_LENGTH + 1 : step + 1]
            e_window = aego_t[step - CONTEXT_LENGTH + 1 : step + 1]
            states = torch.stack([a_window, r_window, v_window, e_window], dim=-1).unsqueeze(0)
            past_lat = lataccel_hist_t[step - CONTEXT_LENGTH : step].detach()
            tokens = _encode_tokens(past_lat, bins).unsqueeze(0)

            logits = model(states, tokens)
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            soft_pred = (probs * bins).sum()
            # Local sensitivity
            soft_pred.backward()
            g_t = float(a_last.grad)
            sensitivities[step - CONTROL_START_IDX] = g_t

            # Hard sample for forward (exact match with real sim)
            with torch.no_grad():
                probs_np = probs.detach().cpu().numpy()
                sample_idx = int(np.random.choice(VOCAB_SIZE, p=probs_np))
                hard_pred = float(bins[sample_idx])
            prev = float(lataccel_hist_t[step - 1])
            pred = float(np.clip(hard_pred, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA))
            lataccel_hist_t[step] = pred
        else:
            with torch.no_grad():
                a_window = all_actions[step - CONTEXT_LENGTH + 1 : step + 1]
                r_window = roll_t[step - CONTEXT_LENGTH + 1 : step + 1]
                v_window = vego_t[step - CONTEXT_LENGTH + 1 : step + 1]
                e_window = aego_t[step - CONTEXT_LENGTH + 1 : step + 1]
                states = torch.stack([a_window, r_window, v_window, e_window], dim=-1).unsqueeze(0)
                past_lat = lataccel_hist_t[step - CONTEXT_LENGTH : step]
                tokens = _encode_tokens(past_lat, bins).unsqueeze(0)
                logits = model(states, tokens)
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                # Advance RNG state, even though prediction is discarded
                probs_np = probs.detach().cpu().numpy()
                _ = int(np.random.choice(VOCAB_SIZE, p=probs_np))

    lataccel_hist = lataccel_hist_t.detach().numpy()
    tgt = data["target_lataccel"][CONTROL_START_IDX:COST_END_IDX]
    prd = lataccel_hist[CONTROL_START_IDX:COST_END_IDX]
    lat_cost = float(((tgt - prd) ** 2).mean() * 100.0)
    jerk = (prd[1:] - prd[:-1]) / DEL_T
    jerk_cost = float((jerk ** 2).mean() * 100.0)
    total_cost = LAT_ACCEL_COST_MULT * lat_cost + jerk_cost
    return lataccel_hist, sensitivities, {
        "lataccel_cost": lat_cost,
        "jerk_cost": jerk_cost,
        "total_cost": total_cost,
    }


def per_step_update(actions, lataccel_hist, target, sensitivities,
                    jerk_pen=1.0, max_step=0.1, alpha=1.0):
    """Compute the locally-optimal action update at each step.

    L_t(a) = 50 * (target[t+1] - (la[t+1] + g*(a - a_t)))^2
           + jerk_pen * ((la[t+1] + g*(a - a_t) - la[t]) / dt)^2

    d/da = 0  =>
       a* = a_t + ( 50*g*(target[t+1] - la[t+1]) - jerk_pen*g*(la[t+1] - la[t])/dt^2 )
                  / ( g^2 * (50 + jerk_pen/dt^2) )

    Applies damped step and clip.
    """
    n = len(actions)
    new_actions = actions.copy()
    for i in range(n):
        t_la = CONTROL_START_IDX + i        # lat_accel index at "t"
        t_next = CONTROL_START_IDX + i + 1  # lat_accel index at "t+1"
        if t_next >= len(lataccel_hist):
            break
        g = float(sensitivities[i])
        if abs(g) < 1e-6:
            continue
        la_next = float(lataccel_hist[t_next])
        la_now = float(lataccel_hist[t_la])
        tgt_next = float(target[t_next])

        # closed-form optimum for the local quadratic
        # numerator: 50*g*(tgt - la_next) - jerk_pen*g*(la_next - la_now)/dt^2
        # denominator: g^2 * (50 + jerk_pen/dt^2)
        num = 50.0 * g * (tgt_next - la_next) - jerk_pen * g * (la_next - la_now) / (DEL_T ** 2)
        den = (g * g) * (50.0 + jerk_pen / (DEL_T ** 2))
        delta = num / max(den, 1e-9)
        delta = float(np.clip(delta * alpha, -max_step, max_step))
        new_actions[i] = float(np.clip(actions[i] + delta, *STEER_RANGE))
    return new_actions


def ilqr_optimize_segment(
    model,
    csv_path,
    init_actions=None,
    n_iters=8,
    jerk_pen=1.0,
    init_alpha=1.0,
    max_step=0.15,
    verbose=False,
):
    # Normalise path so the sim seed matches the real sim's path-hashed seed.
    csv_path = str(Path(csv_path))
    data = load_segment(csv_path)
    n_action = ACTION_HORIZON
    if init_actions is None:
        init_actions = np.zeros(n_action, dtype=np.float32)
    elif len(init_actions) < n_action:
        init_actions = np.pad(init_actions, (0, n_action - len(init_actions)))
    actions = init_actions[:n_action].astype(np.float32).copy()
    seed = sim_seed_from_path(csv_path)
    target = data["target_lataccel"]

    # Initial cost
    lat_hist, sens, cost0 = rollout_with_grad(model, data, actions, seed)
    best_cost = cost0["total_cost"]
    best_actions = actions.copy()
    if verbose:
        print(f"init  total={best_cost:.3f}  lat={cost0['lataccel_cost']:.3f}  "
              f"jerk={cost0['jerk_cost']:.3f}  |g|_mean={np.abs(sens).mean():.3f}")

    alpha = init_alpha
    fail = 0
    for it in range(n_iters):
        new_actions = per_step_update(actions, lat_hist, target, sens,
                                       jerk_pen=jerk_pen, max_step=max_step, alpha=alpha)
        lat_hist_new, sens_new, cost_new = rollout_with_grad(model, data, new_actions, seed)
        if cost_new["total_cost"] < best_cost - 1e-6:
            best_cost = cost_new["total_cost"]
            best_actions = new_actions.copy()
            actions = new_actions
            lat_hist, sens = lat_hist_new, sens_new
            alpha = min(alpha * 1.1, init_alpha * 1.5)
            fail = 0
            if verbose:
                print(f"it={it:02d} total={best_cost:7.3f}  lat={cost_new['lataccel_cost']:.3f}  "
                      f"jerk={cost_new['jerk_cost']:.3f}  alpha={alpha:.3f}  *")
        else:
            alpha *= 0.5
            fail += 1
            if verbose:
                print(f"it={it:02d} total={cost_new['total_cost']:7.3f} "
                      f"(reject; alpha->{alpha:.3f})")
            if alpha < 0.05:
                if verbose:
                    print("  [early stop: alpha too small]")
                break
    return best_actions, best_cost, cost0["total_cost"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--onnx", default="./models/tinyphysics.onnx")
    parser.add_argument("--warm", default=None, help=".npz with init actions")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--jerk_pen", type=float, default=1.0)
    parser.add_argument("--max_step", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model = load_torch_sim(args.onnx)
    print("Model loaded")

    init = None
    if args.warm:
        init = np.load(args.warm)["actions"]

    t0 = time.time()
    actions, best_cost, init_cost = ilqr_optimize_segment(
        model, args.csv,
        init_actions=init,
        n_iters=args.iters,
        jerk_pen=args.jerk_pen,
        max_step=args.max_step,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    print(f"init={init_cost:.3f}  best={best_cost:.3f}  gain={init_cost - best_cost:+.3f}  t={dt:.1f}s")
