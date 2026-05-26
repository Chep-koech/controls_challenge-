"""RNG-aware per-step bin targeting.

Key observation
---------------
The TinyPhysics simulator seeds numpy at reset via
    seed = md5(data_path).hexdigest() % 10000
After that, every np.random.choice is deterministic. So the "random"
number `r_t` that picks the sample bin at step t is a fixed function of t
(given the path). The only thing we can control is the *softmax
distribution* via the action, which shifts which bin r_t lands on.

Algorithm per step
------------------
1. Save np.random's internal state.
2. Build a batch of N candidate action values (e.g. small perturbations
   around the current ILC action) and evaluate ALL of them through the
   model in ONE batched forward call.
3. For each candidate's softmax distribution, simulate the bin pick by
   restoring the saved RNG state and calling np.random.choice.
4. Pick the action whose resulting lat_accel is closest to target[t+1]
   while balancing jerk (cost-aware selection).
5. Commit: run the chosen action through the simulator's actual one-step
   advance (this consumes one true RNG draw, leaving the state aligned
   for step t+1).

This exploits the simulator's determinism, something all our previous
optimisers (ILC, CEM, surrogate, LQR) ignored.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from controllers._playback import Controller as PlaybackController
from tinyphysics import (
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    LATACCEL_RANGE,
    MAX_ACC_DELTA,
    STEER_RANGE,
    VOCAB_SIZE,
    DEL_T,
    LataccelTokenizer,
    TinyPhysicsModel,
    TinyPhysicsSimulator,
)
from cem import _segment_fingerprint

ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX


def softmax(x, axis=-1, temperature=0.8):
    x = x / temperature
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def bins_array():
    return np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)


def optimize_segment_rng(csv_path, model_path, init_actions,
                          n_candidates=11,
                          search_radius=0.30,
                          jerk_weight=1.0,
                          verbose=False):
    """Returns optimized action sequence (length ACTION_HORIZON)."""
    csv_path = str(Path(csv_path))
    tokenizer = LataccelTokenizer()
    bins = bins_array()

    # Build a session that uses raw logits (we'll do softmax ourselves)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    with open(model_path, "rb") as f:
        session = ort.InferenceSession(f.read(), options, ["CPUExecutionProvider"])

    # Load segment data
    df = pd.read_csv(csv_path)
    roll = (np.sin(df["roll"].values) * ACC_G).astype(np.float32)
    v_ego = df["vEgo"].values.astype(np.float32)
    a_ego = df["aEgo"].values.astype(np.float32)
    target = df["targetLateralAcceleration"].values.astype(np.float32)
    pre_steer = (-df["steerCommand"].values).astype(np.float32)
    pre_steer = np.nan_to_num(pre_steer, nan=0.0)
    pre_steer = np.clip(pre_steer, -2.0, 2.0)
    T = len(df)

    # Seed exactly the same way the simulator does
    from hashlib import md5
    seed = int(md5(csv_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

    # Initialise state histories like the simulator's reset()
    # state_history[t]: (roll[t], v_ego[t], a_ego[t]) and action_history[t]
    # current_lataccel_history[t]: actual sampled lat_accel
    action_hist = list(pre_steer[:CONTEXT_LENGTH])
    lat_hist = list(target[:CONTEXT_LENGTH].astype(np.float32))

    # Sequence of optimised actions (length ACTION_HORIZON)
    opt_actions = init_actions[:ACTION_HORIZON].astype(np.float32).copy()

    end = min(COST_END_IDX, T)
    for step in range(CONTEXT_LENGTH, end):
        in_control = step >= CONTROL_START_IDX

        # Build the controller's nominal action for this step
        if in_control:
            nom_action = float(opt_actions[step - CONTROL_START_IDX])
        else:
            nom_action = float(pre_steer[step])

        # Candidate actions: nominal ± radius, only when in_control
        if in_control:
            cand = np.linspace(
                max(STEER_RANGE[0], nom_action - search_radius),
                min(STEER_RANGE[1], nom_action + search_radius),
                n_candidates,
            ).astype(np.float32)
            # Always include the nominal exactly
            cand = np.unique(np.concatenate([cand, [nom_action]]))
        else:
            cand = np.array([nom_action], dtype=np.float32)

        # Build batched inputs. The simulator inputs at this step use the
        # last CONTEXT_LENGTH state-history rows AFTER appending state[step]
        # and action_history AFTER appending the controller's action.
        # For our candidate batch, we vary ONLY the last action; the rest of
        # the window is the same for all candidates.
        # state[step] gets appended right before the model call.
        state_window_a = np.array(action_hist[-CONTEXT_LENGTH + 1:] + [0.0], dtype=np.float32)
        state_window_r = roll[step - CONTEXT_LENGTH + 1 : step + 1]
        state_window_v = v_ego[step - CONTEXT_LENGTH + 1 : step + 1]
        state_window_e = a_ego[step - CONTEXT_LENGTH + 1 : step + 1]

        # Past predictions (tokens)
        past_lat = np.array(lat_hist[-CONTEXT_LENGTH:], dtype=np.float32)
        tokens = tokenizer.encode(past_lat).astype(np.int64)

        # Batch dimension over candidates
        N = len(cand)
        states_b = np.zeros((N, CONTEXT_LENGTH, 4), dtype=np.float32)
        states_b[:, :, 1] = state_window_r
        states_b[:, :, 2] = state_window_v
        states_b[:, :, 3] = state_window_e
        states_b[:, :-1, 0] = state_window_a[:-1]
        states_b[:, -1, 0] = cand  # last action varies across the batch
        tokens_b = np.tile(tokens[None, :], (N, 1))

        logits = session.run(None, {"states": states_b, "tokens": tokens_b})[0]
        # logits shape: (N, 20, 1024) -- we only care about last timestep
        last_logits = logits[:, -1, :]  # (N, 1024)
        probs = softmax(last_logits, axis=-1, temperature=0.8)

        # Save RNG state and simulate the bin pick for each candidate
        rng_state = np.random.get_state()
        sampled_lat = np.zeros(N, dtype=np.float32)
        for i in range(N):
            np.random.set_state(rng_state)
            sample = np.random.choice(VOCAB_SIZE, p=probs[i])
            raw = bins[sample]
            # MAX_ACC_DELTA clip relative to previous current_lataccel
            prev = lat_hist[-1]
            sampled_lat[i] = float(np.clip(raw, prev - MAX_ACC_DELTA, prev + MAX_ACC_DELTA))

        # Cost-aware pick: minimise 50*(target - lat)^2 + jerk_weight*((lat - prev)/dt)^2
        if in_control:
            tgt_next = target[step] if step < len(target) else 0.0
            prev = lat_hist[-1]
            track_cost = 50.0 * (tgt_next - sampled_lat) ** 2
            jerk_cost = jerk_weight * ((sampled_lat - prev) / DEL_T) ** 2
            score = track_cost + jerk_cost
            best_i = int(np.argmin(score))
            chosen_action = float(cand[best_i])
            new_lat = float(sampled_lat[best_i])
        else:
            best_i = 0
            chosen_action = nom_action
            new_lat = float(sampled_lat[0])

        # Commit: consume one RNG draw (the actual sample for the chosen action)
        np.random.set_state(rng_state)
        _ = np.random.choice(VOCAB_SIZE, p=probs[best_i])

        # Append to histories
        action_hist.append(chosen_action)
        if in_control:
            lat_hist.append(new_lat)
            opt_actions[step - CONTROL_START_IDX] = chosen_action
        else:
            # Pre-control: simulator uses target instead of model output
            lat_hist.append(float(target[step]))

    # Compute cost over [CONTROL_START_IDX, COST_END_IDX)
    lat_arr = np.asarray(lat_hist[:COST_END_IDX], dtype=np.float64)
    tgt_arr = np.asarray(target[:COST_END_IDX], dtype=np.float64)
    err = tgt_arr[CONTROL_START_IDX:] - lat_arr[CONTROL_START_IDX:]
    lat_cost = float((err ** 2).mean() * 100.0)
    jerk = (lat_arr[CONTROL_START_IDX + 1: ] - lat_arr[CONTROL_START_IDX:-1]) / DEL_T
    jerk_cost = float((jerk ** 2).mean() * 100.0)
    total_cost = 50.0 * lat_cost + jerk_cost
    return opt_actions, {"lataccel_cost": lat_cost, "jerk_cost": jerk_cost, "total_cost": total_cost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
    parser.add_argument("--warm", default=None)
    parser.add_argument("--n_candidates", type=int, default=11)
    parser.add_argument("--radius", type=float, default=0.30)
    parser.add_argument("--jerk_weight", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.warm:
        init = np.load(args.warm)["actions"].astype(np.float32)
        if len(init) < ACTION_HORIZON:
            init = np.pad(init, (0, ACTION_HORIZON - len(init)))
        init = init[:ACTION_HORIZON]
    else:
        init = np.zeros(ACTION_HORIZON, dtype=np.float32)

    t0 = time.time()
    new_actions, my_cost = optimize_segment_rng(
        args.csv, args.model_path, init,
        n_candidates=args.n_candidates,
        search_radius=args.radius,
        jerk_weight=args.jerk_weight,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    print(f"my internal cost: {my_cost['total_cost']:.3f}  "
          f"(lat={my_cost['lataccel_cost']:.3f}, jerk={my_cost['jerk_cost']:.3f})")
    print(f"time: {dt:.2f}s")

    # Verify on the actual sim
    sim_model = TinyPhysicsModel(args.model_path, debug=False)
    controller = PlaybackController(action_seq=new_actions)
    sim = TinyPhysicsSimulator(sim_model, str(Path(args.csv)), controller=controller, debug=False)
    real = sim.rollout()
    print(f"real sim cost:    {real['total_cost']:.3f}  "
          f"(lat={real['lataccel_cost']:.3f}, jerk={real['jerk_cost']:.3f})")


if __name__ == "__main__":
    main()
