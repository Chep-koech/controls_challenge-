"""Sanity-check the differentiable simulator against the real ONNX simulator.

Run both with the SAME action sequence on the SAME segment, compare:
  - lat_accel trajectory point-by-point
  - cost dict (total_cost, lataccel_cost, jerk_cost)
"""
import argparse

import numpy as np
import torch

from diffsim import (
    CONTROL_START_IDX,
    COST_END_IDX,
    load_segment,
    load_torch_sim,
    differentiable_rollout,
)
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers._playback import Controller as PlaybackController


def real_sim_cost(model_path, csv_path, action_seq):
    sim_model = TinyPhysicsModel(model_path, debug=False)
    controller = PlaybackController(action_seq=action_seq)
    sim = TinyPhysicsSimulator(sim_model, csv_path, controller=controller, debug=False)
    cost = sim.rollout()
    lataccel = np.array(sim.current_lataccel_history)
    return cost, lataccel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./data/00000.csv")
    parser.add_argument("--onnx", default="./models/tinyphysics.onnx")
    parser.add_argument("--init", default=None, help="optional .npz with cached actions")
    args = parser.parse_args()

    # action sequence
    n_action = COST_END_IDX - CONTROL_START_IDX
    if args.init:
        actions = np.load(args.init)["actions"].astype(np.float32)
        if len(actions) < n_action:
            actions = np.pad(actions, (0, n_action - len(actions)))
        actions = actions[:n_action]
    else:
        actions = np.zeros(n_action, dtype=np.float32)

    # Real ONNX sim
    print("Running real ONNX simulator...")
    real_cost, real_lataccel = real_sim_cost(args.onnx, args.csv, actions)
    print(f"  real lataccel_cost: {real_cost['lataccel_cost']:.4f}")
    print(f"  real jerk_cost:     {real_cost['jerk_cost']:.4f}")
    print(f"  real total_cost:    {real_cost['total_cost']:.4f}")

    # Differentiable sim
    print("Running differentiable PyTorch sim...")
    model = load_torch_sim(args.onnx)
    data = load_segment(args.csv)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    with torch.no_grad():
        diff_lataccel, lat_cost, jerk_cost, total_cost = differentiable_rollout(model, data, a_tensor)
    print(f"  diff lataccel_cost: {float(lat_cost):.4f}")
    print(f"  diff jerk_cost:     {float(jerk_cost):.4f}")
    print(f"  diff total_cost:    {float(total_cost):.4f}")

    # Compare trajectories
    diff_np = diff_lataccel.cpu().numpy()
    cost_window = slice(CONTROL_START_IDX, COST_END_IDX)
    abs_diff = np.abs(real_lataccel[cost_window] - diff_np[cost_window])
    print()
    print(f"Trajectory diff over cost window:")
    print(f"  max abs:  {abs_diff.max():.4f}")
    print(f"  mean abs: {abs_diff.mean():.4f}")
    print(f"  first 10 diffs at steps 100..109:")
    for i, k in enumerate(range(CONTROL_START_IDX, CONTROL_START_IDX + 10)):
        print(f"    step {k}: real={real_lataccel[k]:+.4f}  diff={diff_np[k]:+.4f}  err={abs_diff[i]:+.4f}")


if __name__ == "__main__":
    main()
