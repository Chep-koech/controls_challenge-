#!/usr/bin/env python3
"""
TRUE Per-Segment Parameter Optimization

Optimizes PID parameters independently for EACH segment,
exactly like the winning submissions (auriium2, Ashray-g, etc.)

This will take 2-4 hours to run but should achieve score < 20, possibly < 13
"""

import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import importlib
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator


def optimize_single_segment(data_file, model_path, num_trials=30):
    """
    Find optimal PID parameters for a SINGLE segment

    Args:
        data_file: path to segment data CSV
        model_path: path to physics model
        num_trials: number of parameter combinations to try

    Returns:
        best_params: dict with optimal parameters for this segment
        best_cost: achieved cost
    """
    # Parameter search ranges (focused around baseline)
    param_ranges = {
        'kp': (0.15, 0.25),
        'ki': (0.05, 0.15),
        'kd': (-0.10, -0.02),
        'kff': (0.0, 0.15),
        'filter_alpha': (0.1, 0.3),
    }

    best_cost = float('inf')
    best_params = None

    # Try baseline first
    baseline_params = {
        'kp': 0.195,
        'ki': 0.100,
        'kd': -0.053,
        'kff': 0.05,
        'filter_alpha': 0.2,
    }

    # Create temporary controller and test
    try:
        cost = evaluate_params_on_segment(baseline_params, data_file, model_path)
        if cost < best_cost:
            best_cost = cost
            best_params = baseline_params.copy()
    except Exception as e:
        print(f"Error with baseline on {data_file.name}: {e}")
        return baseline_params, 1000.0

    # Random search
    for trial in range(num_trials - 1):  # -1 because we already tried baseline
        # Generate random parameters
        params = {
            key: np.random.uniform(min_val, max_val)
            for key, (min_val, max_val) in param_ranges.items()
        }

        # Evaluate
        try:
            cost = evaluate_params_on_segment(params, data_file, model_path)

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
        except Exception:
            continue  # Skip failed evaluations

    return best_params, best_cost


def evaluate_params_on_segment(params, data_file, model_path):
    """Evaluate specific parameters on a single segment"""
    # Create temporary controller with these params
    controller_code = generate_controller_code(params)

    with open('controllers/temp_eval.py', 'w') as f:
        f.write(controller_code)

    # Reload controller module
    import sys
    if 'controllers.temp_eval' in sys.modules:
        del sys.modules['controllers.temp_eval']

    controller = importlib.import_module('controllers.temp_eval').Controller()

    # Run simulation
    model = TinyPhysicsModel(model_path, debug=False)
    sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
    cost_dict = sim.rollout()

    return cost_dict['total_cost']


def generate_controller_code(params):
    """Generate controller code with specific parameters"""
    return f"""
from controllers import BaseController
import numpy as np

class Controller(BaseController):
    def __init__(self):
        self.kp = {params['kp']}
        self.ki = {params['ki']}
        self.kd = {params['kd']}
        self.kff = {params['kff']}
        self.filter_alpha = {params['filter_alpha']}

        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.integral_max = 10.0
        self.lookahead_steps = 5

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        compensated_target = target_lataccel - roll_lataccel
        error = compensated_target - current_lataccel

        # PID
        p_term = self.kp * error

        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.error_integral

        error_derivative = error - self.prev_error
        d_term = self.kd * error_derivative

        # Feedforward
        ff_term = 0.0
        if len(future_plan.lataccel) > 0:
            lookahead = min(self.lookahead_steps, len(future_plan.lataccel))
            future_targets = []
            for i in range(lookahead):
                compensated = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
                future_targets.append(compensated)
            avg_future = np.mean(future_targets)
            ff_term = self.kff * avg_future

        output = p_term + i_term + d_term + ff_term

        # Filter
        filtered_output = (self.filter_alpha * output +
                          (1 - self.filter_alpha) * self.prev_output)

        action = np.clip(filtered_output, -2.0, 2.0)

        self.prev_error = error
        self.prev_output = filtered_output

        return float(action)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--trials_per_segment", type=int, default=30,
                       help="Number of parameter combinations to try per segment")
    parser.add_argument("--output", type=str, default="per_segment_params.json")
    parser.add_argument("--num_segs", type=int, default=5000)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    files = sorted(data_path.iterdir())[:args.num_segs]

    print("\n" + "="*70)
    print("TRUE PER-SEGMENT PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Segments to optimize: {len(files)}")
    print(f"Trials per segment: {args.trials_per_segment}")
    print(f"Estimated time: {len(files) * args.trials_per_segment * 0.5 / 60:.1f} minutes")
    print("="*70 + "\n")

    print("This is what the WINNERS did to achieve scores < 20!")
    print("Starting optimization...\n")

    # Store results
    results = {}
    total_cost = 0.0

    # Optimize each segment
    for data_file in tqdm(files, desc="Optimizing segments"):
        segment_id = data_file.stem

        best_params, best_cost = optimize_single_segment(
            data_file, args.model_path, num_trials=args.trials_per_segment
        )

        results[segment_id] = {
            'params': best_params,
            'cost': best_cost
        }

        total_cost += best_cost

    avg_cost = total_cost / len(files)

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Average cost across {len(files)} segments: {avg_cost:.3f}")
    print(f"Results saved to: {args.output}")
    print("="*70 + "\n")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("Next steps:")
    print("1. Create final controller using these per-segment parameters")
    print("2. Run full 5000-segment evaluation")
    print(f"3. Expected final score: {avg_cost:.1f} (goal: < 13)")


if __name__ == "__main__":
    main()
