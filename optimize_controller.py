#!/usr/bin/env python3
"""
Per-Segment Controller Optimization

Searches for optimal PID parameters across the dataset
to achieve score < 13 like the winning submissions
"""

import numpy as np
import argparse
from pathlib import Path
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tinyphysics import run_rollout
import json


def evaluate_parameters(params, files, model_path, num_samples=50):
    """
    Evaluate a set of PID parameters on sample segments

    Args:
        params: dict with 'kp', 'ki', 'kd', 'kff' keys
        files: list of data files
        model_path: path to model
        num_samples: number of segments to test on

    Returns:
        average total_cost
    """
    # Create temporary controller file with these parameters
    controller_code = f"""
from controllers import BaseController
import numpy as np

class Controller(BaseController):
    def __init__(self):
        self.kp = {params['kp']}
        self.ki = {params['ki']}
        self.kd = {params['kd']}
        self.kff = {params['kff']}
        self.filter_alpha = {params.get('filter_alpha', 0.2)}

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

    # Write temporary controller
    with open('controllers/temp_optimized.py', 'w') as f:
        f.write(controller_code)

    # Sample random segments for faster evaluation
    sample_files = np.random.choice(files, min(num_samples, len(files)), replace=False)

    # Run evaluation
    try:
        rollout_partial = partial(run_rollout, controller_type='temp_optimized',
                                 model_path=model_path, debug=False)
        results = process_map(rollout_partial, sample_files, max_workers=16,
                             chunksize=5, disable=True)
        costs = [result[0]['total_cost'] for result in results]
        avg_cost = np.mean(costs)
        return avg_cost
    except Exception as e:
        print(f"Error evaluating params {params}: {e}")
        return 1e6  # Return high cost on error


def grid_search_optimization(files, model_path, num_iterations=20):
    """
    Grid search + random search for optimal parameters
    """
    print("\n" + "="*60)
    print("PER-SEGMENT PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Optimizing across {len(files)} segments...")
    print(f"Iterations: {num_iterations}")
    print("="*60 + "\n")

    best_cost = float('inf')
    best_params = None

    # Start from baseline PID parameters (we know these work well: cost ~80)
    param_ranges = {
        'kp': (0.15, 0.25),      # Around baseline 0.195
        'ki': (0.08, 0.15),      # Around baseline 0.100
        'kd': (-0.08, -0.03),    # Around baseline -0.053
        'kff': (0.0, 0.15),      # Feedforward (new)
        'filter_alpha': (0.15, 0.35),  # Output filtering (new)
    }

    print("Starting from baseline PID (cost ~80)...")
    print("Searching parameter space...\n")

    for iteration in range(num_iterations):
        # Generate candidate parameters
        if iteration == 0:
            # Start with baseline
            params = {
                'kp': 0.195,
                'ki': 0.100,
                'kd': -0.053,
                'kff': 0.05,
                'filter_alpha': 0.2,
            }
        else:
            # Random search around best so far
            if best_params:
                # Search around best params
                params = {}
                for key, (min_val, max_val) in param_ranges.items():
                    if np.random.random() < 0.7:  # 70% chance to search near best
                        std = (max_val - min_val) * 0.15
                        params[key] = np.clip(
                            best_params[key] + np.random.normal(0, std),
                            min_val, max_val
                        )
                    else:  # 30% chance to explore
                        params[key] = np.random.uniform(min_val, max_val)
            else:
                # Pure random
                params = {key: np.random.uniform(min_val, max_val)
                         for key, (min_val, max_val) in param_ranges.items()}

        # Evaluate
        cost = evaluate_parameters(params, files, model_path, num_samples=100)

        # Update best
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
            print(f"[*] Iteration {iteration+1:2d}: NEW BEST! Cost = {cost:.3f}")
            print(f"    Parameters: kp={params['kp']:.4f}, ki={params['ki']:.4f}, "
                  f"kd={params['kd']:.4f}, kff={params['kff']:.4f}, "
                  f"filter={params['filter_alpha']:.4f}")
        else:
            print(f"    Iteration {iteration+1:2d}: Cost = {cost:.3f}")

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Best Cost: {best_cost:.3f}")
    print(f"Best Parameters:")
    for key, val in best_params.items():
        print(f"  {key}: {val:.6f}")
    print("="*60 + "\n")

    return best_params, best_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=str, default="optimized_params.json")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    files = sorted(data_path.iterdir())[:5000]  # Limit to 5000 segments for submission

    # Run optimization
    best_params, best_cost = grid_search_optimization(
        files, args.model_path, num_iterations=args.iterations
    )

    # Save results
    results = {
        'parameters': best_params,
        'cost': best_cost,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
    print("\nNext steps:")
    print("1. Review the optimized parameters above")
    print("2. Create final controller with these parameters")
    print("3. Run full 5000-segment evaluation")
