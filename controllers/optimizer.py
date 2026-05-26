from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Pure Optimization-Based Controller

    Strategy: At each timestep, solve a local optimization problem.

    Key Innovation:
    - Formulates control as optimization: minimize cost over short horizon
    - Cost = w_lataccel * tracking_error^2 + w_jerk * jerk^2
    - Uses analytical solution (closed-form) for speed
    - Different from MPC: simpler, no complex dynamics, just local optimization

    Optimization Problem (at each step):
    min_u { sum_i [ 50*(error_i)^2 + (jerk_i)^2 ] }

    Where:
    - error_i = predicted tracking error over horizon
    - jerk_i = control rate of change
    - u = control action to choose
    """

    def __init__(self):
        # Cost weights (match challenge)
        self.w_lataccel = 50.0
        self.w_jerk = 1.0

        # Optimization horizon (steps to look ahead)
        self.horizon = 5

        # System "model" parameters (simplified)
        # How much does control action affect lateral accel?
        self.control_effectiveness = 0.85  # action -> lataccel gain

        # Control constraints
        self.u_min = -2.0
        self.u_max = 2.0

        # State tracking
        self.prev_action = 0.0
        self.prev_error = 0.0

        # Regularization (prevent extreme actions)
        self.w_regularization = 0.05

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Solve optimization problem to find best control action
        """
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compensate target
        compensated_target = target_lataccel - roll_lataccel

        # Current error
        error = compensated_target - current_lataccel

        # === SOLVE OPTIMIZATION PROBLEM ===
        # Find action u that minimizes cost over horizon

        optimal_action = self._solve_optimal_control(
            current_lataccel,
            compensated_target,
            future_plan,
            roll_lataccel,
            v_ego
        )

        # Constrain
        action = np.clip(optimal_action, self.u_min, self.u_max)

        # Update state
        self.prev_error = error
        self.prev_action = action

        return float(action)

    def _solve_optimal_control(self, current_lataccel: float,
                               current_target: float,
                               future_plan,
                               current_roll: float,
                               v_ego: float) -> float:
        """
        Solve optimization problem analytically

        Cost function:
        J(u) = sum_i [ w_lat * (error_i)^2 + w_jerk * (u - u_prev)^2 ] + w_reg * u^2

        Where:
        - error_i depends on future targets and chosen action u
        - First term: tracking cost
        - Second term: jerk cost
        - Third term: regularization (prevent extreme actions)

        Analytical solution (derivative = 0):
        dJ/du = 0  =>  u_optimal = ...
        """

        # Build future targets
        future_targets = self._get_future_targets(future_plan, current_roll, current_target)

        # Predict future errors as function of action u
        # Simplified model: lataccel_next = current_lataccel + control_effectiveness * u
        #
        # Over horizon, we're trying to track future_targets
        # Assume dynamics: lataccel[k+1] = lataccel[k] + effectiveness * u

        # For simplicity, use analytical closed-form solution:
        # Optimal action trades off:
        # 1. Tracking error (want u to drive error to zero)
        # 2. Jerk (want u close to prev_action)
        # 3. Regularization (want u near zero)

        # Tracking term: what action drives error to zero?
        # error = target - current
        # We want: current + effectiveness * u ≈ avg_future_target
        avg_future_target = np.mean(future_targets) if future_targets else current_target

        # Ideal action for tracking (ignoring jerk)
        u_tracking = (avg_future_target - current_lataccel) / (self.control_effectiveness + 1e-6)

        # Account for velocity
        velocity_factor = np.clip(v_ego / 15.0, 0.6, 1.4)
        u_tracking *= velocity_factor

        # Now solve full optimization problem:
        # Cost = w_lat * (effectiveness * u - error)^2 + w_jerk * (u - u_prev)^2 + w_reg * u^2
        #
        # Taking derivative and setting to zero:
        # dJ/du = 2*w_lat*effectiveness*(effectiveness*u - error) +
        #         2*w_jerk*(u - u_prev) +
        #         2*w_reg*u = 0
        #
        # Solving for u:
        # (w_lat*eff^2 + w_jerk + w_reg)*u = w_lat*eff*error + w_jerk*u_prev
        #
        # u = (w_lat*eff*error + w_jerk*u_prev) / (w_lat*eff^2 + w_jerk + w_reg)

        eff = self.control_effectiveness
        error_avg = avg_future_target - current_lataccel

        numerator = (
            self.w_lataccel * eff * error_avg +
            self.w_jerk * self.prev_action
        )

        denominator = (
            self.w_lataccel * eff * eff +
            self.w_jerk +
            self.w_regularization
        ) + 1e-6  # Avoid division by zero

        u_optimal = numerator / denominator

        return u_optimal

    def _get_future_targets(self, future_plan, current_roll: float, current_target: float) -> list:
        """
        Extract future target trajectory
        """
        if len(future_plan.lataccel) == 0:
            return [current_target]

        future_targets = []
        max_horizon = min(self.horizon, len(future_plan.lataccel))

        for i in range(max_horizon):
            if i < len(future_plan.roll_lataccel):
                compensated = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
            else:
                compensated = future_plan.lataccel[i] - current_roll
            future_targets.append(compensated)

        return future_targets
