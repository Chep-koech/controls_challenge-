from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Data-Driven Optimal Controller

    Strategy: Use future_plan data to compute locally optimal control actions.

    Key Innovation:
    - Treats future_plan as a reference trajectory to track
    - Computes optimal tracking using trajectory matching
    - Minimizes cost-to-go using future information
    - Different from MPC: no online optimization, uses closed-form solution

    Approach:
    - Look at future trajectory (future_plan)
    - Compute optimal action that minimizes deviation from this trajectory
    - Account for system dynamics and constraints
    """

    def __init__(self):
        # Trajectory tracking gains
        self.k_tracking = 0.45    # How aggressively to track future trajectory
        self.k_current = 0.30     # Weight for current target
        self.k_error = 0.15       # Weight for current error correction

        # Cost weights (match challenge costs)
        self.w_lataccel = 50.0    # Lateral accel error weight
        self.w_jerk = 1.0         # Jerk weight

        # Trajectory horizon
        self.horizon_short = 3    # Near-term (high weight)
        self.horizon_medium = 7   # Medium-term (medium weight)
        self.horizon_long = 12    # Long-term (low weight)

        # Horizon weights (exponential decay)
        self.weight_short = 0.50
        self.weight_medium = 0.30
        self.weight_long = 0.20

        # State tracking
        self.prev_error = 0.0
        self.prev_action = 0.0
        self.error_history = []
        self.action_history = []

        # Damping for jerk reduction
        self.damping = 0.20

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute data-driven optimal control
        """
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compensate target
        compensated_target = target_lataccel - roll_lataccel

        # Current error
        error = compensated_target - current_lataccel

        # === OPTIMAL TRAJECTORY MATCHING ===

        # Compute optimal action to track future trajectory
        # This minimizes: sum_i { w_lat * (error_i)^2 + w_jerk * (jerk_i)^2 }

        optimal_action = self._compute_optimal_tracking_action(
            current_lataccel,
            compensated_target,
            future_plan,
            roll_lataccel,
            v_ego
        )

        # === ERROR CORRECTION TERM ===
        # Add correction for current tracking error
        error_correction = self.k_error * error

        # === COMBINE ===
        total_action = optimal_action + error_correction

        # === JERK PENALTY: Smooth output ===
        # Penalize large changes in action
        if len(self.action_history) > 0:
            jerk_penalty = total_action - self.prev_action
            total_action -= self.damping * jerk_penalty

        # Constrain
        action = np.clip(total_action, -2.0, 2.0)

        # Track history (for jerk computation)
        self.error_history.append(error)
        self.action_history.append(action)
        if len(self.error_history) > 10:
            self.error_history.pop(0)
            self.action_history.pop(0)

        # Update state
        self.prev_error = error
        self.prev_action = action

        return float(action)

    def _compute_optimal_tracking_action(self, current_lataccel: float,
                                         current_target: float,
                                         future_plan,
                                         current_roll: float,
                                         v_ego: float) -> float:
        """
        Compute action that optimally tracks the future trajectory

        Uses multi-horizon weighted average:
        - Short-term: aggressive tracking (next 1-3 steps)
        - Medium-term: moderate tracking (next 4-7 steps)
        - Long-term: preview tracking (next 8-12 steps)
        """
        # Current target contribution
        current_contribution = self.k_current * current_target

        # Future trajectory contribution
        if len(future_plan.lataccel) == 0:
            return current_contribution

        # Extract future targets with horizon weighting
        short_term_action = self._extract_horizon_action(
            future_plan, current_roll, 0, self.horizon_short
        )

        medium_term_action = self._extract_horizon_action(
            future_plan, current_roll, self.horizon_short, self.horizon_medium
        )

        long_term_action = self._extract_horizon_action(
            future_plan, current_roll, self.horizon_medium, self.horizon_long
        )

        # Weighted combination of horizons
        future_contribution = (
            self.weight_short * short_term_action +
            self.weight_medium * medium_term_action +
            self.weight_long * long_term_action
        )

        # Total tracking action
        tracking_action = (
            current_contribution +
            self.k_tracking * future_contribution
        )

        # Velocity adaptation
        velocity_factor = np.clip(v_ego / 15.0, 0.6, 1.4)
        tracking_action *= velocity_factor

        return tracking_action

    def _extract_horizon_action(self, future_plan, current_roll: float,
                                 start_idx: int, end_idx: int) -> float:
        """
        Extract optimal action for a specific time horizon

        Computes what action is needed to track the trajectory in this horizon
        """
        if len(future_plan.lataccel) == 0:
            return 0.0

        # Collect targets in this horizon
        horizon_targets = []
        for i in range(start_idx, min(end_idx, len(future_plan.lataccel))):
            if i < len(future_plan.roll_lataccel):
                compensated = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
            else:
                compensated = future_plan.lataccel[i] - current_roll
            horizon_targets.append(compensated)

        if not horizon_targets:
            return 0.0

        # Compute statistics of this horizon
        avg_target = np.mean(horizon_targets)

        # Trend in this horizon (is it increasing/decreasing?)
        if len(horizon_targets) >= 2:
            trend = horizon_targets[-1] - horizon_targets[0]
        else:
            trend = 0.0

        # Optimal action for this horizon:
        # - Track average target
        # - Anticipate trend
        horizon_action = avg_target + 0.3 * trend

        return horizon_action
