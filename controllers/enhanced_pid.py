from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Enhanced PID Controller with Feedforward and Gain Scheduling

    Improvements over basic PID:
    1. Feedforward term using future_plan to anticipate changes
    2. Velocity-based gain scheduling for adaptive behavior
    3. Road roll compensation using state information
    4. Jerk reduction through output filtering

    Target: Score < 50 initially, tune toward < 13
    """

    def __init__(self):
        # Base PID gains (back to best performing + small tweaks)
        self.kp_base = 0.32      # Slightly higher for better tracking
        self.ki_base = 0.06      # Small integral
        self.kd_base = 0.12      # Moderate derivative

        # Feedforward gain (anticipate future changes)
        self.kff = 0.28          # Strong feedforward for anticipation

        # Velocity-based gain scheduling parameters
        self.velocity_gain_min = 0.6
        self.velocity_gain_max = 1.4
        self.nominal_velocity = 15.0  # m/s

        # Jerk reduction filter
        self.output_filter_alpha = 0.25  # Light filtering

        # PID state
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

        # Anti-windup
        self.integral_max = 10.0

        # Lookahead for feedforward
        self.lookahead_steps = 5  # 0.5 seconds

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute steering action using enhanced PID

        Args:
            target_lataccel: Desired lateral acceleration
            current_lataccel: Current lateral acceleration
            state: Vehicle state (roll_lataccel, v_ego, a_ego)
            future_plan: Future trajectory plan

        Returns:
            Steering action in range [-2, 2]
        """
        # Extract state
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Velocity-based gain scheduling
        velocity_factor = self._compute_velocity_factor(v_ego)

        # Adjust gains based on velocity
        kp = self.kp_base * velocity_factor
        ki = self.ki_base * velocity_factor
        kd = self.kd_base * velocity_factor

        # Compensate target for road roll
        compensated_target = target_lataccel - roll_lataccel

        # Compute tracking error
        error = compensated_target - current_lataccel

        # PID terms
        p_term = kp * error

        self.error_integral += error
        # Anti-windup: limit integral
        self.error_integral = np.clip(self.error_integral, -self.integral_max, self.integral_max)
        i_term = ki * self.error_integral

        error_derivative = error - self.prev_error
        d_term = kd * error_derivative

        # Feedforward term: anticipate future target changes
        ff_term = self._compute_feedforward(future_plan, roll_lataccel, velocity_factor)

        # Combine all terms
        output = p_term + i_term + d_term + ff_term

        # Jerk reduction: low-pass filter on output
        filtered_output = (self.output_filter_alpha * output +
                          (1 - self.output_filter_alpha) * self.prev_output)

        # Apply constraints
        action = np.clip(filtered_output, -2.0, 2.0)

        # Update state
        self.prev_error = error
        self.prev_output = filtered_output

        return float(action)

    def _compute_velocity_factor(self, v_ego: float) -> float:
        """Compute gain scheduling factor based on velocity"""
        factor = v_ego / self.nominal_velocity
        return np.clip(factor, self.velocity_gain_min, self.velocity_gain_max)

    def _compute_feedforward(self, future_plan, current_roll: float, velocity_factor: float) -> float:
        """
        Compute feedforward term by anticipating future target changes

        Looks ahead in the future_plan to see what's coming and preemptively adjusts
        """
        if len(future_plan.lataccel) == 0:
            return 0.0

        # Look ahead a few steps
        lookahead = min(self.lookahead_steps, len(future_plan.lataccel))

        # Compute average future target (compensated for roll)
        future_targets = []
        for i in range(lookahead):
            compensated = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
            future_targets.append(compensated)

        avg_future_target = np.mean(future_targets)

        # Feedforward is proportional to anticipated target
        # Scale by velocity factor (need more aggressive action at higher speeds)
        ff = self.kff * avg_future_target * velocity_factor

        return ff
