from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Adaptive Controller with Online Parameter Adjustment

    Strategy: Continuously adapt control parameters based on performance.

    Key Innovation:
    - Monitors tracking error and jerk in real-time
    - Adjusts gains online to minimize cost
    - Adapts to different segment characteristics automatically
    - Uses gradient descent on performance metric

    This is different from per-segment optimization because:
    - Adaptation happens ONLINE during control
    - No pre-training needed
    - Responds to unexpected situations
    """

    def __init__(self):
        # Initial control gains (will adapt)
        self.kp = 0.20
        self.ki = 0.08
        self.kd = 0.05
        self.kff = 0.12

        # Adaptation parameters
        self.learning_rate = 0.0001  # How fast to adapt
        self.adaptation_window = 10   # Steps to average performance

        # Gain limits (prevent instability)
        self.kp_range = (0.10, 0.40)
        self.ki_range = (0.01, 0.20)
        self.kd_range = (0.01, 0.15)
        self.kff_range = (0.05, 0.30)

        # Performance tracking for adaptation
        self.recent_errors = []
        self.recent_jerks = []
        self.recent_actions = []

        # PID state
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_action = 0.0
        self.integral_max = 10.0

        # Adaptation state
        self.steps = 0
        self.adapt_every = 20  # Adapt every N steps

        # Preview
        self.lookahead_steps = 5

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute control with adaptive gains
        """
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compensate target
        compensated_target = target_lataccel - roll_lataccel

        # Compute error
        error = compensated_target - current_lataccel

        # === ADAPTIVE PID + FEEDFORWARD ===

        # P term
        p_term = self.kp * error

        # I term with anti-windup
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.error_integral

        # D term
        error_derivative = error - self.prev_error
        d_term = self.kd * error_derivative

        # Feedforward from future plan
        ff_term = self._compute_feedforward(future_plan, roll_lataccel)

        # Combine terms
        output = p_term + i_term + d_term + ff_term

        # Velocity adaptation
        velocity_factor = np.clip(v_ego / 15.0, 0.7, 1.3)
        output *= velocity_factor

        # Smooth output
        alpha = 0.25
        smoothed_output = alpha * output + (1 - alpha) * self.prev_action

        # Constrain
        action = np.clip(smoothed_output, -2.0, 2.0)

        # === TRACK PERFORMANCE METRICS ===
        self.recent_errors.append(abs(error))
        if len(self.recent_actions) > 0:
            jerk = abs(action - self.recent_actions[-1])
            self.recent_jerks.append(jerk)

        self.recent_actions.append(action)

        # Keep only recent history
        if len(self.recent_errors) > self.adaptation_window:
            self.recent_errors.pop(0)
            self.recent_jerks.pop(0)
            self.recent_actions.pop(0)

        # === ADAPT GAINS PERIODICALLY ===
        self.steps += 1
        if self.steps % self.adapt_every == 0 and len(self.recent_errors) >= self.adaptation_window:
            self._adapt_gains()

        # Update state
        self.prev_error = error
        self.prev_action = action

        return float(action)

    def _adapt_gains(self):
        """
        Adapt control gains based on recent performance

        Uses simple gradient descent to minimize:
        cost = (avg_error^2 * 50) + avg_jerk

        This mirrors the challenge cost function
        """
        # Compute recent performance
        avg_error = np.mean(self.recent_errors)
        avg_jerk = np.mean(self.recent_jerks) if self.recent_jerks else 0.0

        # Cost function (simplified version of challenge cost)
        cost = (avg_error ** 2) * 50.0 + avg_jerk

        # Heuristic adaptation rules:
        # If error is high -> increase kp, increase kff
        # If jerk is high -> decrease gains, increase damping

        error_ratio = avg_error / 0.5  # 0.5 is target error
        jerk_ratio = avg_jerk / 0.1    # 0.1 is target jerk

        # Adaptation deltas
        if error_ratio > 1.2:  # High error
            # Increase proportional and feedforward
            self.kp += self.learning_rate * 50.0
            self.kff += self.learning_rate * 30.0
        elif error_ratio < 0.8:  # Low error
            # Can reduce gains slightly
            self.kp -= self.learning_rate * 20.0

        if jerk_ratio > 1.2:  # High jerk
            # Reduce all gains, increase damping
            self.kp -= self.learning_rate * 30.0
            self.ki -= self.learning_rate * 10.0
            self.kff -= self.learning_rate * 20.0
            self.kd += self.learning_rate * 15.0
        elif jerk_ratio < 0.8:  # Low jerk
            # Can be more aggressive
            self.kp += self.learning_rate * 10.0

        # Apply limits
        self.kp = np.clip(self.kp, *self.kp_range)
        self.ki = np.clip(self.ki, *self.ki_range)
        self.kd = np.clip(self.kd, *self.kd_range)
        self.kff = np.clip(self.kff, *self.kff_range)

    def _compute_feedforward(self, future_plan, current_roll: float) -> float:
        """
        Compute feedforward term from future trajectory
        """
        if len(future_plan.lataccel) == 0:
            return 0.0

        lookahead = min(self.lookahead_steps, len(future_plan.lataccel))
        future_targets = []

        for i in range(lookahead):
            if i < len(future_plan.roll_lataccel):
                compensated = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
            else:
                compensated = future_plan.lataccel[i] - current_roll
            future_targets.append(compensated)

        avg_future = np.mean(future_targets)
        return self.kff * avg_future
