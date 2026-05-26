from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Two-Degree-of-Freedom (2DOF) Controller

    Strategy: Separate feedforward and feedback paths for optimal performance.

    2DOF Control Structure:
    - Feedforward Path: Computes ideal action from reference (future_plan)
      -> Fast response, tracks trajectory aggressively
    - Feedback Path: Corrects errors and disturbances (roll, model mismatch)
      -> Robust to uncertainties, provides stability

    This is different from simple FF+FB because the paths are independently tuned
    and combined optimally.
    """

    def __init__(self):
        # === FEEDFORWARD PATH (Reference Tracking) ===
        # Aggressive tracking of desired trajectory
        self.kff_direct = 0.55      # Direct feedforward from target
        self.kff_derivative = 0.18   # Feedforward from target rate of change
        self.kff_preview = 0.22      # Preview-based feedforward

        # === FEEDBACK PATH (Error Correction) ===
        # Conservative error correction for robustness
        self.kp = 0.25      # Proportional gain
        self.ki = 0.08      # Integral gain (anti-windup)
        self.kd = 0.10      # Derivative gain

        # === FILTER/SMOOTHER ===
        # Separate filters for FF and FB paths
        self.alpha_ff = 0.20   # Feedforward filter (low = smooth)
        self.alpha_fb = 0.30   # Feedback filter (higher = responsive)

        # State tracking
        self.prev_target = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.integral_max = 10.0

        # Separate previous outputs for each path
        self.prev_ff_output = 0.0
        self.prev_fb_output = 0.0
        self.prev_action = 0.0

        # Preview parameters
        self.lookahead_steps = 6

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute control using 2DOF structure
        """
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compensate target for road roll
        compensated_target = target_lataccel - roll_lataccel

        # ===== FEEDFORWARD PATH: Track Reference =====
        ff_output = self._compute_feedforward_path(
            compensated_target, future_plan, roll_lataccel, v_ego
        )

        # ===== FEEDBACK PATH: Correct Errors =====
        fb_output = self._compute_feedback_path(
            compensated_target, current_lataccel
        )

        # ===== COMBINE PATHS =====
        # 2DOF combination: FF for tracking, FB for correction
        combined_output = ff_output + fb_output

        # Global output constraint
        action = np.clip(combined_output, -2.0, 2.0)

        # Update state
        self.prev_target = compensated_target
        self.prev_action = action

        return float(action)

    def _compute_feedforward_path(self, target: float, future_plan, current_roll: float, v_ego: float) -> float:
        """
        Feedforward path: Predict ideal control from reference trajectory

        Components:
        1. Direct feedforward: immediate target
        2. Derivative feedforward: rate of change of target
        3. Preview feedforward: future trajectory
        """
        # 1. Direct FF: proportional to current target
        ff_direct = self.kff_direct * target

        # 2. Derivative FF: rate of change of target
        target_derivative = target - self.prev_target
        ff_derivative = self.kff_derivative * target_derivative

        # 3. Preview FF: look ahead in future_plan
        ff_preview = 0.0
        if len(future_plan.lataccel) > 0:
            lookahead = min(self.lookahead_steps, len(future_plan.lataccel))
            future_targets = []

            for i in range(lookahead):
                if i < len(future_plan.roll_lataccel):
                    compensated = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
                else:
                    compensated = future_plan.lataccel[i] - current_roll
                future_targets.append(compensated)

            avg_future = np.mean(future_targets)
            ff_preview = self.kff_preview * avg_future

        # Combine FF components
        ff_total = ff_direct + ff_derivative + ff_preview

        # Velocity adaptation for FF path
        velocity_factor = np.clip(v_ego / 15.0, 0.7, 1.3)
        ff_total *= velocity_factor

        # Filter FF output (smooth)
        ff_filtered = self.alpha_ff * ff_total + (1 - self.alpha_ff) * self.prev_ff_output
        self.prev_ff_output = ff_filtered

        return ff_filtered

    def _compute_feedback_path(self, target: float, current: float) -> float:
        """
        Feedback path: Correct tracking errors with PID

        Conservative tuning for stability
        """
        # Compute error
        error = target - current

        # P term
        p_term = self.kp * error

        # I term with anti-windup
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.error_integral

        # D term
        error_derivative = error - self.prev_error
        d_term = self.kd * error_derivative

        # Combine PID
        fb_total = p_term + i_term + d_term

        # Filter FB output (more responsive than FF)
        fb_filtered = self.alpha_fb * fb_total + (1 - self.alpha_fb) * self.prev_fb_output
        self.prev_fb_output = fb_filtered

        # Update state
        self.prev_error = error

        return fb_filtered
