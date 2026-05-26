from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Pure Preview/Feedforward Controller

    Strategy: Aggressively use future_plan trajectory to compute control actions.
    Instead of reacting to errors (PID), we predict what's needed based on where we're going.

    Key differences from previous approaches:
    - Heavily weighted toward feedforward (90%+)
    - Multi-step lookahead with trajectory prediction
    - Minimal feedback loop (just for stability)
    """

    def __init__(self):
        # Feedforward gains (dominant)
        self.kff_immediate = 0.60   # Weight for immediate next step
        self.kff_near = 0.25        # Weight for near-term (2-5 steps)
        self.kff_far = 0.10         # Weight for far-term (6-10 steps)

        # Minimal feedback for stability
        self.kp_feedback = 0.05     # Small proportional for error correction
        self.kd_feedback = 0.02     # Small derivative for damping

        # Preview parameters
        self.preview_immediate = 1
        self.preview_near = 5
        self.preview_far = 10

        # State
        self.prev_error = 0.0
        self.prev_action = 0.0

        # Output smoothing (reduce jerk)
        self.alpha_smooth = 0.15

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute control using preview of future trajectory
        """
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compensate for road roll
        compensated_target = target_lataccel - roll_lataccel

        # === FEEDFORWARD: Predict what's needed based on future trajectory ===
        ff_action = self._compute_preview_feedforward(future_plan, roll_lataccel, v_ego)

        # === FEEDBACK: Small correction for current error ===
        error = compensated_target - current_lataccel

        # Proportional term (small)
        p_term = self.kp_feedback * error

        # Derivative term for damping (small)
        error_derivative = error - self.prev_error
        d_term = self.kd_feedback * error_derivative

        # Combine: feedforward dominant, feedback for correction
        output = ff_action + p_term + d_term

        # Smooth output to reduce jerk
        smoothed_output = self.alpha_smooth * output + (1 - self.alpha_smooth) * self.prev_action

        # Constrain
        action = np.clip(smoothed_output, -2.0, 2.0)

        # Update state
        self.prev_error = error
        self.prev_action = action

        return float(action)

    def _compute_preview_feedforward(self, future_plan, current_roll: float, v_ego: float) -> float:
        """
        Compute feedforward action using multi-horizon trajectory preview

        Uses weighted combination of:
        - Immediate next step (most important)
        - Near-term trajectory (2-5 steps ahead)
        - Far-term trajectory (6-10 steps ahead)
        """
        if len(future_plan.lataccel) == 0:
            return 0.0

        # Immediate next step target
        immediate_target = 0.0
        if len(future_plan.lataccel) >= self.preview_immediate:
            idx = self.preview_immediate - 1
            if idx < len(future_plan.roll_lataccel):
                immediate_target = future_plan.lataccel[idx] - future_plan.roll_lataccel[idx]
            else:
                immediate_target = future_plan.lataccel[idx] - current_roll

        # Near-term average (next 2-5 steps)
        near_targets = []
        max_near = min(self.preview_near, len(future_plan.lataccel))
        for i in range(2, max_near + 1):
            if i <= len(future_plan.lataccel):
                idx = i - 1
                if idx < len(future_plan.roll_lataccel):
                    comp = future_plan.lataccel[idx] - future_plan.roll_lataccel[idx]
                else:
                    comp = future_plan.lataccel[idx] - current_roll
                near_targets.append(comp)

        near_avg = np.mean(near_targets) if near_targets else immediate_target

        # Far-term average (next 6-10 steps)
        far_targets = []
        max_far = min(self.preview_far, len(future_plan.lataccel))
        for i in range(6, max_far + 1):
            if i <= len(future_plan.lataccel):
                idx = i - 1
                if idx < len(future_plan.roll_lataccel):
                    comp = future_plan.lataccel[idx] - future_plan.roll_lataccel[idx]
                else:
                    comp = future_plan.lataccel[idx] - current_roll
                far_targets.append(comp)

        far_avg = np.mean(far_targets) if far_targets else near_avg

        # Weighted combination with velocity adaptation
        velocity_factor = np.clip(v_ego / 15.0, 0.5, 1.5)

        ff_action = (
            self.kff_immediate * immediate_target +
            self.kff_near * near_avg +
            self.kff_far * far_avg
        ) * velocity_factor

        return ff_action
