from . import BaseController
import numpy as np
from typing import Optional


class Controller(BaseController):
    """
    Iterative Linear Quadratic Regulator (iLQR) Controller
    Optimizes steering actions over a prediction horizon to minimize:
    - Lateral acceleration tracking error
    - Jerk (rate of change of lateral acceleration)

    Designed to achieve score < 13 on comma.ai controls challenge
    """

    def __init__(self):
        # Prediction horizon (use available future plan)
        self.horizon = 50  # 5 seconds at 10 Hz

        # Cost weights (tuned for total_cost = lataccel_cost*50 + jerk_cost)
        self.Q_lataccel = 50.0   # Lateral acceleration tracking weight
        self.R_steer = 0.01      # Steering effort weight (low - we want aggressive tracking)
        self.R_jerk = 5.0        # Jerk penalty weight (higher to reduce jerk_cost)

        # iLQR parameters
        self.max_iterations = 3   # iLQR iterations per control step (faster)
        self.alpha = 0.5          # Line search step size
        self.convergence_tol = 1e-2

        # Vehicle dynamics approximation parameters (more realistic)
        self.lataccel_steer_gain = 1.8  # Approximate gain from steering to lateral accel
        self.lataccel_decay = 0.92      # Lateral accel persistence (higher = more stable)

        # State tracking
        self.prev_action = 0.0
        self.prev_lataccel = 0.0

        # Adaptive parameters based on velocity
        self.velocity_scale_min = 0.7
        self.velocity_scale_max = 1.3
        self.nominal_velocity = 15.0  # m/s (~30 mph)

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute optimal steering action using iLQR optimization

        Args:
            target_lataccel: Desired lateral acceleration
            current_lataccel: Current lateral acceleration
            state: Vehicle state (roll_lataccel, v_ego, a_ego)
            future_plan: Future trajectory plan (lataccel, roll_lataccel, v_ego, a_ego)

        Returns:
            Optimal steering action in range [-2, 2]
        """
        # Extract state information
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Adapt parameters based on velocity
        velocity_scale = self._compute_velocity_scale(v_ego)

        # Compensate target for road roll
        compensated_target = target_lataccel - roll_lataccel

        # Build target trajectory from future plan
        target_trajectory = self._build_target_trajectory(
            compensated_target, future_plan
        )

        # Initialize control sequence (warm start from previous action)
        control_sequence = np.ones(self.horizon) * self.prev_action

        # Run iLQR optimization
        optimal_controls = self._ilqr_optimize(
            current_lataccel,
            target_trajectory,
            control_sequence,
            velocity_scale
        )

        # Extract first control action (MPC-style receding horizon)
        action = optimal_controls[0]

        # Apply constraints
        action = np.clip(action, -2.0, 2.0)

        # Update state tracking
        self.prev_action = action
        self.prev_lataccel = current_lataccel

        return float(action)

    def _compute_velocity_scale(self, v_ego: float) -> float:
        """Adapt control gains based on vehicle velocity"""
        scale = v_ego / self.nominal_velocity
        return np.clip(scale, self.velocity_scale_min, self.velocity_scale_max)

    def _build_target_trajectory(self, current_target: float, future_plan) -> np.ndarray:
        """Build target lateral acceleration trajectory from future plan"""
        trajectory = np.zeros(self.horizon)
        trajectory[0] = current_target

        # Use future plan data (compensate for road roll)
        plan_length = min(len(future_plan.lataccel), self.horizon - 1)
        for i in range(plan_length):
            trajectory[i + 1] = future_plan.lataccel[i] - future_plan.roll_lataccel[i]

        # Extrapolate if future plan is shorter than horizon
        if plan_length < self.horizon - 1:
            trajectory[plan_length + 1:] = trajectory[plan_length]

        return trajectory

    def _ilqr_optimize(
        self,
        initial_lataccel: float,
        target_trajectory: np.ndarray,
        initial_controls: np.ndarray,
        velocity_scale: float
    ) -> np.ndarray:
        """
        Iterative LQR optimization

        Args:
            initial_lataccel: Starting lateral acceleration
            target_trajectory: Desired lateral acceleration over horizon
            initial_controls: Initial guess for control sequence
            velocity_scale: Velocity-based scaling factor

        Returns:
            Optimized control sequence
        """
        controls = initial_controls.copy()

        for iteration in range(self.max_iterations):
            # Forward pass: simulate dynamics with current controls
            states = self._forward_pass(initial_lataccel, controls, velocity_scale)

            # Backward pass: compute optimal control corrections
            control_corrections = self._backward_pass(
                states, controls, target_trajectory, velocity_scale
            )

            # Line search update
            new_controls = controls + self.alpha * control_corrections
            new_controls = np.clip(new_controls, -2.0, 2.0)

            # Check convergence
            if np.linalg.norm(control_corrections) < self.convergence_tol:
                break

            controls = new_controls

        return controls

    def _forward_pass(
        self,
        initial_lataccel: float,
        controls: np.ndarray,
        velocity_scale: float
    ) -> np.ndarray:
        """Simulate vehicle lateral dynamics forward in time"""
        states = np.zeros(self.horizon + 1)
        states[0] = initial_lataccel

        for t in range(self.horizon):
            # Simple lateral dynamics model:
            # lataccel[t+1] = decay * lataccel[t] + gain * steer[t]
            gain = self.lataccel_steer_gain * velocity_scale
            states[t + 1] = self.lataccel_decay * states[t] + gain * controls[t]

        return states

    def _backward_pass(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        target_trajectory: np.ndarray,
        velocity_scale: float
    ) -> np.ndarray:
        """Compute optimal control corrections via dynamic programming"""
        # Initialize value function derivatives
        V_x = 0.0  # Gradient of value function w.r.t. state
        V_xx = 0.0  # Hessian of value function w.r.t. state

        control_corrections = np.zeros(self.horizon)

        # Backward pass through time
        for t in range(self.horizon - 1, -1, -1):
            # State and control at time t
            x = states[t]
            u = controls[t]

            # Tracking error
            error = x - target_trajectory[t]

            # Jerk calculation (change in control)
            if t > 0:
                jerk = u - controls[t - 1]
            else:
                jerk = u - self.prev_action

            # Cost derivatives
            l_x = 2 * self.Q_lataccel * error  # d(cost)/d(state)
            l_u = 2 * self.R_steer * u + 2 * self.R_jerk * jerk  # d(cost)/d(control)
            l_xx = 2 * self.Q_lataccel  # d²(cost)/d(state)²
            l_uu = 2 * self.R_steer + 2 * self.R_jerk  # d²(cost)/d(control)²

            # Dynamics derivatives
            gain = self.lataccel_steer_gain * velocity_scale
            f_x = self.lataccel_decay  # d(next_state)/d(state)
            f_u = gain  # d(next_state)/d(control)

            # Q-function derivatives (state-action value)
            Q_x = l_x + f_x * V_x
            Q_u = l_u + f_u * V_x
            Q_xx = l_xx + f_x * V_xx * f_x
            Q_uu = l_uu + f_u * V_xx * f_u
            Q_ux = f_u * V_xx * f_x

            # Optimal control correction
            # Prevent division by zero
            if abs(Q_uu) > 1e-6:
                control_corrections[t] = -Q_u / Q_uu
            else:
                control_corrections[t] = 0.0

            # Update value function for previous time step
            V_x = Q_x - Q_ux * control_corrections[t]
            V_xx = Q_xx - Q_ux * Q_ux / (Q_uu + 1e-6)

        return control_corrections
