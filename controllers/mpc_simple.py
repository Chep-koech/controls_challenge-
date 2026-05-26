from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Simple Model Predictive Control (MPC) Controller

    Uses direct optimization to minimize the cost function over a short horizon.
    This approach doesn't require an accurate dynamics model - it directly
    optimizes the actions that minimize tracking error and jerk.

    Target: Score < 13 on comma.ai controls challenge
    """

    def __init__(self):
        # MPC horizon (shorter for faster computation)
        self.horizon = 10  # 1 second ahead at 10 Hz

        # Cost weights matching the challenge scoring
        self.lataccel_weight = 50.0  # Matches total_cost = lataccel_cost * 50 + jerk_cost
        self.jerk_weight = 1.0

        # Simple dynamics parameters (learned from data)
        self.response_gain = 0.65  # How much steering affects lataccel
        self.response_delay = 0.15  # Smoothing factor

        # State tracking
        self.prev_action = 0.0
        self.action_history = []

        # Adaptive parameters
        self.nominal_velocity = 15.0  # m/s

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan) -> float:
        """
        Compute optimal steering action using MPC optimization

        Args:
            target_lataccel: Desired lateral acceleration
            current_lataccel: Current lateral acceleration
            state: Vehicle state (roll_lataccel, v_ego, a_ego)
            future_plan: Future trajectory plan

        Returns:
            Optimal steering action in range [-2, 2]
        """
        # Extract state information
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compensate target for road roll
        compensated_target = target_lataccel - roll_lataccel

        # Build future target trajectory
        future_targets = self._build_future_targets(compensated_target, future_plan)

        # Velocity-based scaling
        velocity_scale = np.clip(v_ego / self.nominal_velocity, 0.7, 1.3)

        # Optimize control sequence
        optimal_action = self._optimize_action(
            current_lataccel,
            future_targets,
            velocity_scale
        )

        # Apply constraints
        optimal_action = np.clip(optimal_action, -2.0, 2.0)

        # Update history
        self.prev_action = optimal_action
        self.action_history.append(optimal_action)
        if len(self.action_history) > 100:
            self.action_history.pop(0)

        return float(optimal_action)

    def _build_future_targets(self, current_target: float, future_plan) -> np.ndarray:
        """Build target lateral acceleration trajectory"""
        targets = np.zeros(self.horizon)
        targets[0] = current_target

        # Use future plan (compensated for roll)
        plan_length = min(len(future_plan.lataccel), self.horizon - 1)
        for i in range(plan_length):
            targets[i + 1] = future_plan.lataccel[i] - future_plan.roll_lataccel[i]

        # Extrapolate if needed
        if plan_length < self.horizon - 1:
            targets[plan_length + 1:] = targets[plan_length]

        return targets

    def _optimize_action(
        self,
        current_lataccel: float,
        future_targets: np.ndarray,
        velocity_scale: float
    ) -> float:
        """
        Optimize single steering action to minimize cost

        Uses a simple optimization approach: try multiple actions and pick the best
        This is faster and more robust than complex optimization
        """
        # Sample candidate actions
        num_samples = 50
        actions = np.linspace(-2.0, 2.0, num_samples)

        # Evaluate cost for each action
        costs = np.array([
            self._evaluate_action_cost(
                action, current_lataccel, future_targets, velocity_scale
            )
            for action in actions
        ])

        # Return action with minimum cost
        best_idx = np.argmin(costs)
        return actions[best_idx]

    def _evaluate_action_cost(
        self,
        action: float,
        current_lataccel: float,
        future_targets: np.ndarray,
        velocity_scale: float
    ) -> float:
        """
        Evaluate the cost of a given action over the horizon

        Uses a simple predictive model to estimate future lateral acceleration
        """
        lataccel = current_lataccel
        total_lataccel_error = 0.0
        total_jerk = 0.0

        prev_lataccel = current_lataccel
        prev_action = self.prev_action

        for t in range(self.horizon):
            # Use constant action (receding horizon - we only apply first one)
            current_action = action

            # Simple predictive model: lataccel moves toward action * gain
            target_lataccel = current_action * self.response_gain * velocity_scale
            lataccel = self.response_delay * lataccel + (1 - self.response_delay) * target_lataccel

            # Compute tracking error
            error = lataccel - future_targets[t]
            total_lataccel_error += error ** 2

            # Compute jerk (change in lataccel)
            jerk = (lataccel - prev_lataccel) / 0.1  # dt = 0.1 sec
            total_jerk += jerk ** 2

            prev_lataccel = lataccel
            prev_action = current_action

        # Cost matching the challenge formula
        lataccel_cost = (total_lataccel_error / self.horizon) * 100
        jerk_cost = (total_jerk / (self.horizon - 1 if self.horizon > 1 else 1)) * 100
        total_cost = self.lataccel_weight * lataccel_cost + self.jerk_weight * jerk_cost

        return total_cost
