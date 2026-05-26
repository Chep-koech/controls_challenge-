from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Tuned PID + roll-compensated preview feedforward with output LPF.

    The PID tracks the current target. The FF terms use the next few future
    targets (roll-compensated) to anticipate trajectory bends. The output
    LPF reduces high-frequency content, which lowers jerk_cost.
    """

    def __init__(self):
        self.p = 0.4116248929300425
        self.i = 0.08483982053314855
        self.d = 0.03244252481270672
        self.kff_rate = 0.17158077328557875
        self.kff_preview = 0.31248069240135273
        self.lookahead = 5
        self.alpha = 0.495212043515924

        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        ff_rate = 0.0
        ff_preview = 0.0
        n = min(self.lookahead, len(future_plan.lataccel))
        if n > 0:
            fut_lat = np.asarray(future_plan.lataccel[:n], dtype=np.float64)
            fut_roll = np.asarray(future_plan.roll_lataccel[:n], dtype=np.float64)
            future_avg = float(np.mean(fut_lat - fut_roll))
            compensated_now = target_lataccel - state.roll_lataccel
            ff_rate = self.kff_rate * (future_avg - compensated_now)
            ff_preview = self.kff_preview * future_avg

        raw = (
            self.p * error
            + self.i * self.error_integral
            + self.d * error_diff
            + ff_rate
            + ff_preview
        )

        filtered = self.alpha * raw + (1.0 - self.alpha) * self.prev_filtered
        self.prev_filtered = filtered
        return float(filtered)
