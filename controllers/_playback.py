"""Internal helper controller for CEM evaluation.

Plays back a fixed action sequence given at construction. Used by the
optimizer to evaluate candidate action sequences against the simulator.
"""
from . import BaseController
import numpy as np

# CONTROL_START_IDX - CONTEXT_LENGTH = 100 - 20 = 80
# The simulator overrides controller actions for step_idx < 100, and the
# first update() call comes at step_idx = CONTEXT_LENGTH = 20. So
# action_seq[0] corresponds to step_idx = 100 (first step that matters).
WARMUP_CALLS = 80


class Controller(BaseController):
    def __init__(self, action_seq=None):
        if action_seq is None:
            # If instantiated by importlib without args, fall back to zeros.
            action_seq = np.zeros(400)
        self.action_seq = np.asarray(action_seq, dtype=np.float64)
        self.call_idx = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        i = self.call_idx
        self.call_idx += 1
        if i < WARMUP_CALLS:
            return 0.0
        j = i - WARMUP_CALLS
        if j < len(self.action_seq):
            return float(self.action_seq[j])
        return 0.0
