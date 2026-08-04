from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.doc_opt import make_step_budget_callback


def _drive(callback, *, start_step: int, steps: int):
    """Simulate a training run: on_train_begin at start_step, then `steps` optimizer
    steps. Return the global_step at which training was asked to stop (or None)."""
    args = SimpleNamespace()
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=start_step)
    callback.on_train_begin(args, state, control)
    for _ in range(steps):
        state.global_step += 1
        callback.on_step_end(args, state, control)
        if control.should_training_stop:
            return state.global_step
    return None


class StepBudgetCallbackTest(unittest.TestCase):
    def test_stops_after_limit_from_fresh_start(self) -> None:
        callback = make_step_budget_callback(100)
        self.assertEqual(_drive(callback, start_step=0, steps=500), 100)

    def test_counts_from_resumed_step(self) -> None:
        # Round 2 resumes at global_step=100 and should advance exactly 100 more.
        callback = make_step_budget_callback(100)
        self.assertEqual(_drive(callback, start_step=100, steps=500), 200)

    def test_does_not_stop_before_limit(self) -> None:
        callback = make_step_budget_callback(100)
        self.assertIsNone(_drive(callback, start_step=0, steps=99))


if __name__ == "__main__":
    unittest.main()
