"""Tests for crash-resilient experiment-state tracking."""

from bodhan_bhili.core.constants import STAGE_PREFLIGHT
from bodhan_bhili.core.run_state import RunStateTracker


def test_stage_lifecycle(tmp_path):
    """A stage should move pending -> running -> completed."""
    state_file = tmp_path / "run_state.json"
    tracker = RunStateTracker(state_file=state_file, run_id="test_run")
    initial = tracker.read()

    assert initial["stages"][STAGE_PREFLIGHT]["status"] == "pending"

    tracker.mark_started(STAGE_PREFLIGHT)

    running = tracker.read()

    assert running["stages"][STAGE_PREFLIGHT]["status"] == "running"

    tracker.mark_completed(STAGE_PREFLIGHT)

    completed = tracker.read()

    assert completed["stages"][STAGE_PREFLIGHT]["status"] == "completed"
