"""
Persistent pipeline state.

Example after several stages:

{
    "preflight": {
        "status": "completed"
    },
    "data_preparation": {
        "status": "completed"
    },
    "training": {
        "status": "running"
    }
}

A Colab disconnection therefore does not make us wonder what actually
finished.
"""

from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

from bodhan_bhili.core.constants import (
    ALL_PIPELINE_STAGES,
)
from bodhan_bhili.core.serialization import (
    atomic_write_json,
    read_json,
)


def _utc_now() -> str:
    """Current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


class RunStateTracker:
    """Read and update persistent pipeline stage status."""

    def __init__(
        self,
        state_file: str | Path,
        run_id: str,
    ):
        self.state_file = Path(
            state_file
        )

        self.run_id = run_id

        if not self.state_file.exists():
            self._initialize()


    def _initialize(
        self,
    ) -> None:
        """Create state document for a fresh run."""

        payload = {
            "run_id":
                self.run_id,

            "created_at_utc":
                _utc_now(),

            "updated_at_utc":
                _utc_now(),

            "stages": {
                stage: {
                    "status":
                        "pending",

                    "started_at_utc":
                        None,

                    "completed_at_utc":
                        None,

                    "message":
                        None,
                }
                for stage in ALL_PIPELINE_STAGES
            },
        }

        atomic_write_json(
            payload,
            self.state_file,
        )


    def read(
        self,
    ) -> dict[str, Any]:
        """Return current pipeline state."""

        return read_json(
            self.state_file
        )


    def _validate_stage(
        self,
        stage: str,
    ) -> None:
        """Reject accidental/unknown stage names."""

        if stage not in ALL_PIPELINE_STAGES:
            raise ValueError(
                f"Unknown pipeline stage: {stage!r}. "
                f"Valid stages: {ALL_PIPELINE_STAGES}"
            )


    def mark_started(
        self,
        stage: str,
        message: str | None = None,
    ) -> None:
        """Mark one stage as currently executing."""

        self._validate_stage(
            stage
        )

        state = self.read()

        entry = state[
            "stages"
        ][
            stage
        ]

        entry[
            "status"
        ] = "running"

        entry[
            "started_at_utc"
        ] = _utc_now()

        entry[
            "completed_at_utc"
        ] = None

        entry[
            "message"
        ] = message

        state[
            "updated_at_utc"
        ] = _utc_now()

        atomic_write_json(
            state,
            self.state_file,
        )


    def mark_completed(
        self,
        stage: str,
        message: str | None = None,
    ) -> None:
        """Mark one stage as successfully completed."""

        self._validate_stage(
            stage
        )

        state = self.read()

        entry = state[
            "stages"
        ][
            stage
        ]

        entry[
            "status"
        ] = "completed"

        entry[
            "completed_at_utc"
        ] = _utc_now()

        entry[
            "message"
        ] = message

        state[
            "updated_at_utc"
        ] = _utc_now()

        atomic_write_json(
            state,
            self.state_file,
        )


    def mark_failed(
        self,
        stage: str,
        message: str,
    ) -> None:
        """Record a failed stage without destroying earlier state."""

        self._validate_stage(
            stage
        )

        state = self.read()

        entry = state[
            "stages"
        ][
            stage
        ]

        entry[
            "status"
        ] = "failed"

        entry[
            "completed_at_utc"
        ] = _utc_now()

        entry[
            "message"
        ] = message

        state[
            "updated_at_utc"
        ] = _utc_now()

        atomic_write_json(
            state,
            self.state_file,
        )