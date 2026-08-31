"""Read-only campaign planner simulation and checkpoint reporting.

This module deliberately accepts materialized :class:`CampaignState` values.
It has no emulator, persistence, input, or projection mutation path and uses
the production task selector for every report.
"""

from dataclasses import dataclass
from typing import Iterable

from .campaign_objectives import (
    ObjectiveSelection,
    available_campaign_tasks,
    campaign_task_diagnostics,
    select_available_campaign_task,
)
from .campaign_state import CampaignState


@dataclass(frozen=True, slots=True)
class CampaignCheckpoint:
    """Named materialized campaign state used for planner diagnostics."""

    name: str
    state: CampaignState | None
    unavailable_reason: str | None = None

    @classmethod
    def unavailable(cls, name: str, reason: str) -> "CampaignCheckpoint":
        """Create a checkpoint whose state could not be materialized."""

        return cls(name, None, reason)


@dataclass(frozen=True, slots=True)
class CampaignSimulation:
    """Read-only selector and task report for one campaign checkpoint."""

    checkpoint: str
    status: str
    reason: str | None
    primary_task: str | None
    progression_anchor: tuple[int, int] | None
    selection_reason: str | None
    current_location: tuple[int, int] | None
    campaign_facts: dict[str, object]
    tasks: tuple[dict[str, object], ...]


def simulate_checkpoint(checkpoint: CampaignCheckpoint) -> CampaignSimulation:
    """Run the real read-only task model for one checkpoint."""
    if checkpoint.state is None:
        return CampaignSimulation(
            checkpoint.name,
            "unavailable",
            checkpoint.unavailable_reason,
            None,
            None,
            None,
            None,
            {},
            (),
        )
    state = checkpoint.state
    tasks = campaign_task_diagnostics(state)
    selection: ObjectiveSelection = select_available_campaign_task(state)
    anchor = next(
        (row["progression_destination"] for row in tasks if row["progression_destination"] is not None),
        None,
    )
    return CampaignSimulation(
        checkpoint.name,
        "observed",
        None,
        selection.objective.objective_id if selection.objective else None,
        anchor,
        selection.reason,
        state.raw_map.value if state.raw_map.is_known else None,
        {
            name: {
                "status": getattr(state.campaign_facts, name).status.value,
                "value": getattr(state.campaign_facts, name).value,
            }
            for name in state.campaign_facts.__dataclass_fields__
        },
        tasks,
    )


def simulate_checkpoints(checkpoints: Iterable[CampaignCheckpoint]) -> tuple[CampaignSimulation, ...]:
    """Simulate checkpoints in caller-provided stable order."""
    return tuple(simulate_checkpoint(checkpoint) for checkpoint in checkpoints)
