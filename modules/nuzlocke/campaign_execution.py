"""Pure translation from campaign objectives to tactical execution requests.

This adapter deliberately stops at existing immutable ``Goal`` objects.  A
future controller may hand a READY result to the tactical execution layer;
translation itself performs no emulator or filesystem work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modules.goals import ActivateTrigger, Goal, ReachWarp, EARLY_POKEBALL_TRIGGER_ID
from modules.map_data import MapRSE

from .campaign_objectives import CampaignObjective, ObjectiveSelection, ObjectiveStatus


class CampaignExecutionStatus(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class CampaignExecutionResult:
    """The safe-to-consume result of adapting one objective selection."""

    objective: CampaignObjective | None
    status: CampaignExecutionStatus
    reason: str
    execution_id: str | None = None
    tactical_goal: Goal | None = None


class CampaignExecutionAdapter:
    """Translate known campaign objectives without executing them."""

    @staticmethod
    def translate(selection: ObjectiveSelection) -> CampaignExecutionResult:
        """Return a deterministic tactical request or a structured refusal."""

        if not isinstance(selection, ObjectiveSelection):
            return CampaignExecutionResult(None, CampaignExecutionStatus.UNSUPPORTED, "malformed objective selection")

        status_map = {
            ObjectiveStatus.BLOCKED: CampaignExecutionStatus.BLOCKED,
            ObjectiveStatus.UNKNOWN: CampaignExecutionStatus.UNKNOWN,
            ObjectiveStatus.COMPLETE: CampaignExecutionStatus.COMPLETE,
            ObjectiveStatus.FAILED: CampaignExecutionStatus.FAILED,
        }
        if selection.status in status_map:
            objective = selection.objective
            return CampaignExecutionResult(
                objective,
                status_map[selection.status],
                selection.reason,
                objective.execution_id if objective is not None else None,
            )

        objective = selection.objective
        if selection.status is not ObjectiveStatus.READY:
            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.UNSUPPORTED,
                f"unsupported objective-selection status: {selection.status!r}",
                objective.execution_id if isinstance(objective, CampaignObjective) else None,
            )
        if not isinstance(objective, CampaignObjective):
            return CampaignExecutionResult(None, CampaignExecutionStatus.UNSUPPORTED, "malformed ready objective")

        if objective.objective_id == "reach_oldale":
            goal = objective.tactical_target
            if not isinstance(goal, ReachWarp) or goal.destination_map != MapRSE.OLDALE_TOWN.value:
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "reach_oldale has no valid Oldale navigation goal",
                    objective.execution_id,
                )
            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "translated to the existing Oldale navigation goal",
                objective.execution_id,
                goal,
            )

        if objective.objective_id == "complete_intro_rival":
            goal = objective.tactical_target
            if not isinstance(goal, ActivateTrigger) or goal.trigger_id != "introductory_rival":
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "complete_intro_rival has no valid introductory rival goal",
                    objective.execution_id,
                )
            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "translated to the existing introductory rival tactical goal",
                objective.execution_id,
                goal,
            )

        if objective.objective_id == "confirm_early_pokeballs":
            goal = objective.tactical_target
            if isinstance(goal, ActivateTrigger) and goal.trigger_id == EARLY_POKEBALL_TRIGGER_ID:
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.READY,
                    "translated to the existing Professor Birch interaction goal",
                    objective.execution_id,
                    goal,
                )
            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.UNSUPPORTED,
                "confirm_early_pokeballs has no valid Professor Birch interaction goal",
                objective.execution_id,
            )

        return CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.UNSUPPORTED,
            f"unknown campaign objective ID: {objective.objective_id!r}",
            objective.execution_id,
        )


def adapt_campaign_execution(selection: ObjectiveSelection) -> CampaignExecutionResult:
    """Functional convenience API for :meth:`CampaignExecutionAdapter.translate`."""

    return CampaignExecutionAdapter.translate(selection)
