"""Pure translation from campaign objectives to tactical execution requests.

This adapter deliberately stops at existing immutable ``Goal`` objects.  A
future controller may hand a READY result to the tactical execution layer;
translation itself performs no emulator or filesystem work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

from modules.goals import ActivateTrigger, EngageTrainer, Goal, NavigationGoal, ReachWarp, EARLY_POKEBALL_TRIGGER_ID
from modules.map_data import MapRSE

from .campaign_objectives import CampaignObjective, ObjectiveSelection, ObjectiveStatus
from .emerald_campaign_registry import emerald_capability_definition
from .capture_policy import EncounterMethod


@dataclass(frozen=True)
class _EmeraldCampaignDelegate:
    """Lazy delegate that mounts the ROM-specific Emerald capability."""

    objective_id: str

    def __call__(self, _objective: Goal) -> Iterator[object]:
        """Run the Emerald observation owner for this campaign milestone."""
        from .emerald_capabilities import emerald_campaign_capability

        yield from emerald_campaign_capability(self.objective_id)


class CampaignExecutionStatus(Enum):
    """Translation status returned before any tactical execution begins."""

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
    capability: Callable[[], Iterator[object]] | None = None


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

        if objective.objective_id == "HEAL_PARTY":
            from .resource_runtime import execute_heal_party

            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "mounted observation-driven party healing capability",
                objective.execution_id,
                capability=execute_heal_party,
            )

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
            # The introductory-rival script includes Emerald-owned dialogue,
            # yes/no confirmation, and the starter nickname screen.
            from .resource_runtime import CampaignCapability

            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "translated to the existing introductory rival tactical goal",
                objective.execution_id,
                goal,
                capability=CampaignCapability(
                    objective.objective_id,
                    objective.resource_policy,
                    goal,
                    delegate=_EmeraldCampaignDelegate(objective.objective_id),
                ),
            )

        if objective.objective_id.startswith("obtain_encounter:"):
            goal = objective.tactical_target
            if not isinstance(goal, NavigationGoal):
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "encounter objective has no valid navigation goal",
                    objective.execution_id,
                )
            method = objective.encounter_method or EncounterMethod.LAND
            if not isinstance(method, EncounterMethod):
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    f"encounter objective has malformed encounter method {method!r}",
                    objective.execution_id,
                    goal,
                )
            if method is not EncounterMethod.LAND:
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    f"encounter method {method.value!r} is not supported by the campaign executor",
                    objective.execution_id,
                    goal,
                )
            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                f"translated {method.value} encounter objective to SEEK navigation",
                objective.execution_id,
                goal,
                capability=lambda: __import__(
                    "modules.nuzlocke.resource_runtime", fromlist=["execute_campaign_encounter"]
                ).execute_campaign_encounter(goal.target.target_map, encounter_method=method),
            )

        if objective.objective_id == "prepare_roxanne":
            capability_definition = emerald_capability_definition(objective.objective_id)
            if (
                capability_definition is None
                or capability_definition.semantic_target is None
                or capability_definition.preparation_level is None
            ):
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "prepare_roxanne has no registered Emerald preparation capability",
                    objective.execution_id,
                )
            from .resource_runtime import execute_campaign_preparation

            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "mounted bounded Emerald preparation capability",
                objective.execution_id,
                capability=lambda: execute_campaign_preparation(
                    capability_definition.semantic_target.target_map,
                    target_level=capability_definition.preparation_level,
                ),
            )

        # Preparation decisions supply an explicit trainer target and policy;
        # keep the adapter generic so the live controller can mount the normal
        # navigation loop without ROM-specific battle code here.
        if isinstance(objective.tactical_target, NavigationGoal) and isinstance(
            objective.tactical_target.target, EngageTrainer
        ):
            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "translated trainer preparation objective to targeted navigation",
                objective.execution_id,
                objective.tactical_target,
            )

        capability_definition = emerald_capability_definition(objective.objective_id)
        if capability_definition is not None:
            # Keep the selector ROM-neutral and import emulator code only when
            # a live Emerald objective is actually mounted.
            from .emerald_capabilities import emerald_campaign_capability

            return CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                f"mounted {capability_definition.capability_id} for {objective.objective_id}",
                objective.execution_id,
                capability=lambda: emerald_campaign_capability(capability_definition.objective_id),
                tactical_goal=capability_definition.readiness_goal,
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
                "Poké Ball objective has no valid Professor Birch interaction goal",
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
