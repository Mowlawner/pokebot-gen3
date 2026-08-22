"""Small orchestration layer for the ordered campaign."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generator, Iterator, Any
import json

from modules.agent_control import AgentControlLoop, observe_agent
from modules.goals import Goal

from .campaign_execution import CampaignExecutionResult, CampaignExecutionStatus, adapt_campaign_execution
from .campaign_objectives import ObjectiveSelection, campaign_task_diagnostics, select_campaign_objective
from modules.console import diagnostic_print
from .campaign_state import CampaignState
from .readiness_diagnostics import (
    CampaignReadinessPolicy,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    evaluate_progression_readiness,
)


class CampaignControllerStatus(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CampaignControllerState:
    status: CampaignControllerStatus
    objective_id: str | None
    tactical_goal: Goal | None
    selection: ObjectiveSelection | None
    execution: CampaignExecutionResult | None
    reason: str
    execution_phase: str = "CAMPAIGN"
    recovery_status: str | None = None


TacticalLoopFactory = Callable[[Goal], Iterator[object]]
CampaignBoundaryHandler = Callable[[str], bool]
ReadinessProvider = Callable[[Any, Goal | None], ProgressionReadinessDiagnostic]
RecoveryFactory = Callable[[ProgressionReadinessDiagnostic], Iterator[object]]


class CampaignController:
    """Select and mount one tactical goal, reevaluating at every frame boundary."""

    def __init__(
        self,
        state_provider: Callable[[], CampaignState],
        *,
        tactical_loop_factory: TacticalLoopFactory | None = None,
        selector: Callable[[CampaignState], ObjectiveSelection] = select_campaign_objective,
        adapter: Callable[[ObjectiveSelection], CampaignExecutionResult] = adapt_campaign_execution,
        campaign_boundary_handler: CampaignBoundaryHandler | None = None,
        readiness_provider: ReadinessProvider | None = None,
        recovery_factory: RecoveryFactory | None = None,
        readiness_policy: CampaignReadinessPolicy | None = None,
    ) -> None:
        self._state_provider = state_provider
        self._selector = selector
        self._adapter = adapter
        self._campaign_boundary_handler = campaign_boundary_handler
        self._tactical_loop_factory = tactical_loop_factory or self._default_tactical_loop
        self.current_objective_id: str | None = None
        self.current_tactical_goal: Goal | None = None
        self.last_selection: ObjectiveSelection | None = None
        self.last_execution: CampaignExecutionResult | None = None
        self.status = CampaignControllerStatus.UNKNOWN
        self.transition_reason = "not started"
        self._tactical_loop: Iterator[object] | None = None
        self._readiness_provider = readiness_provider
        self._recovery_factory = recovery_factory
        self._readiness_policy = readiness_policy or CampaignReadinessPolicy()
        self._execution_phase = "CAMPAIGN"
        self._recovery_status: str | None = None

    @staticmethod
    def _default_tactical_loop(goal: Goal) -> Iterator[object]:
        return AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()

    @property
    def state(self) -> CampaignControllerState:
        return CampaignControllerState(
            self.status,
            self.current_objective_id,
            self.current_tactical_goal,
            self.last_selection,
            self.last_execution,
            self.transition_reason,
            self._execution_phase,
            self._recovery_status,
        )

    def _terminal(self, execution: CampaignExecutionResult, status: CampaignControllerStatus) -> None:
        self.current_objective_id = None
        self.current_tactical_goal = None
        self._tactical_loop = None
        self.last_execution = execution
        self.status = status
        self.transition_reason = execution.reason

    def refresh(self) -> CampaignControllerState:
        """Rebuild state, select, adapt, and mount work if needed."""
        observed_state = self._state_provider()
        selection = self._selector(observed_state)
        diagnostic_print(
            lambda: "CAMPAIGN_TASK_DISCOVERY: "
            + json.dumps(
                {
                    "tasks": campaign_task_diagnostics(observed_state),
                    "selected": selection.objective.objective_id if selection.objective else None,
                    "selection_status": selection.status.value,
                    "selection_reason": selection.reason,
                },
                sort_keys=True,
                default=str,
            ),
            trace=True,
        )
        if (
            selection.status.value == "ready"
            and selection.objective is not None
            and selection.objective.objective_id == "start_nuzlocke"
        ):
            if self._campaign_boundary_handler is None or not self._campaign_boundary_handler("start_nuzlocke"):
                self.last_selection = selection
                self._terminal(
                    CampaignExecutionResult(
                        selection.objective,
                        CampaignExecutionStatus.FAILED,
                        "no Nuzlocke campaign-boundary handler is available",
                        selection.objective.execution_id,
                    ),
                    CampaignControllerStatus.FAILED,
                )
                return self.state
            selection = self._selector(self._state_provider())
        self.last_selection = selection
        execution = self._adapter(selection)
        self.last_execution = execution
        terminal_status = {
            CampaignExecutionStatus.BLOCKED: CampaignControllerStatus.BLOCKED,
            CampaignExecutionStatus.UNKNOWN: CampaignControllerStatus.UNKNOWN,
            CampaignExecutionStatus.COMPLETE: CampaignControllerStatus.COMPLETE,
            CampaignExecutionStatus.FAILED: CampaignControllerStatus.FAILED,
            CampaignExecutionStatus.UNSUPPORTED: CampaignControllerStatus.FAILED,
        }
        if execution.status in terminal_status:
            self._terminal(execution, terminal_status[execution.status])
            return self.state
        if execution.status is not CampaignExecutionStatus.READY or execution.objective is None:
            self._terminal(
                CampaignExecutionResult(
                    execution.objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "malformed ready execution result",
                    execution.execution_id,
                ),
                CampaignControllerStatus.FAILED,
            )
            return self.state
        if execution.tactical_goal is None and execution.capability is None:
            self._terminal(
                CampaignExecutionResult(
                    execution.objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "ready execution result has no tactical goal",
                    execution.execution_id,
                ),
                CampaignControllerStatus.FAILED,
            )
            return self.state
        objective_id = execution.objective.objective_id
        # Readiness is tactical/resource gating. Startup Emerald capabilities
        # intentionally have no tactical Goal: they must be allowed to observe
        # title/menu/naming/clock state before overworld readiness exists.
        if (
            self._readiness_provider is not None
            and self._execution_phase == "CAMPAIGN"
            and execution.tactical_goal is not None
        ):
            evaluated = evaluate_progression_readiness(
                self._readiness_provider(execution.objective, execution.tactical_goal),
                self._readiness_policy,
            )
            schedule = getattr(getattr(self._readiness_provider, "__self__", None), "state", None)
            diagnostic_print(
                lambda: "CAMPAIGN_READINESS_LIFECYCLE: "
                + json.dumps(
                    {
                        "controller_tick": schedule.tick_count if schedule is not None else None,
                        "objective_id": objective_id,
                        "current_map": evaluated.current_map,
                        "coordinates": evaluated.current_coordinates,
                        "lowest_hp_ratio": evaluated.lowest_hp_ratio,
                        "usable_pokemon_count": evaluated.usable_count,
                        "trainer_hazard_state": evaluated.trainers,
                        "recovery_availability": evaluated.recovery_availability.value,
                        "readiness_decision": evaluated.readiness_decision.value,
                        "readiness_reason": evaluated.readiness_reason.value,
                        "execution_phase": self._execution_phase,
                        "recovery_status": self._recovery_status,
                        "schedule": (
                            {
                                "status": schedule.status,
                                "age_ticks": schedule.age_ticks,
                                "refresh_count": schedule.refresh_count,
                                "tick_count": schedule.tick_count,
                                "invalidation_reason": schedule.invalidation_reason,
                            }
                            if schedule is not None
                            else None
                        ),
                    },
                    sort_keys=True,
                ),
                trace=True,
            )
            from modules.route102_diagnostics import emit as route102_emit

            route102_emit(
                "readiness",
                controller_tick=schedule.tick_count if schedule is not None else None,
                execution_phase=self._execution_phase,
                objective_id=objective_id,
                objective_status=getattr(execution.objective, "status", None),
                tactical_goal=repr(execution.tactical_goal),
                readiness_decision=evaluated.readiness_decision.value,
                readiness_reason=evaluated.readiness_reason.value,
                scheduler=(
                    schedule.__dict__ if schedule is not None and hasattr(schedule, "__dict__") else repr(schedule)
                ),
                current_map=evaluated.current_map,
                coordinates=evaluated.current_coordinates,
                lowest_hp_ratio=evaluated.lowest_hp_ratio,
                usable_pokemon_count=evaluated.usable_count,
                trainer_availability=evaluated.trainer_availability.value,
                trainers=evaluated.trainers,
                recovery_availability=evaluated.recovery_availability.value,
                recovery_status=self._recovery_status,
            )
            if evaluated.readiness_decision is ReadinessDecision.UNKNOWN:
                self._tactical_loop = None
                self.status = CampaignControllerStatus.BLOCKED
                self._execution_phase = "BLOCKED"
                self.transition_reason = f"readiness deferred: {evaluated.readiness_reason.value}"
                return self.state
            if evaluated.readiness_decision is ReadinessDecision.RECOVER:
                if self._recovery_factory is None:
                    self._tactical_loop = None
                    self.status = CampaignControllerStatus.BLOCKED
                    self._execution_phase = "BLOCKED"
                    self.transition_reason = "readiness requested recovery but no recovery capability is available"
                    return self.state
                self.current_objective_id = objective_id
                self.current_tactical_goal = execution.tactical_goal
                # This is intentionally before invoking the recovery factory:
                # recovery pathfinding is synchronous and may outlive the
                # frame/trace lifecycle (including on KeyboardInterrupt).
                diagnostic_print(
                    lambda: "recovery_start: "
                    + json.dumps(
                        {
                            "objective_id": objective_id,
                            "execution_phase": "RECOVERY",
                            "readiness_decision": evaluated.readiness_decision.value,
                            "readiness_reason": evaluated.readiness_reason.value,
                        },
                        sort_keys=True,
                        default=str,
                    ),
                    trace=True,
                )
                try:
                    self._tactical_loop = self._recovery_factory(evaluated)
                except Exception as error:
                    self._tactical_loop = None
                    self.status = CampaignControllerStatus.BLOCKED
                    self._execution_phase = "BLOCKED"
                    self._recovery_status = "UNAVAILABLE"
                    self.transition_reason = f"recovery unavailable: {error}"
                    return self.state
                self._execution_phase = "RECOVERY"
                self._recovery_status = "ACTIVE"
                self.status = CampaignControllerStatus.READY
                self.transition_reason = f"recover before {objective_id}"
                return self.state
        if (
            self.current_objective_id != objective_id
            or self.current_tactical_goal != execution.tactical_goal
            or self._tactical_loop is None
        ):
            self.current_objective_id = objective_id
            self.current_tactical_goal = execution.tactical_goal
            self._tactical_loop = (
                execution.capability()
                if execution.capability is not None
                else self._tactical_loop_factory(execution.tactical_goal)
            )
            self.transition_reason = f"selected {objective_id}"
        self.status = CampaignControllerStatus.READY
        return self.state

    def step(self) -> CampaignControllerState:
        if self._execution_phase == "BLOCKED":
            return self.state
        if self._execution_phase == "RECOVERY":
            try:
                if self._tactical_loop is None:
                    raise RuntimeError("recovery has no executable capability")
                next(self._tactical_loop)
            except StopIteration:
                self._tactical_loop = None
                self._execution_phase = "CAMPAIGN"
                self._recovery_status = "COMPLETED"
                scheduler = getattr(self._readiness_provider, "__self__", None)
                invalidate = getattr(scheduler, "invalidate", None)
                if callable(invalidate):
                    invalidate("recovery_completed")
                self.status = CampaignControllerStatus.READY
                self.transition_reason = "recovery verified; reevaluating campaign readiness"
            except Exception as error:
                self._tactical_loop = None
                self._execution_phase = "BLOCKED"
                self._recovery_status = "FAILED"
                self.status = CampaignControllerStatus.BLOCKED
                self.transition_reason = f"recovery failed: {error}"
            return self.state
        current = self.refresh()
        if current.status is not CampaignControllerStatus.READY or self._tactical_loop is None:
            return current
        try:
            next(self._tactical_loop)
        except StopIteration:
            if self.last_execution is not None and self.last_execution.capability is not None:
                # A capability may yield control at a ROM boundary while its
                # authoritative completion fact is still false. Re-observe
                # and remount it on the next frame instead of declaring a
                # tactical failure.
                self._tactical_loop = None
                self.transition_reason = "capability boundary; awaiting authoritative completion"
                return self.state
            self._terminal(
                CampaignExecutionResult(
                    self.last_selection.objective if self.last_selection else None,
                    CampaignExecutionStatus.FAILED,
                    "tactical execution stopped before objective completion",
                    self.last_execution.execution_id if self.last_execution else None,
                ),
                CampaignControllerStatus.FAILED,
            )
        except Exception as error:
            # A failed generator cannot be resumed safely. Clear it and leave
            # the controller recoverable so a later Campaign-mode entry can
            # select the current objective and rebuild fresh execution state.
            self._tactical_loop = None
            self.status = CampaignControllerStatus.UNKNOWN
            self.transition_reason = f"tactical execution failed; will re-evaluate: {error}"
        return self.state

    def run(self) -> Generator:
        while True:
            state = self.step()
            if state.status is not CampaignControllerStatus.READY:
                return
            yield


def runtime_campaign_state() -> CampaignState:
    """Build a facade from the already-materialized runtime state."""
    from modules.context import context
    from modules.map_data import get_map_enum
    from .snapshots import get_nuzlocke_snapshot

    snapshot = get_nuzlocke_snapshot()
    runtime = context.nuzlocke_runtime
    canonical_area = None
    if snapshot.player_available and snapshot.player.map_group is not None and snapshot.player.map_number is not None:
        try:
            canonical_area = get_map_enum((snapshot.player.map_group, snapshot.player.map_number)).name
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
    return CampaignState.from_runtime_state(
        snapshot=snapshot,
        observed_projection=runtime.observed_projection if runtime is not None else None,
        rules_projection=runtime.rules_projection if runtime is not None else None,
        canonical_area=canonical_area,
    )


def runtime_campaign_boundary(objective_id: str) -> bool:
    """Execute a Nuzbot-owned campaign boundary through the live runtime."""
    if objective_id != "start_nuzlocke":
        return False
    from modules.context import context

    runtime = context.nuzlocke_runtime
    if runtime is None:
        return False
    runtime.mark_nuzlocke_started()
    return True
