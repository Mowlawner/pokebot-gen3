"""Small orchestration layer for the ordered campaign."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generator, Iterator

from modules.agent_control import AgentControlLoop, observe_agent
from modules.goals import Goal

from .campaign_execution import CampaignExecutionResult, CampaignExecutionStatus, adapt_campaign_execution
from .campaign_objectives import ObjectiveSelection, select_campaign_objective
from .campaign_state import CampaignState


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


TacticalLoopFactory = Callable[[Goal], Iterator[object]]
CampaignBoundaryHandler = Callable[[str], bool]


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
        selection = self._selector(self._state_provider())
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
