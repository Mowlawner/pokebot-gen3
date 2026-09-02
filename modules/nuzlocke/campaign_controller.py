"""Small orchestration layer for the ordered campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Generator, Iterator, Any
import json
import traceback
from functools import lru_cache

from modules.agent_control import AgentControlLoop, observe_agent
from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    Goal,
    NavigationGoal,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
)
from modules.context import context

from .campaign_execution import CampaignExecutionResult, CampaignExecutionStatus, adapt_campaign_execution
from .campaign_objectives import (
    ObjectiveSelection,
    ObjectiveStatus,
    campaign_task_diagnostics,
    campaign_planning_signature,
    select_campaign_objective,
    select_available_campaign_task,
    heal_party_objective,
)
from modules.console import diagnostic_print
from .campaign_status import CampaignStatus, recovery_status
from .campaign_state import CampaignState, FactStatus
from .readiness_diagnostics import (
    CampaignReadinessPolicy,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    ReadinessReason,
)
from .campaign_planner import CampaignPlan, build_campaign_plan, RecoveryStop


class CampaignControllerStatus(Enum):
    """Lifecycle status exposed by the campaign controller."""

    READY = "ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CampaignControllerState:
    """Immutable controller state snapshot for GUI and integration consumers."""

    status: CampaignControllerStatus
    objective_id: str | None
    tactical_goal: Goal | None
    selection: ObjectiveSelection | None
    execution: CampaignExecutionResult | None
    reason: str
    execution_phase: str = "CAMPAIGN"
    recovery_status: str | None = None
    plan: CampaignPlan | None = None


TacticalLoopFactory = Callable[[Goal], Iterator[object]]
CampaignBoundaryHandler = Callable[[str], bool]
ReadinessProvider = Callable[[Any, Goal | None], ProgressionReadinessDiagnostic]
RecoveryFactory = Callable[[RecoveryStop], Iterator[object]]


@lru_cache(maxsize=1)
def _canonical_campaign_objective_ranks() -> dict[str, int]:
    """Return the declarative order used to arbitrate transient ownership.

    The planner is allowed to report an earlier objective briefly while a
    battle or field script is being torn down.  Ownership may move forward
    across that boundary, but an earlier objective must never take ownership
    away from a later one.  Keep this lookup local to the controller so
    custom objective lists used by embedders/tests retain the historical
    fallback behavior below.
    """
    from .campaign_objectives import initial_emerald_campaign

    return {objective.objective_id: index for index, objective in enumerate(initial_emerald_campaign())}


def _retention_order_allows(active_id: str, selected_id: str) -> bool:
    """Return whether a newly selected objective may replace the active one."""

    ranks = _canonical_campaign_objective_ranks()
    active_rank = ranks.get(active_id)
    selected_rank = ranks.get(selected_id)
    if active_rank is None or selected_rank is None:
        # Preserve support for custom objective sequences, which have no
        # canonical campaign order to compare.
        return True
    return active_rank > selected_rank


def _spatial_observation_is_coherent(state: CampaignState) -> bool:
    """Return whether the current save exposes a usable field position.

    Battle start/return and scripted map transitions temporarily remove the
    player projection even though save-backed campaign facts are still being
    read.  Those frames are not evidence that a later active objective has
    regressed; they are an observation boundary at which its generator must
    remain mounted.  A coherent map and coordinate pair, on the other hand,
    is required before allowing this exception to retain ownership across a
    real save regression.
    """
    return state.raw_map.is_known and state.coordinates.is_known


_TRANSIENT_READINESS_REASONS = frozenset(
    {
        ReadinessReason.OVERWORLD_UNAVAILABLE,
        ReadinessReason.RESOURCE_INFORMATION_UNKNOWN,
        ReadinessReason.PARTY_INFORMATION_UNKNOWN,
        ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN,
    }
)


def _readiness_requires_recheck(readiness: ProgressionReadinessDiagnostic) -> bool:
    """Return whether an UNKNOWN result is expected to change next frame."""

    return getattr(readiness, "readiness_reason", None) in _TRANSIENT_READINESS_REASONS


def _campaign_target_for_execution(objective, tactical_goal: Goal | None) -> SemanticTarget | None:
    """Resolve the GUI target for a newly mounted campaign execution.

    Tactical executors are allowed to publish more specific status as they
    observe the ROM.  This initial projection is nevertheless important at a
    handoff: encounter executors do not otherwise publish a status, and a
    completed recovery status must not remain attached to their next route.
    """

    if isinstance(tactical_goal, NavigationGoal) and isinstance(tactical_goal.target, SemanticTarget):
        return tactical_goal.target
    if isinstance(tactical_goal, ReachLocation):
        return SemanticTarget.at(tactical_goal.location)
    if isinstance(tactical_goal, ReachWarp):
        target_map = tactical_goal.destination_map
        if target_map is None and tactical_goal.destination is not None:
            target_map = tactical_goal.destination[0]
        return SemanticTarget.map(target_map) if target_map is not None else None
    if isinstance(tactical_goal, ActivateTrigger):
        target_map = None
        destination = getattr(objective, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            target_map = destination[0]
        return SemanticTarget.interaction(target_map, tactical_goal.trigger_id)
    if isinstance(tactical_goal, EngageTrainer):
        destination = getattr(objective, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            return SemanticTarget.map(destination)
    destination = getattr(objective, "destination", None)
    if isinstance(destination, tuple) and len(destination) == 2:
        return SemanticTarget.map(destination)
    return None


def _post_pokedex_campaign_boundary(state: CampaignState) -> bool:
    """Return whether the current save is past the overworld safety boundary.

    Opening capabilities such as title-screen setup and starter selection are
    intentionally targetless and cannot supply meaningful overworld readiness
    input.  Once the Pokédex has been received, however, targetless Emerald
    capabilities include ordinary route traversal (for example
    ``reach_petalburg`` and ``complete_petalburg_wally``).  Those capabilities
    still need the event-driven post-battle/map readiness check.
    """

    facts = getattr(state, "campaign_facts", None)
    pokedex = getattr(facts, "pokedex_received", None)
    return getattr(pokedex, "status", None) is FactStatus.KNOWN and getattr(pokedex, "value", None) is True


class CampaignController:
    """Select and mount one tactical goal, reevaluating at event boundaries."""

    def __init__(
        self,
        state_provider: Callable[[], CampaignState],
        *,
        tactical_loop_factory: TacticalLoopFactory | None = None,
        # Campaign Progression must use the dependency frontier: optional
        # encounter objectives can be strategic prerequisites for required
        # story progression. The legacy ordered selector cannot discover them.
        selector: Callable[[CampaignState], ObjectiveSelection] = select_available_campaign_task,
        adapter: Callable[[ObjectiveSelection], CampaignExecutionResult] = adapt_campaign_execution,
        campaign_boundary_handler: CampaignBoundaryHandler | None = None,
        readiness_provider: ReadinessProvider | None = None,
        recovery_factory: RecoveryFactory | None = None,
        readiness_policy: CampaignReadinessPolicy | None = None,
    ) -> None:
        """Create a controller around state, selection, and execution adapters."""

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
        self._tactical_loop_objective_id: str | None = None
        self._readiness_provider = readiness_provider
        self._recovery_factory = recovery_factory
        self._readiness_policy = readiness_policy or CampaignReadinessPolicy()
        self._execution_phase = "CAMPAIGN"
        self._recovery_status: str | None = None
        self.interrupted_objective_id: str | None = None
        # A readiness check can land on the same frame as battle/script
        # teardown.  Keep the recheck obligation separate from the scheduler
        # cache so a deferred recovery decision cannot be lost at that
        # boundary.
        self._readiness_recheck_pending = False
        self._pending_readiness_objective_id: str | None = None
        # Campaign selection and adaptation are boundary work.  Keep the
        # mounted tactical generator on the hot path and only rebuild the
        # campaign envelope when an event invalidates it.
        self._refresh_required = True
        self._refresh_required_reason = "initial"
        self._last_readiness_boundary_key = None
        self._campaign_plan: CampaignPlan | None = None
        self._last_published_campaign_status: CampaignStatus | None = None
        # Task diagnostics contain route/encounter discovery and are only a
        # trace aid.  Keep them on the same semantic signature as campaign
        # planning so enabling tracing cannot turn every frame into a second
        # planner pass.
        self._campaign_task_diagnostics_key = None

    @staticmethod
    def _default_tactical_loop(goal: Goal) -> Iterator[object]:
        """Build the normal agent-control generator for a tactical goal."""

        return AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()

    @property
    def state(self) -> CampaignControllerState:
        """Return the controller's immutable externally visible state."""

        return CampaignControllerState(
            self.status,
            self.current_objective_id,
            self.current_tactical_goal,
            self.last_selection,
            self.last_execution,
            self.transition_reason,
            self._execution_phase,
            self._recovery_status,
            self._campaign_plan,
        )

    def _publish_campaign_status(self, intent: str) -> None:
        """Publish a fresh status at a campaign/recovery ownership boundary."""

        objective = self.last_selection.objective if self.last_selection is not None else None
        self._publish_status_for_execution(objective, self.current_tactical_goal, intent)

    def _publish_status_for_execution(self, objective, tactical_goal: Goal | None, intent: str) -> None:
        """Publish status for an execution before its tactical loop is mounted.

        Readiness is evaluated between objective selection and tactical-loop
        mounting. When that observation is deferred, the controller has no
        loop from which to publish status, while the previous capability may
        have left an obsolete GUI projection behind. Keep the projection tied
        to the selected execution so a deferred handoff is truthful without
        granting the execution permission to run.
        """

        if objective is None:
            status = CampaignStatus(None, None, intent)
            if self._last_published_campaign_status == status:
                return
            self._last_published_campaign_status = status
            context.campaign_status = status
            diagnostic_print(
                lambda: f"CAMPAIGN_STATUS_BOUNDARY: objective=None target=None intent={intent!r}",
                trace=True,
                prefix="CAMPAIGN_STATUS_BOUNDARY",
            )
            return
        status = CampaignStatus(
            getattr(objective, "description", None) or objective.objective_id,
            _campaign_target_for_execution(objective, tactical_goal),
            intent,
            tactical_goal,
        )
        if self._last_published_campaign_status == status:
            return
        self._last_published_campaign_status = status
        context.campaign_status = status
        diagnostic_print(
            lambda: (
                "CAMPAIGN_STATUS_BOUNDARY: "
                f"objective={status.objective!r} target={status.target!r} intent={status.intent!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_STATUS_BOUNDARY",
        )

    def _terminal(self, execution: CampaignExecutionResult, status: CampaignControllerStatus) -> None:
        """Clear active execution and publish a terminal controller status."""

        diagnostic_print(
            lambda: (
                "CAMPAIGN_TACTICAL_LOOP_INVALIDATED: "
                f"controller_id={id(self)!r} loop_id={id(self._tactical_loop) if self._tactical_loop else None!r} "
                f"old_objective={self._tactical_loop_objective_id!r} new_objective=None reason='terminal'"
            ),
            trace=True,
        )
        self.current_objective_id = None
        self.current_tactical_goal = None
        self._tactical_loop = None
        self._pending_readiness_objective_id = None
        self._readiness_recheck_pending = False
        self.last_execution = execution
        self.status = status
        self.transition_reason = execution.reason
        diagnostic_print(
            lambda: (
                "CAMPAIGN_STATUS_TRANSITION: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"status={status.value!r} reason={execution.reason!r} terminal=True"
            ),
            trace=True,
        )

    def _emit_campaign_task_diagnostics(self, observed_state: CampaignState, selection: ObjectiveSelection) -> None:
        """Emit task discovery only when map-level planning facts change."""

        # Do this check before constructing the diagnostic key.  Normal runs
        # must not pay even the small signature cost for a trace-only report.
        if not context.debug or not getattr(context, "debug_trace", False):
            return
        try:
            key = campaign_planning_signature(observed_state)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # A partial/custom state is still valid for controller tests and
            # transient startup boundaries.  Its diagnostics are not safe to
            # cache semantically, so leave the trace suppressed for that frame.
            return
        if key == self._campaign_task_diagnostics_key:
            return
        self._campaign_task_diagnostics_key = key
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
            prefix="CAMPAIGN_TASK_DISCOVERY",
        )

    def request_readiness_recheck(self, reason: str) -> None:
        """Require the next campaign refresh to observe readiness afresh.

        Battle completion can change HP and encounter legality without
        changing the cheap scheduler context. Keep that explicit event
        boundary separate from ordinary movement so a stale readiness result
        cannot authorize the next tactical step.
        """

        self._refresh_required = True
        self._refresh_required_reason = reason
        scheduler = getattr(self._readiness_provider, "__self__", None)
        invalidate = getattr(scheduler, "invalidate", None)
        if callable(invalidate):
            invalidate(reason)
        if (
            self._execution_phase == "CAMPAIGN"
            and self.current_objective_id is not None
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self.current_objective_id
        ):
            self._readiness_recheck_pending = True
            self._pending_readiness_objective_id = self.current_objective_id
        diagnostic_print(
            lambda: (
                "CAMPAIGN_READINESS_RECHECK: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"reason={reason!r} objective={self.current_objective_id!r} "
                f"pending={self._readiness_recheck_pending!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_READINESS_RECHECK:",
        )

    def request_refresh(self, reason: str) -> None:
        """Require the campaign envelope to be rebuilt at the next step.

        This is for cheap lifecycle boundaries that do not necessarily alter
        readiness data (for example, a map transition or a capability
        finishing).  Readiness-changing events should use
        :meth:`request_readiness_recheck` so the scheduler is invalidated as
        well.
        """

        self._refresh_required = True
        self._refresh_required_reason = reason
        diagnostic_print(
            lambda: (
                "CAMPAIGN_REFRESH_REQUEST: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"reason={reason!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_REFRESH_REQUEST",
        )

    def refresh(self) -> CampaignControllerState:
        """Rebuild state, select, adapt, and mount work if needed."""
        observed_state = self._state_provider()
        active_safety = (
            self.last_selection
            if self.last_selection is not None
            and self.last_selection.objective is not None
            and self.last_selection.objective.task_kind == "safety"
            and self.current_objective_id == self.last_selection.objective.objective_id
            else None
        )
        if active_safety is not None:
            completion = active_safety.objective.completion.evaluate(observed_state)
            # A safety capability can restore its resource before the ROM has
            # released the interaction that performed that work.  For
            # example, Emerald restores the party during the nurse's healing
            # script and displays the closing messages afterward.  Keep the
            # mounted generator authoritative until it returns; otherwise a
            # parent campaign objective can replace it while its final input
            # prompt is still active.
            safety_loop_active = self._tactical_loop is not None
            if completion.is_known and completion.value is True and not safety_loop_active:
                # A terminal safety objective releases ownership.  The saved
                # parent is deliberately left for the normal selector so it
                # is resumed using the current campaign facts.
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_SAFETY_TERMINAL: "
                        f"controller_id={id(self)!r} objective={active_safety.objective.objective_id!r} "
                        f"status='complete' interrupted_objective={self.interrupted_objective_id!r}"
                    ),
                    trace=True,
                )
                self.current_objective_id = None
                self.current_tactical_goal = None
                self._tactical_loop = None
                self._tactical_loop_objective_id = None
                self.last_selection = None
                self.interrupted_objective_id = None
                selection = self._selector(observed_state)
            else:
                # Safety ownership is an execution invariant, not a planner
                # preference: an incomplete interrupting objective, or a
                # completed objective with a still-running capability, cannot
                # be replaced merely because its parent remains incomplete.
                selection = active_safety
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_SAFETY_OWNERSHIP: "
                        f"controller_id={id(self)!r} objective={selection.objective.objective_id!r} "
                        f"completion_status={completion.status!r} "
                        f"completion_value={completion.value!r} "
                        f"loop_active={safety_loop_active!r} "
                        f"interrupted_objective={self.interrupted_objective_id!r}"
                    ),
                    trace=True,
                )
        else:
            selection = self._selector(observed_state)

        # A deferred readiness observation has not mounted a tactical loop,
        # so the normal active-loop ownership guard below cannot protect its
        # objective. A battle/script transition can nevertheless make the
        # planner report an earlier task for one frame. Keep the objective
        # that owns the pending recheck while it is still incomplete and
        # safe; otherwise a cached RECOVER result could be applied to the
        # transiently selected task instead of to the task that produced the
        # observation.
        pending_readiness_objective = (
            self.last_selection.objective
            if self._readiness_recheck_pending
            and self._pending_readiness_objective_id is not None
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self._pending_readiness_objective_id
            and self.last_execution is not None
            and self.last_execution.objective is not None
            and self.last_execution.objective.objective_id == self._pending_readiness_objective_id
            else None
        )
        if pending_readiness_objective is not None:
            pending_completion = pending_readiness_objective.completion.evaluate(observed_state)
            pending_failure = (
                pending_readiness_objective.failure.evaluate(observed_state)
                if pending_readiness_objective.failure is not None
                else None
            )
            pending_prerequisites_safe = all(
                prerequisite.evaluate(observed_state).value is not False
                for prerequisite in pending_readiness_objective.prerequisites
            )
            pending_completion_value = getattr(pending_completion, "value", None)
            pending_safe = (
                pending_completion_value is not True
                and (pending_failure is None or pending_failure.value is not True)
                and pending_prerequisites_safe
            )
            if (
                selection.objective is None
                or selection.objective.objective_id != pending_readiness_objective.objective_id
            ):
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_OWNERSHIP_CHECK: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"pending={pending_readiness_objective.objective_id!r} "
                        f"planner_selected={getattr(selection.objective, 'objective_id', None)!r} "
                        f"completion_status={getattr(getattr(pending_completion, 'status', None), 'value', None)!r} "
                        f"completion_value={pending_completion_value!r} "
                        f"prerequisites_safe={pending_prerequisites_safe!r} "
                        f"failure_safe={pending_failure is None or pending_failure.value is not True!r} "
                        f"pending_safe={pending_safe!r}"
                    ),
                    trace=True,
                )
                if pending_safe:
                    selection = self.last_selection
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_READINESS_OWNERSHIP: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"retained={pending_readiness_objective.objective_id!r} "
                            f"planner_selected={getattr(selection.objective, 'objective_id', None)!r} "
                            "reason='deferred_readiness_recheck'"
                        ),
                        trace=True,
                    )
                else:
                    self._readiness_recheck_pending = False
                    self._pending_readiness_objective_id = None

        # Keep ownership of an already-mounted objective across transient
        # battle/script observations.  The planner is intentionally rebuilt
        # from ROM-backed facts on every frame, but a battle-start/return
        # boundary can briefly expose an older or incomplete campaign view.
        # Replacing a live capability at that point discards its generator
        # position; preparation then starts over at its map-level handoff.
        #
        # This is deliberately narrower than making the planner sticky: an
        # active objective may be retained only while it is still incomplete,
        # has no known failure, and none of its prerequisites is known false.
        # A genuine save reset or story regression therefore still releases
        # the old objective for normal ROM-fact selection.
        active_objective = (
            self.last_selection.objective
            if self._tactical_loop is not None
            and self.current_objective_id is not None
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self.current_objective_id
            else None
        )
        if active_objective is not None and selection.objective is not None:
            active_completion = active_objective.completion.evaluate(observed_state)
            active_failure = (
                active_objective.failure.evaluate(observed_state) if active_objective.failure is not None else None
            )
            prerequisites_safe = all(
                prerequisite.evaluate(observed_state).value is not False
                for prerequisite in active_objective.prerequisites
            )
            failure_safe = active_failure is None or active_failure.value is not True
            spatial_observation_coherent = _spatial_observation_is_coherent(observed_state)
            if selection.objective.objective_id != active_objective.objective_id:
                retention_order_allows = _retention_order_allows(
                    active_objective.objective_id,
                    selection.objective.objective_id,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_OBJECTIVE_OWNERSHIP_CHECK: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"active={active_objective.objective_id!r} "
                        f"planner_selected={selection.objective.objective_id!r} "
                        f"completion_status={active_completion.status.value!r} "
                        f"completion_value={active_completion.value!r} "
                        f"prerequisites_safe={prerequisites_safe!r} "
                        f"failure_safe={failure_safe!r} "
                        f"spatial_observation_coherent={spatial_observation_coherent!r} "
                        f"retention_order_allows={retention_order_allows!r}"
                    ),
                    trace=True,
                )
            else:
                retention_order_allows = False
            if (
                selection.objective.objective_id != active_objective.objective_id
                and retention_order_allows
                and (
                    active_completion.status is not FactStatus.KNOWN
                    or (
                        active_completion.status is FactStatus.KNOWN
                        and active_completion.value is False
                        and (prerequisites_safe or not spatial_observation_coherent)
                        and failure_safe
                    )
                )
            ):
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_OBJECTIVE_OWNERSHIP: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"retained={active_objective.objective_id!r} "
                        f"planner_selected={selection.objective.objective_id!r} "
                        f"completion_status={active_completion.status.value!r} "
                        f"completion_value={active_completion.value!r} "
                        f"prerequisites_safe={prerequisites_safe!r} "
                        f"spatial_observation_coherent={spatial_observation_coherent!r} "
                        f"failure_safe={failure_safe!r} reason='transient_observation_boundary'"
                    ),
                    trace=True,
                )
                selection = self.last_selection
        previous_id = (
            self.last_selection.objective.objective_id
            if self.last_selection and self.last_selection.objective
            else None
        )
        selected_id = selection.objective.objective_id if selection.objective else None
        diagnostic_print(
            lambda: (
                "CAMPAIGN_SELECTION_HANDOFF: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"selected_objective={selected_id!r} status={selection.status.value!r} "
                f"previous_objective={previous_id!r} changed={selected_id != previous_id!r} "
                f"map={getattr(getattr(observed_state, 'raw_map', None), 'value', None)!r} "
                f"facts={getattr(observed_state, 'campaign_facts', None)!r}"
            ),
            trace=True,
        )
        self._emit_campaign_task_diagnostics(observed_state, selection)
        trace = getattr(context, "stutter_trace", None)
        if trace is not None and callable(getattr(trace, "mark", None)):
            intro_completion = getattr(observed_state.campaign_facts, "intro_rival_battle_complete", None)
            trace.mark("campaign_selected_objective", selected_id)
            trace.mark("campaign_mounted_objective", self._tactical_loop_objective_id)
            trace.mark(
                "campaign_intro_rival_completion",
                (
                    None
                    if intro_completion is None
                    else {
                        "status": getattr(getattr(intro_completion, "status", None), "value", None),
                        "value": getattr(intro_completion, "value", None),
                    }
                ),
            )
            trace.mark("campaign_execution_phase", self._execution_phase)
        self.last_selection = selection
        execution = self._adapter(selection)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_EXECUTION_HANDOFF: "
                f"controller_id={id(self)!r} objective_passed={getattr(execution.objective, 'objective_id', None)!r} "
                f"existing_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r} "
                f"existing_loop_objective={self._tactical_loop_objective_id!r} "
                f"action={'REPLACE_OR_CREATE' if self._tactical_loop is None or self.current_objective_id != getattr(execution.objective, 'objective_id', None) or self.current_tactical_goal != execution.tactical_goal else 'REUSE'} "
                f"tactical_goal={execution.tactical_goal!r} capability={getattr(execution.capability, '__name__', None)!r}"
            ),
            trace=True,
        )
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
            diagnostic_print(
                lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='terminal_execution' status={self.status.value!r} reason={self.transition_reason!r}",
                trace=True,
            )
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
            diagnostic_print(
                lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='malformed_ready' status={self.status.value!r} reason={self.transition_reason!r}",
                trace=True,
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
            diagnostic_print(
                lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='missing_execution_target' status={self.status.value!r} reason={self.transition_reason!r}",
                trace=True,
            )
            return self.state
        objective_id = execution.objective.objective_id
        if self._campaign_plan is None or self._campaign_plan.parent_objective_id != objective_id:
            self._campaign_plan = CampaignPlan(objective_id)
        readiness_map = None
        readiness_coordinates = None
        # Readiness is tactical/resource gating. Startup Emerald capabilities
        # intentionally have no tactical Goal: they must be allowed to observe
        # title/menu/naming/clock state before overworld readiness exists.
        # Most capability-backed opening objectives must be allowed to observe
        # title/menu/script state before an overworld exists.  The Pokédex
        # handoff and every later targetless capability are different: they
        # can own ordinary overworld traversal, so skipping readiness here
        # would bypass a recovery stop exactly when a battle has reduced HP.
        readiness_capability_boundary = objective_id == "receive_pokedex"
        readiness_applies = (
            execution.tactical_goal is not None
            or readiness_capability_boundary
            or _post_pokedex_campaign_boundary(observed_state)
        )
        if self._readiness_provider is not None and self._execution_phase == "CAMPAIGN" and readiness_applies:
            forced_readiness_recheck = False
            if self._readiness_recheck_pending:
                scheduler = getattr(self._readiness_provider, "__self__", None)
                invalidate = getattr(scheduler, "invalidate", None)
                if callable(invalidate):
                    invalidate("deferred_readiness_recheck")
                forced_readiness_recheck = True
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_RECHECK_FORCED: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        "reason='deferred_readiness_recheck'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_READINESS_RECHECK_FORCED:",
                )
                # A recheck request is a one-observation invalidation.  A
                # transient result below may schedule another one; stable
                # policy results must not create a per-frame invalidation loop.
                self._readiness_recheck_pending = False
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_PROVIDER: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"objective={objective_id!r} tactical_goal={execution.tactical_goal!r} "
                    f"provider_id={id(self._readiness_provider)!r} "
                    f"provider_type={type(self._readiness_provider).__name__!r}"
                ),
                trace=True,
            )
            readiness_input = self._readiness_provider(execution.objective, execution.tactical_goal)
            # The provider supplies observations only. CampaignPlan owns the
            # continue/recover decision and candidate selection so recovery
            # cannot be selected in a controller side-channel and then
            # replaced by a different Center during execution.
            plan = build_campaign_plan(
                execution.objective,
                readiness_input,
                policy=self._readiness_policy,
            )
            if isinstance(readiness_input, ProgressionReadinessDiagnostic):
                evaluated = replace(
                    readiness_input,
                    readiness_decision=plan.readiness_decision,
                    readiness_reason=plan.readiness_reason,
                )
            else:
                evaluated = readiness_input
            if evaluated.readiness_decision is None:
                evaluated = replace(
                    evaluated,
                    readiness_decision=getattr(readiness_input, "readiness_decision", None),
                    readiness_reason=getattr(readiness_input, "readiness_reason", None),
                )
            readiness_map = evaluated.current_map
            readiness_coordinates = evaluated.current_coordinates
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_RESULT: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"evaluation_id={id(evaluated)!r} objective={objective_id!r} "
                    f"tactical_goal={execution.tactical_goal!r} "
                    f"decision={evaluated.readiness_decision.value!r} "
                    f"reason={evaluated.readiness_reason.value!r} "
                    f"overworld_availability={evaluated.overworld_availability.value!r} "
                    f"provider_result_type={type(readiness_input).__name__!r} "
                    f"provider_state_type={type(getattr(self._readiness_provider, '__self__', None)).__name__!r} "
                    f"map={evaluated.current_map!r} coordinates={evaluated.current_coordinates!r}"
                ),
                trace=True,
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_DECISION: "
                    f"frame={getattr(getattr(evaluated, 'snapshot', None), 'frame', None)!r} "
                    f"evaluation_id={id(evaluated)} objective={objective_id!r} "
                    f"decision={evaluated.readiness_decision.value if evaluated.readiness_decision else None} "
                    f"reason={evaluated.readiness_reason.value if evaluated.readiness_reason else None} "
                    f"recover_accepted={evaluated.readiness_decision is ReadinessDecision.RECOVER} "
                    f"recovery_factory_available={self._recovery_factory is not None} "
                    f"recovery_not_invoked_reason={None if evaluated.readiness_decision is ReadinessDecision.RECOVER else 'decision_not_recover'} "
                    f"controller_phase_before={self._execution_phase}"
                ),
                trace=True,
            )
            schedule = getattr(getattr(self._readiness_provider, "__self__", None), "state", None)
            readiness_boundary_key = (
                objective_id,
                getattr(schedule, "refresh_count", None),
                getattr(evaluated, "current_map", None),
                getattr(getattr(evaluated, "readiness_decision", None), "value", None),
                getattr(getattr(evaluated, "readiness_reason", None), "value", None),
                getattr(getattr(evaluated, "overworld_availability", None), "value", None),
                getattr(getattr(evaluated, "resource_availability", None), "value", None),
            )
            if readiness_boundary_key != self._last_readiness_boundary_key:
                self._last_readiness_boundary_key = readiness_boundary_key
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_BOUNDARY: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} refresh_count={getattr(schedule, 'refresh_count', None)!r} "
                        f"tick_count={getattr(schedule, 'tick_count', None)!r} "
                        f"current_map={evaluated.current_map!r} "
                        f"decision={getattr(getattr(evaluated, 'readiness_decision', None), 'value', None)!r} "
                        f"reason={getattr(getattr(evaluated, 'readiness_reason', None), 'value', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_READINESS_BOUNDARY",
                )
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
                    default=str,
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
                # An incomplete observation is a defer/reobserve boundary, not
                # permission to transfer ownership to recovery.  In
                # particular, battle and script transitions can temporarily
                # hide the overworld while the campaign context remains the
                # rightful owner. A targetless observation capability still
                # needs to be mounted at this boundary, however: it is the
                # capability that can observe the script handoff and wait for
                # the overworld to become actionable. Leaving it unmounted
                # strands the controller with the completed capability's GUI
                # status and no loop to advance on the next frame.
                diagnostic_print(
                    lambda: f"CAMPAIGN_READINESS_DEFERRED: controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} objective={objective_id!r} reason={evaluated.readiness_reason.value!r}",
                    trace=True,
                )
                # The scheduler retains this bounded UNKNOWN observation, so
                # do not turn it into a per-frame invalidation loop.  An
                # explicit battle-end request has already forced the next
                # observation; if that observation is still transient, let
                # its bounded cache/age policy handle the retry.  Retain the
                # one retry for custom providers that do not expose a cache,
                # preserving the controller's historical startup behavior.
                scheduler = getattr(self._readiness_provider, "__self__", None)
                cached_observation = bool(getattr(scheduler, "last_observation_was_cache_hit", False))
                self._readiness_recheck_pending = (
                    not forced_readiness_recheck and not cached_observation and _readiness_requires_recheck(evaluated)
                )
                self._pending_readiness_objective_id = objective_id if self._readiness_recheck_pending else None
                self._publish_status_for_execution(
                    execution.objective,
                    execution.tactical_goal,
                    "Waiting / readiness temporarily unavailable",
                )
                targetless_capability_boundary = execution.capability is not None and execution.tactical_goal is None
                if not targetless_capability_boundary:
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = f"readiness deferred: {evaluated.readiness_reason.value}"
                    diagnostic_print(
                        lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='readiness_deferred' status={self.status.value!r} reason={self.transition_reason!r}",
                        trace=True,
                    )
                    return self.state
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_CAPABILITY_MOUNT: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} reason={evaluated.readiness_reason.value!r} "
                        "action='mount_targetless_capability'"
                    ),
                    trace=True,
                )
            self._readiness_recheck_pending = False
            self._pending_readiness_objective_id = None
            if evaluated.readiness_decision is ReadinessDecision.RECOVER:
                stop = plan.recovery_stop
                self._campaign_plan = plan
                self.interrupted_objective_id = objective_id
                if stop is None:
                    if (
                        evaluated.game_state != "OVERWORLD"
                        or evaluated.overworld_availability is not None
                        and evaluated.overworld_availability.value != "known"
                    ):
                        # A contradictory recovery result can still be
                        # supplied by a stale/custom readiness provider at a
                        # battle or script boundary.  Preserve campaign
                        # ownership and ask for a stable re-observation; do
                        # not turn the absence of a route into a fallback
                        # Center selection.
                        # A forced retry has already consumed the one
                        # compatibility re-observation. If the second
                        # observation is still contradictory, wait for a new
                        # lifecycle event instead of refreshing every frame.
                        self._readiness_recheck_pending = not forced_readiness_recheck
                        self._pending_readiness_objective_id = objective_id if self._readiness_recheck_pending else None
                        self.status = CampaignControllerStatus.READY
                        self._execution_phase = "CAMPAIGN"
                        self._publish_campaign_status("Recovery deferred; re-evaluating campaign readiness")
                        self.transition_reason = (
                            "recovery deferred: "
                            f"{evaluated.readiness_reason.value if evaluated.readiness_reason else 'recovery requested'}; "
                            "OVERWORLD_UNAVAILABLE (overworld observation unavailable)"
                        )
                        return self.state
                    # A recovery decision without a concrete planner-owned
                    # stop is not permission to reopen the healing catalog.
                    # Block explicitly so the caller can improve route or
                    # resource observations and re-enter the campaign.
                    self.status = CampaignControllerStatus.BLOCKED
                    self._execution_phase = "BLOCKED"
                    self._recovery_status = "UNAVAILABLE"
                    context.campaign_status = CampaignStatus(
                        None,
                        None,
                        "Campaign blocked: no executable recovery route",
                    )
                    self.transition_reason = "campaign planner produced no executable recovery stop"
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RECOVERY_PLAN_UNAVAILABLE: "
                            f"controller_id={id(self)!r} objective={objective_id!r} "
                            "fallback='disabled'"
                        ),
                        trace=True,
                    )
                    return self.state
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
                    if self._recovery_factory is None:
                        raise RuntimeError("recovery factory is unavailable")
                    self._tactical_loop = self._recovery_factory(stop)
                    diagnostic_print(
                        lambda: f"CAMPAIGN_RECOVERY_GENERATOR: event=created controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} frame={getattr(context, 'frame', None)!r}",
                        trace=True,
                    )
                    diagnostic_print(
                        lambda: f"CAMPAIGN_RECOVERY_FACTORY: evaluation_id={id(evaluated)} invoked=True method={getattr(self._recovery_factory, '__name__', type(self._recovery_factory).__name__)}",
                        trace=True,
                    )
                except Exception as error:
                    self._tactical_loop = None
                    self.status = CampaignControllerStatus.BLOCKED
                    self._execution_phase = "BLOCKED"
                    self._recovery_status = "UNAVAILABLE"
                    context.campaign_status = CampaignStatus(
                        None,
                        None,
                        f"Recovery unavailable: {type(error).__name__}",
                    )
                    self.transition_reason = f"recovery unavailable: {error}"
                    diagnostic_print(
                        lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='recovery_unavailable' status={self.status.value!r} reason={self.transition_reason!r}",
                        trace=True,
                    )
                    return self.state
                self._execution_phase = "RECOVERY"
                self._recovery_status = "ACTIVE"
                self.current_objective_id = objective_id
                self.current_tactical_goal = execution.tactical_goal
                self._tactical_loop_objective_id = objective_id
                self.status = CampaignControllerStatus.READY
                context.campaign_status = recovery_status(None, "Recovery planned")
                self.transition_reason = f"recover before {objective_id}"
                diagnostic_print(
                    lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='recovery_mounted' status={self.status.value!r} reason={self.transition_reason!r}",
                    trace=True,
                )
                return self.state
        elif self._readiness_recheck_pending:
            # A lifecycle event can arrive while an opening targetless
            # capability owns the controller.  That capability has no
            # meaningful overworld readiness input, so retaining the pending
            # flag would force ``step()`` to refresh every frame forever.
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_RECHECK_DISCARDED: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"objective={objective_id!r} reason='readiness_not_applicable'"
                ),
                trace=True,
                prefix="CAMPAIGN_READINESS_RECHECK_DISCARDED",
            )
            self._readiness_recheck_pending = False
            self._pending_readiness_objective_id = None
        if (
            self.current_objective_id != objective_id
            or self.current_tactical_goal != execution.tactical_goal
            or self._tactical_loop is None
        ):
            self.current_objective_id = objective_id
            self.current_tactical_goal = execution.tactical_goal
            old_loop_id = id(self._tactical_loop) if self._tactical_loop else None
            old_loop_objective = self._tactical_loop_objective_id
            self._tactical_loop = (
                execution.capability()
                if execution.capability is not None
                else self._tactical_loop_factory(execution.tactical_goal)
            )
            self._tactical_loop_objective_id = objective_id
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_LOOP_BOUNDARY: "
                    f"controller_id={id(self)!r} old_loop_id={old_loop_id!r} old_objective={old_loop_objective!r} "
                    f"new_loop_id={id(self._tactical_loop)!r} new_objective={objective_id!r} reason='mount_or_replace'"
                ),
                trace=True,
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_LOOP_CREATED: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"loop_id={id(self._tactical_loop)!r} objective={objective_id!r} "
                    f"tactical_goal={execution.tactical_goal!r} phase={self._execution_phase!r} "
                    f"status={self.status.value!r} map={readiness_map!r} coordinates={readiness_coordinates!r}"
                ),
                trace=True,
            )
            self._publish_campaign_status("Campaign objective selected")
            self.transition_reason = f"selected {objective_id}"
        self.status = CampaignControllerStatus.READY
        return self.state

    def step(self) -> CampaignControllerState:
        """Advance one mounted capability and return its current status."""

        diagnostic_print(
            lambda: (
                "CAMPAIGN_CONTROLLER_STEP_ENTRY: "
                f"controller_id={id(self)!r} phase={self._execution_phase!r} "
                f"status={self.status.value!r} tactical_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r}"
            ),
            trace=True,
        )
        result = self._step_impl()
        diagnostic_print(
            lambda: (
                "CAMPAIGN_CONTROLLER_STEP_EXIT: "
                f"controller_id={id(self)!r} phase={self._execution_phase!r} "
                f"status={result.status.value!r} tactical_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r}"
            ),
            trace=True,
        )
        return result

    def _step_impl(self) -> CampaignControllerState:
        """Perform one controller step across campaign or recovery phases."""

        if self._execution_phase == "BLOCKED" or self.status in {
            CampaignControllerStatus.COMPLETE,
            CampaignControllerStatus.FAILED,
        }:
            return self.state
        if self._execution_phase == "RECOVERY":
            try:
                if self._tactical_loop is None:
                    raise RuntimeError("recovery has no executable capability")
                diagnostic_print(
                    lambda: f"CAMPAIGN_RECOVERY_GENERATOR: event=advance controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} frame={getattr(context, 'frame', None)!r}",
                    trace=True,
                )
                next(self._tactical_loop)
            except StopIteration:
                completed_recovery_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
                self._tactical_loop = None
                self._refresh_required = True
                self._refresh_required_reason = "recovery_completed"
                self._execution_phase = "CAMPAIGN"
                self._recovery_status = "COMPLETED"
                if self._campaign_plan is not None:
                    self._campaign_plan = self._campaign_plan.complete_active_stop()
                scheduler = getattr(self._readiness_provider, "__self__", None)
                invalidate = getattr(scheduler, "invalidate", None)
                if callable(invalidate):
                    invalidate("recovery_completed")
                self.status = CampaignControllerStatus.READY
                self._publish_campaign_status("Recovery complete; resuming campaign")
                self.transition_reason = "recovery verified; reevaluating campaign readiness"
                diagnostic_print(
                    lambda: f"CAMPAIGN_RECOVERY_COMPLETE: frame={getattr(context, 'frame', None)!r} controller_id={id(self)!r} recovery_loop_id={completed_recovery_loop_id!r} objective={self.current_objective_id!r} invalidation='recovery_completed' resumed_phase={self._execution_phase}",
                    trace=True,
                )
            except Exception as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_EXCEPTION: "
                        f"frame={getattr(context, 'frame', None)!r} "
                        f"emulator_frame={context.emulator.get_frame_count()!r} "
                        f"controller_id={id(self)!r} "
                        f"recovery_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r} "
                        f"exception_type={type(error).__name__!r} exception={str(error)!r} "
                        f"exception_repr={error!r} phase={self._execution_phase!r} "
                        f"status={self.status.value!r} recovery_status={self._recovery_status!r} "
                        f"objective={self.current_objective_id!r} transition_reason={self.transition_reason!r}"
                    ),
                    trace=True,
                )
                self._tactical_loop = None
                self._execution_phase = "BLOCKED"
                self._recovery_status = "FAILED"
                self.status = CampaignControllerStatus.BLOCKED
                context.campaign_status = CampaignStatus(
                    None,
                    None,
                    f"Recovery failed: {type(error).__name__}",
                )
                self.transition_reason = f"recovery failed: {error}"
            return self.state
        try:
            if self._refresh_required or self._readiness_recheck_pending:
                refresh_reason = self._refresh_required_reason
                current = self.refresh()
                # A refresh may deliberately leave no tactical loop mounted
                # while readiness is deferred (for example, while a battle
                # listener owns the emulator).  That is still a handled
                # boundary; at most the one compatibility retry represented
                # by _readiness_recheck_pending may follow it. Further
                # refreshes must come from an explicit event rather than from
                # the absence of a loop alone.
                if current.status in {
                    CampaignControllerStatus.READY,
                    CampaignControllerStatus.UNKNOWN,
                    CampaignControllerStatus.BLOCKED,
                    CampaignControllerStatus.COMPLETE,
                    CampaignControllerStatus.FAILED,
                }:
                    self._refresh_required = False
                    self._refresh_required_reason = None
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_REFRESH_BOUNDARY: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"reason={refresh_reason!r} status={current.status.value!r} "
                        f"loop_id={id(self._tactical_loop) if self._tactical_loop else None!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_REFRESH_BOUNDARY",
                )
            else:
                current = self.state
        except Exception as error:
            # Readiness and planner inputs are assembled from several ROM
            # structures that do not become visible on the same frame after a
            # battle. A transient refresh failure must not unwind the mode
            # generator and silently return the bot to manual mode; discard
            # only the frame-local tactical loop and retry from fresh facts.
            failed_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
            self._tactical_loop = None
            self._tactical_loop_objective_id = None
            self._refresh_required = True
            self._refresh_required_reason = "refresh_exception"
            self.status = CampaignControllerStatus.UNKNOWN
            self.transition_reason = f"campaign refresh failed; will re-evaluate: {type(error).__name__}: {error!r}"
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_REFRESH_EXCEPTION: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"failed_loop_id={failed_loop_id!r} status={self.status.value!r} "
                    f"reason={self.transition_reason!r} traceback={traceback.format_exc()!r}"
                ),
                trace=True,
            )
            return self.state
        if current.status is not CampaignControllerStatus.READY or self._tactical_loop is None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_STEP_RETURN: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"branch={'loop_absent' if self._tactical_loop is None else 'not_ready'} "
                    f"status={current.status.value!r} reason={current.reason!r} input_emitted=False"
                ),
                trace=True,
            )
            return current
        try:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_STEP_HANDOFF: "
                    f"controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} "
                    f"mounted_objective={self._tactical_loop_objective_id!r} "
                    f"planner_objective={getattr(getattr(self.last_selection, 'objective', None), 'objective_id', None)!r} "
                    f"tactical_goal={self.current_tactical_goal!r}"
                ),
                trace=True,
            )
            next(self._tactical_loop)
        except StopIteration:
            if self.last_execution is not None and self.last_execution.capability is not None:
                # A capability may yield control at a ROM boundary while its
                # authoritative completion fact is still false. Re-observe
                # and remount it on the next frame instead of declaring a
                # tactical failure.
                self._tactical_loop = None
                self._refresh_required = True
                self._refresh_required_reason = "capability_boundary"
                self.transition_reason = "capability boundary; awaiting authoritative completion"
                diagnostic_print(
                    lambda: f"CAMPAIGN_STEP_RETURN: controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} branch='capability_boundary' status={self.status.value!r} reason={self.transition_reason!r} input_emitted=unknown",
                    trace=True,
                )
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
            self._refresh_required = True
            self._refresh_required_reason = "tactical_exception"
            self.status = CampaignControllerStatus.UNKNOWN
            self.transition_reason = f"tactical execution failed; will re-evaluate: {type(error).__name__}: {error!r}"
            diagnostic_print(
                lambda: (
                    f"CAMPAIGN_STEP_RETURN: controller_id={id(self)!r} "
                    f"frame={getattr(context, 'frame', None)!r} branch='tactical_exception' "
                    f"status={self.status.value!r} reason={self.transition_reason!r} "
                    f"input_emitted=False traceback={traceback.format_exc()!r}"
                ),
                trace=True,
            )
        return self.state

    def run(self) -> Generator:
        """Yield frame boundaries until the campaign reaches a terminal status."""

        while True:
            state = self.step()
            if state.status is CampaignControllerStatus.UNKNOWN:
                # UNKNOWN is a transient observation boundary.  In
                # particular, UI capabilities can be mounted while the ROM
                # has not yet exposed overworld readiness; do not terminate
                # the controller before the mounted capability gets a chance
                # to execute on the next frame.
                yield
                continue
            if state.status is not CampaignControllerStatus.READY:
                return
            yield


def runtime_campaign_state() -> CampaignState:
    """Build a facade from the already-materialized runtime state."""
    from modules.context import context
    from modules.map_data import get_map_enum
    from .snapshots import get_nuzlocke_snapshot

    runtime = context.nuzlocke_runtime
    # ``NuzlockeRuntime.update`` already materializes the authoritative
    # snapshot for this application frame.  Reusing it avoids a second set of
    # save-block reads between observation and campaign selection, which can
    # otherwise stall the frame loop on loaded mGBA states.
    snapshot = getattr(runtime, "latest_snapshot", None) if runtime is not None else None
    if snapshot is None:
        snapshot = get_nuzlocke_snapshot()
    canonical_area = None
    if snapshot.player_available and snapshot.player.map_group is not None and snapshot.player.map_number is not None:
        try:
            canonical_area = get_map_enum((snapshot.player.map_group, snapshot.player.map_number)).name
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
    history_ready = runtime is None or runtime.campaign_history_ready
    return CampaignState.from_runtime_state(
        snapshot=snapshot,
        observed_projection=runtime.observed_projection if history_ready and runtime is not None else None,
        rules_projection=runtime.rules_projection if history_ready and runtime is not None else None,
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
