"""Pure campaign-plan composition, including transient recovery stops."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from modules.goals import Goal, ReachLocation

from .campaign_objectives import CampaignObjective
from .readiness_diagnostics import (
    CampaignReadinessPolicy,
    ReadinessDecision,
    ReadinessReason,
)


class RecoveryUrgency(Enum):
    """Why the campaign planner inserted a recovery stop."""

    OPPORTUNISTIC = "opportunistic"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class RecoveryStop:
    """A selected, executable recovery waypoint for a campaign objective."""

    source: Goal
    destination: tuple[Any, tuple[int, int]]
    parent_objective_id: str
    route: Any = None
    reason: str = ""
    continuation_route: Any = None
    urgency: RecoveryUrgency = RecoveryUrgency.OPPORTUNISTIC


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """The transient execution envelope around one parent objective."""

    parent_objective_id: str
    stops: tuple[RecoveryStop, ...] = ()
    active_stop: int = 0
    readiness_decision: ReadinessDecision | None = None
    readiness_reason: ReadinessReason | None = None

    @property
    def recovery_stop(self) -> RecoveryStop | None:
        """Return the active recovery stop, if the plan has one remaining."""

        if self.active_stop >= len(self.stops):
            return None
        return self.stops[self.active_stop]

    def complete_active_stop(self) -> "CampaignPlan":
        """Return a plan advanced past its current recovery stop."""

        return CampaignPlan(
            self.parent_objective_id,
            self.stops,
            self.active_stop + 1,
            self.readiness_decision,
            self.readiness_reason,
        )


def _candidate_location(candidate: Any, fallback=None) -> tuple[Any, tuple[int, int]] | None:
    """Extract a concrete outdoor destination from a route candidate."""

    destination = getattr(candidate, "destination", None)
    target = getattr(destination, "target", destination)
    location = getattr(target, "location", None)
    if isinstance(location, tuple) and len(location) == 2 and isinstance(location[1], tuple) and len(location[1]) == 2:
        return location
    # ROM healing goals intentionally target the source interior map, while
    # RecoveryStop retains the catalog's outdoor destination for status and
    # execution.  The readiness observation is the authority for that paired
    # representation; do not guess a door coordinate from the interior map.
    if isinstance(fallback, tuple) and len(fallback) == 2 and isinstance(fallback[1], tuple) and len(fallback[1]) == 2:
        return fallback
    return None


def build_campaign_plan(
    objective: CampaignObjective,
    readiness: Any | None = None,
    *,
    policy: CampaignReadinessPolicy | None = None,
) -> CampaignPlan:
    """Compose a parent objective with the best recovery stop.

    Readiness is an observation bundle. The planner owns the continue/recover
    decision, candidate choice, and exact first/continuation routes. A
    critical recovery may be planned even when the parent route is currently
    unavailable; only the route to the healing source is required to make
    that safety decision executable.
    """

    if readiness is None:
        return CampaignPlan(objective.objective_id)

    # Older callers may provide an already-evaluated diagnostic. Preserve that
    # compatibility while making the live campaign path pass observations
    # directly to the planner. The controller must not make a second,
    # side-channel readiness decision before this function is called.
    decision = getattr(readiness, "readiness_decision", None)
    reason = getattr(readiness, "readiness_reason", None)
    pre_evaluated = decision is not None
    if decision is None:
        evaluation = (policy or CampaignReadinessPolicy()).evaluate(readiness)
        decision = evaluation.decision
        reason = evaluation.reason
    # Keep the historical value semantics for callers that already supplied
    # a policy result. Newly planned observations carry the result on the
    # plan so the live controller can publish the planner's decision.
    base = (
        CampaignPlan(objective.objective_id)
        if pre_evaluated
        else CampaignPlan(objective.objective_id, readiness_decision=decision, readiness_reason=reason)
    )
    if decision is not ReadinessDecision.RECOVER:
        return base
    analysis = getattr(readiness, "route_analysis", None)
    critical = getattr(reason, "value", "") in {
        ReadinessReason.CRITICAL_PARTY_HP.value,
        ReadinessReason.NO_USABLE_POKEMON.value,
        ReadinessReason.FAINTED_PARTY_MEMBER.value,
        ReadinessReason.POISONED_PARTY.value,
    }
    candidates = tuple(
        candidate
        for candidate in getattr(analysis, "candidates", ())
        if getattr(candidate, "reachable", False)
        and getattr(candidate, "first_route", None) is not None
        and (
            critical
            or (getattr(candidate, "total_cost", None) is not None and getattr(candidate, "detour", None) is not None)
        )
    )
    if not candidates:
        # A targetless capability (notably the post-rival Pokédex handoff)
        # has no parent navigation goal to compose against. Critical and
        # already-approved opportunistic recovery still need an executable
        # stop in that case; the route observation has already selected the
        # concrete Center and measured its route. Do not reopen the healing
        # catalog or invent a different destination here.
        recovery = getattr(readiness, "recovery", None)
        destination = getattr(recovery, "center_location", None)
        center_available = bool(recovery and recovery.center_available)
        safe_to_reach_center = bool(recovery and recovery.safe_to_reach_center)
        if (
            (critical or reason is ReadinessReason.OPPORTUNISTIC_RECOVERY)
            and center_available
            and safe_to_reach_center
            and isinstance(destination, tuple)
            and len(destination) == 2
            and isinstance(destination[1], tuple)
            and len(destination[1]) == 2
        ):
            return CampaignPlan(
                objective.objective_id,
                (
                    RecoveryStop(
                        source=ReachLocation(destination),
                        destination=destination,
                        parent_objective_id=objective.objective_id,
                        route=getattr(recovery, "route", None),
                        reason=getattr(reason, "value", "recovery required"),
                        urgency=RecoveryUrgency.CRITICAL if critical else RecoveryUrgency.OPPORTUNISTIC,
                    ),
                ),
                readiness_decision=decision,
                readiness_reason=reason,
            )
    if not candidates:
        return base
    if critical:
        # The parent objective may be temporarily unrouteable because a
        # transition or interaction has not settled.  Prefer the shortest
        # known route to a reachable healing source, then use the composed
        # route when it is available as a deterministic tie-breaker.
        selected = min(
            candidates,
            key=lambda candidate: (
                (
                    getattr(getattr(candidate, "first_route", None), "metrics", None).total_route_cost
                    if getattr(getattr(candidate, "first_route", None), "metrics", None) is not None
                    else float("inf")
                ),
                (
                    getattr(candidate, "total_cost", None)
                    if getattr(candidate, "total_cost", None) is not None
                    else float("inf")
                ),
            ),
        )
    else:
        selected = min(candidates, key=lambda candidate: (candidate.detour, candidate.total_cost))
    location = _candidate_location(
        selected,
        getattr(getattr(readiness, "recovery", None), "center_location", None),
    )
    if location is None:
        return base
    return CampaignPlan(
        objective.objective_id,
        (
            RecoveryStop(
                source=selected.destination,
                destination=location,
                parent_objective_id=objective.objective_id,
                route=selected.first_route,
                reason=getattr(reason, "value", "recovery required"),
                continuation_route=getattr(selected, "continuation_route", None),
                urgency=RecoveryUrgency.CRITICAL if critical else RecoveryUrgency.OPPORTUNISTIC,
            ),
        ),
        readiness_decision=decision,
        readiness_reason=reason,
    )
