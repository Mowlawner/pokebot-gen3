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
    # Preserve the executable goal selected by route recovery, including an
    # exact Center door transition. The public destination remains the
    # outdoor catalog location for status and compatibility. Keep this field
    # last so older positional RecoveryStop construction remains compatible.
    navigation_goal: Any = None
    # Carry the exact healing catalog source selected by route analysis so
    # execution does not have to reverse-map the destination.
    healing_source: Any = None
    # Recovery execution must be explicit.  A same-tile stop is a valid
    # field-item action only when the planner selected it as such; it is not
    # an implicit permission to consume a Potion after a Center search timed
    # out or returned no route.
    execution_kind: str = "center_route"
    # Origin of the route carried by this stop. The controller validates it
    # against the live readiness location before mounting recovery.
    route_origin: Any = None


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


def _readiness_route_origin(readiness: Any) -> tuple[Any, tuple[int, int]] | None:
    """Return a usable live origin for a planner-created recovery stop."""

    current_map = getattr(readiness, "current_map", None)
    current_coordinates = getattr(readiness, "current_coordinates", None)
    if (
        current_map is not None
        and isinstance(current_coordinates, tuple)
        and len(current_coordinates) == 2
    ):
        return current_map, current_coordinates
    return None


def _observed_recovery_stop(
    objective: CampaignObjective,
    readiness: Any,
    reason: ReadinessReason | None,
    critical: bool,
) -> RecoveryStop | None:
    """Build a stop from the recovery worker's verified route.

    Route analysis compares the healing detour with the parent objective. It
    is advisory, however: a completed recovery route remains executable even
    when the parent continuation is unavailable or a route-analysis candidate
    cannot be converted into a public destination. Keep the already-validated
    route as the safety fallback and let the controller replan the parent
    objective after healing.
    """

    recovery = getattr(readiness, "recovery", None)
    destination = getattr(recovery, "center_location", None)
    route = getattr(recovery, "route", None)
    if not (
        recovery is not None
        # ``center_available`` describes the catalog classification. The
        # executable route may instead be reported through the broader
        # healing-source capability, so the route itself is authoritative
        # once the observer has marked it safe.
        and getattr(recovery, "safe_to_reach_center", True) is not False
        and route is not None
        and isinstance(destination, tuple)
        and len(destination) == 2
        and isinstance(destination[1], tuple)
        and len(destination[1]) == 2
    ):
        return None
    route_origin = getattr(recovery, "route_origin", None)
    if route_origin is None:
        route_origin = _readiness_route_origin(readiness)
    return RecoveryStop(
        source=ReachLocation(destination),
        destination=destination,
        parent_objective_id=objective.objective_id,
        route=route,
        route_origin=route_origin,
        navigation_goal=getattr(recovery, "navigation_goal", None),
        healing_source=getattr(recovery, "healing_source", None),
        reason=getattr(reason, "value", "recovery required"),
        urgency=RecoveryUrgency.CRITICAL if critical else RecoveryUrgency.OPPORTUNISTIC,
        execution_kind="center_route",
    )


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
        ReadinessReason.STATUS_CONDITION.value,
    }
    observed_recovery_stop = _observed_recovery_stop(objective, readiness, reason, critical)
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
        recovery = getattr(readiness, "recovery", None)
        # A completed incumbent remains executable even when a later
        # candidate timed out.  The route worker has already validated the
        # selected Center path; only the comparison against other Centers is
        # incomplete. Preserve that safety route and let the campaign replan
        # its parent objective after healing.
        if observed_recovery_stop is not None:
            return CampaignPlan(
                objective.objective_id,
                (observed_recovery_stop,),
                readiness_decision=decision,
                readiness_reason=reason,
            )
        # A bounded route calculation is an incomplete observation. Keep
        # ownership in the campaign controller so it can retry the Center
        # search; consuming a limited field item here would turn search
        # uncertainty into an irreversible resource decision.
        if getattr(recovery, "calculation_pending", False) or getattr(recovery, "calculation_timed_out", False):
            return base
        # A targetless capability (notably the post-rival Pokédex handoff)
        # has no parent navigation goal to compose against. Critical and
        # already-approved opportunistic recovery still need an executable
        # stop in that case; the route observation has already selected the
        # concrete Center and measured its route. Do not reopen the healing
        # catalog or invent a different destination here.
        # Candidate diagnostics describe alternative healing sources. A
        # bounded or pending alternative must not invalidate a completed route
        # already selected by the recovery worker. Only suppress the fallback
        # when there is no executable observed route to mount.
        if any(
            getattr(candidate, "outcome", None) in {"pruned", "pending"}
            for candidate in (getattr(recovery, "candidate_diagnostics", ()) or ())
        ):
            return base
        # A bag cure is an executable recovery stop even when no Center route
        # is available. Keep it at the current location so the recovery
        # factory can consume the item without inventing a navigation target.
        resources = getattr(readiness, "resource_snapshot", None)
        current_map = getattr(readiness, "current_map", None)
        current_coordinates = getattr(readiness, "current_coordinates", None)
        requires_center_recovery = getattr(reason, "value", reason) in {
            ReadinessReason.FAINTED_PARTY_MEMBER.value,
            ReadinessReason.NO_USABLE_POKEMON.value,
        }
        if (
            not requires_center_recovery
            and
            resources is not None
            and getattr(resources, "bag_healing_items", ())
            and isinstance(current_map, tuple)
            and len(current_map) == 2
            and isinstance(current_coordinates, tuple)
            and len(current_coordinates) == 2
        ):
            location = (current_map, current_coordinates)
            return CampaignPlan(
                objective.objective_id,
                (
                    RecoveryStop(
                        source=ReachLocation(location),
                        destination=location,
                        parent_objective_id=objective.objective_id,
                        reason=getattr(reason, "value", "item recovery required"),
                        urgency=RecoveryUrgency.CRITICAL if critical else RecoveryUrgency.OPPORTUNISTIC,
                        execution_kind="field_item",
                        route_origin=(current_map, current_coordinates),
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
        if observed_recovery_stop is not None:
            return CampaignPlan(
                objective.objective_id,
                (observed_recovery_stop,),
                readiness_decision=decision,
                readiness_reason=reason,
            )
        return base
    route_origin = getattr(getattr(readiness, "recovery", None), "route_origin", None)
    if route_origin is None:
        route_origin = _readiness_route_origin(readiness)
    return CampaignPlan(
        objective.objective_id,
        (
            RecoveryStop(
                source=selected.destination,
                destination=location,
                parent_objective_id=objective.objective_id,
                route=selected.first_route,
                route_origin=route_origin,
                navigation_goal=getattr(getattr(readiness, "recovery", None), "navigation_goal", None),
                healing_source=getattr(getattr(readiness, "recovery", None), "healing_source", None),
                reason=getattr(reason, "value", "recovery required"),
                continuation_route=getattr(selected, "continuation_route", None),
                urgency=RecoveryUrgency.CRITICAL if critical else RecoveryUrgency.OPPORTUNISTIC,
                execution_kind="center_route",
            ),
        ),
        readiness_decision=decision,
        readiness_reason=reason,
    )
