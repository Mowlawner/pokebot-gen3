"""Pure campaign-plan composition, including transient recovery stops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.goals import Goal

from .campaign_objectives import CampaignObjective


@dataclass(frozen=True, slots=True)
class RecoveryStop:
    """A selected, executable recovery waypoint for a campaign objective."""

    source: Goal
    destination: tuple[Any, tuple[int, int]]
    parent_objective_id: str
    route: Any = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """The transient execution envelope around one parent objective."""

    parent_objective_id: str
    stops: tuple[RecoveryStop, ...] = ()
    active_stop: int = 0

    @property
    def recovery_stop(self) -> RecoveryStop | None:
        if self.active_stop >= len(self.stops):
            return None
        return self.stops[self.active_stop]

    def complete_active_stop(self) -> "CampaignPlan":
        return CampaignPlan(self.parent_objective_id, self.stops, self.active_stop + 1)


def _candidate_location(candidate: Any) -> tuple[Any, tuple[int, int]] | None:
    destination = getattr(candidate, "destination", None)
    target = getattr(destination, "target", destination)
    location = getattr(target, "location", None)
    if not isinstance(location, tuple) or len(location) != 2:
        return None
    if not isinstance(location[1], tuple) or len(location[1]) != 2:
        return None
    return location


def build_campaign_plan(
    objective: CampaignObjective,
    readiness: Any | None = None,
) -> CampaignPlan:
    """Compose a parent objective with the best currently selected Center.

    Readiness remains an observation/policy input.  This function only turns a
    RECOVER result and its already-composed route analysis into an explicit
    plan step; it never reads emulator state or chooses a new Center.
    """

    if readiness is None or getattr(readiness, "readiness_decision", None) is None:
        return CampaignPlan(objective.objective_id)
    if getattr(readiness.readiness_decision, "value", None) != "RECOVER":
        return CampaignPlan(objective.objective_id)
    analysis = getattr(readiness, "route_analysis", None)
    candidates = tuple(
        candidate
        for candidate in getattr(analysis, "candidates", ())
        if (
            getattr(candidate, "reachable", False)
            and getattr(candidate, "total_cost", None) is not None
            and getattr(candidate, "detour", None) is not None
        )
    )
    if not candidates:
        return CampaignPlan(objective.objective_id)
    selected = min(candidates, key=lambda candidate: (candidate.detour, candidate.total_cost))
    location = _candidate_location(selected)
    if location is None:
        return CampaignPlan(objective.objective_id)
    return CampaignPlan(
        objective.objective_id,
        (
            RecoveryStop(
                source=selected.destination,
                destination=location,
                parent_objective_id=objective.objective_id,
                route=selected.first_route,
                reason=getattr(readiness.readiness_reason, "value", "recovery required"),
            ),
        ),
    )
