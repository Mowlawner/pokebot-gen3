"""Normalize explicit runtime evidence for source-derived gate edges.

Source parsing cannot prove that a flag controls a particular warp or walkable
tile. Collision, warp, and dynamic-object observers can prove that relationship
at runtime, however. This module accepts those explicit observations and
normalizes them into :class:`GateEdgeEffect`; it never derives an edge from a
script command or from a missing transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .gate_discovery import GateCandidate
from .gate_impact import EdgeKey, GateEdgeEffect


_CONFIDENCE_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True, slots=True)
class GateEdgeObservation:
    """One explicit observation of a candidate's effect on one graph edge."""

    candidate_id: str
    edge: EdgeKey
    source_kind: str
    observation_id: str
    blocked: bool
    confidence: str = "unknown"
    guard_state: str = "unsatisfied"
    action_kind: str | None = None
    action_frame: int | None = None
    observation_frame: int | None = None
    session_id: str | None = None
    guard_symbol: str | None = None
    guard_value: int | None = None
    guard_expression: str | None = None
    # Live object positions are supplementary physical evidence.  They are
    # retained with the edge observation so a dynamic blocker can be matched
    # to the ROM object that made the attempted edge inaccessible.
    dynamic_object_positions: tuple[tuple[int, object, object], ...] = ()


@dataclass(frozen=True, slots=True)
class GateEdgeObservationResult:
    """Normalized edge effect plus diagnostics for rejected observations."""

    effect: GateEdgeEffect
    diagnostics: tuple[str, ...] = ()


def _confidence(observations: tuple[GateEdgeObservation, ...]) -> str:
    if not observations:
        return "unknown"
    return min(observations, key=lambda item: _CONFIDENCE_ORDER.get(item.confidence, 0)).confidence


def observe_gate_edge_effect(
    candidate: GateCandidate,
    observations: Iterable[GateEdgeObservation],
) -> GateEdgeObservationResult:
    """Build an edge effect from explicit observations for ``candidate``.

    Observations for another candidate, a satisfied/unknown guard, or an
    unblocked edge are excluded. Missing evidence produces an unknown effect;
    it never proves that the route is open.
    """

    accepted: list[GateEdgeObservation] = []
    diagnostics: list[str] = []
    by_edge: dict[EdgeKey, set[bool]] = {}
    for observation in observations:
        if observation.candidate_id != candidate.candidate_id:
            diagnostics.append(f"ignored observation for {observation.candidate_id!r}")
            continue
        if observation.guard_state != "unsatisfied":
            diagnostics.append(
                f"ignored {observation.observation_id!r}: guard state is {observation.guard_state!r}"
            )
            continue
        by_edge.setdefault(observation.edge, set()).add(observation.blocked)
        if not observation.blocked:
            diagnostics.append(f"ignored {observation.observation_id!r}: edge was observed open")
            continue
        if observation.confidence not in _CONFIDENCE_ORDER:
            diagnostics.append(f"ignored {observation.observation_id!r}: invalid confidence")
            continue
        accepted.append(observation)

    conflicting_edges = {edge for edge, states in by_edge.items() if len(states) > 1}
    if conflicting_edges:
        diagnostics.append("conflicting blocked/open observations for one edge")
        accepted = [observation for observation in accepted if observation.edge not in conflicting_edges]

    unique: dict[EdgeKey, GateEdgeObservation] = {}
    for observation in accepted:
        previous = unique.get(observation.edge)
        if previous is None or _CONFIDENCE_ORDER[observation.confidence] > _CONFIDENCE_ORDER[previous.confidence]:
            unique[observation.edge] = observation
    selected = tuple(unique.values())
    evidence = tuple(
        sorted((observation.source_kind, observation.observation_id) for observation in selected)
    )
    if not selected:
        explanation = "no explicit blocked edge observation"
    else:
        explanation = "explicit runtime edge observations"
    effect = GateEdgeEffect(
        candidate_id=candidate.candidate_id,
        blocked_edges=tuple(sorted(unique, key=repr)),
        confidence=_confidence(selected),
        explanation=explanation,
        evidence=evidence,
    )
    return GateEdgeObservationResult(effect, tuple(sorted(set(diagnostics))))


__all__ = [
    "GateEdgeObservation",
    "GateEdgeObservationResult",
    "observe_gate_edge_effect",
]
