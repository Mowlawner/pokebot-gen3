"""Pure route-impact analysis for source-derived gate candidates.

The source scanner can identify a guarded object or script without proving
which physical map edge it controls.  This module keeps that distinction
explicit: callers provide the edge effect when collision, warp, or live object
analysis has established it, and this layer measures the resulting route
impact.  A missing edge effect therefore produces ``unknown`` evidence rather
than an optimistic route.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Iterable

from modules.world_navigation import WorldEdge, WorldMapGraph, WorldNavigationError, WorldRoute, route_with_edge_filter

from .gate_discovery import GateCandidate, RouteEvidence


@dataclass(frozen=True, slots=True)
class EdgeKey:
    """Stable identity for one physical map edge."""

    source_map: object
    destination_map: object
    kind: str
    warp_index: int | None = None

    @classmethod
    def from_edge(cls, edge: WorldEdge) -> "EdgeKey":
        return cls(edge.source_map, edge.destination_map, edge.kind, edge.warp_index)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_map": self.source_map,
            "destination_map": self.destination_map,
            "kind": self.kind,
            "warp_index": self.warp_index,
        }


@dataclass(frozen=True, slots=True)
class GateEdgeEffect:
    """The topology effect of one candidate while its guard is unsatisfied."""

    candidate_id: str
    blocked_edges: tuple[EdgeKey, ...] = ()
    confidence: str = "unknown"
    explanation: str = "edge effect not established"
    # The runtime/collision observer that established the edge relationship
    # can retain its stable reference for later promotion diagnostics.
    evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class GateRouteImpact:
    """Comparison of routes with and without a candidate's guard satisfied."""

    candidate_id: str
    origin: object
    destinations: tuple[object, ...]
    blocked_destinations: tuple[object, ...] = ()
    alternate_destinations: tuple[object, ...] = ()
    newly_reachable_destinations: tuple[object, ...] = ()
    satisfied_routes: tuple[tuple[object, WorldRoute], ...] = ()
    unsatisfied_routes: tuple[tuple[object, WorldRoute], ...] = ()
    cut_set_confidence: str = "unknown"
    notes: tuple[str, ...] = ()
    edge_evidence: tuple[tuple[str, str], ...] = ()

    @property
    def changes_reachability(self) -> bool:
        return bool(self.blocked_destinations or self.newly_reachable_destinations)

    def to_dict(self) -> dict[str, object]:
        """Return stable diagnostic data without serializing map edge details twice."""

        return {
            "candidate_id": self.candidate_id,
            "origin": self.origin,
            "destinations": list(self.destinations),
            "blocked_destinations": list(self.blocked_destinations),
            "alternate_destinations": list(self.alternate_destinations),
            "newly_reachable_destinations": list(self.newly_reachable_destinations),
            "cut_set_confidence": self.cut_set_confidence,
            "notes": list(self.notes),
            "edge_evidence": [list(item) for item in self.edge_evidence],
        }


class GateImpactCache:
    """Small pure cache with explicit invalidation boundaries."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, str, str, GateRouteImpact]] = {}

    def get_or_compute(
        self,
        graph: WorldMapGraph,
        candidate: GateCandidate,
        *,
        topology_revision: str,
        guard_state: str,
        origin: object,
        destinations: Iterable[object],
        effect: GateEdgeEffect | None = None,
    ) -> GateRouteImpact:
        ordered_destinations = tuple(destinations)
        key = gate_impact_cache_key(
            candidate,
            topology_revision=topology_revision,
            guard_state=guard_state,
            origin=origin,
            destinations=ordered_destinations,
            effect=effect,
        )
        cached = self._entries.get(key)
        if cached is not None:
            return cached[3]
        impact = assess_gate_route_impact(
            graph,
            candidate,
            origin=origin,
            destinations=ordered_destinations,
            effect=effect,
        )
        self._entries[key] = (candidate.candidate_id, topology_revision, candidate.source_revision, impact)
        return impact

    def invalidate(
        self,
        *,
        candidate_ids: Iterable[str] = (),
        topology_revision: str | None = None,
        source_revision: str | None = None,
    ) -> int:
        """Drop entries affected by a cleared gate or changed topology."""

        selected = set(candidate_ids)
        if not selected and topology_revision is None and source_revision is None:
            count = len(self._entries)
            self._entries.clear()
            return count
        stale = [
            key
            for key, (candidate_id, entry_topology, entry_source, _) in self._entries.items()
            if (selected and candidate_id in selected)
            or (topology_revision is not None and entry_topology == topology_revision)
            or (source_revision is not None and entry_source == source_revision)
        ]
        for key in stale:
            del self._entries[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._entries)


def _edge_allowed(blocked_edges: frozenset[EdgeKey]):
    return lambda edge: EdgeKey.from_edge(edge) not in blocked_edges


def _route(
    graph: WorldMapGraph,
    origin: object,
    destination: object,
    blocked_edges: frozenset[EdgeKey],
) -> WorldRoute | None:
    try:
        if not blocked_edges:
            return graph.route(origin, destination)
        return route_with_edge_filter(graph, origin, destination, _edge_allowed(blocked_edges))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, WorldNavigationError):
        return None


def _count_routes(
    graph: WorldMapGraph,
    origin: object,
    destination: object,
    blocked_edges: frozenset[EdgeKey],
    *,
    limit: int = 2,
    max_depth: int = 32,
) -> int:
    """Count simple routes up to ``limit`` without exploring graph cycles."""

    if origin == destination:
        return 1
    count = 0

    def visit(current: object, visited: frozenset[object], depth: int) -> None:
        nonlocal count
        if count >= limit or depth > max_depth:
            return
        for edge in graph.outgoing(current):
            if EdgeKey.from_edge(edge) in blocked_edges or edge.destination_map in visited:
                continue
            if edge.destination_map == destination:
                count += 1
                if count >= limit:
                    return
                continue
            visit(edge.destination_map, visited | {edge.destination_map}, depth + 1)
            if count >= limit:
                return

    visit(origin, frozenset({origin}), 0)
    return count


def assess_gate_route_impact(
    graph: WorldMapGraph,
    candidate: GateCandidate,
    *,
    origin: object,
    destinations: Iterable[object],
    effect: GateEdgeEffect | None = None,
) -> GateRouteImpact:
    """Measure the candidate's route impact for a fixed origin and targets.

    ``effect.blocked_edges`` describes edges unavailable while the guard is
    unsatisfied.  If ``effect`` is missing or has no edge identities, the
    result records that route impact is unknown.  A route that remains
    available in the unsatisfied graph is an alternate route and the
    candidate cannot be promoted as a cut set for that destination.
    """

    ordered_destinations = tuple(sorted(set(destinations), key=repr))
    blocked_edges = frozenset(effect.blocked_edges if effect else ())
    satisfied_routes: list[tuple[object, WorldRoute]] = []
    unsatisfied_routes: list[tuple[object, WorldRoute]] = []
    blocked: list[object] = []
    alternates: list[object] = []
    newly_reachable: list[object] = []
    notes: list[str] = []

    if effect is not None and effect.candidate_id != candidate.candidate_id:
        notes.append("edge effect belongs to a different candidate")
        blocked_edges = frozenset()
        effect = None
    if effect is None or not effect.blocked_edges:
        notes.append("guard-to-edge effect is unresolved")
    elif effect.confidence == "unknown":
        notes.append(effect.explanation)

    for destination in ordered_destinations:
        satisfied_route = _route(graph, origin, destination, frozenset())
        unsatisfied_route = _route(graph, origin, destination, blocked_edges)
        if satisfied_route is not None:
            satisfied_routes.append((destination, satisfied_route))
        if unsatisfied_route is not None:
            unsatisfied_routes.append((destination, unsatisfied_route))
        if satisfied_route is not None and unsatisfied_route is None:
            blocked.append(destination)
        elif satisfied_route is None and unsatisfied_route is not None:
            newly_reachable.append(destination)
        elif unsatisfied_route is not None and origin != destination:
            if _count_routes(graph, origin, destination, blocked_edges) > 0:
                alternates.append(destination)

    if not blocked_edges:
        confidence = "unknown"
    elif blocked:
        confidence = "high" if effect is not None and effect.confidence == "high" else "medium"
    elif alternates:
        confidence = "medium"
    else:
        confidence = "low"
        notes.append("candidate did not remove a route to the supplied destinations")
    if effect is not None and effect.explanation and effect.explanation not in notes:
        notes.append(effect.explanation)

    return GateRouteImpact(
        candidate_id=candidate.candidate_id,
        origin=origin,
        destinations=ordered_destinations,
        blocked_destinations=tuple(blocked),
        alternate_destinations=tuple(alternates),
        newly_reachable_destinations=tuple(newly_reachable),
        satisfied_routes=tuple(satisfied_routes),
        unsatisfied_routes=tuple(unsatisfied_routes),
        cut_set_confidence=confidence,
        notes=tuple(sorted(set(notes))),
        edge_evidence=tuple(sorted(effect.evidence if effect is not None else ())),
    )


def annotate_candidate_with_route_impact(
    candidate: GateCandidate,
    impact: GateRouteImpact,
) -> GateCandidate:
    """Return a new candidate containing route evidence from an assessment."""

    blocked = tuple(sorted((repr(destination) for destination in impact.blocked_destinations)))
    alternates = tuple(sorted((repr(destination) for destination in impact.alternate_destinations)))
    existing_notes = set(candidate.route_evidence.notes)
    existing_notes.update(impact.notes)
    existing_evidence = set(candidate.route_evidence.evidence)
    existing_evidence.update(impact.edge_evidence)
    evidence = RouteEvidence(
        blocked_routes=blocked,
        alternate_routes=alternates,
        cut_set_confidence=impact.cut_set_confidence,
        notes=tuple(sorted(existing_notes)),
        evidence=tuple(sorted(existing_evidence)),
    )
    return replace(candidate, route_evidence=evidence)


def gate_impact_cache_key(
    candidate: GateCandidate,
    *,
    topology_revision: str,
    guard_state: str,
    origin: object,
    destinations: Iterable[object],
    effect: GateEdgeEffect | None = None,
) -> str:
    """Build a stable cache key for topology, candidate, and observed state."""

    payload = {
        "candidate_id": candidate.candidate_id,
        "source_revision": candidate.source_revision,
        "topology_revision": topology_revision,
        "guard_state": guard_state,
        "origin": origin,
        "destinations": sorted(set(destinations), key=repr),
        "effect": {
            "candidate_id": effect.candidate_id,
            "blocked_edges": [edge.to_dict() for edge in effect.blocked_edges],
            "confidence": effect.confidence,
            "evidence": effect.evidence,
        }
        if effect is not None
        else None,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "EdgeKey",
    "GateEdgeEffect",
    "GateImpactCache",
    "GateRouteImpact",
    "annotate_candidate_with_route_impact",
    "assess_gate_route_impact",
    "gate_impact_cache_key",
]
