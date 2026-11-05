from __future__ import annotations

import unittest

from modules.nuzlocke.gate_discovery import GateCandidate, SourceLocation
from modules.nuzlocke.gate_impact import (
    EdgeKey,
    GateEdgeEffect,
    GateImpactCache,
    annotate_candidate_with_route_impact,
    assess_gate_route_impact,
    gate_impact_cache_key,
)
from modules.world_navigation import WorldEdge, WorldMapGraph


def _graph() -> tuple[WorldMapGraph, WorldEdge, WorldEdge, WorldEdge, WorldEdge]:
    origin = (0, 0)
    middle = (0, 1)
    alternate = (0, 2)
    target = (0, 3)
    gated = WorldEdge(origin, middle, "warp", ((0, 0),), ((0, 0),), warp_index=0)
    middle_target = WorldEdge(middle, target, "warp", ((0, 0),), ((0, 0),), warp_index=0)
    detour = WorldEdge(origin, alternate, "warp", ((0, 0),), ((0, 0),), warp_index=1)
    alternate_target = WorldEdge(alternate, target, "warp", ((0, 0),), ((0, 0),), warp_index=0)
    return WorldMapGraph((gated, middle_target, detour, alternate_target)), gated, middle_target, detour, alternate_target


def _candidate() -> GateCandidate:
    return GateCandidate(
        candidate_id="emerald:script_event:fixture",
        game="emerald",
        source_revision="fixture-1",
        kind="script_event",
        source=SourceLocation("data/maps/Fixture/scripts.inc", 4, "Fixture_Open"),
        guard_expression="goto_if_set FLAG_FIXTURE_OPEN, Fixture_Done",
        guard_symbol="FLAG_FIXTURE_OPEN",
    )


class GateImpactTests(unittest.TestCase):
    def test_alternate_route_prevents_cut_set_classification(self):
        graph, gated, _, _, _ = _graph()
        impact = assess_gate_route_impact(
            graph,
            _candidate(),
            origin=(0, 0),
            destinations=((0, 3),),
            effect=GateEdgeEffect(
                "emerald:script_event:fixture",
                (EdgeKey.from_edge(gated),),
                confidence="high",
                explanation="fixture blocker",
            ),
        )
        self.assertEqual(impact.blocked_destinations, ())
        self.assertEqual(impact.alternate_destinations, ((0, 3),))
        self.assertEqual(impact.cut_set_confidence, "medium")

    def test_removed_edge_is_a_high_confidence_cut_set(self):
        graph, gated, _, detour, alternate_target = _graph()
        del detour, alternate_target
        # The target graph is rebuilt without the alternate route for this case.
        graph = WorldMapGraph(tuple(edge for edge in graph.edges if edge.source_map != (0, 2)))
        impact = assess_gate_route_impact(
            graph,
            _candidate(),
            origin=(0, 0),
            destinations=((0, 3),),
            effect=GateEdgeEffect(
                "emerald:script_event:fixture",
                (EdgeKey.from_edge(gated),),
                confidence="high",
                explanation="fixture blocker",
            ),
        )
        self.assertEqual(impact.blocked_destinations, ((0, 3),))
        self.assertEqual(impact.cut_set_confidence, "high")

    def test_unresolved_edge_effect_stays_unknown(self):
        graph, _, _, _, _ = _graph()
        impact = assess_gate_route_impact(
            graph,
            _candidate(),
            origin=(0, 0),
            destinations=((0, 3),),
        )
        self.assertEqual(impact.cut_set_confidence, "unknown")
        self.assertIn("guard-to-edge effect is unresolved", impact.notes)

    def test_annotation_and_cache_key_are_revision_and_state_aware(self):
        graph, gated, _, _, _ = _graph()
        candidate = _candidate()
        effect = GateEdgeEffect(
            candidate.candidate_id,
            (EdgeKey.from_edge(gated),),
            confidence="high",
            evidence=(("dynamic_object", "fixture:object:1"),),
        )
        impact = assess_gate_route_impact(
            graph,
            candidate,
            origin=(0, 0),
            destinations=((0, 3),),
            effect=effect,
        )
        annotated = annotate_candidate_with_route_impact(candidate, impact)
        self.assertIsNot(annotated, candidate)
        self.assertEqual(annotated.route_evidence.alternate_routes, ("(0, 3)",))
        self.assertEqual(annotated.route_evidence.evidence, (("dynamic_object", "fixture:object:1"),))
        first_key = gate_impact_cache_key(
            candidate,
            topology_revision="topology-1",
            guard_state="unknown",
            origin=(0, 0),
            destinations=((0, 3),),
            effect=effect,
        )
        second_key = gate_impact_cache_key(
            candidate,
            topology_revision="topology-1",
            guard_state="satisfied",
            origin=(0, 0),
            destinations=((0, 3),),
            effect=effect,
        )
        self.assertNotEqual(first_key, second_key)

    def test_cache_reuses_and_invalidates_gate_state(self):
        graph, gated, _, _, _ = _graph()
        candidate = _candidate()
        effect = GateEdgeEffect(candidate.candidate_id, (EdgeKey.from_edge(gated),), confidence="high")
        cache = GateImpactCache()
        first = cache.get_or_compute(
            graph,
            candidate,
            topology_revision="topology-1",
            guard_state="unknown",
            origin=(0, 0),
            destinations=((0, 3),),
            effect=effect,
        )
        second = cache.get_or_compute(
            graph,
            candidate,
            topology_revision="topology-1",
            guard_state="unknown",
            origin=(0, 0),
            destinations=((0, 3),),
            effect=effect,
        )
        self.assertIs(first, second)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.invalidate(candidate_ids=(candidate.candidate_id,)), 1)
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
