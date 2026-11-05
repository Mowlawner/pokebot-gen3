from __future__ import annotations

import unittest

from modules.nuzlocke.gate_discovery import GateCandidate, SourceLocation
from modules.nuzlocke.gate_impact import EdgeKey
from modules.nuzlocke.gate_runtime_evidence import (
    GateEdgeObservation,
    observe_gate_edge_effect,
)


def _candidate() -> GateCandidate:
    return GateCandidate(
        candidate_id="emerald:object_visibility:fixture",
        game="emerald",
        source_revision="fixture-1",
        kind="object_visibility",
        source=SourceLocation("fixture/map.json", 8),
    )


def _edge(index: int) -> EdgeKey:
    return EdgeKey("MAP_START", f"MAP_TARGET_{index}", "warp", index)


class GateRuntimeEvidenceTests(unittest.TestCase):
    def test_combines_explicit_warp_and_dynamic_object_evidence(self):
        candidate = _candidate()
        result = observe_gate_edge_effect(
            candidate,
            (
                GateEdgeObservation(
                    candidate.candidate_id,
                    _edge(0),
                    "warp",
                    "transition:0",
                    True,
                    confidence="high",
                ),
                GateEdgeObservation(
                    candidate.candidate_id,
                    _edge(1),
                    "dynamic_object",
                    "object:7",
                    True,
                    confidence="high",
                ),
            ),
        )

        self.assertEqual(result.effect.confidence, "high")
        self.assertEqual(result.effect.blocked_edges, (_edge(0), _edge(1)))
        self.assertEqual(
            result.effect.evidence,
            (("dynamic_object", "object:7"), ("warp", "transition:0")),
        )
        self.assertEqual(result.diagnostics, ())

    def test_missing_or_uncertain_runtime_evidence_stays_unknown(self):
        candidate = _candidate()
        result = observe_gate_edge_effect(
            candidate,
            (
                GateEdgeObservation(
                    candidate.candidate_id,
                    _edge(0),
                    "collision",
                    "collision:unknown",
                    True,
                    confidence="unknown",
                    guard_state="unknown",
                ),
                GateEdgeObservation(
                    candidate.candidate_id,
                    _edge(1),
                    "warp",
                    "transition:open",
                    False,
                    confidence="high",
                ),
            ),
        )

        self.assertEqual(result.effect.blocked_edges, ())
        self.assertEqual(result.effect.confidence, "unknown")
        self.assertTrue(any("guard state is 'unknown'" in item for item in result.diagnostics))
        self.assertTrue(any("edge was observed open" in item for item in result.diagnostics))

    def test_mismatched_and_conflicting_observations_do_not_promote_an_edge(self):
        candidate = _candidate()
        result = observe_gate_edge_effect(
            candidate,
            (
                GateEdgeObservation(
                    "emerald:object_visibility:other",
                    _edge(0),
                    "warp",
                    "other:0",
                    True,
                    confidence="high",
                ),
                GateEdgeObservation(
                    candidate.candidate_id,
                    _edge(0),
                    "collision",
                    "collision:blocked",
                    True,
                    confidence="high",
                ),
                GateEdgeObservation(
                    candidate.candidate_id,
                    _edge(0),
                    "warp",
                    "transition:open",
                    False,
                    confidence="high",
                ),
            ),
        )

        self.assertEqual(result.effect.blocked_edges, ())
        self.assertIn("conflicting blocked/open observations for one edge", result.diagnostics)
        self.assertTrue(any("other" in diagnostic for diagnostic in result.diagnostics))


if __name__ == "__main__":
    unittest.main()
