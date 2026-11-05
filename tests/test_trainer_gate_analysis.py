from __future__ import annotations

import unittest

from modules.nuzlocke.gate_discovery import GateCandidate, RouteEvidence, SourceLocation
from modules.nuzlocke.trainer_gate_analysis import classify_trainer_candidate


def _candidate(route_evidence: RouteEvidence = RouteEvidence()) -> GateCandidate:
    return GateCandidate(
        candidate_id="emerald:trainer_checkpoint:fixture",
        game="emerald",
        source_revision="fixture-1",
        kind="trainer_checkpoint",
        source=SourceLocation("data/maps/Route102/scripts.inc", 20, "Route102_EventScript_Calvin"),
        route_evidence=route_evidence,
    )


class TrainerGateAnalysisTests(unittest.TestCase):
    def test_high_confidence_cut_set_is_mandatory(self):
        assessment = classify_trainer_candidate(
            _candidate(RouteEvidence(blocked_routes=("Route102->Petalburg",), cut_set_confidence="high")),
            trainer_id="trainer:(0, 17):2",
        )

        self.assertEqual(assessment.classification, "mandatory")
        self.assertEqual(assessment.confidence, "high")

    def test_route_risk_distinguishes_costly_and_avoidable_trainers(self):
        costly = classify_trainer_candidate(
            _candidate(),
            avoidability="costly",
            route_confidence="high",
            route_reason="avoidance requires a detour",
        )
        avoidable = classify_trainer_candidate(
            _candidate(),
            avoidability="avoidable",
            route_confidence="high",
        )

        self.assertEqual(costly.classification, "costly_to_avoid")
        self.assertEqual(avoidable.classification, "avoidable")

    def test_unknown_geometry_stays_unknown(self):
        assessment = classify_trainer_candidate(
            _candidate(),
            avoidability="unknown",
            route_confidence="unknown",
        )

        self.assertEqual(assessment.classification, "unknown")
        self.assertEqual(assessment.confidence, "unknown")

    def test_alternate_route_prevents_static_mandatory_promotion(self):
        assessment = classify_trainer_candidate(
            _candidate(
                RouteEvidence(
                    blocked_routes=("Route102->Petalburg",),
                    alternate_routes=("detour",),
                    cut_set_confidence="high",
                )
            ),
            avoidability="avoidable",
            route_confidence="high",
        )

        self.assertEqual(assessment.classification, "avoidable")


if __name__ == "__main__":
    unittest.main()
