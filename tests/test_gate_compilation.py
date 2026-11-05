from __future__ import annotations

import unittest

from modules.nuzlocke.campaign_access import CampaignAccessRequirement
from modules.nuzlocke.gate_compilation import (
    AccessRequirementOverride,
    TrainerCheckpointOverride,
    compile_gate_candidates,
)
from modules.nuzlocke.gate_discovery import (
    GateCandidate,
    GateDiscoveryReport,
    RouteEvidence,
    SourceLocation,
)
from modules.nuzlocke.gate_impact import (
    EdgeKey,
    GateEdgeEffect,
    annotate_candidate_with_route_impact,
    assess_gate_route_impact,
)
from modules.trainer_checkpoints import TrainerPokemon
from modules.world_navigation import WorldEdge, WorldMapGraph


def _candidate(candidate_id: str, kind: str, *, metadata=(), route=True) -> GateCandidate:
    return GateCandidate(
        candidate_id=candidate_id,
        game="emerald",
        source_revision="fixture-1",
        kind=kind,
        source=SourceLocation("data/maps/Fixture/scripts.inc", 4, "Fixture_Event"),
        guard_expression="goto_if_set FLAG_FIXTURE_OPEN, Done",
        guard_symbol="FLAG_FIXTURE_OPEN",
        affected_maps=("MAP_FIXTURE",),
        route_evidence=(
            RouteEvidence(blocked_routes=("(0, 1)",), cut_set_confidence="high")
            if route
            else RouteEvidence()
        ),
        metadata=tuple(metadata),
    )


def _report(*candidates: GateCandidate) -> GateDiscoveryReport:
    return GateDiscoveryReport(
        schema_version=1,
        scanner_version="fixture",
        game="emerald",
        source_root="/fixture",
        source_revision="fixture-1",
        candidates=tuple(candidates),
    )


class GateCompilationTests(unittest.TestCase):
    def test_access_provenance_survives_compilation(self):
        candidate = _candidate("emerald:script_event:access", "script_event")
        result = compile_gate_candidates(
            _report(candidate),
            access_overrides=(
                AccessRequirementOverride(
                    candidate.candidate_id,
                    (1, 2),
                    required_facts=("fixture_complete",),
                ),
            ),
        )

        self.assertEqual(result.diagnostics, ())
        requirement = result.access_requirements[0]
        self.assertEqual(requirement, CampaignAccessRequirement(
            requirement_id=f"source:{candidate.candidate_id}",
            map_id=(1, 2),
            required_facts=("fixture_complete",),
            description="compiled campaign access requirement",
            candidate_id=candidate.candidate_id,
            provenance=(
                ("gate_candidate", candidate.candidate_id),
                ("source", "data/maps/Fixture/scripts.inc:4#Fixture_Event"),
                ("source_revision", "fixture-1"),
                ("guard", "goto_if_set FLAG_FIXTURE_OPEN, Done"),
                ("producer_actions", ""),
                ("route_confidence", "high"),
                ("blocked_routes", "(0, 1)"),
                ("alternate_routes", ""),
                ("route_evidence", ""),
                ("route_notes", ""),
            ),
            confidence="high",
            verification_status="verified",
        ))
        self.assertEqual(result.promoted_candidate_ids, (candidate.candidate_id,))

    def test_route_impact_is_required_before_access_promotion(self):
        candidate = _candidate("emerald:script_event:corridor", "script_event", route=False)
        gated_edge = WorldEdge((0, 0), (0, 1), "warp", (), (), warp_index=0)
        graph = WorldMapGraph((gated_edge,))
        impact = assess_gate_route_impact(
            graph,
            candidate,
            origin=(0, 0),
            destinations=((0, 1),),
            effect=GateEdgeEffect(
                candidate.candidate_id,
                (EdgeKey.from_edge(gated_edge),),
                confidence="high",
                explanation="fixture corridor transition",
            ),
        )
        annotated = annotate_candidate_with_route_impact(candidate, impact)
        result = compile_gate_candidates(
            _report(annotated),
            access_overrides=(
                AccessRequirementOverride(
                    annotated.candidate_id,
                    (1, 2),
                    required_facts=("fixture_complete",),
                ),
            ),
        )

        self.assertEqual(result.diagnostics, ())
        self.assertEqual(result.access_requirements[0].candidate_id, candidate.candidate_id)

    def test_unresolved_route_candidate_remains_diagnostic_only(self):
        candidate = _candidate("emerald:script_event:unknown", "script_event", route=False)
        result = compile_gate_candidates(
            _report(candidate),
            access_overrides=(
                AccessRequirementOverride(
                    candidate.candidate_id,
                    (1, 2),
                    required_facts=("fixture_complete",),
                ),
            ),
        )

        self.assertEqual(result.access_requirements, ())
        self.assertEqual(result.diagnostics[0].code, "route_impact_unverified")

    def test_trainer_requires_explicit_classification_clearance_and_roster(self):
        candidate = _candidate(
            "emerald:trainer_checkpoint:calvin",
            "trainer_checkpoint",
            metadata=(
                ("roster_resolution", "resolved"),
            ),
        )
        result = compile_gate_candidates(
            _report(candidate),
            trainer_overrides=(
                TrainerCheckpointOverride(
                    candidate_id=candidate.candidate_id,
                    checkpoint_id="fixture:calvin",
                    map_id=(1, 2),
                    trainer_affordance_id="trainer:(1, 2):2",
                    display_name="Calvin",
                    party=(TrainerPokemon("Poochyena", 5, ("Tackle",)),),
                    mandatory=True,
                    clearance_fact="calvin_cleared",
                ),
            ),
        )

        self.assertEqual(result.diagnostics, ())
        checkpoint = result.trainer_checkpoints[0]
        self.assertEqual(checkpoint.candidate_id, candidate.candidate_id)
        self.assertEqual(checkpoint.source_records[0], ("gate_candidate", candidate.candidate_id))
        self.assertEqual(checkpoint.clearance_fact, "calvin_cleared")
        self.assertTrue(checkpoint.mandatory)

    def test_trainer_without_explicit_mandatory_classification_is_rejected(self):
        candidate = _candidate("emerald:trainer_checkpoint:unknown", "trainer_checkpoint")
        result = compile_gate_candidates(
            _report(candidate),
            trainer_overrides=(
                TrainerCheckpointOverride(
                    candidate_id=candidate.candidate_id,
                    checkpoint_id="fixture:unknown",
                    map_id=(1, 2),
                    trainer_affordance_id="trainer:(1, 2):2",
                    display_name="Unknown",
                    party=(TrainerPokemon("Poochyena", 5, ("Tackle",)),),
                ),
            ),
        )

        self.assertEqual(result.trainer_checkpoints, ())
        self.assertEqual(result.diagnostics[0].code, "mandatory_classification_required")

    def test_unresolved_trainer_roster_is_not_promoted(self):
        candidate = _candidate(
            "emerald:trainer_checkpoint:unresolved",
            "trainer_checkpoint",
            metadata=(("roster_resolution", "unresolved"),),
        )
        result = compile_gate_candidates(
            _report(candidate),
            trainer_overrides=(
                TrainerCheckpointOverride(
                    candidate_id=candidate.candidate_id,
                    checkpoint_id="fixture:unresolved",
                    map_id=(1, 2),
                    trainer_affordance_id="trainer:(1, 2):2",
                    display_name="Unresolved",
                    party=(TrainerPokemon("Poochyena", 5, ("Tackle",)),),
                    mandatory=False,
                ),
            ),
        )

        self.assertEqual(result.trainer_checkpoints, ())
        self.assertEqual(result.diagnostics[0].code, "trainer_roster_unresolved")


if __name__ == "__main__":
    unittest.main()
