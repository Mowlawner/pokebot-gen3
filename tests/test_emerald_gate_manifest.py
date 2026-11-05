from __future__ import annotations

import unittest

from modules.map_data import MapRSE
from modules.nuzlocke.emerald_gate_manifest import (
    EMERALD_GATE_MANIFEST,
    EMERALD_SCANNER_VERSION,
    EMERALD_SOURCE_REVISION,
    RouteImpactContext,
    compile_emerald_gate_manifest,
    validate_emerald_gate_manifest,
)
from modules.nuzlocke.gate_discovery import (
    GateCandidate,
    GateDiscoveryReport,
    SourceLocation,
)
from modules.nuzlocke.gate_impact import EdgeKey, GateEdgeEffect
from modules.world_navigation import WorldEdge, WorldMapGraph


WALLY_ID = "emerald:script_event:3b0fd3f96b9dcdd8ed31"
OLDALE_ID = "emerald:script_event:34df85740152dad2b963"
CALVIN_ID = "emerald:trainer_checkpoint:baa9b80dae45f7382c78"


def _wally_candidate(*, source_revision: str = EMERALD_SOURCE_REVISION) -> GateCandidate:
    return GateCandidate(
        candidate_id=WALLY_ID,
        game="emerald",
        source_revision=source_revision,
        kind="script_event",
        source=SourceLocation(
            "data/maps/PetalburgCity/scripts.inc",
            28,
            "PetalburgCity_OnFrame",
        ),
        guard_expression=(
            "map_script_2 VAR_PETALBURG_CITY_STATE, 2, "
            "PetalburgCity_EventScript_WallyTutorial"
        ),
        guard_symbol="VAR_PETALBURG_CITY_STATE",
        guard_value="2",
        affected_maps=("MAP_PETALBURG_CITY",),
        metadata=(
            ("command", "map_script_2"),
            ("guard_kind", "map_script_value"),
        ),
    )


def _calvin_candidate(*, source_revision: str = EMERALD_SOURCE_REVISION) -> GateCandidate:
    return GateCandidate(
        candidate_id=CALVIN_ID,
        game="emerald",
        source_revision=source_revision,
        kind="trainer_checkpoint",
        source=SourceLocation(
            "data/maps/Route102/scripts.inc",
            21,
            "Route102_EventScript_Calvin",
        ),
        affected_maps=("MAP_ROUTE102",),
        metadata=(
            ("affordance", "object_event"),
            ("command", "trainerbattle_single"),
            ("local_id", "2"),
            ("script", "Route102_EventScript_Calvin"),
            ("trainer", "TRAINER_CALVIN_1"),
            ("trainer_id", "318"),
            ("trainer_name", "CALVIN"),
            ("roster_resolution", "resolved"),
        ),
    )


def _oldale_candidate(*, source_revision: str = EMERALD_SOURCE_REVISION) -> GateCandidate:
    return GateCandidate(
        candidate_id=OLDALE_ID,
        game="emerald",
        source_revision=source_revision,
        kind="script_event",
        source=SourceLocation(
            "data/maps/OldaleTown/scripts.inc",
            8,
            "OldaleTown_OnTransition",
        ),
        guard_expression=(
            "call_if_unset FLAG_ADVENTURE_STARTED, "
            "OldaleTown_EventScript_BlockWestEntrance"
        ),
        guard_symbol="FLAG_ADVENTURE_STARTED",
        guard_value="OldaleTown_EventScript_BlockWestEntrance",
        affected_maps=("MAP_OLDALE_TOWN",),
        metadata=(
            ("command", "call_if_unset"),
            ("guard_kind", "flag_unset"),
        ),
    )


def _report(*, source_revision: str = EMERALD_SOURCE_REVISION) -> GateDiscoveryReport:
    return GateDiscoveryReport(
        schema_version=1,
        scanner_version=EMERALD_SCANNER_VERSION,
        game="emerald",
        source_root="/fixture/pokeemerald",
        source_revision=source_revision,
        candidates=(
            _wally_candidate(source_revision=source_revision),
            _oldale_candidate(source_revision=source_revision),
            _calvin_candidate(source_revision=source_revision),
        ),
    )


def _route_context(candidate_id: str) -> tuple[GateEdgeEffect, RouteImpactContext]:
    edge = WorldEdge((0, 0), (0, 1), "warp", (), (), warp_index=0)
    graph = WorldMapGraph((edge,))
    effect = GateEdgeEffect(
        candidate_id,
        (EdgeKey.from_edge(edge),),
        confidence="high",
        explanation="fixture gate edge",
        evidence=(("fixture", candidate_id),),
    )
    return effect, RouteImpactContext(graph, (0, 0), ((0, 1),))


class EmeraldGateManifestTests(unittest.TestCase):
    def test_reviewed_manifest_matches_pinned_candidates(self):
        validation = validate_emerald_gate_manifest(_report(), EMERALD_GATE_MANIFEST)

        self.assertEqual(validation.diagnostics, ())
        self.assertEqual(
            tuple(item.candidate_id for item in validation.access_overrides),
            (OLDALE_ID, WALLY_ID),
        )
        self.assertEqual(
            tuple(item.candidate_id for item in validation.trainer_overrides),
            (CALVIN_ID,),
        )
        self.assertEqual(
            validation.access_overrides[0].description,
            "Oldale's westward traversal into Route 102 requires the Pokédex handoff that sets FLAG_ADVENTURE_STARTED",
        )

    def test_stale_source_revision_is_rejected(self):
        validation = validate_emerald_gate_manifest(
            _report(source_revision="stale-revision"),
            EMERALD_GATE_MANIFEST,
        )

        self.assertEqual(validation.access_overrides, ())
        self.assertEqual(validation.trainer_overrides, ())
        self.assertIn(
            "manifest_source_revision_mismatch",
            {diagnostic.code for diagnostic in validation.diagnostics},
        )

    def test_missing_route_evidence_stays_diagnostic_only(self):
        result = compile_emerald_gate_manifest(
            _report(),
            manifest=EMERALD_GATE_MANIFEST,
        )

        self.assertEqual(result.access_requirements, ())
        self.assertEqual(result.trainer_checkpoints, ())
        self.assertEqual(
            {diagnostic.code for diagnostic in result.diagnostics},
            {"route_impact_unverified"},
        )

    def test_explicit_edge_route_context_promotes_reviewed_entries(self):
        wally_effect, wally_context = _route_context(WALLY_ID)
        oldale_effect, oldale_context = _route_context(OLDALE_ID)
        calvin_effect, calvin_context = _route_context(CALVIN_ID)

        result = compile_emerald_gate_manifest(
            _report(),
            manifest=EMERALD_GATE_MANIFEST,
            edge_effects=(wally_effect, oldale_effect, calvin_effect),
            route_contexts={
                WALLY_ID: wally_context,
                OLDALE_ID: oldale_context,
                CALVIN_ID: calvin_context,
            },
        )

        self.assertEqual(result.diagnostics, ())
        self.assertEqual(
            tuple(item.candidate_id for item in result.access_requirements),
            (OLDALE_ID, WALLY_ID),
        )
        self.assertEqual(
            tuple(item.candidate_id for item in result.trainer_checkpoints),
            (CALVIN_ID,),
        )
        self.assertEqual(
            next(item for item in result.access_requirements if item.candidate_id == WALLY_ID).map_id,
            MapRSE.ROUTE104.value,
        )
        self.assertEqual(
            next(item for item in result.access_requirements if item.candidate_id == OLDALE_ID).map_id,
            MapRSE.ROUTE102.value,
        )


if __name__ == "__main__":
    unittest.main()
