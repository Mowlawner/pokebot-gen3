"""Reviewed source-gate manifest for the pinned Pokémon Emerald source.

The source scanner is deliberately broader than the campaign registry.  This
module records the small set of candidates that have received a semantic
review for the current Emerald slice, including the exact source fingerprint
that was reviewed.  It does not make route impact implicit: every promotion
still requires a high-confidence route annotation supplied by route analysis
or an explicit edge observation evaluated against a graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable, Mapping

from modules.map_data import MapRSE
from modules.world_navigation import WorldMapGraph

from .gate_compilation import (
    AccessRequirementOverride,
    GateCompilationDiagnostic,
    GateCompilationResult,
    TrainerCheckpointOverride,
    compile_gate_candidates,
)
from .gate_discovery import (
    GateCandidate,
    GateDiscoveryReport,
    SourceLocation,
)
from .gate_impact import (
    GateEdgeEffect,
    GateRouteImpact,
    annotate_candidate_with_route_impact,
    assess_gate_route_impact,
)
from modules.trainer_checkpoints import TrainerPokemon


EMERALD_GAME_ID = "emerald"
EMERALD_SOURCE_REVISION = "5eff78649e7170a877b961ef0b3da13b81a16038"
EMERALD_SCANNER_VERSION = "emerald-source-1"
EMERALD_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReviewedCandidateExpectation:
    """Source identity that must match before a reviewed override is used."""

    candidate_id: str
    kind: str
    source: SourceLocation
    guard_expression: str | None = None
    guard_symbol: str | None = None
    guard_value: str | None = None
    affected_maps: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class EmeraldGateManifest:
    """Pinned source revision and reviewed semantic overrides for Emerald."""

    manifest_version: int
    game: str
    source_revision: str
    scanner_version: str
    expectations: tuple[ReviewedCandidateExpectation, ...] = ()
    access_overrides: tuple[AccessRequirementOverride, ...] = ()
    trainer_overrides: tuple[TrainerCheckpointOverride, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteImpactContext:
    """Graph inputs needed to turn an explicit edge effect into route evidence."""

    graph: WorldMapGraph
    origin: object
    destinations: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ManifestValidationResult:
    """Validated overrides and diagnostics from one manifest/report comparison."""

    access_overrides: tuple[AccessRequirementOverride, ...] = ()
    trainer_overrides: tuple[TrainerCheckpointOverride, ...] = ()
    diagnostics: tuple[GateCompilationDiagnostic, ...] = ()


def _json_value(value):
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _diagnostic(candidate_id: str | None, code: str, message: str) -> GateCompilationDiagnostic:
    return GateCompilationDiagnostic(candidate_id, code, message)


def _candidate_metadata(candidate: GateCandidate) -> dict[str, str]:
    return dict(candidate.metadata)


def _expectation_diagnostics(
    candidate: GateCandidate,
    expectation: ReviewedCandidateExpectation,
) -> list[GateCompilationDiagnostic]:
    diagnostics: list[GateCompilationDiagnostic] = []
    if candidate.kind != expectation.kind:
        diagnostics.append(
            _diagnostic(
                candidate.candidate_id,
                "manifest_kind_mismatch",
                f"reviewed candidate kind is {expectation.kind!r}, report has {candidate.kind!r}",
            )
        )
    if candidate.source != expectation.source:
        diagnostics.append(
            _diagnostic(
                candidate.candidate_id,
                "manifest_source_mismatch",
                "reviewed source location does not match the discovery report",
            )
        )
    for field_name in ("guard_expression", "guard_symbol", "guard_value"):
        expected = getattr(expectation, field_name)
        observed = getattr(candidate, field_name)
        if observed != expected:
            diagnostics.append(
                _diagnostic(
                    candidate.candidate_id,
                    "manifest_guard_mismatch",
                    f"reviewed {field_name} does not match the discovery report",
                )
            )
    if tuple(candidate.affected_maps) != tuple(expectation.affected_maps):
        diagnostics.append(
            _diagnostic(
                candidate.candidate_id,
                "manifest_affected_maps_mismatch",
                "reviewed affected maps do not match the discovery report",
            )
        )
    metadata = _candidate_metadata(candidate)
    for key, expected in expectation.metadata:
        if metadata.get(key) != expected:
            diagnostics.append(
                _diagnostic(
                    candidate.candidate_id,
                    "manifest_metadata_mismatch",
                    f"reviewed metadata {key!r} does not match the discovery report",
                )
            )
    return diagnostics


def validate_emerald_gate_manifest(
    report: GateDiscoveryReport,
    manifest: EmeraldGateManifest,
) -> ManifestValidationResult:
    """Validate reviewed entries against one immutable discovery report."""

    diagnostics: list[GateCompilationDiagnostic] = []
    if manifest.manifest_version != EMERALD_MANIFEST_VERSION:
        diagnostics.append(
            _diagnostic(
                None,
                "manifest_version_mismatch",
                f"unsupported reviewed manifest version {manifest.manifest_version!r}",
            )
        )
    if report.game != manifest.game:
        diagnostics.append(
            _diagnostic(
                None,
                "manifest_game_mismatch",
                f"manifest targets {manifest.game!r}, report is for {report.game!r}",
            )
        )
    if report.source_revision != manifest.source_revision:
        diagnostics.append(
            _diagnostic(
                None,
                "manifest_source_revision_mismatch",
                "discovery report revision does not match the reviewed manifest",
            )
        )
    if report.scanner_version != manifest.scanner_version:
        diagnostics.append(
            _diagnostic(
                None,
                "manifest_scanner_version_mismatch",
                "discovery scanner version does not match the reviewed manifest",
            )
        )

    candidates = {candidate.candidate_id: candidate for candidate in report.candidates}
    expectations: dict[str, ReviewedCandidateExpectation] = {}
    for expectation in manifest.expectations:
        if expectation.candidate_id in expectations:
            diagnostics.append(
                _diagnostic(
                    expectation.candidate_id,
                    "manifest_duplicate_expectation",
                    "manifest contains duplicate reviewed candidate expectations",
                )
            )
            continue
        expectations[expectation.candidate_id] = expectation

    overrides: list[tuple[str, object]] = []
    overrides.extend(("access", override) for override in manifest.access_overrides)
    overrides.extend(("trainer", override) for override in manifest.trainer_overrides)
    access_overrides: list[AccessRequirementOverride] = []
    trainer_overrides: list[TrainerCheckpointOverride] = []
    seen_overrides: set[str] = set()
    for override_kind, override in overrides:
        candidate_id = override.candidate_id
        if candidate_id in seen_overrides:
            diagnostics.append(
                _diagnostic(
                    candidate_id,
                    "manifest_duplicate_override",
                    "manifest promotes one candidate more than once",
                )
            )
            continue
        seen_overrides.add(candidate_id)
        expectation = expectations.get(candidate_id)
        candidate = candidates.get(candidate_id)
        if expectation is None:
            diagnostics.append(
                _diagnostic(
                    candidate_id,
                    "manifest_expectation_missing",
                    "manifest override has no reviewed source expectation",
                )
            )
            continue
        if candidate is None:
            diagnostics.append(
                _diagnostic(
                    candidate_id,
                    "manifest_candidate_not_found",
                    "reviewed candidate is absent from the discovery report",
                )
            )
            continue
        candidate_diagnostics = _expectation_diagnostics(candidate, expectation)
        if candidate.source_revision != manifest.source_revision:
            candidate_diagnostics.append(
                _diagnostic(
                    candidate_id,
                    "manifest_candidate_revision_mismatch",
                    "candidate source revision does not match the reviewed manifest",
                )
            )
        if candidate_diagnostics:
            diagnostics.extend(candidate_diagnostics)
            continue
        if override_kind == "access":
            access_overrides.append(override)
        else:
            trainer_overrides.append(override)

    valid_override_ids = seen_overrides
    for expectation_id in expectations:
        if expectation_id not in valid_override_ids:
            diagnostics.append(
                _diagnostic(
                    expectation_id,
                    "manifest_expectation_unused",
                    "manifest reviews a candidate without a semantic override",
                )
            )

    diagnostics.sort(key=lambda item: (item.candidate_id or "", item.code, item.message))
    return ManifestValidationResult(
        access_overrides=tuple(sorted(access_overrides, key=lambda item: item.candidate_id)),
        trainer_overrides=tuple(sorted(trainer_overrides, key=lambda item: item.candidate_id)),
        diagnostics=tuple(diagnostics),
    )


def _annotate_report(
    report: GateDiscoveryReport,
    *,
    route_impacts: Iterable[GateRouteImpact],
    edge_effects: Iterable[GateEdgeEffect],
    route_contexts: Mapping[str, RouteImpactContext],
) -> tuple[GateDiscoveryReport, tuple[GateCompilationDiagnostic, ...]]:
    candidates = {candidate.candidate_id: candidate for candidate in report.candidates}
    annotations: dict[str, GateRouteImpact] = {}
    diagnostics: list[GateCompilationDiagnostic] = []

    for impact in route_impacts:
        if impact.candidate_id not in candidates:
            diagnostics.append(
                _diagnostic(
                    impact.candidate_id,
                    "route_impact_candidate_not_found",
                    "route impact references no candidate in the discovery report",
                )
            )
        elif impact.candidate_id in annotations:
            diagnostics.append(
                _diagnostic(
                    impact.candidate_id,
                    "duplicate_route_impact",
                    "more than one route impact was supplied for one candidate",
                )
            )
        else:
            annotations[impact.candidate_id] = impact

    for effect in edge_effects:
        candidate = candidates.get(effect.candidate_id)
        if candidate is None:
            diagnostics.append(
                _diagnostic(
                    effect.candidate_id,
                    "edge_effect_candidate_not_found",
                    "edge effect references no candidate in the discovery report",
                )
            )
            continue
        if effect.candidate_id in annotations:
            diagnostics.append(
                _diagnostic(
                    effect.candidate_id,
                    "duplicate_route_impact",
                    "route impact and edge effect were both supplied for one candidate",
                )
            )
            continue
        context = route_contexts.get(effect.candidate_id)
        if context is None:
            diagnostics.append(
                _diagnostic(
                    effect.candidate_id,
                    "edge_effect_requires_route_context",
                    "an edge effect requires a graph, origin, and destinations before promotion",
                )
            )
            continue
        annotations[effect.candidate_id] = assess_gate_route_impact(
            context.graph,
            candidate,
            origin=context.origin,
            destinations=context.destinations,
            effect=effect,
        )

    annotated = tuple(
        annotate_candidate_with_route_impact(candidate, annotations[candidate.candidate_id])
        if candidate.candidate_id in annotations
        else candidate
        for candidate in report.candidates
    )
    return (
        GateDiscoveryReport(
            schema_version=report.schema_version,
            scanner_version=report.scanner_version,
            game=report.game,
            source_root=report.source_root,
            source_revision=report.source_revision,
            candidates=annotated,
            diagnostics=report.diagnostics,
        ),
        tuple(sorted(diagnostics, key=lambda item: (item.candidate_id or "", item.code, item.message))),
    )


def compile_emerald_gate_manifest(
    report: GateDiscoveryReport,
    *,
    manifest: EmeraldGateManifest | None = None,
    route_impacts: Iterable[GateRouteImpact] = (),
    edge_effects: Iterable[GateEdgeEffect] = (),
    route_contexts: Mapping[str, RouteImpactContext] | None = None,
) -> GateCompilationResult:
    """Validate and report-only compile one reviewed Emerald manifest.

    ``route_impacts`` are already assessed annotations.  ``edge_effects`` can
    be assessed here when a matching :class:`RouteImpactContext` is supplied;
    an edge effect without route context remains diagnostic-only.
    """

    selected_manifest = manifest or EMERALD_GATE_MANIFEST
    validation = validate_emerald_gate_manifest(report, selected_manifest)
    annotated_report, route_diagnostics = _annotate_report(
        report,
        route_impacts=route_impacts,
        edge_effects=edge_effects,
        route_contexts=route_contexts or {},
    )
    compiled = compile_gate_candidates(
        annotated_report,
        access_overrides=validation.access_overrides,
        trainer_overrides=validation.trainer_overrides,
    )
    diagnostics = tuple(
        sorted(
            (*validation.diagnostics, *route_diagnostics, *compiled.diagnostics),
            key=lambda item: (item.candidate_id or "", item.code, item.message),
        )
    )
    return GateCompilationResult(
        access_requirements=compiled.access_requirements,
        trainer_checkpoints=compiled.trainer_checkpoints,
        diagnostics=diagnostics,
    )


def manifest_to_json(manifest: EmeraldGateManifest) -> str:
    """Serialize a manifest into a stable review artifact."""

    return json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n"


EMERALD_GATE_MANIFEST = EmeraldGateManifest(
    manifest_version=EMERALD_MANIFEST_VERSION,
    game=EMERALD_GAME_ID,
    source_revision=EMERALD_SOURCE_REVISION,
    scanner_version=EMERALD_SCANNER_VERSION,
    expectations=(
        ReviewedCandidateExpectation(
            candidate_id="emerald:script_event:3b0fd3f96b9dcdd8ed31",
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
        ),
        ReviewedCandidateExpectation(
            candidate_id="emerald:script_event:34df85740152dad2b963",
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
        ),
        ReviewedCandidateExpectation(
            candidate_id="emerald:trainer_checkpoint:baa9b80dae45f7382c78",
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
        ),
    ),
    access_overrides=(
        AccessRequirementOverride(
            candidate_id="emerald:script_event:34df85740152dad2b963",
            map_id=MapRSE.ROUTE102.value,
            required_facts=("pokedex_received",),
            description=(
                "Oldale's westward traversal into Route 102 requires the "
                "Pokédex handoff that sets FLAG_ADVENTURE_STARTED"
            ),
        ),
        AccessRequirementOverride(
            candidate_id="emerald:script_event:3b0fd3f96b9dcdd8ed31",
            map_id=MapRSE.ROUTE104.value,
            required_facts=("petalburg_wally_scene_complete",),
            description=(
                "Petalburg westward traversal into Route 104 requires the "
                "completed Wally scene"
            ),
        ),
    ),
    trainer_overrides=(
        TrainerCheckpointOverride(
            candidate_id="emerald:trainer_checkpoint:baa9b80dae45f7382c78",
            checkpoint_id="route102:first_poochyena",
            map_id=MapRSE.ROUTE102.value,
            trainer_affordance_id=f"trainer:{MapRSE.ROUTE102.value}:2",
            display_name="Calvin",
            party=(TrainerPokemon("Poochyena", 5, ("Tackle", "Howl")),),
            mandatory=True,
            clearance_fact="visited_petalburg",
            source="emerald_reviewed_gate_manifest",
            rom_trainer_id=318,
            script_symbol="Route102_EventScript_Calvin",
        ),
    ),
)


def emerald_gate_manifest() -> EmeraldGateManifest:
    """Return the reviewed immutable Emerald manifest."""

    return EMERALD_GATE_MANIFEST


__all__ = [
    "EMERALD_GAME_ID",
    "EMERALD_GATE_MANIFEST",
    "EMERALD_MANIFEST_VERSION",
    "EMERALD_SCANNER_VERSION",
    "EMERALD_SOURCE_REVISION",
    "EmeraldGateManifest",
    "ManifestValidationResult",
    "ReviewedCandidateExpectation",
    "RouteImpactContext",
    "compile_emerald_gate_manifest",
    "emerald_gate_manifest",
    "manifest_to_json",
    "validate_emerald_gate_manifest",
]
