"""Report-only promotion of source gate candidates into campaign contracts.

The source scanner intentionally produces evidence rather than campaign
policy.  This module is the narrow seam between those layers: a caller must
provide the semantic override that names the map, completion facts, trainer
classification, and any clearance fact.  Route evidence must already show a
high-confidence cut set before a candidate can enter the campaign graph.

No registry or runtime state is mutated here.  The result can be reviewed,
tested, or written to a generated artifact before a game-specific registry
chooses to consume it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable

from modules.trainer_checkpoints import TrainerCheckpoint, TrainerPokemon

from .campaign_access import CampaignAccessRequirement
from .gate_discovery import GateCandidate, GateDiscoveryReport


_ACCESS_KINDS = frozenset({"map_access", "object_visibility", "script_event"})
_VERIFIED_STATUSES = frozenset({"verified", "verified_registry"})


@dataclass(frozen=True, slots=True)
class AccessRequirementOverride:
    """Explicit semantics required to promote an access-gate candidate."""

    candidate_id: str
    map_id: tuple[int, int]
    required_badges: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    description: str = "compiled campaign access requirement"
    confidence: str = "high"
    verification_status: str = "verified"


@dataclass(frozen=True, slots=True)
class TrainerCheckpointOverride:
    """Explicit semantics required to promote a trainer candidate."""

    candidate_id: str
    checkpoint_id: str
    map_id: tuple[int, int]
    trainer_affordance_id: str
    display_name: str
    party: tuple[TrainerPokemon, ...] = ()
    # ``None`` forces the registry author to classify the hazard deliberately.
    mandatory: bool | None = None
    clearance_fact: str | None = None
    source: str = "compiled_gate_candidate"
    rom_trainer_id: int | None = None
    trainer_class: int | None = None
    ai_flags: int | None = None
    gym_trainer: bool = False
    script_symbol: str | None = None
    alternate_script_symbols: tuple[str, ...] = ()
    battle_script_symbols_by_context: tuple[tuple[str, str], ...] = ()
    confidence: str = "high"
    verification_status: str = "verified"


@dataclass(frozen=True, slots=True)
class GateCompilationDiagnostic:
    """A non-promoting result explaining why evidence stayed diagnostic-only."""

    candidate_id: str | None
    code: str
    message: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateCompilationResult:
    """Compiled contracts and diagnostics from one explicit promotion pass."""

    access_requirements: tuple[CampaignAccessRequirement, ...] = ()
    trainer_checkpoints: tuple[TrainerCheckpoint, ...] = ()
    diagnostics: tuple[GateCompilationDiagnostic, ...] = ()

    @property
    def promoted_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                candidate_id
                for candidate_id in (
                    *(requirement.candidate_id for requirement in self.access_requirements),
                    *(checkpoint.candidate_id for checkpoint in self.trainer_checkpoints),
                )
                if candidate_id is not None
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable, reviewable representation of the compile pass."""

        return {
            "access_requirements": [asdict(requirement) for requirement in self.access_requirements],
            "trainer_checkpoints": [asdict(checkpoint) for checkpoint in self.trainer_checkpoints],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }

    def to_json(self) -> str:
        """Serialize the compile pass without depending on object identity."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _source_reference(candidate: GateCandidate) -> str:
    symbol = f"#{candidate.source.symbol}" if candidate.source.symbol else ""
    return f"{candidate.source.path}:{candidate.source.line}{symbol}"


def _provenance(candidate: GateCandidate) -> tuple[tuple[str, str], ...]:
    """Return stable provenance pairs for existing diagnostic renderers."""

    producers = ";".join(
        f"{producer.action}:{producer.symbol}:{producer.source.path}:{producer.source.line}"
        for producer in candidate.producer_actions
    )
    return (
        ("gate_candidate", candidate.candidate_id),
        ("source", _source_reference(candidate)),
        ("source_revision", candidate.source_revision),
        ("guard", candidate.guard_expression or ""),
        ("producer_actions", producers),
        ("route_confidence", candidate.route_evidence.cut_set_confidence),
        ("blocked_routes", ";".join(candidate.route_evidence.blocked_routes)),
        ("alternate_routes", ";".join(candidate.route_evidence.alternate_routes)),
        ("route_evidence", ";".join(
            f"{kind}:{value}" for kind, value in candidate.route_evidence.evidence
        )),
        ("route_notes", ";".join(candidate.route_evidence.notes)),
    )


def _metadata(candidate: GateCandidate) -> dict[str, str]:
    return dict(candidate.metadata)


def _valid_map_id(map_id: object) -> bool:
    return (
        isinstance(map_id, tuple)
        and len(map_id) == 2
        and all(isinstance(value, int) for value in map_id)
    )


def _route_evidence_diagnostic(candidate: GateCandidate) -> GateCompilationDiagnostic | None:
    evidence = candidate.route_evidence
    if evidence.cut_set_confidence != "high":
        return GateCompilationDiagnostic(
            candidate.candidate_id,
            "route_impact_unverified",
            "candidate lacks high-confidence route cut-set evidence",
        )
    if not evidence.blocked_routes:
        return GateCompilationDiagnostic(
            candidate.candidate_id,
            "route_impact_empty",
            "candidate has no blocked destination in its route evidence",
        )
    if evidence.alternate_routes:
        return GateCompilationDiagnostic(
            candidate.candidate_id,
            "alternate_route_exists",
            "candidate has an alternate route and cannot be promoted as a blocker",
        )
    return None


def _override_diagnostic(
    candidate: GateCandidate,
    *,
    confidence: str,
    verification_status: str,
) -> GateCompilationDiagnostic | None:
    if confidence != "high":
        return GateCompilationDiagnostic(
            candidate.candidate_id,
            "semantic_confidence_insufficient",
            "promotion requires a high-confidence semantic override",
        )
    if verification_status not in _VERIFIED_STATUSES:
        return GateCompilationDiagnostic(
            candidate.candidate_id,
            "verification_incomplete",
            "promotion requires verified semantic status",
        )
    if candidate.verification_status == "rejected":
        return GateCompilationDiagnostic(
            candidate.candidate_id,
            "candidate_rejected",
            "source report marks this candidate as rejected",
        )
    return None


def _candidate_index(
    report: GateDiscoveryReport,
) -> tuple[dict[str, GateCandidate], tuple[GateCompilationDiagnostic, ...]]:
    candidates: dict[str, GateCandidate] = {}
    diagnostics: list[GateCompilationDiagnostic] = []
    for candidate in report.candidates:
        if candidate.candidate_id in candidates:
            diagnostics.append(
                GateCompilationDiagnostic(
                    candidate.candidate_id,
                    "duplicate_candidate_id",
                    "source report contains duplicate candidate IDs",
                )
            )
            continue
        candidates[candidate.candidate_id] = candidate
    return candidates, tuple(diagnostics)


def _candidate_for_override(
    candidate_index: dict[str, GateCandidate],
    candidate_id: str,
    used: set[str],
) -> tuple[GateCandidate | None, GateCompilationDiagnostic | None]:
    candidate = candidate_index.get(candidate_id)
    if candidate is None:
        return None, GateCompilationDiagnostic(candidate_id, "candidate_not_found", "override references no source candidate")
    if candidate_id in used:
        return None, GateCompilationDiagnostic(candidate_id, "duplicate_override", "candidate has more than one promotion override")
    used.add(candidate_id)
    return candidate, None


def _compile_access(
    candidate: GateCandidate,
    override: AccessRequirementOverride,
) -> tuple[CampaignAccessRequirement | None, GateCompilationDiagnostic | None]:
    if candidate.kind not in _ACCESS_KINDS:
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "candidate_kind_mismatch",
            f"{candidate.kind!r} cannot compile into a campaign access requirement",
        )
    diagnostic = _override_diagnostic(
        candidate,
        confidence=override.confidence,
        verification_status=override.verification_status,
    ) or _route_evidence_diagnostic(candidate)
    if diagnostic is not None:
        return None, diagnostic
    if not _valid_map_id(override.map_id):
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "map_id_required",
            "access promotion requires an explicit (map_group, map_number) map ID",
        )
    if not (override.required_badges or override.required_facts or override.required_capabilities):
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "completion_semantics_required",
            "access promotion requires at least one badge, campaign fact, or capability",
        )
    provenance = _provenance(candidate)
    return (
        CampaignAccessRequirement(
            requirement_id=f"source:{candidate.candidate_id}",
            map_id=override.map_id,
            required_badges=tuple(override.required_badges),
            required_facts=tuple(override.required_facts),
            required_capabilities=tuple(override.required_capabilities),
            description=override.description,
            candidate_id=candidate.candidate_id,
            provenance=provenance,
            confidence=override.confidence,
            verification_status=override.verification_status,
        ),
        None,
    )


def _compile_trainer(
    candidate: GateCandidate,
    override: TrainerCheckpointOverride,
) -> tuple[TrainerCheckpoint | None, GateCompilationDiagnostic | None]:
    if candidate.kind != "trainer_checkpoint":
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "candidate_kind_mismatch",
            f"{candidate.kind!r} cannot compile into a trainer checkpoint",
        )
    diagnostic = _override_diagnostic(
        candidate,
        confidence=override.confidence,
        verification_status=override.verification_status,
    ) or _route_evidence_diagnostic(candidate)
    if diagnostic is not None:
        return None, diagnostic
    if not _valid_map_id(override.map_id):
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "map_id_required",
            "trainer promotion requires an explicit (map_group, map_number) map ID",
        )
    if not override.checkpoint_id or not override.trainer_affordance_id or not override.display_name:
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "trainer_identity_required",
            "trainer promotion requires checkpoint, affordance, and display identities",
        )
    if override.mandatory is None:
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "mandatory_classification_required",
            "trainer promotion requires an explicit mandatory classification",
        )
    if override.mandatory and not override.clearance_fact:
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "clearance_fact_required",
            "mandatory trainer promotion requires an explicit clearance fact",
        )
    if not override.party:
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "trainer_roster_unresolved",
            "trainer promotion requires a resolved non-empty party roster",
        )
    metadata = _metadata(candidate)
    if metadata.get("roster_resolution") == "unresolved":
        return None, GateCompilationDiagnostic(
            candidate.candidate_id,
            "trainer_roster_unresolved",
            "source report does not resolve the trainer roster",
        )
    provenance = _provenance(candidate)
    return (
        TrainerCheckpoint(
            checkpoint_id=override.checkpoint_id,
            map_id=override.map_id,
            trainer_affordance_id=override.trainer_affordance_id,
            display_name=override.display_name,
            party=tuple(override.party),
            mandatory=override.mandatory,
            source=override.source,
            rom_trainer_id=override.rom_trainer_id,
            trainer_class=override.trainer_class,
            ai_flags=override.ai_flags,
            gym_trainer=override.gym_trainer,
            clearance_fact=override.clearance_fact,
            script_symbol=override.script_symbol or candidate.source.symbol,
            alternate_script_symbols=tuple(override.alternate_script_symbols),
            battle_script_symbols_by_context=tuple(override.battle_script_symbols_by_context),
            source_records=(("gate_candidate", candidate.candidate_id), ("source", _source_reference(candidate))),
            candidate_id=candidate.candidate_id,
            provenance=provenance,
            confidence=override.confidence,
            verification_status=override.verification_status,
        ),
        None,
    )


def compile_gate_candidates(
    report: GateDiscoveryReport,
    *,
    access_overrides: Iterable[AccessRequirementOverride] = (),
    trainer_overrides: Iterable[TrainerCheckpointOverride] = (),
) -> GateCompilationResult:
    """Compile explicitly overridden candidates without changing runtime state."""

    candidate_index, initial_diagnostics = _candidate_index(report)
    diagnostics = list(initial_diagnostics)
    used: set[str] = set()
    access_requirements: list[CampaignAccessRequirement] = []
    trainer_checkpoints: list[TrainerCheckpoint] = []

    for override in sorted(tuple(access_overrides), key=lambda item: item.candidate_id):
        candidate, diagnostic = _candidate_for_override(candidate_index, override.candidate_id, used)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
            continue
        requirement, diagnostic = _compile_access(candidate, override)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        elif requirement is not None:
            access_requirements.append(requirement)

    for override in sorted(tuple(trainer_overrides), key=lambda item: item.candidate_id):
        candidate, diagnostic = _candidate_for_override(candidate_index, override.candidate_id, used)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
            continue
        checkpoint, diagnostic = _compile_trainer(candidate, override)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        elif checkpoint is not None:
            trainer_checkpoints.append(checkpoint)

    diagnostics.sort(key=lambda item: (item.candidate_id or "", item.code, item.message))
    return GateCompilationResult(
        access_requirements=tuple(sorted(access_requirements, key=lambda item: item.requirement_id)),
        trainer_checkpoints=tuple(sorted(trainer_checkpoints, key=lambda item: item.checkpoint_id)),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "AccessRequirementOverride",
    "GateCompilationDiagnostic",
    "GateCompilationResult",
    "TrainerCheckpointOverride",
    "compile_gate_candidates",
]
