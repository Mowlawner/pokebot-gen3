"""Classify source-discovered trainer candidates from route-risk evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .gate_discovery import GateCandidate


TRAINER_GATE_CLASSIFICATIONS = frozenset({"mandatory", "costly_to_avoid", "avoidable", "unknown"})


@dataclass(frozen=True, slots=True)
class TrainerGateAssessment:
    """Pure classification evidence for one source trainer candidate."""

    candidate_id: str
    trainer_id: str | None
    classification: str
    confidence: str
    reason: str
    provenance: tuple[tuple[str, str], ...] = ()


def _provenance(candidate: GateCandidate) -> tuple[tuple[str, str], ...]:
    symbol = f"#{candidate.source.symbol}" if candidate.source.symbol else ""
    return (
        ("gate_candidate", candidate.candidate_id),
        ("source", f"{candidate.source.path}:{candidate.source.line}{symbol}"),
        ("source_revision", candidate.source_revision),
    )


def classify_trainer_candidate(
    candidate: GateCandidate,
    *,
    trainer_id: str | None = None,
    avoidability: str | None = None,
    route_confidence: str | None = None,
    route_reason: str = "",
) -> TrainerGateAssessment:
    """Classify a trainer without treating missing route evidence as safety.

    ``avoidability`` is the normalized value published by route analysis. A
    high-confidence source route cut set can establish ``mandatory`` directly;
    all other classifications require an explicit route-risk result. The
    result is diagnostic evidence until a registry supplies the checkpoint's
    semantic override and clearance fact.
    """

    if candidate.kind != "trainer_checkpoint":
        return TrainerGateAssessment(
            candidate.candidate_id,
            trainer_id,
            "unknown",
            "low",
            f"candidate kind {candidate.kind!r} is not a trainer checkpoint",
            _provenance(candidate),
        )

    route_evidence = candidate.route_evidence
    if (
        route_evidence.cut_set_confidence == "high"
        and route_evidence.blocked_routes
        and not route_evidence.alternate_routes
    ):
        return TrainerGateAssessment(
            candidate.candidate_id,
            trainer_id,
            "mandatory",
            "high",
            "candidate is a high-confidence route cut set",
            _provenance(candidate),
        )

    if route_confidence in {None, "unknown"} or avoidability in {None, "unknown"}:
        return TrainerGateAssessment(
            candidate.candidate_id,
            trainer_id,
            "unknown",
            "unknown",
            route_reason or "trainer route exposure is unresolved",
            _provenance(candidate),
        )

    if avoidability in {"unavoidable", "imminent"}:
        classification = "mandatory"
    elif avoidability in {"costly", "practically_unavoidable"}:
        classification = "costly_to_avoid"
    elif avoidability == "avoidable":
        classification = "avoidable"
    else:
        classification = "unknown"

    confidence = route_confidence if classification != "unknown" else "low"
    return TrainerGateAssessment(
        candidate.candidate_id,
        trainer_id,
        classification,
        confidence,
        route_reason or f"route analysis classified trainer as {classification}",
        _provenance(candidate),
    )


__all__ = [
    "TRAINER_GATE_CLASSIFICATIONS",
    "TrainerGateAssessment",
    "classify_trainer_candidate",
]
