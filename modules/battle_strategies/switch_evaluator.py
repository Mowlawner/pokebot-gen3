"""Shared, policy-weighted evaluation of prospective battle switches.

The battle strategies use different priorities when selecting a replacement:
an emergency switch should preserve the party, while a leveling switch should
prefer a safe, fresh, under-leveled member.  This module keeps the normalized
facts and weighting in one place without deciding how those facts are read
from the live emulator.

Safety is intentionally split from scoring.  A policy may rank legal
candidates differently, but a finite score cannot override a hard safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class SwitchEvaluationContext(Enum):
    """The policy context in which a switch is being considered."""

    EMERGENCY = "emergency"
    LEVELING = "leveling"
    SUPPORT = "support"
    TRAINER_REPLACEMENT = "trainer_replacement"
    CAPTURE = "capture"
    PLANNER = "planner"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class SwitchPolicyWeights:
    """Weights for normalized switch factors.

    All terms are normalized to approximately ``0..1``. Positive weights are
    rewards, except ``poison_risk`` and ``unknown_response`` which are costs.
    ``minimum_survival_probability`` and ``require_response_facts`` are hard
    policy gates, not score terms.
    """

    survival_probability: float
    critical_safety: float
    response_margin: float
    offensive_progress: float
    guaranteed_ko: float
    training_priority: float
    fresh_participant: float
    health_ratio: float
    healthy_status: float
    continuity: float
    poison_risk: float
    unknown_response: float
    minimum_survival_probability: float | None
    require_response_facts: bool


_POLICIES: dict[SwitchEvaluationContext, SwitchPolicyWeights] = {
    SwitchEvaluationContext.EMERGENCY: SwitchPolicyWeights(
        survival_probability=8.0,
        critical_safety=4.0,
        response_margin=4.0,
        offensive_progress=1.5,
        guaranteed_ko=1.0,
        training_priority=0.0,
        fresh_participant=0.0,
        health_ratio=1.0,
        healthy_status=1.0,
        continuity=0.0,
        poison_risk=2.0,
        unknown_response=2.0,
        minimum_survival_probability=0.85,
        require_response_facts=True,
    ),
    SwitchEvaluationContext.LEVELING: SwitchPolicyWeights(
        survival_probability=7.0,
        critical_safety=3.5,
        response_margin=3.0,
        offensive_progress=1.0,
        guaranteed_ko=1.0,
        training_priority=3.0,
        fresh_participant=2.5,
        health_ratio=0.75,
        healthy_status=1.5,
        continuity=0.0,
        poison_risk=2.0,
        unknown_response=2.0,
        minimum_survival_probability=0.85,
        require_response_facts=True,
    ),
    SwitchEvaluationContext.SUPPORT: SwitchPolicyWeights(
        survival_probability=8.0,
        critical_safety=4.0,
        response_margin=4.0,
        offensive_progress=2.0,
        guaranteed_ko=1.5,
        training_priority=0.0,
        fresh_participant=0.0,
        health_ratio=1.0,
        healthy_status=1.0,
        continuity=0.0,
        poison_risk=2.0,
        unknown_response=2.0,
        minimum_survival_probability=0.85,
        require_response_facts=True,
    ),
    SwitchEvaluationContext.TRAINER_REPLACEMENT: SwitchPolicyWeights(
        survival_probability=7.0,
        critical_safety=3.0,
        response_margin=3.0,
        offensive_progress=1.0,
        guaranteed_ko=1.5,
        training_priority=2.5,
        fresh_participant=1.0,
        health_ratio=1.0,
        healthy_status=1.5,
        continuity=0.75,
        poison_risk=2.5,
        unknown_response=2.0,
        # A known response is screened by the trainer-replacement caller
        # using its visible-response rule. Unknown responses remain legal as
        # comparative candidates, but receive an explicit uncertainty cost.
        minimum_survival_probability=None,
        require_response_facts=False,
    ),
    SwitchEvaluationContext.CAPTURE: SwitchPolicyWeights(
        survival_probability=8.0,
        critical_safety=4.0,
        response_margin=4.0,
        offensive_progress=1.5,
        guaranteed_ko=0.5,
        training_priority=0.5,
        fresh_participant=1.0,
        health_ratio=1.0,
        healthy_status=1.5,
        continuity=0.0,
        poison_risk=2.0,
        unknown_response=2.0,
        minimum_survival_probability=0.85,
        require_response_facts=False,
    ),
    SwitchEvaluationContext.PLANNER: SwitchPolicyWeights(
        survival_probability=5.0,
        critical_safety=3.0,
        response_margin=3.0,
        offensive_progress=2.0,
        guaranteed_ko=1.5,
        training_priority=0.0,
        fresh_participant=0.0,
        health_ratio=1.0,
        healthy_status=0.0,
        continuity=0.0,
        poison_risk=0.0,
        unknown_response=1.0,
        minimum_survival_probability=None,
        require_response_facts=False,
    ),
    SwitchEvaluationContext.FALLBACK: SwitchPolicyWeights(
        survival_probability=2.0,
        critical_safety=1.0,
        response_margin=1.0,
        offensive_progress=1.0,
        guaranteed_ko=0.5,
        training_priority=1.5,
        fresh_participant=0.5,
        health_ratio=1.0,
        healthy_status=0.5,
        continuity=0.0,
        poison_risk=0.5,
        unknown_response=0.5,
        minimum_survival_probability=None,
        require_response_facts=False,
    ),
}


@dataclass(frozen=True, slots=True)
class SwitchCandidateFacts:
    """Normalized facts consumed by the shared evaluator."""

    party_index: int
    hp: int
    max_hp: int
    opponent_hp: int
    damage_min: int = 0
    damage_max: int = 0
    incoming_damage_max: int | None = None
    incoming_critical_damage_max: int | None = None
    critical_ko_probability: float | None = None
    survival_probability: float | None = None
    strict_safe: bool | None = None
    probabilistically_safe: bool | None = None
    response_facts_known: bool = True
    can_damage: bool = True
    cap_legal: bool = True
    fresh_participant: bool = True
    training_priority: float = 0.0
    healthy: bool = True
    poison_threat: bool = False
    guaranteed_ko: bool | None = None
    is_current: bool = False
    # Hard exclusion for every party slot currently deployed on the player's
    # side.  A caller may keep the current slot as a "stay in" comparison
    # candidate (trainer replacement does this), so the set is intentionally
    # supplied by the caller rather than inferred from ``is_current``.
    active_party_indices: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class SwitchCandidateEvaluation:
    """The decision-ready result for one prospective switch."""

    party_index: int
    context: SwitchEvaluationContext
    eligible: bool
    score: float
    reason: str
    components: tuple[tuple[str, float], ...]

    @property
    def component_map(self) -> dict[str, float]:
        return dict(self.components)


def get_switch_policy(policy: SwitchEvaluationContext | SwitchPolicyWeights) -> SwitchPolicyWeights:
    """Resolve a named policy or accept a caller-supplied weight profile."""

    if isinstance(policy, SwitchPolicyWeights):
        return policy
    return _POLICIES[policy]


def relative_training_score(priority: tuple[int, ...], priorities: tuple[tuple[int, ...], ...]) -> float:
    """Convert a lexicographic training priority into a normalized reward."""

    if not priorities:
        return 0.0
    ordered = sorted(priorities)
    try:
        rank = ordered.index(priority)
    except ValueError:
        rank = len(ordered) - 1
    if len(ordered) == 1:
        return 1.0
    return 1.0 - (rank / (len(ordered) - 1))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not isfinite(value):
        return lower
    return max(lower, min(upper, value))


def _response_margin(facts: SwitchCandidateFacts) -> float:
    if facts.incoming_damage_max is None or facts.hp <= 0:
        return 0.0
    return _clamp(1.0 - (facts.incoming_damage_max / facts.hp))


def _strict_safety(facts: SwitchCandidateFacts) -> bool:
    if facts.strict_safe is not None:
        return facts.strict_safe
    return (
        facts.hp > 0
        and facts.incoming_damage_max is not None
        and facts.incoming_damage_max < facts.hp
        and (
            facts.incoming_critical_damage_max is None
            or facts.incoming_critical_damage_max < facts.hp
        )
    )


def _survival_probability(facts: SwitchCandidateFacts) -> float | None:
    if facts.survival_probability is not None:
        return _clamp(facts.survival_probability)
    if _strict_safety(facts):
        return 1.0
    if facts.incoming_damage_max is None:
        return None
    return 0.0


def evaluate_switch_candidate(
    facts: SwitchCandidateFacts,
    policy: SwitchEvaluationContext | SwitchPolicyWeights,
) -> SwitchCandidateEvaluation:
    """Apply hard legality/safety gates and calculate a weighted score."""

    context = policy if isinstance(policy, SwitchEvaluationContext) else SwitchEvaluationContext.FALLBACK
    weights = get_switch_policy(policy)

    def rejected(rejection_reason: str) -> SwitchCandidateEvaluation:
        return SwitchCandidateEvaluation(
            facts.party_index,
            context,
            False,
            float("-inf"),
            rejection_reason,
            (),
        )

    if facts.hp <= 0 or facts.max_hp <= 0:
        return rejected("party member is fainted or has unavailable HP")
    if facts.party_index in facts.active_party_indices:
        return rejected("replacement is already active in battle")
    if not facts.can_damage or facts.damage_max <= 0:
        return rejected("replacement has no effective damaging move against the opponent")
    if not facts.cap_legal:
        return rejected("projected experience would violate the active level cap")
    if weights.require_response_facts and not facts.response_facts_known:
        return rejected("opponent response facts are incomplete")

    strict_safe = _strict_safety(facts)
    survival_probability = _survival_probability(facts)
    minimum_survival = weights.minimum_survival_probability
    probabilistically_safe = (
        survival_probability is not None
        and minimum_survival is not None
        and survival_probability >= minimum_survival
    )
    if minimum_survival is not None and not strict_safe and not probabilistically_safe:
        return rejected("switch survival probability is below the policy threshold")

    if facts.guaranteed_ko is None:
        guaranteed_ko = facts.damage_min >= facts.opponent_hp > 0
    else:
        guaranteed_ko = facts.guaranteed_ko
    offensive_progress = _clamp(
        facts.damage_max / facts.opponent_hp if facts.opponent_hp > 0 else 0.0
    )
    critical_ko_probability = _clamp(
        facts.critical_ko_probability if facts.critical_ko_probability is not None else (0.0 if strict_safe else 1.0)
    )
    survival_reward = survival_probability if survival_probability is not None else 0.5
    components = {
        "survival_probability": survival_reward,
        "critical_safety": 1.0 - critical_ko_probability,
        "response_margin": _response_margin(facts),
        "offensive_progress": offensive_progress,
        "guaranteed_ko": float(guaranteed_ko),
        "training_priority": _clamp(facts.training_priority),
        "fresh_participant": float(facts.fresh_participant),
        "health_ratio": _clamp(facts.hp / facts.max_hp if facts.max_hp > 0 else 0.0),
        "healthy_status": float(facts.healthy),
        "continuity": float(facts.is_current),
        "poison_risk": float(facts.poison_threat and not facts.healthy),
        "unknown_response": float(not facts.response_facts_known),
    }
    score = (
        weights.survival_probability * components["survival_probability"]
        + weights.critical_safety * components["critical_safety"]
        + weights.response_margin * components["response_margin"]
        + weights.offensive_progress * components["offensive_progress"]
        + weights.guaranteed_ko * components["guaranteed_ko"]
        + weights.training_priority * components["training_priority"]
        + weights.fresh_participant * components["fresh_participant"]
        + weights.health_ratio * components["health_ratio"]
        + weights.healthy_status * components["healthy_status"]
        + weights.continuity * components["continuity"]
        - weights.poison_risk * components["poison_risk"]
        - weights.unknown_response * components["unknown_response"]
    )
    safety_detail = (
        "survives normal and critical response ceilings"
        if strict_safe
        else f"estimated survival={survival_reward:.1%}"
        if survival_probability is not None
        else "response survival is unknown"
    )
    reason = (
        f"{safety_detail}; damage={facts.damage_min}-{facts.damage_max}; "
        f"fresh={facts.fresh_participant}; training={components['training_priority']:.2f}; "
        f"weighted score={score:.2f}"
    )
    return SwitchCandidateEvaluation(
        facts.party_index,
        context,
        True,
        score,
        reason,
        tuple(components.items()),
    )


def rank_switch_candidates(
    candidates: tuple[SwitchCandidateFacts, ...] | list[SwitchCandidateFacts],
    policy: SwitchEvaluationContext | SwitchPolicyWeights,
) -> tuple[SwitchCandidateEvaluation, ...]:
    """Evaluate and rank candidates from best to worst."""

    evaluations = tuple(evaluate_switch_candidate(candidate, policy) for candidate in candidates)
    return tuple(sorted(evaluations, key=lambda evaluation: evaluation.score, reverse=True))
