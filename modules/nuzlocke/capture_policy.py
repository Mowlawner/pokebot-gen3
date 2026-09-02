"""Pure deliberate-capture policy.

The policy decides which already-legal encounter opportunity to pursue.  It
does not read the emulator, consume an encounter, or decide battle turns.
"""

from dataclasses import dataclass
from enum import Enum


class EncounterMethod(str, Enum):
    LAND = "land"
    SURF = "surf"
    FISHING = "fishing"
    FISHING_OLD_ROD = "fishing_old_rod"
    FISHING_GOOD_ROD = "fishing_good_rod"
    FISHING_SUPER_ROD = "fishing_super_rod"
    ROCK_SMASH = "rock_smash"
    STATIC = "static"
    GIFT = "gift"
    FORCED = "forced"
    SAFARI = "safari"
    ROAMER = "roamer"
    EGG = "egg"


@dataclass(frozen=True, slots=True)
class EncounterCandidate:
    """ROM-backed opportunity enriched with planner-owned strategic facts."""

    area: tuple[int, int]
    method: EncounterMethod
    species_pool: tuple[str, ...] = ()
    type_pool: tuple[str, ...] = ()
    reachable: bool = True
    pending: bool = True
    route_cost: int | None = None
    detour_cost: int | None = None
    resource_cost: int = 0
    future_only: bool = False
    revisit_likelihood: float = 1.0
    future_value: float = 0.0


@dataclass(frozen=True, slots=True)
class CapturePolicyWeights:
    """Configurable scoring weights for one campaign decision."""

    boss_type_coverage: float = 10.0
    team_type_coverage: float = 3.0
    future_value: float = 2.0
    route_cost: float = 0.05
    detour_cost: float = 0.1
    resource_cost: float = 0.2
    revisit_risk: float = 4.0


@dataclass(frozen=True, slots=True)
class CapturePolicyContext:
    """Current strategic state supplied by campaign planning."""

    boss_weakness_types: frozenset[str] = frozenset()
    team_types: frozenset[str] = frozenset()
    available_future_methods: frozenset[EncounterMethod] = frozenset()
    weights: CapturePolicyWeights = CapturePolicyWeights()


@dataclass(frozen=True, slots=True)
class CaptureRecommendation:
    """Deterministic result with an auditable score and reason."""

    candidate: EncounterCandidate | None
    pursue_now: bool
    score: float | None
    reason: str


def score_encounter(candidate: EncounterCandidate, context: CapturePolicyContext) -> float:
    """Score a legal, currently executable candidate."""

    weights = context.weights
    types = set(candidate.type_pool)
    boss_gain = len(types & context.boss_weakness_types)
    team_gain = len(types - context.team_types)
    revisit_risk = max(0.0, 1.0 - candidate.revisit_likelihood)
    return (
        boss_gain * weights.boss_type_coverage
        + team_gain * weights.team_type_coverage
        + candidate.future_value * weights.future_value
        - (candidate.route_cost or 0) * weights.route_cost
        - (candidate.detour_cost or 0) * weights.detour_cost
        - candidate.resource_cost * weights.resource_cost
        - revisit_risk * weights.revisit_risk
    )


def choose_encounter(
    candidates: tuple[EncounterCandidate, ...],
    context: CapturePolicyContext,
) -> CaptureRecommendation:
    """Choose an encounter, deferring only for a superior known future method."""

    current = tuple(c for c in candidates if c.pending and c.reachable and not c.future_only)
    if not current:
        return CaptureRecommendation(None, False, None, "no reachable pending encounter")

    ranked = sorted(current, key=lambda c: (-score_encounter(c, context), c.area, c.method.value))
    selected = ranked[0]
    selected_score = score_encounter(selected, context)

    future = tuple(
        c
        for c in candidates
        if c.pending and c.reachable and c.future_only and c.method in context.available_future_methods
    )
    if future:
        best_future = max(future, key=lambda c: (score_encounter(c, context), c.area, c.method.value))
        future_score = score_encounter(best_future, context)
        if future_score > selected_score and selected.revisit_likelihood > 0:
            return CaptureRecommendation(
                best_future,
                False,
                future_score,
                "defer current encounter for a superior reachable future method",
            )

    return CaptureRecommendation(selected, True, selected_score, "pursue the best currently available legal encounter")
