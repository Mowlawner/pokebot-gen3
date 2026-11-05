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
    # Raw encounter-table weights and per-species type facts.  These remain
    # optional so callers with only a coarse pool can keep using the policy.
    species_rates: tuple[tuple[str, float], ...] = ()
    species_types: tuple[tuple[str, tuple[str, ...]], ...] = ()
    min_level: int | None = None
    max_level: int | None = None
    trainer_exposure_cost: float = 0.0
    training_value: float = 0.0


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
    open_party_slot: float = 8.0
    species_clause_conditioning: float = 3.0
    # A Species Clause reroll can turn a mixed table into a highly predictable
    # legal result. Price that certainty separately from duplicate mass;
    # otherwise a pool with one guaranteed remaining species looks the same as
    # a pool with many possible remaining species.
    # A guaranteed legal result under Species Clause is a meaningful campaign
    # resource when the party has capacity. Give it enough weight to justify a
    # bounded detour, while leaving type coverage and route cost in the same
    # score so the choice remains policy-driven.
    conditioned_pool_certainty: float = 20.0


@dataclass(frozen=True, slots=True)
class CapturePolicyContext:
    """Current strategic state supplied by campaign planning."""

    boss_weakness_types: frozenset[str] = frozenset()
    team_types: frozenset[str] = frozenset()
    available_future_methods: frozenset[EncounterMethod] = frozenset()
    species_clause_enabled: bool = False
    captured_species: frozenset[str] = frozenset()
    open_party_slots: int | None = None
    weights: CapturePolicyWeights = CapturePolicyWeights()


@dataclass(frozen=True, slots=True)
class EncounterPortfolioContext:
    """Campaign-level inputs for selecting a sequence of encounter areas."""

    capture: CapturePolicyContext = CapturePolicyContext()
    current_level: int | None = None
    target_level: int | None = None
    max_encounters: int | None = None
    minimum_score: float = float("-inf")


@dataclass(frozen=True, slots=True)
class EncounterPortfolioPlan:
    """Deterministic portfolio result with a reviewable rationale."""

    candidates: tuple[EncounterCandidate, ...]
    selected: EncounterCandidate | None
    score: float
    expected_legal_captures: float
    expected_new_species: float
    reason: str
    # Keep the complete ranked frontier separate from the bounded slice that
    # can be mounted in the current party.  Consumers that only need the
    # selected portfolio continue to use ``candidates``.
    ordered_candidates: tuple[EncounterCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class CaptureRecommendation:
    """Deterministic result with an auditable score and reason."""

    candidate: EncounterCandidate | None
    pursue_now: bool
    score: float | None
    reason: str


def _species_key(species: str) -> str:
    return str(species).strip().casefold()


def conditioned_species_rates(
    candidate: EncounterCandidate,
    context: CapturePolicyContext,
) -> tuple[tuple[str, float], ...]:
    """Condition one encounter table on the configured Species Clause.

    The result is a normalized distribution over eligible species. A
    duplicate is a reroll opportunity, so common duplicate outcomes are
    removed before the remaining species are normalized. Exact species
    identity is the policy boundary; evolutionary-family legality remains
    owned by the Nuzlocke rules projection.
    """

    raw = tuple(candidate.species_rates)
    if not raw:
        pool = tuple(dict.fromkeys(candidate.species_pool))
        raw = tuple((species, 1.0) for species in pool) or (("__unknown__", 1.0),)
    merged: dict[str, tuple[str, float]] = {}
    for species, rate in raw:
        try:
            weight = max(0.0, float(rate))
        except (TypeError, ValueError):
            continue
        key = _species_key(species)
        if not key or weight <= 0:
            continue
        display, previous = merged.get(key, (str(species), 0.0))
        merged[key] = (display, previous + weight)
    if not merged:
        return ()
    if context.species_clause_enabled:
        captured = {_species_key(species) for species in context.captured_species}
        eligible = {key: value for key, value in merged.items() if key not in captured}
        if eligible:
            merged = eligible
        else:
            return ()
    total = sum(weight for _, weight in merged.values())
    if total <= 0:
        return ()
    return tuple(
        (display, weight / total)
        for display, weight in sorted(merged.values(), key=lambda item: _species_key(item[0]))
    )


def _species_type_map(candidate: EncounterCandidate) -> dict[str, frozenset[str]]:
    mapped = {
        _species_key(species): frozenset(str(type_name).casefold() for type_name in types)
        for species, types in candidate.species_types
    }
    fallback = frozenset(str(type_name).casefold() for type_name in candidate.type_pool)
    mapped.setdefault("__unknown__", fallback)
    for species in candidate.species_pool:
        mapped.setdefault(_species_key(species), fallback)
    return mapped


def conditioned_pool_certainty(candidate: EncounterCandidate, context: CapturePolicyContext) -> float:
    """Return the probability of the most likely legal species after rerolls."""

    rates = conditioned_species_rates(candidate, context)
    return max((probability for _, probability in rates), default=0.0)


def score_encounter(candidate: EncounterCandidate, context: CapturePolicyContext) -> float:
    """Score a legal, currently executable candidate."""

    weights = context.weights
    rates = conditioned_species_rates(candidate, context)
    if not rates:
        return float("-inf")
    type_map = _species_type_map(candidate)
    boss_gain = sum(
        probability * len(type_map.get(_species_key(species), frozenset()) & context.boss_weakness_types)
        for species, probability in rates
    )
    team_gain = sum(
        probability * len(type_map.get(_species_key(species), frozenset()) - context.team_types)
        for species, probability in rates
    )
    raw_rates = tuple(candidate.species_rates) or tuple((species, 1.0) for species in candidate.species_pool)
    raw_total = sum(max(0.0, float(rate)) for _, rate in raw_rates)
    raw_duplicate = 0.0
    if context.species_clause_enabled and raw_total > 0:
        captured = {_species_key(species) for species in context.captured_species}
        raw_duplicate = sum(
            max(0.0, float(rate))
            for species, rate in raw_rates
            if _species_key(species) in captured
        ) / raw_total
    revisit_risk = max(0.0, 1.0 - candidate.revisit_likelihood)
    capacity_gain = (
        max(0, int(context.open_party_slots)) * weights.open_party_slot
        if context.open_party_slots is not None
        else 0.0
    )
    clause_certainty = (
        conditioned_pool_certainty(candidate, context)
        if context.species_clause_enabled
        else 0.0
    )
    return (
        boss_gain * weights.boss_type_coverage
        + team_gain * weights.team_type_coverage
        + candidate.future_value * weights.future_value
        + capacity_gain
        + raw_duplicate * weights.species_clause_conditioning
        + clause_certainty * weights.conditioned_pool_certainty
        - (candidate.route_cost or 0) * weights.route_cost
        - (candidate.detour_cost or 0) * weights.detour_cost
        - candidate.resource_cost * weights.resource_cost
        - candidate.trainer_exposure_cost * weights.resource_cost
        - revisit_risk * weights.revisit_risk
    )


def encounter_level_fit(candidate: EncounterCandidate, context: EncounterPortfolioContext) -> float:
    """Return a bounded preference for encounters at the current level."""

    if context.current_level is None or candidate.min_level is None or candidate.max_level is None:
        return 0.0
    low = min(candidate.min_level, candidate.max_level)
    high = max(candidate.min_level, candidate.max_level)
    if low <= context.current_level <= high:
        return 1.0
    distance = low - context.current_level if context.current_level < low else context.current_level - high
    return 1.0 / (1.0 + max(0, distance))


def score_portfolio_candidate(candidate: EncounterCandidate, context: EncounterPortfolioContext) -> float:
    """Score one candidate while retaining capture-policy compatibility."""

    score = score_encounter(candidate, context.capture)
    if score == float("-inf"):
        return score
    level_score = encounter_level_fit(candidate, context)
    target_bonus = 0.0
    if context.target_level is not None and candidate.max_level is not None:
        target_distance = abs(candidate.max_level - context.target_level)
        target_bonus = 1.0 / (1.0 + target_distance)
    legal_rates = conditioned_species_rates(candidate, context.capture)
    legal_probability = min(1.0, sum(probability for _, probability in legal_rates))
    return score + candidate.training_value + level_score + target_bonus + legal_probability


def choose_encounter_portfolio(
    candidates: tuple[EncounterCandidate, ...],
    context: EncounterPortfolioContext,
) -> EncounterPortfolioPlan:
    """Choose a bounded, ordered encounter portfolio.

    This is intentionally deterministic.  It removes candidates with no
    legal conditioned outcome, ranks the remaining areas by capture value,
    level fit, route cost, and trainer exposure, then keeps the useful
    frontier up to ``max_encounters``.  The campaign can re-run it after each
    capture so species history changes the next portfolio.
    """

    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.pending
        and candidate.reachable
        and not candidate.future_only
        and conditioned_species_rates(candidate, context.capture)
        and score_portfolio_candidate(candidate, context) >= context.minimum_score
    )
    # Preserve the caller's deterministic campaign order when scores tie.
    # Area identity remains a useful diagnostic field, but it is not a reason
    # to reorder otherwise equivalent encounter opportunities and can fight
    # the route sequence supplied by the campaign registry.
    ranked = tuple(
        candidate
        for _index, candidate in sorted(
            enumerate(eligible),
            key=lambda item: (-score_portfolio_candidate(item[1], context), item[0]),
        )
    )
    if context.max_encounters is not None:
        selected_portfolio = ranked[: max(0, int(context.max_encounters))]
    else:
        selected_portfolio = ranked
    expected_new_species = sum(
        1.0
        for candidate in selected_portfolio
        if conditioned_species_rates(candidate, context.capture)
    )
    total_score = sum(score_portfolio_candidate(candidate, context) for candidate in selected_portfolio)
    if not selected_portfolio:
        reason = "no pending reachable encounter has a legal conditioned outcome"
    else:
        reason = (
            f"selected {len(selected_portfolio)} encounter areas by expected legal capture value, "
            "level fit, route cost, and trainer exposure"
        )
    return EncounterPortfolioPlan(
        selected_portfolio,
        selected_portfolio[0] if selected_portfolio else None,
        total_score,
        float(len(selected_portfolio)),
        expected_new_species,
        reason,
        ranked,
    )


def choose_encounter(
    candidates: tuple[EncounterCandidate, ...],
    context: CapturePolicyContext,
) -> CaptureRecommendation:
    """Choose an encounter, deferring only for a superior known future method."""

    portfolio = choose_encounter_portfolio(
        candidates,
        EncounterPortfolioContext(capture=context, max_encounters=1),
    )
    current = tuple(
        c
        for c in candidates
        if c.pending
        and c.reachable
        and not c.future_only
        and conditioned_species_rates(c, context)
    )
    if not current:
        return CaptureRecommendation(None, False, None, "no reachable pending encounter")

    if portfolio.selected is not None:
        selected = portfolio.selected
        selected_score = score_encounter(selected, context)
    else:
        selected = None
        selected_score = None

    if selected is None:
        ranked = sorted(current, key=lambda c: (-score_encounter(c, context), c.area, c.method.value))
        selected = ranked[0]
        selected_score = score_encounter(selected, context)

    future = tuple(
        c
        for c in candidates
        if c.pending
        and c.reachable
        and c.future_only
        and c.method in context.available_future_methods
        and conditioned_species_rates(c, context)
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
