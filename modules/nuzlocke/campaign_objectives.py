"""Pure campaign-objective planning and selection.

Objectives describe what the campaign should accomplish.  They do not read
the emulator or execute tactical goals; a future integration layer can use
``execution_id`` and ``tactical_target`` after selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from modules.goals import (
    EncounterMode,
    Goal,
    NavigationGoal,
    ReachLocation,
    ReachWarp,
    early_pokeball_goal,
    introductory_rival_goal,
)
from modules.map_data import MapRSE

from .campaign_state import CampaignState, Fact, FactStatus, RunStatus
from .resource_policy import EncounterPolicy, ReadinessImportance, ResourceObjective
from modules.world_navigation import WorldMapGraph, WorldNavigationError, get_world_map_graph
from .encounter_catalog import EncounterOpportunity, encounter_opportunities
from .emerald_campaign_registry import emerald_bosses, evaluate_emerald_fact
from .snapshots import CampaignObservationLifecycle


class ObjectiveStatus(Enum):
    """Evaluation status used when selecting a campaign objective."""

    READY = "ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"


class RouteRelation(Enum):
    """Relationship between an optional task route and progression route."""

    ON_ROUTE = "on_route"
    DETOUR = "detour"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class EncounterClassification(Enum):
    """Geometric classification of an encounter's route detour."""

    ON_ROUTE = "on_route"
    SMALL_DETOUR = "small_detour"
    LARGE_DETOUR = "large_detour"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class EncounterRecommendation(Enum):
    """Policy recommendation for an available encounter opportunity."""

    PREFER = "prefer"
    DEFER = "defer"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EncounterEvaluationPolicy:
    """Thresholds used to classify encounter detour costs."""

    small_detour_cost: int = 10
    small_detour_ratio: float = 0.25


@dataclass(frozen=True, slots=True)
class EncounterEvaluation:
    """Measured route costs and recommendation for one encounter task."""

    task_id: str
    location: tuple[int, int]
    progression_destination: tuple[int, int] | None
    classification: EncounterClassification
    recommendation: EncounterRecommendation
    direct_progression_cost: int | None
    encounter_cost: int | None
    encounter_to_progression_cost: int | None
    via_encounter_cost: int | None
    detour_cost: int | None
    detour_ratio: float | None


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Measured relationship between a task destination and progression."""

    current_location: tuple[int, int]
    progression_destination: tuple[int, int] | None
    task_destination: tuple[int, int]
    direct_distance: int | None
    via_task_distance: int | None
    detour_distance: int | None
    relation: RouteRelation


@dataclass(frozen=True, slots=True)
class CampaignPredicate:
    """A named pure predicate over :class:`CampaignState`."""

    predicate_id: str
    description: str
    evaluator: Callable[[CampaignState], Fact[bool]]

    def evaluate(self, state: CampaignState) -> Fact[bool]:
        """Evaluate this predicate against an already-observed campaign state."""

        return self.evaluator(state)


def party_fully_restored() -> CampaignPredicate:
    """Freshly observed party restoration fact used by safety objectives."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        """Evaluate the target level against the living observed party."""

        """Report whether every observed non-egg party member is restored."""

        if not state.party.is_known:
            return Fact(None, state.party.status)
        members = tuple(p for p in (state.party.value or ()) if not getattr(p, "is_egg", getattr(p, "egg", False)))
        return Fact.known(
            bool(members)
            and all(
                p.current_hp == getattr(p, "total_hp", getattr(p, "max_hp", None))
                and getattr(getattr(p, "status_condition", None), "value", getattr(p, "status", None)) == "none"
                for p in members
            )
        )

    return CampaignPredicate("party_fully_restored", "party is fully restored", evaluate)


def heal_party_objective() -> CampaignObjective:
    """Return the transient safety objective that restores the party."""

    return CampaignObjective(
        "HEAL_PARTY",
        "Restore the party",
        (),
        party_fully_restored(),
        execution_id="heal_party",
        task_kind="safety",
        priority=100,
    )


@dataclass(frozen=True, slots=True)
class CampaignObjective:
    """Declarative campaign task with predicates, dependencies, and execution metadata."""

    objective_id: str
    description: str
    prerequisites: tuple[CampaignPredicate, ...]
    completion: CampaignPredicate
    failure: CampaignPredicate | None = None
    execution_id: str | None = None
    tactical_target: Goal | None = None
    resource_policy: ResourceObjective | None = None
    # Kept on the existing objective model so execution/capabilities remain
    # compatible while callers can treat objectives as discoverable tasks.
    task_kind: str = "required"
    priority: int = 0
    destination: tuple[int, int] | None = None
    route_context: RouteContext | None = None
    observed: bool | None = None
    reachable: bool | None = None
    encounter_evaluation: EncounterEvaluation | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveSelection:
    """Result of resolving one campaign objective or task frontier."""

    objective: CampaignObjective | None
    status: ObjectiveStatus
    reason: str


@dataclass(frozen=True, slots=True)
class CampaignGoal:
    """A domain-level campaign target resolved through objective producers."""

    goal_id: str
    description: str
    completion: CampaignPredicate
    implemented_through: str | None = None


def ultimate_emerald_campaign_goal() -> CampaignGoal:
    """Return the ultimate Emerald goal represented by this campaign slice.

    The full Elite Four chain is not implemented yet.  Until its authoritative
    completion fact exists, the current slice is explicitly represented by its
    implemented endpoint rather than pretending the slice completes the full
    game.
    """
    return CampaignGoal(
        "beat_elite_four",
        "Beat the Elite Four",
        campaign_fact("first_badge_obtained"),
        implemented_through="defeat_roxanne",
    )


def campaign_objective_producers(
    objectives: tuple[CampaignObjective, ...] | None = None,
) -> dict[str, tuple[CampaignObjective, ...]]:
    """Index objectives by the observable predicate they establish.

    Completion predicates are the existing source of truth for what an
    objective produces.  Keeping this index derived avoids a second,
    manually-maintained producer table.
    """
    ordered = initial_emerald_campaign() if objectives is None else objectives
    producers: dict[str, list[CampaignObjective]] = {}
    for objective in ordered:
        producers.setdefault(objective.completion.predicate_id, []).append(objective)
    return {predicate_id: tuple(items) for predicate_id, items in producers.items()}


def current_area_is(area: str) -> CampaignPredicate:
    """Build a predicate requiring the observed canonical area name."""

    return CampaignPredicate(
        f"current_area_is:{area}",
        f"current canonical area is {area}",
        lambda state: (
            Fact.known(state.canonical_area.value == area)
            if state.canonical_area.is_known
            else Fact(None, state.canonical_area.status)
        ),
    )


def has_item(item: str, minimum: int = 1) -> CampaignPredicate:
    """Build a predicate requiring at least ``minimum`` copies of an item."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        """Compare the observed item quantity with the requested minimum."""

        quantity = state.item_quantity(item)
        if not quantity.is_known:
            return Fact(None, quantity.status)
        return Fact.known(quantity.value >= minimum)

    return CampaignPredicate(
        f"has_item:{item}:{minimum}",
        f"has at least {minimum} {item}",
        evaluate,
    )


def battle_completed_at(location: tuple[int, int], *, trainer: bool = True) -> CampaignPredicate:
    """Build a predicate for a successful battle at a specific map location."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        """Match the last completed battle against the requested boundary."""

        battle = state.last_completed_battle
        if not battle.is_known:
            return Fact(None, battle.status)
        if battle.value is None:
            return Fact.known(False)
        successful = battle.value.outcome.lower() in {"won", "caught"}
        return Fact.known(successful and battle.value.location == location and battle.value.is_trainer == trainer)

    return CampaignPredicate(
        f"battle_completed_at:{location}:{trainer}",
        f"completed a {'trainer' if trainer else 'wild'} battle at {location}",
        evaluate,
    )


def party_has_usable_pokemon() -> CampaignPredicate:
    """Build a predicate requiring one living, non-fainted party member."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        """Evaluate party usability from current party and rules projections."""

        if not state.party.is_known:
            return Fact(None, state.party.status)
        dead = state.dead_pokemon
        if not dead.is_known:
            return Fact(None, dead.status)
        usable = any(
            pokemon.current_hp > 0 and (pokemon.identity is None or pokemon.identity not in dead.value)
            for pokemon in state.party.value or ()
        )
        return Fact.known(usable)

    return CampaignPredicate("party_has_usable_pokemon", "party has a usable Pokémon", evaluate)


def party_meets_level_target(target_level: int) -> CampaignPredicate:
    """Require every currently living, non-egg party member to meet a level.

    Party levels are current-save observations, not durable campaign events.
    Dead Pokémon are excluded using the rules projection so a healed fainted
    Pokémon cannot make preparation appear complete or be selected for battle.
    """

    if target_level < 1:
        raise ValueError("level target must be positive")

    def evaluate(state: CampaignState) -> Fact[bool]:
        """Evaluate the target level against the living observed party."""

        if not state.party.is_known:
            return Fact(None, state.party.status)
        if not state.dead_pokemon.is_known:
            return Fact(None, state.dead_pokemon.status)
        members = tuple(
            pokemon
            for pokemon in state.party.value or ()
            if not getattr(pokemon, "egg", getattr(pokemon, "is_egg", False))
            and (
                getattr(pokemon, "identity", None) is None
                or pokemon.identity not in (state.dead_pokemon.value or frozenset())
            )
        )
        return Fact.known(bool(members) and all(pokemon.level >= target_level for pokemon in members))

    return CampaignPredicate(
        f"party_meets_level_target:{target_level}",
        f"all living party Pokémon are at least level {target_level}",
        evaluate,
    )


def encounter_available() -> CampaignPredicate:
    """Build a predicate for an eligible, unconsumed current-area encounter."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        """Evaluate current-area encounter availability without guessing."""

        encounter = state.current_encounter
        if not encounter.is_known:
            return Fact(None, encounter.status)
        if encounter.value.status == "unknown":
            return Fact.unknown()
        return Fact.known(encounter.value.status == "none" and encounter.value.eligible)

    return CampaignPredicate("encounter_available", "current area has an unused encounter", evaluate)


def campaign_fact(name: str) -> CampaignPredicate:
    """Build a predicate backed by a registered ROM-observed campaign fact."""

    return CampaignPredicate(
        f"campaign_fact:{name}",
        name.replace("_", " "),
        lambda state: evaluate_emerald_fact(state, name),
    )


def initial_emerald_campaign() -> tuple[CampaignObjective, ...]:
    """Return Emerald's ordered, declarative early campaign slice."""

    return (
        CampaignObjective(
            objective_id="set_text_speed",
            description="Set text speed to Fast",
            prerequisites=(),
            completion=campaign_fact("text_speed_fast"),
            execution_id="set_text_speed",
        ),
        CampaignObjective(
            objective_id="complete_new_game_setup",
            description="Complete the introductory name, gender, and house setup",
            prerequisites=(campaign_fact("text_speed_fast"),),
            completion=campaign_fact("new_game_setup_complete"),
            execution_id="new_game_setup",
        ),
        CampaignObjective(
            objective_id="set_wall_clock",
            description="Set the wall clock",
            prerequisites=(campaign_fact("new_game_setup_complete"),),
            completion=campaign_fact("wall_clock_set"),
            execution_id="set_wall_clock",
        ),
        CampaignObjective(
            objective_id="meet_rival",
            description="Meet the rival",
            prerequisites=(campaign_fact("wall_clock_set"),),
            completion=campaign_fact("rival_met"),
            execution_id="meet_rival",
        ),
        CampaignObjective(
            objective_id="rescue_birch",
            description="Rescue Professor Birch",
            prerequisites=(campaign_fact("rival_met"),),
            completion=campaign_fact("birch_rescued"),
            execution_id="rescue_birch",
        ),
        CampaignObjective(
            objective_id="obtain_starter",
            description="Obtain the starter Pokémon",
            prerequisites=(campaign_fact("birch_rescued"),),
            completion=campaign_fact("starter_obtained"),
            execution_id="obtain_starter",
        ),
        CampaignObjective(
            objective_id="complete_intro_rival",
            description="Complete the introductory rival battle",
            # The rival objective needs a living battle-capable party. Full
            # restoration remains an explicit recovery objective, but a small
            # amount of ordinary post-Birch damage must not force a Center
            # visit before the ROM battle begins.
            prerequisites=(campaign_fact("starter_obtained"), party_has_usable_pokemon()),
            completion=campaign_fact("intro_rival_battle_complete"),
            execution_id="intro_rival",
            tactical_target=introductory_rival_goal(),
            resource_policy=ResourceObjective(
                "complete_intro_rival",
                readiness=ReadinessImportance.IMPORTANT,
                encounters=EncounterPolicy.PRESERVE,
                mandatory_battle=True,
                recover_before_completion=True,
            ),
            destination=MapRSE.ROUTE103.value,
        ),
        CampaignObjective(
            objective_id="receive_pokedex",
            description="Receive the Pokédex",
            prerequisites=(campaign_fact("intro_rival_battle_complete"),),
            completion=campaign_fact("pokedex_received"),
            execution_id="receive_pokedex",
            destination=MapRSE.LITTLEROOT_TOWN.value,
        ),
        CampaignObjective(
            objective_id="receive_pokeballs",
            description="Receive Poké Balls",
            prerequisites=(campaign_fact("pokedex_received"),),
            completion=campaign_fact("pokeballs_ready"),
            execution_id="receive_pokeballs",
            tactical_target=early_pokeball_goal(),
            destination=MapRSE.LITTLEROOT_TOWN.value,
        ),
        CampaignObjective(
            objective_id="reach_petalburg",
            description="Reach Petalburg City after receiving the Pokédex",
            # The Pokédex opens the ROM route west from Oldale.  Poké Balls
            # are a campaign safety prerequisite so the first encounter
            # opportunity is not intentionally passed while the party cannot
            # catch it; they are not being misrepresented as a ROM map gate.
            prerequisites=(campaign_fact("pokedex_received"), campaign_fact("pokeballs_ready")),
            completion=campaign_fact("visited_petalburg"),
            execution_id="reach_petalburg",
            destination=MapRSE.PETALBURG_CITY.value,
        ),
        CampaignObjective(
            objective_id="complete_petalburg_wally",
            description="Talk to Norman and complete Wally's tutorial",
            prerequisites=(campaign_fact("visited_petalburg"),),
            completion=campaign_fact("petalburg_wally_scene_complete"),
            execution_id="complete_petalburg_wally",
            destination=MapRSE.PETALBURG_CITY_GYM.value,
        ),
        CampaignObjective(
            objective_id="complete_petalburg_woods",
            description="Complete the Petalburg Woods Devon researcher scene",
            prerequisites=(campaign_fact("petalburg_wally_scene_complete"),),
            completion=campaign_fact("petalburg_woods_scene_complete"),
            execution_id="complete_petalburg_woods",
            destination=MapRSE.PETALBURG_WOODS.value,
        ),
        CampaignObjective(
            objective_id="reach_rustboro",
            description="Reach Rustboro City after the Petalburg Woods progression",
            prerequisites=(campaign_fact("petalburg_woods_scene_complete"),),
            completion=campaign_fact("visited_rustboro"),
            execution_id="reach_rustboro",
            destination=MapRSE.RUSTBORO_CITY.value,
        ),
        CampaignObjective(
            objective_id="complete_rustboro_goods_stolen",
            description="Complete the Rustboro Devon Goods theft scene",
            prerequisites=(campaign_fact("visited_rustboro"),),
            completion=campaign_fact("devon_goods_stolen"),
            execution_id="complete_rustboro_goods_stolen",
            destination=MapRSE.RUSTBORO_CITY.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="report_devon_goods",
            description="Report the stolen Devon Goods to the Devon employee",
            prerequisites=(campaign_fact("devon_goods_stolen"),),
            completion=campaign_fact("devon_goods_reported"),
            execution_id="report_devon_goods",
            destination=MapRSE.RUSTBORO_CITY.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="recover_devon_goods",
            description="Recover the Devon Goods from the Rusturf Tunnel grunt",
            prerequisites=(campaign_fact("devon_goods_stolen"),),
            completion=campaign_fact("devon_goods_recovered"),
            execution_id="recover_devon_goods",
            destination=MapRSE.RUSTURF_TUNNEL.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="return_devon_goods",
            description="Return the Devon Goods to the Rustboro employee",
            prerequisites=(campaign_fact("devon_goods_recovered"),),
            completion=campaign_fact("devon_goods_returned"),
            execution_id="return_devon_goods",
            destination=MapRSE.RUSTBORO_CITY.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="meet_mr_stone",
            description="Meet Mr. Stone in Devon Corporation",
            prerequisites=(campaign_fact("devon_goods_returned"),),
            completion=campaign_fact("devon_corp_3f_scene_complete"),
            execution_id="meet_mr_stone",
            destination=MapRSE.RUSTBORO_CITY_DEVON_CORP_3F.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="prepare_roxanne",
            description="Prepare the living party for Roxanne",
            prerequisites=(campaign_fact("visited_rustboro"),),
            completion=party_meets_level_target(emerald_bosses()[0].preparation_level),
            execution_id="prepare_roxanne",
            destination=MapRSE.ROUTE116.value,
        ),
        CampaignObjective(
            objective_id="defeat_roxanne",
            description="Defeat Roxanne and earn the first badge",
            # Preparation is a real executable dependency, not merely an
            # available side task.  Reusing the same predicate ID lets the
            # dependency planner resolve the preparation producer when the
            # live party is below the registered target, while allowing this
            # prerequisite to pass directly once the target is met.
            prerequisites=(
                party_meets_level_target(emerald_bosses()[0].preparation_level),
                campaign_fact("visited_rustboro"),
            ),
            completion=campaign_fact("first_badge_obtained"),
            execution_id="defeat_roxanne",
            destination=MapRSE.RUSTBORO_CITY_GYM.value,
            resource_policy=ResourceObjective(
                "defeat_roxanne",
                readiness=ReadinessImportance.IMPORTANT,
                encounters=EncounterPolicy.PRESERVE,
                mandatory_battle=True,
                recover_before_completion=True,
                minimum_hp_ratio=0.75,
            ),
        ),
    )


def encounter_task(
    location: tuple[int, int], *, name: str | None = None, observed: bool | None = None
) -> CampaignObjective:
    """Describe an opportunistic encounter without choosing encounter policy."""
    task_id = name or f"obtain_encounter:{location[0]}:{location[1]}"

    def accessible(state: CampaignState) -> Fact[bool]:
        """Report whether this location still has an eligible encounter."""

        if not state.raw_map.is_known or not state.encounters.is_known:
            return Fact(None, state.raw_map.status if not state.raw_map.is_known else state.encounters.status)
        encounter = state.encounter_for(location)
        return Fact.known(encounter.value.status == "none" and encounter.value.eligible)

    return CampaignObjective(
        task_id,
        f"Obtain the first encounter at {location}",
        (CampaignPredicate(f"encounter_accessible:{location}", "location is accessible and unconsumed", accessible),),
        CampaignPredicate(
            f"encounter_complete:{location}",
            "encounter is consumed",
            lambda state: (
                Fact.known(state.encounter_for(location).value.status != "none")
                if state.encounter_for(location).is_known
                else Fact(None, state.encounter_for(location).status)
            ),
        ),
        task_kind="optional",
        priority=100,
        destination=location,
        # Encounter acquisition is a reusable tactical navigation request;
        # the selector decides whether this optional task is worth mounting.
        tactical_target=NavigationGoal(ReachLocation(location), encounter_mode=EncounterMode.SEEK),
        observed=observed,
    )


def measure_route_context(
    state: CampaignState,
    task: CampaignObjective,
    progression_destination: tuple[int, int] | None,
    *,
    graph: WorldMapGraph | None = None,
) -> RouteContext | None:
    """Measure map-level route relationship using the existing world graph."""
    if not state.raw_map.is_known or task.destination is None:
        return None
    current = state.raw_map.value
    if progression_destination is None:
        return RouteContext(current, None, task.destination, None, None, None, RouteRelation.UNKNOWN)
    try:
        route_graph = graph or get_world_map_graph()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return RouteContext(current, progression_destination, task.destination, None, None, None, RouteRelation.UNKNOWN)
    try:
        direct = route_graph.route(current, progression_destination).estimated_cost
        to_task = route_graph.route(current, task.destination).estimated_cost
        onward = route_graph.route(task.destination, progression_destination).estimated_cost
    except WorldNavigationError:
        return RouteContext(
            current, progression_destination, task.destination, None, None, None, RouteRelation.UNREACHABLE
        )
    via = to_task + onward
    detour = max(0, via - direct)
    relation = RouteRelation.ON_ROUTE if detour == 0 else RouteRelation.DETOUR
    return RouteContext(current, progression_destination, task.destination, direct, via, detour, relation)


def evaluate_encounter_opportunity(
    state: CampaignState,
    opportunity: EncounterOpportunity,
    progression_destination: tuple[int, int] | None,
    *,
    task_id: str | None = None,
    policy: EncounterEvaluationPolicy = EncounterEvaluationPolicy(),
    graph: WorldMapGraph | None = None,
) -> EncounterEvaluation:
    """Purely evaluate encounter logistics; never changes the projection."""
    task_id = task_id or f"obtain_encounter:{opportunity.location[0]}:{opportunity.location[1]}"
    base = dict(
        task_id=task_id,
        location=opportunity.location,
        progression_destination=progression_destination,
    )
    if opportunity.consumed or not opportunity.eligible:
        return EncounterEvaluation(
            **base,
            direct_progression_cost=None,
            encounter_cost=None,
            encounter_to_progression_cost=None,
            via_encounter_cost=None,
            detour_cost=None,
            detour_ratio=None,
            classification=EncounterClassification.UNAVAILABLE,
            recommendation=EncounterRecommendation.UNAVAILABLE,
        )
    if not state.raw_map.is_known or progression_destination is None:
        return EncounterEvaluation(
            **base,
            direct_progression_cost=None,
            encounter_cost=None,
            encounter_to_progression_cost=None,
            via_encounter_cost=None,
            detour_cost=None,
            detour_ratio=None,
            classification=EncounterClassification.UNKNOWN,
            recommendation=EncounterRecommendation.UNKNOWN,
        )
    try:
        route_graph = graph or get_world_map_graph()
        direct = route_graph.route(state.raw_map.value, progression_destination).estimated_cost
        to_encounter = route_graph.route(state.raw_map.value, opportunity.location).estimated_cost
        onward = route_graph.route(opportunity.location, progression_destination).estimated_cost
    except (WorldNavigationError, AttributeError, RuntimeError, TypeError, ValueError):
        return EncounterEvaluation(
            **base,
            direct_progression_cost=None,
            encounter_cost=None,
            encounter_to_progression_cost=None,
            via_encounter_cost=None,
            detour_cost=None,
            detour_ratio=None,
            classification=EncounterClassification.UNREACHABLE,
            recommendation=EncounterRecommendation.UNAVAILABLE,
        )
    via = to_encounter + onward
    detour = max(0, via - direct)
    ratio = detour / direct if direct else (0.0 if detour == 0 else float("inf"))
    if detour == 0:
        classification = EncounterClassification.ON_ROUTE
        recommendation = EncounterRecommendation.PREFER
    elif detour <= policy.small_detour_cost or ratio <= policy.small_detour_ratio:
        classification = EncounterClassification.SMALL_DETOUR
        recommendation = EncounterRecommendation.PREFER
    else:
        classification = EncounterClassification.LARGE_DETOUR
        recommendation = EncounterRecommendation.DEFER
    return EncounterEvaluation(
        **base,
        classification=classification,
        recommendation=recommendation,
        direct_progression_cost=direct,
        encounter_cost=to_encounter,
        encounter_to_progression_cost=onward,
        via_encounter_cost=via,
        detour_cost=detour,
        detour_ratio=ratio,
    )


def progression_destination(
    tasks: tuple[CampaignObjective, ...],
) -> tuple[tuple[int, int] | None, str]:
    """Return the sole required spatial anchor, or explain ambiguity."""
    destinations = tuple(
        dict.fromkeys(
            task.destination for task in tasks if task.task_kind == "required" and task.destination is not None
        )
    )
    if len(destinations) == 1:
        return destinations[0], "single available required destination"
    if not destinations:
        return None, "no available required destination"
    return None, "multiple available required destinations"


def available_campaign_tasks(
    state: CampaignState,
    tasks: tuple[CampaignObjective, ...] | None = None,
    encounter_locations: tuple[tuple[int, int], ...] | None = None,
) -> tuple[CampaignObjective, ...]:
    """Enumerate ready tasks; unavailable/unknown tasks are omitted."""
    candidates = initial_emerald_campaign() if tasks is None else tasks
    if tasks is None and state.encounters.is_known:
        # The rules projection is deliberately the source of truth here.  It
        # knows only locations it has observed, so discovery must not invent
        # map encounters from the ROM's wild-data tables.
        candidates += tuple(
            encounter_task(encounter.location)
            for encounter in state.encounters.value or ()
            if encounter.eligible and encounter.status == "none"
        )
        try:
            opportunities = encounter_opportunities(state, encounter_locations)
        except (AttributeError, NameError, RuntimeError, TypeError, ValueError):
            opportunities = ()
        candidates += tuple(
            encounter_task(item.location, observed=item.observed)
            for item in opportunities
            if item.eligible and not item.consumed
        )
        # De-duplicate observed locations when the compatibility fallback
        # above and the world catalog describe the same task.
        candidates = tuple({task.objective_id: task for task in candidates}.values())
    available: list[CampaignObjective] = []
    for task in candidates:
        completion = task.completion.evaluate(state)
        if completion.status is not FactStatus.KNOWN or completion.value:
            continue
        if task.failure is not None:
            failure = task.failure.evaluate(state)
            if failure.status is not FactStatus.KNOWN or failure.value:
                continue
        results = tuple(prerequisite.evaluate(state) for prerequisite in task.prerequisites)
        if all(result.status is FactStatus.KNOWN and result.value for result in results):
            available.append(task)
    progression, _ = progression_destination(tuple(available))
    result = []
    for task in available:
        route = measure_route_context(state, task, progression) if task.destination is not None else None
        evaluation = task.encounter_evaluation
        if task.task_kind == "optional" and task.destination is not None:
            fact = state.encounter_for(task.destination)
            opportunity = EncounterOpportunity(
                task.destination,
                task.observed if task.observed is not None else fact.is_known,
                fact.value.status != "none" if fact.is_known else False,
                fact.value.eligible if fact.is_known else True,
            )
            evaluation = evaluate_encounter_opportunity(state, opportunity, progression, task_id=task.objective_id)
        result.append(replace(task, route_context=route, encounter_evaluation=evaluation))
    return tuple(result)


def _select_available_campaign_tasks(available: tuple[CampaignObjective, ...]) -> ObjectiveSelection:
    """Select the deterministic best task from one discovery result."""

    required = tuple(task for task in available if task.task_kind == "required")
    if required:
        # Required progression remains the primary task. Preferred encounter
        # work is retained on the same discovery result as opportunistic work.
        task = required[0]
        return ObjectiveSelection(task, ObjectiveStatus.READY, "selected required campaign progression task")
    selectable = tuple(
        task
        for task in available
        if task.task_kind != "optional"
        or (
            task.encounter_evaluation is not None
            and task.encounter_evaluation.recommendation is EncounterRecommendation.PREFER
        )
    )
    if selectable:
        task = min(
            selectable,
            key=lambda item: (
                item.encounter_evaluation.detour_cost if item.encounter_evaluation else float("inf"),
                item.encounter_evaluation.encounter_cost if item.encounter_evaluation else float("inf"),
                item.objective_id,
            ),
        )
        return ObjectiveSelection(task, ObjectiveStatus.READY, "selected from currently available tasks")
    return ObjectiveSelection(None, ObjectiveStatus.COMPLETE, "no campaign task is currently available")


def select_available_campaign_task(
    state: CampaignState,
    tasks: tuple[CampaignObjective, ...] | None = None,
    encounter_locations: tuple[tuple[int, int], ...] | None = None,
) -> ObjectiveSelection:
    """Select the deterministic best currently available task.

    Priority is intentionally simple and replaceable: required tasks precede
    optional tasks, then explicit priority and declaration order decide ties.
    """

    return _select_available_campaign_tasks(available_campaign_tasks(state, tasks, encounter_locations))


def campaign_task_diagnostics(
    state: CampaignState,
    tasks: tuple[CampaignObjective, ...] | None = None,
    encounter_locations: tuple[tuple[int, int], ...] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return structured, read-only diagnostics for currently available tasks."""
    available = available_campaign_tasks(state, tasks, encounter_locations)
    progression, progression_reason = progression_destination(available)
    # Selection must use the same discovery result as the rows below.  The
    # discovery pass performs route and encounter analysis; recomputing it
    # here doubled that cost on every controller refresh.
    selection = _select_available_campaign_tasks(available)
    preferred = sorted(
        (
            task
            for task in available
            if task.task_kind == "optional"
            and task.encounter_evaluation is not None
            and task.encounter_evaluation.recommendation is EncounterRecommendation.PREFER
        ),
        key=lambda item: (
            item.encounter_evaluation.detour_cost,
            item.encounter_evaluation.encounter_cost,
            item.objective_id,
        ),
    )
    ranks = {task.objective_id: index for index, task in enumerate(preferred, 1)}
    rows = []
    for task in available:
        route = task.route_context
        encounter = state.current_encounter.value if state.current_encounter.is_known else None
        rows.append(
            {
                "task_id": task.objective_id,
                "kind": task.task_kind,
                "required": task.task_kind == "required",
                "priority": task.priority,
                "selection_rank": ranks.get(task.objective_id),
                "selected": selection.objective is not None and selection.objective.objective_id == task.objective_id,
                "selected_task": selection.objective.objective_id if selection.objective else None,
                "destination": task.destination,
                "observed": task.observed,
                "prerequisites": tuple(
                    {"id": p.predicate_id, "satisfied": p.evaluate(state).value} for p in task.prerequisites
                ),
                "encounter": (
                    {
                        "location": task.destination,
                        "eligible": state.encounter_for(task.destination).value.eligible,
                        "consumed": state.encounter_for(task.destination).value.status != "none",
                    }
                    if task.task_kind == "optional"
                    and task.objective_id.startswith("obtain_encounter:")
                    and task.destination is not None
                    and state.encounter_for(task.destination).is_known
                    else None
                ),
                "progression_destination": progression,
                "progression_reason": progression_reason,
                "route_relation": route.relation.value if route else None,
                "direct_distance": route.direct_distance if route else None,
                "via_task_distance": route.via_task_distance if route else None,
                "detour_distance": route.detour_distance if route else None,
                "current_encounter": encounter.status if encounter else None,
                "encounter_evaluation": (
                    {
                        "classification": task.encounter_evaluation.classification.value,
                        "recommendation": task.encounter_evaluation.recommendation.value,
                        "direct_progression_cost": task.encounter_evaluation.direct_progression_cost,
                        "encounter_cost": task.encounter_evaluation.encounter_cost,
                        "encounter_to_progression_cost": task.encounter_evaluation.encounter_to_progression_cost,
                        "via_encounter_cost": task.encounter_evaluation.via_encounter_cost,
                        "detour_cost": task.encounter_evaluation.detour_cost,
                        "detour_ratio": task.encounter_evaluation.detour_ratio,
                    }
                    if task.encounter_evaluation
                    else None
                ),
            }
        )
    return tuple(rows)


def select_campaign_objective(
    state: CampaignState,
    objectives: tuple[CampaignObjective, ...] | None = None,
) -> ObjectiveSelection:
    """Select the first non-complete objective in deterministic order."""

    if state.run_status.status is not FactStatus.KNOWN:
        return ObjectiveSelection(None, ObjectiveStatus.UNKNOWN, "run status is unavailable")
    if state.run_status.value is RunStatus.LOST:
        return ObjectiveSelection(None, ObjectiveStatus.FAILED, "run is lost")

    legal = state.run_is_legal()
    if legal.status is not FactStatus.KNOWN:
        return ObjectiveSelection(None, ObjectiveStatus.UNKNOWN, "run legality is unavailable")
    if not legal.value:
        return ObjectiveSelection(None, ObjectiveStatus.FAILED, "run is not legal")

    ordered = initial_emerald_campaign() if objectives is None else objectives
    # Optional story work is discoverable through ``available_campaign_tasks``
    # but must not block the executable required path to the campaign goal.
    for objective in ordered:
        if objective.task_kind == "optional":
            continue
        if objective.failure is not None:
            failure = objective.failure.evaluate(state)
            if failure.status is not FactStatus.KNOWN:
                return ObjectiveSelection(
                    objective, ObjectiveStatus.UNKNOWN, f"failure state is {failure.status.value}"
                )
            if failure.value:
                return ObjectiveSelection(objective, ObjectiveStatus.FAILED, "objective failure predicate is true")

        completion = objective.completion.evaluate(state)
        if completion.status is not FactStatus.KNOWN:
            return ObjectiveSelection(objective, ObjectiveStatus.UNKNOWN, f"completion is {completion.status.value}")
        if completion.value:
            continue

        for prerequisite in objective.prerequisites:
            result = prerequisite.evaluate(state)
            if result.status is not FactStatus.KNOWN:
                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.UNKNOWN,
                    f"prerequisite {prerequisite.predicate_id} is {result.status.value}",
                )
            if not result.value:
                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.BLOCKED,
                    f"prerequisite {prerequisite.predicate_id} is false",
                )
        return ObjectiveSelection(objective, ObjectiveStatus.READY, "all prerequisites are satisfied")

    return ObjectiveSelection(None, ObjectiveStatus.COMPLETE, "all campaign objectives are complete")


def plan_campaign(
    state: CampaignState,
    objectives: tuple[CampaignObjective, ...] | None = None,
    encounter_locations: tuple[tuple[int, int], ...] | None = None,
    goal: CampaignGoal | None = None,
) -> ObjectiveSelection:
    """Resolve the executable frontier for the campaign's domain goal.

    Planning starts at the goal completion predicate and walks backwards
    through objective prerequisites.  Objective tuple order is not consulted;
    when multiple producers exist, objective ID provides a deterministic,
    domain-neutral tie-breaker.

    This function deliberately does not inspect ``GameState`` or classify
    emulator phases.  Menu, dialogue, overworld, and battle observations are
    execution concerns for the selected objective's capability.
    """
    ordered = initial_emerald_campaign() if objectives is None else objectives

    # A title/main-menu observation is a valid opening boundary, not a
    # completed or failed campaign.  Save-backed facts are intentionally
    # unavailable there because mGBA can expose an all-0xff placeholder save;
    # select the opening capability directly so the planner does not stall on
    # an unavailable terminal predicate.  This is lifecycle dispatch, not a
    # remembered phase: the next frame re-evaluates the ROM observation.
    if (
        getattr(state, "campaign_lifecycle", CampaignObservationLifecycle.UNAVAILABLE)
        is CampaignObservationLifecycle.FRESH_START
    ):
        startup = next((item for item in ordered if item.objective_id == "set_text_speed"), None)
        if startup is not None:
            return ObjectiveSelection(startup, ObjectiveStatus.READY, "fresh ROM startup requires opening setup")

    # ``CampaignProgressionMode`` uses this dependency-aware planner instead
    # of the legacy tuple-order selector.  Keep the terminal Nuzlocke rule at
    # this boundary too: a durable WhiteoutOccurred must prevent a restarted
    # process from mounting campaign movement against an older save-state.
    if state.run_status.status is not FactStatus.KNOWN:
        return ObjectiveSelection(None, ObjectiveStatus.UNKNOWN, "run status is unavailable")
    if state.run_status.value is RunStatus.LOST:
        return ObjectiveSelection(None, ObjectiveStatus.FAILED, "run is lost")

    legal = state.run_is_legal()
    if legal.status is not FactStatus.KNOWN:
        return ObjectiveSelection(None, ObjectiveStatus.UNKNOWN, "run legality is unavailable")
    if not legal.value:
        return ObjectiveSelection(None, ObjectiveStatus.FAILED, "run is not legal")

    target = goal or ultimate_emerald_campaign_goal()
    producers = campaign_objective_producers(ordered)
    # Party restoration is an executable safety dependency, not a public
    # campaign milestone.  Keep it out of ``initial_emerald_campaign`` (and
    # therefore preserve its stable public ordering), while making the
    # dependency-aware planner able to resolve it before the first legal
    # encounter.
    restoration = heal_party_objective()
    producers.setdefault(restoration.completion.predicate_id, (restoration,))
    completion = target.completion.evaluate(state)

    if completion.status is not FactStatus.KNOWN:
        return ObjectiveSelection(
            None,
            ObjectiveStatus.UNKNOWN,
            f"campaign goal completion is {completion.status.value}",
        )
    if completion.value:
        return ObjectiveSelection(
            None,
            ObjectiveStatus.COMPLETE,
            f"campaign goal {target.goal_id} is complete",
        )

    visiting: set[str] = set()

    def resolve(objective: CampaignObjective) -> ObjectiveSelection:
        """Resolve one objective recursively through its prerequisite producers."""

        if objective.objective_id in visiting:
            return ObjectiveSelection(
                objective,
                ObjectiveStatus.BLOCKED,
                f"dependency cycle detected at {objective.objective_id}",
            )
        visiting.add(objective.objective_id)
        try:
            completed = objective.completion.evaluate(state)
            if completed.status is not FactStatus.KNOWN:
                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.UNKNOWN,
                    f"completion is {completed.status.value}",
                )
            if completed.value:
                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.COMPLETE,
                    f"objective {objective.objective_id} is complete",
                )

            if objective.failure is not None:
                failure = objective.failure.evaluate(state)
                if failure.status is not FactStatus.KNOWN:
                    return ObjectiveSelection(
                        objective,
                        ObjectiveStatus.UNKNOWN,
                        f"failure state is {failure.status.value}",
                    )
                if failure.value:
                    return ObjectiveSelection(
                        objective,
                        ObjectiveStatus.FAILED,
                        "objective failure predicate is true",
                    )

            for prerequisite in objective.prerequisites:
                result = prerequisite.evaluate(state)
                if result.status is not FactStatus.KNOWN:
                    return ObjectiveSelection(
                        objective,
                        ObjectiveStatus.UNKNOWN,
                        f"prerequisite {prerequisite.predicate_id} is {result.status.value}",
                    )
                if result.value:
                    continue

                candidates = sorted(
                    producers.get(prerequisite.predicate_id, ()),
                    key=lambda item: item.objective_id,
                )
                if not candidates:
                    return ObjectiveSelection(
                        objective,
                        ObjectiveStatus.BLOCKED,
                        f"no producer for unmet prerequisite {prerequisite.predicate_id}",
                    )

                unknown_producer: ObjectiveSelection | None = None
                blocked_producer: ObjectiveSelection | None = None
                for producer in candidates:
                    candidate = resolve(producer)
                    if candidate.status is ObjectiveStatus.READY:
                        return candidate
                    if candidate.status is ObjectiveStatus.UNKNOWN:
                        unknown_producer = candidate
                    if candidate.status is ObjectiveStatus.BLOCKED:
                        blocked_producer = candidate

                if unknown_producer is not None:
                    return unknown_producer
                if blocked_producer is not None:
                    return blocked_producer

                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.BLOCKED,
                    f"no executable producer for unmet prerequisite {prerequisite.predicate_id}",
                )

            return ObjectiveSelection(
                objective,
                ObjectiveStatus.READY,
                f"resolved executable frontier for campaign goal {target.goal_id}",
            )
        finally:
            visiting.remove(objective.objective_id)

    goal_producers = sorted(
        producers.get(target.completion.predicate_id, ()),
        key=lambda item: item.objective_id,
    )
    if not goal_producers:
        return ObjectiveSelection(
            None,
            ObjectiveStatus.BLOCKED,
            f"no producer for incomplete campaign goal {target.goal_id}",
        )

    unknown_producer: ObjectiveSelection | None = None
    blocked_producer: ObjectiveSelection | None = None
    for producer in goal_producers:
        result = resolve(producer)
        if result.status is ObjectiveStatus.READY:
            return result
        if result.status is ObjectiveStatus.UNKNOWN:
            unknown_producer = result
        if result.status is ObjectiveStatus.BLOCKED:
            blocked_producer = result

    if unknown_producer is not None:
        return unknown_producer
    if blocked_producer is not None:
        return blocked_producer

    return ObjectiveSelection(
        goal_producers[0],
        ObjectiveStatus.BLOCKED,
        f"no executable producer for campaign goal {target.goal_id}",
    )
