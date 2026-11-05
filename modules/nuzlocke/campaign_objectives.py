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
    GoalConstraints,
    NavigationGoal,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    introductory_rival_goal,
)
from modules.map_data import MapRSE
from modules.console import diagnostic_print
from modules.context import context

from .campaign_state import CampaignState, Fact, FactStatus, RunStatus
from .resource_policy import (
    EncounterPolicy,
    PokeballRestockPolicy,
    ReadinessImportance,
    RecoverySupplyPolicy,
    ResourceObjective,
)
from modules.overworld import static_map_transitions, WorldTransition
from modules.world_navigation import (
    WorldMapGraph,
    WorldNavigationError,
    get_world_map_graph,
    route_with_edge_filter,
)
from .encounter_catalog import (
    EncounterOpportunity,
    campaign_encounter_methods,
    encounter_candidates_for_location,
    encounter_opportunities,
)
from .capture_policy import (
    CapturePolicyContext,
    EncounterCandidate,
    EncounterMethod,
    EncounterPortfolioContext,
    choose_encounter_portfolio,
    score_encounter,
    score_portfolio_candidate,
)
from .trainer_policy import TrainerPolicyInput, choose_trainer_mode
from .emerald_campaign_registry import emerald_bosses, evaluate_emerald_fact
from .snapshots import CampaignObservationLifecycle
from .campaign_reachability import (
    CampaignReachabilityStatus,
    assess_campaign_reachability,
)


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
    # A positive value means that the encounter's known species pool adds
    # offensive coverage against the next unavoidable trainer checkpoint.
    # This is intentionally separate from route geometry: a useful capture
    # can justify a detour that would otherwise be deferred.
    trainer_preparation_gain: float = 0.0
    trainer_checkpoint_id: str | None = None
    # Static route cost and conditioned encounter utility may still be useful
    # for a future objective, but an unresolved dynamic gate must prevent the
    # optional task from taking execution ownership now.
    reachability_status: CampaignReachabilityStatus = CampaignReachabilityStatus.REACHABLE_NOW
    reachability_blockers: tuple[str, ...] = ()

    @property
    def executable_now(self) -> bool:
        """Return whether encounter policy may mount this task immediately."""

        return self.reachability_status is CampaignReachabilityStatus.REACHABLE_NOW


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
    # The objective is location-level until strategic selection resolves a
    # concrete ROM encounter method.  Keeping the method here lets execution
    # validate that policy and tactical behavior agree.
    encounter_method: EncounterMethod | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveSelection:
    """Result of resolving one campaign objective or task frontier."""

    objective: CampaignObjective | None
    status: ObjectiveStatus
    reason: str
    # Populated by the executable-task selector when it has already performed
    # discovery. Consumers such as the Rich frontier can render the exact
    # executable set without running route analysis a second time.
    frontier_objective_ids: tuple[str, ...] = ()
    # Structured evidence for the human-facing frontier report. These are
    # immutable because the selector has already done the expensive route and
    # encounter work by the time a selection is published.
    selection_diagnostics: tuple[dict[str, object], ...] = ()
    # Ordered encounter candidates produced by the same policy call that
    # selected the optional task.  This is kept separate from per-task
    # tradeoffs so the Rich console can show the portfolio order explicitly.
    encounter_portfolio: tuple[dict[str, object], ...] = ()


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


def birch_lab_handoff_complete() -> CampaignPredicate:
    """Return the completion boundary for Birch's combined lab handoff."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        # Emerald sets the Pokédex flag before the same ROM-owned dialogue
        # grants the initial Poké Balls. Keep both facts available to the
        # planner, but let one capability own the complete script.
        pokedex = state.campaign_facts.pokedex_received
        pokeballs = state.campaign_facts.pokeballs_received
        if pokedex.is_known and pokeballs.is_known:
            return Fact.known(bool(pokedex.value and pokeballs.value))
        if (pokedex.is_known and pokedex.value is False) or (
            pokeballs.is_known and pokeballs.value is False
        ):
            return Fact.known(False)
        return Fact(None, pokedex.status if not pokedex.is_known else pokeballs.status)

    return CampaignPredicate(
        "campaign_fact:birch_lab_handoff_complete",
        "Birch's Pokédex and Poké Ball handoff is complete",
        evaluate,
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
            ),
            destination=MapRSE.ROUTE103.value,
        ),
        CampaignObjective(
            objective_id="receive_pokedex",
            description="Receive the Pokédex and initial Poké Balls",
            prerequisites=(campaign_fact("intro_rival_battle_complete"),),
            # The Pokédex flag becomes true before Birch finishes the same
            # script by handing over the initial Poké Balls. Keep this one
            # capability mounted until the final lab state.
            completion=birch_lab_handoff_complete(),
            execution_id="receive_pokedex",
            destination=MapRSE.LITTLEROOT_TOWN.value,
        ),
        CampaignObjective(
            objective_id="complete_petalburg_wally",
            description="Talk to Norman and complete Wally's tutorial",
            # Petalburg arrival is a consequence of pursuing this scene. The
            # Birch handoff is the actual prerequisite that opens the route;
            # ``visited_petalburg`` remains an observed ROM fact and is not a
            # separate campaign milestone.
            prerequisites=(birch_lab_handoff_complete(),),
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
            resource_policy=ResourceObjective(
                "complete_petalburg_woods",
                readiness=ReadinessImportance.IMPORTANT,
                encounters=EncounterPolicy.PRESERVE,
                mandatory_battle=True,
            ),
            destination=MapRSE.PETALBURG_WOODS.value,
        ),
        CampaignObjective(
            objective_id="complete_rustboro_goods_stolen",
            description="Complete the Rustboro Devon Goods theft scene",
            # Rustboro arrival is a consequence of pursuing this ROM-owned
            # scene. The scene's completion flag, rather than the map visit,
            # is the progression boundary.
            prerequisites=(campaign_fact("petalburg_woods_scene_complete"),),
            completion=campaign_fact("devon_goods_stolen"),
            execution_id="complete_rustboro_goods_stolen",
            destination=MapRSE.RUSTBORO_CITY.value,
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
            resource_policy=ResourceObjective(
                "recover_devon_goods",
                readiness=ReadinessImportance.IMPORTANT,
                encounters=EncounterPolicy.PRESERVE,
                mandatory_battle=True,
            ),
            destination=MapRSE.RUSTURF_TUNNEL.value,
        ),
        CampaignObjective(
            objective_id="return_devon_goods",
            description="Return the Devon Goods to the Rustboro employee",
            prerequisites=(campaign_fact("devon_goods_recovered"), campaign_fact("visited_rustboro")),
            completion=campaign_fact("devon_goods_returned"),
            execution_id="return_devon_goods",
            destination=MapRSE.RUSTBORO_CITY.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="meet_mr_stone",
            description="Meet Mr. Stone in Devon Corporation",
            prerequisites=(campaign_fact("devon_goods_returned"), campaign_fact("visited_rustboro")),
            completion=campaign_fact("devon_corp_3f_scene_complete"),
            execution_id="meet_mr_stone",
            destination=MapRSE.RUSTBORO_CITY_DEVON_CORP_3F.value,
            task_kind="optional",
        ),
        CampaignObjective(
            objective_id="prepare_roxanne",
            description="Prepare the living party for Roxanne",
            # Rustboro arrival is an observed side effect of the Devon Goods
            # corridor. The recovered-goods boundary is the durable story
            # prerequisite that opens the gym preparation route.
            prerequisites=(campaign_fact("devon_goods_recovered"),),
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
                campaign_fact("devon_goods_recovered"),
            ),
            completion=campaign_fact("first_badge_obtained"),
            execution_id="defeat_roxanne",
            destination=MapRSE.RUSTBORO_CITY_GYM.value,
            resource_policy=ResourceObjective(
                "defeat_roxanne",
                readiness=ReadinessImportance.IMPORTANT,
                encounters=EncounterPolicy.PRESERVE,
                mandatory_battle=True,
                minimum_hp_ratio=0.75,
            ),
        ),
    )


def restock_pokeballs_objective() -> CampaignObjective:
    """Return the dynamic shopping task used when the capture reserve is low."""

    return CampaignObjective(
        "restock_pokeballs",
        "Restock Poké Balls before pursuing another encounter",
        # Pokédex receipt is the durable post-intro campaign boundary.  The
        # ``pokeballs_ready`` fact is intentionally reserved for the one-time
        # Birch handoff and may remain false while a later shop is perfectly
        # capable of selling balls.
        (campaign_fact("pokedex_received"),),
        campaign_fact("pokeballs_sufficient"),
        execution_id="restock_pokeballs",
        priority=10,
    )


def recovery_supply_policy() -> RecoverySupplyPolicy:
    """Return the configured medicine reserve policy."""

    runtime = getattr(context, "nuzlocke_runtime", None)
    configured = getattr(getattr(runtime, "rule_config", None), "recovery_supply_policy", None)
    return configured if isinstance(configured, RecoverySupplyPolicy) else RecoverySupplyPolicy()


def pokeball_restock_policy() -> PokeballRestockPolicy:
    """Return the configured Poké Ball reserve policy."""

    runtime = getattr(context, "nuzlocke_runtime", None)
    configured = getattr(getattr(runtime, "rule_config", None), "pokeball_policy", None)
    return configured if isinstance(configured, PokeballRestockPolicy) else PokeballRestockPolicy()


def _observed_recovery_quantities(state: CampaignState) -> tuple[int, int] | None:
    if not state.inventory.is_known or state.inventory.value is None:
        return None
    quantities = {item.name.casefold(): item.quantity for item in state.inventory.value.items}
    return quantities.get("antidote", 0), quantities.get("potion", 0)


def campaign_restock_item_summary(state: CampaignState, objective_id: str | None) -> str | None:
    """Describe the baseline transaction for a selected restock objective.

    Selection runs from an already observed ``CampaignState``. Keep this
    summary read-only and report policy-target quantities before the execution
    layer applies the live wallet and cash-floor limits.
    """

    if objective_id not in {"restock_pokeballs", "restock_recovery_items"}:
        return None
    inventory = state.inventory
    if not inventory.is_known or inventory.value is None:
        return "Antidote, Potion, and Poké Balls (quantities unavailable)"

    quantities = {}
    for item in (*inventory.value.items, *inventory.value.poke_balls):
        name = item.name.casefold()
        quantities[name] = quantities.get(name, 0) + item.quantity

    planned: list[tuple[str, int]] = []
    antidotes = quantities.get("antidote", 0)
    potions = quantities.get("potion", 0)
    recovery = recovery_supply_policy()
    recovery_required = recovery.needs_restock(antidotes, potions)
    if recovery_required:
        desired = recovery.desired_quantities(antidotes, potions)
        planned.extend((name, quantity) for name, quantity in desired.items() if quantity)

    balls = sum(
        quantity
        for name, quantity in quantities.items()
        if name != "master ball" and "ball" in name
    )
    pokeballs = pokeball_restock_policy()
    pokeballs_required = pokeballs.needs_restock(balls)
    if pokeballs_required:
        planned.append(("Poké Ball", pokeballs.quantity_to_buy(balls)))

    if not planned:
        return "No additional items (observed reserve is at target)"
    return (
        ", ".join(f"{quantity}x {name}" for name, quantity in planned)
        + " (policy targets; wallet/cash-floor limits may reduce purchase)"
    )


def _restock_is_deferred_on_current_map(state: CampaignState, objective_id: str) -> bool:
    """Return whether a failed funding attempt should yield this map to travel.

    The dynamic restock objective is normally an immediate interrupt.  When a
    Mart transaction is unaffordable and the live capability has already
    established that no undefeated trainer can fund it here, remounting that
    same objective every frame creates an ``earn_money`` standstill.  Keep the
    deferral narrow and transient: it applies only to the objective and map
    that produced it, and a map transition clears it.
    """

    marker = getattr(context, "campaign_restock_deferral", None)
    if not isinstance(marker, tuple) or len(marker) != 2:
        return False
    deferred_objective, deferred_map = marker
    if deferred_objective != objective_id:
        return False
    # A transient map-unavailable observation does not establish that the
    # player left the map. Preserve the marker until the next coherent map
    # observation; otherwise the controller can remount the same unfundable
    # restock during every script/warp transition frame.
    if not state.raw_map.is_known:
        return True
    if state.raw_map.value != deferred_map:
        try:
            context.campaign_restock_deferral = None
        except AttributeError:
            pass
        return False
    return True


def _active_story_scene_blocks_restock(state: CampaignState) -> bool:
    """Return whether the current map is inside the unfinished Wally scene."""

    raw_map = state.raw_map
    wally_complete = getattr(state.campaign_facts, "petalburg_wally_scene_complete", Fact.unavailable())
    return (
        raw_map.is_known
        and raw_map.value in {MapRSE.PETALBURG_CITY.value, MapRSE.PETALBURG_CITY_GYM.value}
        and wally_complete.status is FactStatus.KNOWN
        and wally_complete.value is False
    )


def _recovery_restock_is_required(state: CampaignState) -> bool:
    pokedex = state.campaign_facts.pokedex_received
    pokeballs_received = state.campaign_facts.pokeballs_received
    counts = _observed_recovery_quantities(state)
    area = state.canonical_area.value if state.canonical_area.is_known else None
    # The opening Littleroot handoff has no purchase route and its empty item
    # pocket is not evidence that a post-route reserve is overdue. Begin the
    # mutable medicine policy once the player has left the opening town.
    if (
        area == "LITTLEROOT_TOWN"
        or not pokedex.is_known
        or pokedex.value is not True
        or not pokeballs_received.is_known
        or pokeballs_received.value is not True
        or counts is None
    ):
        return False
    return recovery_supply_policy().needs_restock(*counts)


def recovery_supply_sufficient() -> CampaignPredicate:
    """Return the mutable inventory fact used by the medicine interrupt."""

    def evaluate(state: CampaignState) -> Fact[bool]:
        counts = _observed_recovery_quantities(state)
        if counts is None:
            return Fact(None, state.inventory.status)
        return Fact.known(not recovery_supply_policy().needs_restock(*counts))

    return CampaignPredicate("recovery_items_sufficient", "Antidote and Potion reserves are sufficient", evaluate)


def restock_recovery_items_objective() -> CampaignObjective:
    """Return the dynamic medicine restock task."""

    return CampaignObjective(
        "restock_recovery_items",
        "Restock Antidotes and Potions before continuing campaign travel",
        (birch_lab_handoff_complete(),),
        recovery_supply_sufficient(),
        execution_id="restock_recovery_items",
        priority=11,
    )


def _observed_pokeball_count(state: CampaignState) -> int | None:
    """Return the observed ordinary Poké Ball count for diagnostics.

    The boolean campaign facts intentionally remain the planner contract.  A
    count alongside the task-discovery trace makes it possible to distinguish
    a real low-reserve decision from an unavailable or stale inventory read.
    """

    inventory = state.inventory
    if not inventory.is_known or inventory.value is None:
        return None
    try:
        return sum(item.quantity for item in inventory.value.poke_balls if item.name.casefold() != "master ball")
    except (AttributeError, TypeError, ValueError):
        return None


def _restock_is_required(state: CampaignState) -> bool:
    """Return whether the observed campaign reserve requires shopping."""

    # Pokédex receipt enables encounters, while the live inventory is the
    # authoritative source for the mutable reserve.  The Birch-lab variable
    # can lag behind the inventory after a save-state or scripted handoff, so
    # it must not suppress restocking when ordinary balls are visibly present.
    ready = state.campaign_facts.pokedex_received
    received = state.campaign_facts.pokeballs_received
    sufficient = state.campaign_facts.pokeballs_sufficient
    count = _observed_pokeball_count(state)
    if ready.status is not FactStatus.KNOWN or ready.value is not True:
        return False
    if count is None:
        return False
    if sufficient.status is not FactStatus.KNOWN or sufficient.value is not False:
        return False
    # A nonzero live reserve proves the opening handoff has already happened,
    # even if its durable fact is unavailable.  For zero balls, retain the
    # Receipt guard so this dynamic task cannot pre-empt Birch's combined
    # Pokédex/Poké Ball handoff at the start of a run.
    return received.status is FactStatus.KNOWN and received.value is True or count > 0


def _campaign_progression_destination_for_restock(state: CampaignState) -> tuple[int, int] | None:
    """Return the current required campaign destination without resource interrupts.

    Poké Ball restock gating runs before the dynamic restock task is added to
    the frontier.  Resolve the ordinary required story tasks here so the
    encounter route can be measured against the destination the campaign is
    already prepared to pursue.
    """

    available: list[CampaignObjective] = []
    for task in initial_emerald_campaign():
        completion = task.completion.evaluate(state)
        if completion.status is not FactStatus.KNOWN or completion.value:
            continue
        if task.failure is not None:
            failure = task.failure.evaluate(state)
            if failure.status is not FactStatus.KNOWN or failure.value:
                continue
        prerequisites = tuple(prerequisite.evaluate(state) for prerequisite in task.prerequisites)
        if task.task_kind == "required" and all(
            result.status is FactStatus.KNOWN and result.value for result in prerequisites
        ):
            available.append(task)
    return progression_destination(tuple(available))[0]


def _observed_pending_encounter_opportunities(state: CampaignState) -> tuple[EncounterOpportunity, ...]:
    """Build encounter opportunities from the authoritative observed projection."""

    if not state.encounters.is_known:
        return ()
    return tuple(
        EncounterOpportunity(encounter.location, True, False, True)
        for encounter in state.encounters.value or ()
        if encounter.eligible and encounter.status == "none"
    )


def _pokeball_restock_is_warranted(
    state: CampaignState,
    opportunities: tuple[EncounterOpportunity, ...] | None = None,
    progression: tuple[int, int] | None = None,
    *,
    graph: WorldMapGraph | None = None,
) -> bool:
    """Return whether a low ball reserve is needed before current progression.

    A low reserve alone does not justify a Mart detour.  The selector must have
    a known required destination and an eligible, unconsumed encounter that is
    on that route or within the normal small-detour policy.  Unknown route or
    encounter data is treated as no planned encounter so the campaign can
    continue and reevaluate after its next authoritative observation.
    """

    if not _restock_is_required(state) or not state.encounters.is_known:
        return False
    progression = progression or _campaign_progression_destination_for_restock(state)
    if progression is None:
        return False
    if opportunities is None:
        try:
            opportunities = encounter_opportunities(state)
        except (AttributeError, NameError, RuntimeError, TypeError, ValueError):
            opportunities = ()
    merged = {opportunity.location: opportunity for opportunity in opportunities}
    for opportunity in _observed_pending_encounter_opportunities(state):
        merged.setdefault(opportunity.location, opportunity)

    preferred_locations: list[tuple[int, int]] = []
    for opportunity in merged.values():
        if opportunity.consumed or not opportunity.eligible:
            continue
        try:
            if not _has_current_campaign_encounter(state, opportunity.location):
                continue
            evaluation = evaluate_encounter_opportunity(
                state,
                opportunity,
                progression,
                graph=graph,
            )
        except (AttributeError, NameError, RuntimeError, TypeError, ValueError):
            continue
        if evaluation.recommendation is EncounterRecommendation.PREFER:
            preferred_locations.append(opportunity.location)

    selected = bool(preferred_locations)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_POKEBALL_RESTOCK_GATE: "
            f"reserve_low=True progression={progression!r} "
            f"candidate_count={len(merged)!r} "
            f"preferred_locations={tuple(preferred_locations)!r} selected={selected!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_POKEBALL_RESTOCK_GATE",
    )
    return selected


def encounter_task(
    location: tuple[int, int], *, name: str | None = None, observed: bool | None = None
) -> CampaignObjective:
    """Describe an opportunistic encounter without choosing encounter policy."""
    task_id = name or f"obtain_encounter:{location[0]}:{location[1]}"
    try:
        location_name = MapRSE(location).pretty_name
    except (TypeError, ValueError):
        location_name = str(location)

    def accessible(state: CampaignState) -> Fact[bool]:
        """Report whether this location still has an eligible encounter."""

        if not state.raw_map.is_known or not state.encounters.is_known:
            return Fact(None, state.raw_map.status if not state.raw_map.is_known else state.encounters.status)
        encounter = state.encounter_for(location)
        if encounter.value.status != "none" or not encounter.value.eligible:
            return Fact.known(False)
        return Fact.known(_has_current_campaign_encounter(state, location))

    return CampaignObjective(
        task_id,
        f"Obtain the first encounter at {location_name}",
        (
            # Encounter legality begins at Pokédex receipt.  Before that ROM
            # boundary, an observed or catalogued grass table must not become
            # an executable campaign task.
            campaign_fact("pokedex_received"),
            # A legal encounter is not executable until the campaign has
            # actually acquired Poké Balls.  Keeping this prerequisite
            # explicit prevents an optional encounter from outranking the
            # Poké Ball capability and mounting a capture route with no way
            # to complete it.
            campaign_fact("pokeballs_ready"),
            # Inventory is mutable after the initial Birch handoff. An
            # encounter task must wait for a known reserve check; a false
            # value is surfaced as the dynamic restock objective instead.
            campaign_fact("pokeballs_sufficient"),
            CampaignPredicate(
                f"encounter_accessible:{location}",
                "location is accessible and unconsumed",
                accessible,
            ),
        ),
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
        tactical_target=NavigationGoal(
            SemanticTarget.map(location),
            constraints=GoalConstraints(trainer_mode=choose_trainer_mode(TrainerPolicyInput(seeking_encounter=True))),
            encounter_mode=EncounterMode.SEEK,
        ),
        observed=observed,
    )


def _current_campaign_encounter_candidates(
    state: CampaignState, location: tuple[int, int]
) -> tuple[EncounterCandidate, ...]:
    """Return encounter candidates usable by the current campaign runtime."""

    return encounter_candidates_for_location(
        location,
        available_methods=campaign_encounter_methods(state),
    )


def _has_current_campaign_encounter(state: CampaignState, location: tuple[int, int]) -> bool:
    """Return whether a location has a table the campaign can execute now."""

    try:
        return bool(_current_campaign_encounter_candidates(state, location))
    except AttributeError:
        # Lightweight pure fixtures may intentionally omit a ROM.  In that
        # boundary an observed encounter record remains authoritative; a live
        # campaign always has a ROM before this path is mounted.
        return getattr(context, "rom", None) is None
    except (RuntimeError, TypeError, ValueError):
        return False


def _rom_backed_campaign_route(graph: WorldMapGraph, source_map, target_map):
    """Use the ROM transition catalog when measuring campaign routes."""
    transition_cache: dict[object, tuple[WorldTransition, ...] | None] = {}

    def edge_is_executable(edge) -> bool:
        if not getattr(edge, "source_coordinates", ()):
            # Geometry-free WorldEdge instances are abstract planning
            # fixtures. They intentionally model costs without requiring a
            # live ROM transition catalog.
            return True
        if edge.source_map not in transition_cache:
            try:
                transition_cache[edge.source_map] = tuple(static_map_transitions(edge.source_map))
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # Pure planning fixtures may provide a graph without an
                # emulator backed transition catalog. Retain their graph
                # semantics; live campaign runs have ROM evidence here.
                transition_cache[edge.source_map] = None
        transitions = transition_cache[edge.source_map]
        if transitions is None:
            return True
        expected_kind = "map_connection" if edge.kind == "connection" else edge.kind
        return any(
            transition.destination is not None
            and transition.destination[0] == edge.destination_map
            and transition.kind == expected_kind
            for transition in transitions
        )

    return route_with_edge_filter(graph, source_map, target_map, edge_is_executable)


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
        try:
            # One forward and one reverse table answer all three distances for
            # the full ROM encounter catalog. Calling route() independently
            # for every location repeats the same Dijkstra work and can make
            # a post-battle refresh look like a long emulator stall.
            forward_costs = dict(route_graph.map_costs(current))
            reverse_costs = dict(route_graph.map_costs_to(progression_destination))
            direct = forward_costs.get(progression_destination)
            to_task = forward_costs.get(task.destination)
            onward = reverse_costs.get(task.destination)
            if direct is None or to_task is None or onward is None:
                raise WorldNavigationError("task route is not connected to campaign progression")
            direct_route = _rom_backed_campaign_route(route_graph, current, progression_destination)
        except AttributeError:
            # Keep compatibility with lightweight graph doubles supplied by
            # embedders and older tests that expose only route().
            direct = route_graph.route(current, progression_destination).estimated_cost
            to_task = route_graph.route(current, task.destination).estimated_cost
            onward = route_graph.route(task.destination, progression_destination).estimated_cost
            direct_route = route_graph.route(current, progression_destination)
    except WorldNavigationError:
        return RouteContext(
            current, progression_destination, task.destination, None, None, None, RouteRelation.UNREACHABLE
        )
    via = to_task + onward
    # Route cost alone cannot express that Route 101 is literally the first
    # map on the Littleroot-to-Petalburg corridor.  Use the ROM-backed direct
    # route's map sequence as the primary membership fact; a map on that
    # corridor has no campaign detour even when its local map edge costs make
    # the arithmetic look more expensive.
    on_direct_corridor = task.destination in getattr(direct_route, "maps", ())
    detour = 0 if on_direct_corridor else max(0, via - direct)
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
    trainer_checkpoint=None,
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
        reachability = assess_campaign_reachability(state, opportunity.location, graph=route_graph)
        direct_route = None
        if callable(getattr(route_graph, "outgoing", None)):
            direct_route = _rom_backed_campaign_route(
                route_graph, state.raw_map.value, progression_destination
            )
            direct = _rom_backed_campaign_route(
                route_graph, state.raw_map.value, progression_destination
            ).estimated_cost
            to_encounter = _rom_backed_campaign_route(
                route_graph, state.raw_map.value, opportunity.location
            ).estimated_cost
            onward = _rom_backed_campaign_route(
                route_graph, opportunity.location, progression_destination
            ).estimated_cost
        else:
            direct_route = route_graph.route(state.raw_map.value, progression_destination)
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
            reachability_status=CampaignReachabilityStatus.UNREACHABLE,
            reachability_blockers=("no executable ROM map route",),
        )
    via = to_encounter + onward
    on_direct_corridor = opportunity.location in getattr(direct_route, "maps", ())
    detour = 0 if on_direct_corridor else max(0, via - direct)
    ratio = detour / direct if direct else (0.0 if detour == 0 else float("inf"))
    if not reachability.executable_now:
        return EncounterEvaluation(
            **base,
            direct_progression_cost=direct,
            encounter_cost=to_encounter,
            encounter_to_progression_cost=onward,
            via_encounter_cost=via,
            detour_cost=max(0, via - direct),
            detour_ratio=ratio,
            classification=EncounterClassification.UNREACHABLE,
            recommendation=EncounterRecommendation.UNAVAILABLE,
            reachability_status=reachability.status,
            reachability_blockers=reachability.blockers,
        )
    if detour == 0:
        classification = EncounterClassification.ON_ROUTE
        recommendation = EncounterRecommendation.PREFER
    elif detour <= policy.small_detour_cost or ratio <= policy.small_detour_ratio:
        classification = EncounterClassification.SMALL_DETOUR
        recommendation = EncounterRecommendation.PREFER
    else:
        classification = EncounterClassification.LARGE_DETOUR
        recommendation = EncounterRecommendation.DEFER
    preparation_gain, checkpoint_id = _encounter_trainer_preparation_gain(
        state,
        opportunity.location,
        trainer_checkpoint,
    )
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
        trainer_preparation_gain=preparation_gain,
        trainer_checkpoint_id=checkpoint_id,
        reachability_status=reachability.status,
        reachability_blockers=reachability.blockers,
    )


def _encounter_trainer_preparation_gain(
    state: CampaignState,
    location: tuple[int, int],
    trainer_checkpoint=None,
) -> tuple[float, str | None]:
    """Estimate whether an encounter can add coverage for the next forced fight.

    The campaign selector does not have the eventual caught Pokémon's exact
    level, IVs, or moveset. It therefore uses a deliberately modest proxy:
    compare the best known level-up offensive coverage in the encounter pool
    with the coverage already present in the observed party. A detour is
    promoted only when the pool adds a type-effective line against a known
    unavoidable checkpoint; unknown trainers remain unknown.
    """

    if trainer_checkpoint is None or not state.party.is_known:
        return 0.0, getattr(trainer_checkpoint, "checkpoint_id", None)
    try:
        from modules.pokemon import get_move_by_name, get_species_by_name

        opponents = tuple(
            get_species_by_name(member.species)
            for member in getattr(trainer_checkpoint, "party", ())
        )
        if not opponents:
            return 0.0, getattr(trainer_checkpoint, "checkpoint_id", None)

        def coverage(move_types) -> int:
            return sum(
                1
                for opponent in opponents
                if any(
                    _move_is_effective_against(move_type, opponent)
                    for move_type in move_types
                )
            )

        current_move_types = []
        for member in state.party.value or ():
            for move in getattr(member, "moves", ()):
                try:
                    current_move_types.append(get_move_by_name(move.name).type)
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    continue
        current_coverage = coverage(current_move_types)
        best_candidate_coverage = current_coverage
        for candidate in _current_campaign_encounter_candidates(state, location):
            for species_name in getattr(candidate, "species_pool", ()):
                try:
                    species = get_species_by_name(species_name)
                    candidate_move_types = [
                        entry.move.type
                        for entry in getattr(getattr(species, "learnset", None), "level_up", ())
                        if int(getattr(entry, "level", 0) or 0) <= 10
                        and getattr(getattr(entry, "move", None), "base_power", 0) > 0
                    ]
                    best_candidate_coverage = max(best_candidate_coverage, coverage(candidate_move_types))
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    continue
        gain = float(max(0, best_candidate_coverage - current_coverage))
        if gain == 0 and len(state.party.value or ()) < 6:
            # A legal, still-open party slot is itself a meaningful
            # probability increase before an unavoidable trainer: it adds a
            # fresh HP pool and another chance to finish the fight. Keep this
            # contribution below type-effective coverage so a real matchup
            # improvement always wins the detour comparison.
            gain = 0.25
        return gain, getattr(trainer_checkpoint, "checkpoint_id", None)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return 0.0, getattr(trainer_checkpoint, "checkpoint_id", None)


def _move_is_effective_against(move_type, opponent) -> bool:
    """Return whether one move type deals super-effective damage."""

    try:
        effectiveness = 1.0
        for defender_type in opponent.types:
            effectiveness *= move_type.get_effectiveness_against(defender_type)
        return effectiveness > 1.0
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _next_unavoidable_trainer_checkpoint(state: CampaignState, progression):
    """Find the registered trainer on the current map before progression."""

    if not state.raw_map.is_known or progression is None:
        return None
    try:
        from modules.trainer_checkpoints import KNOWN_TRAINER_CHECKPOINTS

        current_map = state.raw_map.value
        return next(
            (
                checkpoint
                for checkpoint in KNOWN_TRAINER_CHECKPOINTS
                if checkpoint.map_id == current_map
                and getattr(checkpoint, "mandatory", False)
                and not (
                    getattr(checkpoint, "clearance_fact", None)
                    and getattr(state.campaign_facts, checkpoint.clearance_fact, None) is not None
                    and getattr(state.campaign_facts, checkpoint.clearance_fact).status is FactStatus.KNOWN
                    and getattr(state.campaign_facts, checkpoint.clearance_fact).value is True
                )
            ),
            None,
        )
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None


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


def _frontier_encounter_opportunities(
    state: CampaignState,
    opportunities: tuple[EncounterOpportunity, ...],
    progression: tuple[int, int] | None,
    *,
    graph: WorldMapGraph | None = None,
) -> tuple[EncounterOpportunity, ...]:
    """Keep only encounter maps worth considering for the current frontier.

    The ROM catalog contains every encounter map in the game.  The campaign
    selector only needs opportunities that are on the current static route or
    within the same small-detour threshold used by encounter evaluation.
    Static forward and reverse distance tables make this one bounded graph
    pass instead of running a fresh shortest-path search for every catalog
    entry. Observed pending locations are still supplied separately by the
    caller, so this filter cannot discard a rule projection that has already
    become relevant.
    """

    if not opportunities or progression is None or not state.raw_map.is_known:
        return opportunities
    try:
        route_graph = graph or get_world_map_graph()
        forward = dict(route_graph.map_costs(state.raw_map.value))
        reverse = dict(route_graph.map_costs_to(progression))
        direct = forward.get(progression)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # A static graph is an avenue to try, not an authority that may
        # convert a live campaign into a terminal failure. The normal route
        # evaluator will report an unavailable relationship if no graph can
        # be constructed, while this live frontier remains bounded.
        return ()
    if direct is None:
        return ()
    policy = EncounterEvaluationPolicy()
    detour_limit = max(policy.small_detour_cost, direct * policy.small_detour_ratio)
    return tuple(
        opportunity
        for opportunity in opportunities
        if (
            (to_encounter := forward.get(opportunity.location)) is not None
            and (encounter_to_progression := reverse.get(opportunity.location)) is not None
            and max(0, to_encounter + encounter_to_progression - direct) <= detour_limit
        )
    )


def available_campaign_tasks(
    state: CampaignState,
    tasks: tuple[CampaignObjective, ...] | None = None,
    encounter_locations: tuple[tuple[int, int], ...] | None = None,
) -> tuple[CampaignObjective, ...]:
    """Enumerate ready tasks; unavailable/unknown tasks are omitted."""
    candidates = initial_emerald_campaign() if tasks is None else tasks
    encounter_catalog_error = None
    opportunity_count = 0
    catalog_count = 0
    opportunities = ()
    route_graph = None

    # This is intentionally discovered dynamically rather than inserted into
    # the ordered story list: after the initial Birch handoff, inventory is a
    # mutable resource and a low-ball observation should create a task only
    # when it is actually needed.
    if tasks is None:
        if (
            _recovery_restock_is_required(state)
            and not _active_story_scene_blocks_restock(state)
            and not _restock_is_deferred_on_current_map(state, "restock_recovery_items")
        ):
            recovery_restock = restock_recovery_items_objective()
            insertion = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.objective_id == "complete_petalburg_wally"
                ),
                len(candidates),
            )
            candidates = candidates[:insertion] + (recovery_restock,) + candidates[insertion:]

    def task_is_available(task: CampaignObjective) -> bool:
        """Return whether one task is complete, safe, and executable now."""

        completion = task.completion.evaluate(state)
        if completion.status is not FactStatus.KNOWN or completion.value:
            return False
        if task.failure is not None:
            failure = task.failure.evaluate(state)
            if failure.status is not FactStatus.KNOWN or failure.value:
                return False
        results = tuple(prerequisite.evaluate(state) for prerequisite in task.prerequisites)
        return all(result.status is FactStatus.KNOWN and result.value for result in results)

    catalog_allowed = (
        tasks is None
        and state.encounters.is_known
        and state.campaign_facts.pokedex_received.is_known
        and state.campaign_facts.pokedex_received.value is True
    )
    if tasks is None and state.encounters.is_known:
        # The rules projection is deliberately the source of truth here.  It
        # knows only locations it has observed, so discovery must not invent
        # map encounters from the ROM's wild-data tables.
        candidates += tuple(
            encounter_task(encounter.location)
            for encounter in state.encounters.value or ()
            if encounter.eligible and encounter.status == "none"
        )
    base_available = tuple(task for task in candidates if task_is_available(task))
    progression, _ = progression_destination(base_available)
    catalog_attempted = tasks is None and state.encounters.is_known
    if catalog_attempted:
        try:
            catalog_opportunities = encounter_opportunities(state, encounter_locations)
            catalog_count = len(catalog_opportunities)
            # Discovery should expose every unconsumed executable encounter
            # location. Route preference belongs to evaluation/selection;
            # filtering the catalog here made future route encounters vanish
            # from diagnostics and from the campaign task frontier.
            opportunities = catalog_opportunities
            if catalog_allowed and encounter_locations is None:
                try:
                    route_graph = get_world_map_graph()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    # Route metadata is optional for discovery. An encounter
                    # can remain visible with UNKNOWN route evaluation and be
                    # reconsidered once the graph is available.
                    route_graph = None
        except (AttributeError, NameError, RuntimeError, TypeError, ValueError) as error:
            opportunities = ()
            encounter_catalog_error = f"{type(error).__name__}: {error}"
        opportunity_count = len(opportunities)
        candidates += tuple(
            encounter_task(item.location, observed=item.observed)
            for item in opportunities
            if item.eligible and not item.consumed
        )
        # De-duplicate observed locations when the compatibility fallback
        # above and the world catalog describe the same task.
        candidates = tuple({task.objective_id: task for task in candidates}.values())
    elif tasks is None:
        # An unavailable encounter projection is not safe to use for task
        # discovery, even if the campaign facts are otherwise complete.
        catalog_count = 0

    # Unlike recovery supplies, Poké Balls are only an immediate campaign
    # requirement when a legal encounter is planned on the current route.
    # Evaluate this after encounter discovery so a low reserve cannot create a
    # Mart detour with no encounter between the current map and progression.
    if (
        tasks is None
        and _restock_is_required(state)
        and not _active_story_scene_blocks_restock(state)
        and not _restock_is_deferred_on_current_map(state, "restock_pokeballs")
        and _pokeball_restock_is_warranted(
            state,
            opportunities=opportunities,
            progression=progression,
            graph=route_graph,
        )
    ):
        restock = restock_pokeballs_objective()
        insertion = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if candidate.objective_id == "complete_petalburg_wally"
            ),
            len(candidates),
        )
        candidates = candidates[:insertion] + (restock,) + candidates[insertion:]
    diagnostic_print(
        lambda: (
            "CAMPAIGN_ENCOUNTER_DISCOVERY: "
            f"encounters_status={state.encounters.status.value!r} "
            f"encounters_known={state.encounters.is_known!r} "
            f"catalog_attempted={catalog_attempted!r} "
            f"catalog_allowed={catalog_allowed!r} "
            f"catalog_count={catalog_count!r} "
            f"opportunity_count={opportunity_count!r} "
            f"catalog_error={encounter_catalog_error!r} "
            f"pokeball_count={_observed_pokeball_count(state)!r} "
            f"pokeballs_received={state.campaign_facts.pokeballs_received.value!r} "
            f"pokeballs_received_status={state.campaign_facts.pokeballs_received.status.value!r} "
            f"pokeballs_sufficient={state.campaign_facts.pokeballs_sufficient.value!r} "
            f"pokeballs_sufficient_status={state.campaign_facts.pokeballs_sufficient.status.value!r} "
            f"campaign_methods={tuple(sorted(method.value for method in campaign_encounter_methods(state)))!r} "
            f"early_locations={tuple(item.location for item in opportunities if item.location in {(0, 16), (0, 17), (0, 18), (0, 19)})!r}"
        ),
        trace=True,
    )
    available: list[CampaignObjective] = []
    for task in candidates:
        if task_is_available(task):
            available.append(task)
    progression, _ = progression_destination(tuple(available))
    imminent_trainer = _next_unavoidable_trainer_checkpoint(state, progression)
    result = []
    for task in available:
        route = (
            measure_route_context(state, task, progression, graph=route_graph) if task.destination is not None else None
        )
        evaluation = task.encounter_evaluation
        if task.task_kind == "optional" and task.destination is not None:
            fact = state.encounter_for(task.destination)
            opportunity = EncounterOpportunity(
                task.destination,
                task.observed if task.observed is not None else fact.is_known,
                fact.value.status != "none" if fact.is_known else False,
                fact.value.eligible if fact.is_known else True,
            )
            evaluation = evaluate_encounter_opportunity(
                state,
                opportunity,
                progression,
                task_id=task.objective_id,
                graph=route_graph,
                trainer_checkpoint=imminent_trainer,
            )
        result.append(replace(task, route_context=route, encounter_evaluation=evaluation))
    diagnostic_print(
        lambda: (
            "CAMPAIGN_ENCOUNTER_FRONTIER: "
            f"optional={[{'id': task.objective_id, 'recommendation': task.encounter_evaluation.recommendation.value if task.encounter_evaluation else None, 'classification': task.encounter_evaluation.classification.value if task.encounter_evaluation else None, 'reachability': task.encounter_evaluation.reachability_status.value if task.encounter_evaluation else None, 'blockers': task.encounter_evaluation.reachability_blockers if task.encounter_evaluation else (), 'trainer_preparation_gain': task.encounter_evaluation.trainer_preparation_gain if task.encounter_evaluation else None, 'trainer_checkpoint': task.encounter_evaluation.trainer_checkpoint_id if task.encounter_evaluation else None} for task in result if task.task_kind == 'optional']!r}"
        ),
        trace=True,
    )
    return tuple(result)


def _select_available_campaign_tasks(
    available: tuple[CampaignObjective, ...], state: CampaignState | None = None
) -> ObjectiveSelection:
    """Select the deterministic best task from one discovery result."""

    required = tuple(task for task in available if task.task_kind == "required")
    # Optional legal encounters form one policy frontier.  Route geometry still
    # contributes to each candidate's score, but a large detour is no longer a
    # hard exclusion that can disagree with the ordered portfolio displayed to
    # the operator.
    strategic_optional = tuple(
        task
        for task in available
        if task.task_kind == "optional"
        and task.encounter_evaluation is not None
        and task.encounter_evaluation.executable_now
        and task.encounter_evaluation.recommendation
        in {EncounterRecommendation.PREFER, EncounterRecommendation.DEFER}
        and task.encounter_evaluation.classification
        in {
            EncounterClassification.ON_ROUTE,
            EncounterClassification.SMALL_DETOUR,
            EncounterClassification.LARGE_DETOUR,
        }
    )
    encounter_portfolio: tuple[dict[str, object], ...] = ()

    def strategic_candidate(task: CampaignObjective):
        try:
            candidates = _current_campaign_encounter_candidates(state, task.destination) if task.destination else ()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            candidates = ()
        if not candidates:
            return None
        evaluation = getattr(task, "encounter_evaluation", None)
        # Route measurements belong to the objective evaluation, while the
        # pure capture policy owns the encounter distribution. Join them at
        # this boundary so detour cost is priced by the same score used for
        # species-clause conditioning.
        if evaluation is not None:
            candidates = tuple(
                replace(
                    candidate,
                    route_cost=evaluation.encounter_cost,
                    detour_cost=evaluation.detour_cost,
                    # The route evaluator already has the best available
                    # proxy for trainer preparation value: known type
                    # coverage and the value of an additional living party
                    # slot against the next checkpoint. Carry it into the
                    # portfolio scorer instead of allowing the encounter
                    # ranking to discard that campaign fact.
                    training_value=evaluation.trainer_preparation_gain,
                )
                for candidate in candidates
            )
        policy_context = _capture_policy_context(state)
        return max(
            candidates,
            key=lambda candidate: (
                score_encounter(candidate, policy_context),
                candidate.method.value,
            ),
        )

    def strategic_score(task: CampaignObjective) -> float:
        candidate = strategic_candidate(task)
        if candidate is None:
            return float("-inf")
        current_level = None
        if state is not None and state.party.is_known:
            levels = tuple(
                int(getattr(member, "level", 0) or 0)
                for member in (state.party.value or ())
                if int(getattr(member, "level", 0) or 0) > 0
            )
            current_level = max(levels, default=None)
        return score_portfolio_candidate(
            candidate,
            EncounterPortfolioContext(
                capture=_capture_policy_context(state),
                current_level=current_level,
                max_encounters=1,
            ),
        )

    def portfolio_task(tasks: tuple[CampaignObjective, ...]):
        """Select one task from a joint encounter portfolio."""

        nonlocal encounter_portfolio

        entries = []
        for task in tasks:
            candidate = strategic_candidate(task)
            if candidate is not None:
                entries.append((task, candidate))
        entries = tuple(entries)
        if not entries:
            encounter_portfolio = ()
            return None, None
        portfolio_capture_context = _capture_policy_context(state)
        current_level = None
        if state is not None and state.party.is_known:
            levels = tuple(
                int(getattr(member, "level", 0) or 0)
                for member in (state.party.value or ())
                if int(getattr(member, "level", 0) or 0) > 0
            )
            current_level = max(levels, default=None)
        plan = choose_encounter_portfolio(
            tuple(candidate for _, candidate in entries),
            EncounterPortfolioContext(
                capture=portfolio_capture_context,
                current_level=current_level,
                # Keep the full ordered frontier available for diagnostics,
                # while mounting only its first task. The next capture
                # re-plans because Species Clause history changes the legal
                # portfolio.
                max_encounters=max(1, portfolio_capture_context.open_party_slots or 1),
            ),
        )
        task_by_candidate = {candidate: task for task, candidate in entries}
        portfolio_context = EncounterPortfolioContext(
            capture=portfolio_capture_context,
            current_level=current_level,
            max_encounters=1,
        )
        portfolio_rows = []
        for rank, candidate in enumerate(plan.ordered_candidates, 1):
            task = task_by_candidate[candidate]
            evaluation = task.encounter_evaluation
            portfolio_rows.append(
                {
                    "rank": rank,
                    "task_id": task.objective_id,
                    "area": candidate.area,
                    "method": candidate.method.value,
                    "selected": candidate == plan.selected,
                    "portfolio_selected": candidate in plan.candidates,
                    "score": score_portfolio_candidate(candidate, portfolio_context),
                    "policy": evaluation.recommendation.value if evaluation is not None else "unknown",
                    "classification": evaluation.classification.value if evaluation is not None else "unknown",
                    "route_cost": candidate.route_cost,
                    "detour_cost": candidate.detour_cost,
                    "trainer_preparation_gain": candidate.training_value,
                    "species": candidate.species_pool,
                }
            )
        encounter_portfolio = tuple(portfolio_rows)
        selected = plan.selected
        if selected is None:
            return None, None
        return next(
            ((task, candidate) for task, candidate in entries if candidate == selected),
            (None, None),
        )

    def party_has_open_slot() -> bool:
        return bool(
            state is not None
            and state.party.is_known
            and len(state.party.value or ()) < 6
        )

    def nearest_preferred_encounter(tasks: tuple[CampaignObjective, ...]) -> CampaignObjective:
        return min(
            tasks,
            key=lambda task: (
                task.encounter_evaluation.detour_cost
                if task.encounter_evaluation and task.encounter_evaluation.detour_cost is not None
                else float("inf"),
                task.encounter_evaluation.encounter_cost
                if task.encounter_evaluation and task.encounter_evaluation.encounter_cost is not None
                else float("inf"),
                task.objective_id,
            ),
        )

    def finish(objective: CampaignObjective | None, status: ObjectiveStatus, reason: str) -> ObjectiveSelection:
        """Attach the evidence that explains this frontier decision.

        Encounter selection is intentionally policy-owned by this function.
        The diagnostics are therefore assembled at the same boundary as the
        decision instead of asking the console to infer intent from a route
        table after the fact.
        """

        selected_id = objective.objective_id if objective is not None else None
        diagnostics: list[dict[str, object]] = []
        for task in available:
            evaluation = task.encounter_evaluation
            if task.task_kind != "optional" or evaluation is None:
                continue
            candidate = strategic_candidate(task)
            score = strategic_score(task) if candidate is not None else None
            if task.objective_id == selected_id:
                task_reason = "selected by the campaign encounter policy"
            elif not evaluation.executable_now:
                blockers = ", ".join(evaluation.reachability_blockers) or evaluation.reachability_status.value
                task_reason = f"skipped because it is not executable now ({blockers})"
            elif objective is not None and objective.task_kind == "required":
                if evaluation.recommendation is EncounterRecommendation.DEFER:
                    task_reason = "deferred because its detour is not worth interrupting required progression"
                else:
                    task_reason = "skipped because required progression retained ownership of the route"
            elif evaluation.recommendation is EncounterRecommendation.DEFER:
                task_reason = "deferred by encounter policy"
            else:
                task_reason = "skipped because another available task ranked higher"
            diagnostics.append(
                {
                    "task_id": task.objective_id,
                    "selected": task.objective_id == selected_id,
                    "reason": task_reason,
                    "classification": evaluation.classification.value,
                    "recommendation": evaluation.recommendation.value,
                    "reachability_status": evaluation.reachability_status.value,
                    "reachability_blockers": evaluation.reachability_blockers,
                    "route_cost": evaluation.encounter_cost,
                    "detour_cost": evaluation.detour_cost,
                    "direct_progression_cost": evaluation.direct_progression_cost,
                    "via_encounter_cost": evaluation.via_encounter_cost,
                    "detour_ratio": evaluation.detour_ratio,
                    "trainer_preparation_gain": evaluation.trainer_preparation_gain,
                    "trainer_checkpoint_id": evaluation.trainer_checkpoint_id,
                    "strategic_score": score,
                    "candidate_method": candidate.method.value if candidate is not None else None,
                    "candidate_species": candidate.species_pool if candidate is not None else (),
                    "candidate_training_value": candidate.training_value if candidate is not None else None,
                }
            )
        # The task selector may have evaluated only the narrow executable
        # subset needed for its branch (for example, the current on-route
        # encounter). Rebuild a read-only full portfolio for the report so
        # deferred but legal locations remain visible beside the selected
        # task. This does not feed back into selection.
        portfolio_entries = []
        for task in available:
            if task.task_kind != "optional" or task.encounter_evaluation is None:
                continue
            if not task.encounter_evaluation.executable_now:
                continue
            candidate = strategic_candidate(task)
            if candidate is not None:
                portfolio_entries.append((task, candidate))
        display_portfolio = encounter_portfolio
        if portfolio_entries:
            capture_context = _capture_policy_context(state)
            current_level = None
            if state is not None and state.party.is_known:
                levels = tuple(
                    int(getattr(member, "level", 0) or 0)
                    for member in (state.party.value or ())
                    if int(getattr(member, "level", 0) or 0) > 0
                )
                current_level = max(levels, default=None)
            display_context = EncounterPortfolioContext(
                capture=capture_context,
                current_level=current_level,
                max_encounters=None,
            )
            task_by_candidate = {candidate: task for task, candidate in portfolio_entries}
            display_plan = choose_encounter_portfolio(
                tuple(candidate for _, candidate in portfolio_entries),
                display_context,
            )
            decision_ids = {
                str(detail.get("task_id"))
                for detail in encounter_portfolio
                if isinstance(detail, dict)
            }
            display_rows = []
            for rank, candidate in enumerate(display_plan.ordered_candidates, 1):
                task = task_by_candidate[candidate]
                evaluation = task.encounter_evaluation
                display_rows.append(
                    {
                        "rank": rank,
                        "task_id": task.objective_id,
                        "area": candidate.area,
                        "method": candidate.method.value,
                        "selected": task.objective_id == selected_id,
                        "portfolio_selected": task.objective_id in decision_ids,
                        "score": score_portfolio_candidate(candidate, display_context),
                        "policy": evaluation.recommendation.value if evaluation is not None else "unknown",
                        "classification": evaluation.classification.value if evaluation is not None else "unknown",
                        "route_cost": candidate.route_cost,
                        "detour_cost": candidate.detour_cost,
                        "trainer_preparation_gain": candidate.training_value,
                        "species": candidate.species_pool,
                    }
                )
            display_portfolio = tuple(display_rows)
        return ObjectiveSelection(
            objective,
            status,
            reason,
            selection_diagnostics=tuple(diagnostics),
            encounter_portfolio=display_portfolio,
        )

    # Restocking is an explicit resource interrupt. It must take ownership
    # immediately when it is available, even if an encounter or story task
    # was already mounted and would otherwise win the normal required-task
    # ordering.
    restock = next(
        (task for task in required if task.objective_id in {"restock_pokeballs", "restock_recovery_items"}),
        None,
    )
    if restock is not None:
        reason = (
            "selected required recovery-item restock task"
            if restock.objective_id == "restock_recovery_items"
            else "selected required Poké Ball restock task"
        )
        return finish(restock, ObjectiveStatus.READY, reason)

    progression_anchor, _ = progression_destination(available)
    raw_map = getattr(state, "raw_map", None) if state is not None else None
    current_map = (
        getattr(raw_map, "value", None)
        if raw_map is not None and getattr(raw_map, "is_known", False)
        else None
    )
    # When the live party is already on a zero-detour map immediately before
    # the next required map, evaluate that encounter in the same portfolio as
    # the required-map encounter. This is the Route 104 -> Petalburg Woods
    # boundary: consuming Route 104 first preserves a legal party slot and
    # avoids treating the woods encounter as an exclusive side quest.
    progression_encounters = tuple(
        task
        for task in available
        if task.task_kind == "optional"
        and task.encounter_evaluation is not None
        and task.encounter_evaluation.executable_now
        and task.encounter_evaluation.recommendation
        in {EncounterRecommendation.PREFER, EncounterRecommendation.DEFER}
        and (
            task.destination == progression_anchor
            or (
                current_map is not None
                and task.destination == current_map
                and task.encounter_evaluation.detour_cost == 0
            )
        )
    )
    if party_has_open_slot() and progression_encounters:
        selected_optional, selected_candidate = portfolio_task(progression_encounters)
        if selected_candidate is not None:
            return finish(
                replace(selected_optional, encounter_method=selected_candidate.method),
                ObjectiveStatus.READY,
                "selected encounter on the current required progression map",
            )

    # An encounter already on the map the campaign is traversing does not
    # create the kind of detour that DEFER normally describes.  This is the
    # Route 102/Calvin case: if the party has room and is already there, take
    # the legal encounter before walking into the mandatory checkpoint.
    current_map_encounters = tuple(
        task
        for task in available
        if (
            party_has_open_slot()
            and current_map is not None
            and task.task_kind == "optional"
            and task.destination == current_map
            and task.encounter_evaluation is not None
            and task.encounter_evaluation.executable_now
            and strategic_candidate(task) is not None
            and strategic_score(task) > 0
        )
    )
    if current_map_encounters:
        selected_optional, selected_candidate = portfolio_task(current_map_encounters)
        if selected_candidate is not None:
            return finish(
                replace(selected_optional, encounter_method=selected_candidate.method),
                ObjectiveStatus.READY,
                "selected encounter on the current required progression map",
            )

    # The same portfolio now ranks on-route, small-detour, and legal large-
    # detour encounters together.  Route cost, level fit, species-clause
    # conditioning, trainer preparation value, and party capacity are priced
    # by the capture policy instead of being split across mutually exclusive
    # selector branches.
    if party_has_open_slot() and strategic_optional:
        selected_optional, selected_candidate = portfolio_task(strategic_optional)
        if selected_candidate is not None:
            evaluation = selected_optional.encounter_evaluation
            if evaluation is not None and evaluation.trainer_preparation_gain > 0:
                reason = (
                    "selected encounter detour to improve odds against "
                    f"{evaluation.trainer_checkpoint_id}"
                )
            elif evaluation is not None and evaluation.recommendation is EncounterRecommendation.DEFER:
                reason = (
                    "selected deferred encounter from the legal encounter portfolio"
                    if required
                    else "selected deferred encounter because no required task is executable"
                )
            else:
                reason = (
                    "selected nearest on-route encounter while party has an open slot"
                    if all(
                        task.encounter_evaluation.recommendation is EncounterRecommendation.PREFER
                        for task in strategic_optional
                        if task.encounter_evaluation is not None
                    )
                    else "selected strategic encounter from the legal encounter portfolio"
                )
            return finish(
                replace(selected_optional, encounter_method=selected_candidate.method),
                ObjectiveStatus.READY,
                reason,
            )

    if required:
        # Required progression is the executable fallback when no encounter
        # has enough utility, capacity, or route safety to own the frontier.
        task = required[0]
        return finish(task, ObjectiveStatus.READY, "selected required campaign progression task")
    selectable = tuple(
        task
        for task in available
        if task.task_kind != "optional"
        or (
            task.encounter_evaluation is not None
            # DEFER means "not worth interrupting required progression for";
            # it is still a known reachable encounter. If no required task
            # is executable, it must remain a valid fallback instead of
            # turning the campaign frontier into a terminal COMPLETE result.
            and task.encounter_evaluation.recommendation
            in {EncounterRecommendation.PREFER, EncounterRecommendation.DEFER}
        )
    )
    if selectable:
        # Strategic ranking is intentionally applied only when candidate
        # metadata is available.  Route evaluation remains the stable
        # fallback for fixtures and partially observed campaign states.
        task = min(
            selectable,
            key=lambda item: (
                -strategic_score(item),
                item.encounter_evaluation.detour_cost if item.encounter_evaluation else float("inf"),
                item.encounter_evaluation.encounter_cost if item.encounter_evaluation else float("inf"),
                item.objective_id,
            ),
        )
        candidate = strategic_candidate(task)
        if candidate is not None:
            task = replace(task, encounter_method=candidate.method)
        return finish(
            task,
            ObjectiveStatus.READY,
            (
                "selected from currently available tasks"
                if task.encounter_evaluation is None
                or task.encounter_evaluation.recommendation is EncounterRecommendation.PREFER
                else "selected deferred encounter because no required task is executable"
            ),
        )
    return finish(None, ObjectiveStatus.COMPLETE, "no campaign task is currently available")


def _capture_policy_context(state: CampaignState | None = None) -> CapturePolicyContext:
    """Return the initial strategic context used by pure task ranking.

    The boss profile is currently the registered Emerald Roxanne slice. Party
    types are derived from the immutable observed party when available.
    """

    team_types: set[str] = set()
    captured_species: frozenset[str] = frozenset()
    open_party_slots: int | None = None
    species_clause_enabled = False
    if state is not None and state.party.is_known:
        open_party_slots = max(0, 6 - len(state.party.value or ()))
        from modules.pokemon import get_species_by_name

        for member in state.party.value or ():
            try:
                team_types.update(type_.name.lower() for type_ in get_species_by_name(member.species).types)
            except (KeyError, AttributeError, RuntimeError, TypeError, ValueError):
                continue
    if state is not None:
        species_clause_enabled = bool(getattr(state, "species_clause_enabled", False))
        captured_fact = getattr(state, "captured_species", None)
        if captured_fact is not None and captured_fact.is_known:
            captured_species = frozenset(captured_fact.value or ())
    return CapturePolicyContext(
        boss_weakness_types=frozenset({"grass", "water", "fighting", "ground", "steel"}),
        team_types=frozenset(team_types),
        species_clause_enabled=species_clause_enabled,
        captured_species=captured_species,
        open_party_slots=open_party_slots,
    )


def campaign_planning_signature(state: CampaignState) -> tuple[object, ...]:
    """Return the dynamic facts that can change the campaign selection.

    Campaign selection is map-level planning.  Avatar coordinates and current
    HP are intentionally excluded because those change during ordinary
    movement and are handled by tactical navigation/readiness instead.  The
    encounter projection, campaign facts, party composition/levels, and
    resource facts remain part of the key because each can change the
    executable frontier.
    """

    def fact_signature(fact: Fact[object]) -> tuple[object, object]:
        return (fact.status.value, repr(fact.value))

    party_signature = tuple(
        sorted(
            (
                getattr(pokemon, "party_index", None),
                getattr(pokemon, "species", None),
                getattr(pokemon, "level", None),
                getattr(pokemon, "identity", None),
                getattr(pokemon, "egg", getattr(pokemon, "is_egg", False)),
                getattr(pokemon, "current_hp", 0) > 0,
                tuple(getattr(move, "name", None) for move in getattr(pokemon, "moves", ())),
            )
            for pokemon in state.party.value or ()
        )
        if state.party.is_known
        else ()
    )
    campaign_fact_signature = tuple(
        (
            name,
            fact_signature(getattr(state.campaign_facts, name)),
        )
        for name in state.campaign_facts.__dataclass_fields__
    )
    encounter_signature = (
        state.encounters.status.value,
        (
            tuple(
                sorted(
                    (
                        encounter.location,
                        encounter.status,
                        encounter.eligible,
                        encounter.species,
                    )
                    for encounter in state.encounters.value or ()
                )
            )
            if state.encounters.is_known
            else ()
        ),
    )
    return (
        fact_signature(state.raw_map),
        fact_signature(state.badges),
        fact_signature(state.inventory),
        fact_signature(state.run_status),
        fact_signature(state.dead_pokemon),
        fact_signature(state.rules_legal),
        fact_signature(state.last_completed_battle),
        party_signature,
        encounter_signature,
        campaign_fact_signature,
        fact_signature(state.captured_species),
        state.species_clause_enabled,
        getattr(state.campaign_lifecycle, "value", state.campaign_lifecycle),
    )


def select_available_campaign_task(
    state: CampaignState,
    tasks: tuple[CampaignObjective, ...] | None = None,
    encounter_locations: tuple[tuple[int, int], ...] | None = None,
) -> ObjectiveSelection:
    """Select the deterministic best currently available task.

    Priority is intentionally simple and replaceable: required tasks precede
    optional tasks, then explicit priority and declaration order decide ties.
    """

    available = available_campaign_tasks(state, tasks, encounter_locations)
    selection = _select_available_campaign_tasks(available, state)
    # Keep the full discovered set available to the policy so a deferred
    # encounter can become valuable after a capability/story transition. The
    # human-facing frontier should describe immediate locations only; future
    # encounter locations otherwise make an early-game run look as though it
    # has executable choices that the route model explicitly rejected.
    visible = tuple(
        task.objective_id
        for task in available
        if task.task_kind != "optional"
        or task.encounter_evaluation is None
        or task.encounter_evaluation.executable_now
    )
    return replace(selection, frontier_objective_ids=visible)


def campaign_frontier_summary(
    state: CampaignState,
    objectives: tuple[CampaignObjective, ...] | None = None,
    executable_objective_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return concise completed and currently available story objectives.

    When ``executable_objective_ids`` is supplied, it is the already-resolved
    frontier from the selector and is rendered verbatim. This keeps the
    human-facing report truthful without repeating route/encounter discovery.
    Without it, the function retains its cheap declarative fallback for
    callers that do not have a selector result.
    """

    completed: list[str] = []
    available: list[str] = []
    ordered = initial_emerald_campaign() if objectives is None else objectives
    executable_ids = set(executable_objective_ids or ())
    for objective in ordered:
        completion = objective.completion.evaluate(state)
        if completion.status is FactStatus.KNOWN and completion.value is True:
            completed.append(objective.objective_id)
            continue
        if executable_objective_ids is not None:
            if objective.objective_id in executable_ids:
                available.append(objective.objective_id)
            continue
        if completion.status is not FactStatus.KNOWN or completion.value:
            continue
        if objective.failure is not None:
            failure = objective.failure.evaluate(state)
            if failure.status is not FactStatus.KNOWN or failure.value:
                continue
        prerequisites = tuple(prerequisite.evaluate(state) for prerequisite in objective.prerequisites)
        if all(result.status is FactStatus.KNOWN and result.value for result in prerequisites):
            available.append(objective.objective_id)
    if executable_objective_ids is not None:
        known_ids = {objective.objective_id for objective in ordered}
        available.extend(objective_id for objective_id in executable_objective_ids if objective_id not in known_ids)
    return tuple(completed), tuple(available)


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
    selection = _select_available_campaign_tasks(available, state)
    selection_details = {
        detail["task_id"]: detail
        for detail in selection.selection_diagnostics
        if isinstance(detail, dict) and "task_id" in detail
    }
    preferred = sorted(
        (
        task
        for task in available
        if task.task_kind == "optional"
        and task.encounter_evaluation is not None
        and task.encounter_evaluation.executable_now
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
                        "trainer_preparation_gain": task.encounter_evaluation.trainer_preparation_gain,
                        "trainer_checkpoint_id": task.encounter_evaluation.trainer_checkpoint_id,
                        "reachability_status": task.encounter_evaluation.reachability_status.value,
                        "reachability_blockers": task.encounter_evaluation.reachability_blockers,
                    }
                    if task.encounter_evaluation
                    else None
                ),
                "selected_encounter_method": (
                    selection.objective.encounter_method.value
                    if selection.objective is not None
                    and selection.objective.objective_id == task.objective_id
                    and selection.objective.encounter_method is not None
                    else None
                ),
                "selection_diagnostic": selection_details.get(task.objective_id),
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
        # Encounter tasks are optional in the story graph, but a strategically
        # useful legal encounter is an interruptible prerequisite for safe
        # progression. Let the shared frontier selector preempt this ordinary
        # story task when it has enough observed information to justify doing so.
        frontier = select_available_campaign_task(state)
        if frontier.objective is not None and frontier.objective.task_kind == "optional":
            return frontier
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

    # Recovery restocking is a mutable-resource interrupt, not a prerequisite
    # encoded into every future objective. Campaign Progression mounts this
    # dependency-aware planner directly, so it must see the safety interrupt
    # here as well as through the available-task compatibility selector.
    # Poké Ball restocking is deliberately gated later, after route encounter
    # evaluation, because a low reserve alone does not justify a Mart detour.
    if (
        objectives is None
        and _recovery_restock_is_required(state)
        and not _active_story_scene_blocks_restock(state)
        and not _restock_is_deferred_on_current_map(state, "restock_recovery_items")
    ):
        restock = restock_recovery_items_objective()
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_RESTOCK_DECISION: "
                f"selected=True objective={restock.objective_id!r} "
                f"reason='medicine reserve below lower threshold'"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_RESTOCK_DECISION",
        )
        return ObjectiveSelection(
            restock,
            ObjectiveStatus.READY,
            "selected required recovery-item restock task",
        )
    if (
        objectives is None
        and _restock_is_required(state)
        and not _active_story_scene_blocks_restock(state)
        and not _restock_is_deferred_on_current_map(state, "restock_pokeballs")
        and _pokeball_restock_is_warranted(state)
    ):
        # This interrupt must be checked before dependency recursion. The
        # ordinary story graph can reach the parent Wally/Woods objective
        # before its shared optional frontier is consulted, which used to
        # mount story travel with too few balls even though discovery had
        # already exposed the Mart task.
        restock = restock_pokeballs_objective()
        diagnostic_print(
            lambda: (
                "CAMPAIGN_POKEBALL_RESTOCK_DECISION: "
                f"selected=True objective={restock.objective_id!r} "
                "reason='planned encounter requires reserve before dependency recursion'"
            ),
            trace=True,
            prefix="CAMPAIGN_POKEBALL_RESTOCK_DECISION",
        )
        return ObjectiveSelection(
            restock,
            ObjectiveStatus.READY,
            "selected required Poké Ball restock task",
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

            frontier = select_available_campaign_task(state)
            if frontier.objective is not None and frontier.objective.task_kind == "optional":
                return frontier
            return ObjectiveSelection(
                objective,
                ObjectiveStatus.READY,
                f"resolved executable frontier for campaign goal {target.goal_id}",
                frontier.frontier_objective_ids,
                frontier.selection_diagnostics,
                frontier.encounter_portfolio,
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
