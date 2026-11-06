"""Runtime adapters for the pure campaign resource policy."""

from collections import Counter, deque, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from itertools import count
import traceback

from modules.items import InvalidItemIndexError, get_item_bag, get_item_storage, get_item_by_name
from modules.context import context
from modules.agent_control import (
    AgentAction,
    AgentActionExecutor,
    AgentActionType,
    AgentControlLoop,
    observe_agent,
    select_action,
)
from modules.interaction_state import InteractionPhase, InteractionType
from modules.memory import GameState, get_game_state
from modules.goals import (
    ActivateTrigger,
    EncounterMode,
    EngageTrainer,
    Goal,
    GoalConstraints,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    TrainerMode,
)
from modules.map_data import MapFRLG, MapRSE, PokemonCenter
from modules.map import get_map_all_tiles, get_map_data, get_map_metadata
from modules.modes.util.items import use_item_from_bag
from modules.modes.util.walking import navigate_to
from modules.menuing import parse_start_menu, scroll_to_party_menu_index
from modules.modes.util.map import find_closest_pokemon_center, pokemon_center_candidates
from modules.map_path import PathFindingError, Direction
from modules.player import get_player_avatar, get_player_location
from modules.navigation import (
    NavigationError,
    NavigationSearchLimitExceeded,
    GoalAwareNavigator,
    NavigationWorld,
    NavigationAction,
    NavigationActionType,
    NavigationPlan,
    plan_with_world_navigation,
    static_navigation_tile_snapshot,
    transition_approach_position,
    transitions_match,
)
from modules.overworld import (
    MovementState,
    OverworldObservationResult,
    TriggerObservation,
    perceive_overworld,
    prewarm_static_map_observation,
    static_map_transitions,
    static_trainer_observations,
    trainer_hazard_geometry_is_exact,
    WorldTransition,
)
from modules.world_navigation import WorldRoute, get_world_map_graph, route_with_edge_filter
from modules.modes.util.pc_interaction import PCAction, interact_with_pc
from modules.modes._interface import BotModeError
from modules.pokemon_party import get_party
from modules.pokemon_storage import get_pokemon_storage
from modules.console import (
    diagnostic_print,
    print_always,
    print_campaign_encounter_stage,
    print_recovery_attempt,
    print_recovery_failure,
    print_recovery_success,
)
from modules.tasks import task_is_active
from .campaign_status import recovery_status
from .preparation_status import describe_preparation_target
from modules.goals import SemanticTarget
from .capture_policy import EncounterMethod
from .trainer_policy import TrainerPolicyInput, choose_trainer_mode
from modules.trainer_checkpoints import KNOWN_TRAINER_CHECKPOINTS, trainer_checkpoint_for_source
from modules.trainer_readiness import assess_live_trainer_checkpoint
from .identity import PokemonIdentity
from .emerald_healing_catalog import (
    emerald_healing_source_for_destination,
    emerald_healing_sources,
)

from .resource_policy import (
    HealingResource,
    PartyResource,
    ResourceObservationStatus,
    ResourceSnapshot,
    best_hp_recovery_item,
    classify_recovery_item,
)
from .resource_policy import RecoveryCandidateDiagnostic, ResourceDecision, RouteRecovery, assess_campaign_resources
from .trainer_preparation import (
    TrainerSource,
    build_exp_budget,
    projected_trainer_experience,
    select_trainer_source,
)
from .campaign_preparation_loop import CampaignPreparationLoopController


def _recovery_agent_logger(message: str) -> None:
    """Expose only recovery route boundaries from the tactical loop.

    Recovery uses the same agent controller as ordinary campaign navigation,
    but its default diagnostic stream is intentionally filtered. Keep the
    useful route/input evidence visible without re-enabling every per-frame
    agent record (which materially slows the emulator).
    """

    visible = (
        "AGENT_ROUTE_PLAN",
        "AGENT_REPLAN",
        "AGENT_MOVE_BATCH",
        "AGENT_BATTLE_RETURN",
        "AGENT_WATCHDOG",
        "AGENT_WORLD:",
        "AGENT_TRANSITION",
        "AGENT_ACTION:",
        "AGENT_GOAL:",
        "AGENT_PLAN:",
    )
    if not message.startswith(visible):
        return
    diagnostic_print(
        lambda: f"CAMPAIGN_RECOVERY_STAGE: {message}",
        trace=True,
        prefix="CAMPAIGN_RECOVERY_STAGE",
    )


def _pulse_toward_entry(observation, destination) -> bool:
    """Issue one fresh cardinal input toward a stalled healing entrance."""
    overworld = getattr(observation, "overworld", None)
    current = getattr(overworld, "player_coordinates", None)
    target = destination[1] if isinstance(destination, tuple) and len(destination) == 2 else None
    if current is None or target is None:
        return False
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) + abs(dy) != 1:
        return False
    direction = (
        Direction.East if dx == 1 else Direction.West if dx == -1 else Direction.South if dy == 1 else Direction.North
    )
    press_direction = getattr(context.emulator, "press_direction", None)
    if callable(press_direction):
        press_direction(direction.button_name, run=False, fresh=True)
    else:
        press_button_fresh = getattr(context.emulator, "press_button_fresh", None)
        if not callable(press_button_fresh):
            return False
        press_button_fresh(direction.button_name)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_ENTRY_WARP_FALLBACK: " f"from={current!r} toward={target!r} direction={direction.name!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_ENTRY_WARP_FALLBACK",
    )
    return True


def _map_id_value(map_id):
    """Normalize map enums to the tuple emitted by overworld perception."""
    return getattr(map_id, "value", map_id)


def _normalized_location(location):
    """Normalize a map/coordinate pair for graph and catalog comparisons."""

    if not isinstance(location, tuple) or len(location) != 2:
        return location
    return _map_id_value(location[0]), location[1]


def _cataloged_healing_warp(destination, source=None) -> WorldTransition | None:
    """Return the ROM door transition for a cataloged healing source.

    The catalog's outdoor coordinate is the Center door tile.  Resolving that
    tile to its exact transition while the frame thread owns ROM observation
    lets the worker plan toward one executable door instead of searching every
    warp on the destination map.
    """

    source = source or emerald_healing_source_for_destination(destination)
    interior_map = getattr(source, "interior_map", None)
    if interior_map is None:
        return None
    destination_map = _map_id_value(interior_map)
    outdoor_location = _normalized_location(destination)
    if not isinstance(outdoor_location, tuple) or len(outdoor_location) != 2:
        return None
    try:
        # Healing-source catalog entries often use MapRSE enum members while
        # the static transition cache is keyed by raw map tuples. Normalize at
        # this frame-thread cache boundary so a valid Oldale map cannot become
        # ``KeyError: 6``.
        transitions = static_map_transitions(_map_id_value(outdoor_location[0]))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    candidates = tuple(
        transition
        for transition in transitions
        if transition.destination is not None and _normalized_location(transition.destination)[0] == destination_map
    )
    exact = tuple(
        transition
        for transition in candidates
        if _normalized_location(transition.entry) == outdoor_location
        or any(
            _normalized_location(location) == outdoor_location
            for location in getattr(transition, "activation_locations", ())
        )
    )
    if exact:
        return exact[0]
    # Most Center maps have one door warp. Keep that safe fallback for catalog
    # providers whose outdoor coordinate names the adjacent activation tile.
    return candidates[0] if len(candidates) == 1 else None


_OPPORTUNISTIC_RECOVERY_ENCOUNTER_PENALTY = 8


def _recovery_navigation_goal(
    destination,
    source=None,
    warp: WorldTransition | None = None,
    *,
    critical: bool = False,
):
    """Return the executable goal for a cataloged healing destination."""
    source = source or emerald_healing_source_for_destination(destination)
    interior_map = getattr(source, "interior_map", None)
    if interior_map is None:
        # Preserve compatibility for non-Emerald providers that expose only a
        # standable destination coordinate.
        target = ReachLocation(destination)
    else:
        selected_warp = warp
        target = ReachWarp(
            destination_map=_map_id_value(interior_map),
            destination=selected_warp.destination if selected_warp is not None else None,
            warp=selected_warp,
        )
    # Both recovery urgencies avoid undefeated trainer sight lines. Critical
    # recovery also keeps the lexicographic encounter-first objective, while
    # opportunistic recovery treats grass as a finite route cost so a nearby
    # Center can beat a much longer encounter-free detour.
    return NavigationGoal(
        target,
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        encounter_mode=EncounterMode.AVOID if critical else EncounterMode.MOSTLY_AVOID,
        encounter_penalty=_OPPORTUNISTIC_RECOVERY_ENCOUNTER_PENALTY,
    )


def _prewarm_interior_map_identity(interior_map_id) -> None:
    """Make the live map header sufficient to identify a healing interior.

    During a door warp, SaveBlock1 can retain the exterior map after the
    interior is already visible.  ``perceive_overworld`` resolves the live
    ``gMapHeader`` only against headers that have previously been cached.
    Recovery must warm its known destination before waiting on that
    observation; otherwise a stale exterior observation prevents the warp
    controller from being created, which in turn prevents the cache warm-up.
    """
    try:
        get_map_metadata(interior_map_id)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        # Metadata warming is an observation aid, not a prerequisite for a
        # valid recovery route. The normal handoff still reports a meaningful
        # navigation failure if the destination itself is unavailable.
        return


# A recovery handoff is a map-observation boundary, not a second navigation
# phase.  Bound a failed live-map observation so a bad ROM observation is
# reported instead of leaving the bot motionless forever.
_MAP_IDENTITY_RESOLUTION_TIMEOUT = 300
_RECOVERY_CATALOG_CANDIDATE_LIMIT = 4
# Recovery candidate validation is bounded speculative work. Keep both a
# geometric incumbent bound and a hard finite budget so an unreachable/future
# healing source cannot monopolize the route worker.
_RECOVERY_SEARCH_MAX_EXPANSIONS = 4000
_RECOVERY_SEARCH_MAX_ROUTE_COST = 512
# A critical recovery route may be forced to cross encounter terrain. Keep
# that fallback finite, but give it enough room to reach a known Center after
# the strict encounter-free search has exhausted its budget.
_RECOVERY_EMERGENCY_SEARCH_MAX_EXPANSIONS = _RECOVERY_SEARCH_MAX_EXPANSIONS * 2
_RECOVERY_EMERGENCY_SEARCH_MAX_ROUTE_COST = _RECOVERY_SEARCH_MAX_ROUTE_COST * 2
_RECOVERY_STATIC_CATALOG_CACHE_LIMIT = 16
_recovery_static_catalog_cache: dict[tuple, tuple[tuple, tuple]] = {}


class RecoveryPlanningError(PathFindingError):
    """A bounded recovery search failed after checking its candidates.

    Keeping the candidate records on the exception lets callers show the
    complete destination table even when no executable route exists.
    """

    def __init__(self, message: str, candidate_diagnostics=()):
        super().__init__(message)
        self.candidate_diagnostics = tuple(candidate_diagnostics)


@dataclass(frozen=True, slots=True)
class RouteRecoveryRequest:
    """Frozen inputs for the expensive portion of recovery observation.

    The request is assembled while the emulator is owned by the frame loop.
    Route search can then run against this immutable snapshot without reading
    emulator memory or publishing frame-scoped diagnostics.
    """

    location: object
    world: NavigationWorld
    graph: object
    ordered_sources: tuple[tuple[int, object], ...]
    center_location: object | None
    candidate_limit: int
    critical: bool = False
    # Resolved on the frame thread; route workers must not read ROM metadata.
    healing_warps: tuple[tuple[int, WorldTransition], ...] = ()
    # Static route inputs are also prepared on the frame thread.  The recovery
    # worker must be able to plan without touching the emulator or shared ROM
    # caches while a map transition is in flight.
    static_tile_catalog: tuple[tuple[tuple[int, int], tuple[tuple[object, object], ...]], ...] = ()
    static_transition_catalog: tuple[tuple[tuple[int, int], tuple[WorldTransition, ...]], ...] = ()

    @property
    def key(self) -> tuple:
        """Return the dynamic-world identity used to reject stale results."""
        # NavigationWorld computes these revisions while the live observation
        # is assembled.  Do not walk the full static tile mapping here: this
        # property is polled once per frame while a worker is pending.
        blocked, transitions = self.world.revision_key()
        sources = tuple(
            (
                getattr(source, "source_id", None),
                getattr(source, "outdoor_location", None),
                getattr(source, "interior_map", None),
            )
            for _, source in self.ordered_sources
        )
        healing_warps = tuple(
            (index, transition.entry, transition.destination, transition.kind)
            for index, transition in self.healing_warps
        )
        static_maps = tuple(
            map_id for map_id, _tiles in self.static_tile_catalog
        )
        static_transition_maps = tuple(
            map_id for map_id, _transitions in self.static_transition_catalog
        )
        return (
            self.location,
            self.candidate_limit,
            getattr(self.world, "facing", None),
            getattr(self.world, "running_shoes", None),
            getattr(self.world, "player_elevation", None),
            getattr(self.world, "surfing", None),
            transitions,
            blocked,
            sources,
            healing_warps,
            static_maps,
            static_transition_maps,
            self.critical,
        )


class HealingSourceType(Enum):
    """Kinds of full-party healing affordances recognized at runtime."""

    FULL_PARTY_PROVIDER = "full_party_provider"
    POKEMON_CENTER_NURSE = "pokemon_center_nurse"


@dataclass(frozen=True, slots=True)
class HealingSource:
    """Observed healing affordance with navigation and interaction metadata."""

    source_id: str
    source_type: HealingSourceType
    location: object
    interaction_target: object
    full_party: bool = True
    usable: bool = True
    requires_dialogue: bool = True
    interaction_trigger_id: str | None = None

    @property
    def navigation_goal(self) -> Goal:
        """Semantic target for reaching this source's usable interaction point."""
        if self.interaction_trigger_id is not None:
            return ReachInteractionPosition(self.interaction_trigger_id)
        if self.location is not None:
            return ReachLocation(self.location)
        raise RuntimeError(f"Healing source {self.source_id!r} has no navigation target")


# These are interaction affordances, not map types.  The decompilation uses
# the same full-party healing primitive from several scripts: a Center nurse,
# the player's mother, and the Route 111 rest stop are the ordinary
# overworld-accessible examples.  Keep the catalog keyed by script identity so
# other games/ROM capabilities can extend it without changing recovery policy.
_FULL_PARTY_HEALING_SCRIPT_NAMES = frozenset(
    {
        "PlayersHouse_1F_EventScript_Mom",
        "Route111_OldLadysRestStop_EventScript_OldLady",
        "Route119_WeatherInstitute_1F_EventScript_Bed",
        "SSTidalRooms_EventScript_Bed",
    }
)


def _healing_source_type(identity: str) -> HealingSourceType | None:
    """Classify an observed interaction script as a healing source."""

    if identity.endswith("_PokemonCenter_1F_EventScript_Nurse"):
        return HealingSourceType.POKEMON_CENTER_NURSE
    if identity in _FULL_PARTY_HEALING_SCRIPT_NAMES:
        return HealingSourceType.FULL_PARTY_PROVIDER
    return None


def discover_healing_source(location=None, observation=None) -> HealingSource | None:
    """Resolve a destination or an observed full-party healing affordance.

    A source is deliberately recognized from its observed interaction script,
    not from the building it happens to occupy.  This keeps the runtime open
    to mother/rest-stop/facility sources and lets a ROM capability add more
    script identities without teaching the planner about map geometry.
    """
    if observation is not None and observation.overworld is not None:
        for trigger in getattr(observation.overworld, "triggers", ()):
            identity = trigger.affordance_id or trigger.script_symbol or ""
            source_type = _healing_source_type(identity)
            if trigger.kind == "object_interaction" and source_type is not None and trigger.activation_locations:
                activation_location = next(iter(trigger.activation_locations), None)
                return HealingSource(
                    identity,
                    source_type,
                    activation_location,
                    trigger.trigger_id,
                    True,
                    True,
                    True,
                    trigger.trigger_id,
                )
        return None
    if location is None:
        location = get_player_location()
    try:
        center = find_closest_pokemon_center(location)
    except (BotModeError, PathFindingError, RuntimeError):
        return None
    return HealingSource(
        "nearest_full_party_source",
        HealingSourceType.FULL_PARTY_PROVIDER,
        center.value,
        center,
        True,
        True,
        True,
    )


def execute_existing_tactical_goal(goal: Goal) -> Iterator[object]:
    """Run the controller's normal tactical loop for an objective goal."""
    if isinstance(goal, ActivateTrigger) and goal.trigger_id == "introductory_rival":
        # Preserve the public campaign trigger goal while making its travel
        # policy explicit at the observation-driven execution boundary.
        goal = NavigationGoal(goal, encounter_mode=EncounterMode.AVOID)
    return AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()


def observe_resource_snapshot() -> ResourceSnapshot:
    """Read party and healing-item resources with explicit error provenance."""

    trace = getattr(context, "stutter_trace", None)
    started = trace.now() if trace is not None else 0
    party = get_party()
    bag = get_item_bag()
    storage = get_item_storage()
    unavailable = tuple(
        name for name, value in (("party", party), ("item_bag", bag), ("item_storage", storage)) if value is None
    )
    if unavailable:
        result = ResourceSnapshot(
            observation_status=ResourceObservationStatus.UNAVAILABLE,
            observation_error=";".join(f"{component}_unavailable" for component in unavailable),
            unavailable_components=unavailable,
        )
        if trace is not None:
            trace.duration("campaign_resource_observation_duration_ms", started)
            trace.mark("campaign_resource_observation_status", result.observation_status.value)
            trace.mark("campaign_resource_unavailable_components", ",".join(unavailable))
        return result
    party_resources = tuple(
        PartyResource(p.current_hp, p.total_hp, p.status_condition.value, p.current_hp <= 0) for p in party
    )
    bag_items = []
    try:
        bag_slots = bag.items
    except InvalidItemIndexError as error:
        return ResourceSnapshot(
            party_resources,
            observation_status=ResourceObservationStatus.MALFORMED,
            invalid_item_index=error.index,
            invalid_item_slot=error.slot,
            invalid_item_storage=error.storage,
        )
    for slot in bag_slots:
        if slot.item.battle_use.value == "healing" or slot.item.field_use.value == "healing":
            bag_items.append(
                HealingResource(
                    slot.item.name,
                    slot.quantity,
                    slot.item.parameter,
                    "bag",
                    kind=classify_recovery_item(slot.item.name, slot.item.parameter),
                )
            )
    pc_items = []
    try:
        storage_items = storage.items
        for slot in storage_items:
            if slot.item.battle_use.value == "healing" or slot.item.field_use.value == "healing":
                pc_items.append(
                    HealingResource(
                        slot.item.name,
                        slot.quantity,
                        slot.item.parameter,
                        "pc",
                        kind=classify_recovery_item(slot.item.name, slot.item.parameter),
                    )
                )
    except InvalidItemIndexError as error:
        result = ResourceSnapshot(
            party_resources,
            tuple(bag_items),
            (),
            observation_status=ResourceObservationStatus.MALFORMED,
            invalid_item_index=error.index,
            invalid_item_slot=error.slot,
            invalid_item_storage=error.storage,
        )
        if trace is not None:
            trace.duration("campaign_resource_observation_duration_ms", started)
            trace.mark("campaign_resource_observation_status", result.observation_status.value)
            trace.mark("campaign_resource_invalid_item_index", error.index)
        return result
    result = ResourceSnapshot(
        party_resources,
        tuple(bag_items),
        tuple(pc_items),
        observation_status=ResourceObservationStatus.VALID,
    )
    if trace is not None:
        trace.duration("campaign_resource_observation_duration_ms", started)
        trace.mark("campaign_resource_observation_status", result.observation_status.value)
    return result


def party_is_restored() -> bool:
    """Return whether every observed non-egg party member is fully restored."""

    party = get_party()
    return bool(party) and all(
        p.current_hp == p.total_hp and p.status_condition.value == "none" for p in party if not p.is_egg
    )


def execute_heal_party() -> Iterator[object]:
    """Observation/action slice for HEAL_PARTY.

    Navigation and interaction are separate observation-driven phases.  The
    source's concrete location is only used to reach it; the interaction is
    then handled one bounded action at a time by the common agent executor.
    """
    if party_is_restored():
        # A full party does not need a nurse interaction, but a Center visit
        # is still a useful opportunity to collect medicine from Emerald's
        # item PC. The helper is a no-op outside a Center, so this does not
        # create a detour merely to retrieve stored stock.
        yield from _withdraw_pc_healing_item_at_current_location()
        return
    destination_source = None
    while True:
        observation = observe_agent()
        if observation.interaction_type.name == "DIALOGUE" or not observation.interaction.controllable:
            decision = select_action(observation)
            AgentActionExecutor().execute(decision.action, observation)
            yield
            continue
        source = discover_healing_source(observation=observation)
        if source is None:
            if destination_source is not None:
                raise RuntimeError("SOURCE_UNAVAILABLE")
            destination_source = discover_healing_source(
                location=(observation.overworld.map_id, observation.overworld.player_coordinates)
            )
            if destination_source is None:
                raise RuntimeError("SOURCE_UNAVAILABLE")
            source = destination_source
        if source.interaction_trigger_id is None:
            destination = source.location
            navigation_goal = _recovery_navigation_goal(destination)
            yield from AgentControlLoop(lambda: observe_agent(goal=navigation_goal), goal=navigation_goal).run()
            destination_source = None
            source = None
            while source is None:
                yield
                interior_observation = observe_agent()
                source = discover_healing_source(observation=interior_observation)
        navigation_goal = source.navigation_goal
        yield from AgentControlLoop(
            lambda: observe_agent(goal=navigation_goal),
            goal=navigation_goal,
        ).run()
        yield from _execute_healing_source_interaction(source)
        return


def _preparation_training_location(training_map) -> tuple[tuple, tuple[tuple[int, int], ...]]:
    """Find ROM-authoritative encounter tiles on the selected training map.

    Encounter availability alone does not imply that a tile is reachable from
    the map's current entry.  Return the complete candidate set so the live
    observed world can choose a reachable tile after the map transition.
    """
    map_id = _map_id_value(training_map)
    map_data = get_map_data(map_id, (0, 0))
    candidates = tuple(sorted(tile.local_position for tile in get_map_all_tiles(map_data) if tile.has_encounters))
    if not candidates:
        raise BotModeError(f"no land encounter tile is available on training map {map_id!r}")
    return map_id, candidates


_PREPARATION_NAVIGATION_RETRY_LIMIT = 12
_ENCOUNTER_STAGE_RETRY_DELAY_MAX = 64


@dataclass(frozen=True, slots=True)
class _EncounterStageProbe:
    """One immutable result from an encounter-stage transition probe."""

    index: int
    transition: object
    approach: object | None
    status: str
    plan: NavigationPlan | None = None
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _EncounterStageBatch:
    """Cached evidence for one complete candidate batch."""

    signature: tuple
    trainer_mode: TrainerMode
    probes: tuple[_EncounterStageProbe, ...]
    shared_search: bool = False


@dataclass(frozen=True, slots=True)
class EncounterStageCandidateEvidence:
    """Stable, human-readable evidence for one encounter-stage exit."""

    index: int
    entry: object | None
    approach: object | None
    destination: object | None
    status: str
    route_cost: int | None = None
    error: str | None = None
    # Route-level facts are kept per candidate so a permissive path that
    # happens to be cheaper cannot be mistaken for a trainer-free path.
    trainer_ids: tuple[str, ...] = ()
    trainer_free: bool | None = None
    encounter_opportunities: int | None = None
    probe_mode: str | None = None


@dataclass(frozen=True, slots=True)
class EncounterStageTrainerEvidence:
    """ROM/runtime trainer geometry visible at an encounter-stage boundary."""

    trainer_id: str | None
    object_id: int | None
    location: object | None
    trainer_type: str | None
    trainer_range: int | None
    facing: object | None
    movement_type: str | None
    movement_range: tuple | None
    movement_origin: object | None
    defeated: bool | None
    hazard_locations: tuple = ()
    engagement_locations: tuple = ()
    activation_locations: tuple = ()


@dataclass(frozen=True, slots=True)
class EncounterStageEvidence:
    """The complete decision envelope for the current encounter stage.

    This is published at the capability boundary so campaign readiness can
    promote a known route trainer after the tactical stage probe returns. A
    failed safe probe is not itself permission to force exposure: unresolved
    hazards remain explicit and are consumed as unknown by campaign policy.
    """

    current_map: object | None
    next_map: object | None
    world_signature: tuple
    safe_candidates: tuple[EncounterStageCandidateEvidence, ...]
    permissive_candidates: tuple[EncounterStageCandidateEvidence, ...]
    selected_candidate: int | None
    readiness: bool | None
    mandatory_hazard_ids: tuple[str, ...] = ()
    known_hazard_ids: tuple[str, ...] = ()
    unresolved_hazard_ids: tuple[str, ...] = ()
    # Preserve the source-resolved roster facts found while probing the
    # encounter boundary. The probe can run immediately before the campaign
    # readiness pass, and the live trigger/script that supplied the fact may
    # no longer be present when that pass re-reads the overworld.
    known_hazard_checkpoints: tuple[object, ...] = ()
    selected_trainer_id: str | None = None
    trainer_evidence: tuple[EncounterStageTrainerEvidence, ...] = ()
    reason: str = ""
    blocked: bool = False
    shared_search: bool = False


def _encounter_stage_candidate_evidence(
    batch: _EncounterStageBatch,
    overworld=None,
) -> tuple[EncounterStageCandidateEvidence, ...]:
    return tuple(
        _candidate_evidence_for_probe(batch, probe, overworld)
        for probe in batch.probes
    )


def _encounter_stage_trainer_evidence(overworld) -> tuple[EncounterStageTrainerEvidence, ...]:
    """Collect deterministic trainer geometry for the normal stage report."""

    records: dict[str, dict] = {}

    def record_for(trainer_id):
        return records.setdefault(
            trainer_id,
            {
                "trainer_id": trainer_id,
                "object_id": None,
                "location": None,
                "trainer_type": None,
                "trainer_range": None,
                "facing": None,
                "movement_type": None,
                "movement_range": None,
                "movement_origin": None,
                "defeated": None,
                "hazard_locations": set(),
                "engagement_locations": set(),
                "activation_locations": set(),
            },
        )

    trainer_objects = list(tuple(getattr(overworld, "objects", ()) or ()))
    try:
        trainer_objects.extend(
            static_trainer_observations(
                getattr(overworld, "map_id", None),
                get_map_metadata(getattr(overworld, "map_id", None)),
                tuple(trainer_objects),
            )
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        # Geometry reporting is diagnostic only. A partial map boundary must
        # not turn a valid encounter-stage observation into a routing failure.
        pass

    for obj in tuple(trainer_objects):
        try:
            trainer_id = getattr(obj, "trainer_id", None)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            trainer_id = None
        if not trainer_id:
            continue
        item = record_for(trainer_id)
        item.update(
            object_id=getattr(obj, "local_id", None),
            location=getattr(obj, "location", None),
            trainer_type=getattr(obj, "trainer_type", None),
            trainer_range=getattr(obj, "trainer_range", None),
            facing=getattr(obj, "facing", None),
            movement_type=getattr(obj, "movement_type", None),
            movement_range=getattr(obj, "movement_range", None),
            movement_origin=getattr(obj, "movement_origin", None),
            defeated=getattr(obj, "trainer_defeated", None),
        )

    for trigger in tuple(getattr(overworld, "triggers", ()) or ()):
        if getattr(trigger, "hazard_kind", None) not in {"trainer", "trainer_hazard"}:
            continue
        trainer_id = getattr(trigger, "affordance_id", None) or getattr(trigger, "trigger_id", None)
        if not trainer_id:
            continue
        item = record_for(trainer_id)
        item["hazard_locations"].update(getattr(trigger, "hazard_locations", ()) or ())
        item["engagement_locations"].update(getattr(trigger, "engagement_locations", ()) or ())
        item["activation_locations"].update(getattr(trigger, "activation_locations", ()) or ())
        if item["defeated"] is None:
            item["defeated"] = getattr(trigger, "trainer_defeated", None)
        trigger_id = str(getattr(trigger, "trigger_id", ""))
        if item["object_id"] is None and trigger_id.startswith("static_object:"):
            try:
                item["object_id"] = int(trigger_id.split(":", 2)[1])
            except (IndexError, TypeError, ValueError):
                pass
        if item["location"] is None:
            locations = tuple(getattr(trigger, "locations", ()) or ())
            item["location"] = locations[0] if locations else None

    return tuple(
        EncounterStageTrainerEvidence(
            trainer_id=item["trainer_id"],
            object_id=item["object_id"],
            location=item["location"],
            trainer_type=item["trainer_type"],
            trainer_range=item["trainer_range"],
            facing=item["facing"],
            movement_type=item["movement_type"],
            movement_range=item["movement_range"],
            movement_origin=item["movement_origin"],
            defeated=item["defeated"],
            hazard_locations=tuple(sorted(item["hazard_locations"], key=repr)),
            engagement_locations=tuple(sorted(item["engagement_locations"], key=repr)),
            activation_locations=tuple(sorted(item["activation_locations"], key=repr)),
        )
        for _trainer_id, item in sorted(records.items(), key=lambda pair: repr(pair[0]))
    )


def _candidate_evidence_for_probe(
    batch: _EncounterStageBatch,
    probe: _EncounterStageProbe,
    overworld,
) -> EncounterStageCandidateEvidence:
    metrics = getattr(probe.plan, "metrics", None)
    trainer_ids = (
        _encounter_stage_trainer_ids(overworld, probe.plan)
        if overworld is not None and probe.plan is not None
        else ()
    )
    return EncounterStageCandidateEvidence(
        index=probe.index,
        entry=getattr(probe.transition, "entry", None),
        approach=probe.approach,
        destination=getattr(probe.transition, "destination", None),
        status=probe.status,
        route_cost=getattr(metrics, "total_route_cost", None),
        error=probe.error,
        trainer_ids=trainer_ids,
        trainer_free=(not trainer_ids if probe.plan is not None else None),
        encounter_opportunities=getattr(metrics, "encounter_opportunities", None),
        probe_mode=batch.trainer_mode.name,
    )


_encounter_stage_request_ids = count(1)
_encounter_stage_probe_cache: OrderedDict[tuple, _EncounterStageBatch] = OrderedDict()
_MAX_ENCOUNTER_STAGE_PROBE_CACHE = 128


def _encounter_stage_transition_signature(transition) -> tuple:
    """Return executable transition facts for stage diagnostics and caching."""

    return (
        getattr(transition, "kind", None),
        getattr(transition, "entry", None),
        transition_approach_position(transition),
        getattr(transition, "destination", None),
        getattr(transition, "required_facing", None),
        tuple(sorted(getattr(transition, "activation_locations", ()) or (), key=repr)),
        getattr(transition, "activation_direction", None),
    )


def _encounter_stage_world_signature(overworld, transitions, next_map) -> tuple:
    """Build a complete topology/blocker key for one staging observation."""

    staged_overworld = replace(overworld, transitions=tuple(transitions))
    world = NavigationWorld.from_overworld(staged_overworld)
    blocked, transition_revision = world.revision_key()
    hazards = tuple(
        sorted(
            (
                (
                    getattr(trigger, "affordance_id", None),
                    getattr(trigger, "trainer_defeated", None),
                    tuple(sorted(getattr(trigger, "hazard_locations", ()) or (), key=repr)),
                    tuple(sorted(getattr(trigger, "engagement_locations", ()) or (), key=repr)),
                )
                for trigger in getattr(overworld, "triggers", ()) or ()
            ),
            key=repr,
        )
    )
    return (
        getattr(overworld, "map_id", None),
        getattr(overworld, "player_coordinates", None),
        next_map,
        tuple(_encounter_stage_transition_signature(transition) for transition in transitions),
        blocked,
        transition_revision,
        hazards,
    )


def _encounter_stage_probe_cache_get(key, trainer_mode) -> _EncounterStageBatch | None:
    cache_key = (key, trainer_mode)
    batch = _encounter_stage_probe_cache.get(cache_key)
    if batch is not None:
        _encounter_stage_probe_cache.move_to_end(cache_key)
    return batch


def _encounter_stage_probe_cache_put(batch: _EncounterStageBatch) -> None:
    cache_key = (batch.signature, batch.trainer_mode)
    _encounter_stage_probe_cache[cache_key] = batch
    _encounter_stage_probe_cache.move_to_end(cache_key)
    while len(_encounter_stage_probe_cache) > _MAX_ENCOUNTER_STAGE_PROBE_CACHE:
        _encounter_stage_probe_cache.popitem(last=False)


def _encounter_stage_battle_ready() -> bool | None:
    """Return only affirmative readiness for an explicitly forced exposure."""

    for candidate in (
        getattr(context, "campaign_trainer_readiness", None),
        getattr(context, "battle_readiness", None),
    ):
        ready = getattr(candidate, "ready", None)
        if ready is not None:
            return ready
    return None


def _encounter_stage_route_cut_hazards(
    overworld,
    *,
    safe_candidates=None,
    permissive_candidates=None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return known and unresolved trainers common to every usable exit.

    A blocked AVOID search is not enough to promote a trainer: one exit may
    be blocked by Cindy while another is blocked by Billy. Only the
    intersection of all usable IGNORE exits is a route cut for this stage.
    Source resolution is kept here so an unresolved common trainer remains an
    explicit unknown instead of becoming an invented mandatory checkpoint.

    A failed AVOID search is proof of a route cut only when every hazard that
    could have blocked it is exact.  A trainer whose hazard area is partly a
    guess (unknown facing, over-covered walker) or whose geometry is not
    observed at all stays unresolved, so a pessimistic model cannot promote an
    avoidable trainer into a mandatory fight.
    """

    evidence = getattr(context, "campaign_encounter_stage", None)
    if safe_candidates is None:
        safe_candidates = getattr(evidence, "safe_candidates", ()) if evidence is not None else ()
    if permissive_candidates is None:
        permissive_candidates = (
            getattr(evidence, "permissive_candidates", ()) if evidence is not None else ()
        )

    usable_safe = tuple(
        candidate
        for candidate in tuple(safe_candidates or ())
        if getattr(candidate, "status", None) == "found"
    )
    if usable_safe:
        return (), ()
    usable_permissive = tuple(
        candidate
        for candidate in tuple(permissive_candidates or ())
        if getattr(candidate, "status", None) == "found"
    )
    if not usable_permissive:
        return (), ()

    common = set(getattr(usable_permissive[0], "trainer_ids", ()) or ())
    for candidate in usable_permissive[1:]:
        common.intersection_update(getattr(candidate, "trainer_ids", ()) or ())
    common.discard(None)
    known = []
    unresolved = []
    exact_geometry = {
        record.trainer_id: trainer_hazard_geometry_is_exact(
            record.trainer_type, record.movement_type, record.facing
        )
        for record in (_encounter_stage_trainer_evidence(overworld) if common else ())
    }
    for trainer_id in sorted(common, key=repr):
        checkpoint = trainer_checkpoint_for_source(
            trainer_id,
            script_symbol=_encounter_stage_trainer_script(overworld, trainer_id),
        )
        if checkpoint is not None and exact_geometry.get(trainer_id, False):
            known.append(trainer_id)
        else:
            unresolved.append(trainer_id)
    return tuple(known), tuple(unresolved)


def _encounter_stage_mandatory_hazards(
    overworld,
    *,
    safe_candidates=None,
    permissive_candidates=None,
) -> tuple[str, ...]:
    """Return registry and route-cut checkpoint identities for this stage."""

    mandatory_ids = {
        getattr(checkpoint, "trainer_affordance_id", None)
        for checkpoint in KNOWN_TRAINER_CHECKPOINTS
        if getattr(checkpoint, "mandatory", False)
        and getattr(checkpoint, "map_id", None) == getattr(overworld, "map_id", None)
    }
    route_cut_known, _route_cut_unresolved = _encounter_stage_route_cut_hazards(
        overworld,
        safe_candidates=safe_candidates,
        permissive_candidates=permissive_candidates,
    )
    mandatory_ids.update(route_cut_known)
    visible = {
        getattr(trigger, "affordance_id", None)
        for trigger in getattr(overworld, "triggers", ()) or ()
        if getattr(trigger, "affordance_id", None) in mandatory_ids
        and getattr(trigger, "trainer_defeated", None) is not True
        and (
            getattr(trigger, "hazard_locations", ())
            or getattr(trigger, "engagement_locations", ())
            or getattr(trigger, "activation_locations", ())
        )
    }
    return tuple(sorted(visible.union(route_cut_known), key=repr))


def _encounter_stage_trainer_script(overworld, trainer_id: str | None) -> str | None:
    """Return the ROM script paired with a live/static trainer affordance."""

    if not trainer_id:
        return None
    for obj in getattr(overworld, "objects", ()) or ():
        if getattr(obj, "trainer_id", None) == trainer_id:
            return getattr(obj, "script", None)
    for trigger in getattr(overworld, "triggers", ()) or ():
        if getattr(trigger, "affordance_id", None) != trainer_id:
            continue
        script = getattr(trigger, "script_symbol", None)
        if script:
            return script
        trigger_id = getattr(trigger, "trigger_id", "")
        if isinstance(trigger_id, str) and trigger_id.startswith("static_object:"):
            parts = trigger_id.split(":", 2)
            return parts[2] if len(parts) == 3 else None
    return None


def _encounter_stage_route_locations(plan) -> frozenset:
    if plan is None:
        return frozenset()
    locations = set()
    for action in getattr(plan, "actions", ()) or ():
        for location in (getattr(action, "source", None), getattr(action, "destination", None)):
            if isinstance(location, tuple) and len(location) == 2:
                locations.add(_normalized_location(location))
    destination = getattr(plan, "destination", None)
    if isinstance(destination, tuple) and len(destination) == 2:
        locations.add(_normalized_location(destination))
    return frozenset(locations)


def _encounter_stage_trainer_ids(overworld, plan) -> tuple[str, ...]:
    """Find active trainer hazards intersecting the permissive stage route."""

    route_locations = _encounter_stage_route_locations(plan)
    if not route_locations:
        return ()
    trainer_ids = []
    for trigger in getattr(overworld, "triggers", ()) or ():
        trainer_id = getattr(trigger, "affordance_id", None)
        if not trainer_id:
            continue
        is_trainer = getattr(trigger, "hazard_kind", None) in {"trainer", "trainer_hazard"} or str(
            trainer_id
        ).startswith(("trainer:", "rom_trainer:"))
        if not is_trainer or getattr(trigger, "trainer_defeated", None) is True:
            continue
        geometry = tuple(
            getattr(trigger, field, ()) or ()
            for field in ("hazard_locations", "engagement_locations", "activation_locations", "locations")
        )
        normalized_geometry = {
            _normalized_location(location)
            for locations in geometry
            for location in locations
            if isinstance(location, tuple) and len(location) == 2
        }
        if normalized_geometry.intersection(route_locations):
            trainer_ids.append(trainer_id)
    return tuple(sorted(set(trainer_ids), key=repr))


def _publish_encounter_stage_evidence(
    overworld,
    next_map,
    safe_batch: _EncounterStageBatch,
    permissive_batch: _EncounterStageBatch | None,
    *,
    selected_candidate: int | None,
    readiness: bool | None,
    mandatory_hazards: tuple[str, ...] = (),
    reason: str,
    blocked: bool,
) -> EncounterStageEvidence:
    safe_candidates = _encounter_stage_candidate_evidence(safe_batch, overworld)
    permissive_candidates = (
        _encounter_stage_candidate_evidence(permissive_batch, overworld)
        if permissive_batch is not None
        else ()
    )
    known_hazards, unresolved_hazards = _encounter_stage_route_cut_hazards(
        overworld,
        safe_candidates=safe_candidates,
        permissive_candidates=permissive_candidates,
    )
    known_hazard_checkpoints = []
    for trainer_id in known_hazards:
        checkpoint = trainer_checkpoint_for_source(
            trainer_id,
            script_symbol=_encounter_stage_trainer_script(overworld, trainer_id),
        )
        if checkpoint is not None:
            known_hazard_checkpoints.append(checkpoint)
    evidence = EncounterStageEvidence(
        current_map=getattr(overworld, "map_id", None),
        next_map=next_map,
        world_signature=safe_batch.signature,
        safe_candidates=safe_candidates,
        permissive_candidates=permissive_candidates,
        selected_candidate=selected_candidate,
        readiness=readiness,
        mandatory_hazard_ids=tuple(sorted(set(mandatory_hazards), key=repr)),
        known_hazard_ids=tuple(known_hazards),
        unresolved_hazard_ids=tuple(unresolved_hazards),
        known_hazard_checkpoints=tuple(known_hazard_checkpoints),
        selected_trainer_id=known_hazards[0] if known_hazards else (unresolved_hazards[0] if unresolved_hazards else None),
        trainer_evidence=_encounter_stage_trainer_evidence(overworld),
        reason=reason,
        blocked=blocked,
        shared_search=bool(safe_batch.shared_search or getattr(permissive_batch, "shared_search", False)),
    )
    previous = getattr(context, "campaign_encounter_stage", None)
    context.campaign_encounter_stage = evidence
    if previous != evidence:
        print_campaign_encounter_stage(evidence)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_ENCOUNTER_STAGE_EVIDENCE: "
            f"current_map={evidence.current_map!r} next_map={evidence.next_map!r} "
            f"blocked={evidence.blocked!r} safe={[(item.index, item.status, item.trainer_ids, item.encounter_opportunities) for item in evidence.safe_candidates]!r} "
            f"permissive={[(item.index, item.status, item.trainer_ids, item.encounter_opportunities) for item in evidence.permissive_candidates]!r} "
            f"known_hazards={evidence.known_hazard_ids!r} unresolved_hazards={evidence.unresolved_hazard_ids!r} "
            f"trainer_evidence={evidence.trainer_evidence!r} "
            f"reason={evidence.reason!r} shared_search={evidence.shared_search!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_ENCOUNTER_STAGE_EVIDENCE",
    )
    return evidence


def _log_encounter_stage_probe(request_id, batch: _EncounterStageBatch, *, cache_hit: bool) -> None:
    """Emit one bounded, reviewable record for every stage candidate."""

    world_signature = batch.signature
    for probe in batch.probes:
        transition = probe.transition
        diagnostic_print(
            lambda probe=probe, transition=transition: (
                "CAMPAIGN_ENCOUNTER_STAGE_PROBE: "
                f"request_id={request_id} cache_hit={cache_hit!r} mode={batch.trainer_mode.name!r} "
                f"candidate={probe.index} status={probe.status!r} "
                f"entry={getattr(transition, 'entry', None)!r} approach={probe.approach!r} "
                f"destination={getattr(transition, 'destination', None)!r} "
                f"error_type={probe.error_type!r} error={probe.error!r} "
                f"blocked={world_signature[4]!r} transition_revision={world_signature[5]!r} "
                f"hazards={world_signature[6]!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_ENCOUNTER_STAGE_PROBE",
        )


def _shared_encounter_stage_routes(overworld, transitions, next_map, trainer_mode):
    """Resolve map-connection candidates with one local multi-target search."""

    if not transitions or not all(getattr(item, "kind", None) == "map_connection" for item in transitions):
        return None
    approaches = tuple(transition_approach_position(item) for item in transitions)
    if any(approach is None for approach in approaches):
        return None
    staged_overworld = replace(overworld, transitions=tuple(transitions))
    world = NavigationWorld.from_overworld(staged_overworld)
    local_world = replace(world, transitions=(), warps=())
    start = (overworld.map_id, overworld.player_coordinates)
    constraints = GoalConstraints(trainer_mode=trainer_mode)
    local_plans = GoalAwareNavigator(local_world).plan_many_locations(
        start,
        approaches,
        target_groups=tuple((approach,) for approach in approaches),
        encounter_mode=EncounterMode.AVOID,
        constraints=constraints,
    )
    results = []
    navigator = GoalAwareNavigator(local_world)
    for index, (transition, approach) in enumerate(zip(transitions, approaches)):
        local_plan = local_plans.get(approach)
        if local_plan is None:
            results.append(
                _EncounterStageProbe(index, transition, approach, "not_found", error="shared local search found no approach")
            )
            continue
        direction = getattr(transition, "required_facing", None)
        if direction is None:
            results.append(
                _EncounterStageProbe(index, transition, approach, "not_found", error="map connection has no crossing direction")
            )
            continue
        boundary_move = NavigationAction(
            NavigationActionType.MOVE,
            direction,
            approach,
            transition.entry,
        )
        crossing = NavigationAction(
            NavigationActionType.WARP,
            direction,
            transition.entry,
            transition.destination,
            transition_kind="map_connection",
        )
        actions = local_plan.actions + (boundary_move, crossing)
        plan = NavigationPlan(
            actions,
            transition.entry,
            navigator._metrics(actions, EncounterMode.AVOID),
        )
        results.append(_EncounterStageProbe(index, transition, approach, "found", plan=plan))
    return tuple(results)


def _active_preparation_level_cap_for_runtime() -> int | None:
    """Read the active hard cap for trainer-source selection.

    Preparation selection is allowed to operate only with a known cap.  A
    missing campaign projection therefore returns ``None`` and the caller
    defers rather than authorizing an unbounded trainer battle.
    """

    try:
        from .campaign_controller import runtime_campaign_state
        from .level_cap import assess_level_cap
        from .rule_config import CampaignRuleId

        runtime = getattr(context, "nuzlocke_runtime", None)
        rule_config = getattr(runtime, "rule_config", None)
        if rule_config is None or not rule_config.is_enabled(CampaignRuleId.LEVEL_CAP):
            return None
        assessment = assess_level_cap(runtime_campaign_state().campaign_facts, get_party())
        return assessment.level_cap if assessment.status.value == "known" else None
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _is_gym_map(map_id) -> bool:
    """Recognize gym maps from the ROM map enum without name guessing."""

    try:
        return "GYM" in MapFRLG(map_id).name
    except (AttributeError, KeyError, TypeError, ValueError):
        try:
            return "GYM" in MapRSE(map_id).name
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


def _encounter_stage_goal(overworld, transition, next_map, trainer_mode):
    """Build the exact local goal for one ROM-backed encounter exit."""

    live_transitions = tuple(getattr(overworld, "transitions", ()) or overworld.warps)
    if not any(transitions_match(transition, candidate) for candidate in live_transitions):
        live_transitions += (transition,)
    staged_overworld = replace(overworld, transitions=live_transitions)
    goal = NavigationGoal(
        ReachWarp(
            destination_map=next_map,
            destination=transition.destination,
            warp=transition,
        ),
        constraints=GoalConstraints(trainer_mode=trainer_mode),
        encounter_mode=EncounterMode.AVOID,
    )
    return staged_overworld, goal


def _probe_encounter_stage_route(overworld, transition, next_map, trainer_mode):
    """Probe one exit without sending input to the emulator."""

    staged_overworld, goal = _encounter_stage_goal(overworld, transition, next_map, trainer_mode)
    plan, _ = plan_with_world_navigation(
        NavigationWorld.from_overworld(staged_overworld),
        (overworld.map_id, overworld.player_coordinates),
        goal,
    )
    return plan


def _encounter_stage_probe_batch(overworld, transitions, next_map, trainer_mode) -> _EncounterStageBatch:
    """Probe one transition set once, preferring a shared local search."""

    signature = _encounter_stage_world_signature(overworld, transitions, next_map)
    cached = _encounter_stage_probe_cache_get(signature, trainer_mode)
    if cached is not None:
        return cached

    shared = False
    probes = None
    if len(transitions) > 1:
        try:
            probes = _shared_encounter_stage_routes(overworld, transitions, next_map, trainer_mode)
            shared = probes is not None
        except (NavigationError, PathFindingError, RuntimeError, TypeError, ValueError):
            # A partial or synthetic transition table may not support the
            # shared local adapter. Preserve the per-candidate diagnostics and
            # compatibility behavior by falling back to exact probes below.
            probes = None

    if probes is None:
        records = []
        for index, transition in enumerate(transitions):
            approach = transition_approach_position(transition)
            try:
                plan = _probe_encounter_stage_route(overworld, transition, next_map, trainer_mode)
            except (NavigationError, PathFindingError, RuntimeError, TypeError, ValueError) as error:
                records.append(
                    _EncounterStageProbe(
                        index,
                        transition,
                        approach,
                        "not_found",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                )
                continue
            if getattr(plan, "forced_trainer_exposure", False):
                records.append(
                    _EncounterStageProbe(
                        index,
                        transition,
                        approach,
                        "forced_exposure",
                        plan=plan,
                        error="safe route requires trainer exposure",
                    )
                )
            else:
                records.append(_EncounterStageProbe(index, transition, approach, "found", plan=plan))
        probes = tuple(records)

    batch = _EncounterStageBatch(signature, trainer_mode, tuple(probes), shared_search=shared)
    _encounter_stage_probe_cache_put(batch)
    return batch


def _cached_encounter_stage_plan(overworld, transitions, next_map, transition, trainer_mode):
    """Return the plan that justified the selected encounter-stage exit.

    Stage selection already performs the safe route search.  Reusing its
    immutable plan avoids launching a second search when the execution loop
    is mounted.  The caller still gives the plan to ``AgentControlLoop``,
    whose normal origin, live-blocker, and trainer-hazard checks validate it
    against the first runtime observation.
    """

    signature = _encounter_stage_world_signature(overworld, transitions, next_map)
    modes = [trainer_mode]
    # An explicitly promoted trainer route is selected from the permissive
    # IGNORE probe, while the execution policy is ENGAGE.  Keep that mapping
    # local to the cache lookup so the public stage-selection contract remains
    # a three-value tuple.
    if trainer_mode is TrainerMode.ENGAGE:
        modes.append(TrainerMode.IGNORE)
    for mode in modes:
        batch = _encounter_stage_probe_cache_get(signature, mode)
        if batch is None:
            continue
        probe = next(
            (
                candidate
                for candidate in batch.probes
                if transitions_match(getattr(candidate, "transition", None), transition)
                and getattr(candidate, "status", None) == "found"
                and getattr(candidate, "plan", None) is not None
            ),
            None,
        )
        if probe is not None:
            return probe.plan
    return None


def _best_encounter_stage_probe(batch: _EncounterStageBatch):
    """Return the safest, then least-cost usable probe in a batch.

    Each probe is planned independently, so the route search's lexicographic
    encounter objective does not rank different exits against one another.
    Keep that same policy at the stage boundary: a trainer-free grass
    corridor may be longer than a paved route that approaches a trainer, and
    the encounter count must be compared before practical route cost.
    """

    usable = tuple(probe for probe in batch.probes if probe.status == "found" and probe.plan is not None)
    return min(
        usable,
        key=lambda probe: (
            getattr(getattr(probe.plan, "metrics", None), "encounter_opportunities", float("inf")),
            getattr(getattr(probe.plan, "metrics", None), "total_route_cost", float("inf")),
            getattr(getattr(probe.plan, "metrics", None), "movement_actions", float("inf")),
            probe.index,
        ),
        default=None,
    )


def _encounter_stage_lookahead_candidate(
    overworld,
    transitions,
    next_map,
    final_map,
    graph,
):
    """Return the current exit selected by a safe full encounter route.

    A locally cheapest map connection can deposit the player at an entry tile
    from which the next trainer-free segment is unreachable.  Encounter stages
    therefore use the semantic planner's weighted portal route when the
    current segment is only an intermediate map.  The returned index is still
    checked against the current stage's independent AVOID probe before it is
    accepted for execution.
    """

    if final_map is None or final_map == next_map or final_map == getattr(overworld, "map_id", None):
        return None
    start = (getattr(overworld, "map_id", None), getattr(overworld, "player_coordinates", None))
    if start[0] is None or start[1] is None:
        return None
    try:
        staged_overworld = replace(overworld, transitions=tuple(transitions))
        world = NavigationWorld.from_overworld(staged_overworld)
        goal = NavigationGoal(
            SemanticTarget.map(final_map),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            encounter_mode=EncounterMode.AVOID,
        )
        _plan, route = plan_with_world_navigation(
            world,
            start,
            goal,
            graph=graph,
            caller="campaign.encounter_stage.lookahead",
        )
    except (NavigationError, NavigationSearchLimitExceeded, PathFindingError, RuntimeError, TypeError, ValueError):
        return None
    selected = tuple(getattr(route, "selected_transitions", ()) or ()) if route is not None else ()
    if not selected:
        return None
    first = selected[0]
    return next(
        (
            index
            for index, transition in enumerate(transitions)
            if transitions_match(first, transition)
        ),
        None,
    )


def _encounter_stage_trainer_mode(overworld, transitions, next_map, *, final_map=None, graph=None):
    """Choose the trainer policy for one cross-map encounter stage.

    Encounter acquisition should preserve an available first encounter, but
    it must not deadlock when an undefeated trainer's sight line partitions
    the map. Probe every ROM-backed exit to the destination, preferring a
    route that avoids trainers. If no safe exit exists, the policy is allowed
    to let a trainer battle happen as part of normal travel; the campaign
    battle listener will own the battle and readiness will be checked again
    when it ends.

    Return the selected transition along with its policy so callers do not
    probe one exit and then execute a different, arbitrarily chosen exit.
    """
    current_map = getattr(overworld, "map_id", None)
    current_coordinates = getattr(overworld, "player_coordinates", None)
    if current_map is None or current_coordinates is None:
        return transitions[0], choose_trainer_mode(TrainerPolicyInput(seeking_encounter=True)), "position_unavailable"

    request_id = next(_encounter_stage_request_ids)
    safe_signature = _encounter_stage_world_signature(overworld, transitions, next_map)
    safe_cached = _encounter_stage_probe_cache_get(safe_signature, TrainerMode.AVOID)
    safe_batch = _encounter_stage_probe_batch(overworld, transitions, next_map, TrainerMode.AVOID)
    _log_encounter_stage_probe(request_id, safe_batch, cache_hit=safe_cached is not None)
    safe_probe = _best_encounter_stage_probe(safe_batch)
    lookahead_index = _encounter_stage_lookahead_candidate(
        overworld,
        transitions,
        next_map,
        final_map,
        graph,
    )
    lookahead_probe = next(
        (
            probe
            for probe in safe_batch.probes
            if probe.index == lookahead_index and probe.status == "found" and probe.plan is not None
        ),
        None,
    )
    if lookahead_probe is not None:
        _publish_encounter_stage_evidence(
            overworld,
            next_map,
            safe_batch,
            None,
            selected_candidate=lookahead_probe.index,
            readiness=_encounter_stage_battle_ready(),
            reason=(
                f"safe_route_available:lookahead_candidate={lookahead_probe.index}:"
                f"shared_search={safe_batch.shared_search!r}"
            ),
            blocked=False,
        )
        return (
            lookahead_probe.transition,
            choose_trainer_mode(TrainerPolicyInput(seeking_encounter=True)),
            f"safe_route_available:lookahead_candidate={lookahead_probe.index}:"
            f"shared_search={safe_batch.shared_search!r}",
        )
    if safe_probe is not None:
        _publish_encounter_stage_evidence(
            overworld,
            next_map,
            safe_batch,
            None,
            selected_candidate=safe_probe.index,
            readiness=_encounter_stage_battle_ready(),
            reason=(
                f"safe_route_available:candidate={safe_probe.index}:"
                f"shared_search={safe_batch.shared_search!r}"
            ),
            blocked=False,
        )
        return (
            safe_probe.transition,
            choose_trainer_mode(TrainerPolicyInput(seeking_encounter=True)),
            f"safe_route_available:candidate={safe_probe.index}:shared_search={safe_batch.shared_search!r}",
        )

    # A second probe under IGNORE is diagnostic evidence only. It distinguishes
    # an actual geometry failure from a trainer hazard/promotion problem and
    # lets a known-ready party explicitly opt into an unavoidable battle.
    permissive_signature = _encounter_stage_world_signature(overworld, transitions, next_map)
    permissive_cached = _encounter_stage_probe_cache_get(permissive_signature, TrainerMode.IGNORE)
    permissive_batch = _encounter_stage_probe_batch(overworld, transitions, next_map, TrainerMode.IGNORE)
    _log_encounter_stage_probe(request_id, permissive_batch, cache_hit=permissive_cached is not None)
    permissive_probe = _best_encounter_stage_probe(permissive_batch)
    if permissive_probe is None:
        _publish_encounter_stage_evidence(
            overworld,
            next_map,
            safe_batch,
            permissive_batch,
            selected_candidate=None,
            readiness=_encounter_stage_battle_ready(),
            reason=(
                "stage_unavailable:avoid_and_ignore_unreachable; "
                f"candidate_count={len(transitions)}"
            ),
            blocked=True,
        )
        return None, TrainerMode.AVOID, (
            "stage_unavailable:avoid_and_ignore_unreachable; "
            f"candidate_count={len(transitions)}"
        )

    readiness = _encounter_stage_battle_ready()
    safe_candidates = _encounter_stage_candidate_evidence(safe_batch, overworld)
    permissive_candidates = _encounter_stage_candidate_evidence(permissive_batch, overworld)
    mandatory_hazards = _encounter_stage_mandatory_hazards(
        overworld,
        safe_candidates=safe_candidates,
        permissive_candidates=permissive_candidates,
    )
    route_cut_known, route_cut_unresolved = _encounter_stage_route_cut_hazards(
        overworld,
        safe_candidates=safe_candidates,
        permissive_candidates=permissive_candidates,
    )
    if readiness is True and mandatory_hazards:
        reason = (
            f"forced_trainer_exposure:avoid_unavailable_ignore_available:candidate={permissive_probe.index}:"
            f"mandatory={mandatory_hazards!r}"
            + (
                ":evidence=route_cut_common_to_all_permissive_exits"
                if route_cut_known
                else ""
            )
        )
        _publish_encounter_stage_evidence(
            overworld,
            next_map,
            safe_batch,
            permissive_batch,
            selected_candidate=permissive_probe.index,
            readiness=readiness,
            mandatory_hazards=mandatory_hazards,
            reason=reason,
            blocked=False,
        )
        mode = choose_trainer_mode(
            TrainerPolicyInput(
                trainer_required=True,
                trainer_blocks_route=True,
                party_can_survive=True,
                strategic_promotion=True,
            )
        )
        return (
            permissive_probe.transition,
            mode,
            reason,
        )

    if readiness is False:
        reason = "stage_unavailable:trainer_exposure_requires_preparation"
    elif not mandatory_hazards:
        reason = "stage_unavailable:optional_or_unknown_trainer_exposure_requires_explicit_promotion"
        if route_cut_unresolved:
            reason += ":unknown_route_cut=" + repr(route_cut_unresolved)
    else:
        reason = "stage_unavailable:trainer_exposure_readiness_unknown"
    _publish_encounter_stage_evidence(
        overworld,
        next_map,
        safe_batch,
        permissive_batch,
        selected_candidate=permissive_probe.index,
        readiness=readiness,
        mandatory_hazards=mandatory_hazards,
        reason=reason,
        blocked=True,
    )
    return None, TrainerMode.AVOID, reason


def _reachable_local_positions(world, start, training_map):
    """Return the current map component reachable from ``start``.

    Preparation used to run a full weighted path search once per grass tile.
    Route 116 exposes enough encounter tiles for that repeated search to
    monopolize an application frame.  A plain ReachLocation goal has no
    route preference or interaction-facing requirement, so its candidate
    membership is exactly the walkable component produced by the observed
    world neighbors.  Keep non-``NavigationWorld`` values permissive for
    lightweight callers and existing test doubles.
    """
    if not isinstance(world, NavigationWorld):
        return None
    if not isinstance(start, tuple) or len(start) != 2 or start[0] != training_map:
        return frozenset()
    pending = deque((start,))
    reachable = {start[1]}
    while pending:
        current = pending.popleft()
        for _direction, destination, _is_warp in world.neighbors(current):
            if destination[0] != training_map or destination[1] in reachable:
                continue
            reachable.add(destination[1])
            pending.append(destination)
    return frozenset(reachable)


def _mandatory_checkpoint_trigger_ids(overworld, training_map) -> frozenset[str]:
    """Return ROM trigger IDs that can activate a mandatory checkpoint.

    Story checkpoints are not always represented as trainer objects.  In
    Petalburg Woods the researcher is a non-trainer object and the scene is
    entered through two coordinate events whose script symbols select the
    left/right branch.  Keep those event IDs in the encounter route's hard
    avoidance set so a grass search can route around the scene tiles.
    """

    checkpoint_symbols = {
        symbol
        for checkpoint in KNOWN_TRAINER_CHECKPOINTS
        if getattr(checkpoint, "mandatory", False)
        and getattr(checkpoint, "map_id", None) == training_map
        for symbol in (
            getattr(checkpoint, "script_symbol", None),
            *getattr(checkpoint, "alternate_script_symbols", ()),
        )
        if symbol
    }
    checkpoint_ids = {
        getattr(checkpoint, "trainer_affordance_id", None)
        for checkpoint in KNOWN_TRAINER_CHECKPOINTS
        if getattr(checkpoint, "mandatory", False)
        and getattr(checkpoint, "map_id", None) == training_map
        and getattr(checkpoint, "trainer_affordance_id", None)
    }
    result = set()
    for trigger in _mandatory_checkpoint_triggers(overworld, training_map):
        trigger_id = getattr(trigger, "trigger_id", None)
        if not trigger_id:
            continue
        identities = (
            getattr(trigger, "trigger_id", None),
            getattr(trigger, "affordance_id", None),
            getattr(trigger, "script_symbol", None),
        )
        if checkpoint_ids.intersection(identities) or checkpoint_symbols.intersection(identities):
            result.add(trigger_id)
    return frozenset(result)


def _mandatory_checkpoint_triggers(overworld, training_map) -> tuple:
    """Return live and ROM-static triggers for mandatory story checkpoints.

    Coordinate events are part of the map's ROM metadata, but they are not
    guaranteed to survive as live affordances in every settled observation.
    In particular, the Petalburg Woods researcher scene can be absent from
    ``overworld.triggers`` while the player is standing at the Woods entrance.
    That absence is a camera/observation boundary, not evidence that the
    scene has no geometry.  Reconstruct only the checkpoint-owned coordinate
    events from the active map metadata; an unavailable metadata read still
    leaves the safety gate unresolved.
    """

    observed = list(getattr(overworld, "triggers", ()) or ())
    observed_ids = {getattr(trigger, "trigger_id", None) for trigger in observed}
    checkpoint_symbols = {
        symbol
        for checkpoint in KNOWN_TRAINER_CHECKPOINTS
        if getattr(checkpoint, "mandatory", False)
        and getattr(checkpoint, "map_id", None) == training_map
        for symbol in (
            getattr(checkpoint, "script_symbol", None),
            *getattr(checkpoint, "alternate_script_symbols", ()),
        )
        if symbol
    }
    if not checkpoint_symbols:
        return tuple(observed)

    try:
        map_metadata = get_map_metadata(training_map)
        for index, event in enumerate(getattr(map_metadata, "coord_events", ()) or ()):
            if getattr(event, "type", None) == "weather":
                continue
            script_symbol = getattr(event, "script_symbol", None)
            if script_symbol not in checkpoint_symbols:
                continue
            trigger_id = (
                f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}"
            )
            if trigger_id in observed_ids:
                continue
            location = (training_map, event.local_coordinates)
            observed.append(
                TriggerObservation(
                    trigger_id=trigger_id,
                    locations=frozenset({location}),
                    activation_locations=frozenset({location}),
                    kind="trigger",
                    elevation=getattr(event, "elevation", None),
                    script_symbol=script_symbol,
                    condition_variable_number=getattr(event, "trigger_var_number", None),
                    condition_required_value=getattr(event, "trigger_value", None),
                    affordance_id=script_symbol,
                    requires_input=False,
                )
            )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        # Unknown ROM metadata must not open a route through an unresolved
        # mandatory checkpoint. The caller will retain the live observations
        # and fail closed if the checkpoint remains absent from them.
        pass
    return tuple(observed)


def _mandatory_trainer_hazard_state(overworld, training_map):
    """Return the unresolved mandatory trainer hazard on an encounter map.

    Encounter preparation used to treat the walkable component as sufficient
    evidence that a grass tile was safe.  That is not true on a map where a
    mandatory trainer partitions the route: a grass tile can be reachable in
    the ordinary tile graph while every executable route to it crosses the
    trainer's sight line.  The static/object trigger must also provide
    geometry before the encounter capability is allowed to move.  An absent
    or empty geometry envelope is therefore an unresolved safety fact.

    The helper returns ``(trainer_id, reason)`` or ``None``.  It deliberately
    does not promote the trainer to an engagement goal.  During first
    encounter acquisition, fighting that trainer is precisely the outcome
    this gate is meant to prevent.
    """

    candidates = tuple(
        checkpoint
        for checkpoint in KNOWN_TRAINER_CHECKPOINTS
        if getattr(checkpoint, "mandatory", False)
        and getattr(checkpoint, "map_id", None) == training_map
    )
    if not candidates:
        return None

    triggers = _mandatory_checkpoint_triggers(overworld, training_map)
    objects = tuple(getattr(overworld, "objects", ()) or ())
    for checkpoint in candidates:
        try:
            # Keep this import local: campaign_controller imports the runtime
            # adapters during application setup. Once execution begins, the
            # shared campaign state is the authoritative proof that a trainer
            # already crossed by the player no longer gates duplicate grass
            # preparation.
            from modules.nuzlocke.campaign_controller import runtime_campaign_state

            facts = getattr(runtime_campaign_state(), "campaign_facts", None)
            clearance = getattr(facts, getattr(checkpoint, "clearance_fact", ""), None)
            if (
                getattr(getattr(clearance, "status", None), "name", None) == "KNOWN"
                and getattr(clearance, "value", None) is True
            ):
                continue
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            pass
        trainer_id = getattr(checkpoint, "trainer_affordance_id", None)
        matching = tuple(
            trigger
            for trigger in triggers
            if (
                getattr(trigger, "affordance_id", None) == trainer_id
                or getattr(trigger, "script_symbol", None)
                in {
                    symbol
                    for symbol in (
                        getattr(checkpoint, "script_symbol", None),
                        *getattr(checkpoint, "alternate_script_symbols", ()),
                    )
                    if symbol
                }
            )
        )
        matching_objects = tuple(
            obj
            for obj in objects
            if getattr(obj, "trainer_id", None) == trainer_id
        )
        if any(getattr(item, "trainer_defeated", None) is True for item in (*matching, *matching_objects)):
            continue
        if not matching and not matching_objects:
            return trainer_id, "mandatory trainer is absent from the current hazard observation"
        if not any(
            getattr(item, "hazard_locations", ())
            or getattr(item, "engagement_locations", ())
            or getattr(item, "activation_locations", ())
            or getattr(item, "locations", ())
            for item in matching
        ):
            return trainer_id, "mandatory trainer sight-line geometry is unresolved"
    return None


_last_preparation_no_line_signature: tuple | None = None


def _describe_trainer_readiness(item) -> str:
    """Summarize the readiness verdict behind one trainer's rejection.

    ``BattleReadiness.win_probability`` is ``None`` when no search established
    the outcome, so it is not shown as a number.  The per-lead assessments carry
    the evidence that explains a "not proven safe" verdict: how many roster slots
    each party member's lines cover, and its worst win and faint probabilities.
    """

    readiness = getattr(item, "readiness", None)
    if readiness is None:
        return "readiness unavailable"
    status = getattr(getattr(readiness, "status", None), "value", getattr(readiness, "status", None))
    roster = len(getattr(getattr(item.source, "checkpoint", None), "party", ()) or ())
    leads = sorted(
        getattr(readiness, "lead_assessments", ()) or (),
        key=lambda lead: getattr(lead, "rank", ()),
        reverse=True,
    )[:3]
    lead_text = "; ".join(
        f"{lead.name} covers {len(lead.lines)}/{roster} slots, "
        f"min win {lead.minimum_win_probability:.0%}, max faint {lead.maximum_faint_probability:.0%}, "
        f"{'safe' if lead.safe else 'not safe'}"
        for lead in leads
    )
    reason = getattr(readiness, "reason", "") or "no reason given"
    return f"status={status}, {reason}" + (f" [{lead_text}]" if lead_text else " [no party member assessed]")


def _describe_preparation_failure(
    *,
    allow_wild: bool,
    grass_candidates: int,
    grass_reachable: int | None,
    grass_rejections: Counter,
    trainers_observed: int,
    trainers_without_checkpoint: tuple[str, ...],
    evaluations,
    level_cap,
    target_level,
) -> str:
    """Explain, stage by stage, why preparation found nothing to train on.

    Grass training and trainer clearing are tried in that order; a bare "no
    safe line" hides which stage failed and why.  This names each stage, and
    for every trainer the roster-versus-party numbers and the rejection reason.
    """

    if not allow_wild:
        grass = "grass: not used, the first encounter of this area is already resolved"
    elif grass_candidates == 0:
        grass = "grass: no encounter tiles known on this map"
    else:
        rejected = ", ".join(f"{reason}={count}" for reason, count in sorted(grass_rejections.items())) or "none"
        reachable = "unknown" if grass_reachable is None else str(grass_reachable)
        grass = f"grass: {grass_candidates} tiles, {reachable} reachable, rejected [{rejected}]"
    if trainers_observed == 0:
        trainers = "trainers: no undefeated trainer observed on this map"
    else:
        parts = []
        for item in evaluations:
            numbers = f"exp={item.projected_exp} headroom={item.recipient_headroom} cap_safe={item.cap_safe}"
            verdict = "selected" if item.selected else f"rejected: {item.rejection_reason or 'unspecified'}"
            parts.append(f"{item.source.trainer_id} ({numbers}) {verdict} ({_describe_trainer_readiness(item)})")
        if trainers_without_checkpoint:
            parts.append(f"no ROM checkpoint, not evaluated: {', '.join(trainers_without_checkpoint)}")
        trainers = f"trainers: {trainers_observed} observed; " + "; ".join(parts)
    return f"{grass}; {trainers}; target_level={target_level} level_cap={level_cap}"


def _report_preparation_no_line(map_id, diagnosis: str) -> None:
    """Write the failure explanation to the normal log, once per distinct explanation."""

    global _last_preparation_no_line_signature
    signature = (map_id, diagnosis)
    if signature == _last_preparation_no_line_signature:
        return
    _last_preparation_no_line_signature = signature
    print_always(lambda: f"CAMPAIGN_PREPARATION_NO_LINE: map={map_id!r} {diagnosis}")


def _preparation_navigation_goal(
    overworld,
    training_map,
    training_candidates: tuple[tuple[int, int], ...],
    *,
    allow_wild: bool = True,
    target_level: int | None = None,
    trainer_mode: TrainerMode = TrainerMode.AVOID,
):
    """Select a reachable grass target, or a trainer that blocks the route.

    A battle can return the avatar several tiles away from the target that
    was planned before the battle.  Dynamic trainer occupancy can also split
    the currently observed map into components.  Reusing the old fixed
    target in either case turns a recoverable battle into a failed
    preparation capability.  Build one fresh local world and use it both to
    choose a new grass tile and, when necessary, to deliberately clear a
    reachable trainer blocking the remaining grass component.
    """
    start = (overworld.map_id, overworld.player_coordinates)
    world = NavigationWorld.from_overworld(overworld)
    last_error = None
    mandatory_trigger_ids = (
        _mandatory_checkpoint_trigger_ids(overworld, training_map)
        if allow_wild and trainer_mode is TrainerMode.AVOID
        else frozenset()
    )
    mandatory_hazard = (
        _mandatory_trainer_hazard_state(overworld, training_map)
        if allow_wild and trainer_mode is TrainerMode.AVOID
        else None
    )
    if mandatory_hazard is not None:
        trainer_id, reason = mandatory_hazard
        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_TRAINER_GATE: "
                f"map={training_map!r} trainer={trainer_id!r} action='wait_for_safe_geometry' "
                f"reason={reason!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_ENCOUNTER_TRAINER_GATE",
        )
        return None, f"{trainer_id}: {reason}"
    candidates = sorted(
        training_candidates,
        key=lambda position: (
            abs(position[0] - start[1][0]) + abs(position[1] - start[1][1]),
            position[0],
            position[1],
        ),
    )
    grass_rejections: Counter = Counter()
    grass_reachable: int | None = None
    if allow_wild:
        reachable_positions = _reachable_local_positions(world, start, training_map)
        if reachable_positions is not None:
            grass_reachable = sum(coordinates in reachable_positions for coordinates in candidates)
            for coordinates in candidates:
                if coordinates not in reachable_positions:
                    continue
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_PREPARATION_CANDIDATE_SELECTED: "
                        f"map={training_map!r} start={start!r} coordinates={coordinates!r} "
                        f"reachable_candidates={sum(candidate in reachable_positions for candidate in candidates)}"
                    ),
                    trace=True,
                )
                target = ReachLocation((training_map, coordinates))
                if trainer_mode is TrainerMode.IGNORE:
                    return target, None
                navigation_goal = NavigationGoal(
                    target,
                    constraints=GoalConstraints(
                        trainer_mode=trainer_mode,
                        avoid_trigger_ids=mandatory_trigger_ids,
                    ),
                    encounter_mode=EncounterMode.AVOID,
                )
                try:
                    planned = plan_with_world_navigation(world, start, navigation_goal)[0]
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    NavigationError,
                    PathFindingError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    last_error = error
                    grass_rejections["trainer_safe_route_unavailable"] += 1
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_PREPARATION_CANDIDATE_REJECTED: "
                            f"map={training_map!r} coordinates={coordinates!r} "
                            f"reason='trainer_safe_route_unavailable' error={error!r}"
                        ),
                        trace=True,
                    )
                    continue
                if getattr(planned, "forced_trainer_exposure", False):
                    grass_rejections["forced_trainer_exposure"] += 1
                    last_error = NavigationError("trainer-safe route required trainer exposure")
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_PREPARATION_CANDIDATE_REJECTED: "
                            f"map={training_map!r} coordinates={coordinates!r} "
                            "reason='forced_trainer_exposure'"
                        ),
                        trace=True,
                    )
                    continue
                return navigation_goal, None
        else:
            for coordinates in candidates:
                try:
                    target = ReachLocation((training_map, coordinates))
                    candidate_goal = (
                        target
                        if trainer_mode is TrainerMode.IGNORE
                        else NavigationGoal(
                            target,
                            constraints=GoalConstraints(
                                trainer_mode=trainer_mode,
                                avoid_trigger_ids=mandatory_trigger_ids,
                            ),
                            encounter_mode=EncounterMode.AVOID,
                        )
                    )
                    planned = plan_with_world_navigation(world, start, candidate_goal)[0]
                    if getattr(planned, "forced_trainer_exposure", False):
                        raise NavigationError("trainer-safe route required trainer exposure")
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    NavigationError,
                    PathFindingError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    last_error = error
                    grass_rejections[type(error).__name__] += 1
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_PREPARATION_CANDIDATE_REJECTED: "
                            f"map={training_map!r} coordinates={coordinates!r} "
                            f"start={start!r} error_type={type(error).__name__!r} error={error!r}"
                        ),
                        trace=True,
                    )
                    continue
                return candidate_goal, None
    else:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_WILD_DISABLED: "
                f"map={training_map!r} start={start!r} reason='area encounter already resolved'"
            ),
            trace=False,
        )

    # Runtime object events are camera-scoped and may disappear immediately
    # after a battle.  Perception also publishes ROM-backed trainer hazard
    # triggers for objects that have not spawned yet; use both sources so a
    # defeated live object cannot make the remaining training route vanish.
    trainer_positions: dict[str, set[tuple]] = {}
    trainer_scripts: dict[str, str | None] = {}
    for obj in getattr(overworld, "objects", ()):
        trainer_id = getattr(obj, "trainer_id", None)
        location = getattr(obj, "location", None)
        if trainer_id is None or getattr(obj, "trainer_defeated", None) is not False or location is None:
            continue
        trainer_positions.setdefault(trainer_id, set()).add(location)
        trainer_scripts.setdefault(trainer_id, getattr(obj, "script", None))
    for trigger in getattr(overworld, "triggers", ()):
        trainer_id = getattr(trigger, "affordance_id", None)
        if trainer_id is None:
            continue
        if not (
            getattr(trigger, "hazard_kind", None) == "trainer"
            or getattr(trigger, "kind", None) in {"trainer", "trainer_hazard"}
        ):
            continue
        positions = getattr(trigger, "engagement_locations", ()) or getattr(trigger, "hazard_locations", ())
        positions = tuple(position for position in positions if position is not None)
        if positions:
            trainer_positions.setdefault(trainer_id, set()).update(positions)
        script = getattr(trigger, "script_symbol", None)
        if not script:
            trigger_id = getattr(trigger, "trigger_id", "")
            if isinstance(trigger_id, str) and trigger_id.startswith("static_object:"):
                parts = trigger_id.split(":", 2)
                script = parts[2] if len(parts) == 3 else None
        trainer_scripts.setdefault(trainer_id, script)

    def trainer_distance(item):
        trainer_id, positions = item
        same_map = [position for position in positions if position[0] == start[0]]
        return (
            min(
                abs(position[1][0] - start[1][0]) + abs(position[1][1] - start[1][1])
                for position in same_map
            )
            if same_map
            else float("inf"),
            trainer_id,
        )

    trainers = sorted(trainer_positions.items(), key=trainer_distance)

    # The fallback below deliberately returns an EngageTrainer goal for a
    # trainer that blocks ordinary preparation.  That is valid after the
    # area's first encounter has been secured, but it violates the encounter
    # objective while ``allow_wild`` is still true: it would turn the missing
    # safe grass route into an early Calvin battle. Fail closed and let the
    # campaign reobserve the map instead.
    if allow_wild:
        mandatory_ids = {
            getattr(checkpoint, "trainer_affordance_id", None)
            for checkpoint in KNOWN_TRAINER_CHECKPOINTS
            if getattr(checkpoint, "mandatory", False)
            and getattr(checkpoint, "map_id", None) == training_map
        }
        blocking_mandatory = next((trainer_id for trainer_id, _ in trainers if trainer_id in mandatory_ids), None)
        if blocking_mandatory is not None:
            return None, f"{blocking_mandatory}: safe encounter route is unavailable"

    # Once an encounter has been consumed, trainer preparation is a source
    # selection problem rather than a nearest-object problem.  Use the active
    # ROM catalog when it is available and keep unknown rosters out of the
    # selected set.  The legacy distance ordering remains for lightweight
    # test doubles and ROM boundaries where no catalog facts can be read.
    selected_trainer_ids = [trainer_id for trainer_id, _ in trainers]
    catalog_sources = []
    for trainer_id, positions in trainers:
        checkpoint = trainer_checkpoint_for_source(
            trainer_id,
            script_symbol=trainer_scripts.get(trainer_id),
        )
        if checkpoint is None:
            continue
        same_map = [position for position in positions if position[0] == start[0]]
        distance = min(
            (abs(position[1][0] - start[1][0]) + abs(position[1][1] - start[1][1]) for position in same_map),
            default=10_000,
        )
        catalog_sources.append(
            TrainerSource(
                trainer_id,
                checkpoint,
                distance=distance,
                route_simplification=100 if not allow_wild else 0,
                gym_trainer=bool(getattr(checkpoint, "gym_trainer", False) or _is_gym_map(training_map)),
                mandatory=not allow_wild,
            )
        )

    evaluations = ()
    level_cap = None
    if catalog_sources:
        level_cap = _active_preparation_level_cap_for_runtime()
        party = get_party()
        if level_cap is not None and all(
            hasattr(pokemon, "total_exp") and hasattr(pokemon, "species") for pokemon in party if not pokemon.is_egg
        ):
            budget = build_exp_budget(
                party,
                target_level=target_level,
                level_cap=level_cap,
            )
            context.campaign_preparation_exp_budget = budget
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PREPARATION_EXP_BUDGET: "
                    f"target_level={budget.target_level!r} level_cap={budget.level_cap!r} "
                    f"required_exp={budget.required_exp!r} unavoidable_exp={budget.unavoidable_exp!r} "
                    f"remaining_deficit={budget.remaining_deficit!r} cap_headroom={budget.cap_headroom!r} "
                    f"mandatory_exp_is_legal={budget.mandatory_exp_is_legal!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_PREPARATION_EXP_BUDGET",
            )
            selected, evaluations = select_trainer_source(
                catalog_sources,
                party,
                target_level=target_level,
                level_cap=level_cap,
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_SOURCE_EVALUATION: "
                    f"map={training_map!r} level_cap={level_cap!r} "
                    f"candidates={[{'trainer': item.source.trainer_id, 'exp': item.projected_exp, 'win': round(item.win_probability, 4), 'catastrophic': round(item.catastrophic_probability, 4), 'recipient': item.best_recipient, 'headroom': item.recipient_headroom, 'cap_safe': item.cap_safe, 'route_value': item.source.route_simplification, 'gym': item.source.gym_trainer, 'selected': item.selected, 'rejected': item.rejection_reason} for item in evaluations]!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_SOURCE_EVALUATION",
            )
            if selected is None:
                evaluated_ids = {item.source.trainer_id for item in evaluations}
                diagnosis = _describe_preparation_failure(
                    allow_wild=allow_wild,
                    grass_candidates=len(candidates),
                    grass_reachable=grass_reachable,
                    grass_rejections=grass_rejections,
                    trainers_observed=len(trainers),
                    trainers_without_checkpoint=tuple(
                        trainer_id for trainer_id, _positions in trainers if trainer_id not in evaluated_ids
                    ),
                    evaluations=evaluations,
                    level_cap=level_cap,
                    target_level=target_level,
                )
                _report_preparation_no_line(training_map, diagnosis)
                return None, RuntimeError(
                    f"no available trainer has a proven safe, cap-compatible preparation line ({diagnosis})"
                )
            selected_trainer_ids = [selected.source.trainer_id]
        elif catalog_sources:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_SOURCE_DEFERRED: "
                    f"map={training_map!r} reason='level-cap or party EXP facts unavailable'"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_SOURCE_DEFERRED",
            )
            return None, RuntimeError("trainer preparation deferred because cap or EXP facts are unavailable")

    for trainer_id, _positions in trainers:
        if trainer_id not in selected_trainer_ids:
            continue
        trainer_goal = NavigationGoal(
            EngageTrainer(trainer_id),
            constraints=GoalConstraints(trainer_mode=TrainerMode.ENGAGE),
            encounter_mode=EncounterMode.IGNORE,
        )
        try:
            plan_with_world_navigation(world, start, trainer_goal)
        except (
            AttributeError,
            IndexError,
            KeyError,
            NavigationError,
            PathFindingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            last_error = error
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PREPARATION_TRAINER_REJECTED: "
                    f"map={training_map!r} trainer={trainer_id!r} "
                    f"start={start!r} "
                    f"static_blocked={tuple(sorted(getattr(overworld, 'static_blocked_coordinates', ())))!r} "
                    f"error_type={type(error).__name__!r} error={error!r}"
                ),
                trace=True,
            )
            continue
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_TRAINER_SELECTED: "
                f"map={training_map!r} start={start!r} trainer={trainer_id!r}"
            ),
            trace=True,
        )
        checkpoint = trainer_checkpoint_for_source(
            trainer_id,
            script_symbol=trainer_scripts.get(trainer_id),
        )
        if checkpoint is not None:
            context.campaign_trainer_checkpoint = checkpoint
            try:
                readiness = assess_live_trainer_checkpoint(
                    checkpoint,
                    get_party(),
                    expected_exp=projected_trainer_experience(checkpoint),
                )
                context.campaign_trainer_readiness = readiness
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                readiness = None
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_CHECKPOINT: "
                    f"checkpoint={checkpoint.checkpoint_id!r} trainer={trainer_id!r} "
                    f"party={[(member.species, member.level, member.moves) for member in checkpoint.party]!r} "
                    f"readiness={getattr(getattr(readiness, 'status', None), 'value', None)!r} "
                    f"reason={getattr(readiness, 'reason', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_CHECKPOINT",
            )
        else:
            context.campaign_trainer_checkpoint = None
            context.campaign_trainer_readiness = None
        return trainer_goal, None

    evaluated_ids = {item.source.trainer_id for item in evaluations}
    diagnosis = _describe_preparation_failure(
        allow_wild=allow_wild,
        grass_candidates=len(candidates),
        grass_reachable=grass_reachable,
        grass_rejections=grass_rejections,
        trainers_observed=len(trainers),
        trainers_without_checkpoint=tuple(
            trainer_id for trainer_id, _positions in trainers if trainer_id not in evaluated_ids
        ),
        evaluations=evaluations,
        level_cap=level_cap,
        target_level=target_level,
    )
    _report_preparation_no_line(training_map, diagnosis)
    if last_error is None:
        last_error = RuntimeError(f"no preparation line ({diagnosis})")
    return None, last_error


def _preparation_encounter_resolved(training_map) -> bool:
    """Return whether the selected area's first encounter is no longer open.

    This is the ownership boundary for the first encounter objective. Once it
    reaches a terminal outcome, later wild battles remain legal training
    opportunities; they simply must not become another capture objective.
    Read this from the rules projection rather than from the event-store cache
    or the current party so the live policy and replay use the same authority.
    """
    runtime = getattr(context, "nuzlocke_runtime", None)
    projection = getattr(runtime, "rules_projection", None)
    state = getattr(projection, "state", None)
    if state is None:
        return False
    location = _map_id_value(training_map)
    encounter = next(
        (
            item
            for item in getattr(state, "encounters", ())
            if getattr(item, "eligible", True) and item.location == location
        ),
        None,
    )
    status = getattr(encounter, "status", None) if encounter is not None else None
    resolved = encounter is not None and status not in {"none", "pending", "unknown"}
    diagnostic_print(
        lambda: (
            "CAMPAIGN_ENCOUNTER_PROJECTION: "
            f"map={location!r} found={encounter is not None!r} "
            f"status={status!r} eligible={getattr(encounter, 'eligible', None)!r} "
            f"resolved={resolved!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_ENCOUNTER_PROJECTION",
    )
    return resolved


def _encounter_roll_direction(overworld) -> tuple[Direction | None, str]:
    """Choose an input that rolls for an encounter without leaving grass.

    Gen III checks for a wild encounter on a movement/turn input, not while
    the avatar is merely standing on an encounter tile.  When the current
    tile has a blocked edge, turn into that edge first.  This keeps the avatar
    on the first reachable grass tile instead of repeatedly walking across a
    large patch and rebuilding navigation state for every roll.  A navigation
    world remains the fallback when the local projection cannot classify a
    legal roll neighbor, because forced movement and field transitions need
    the full neighbor semantics.
    """
    if overworld is None:
        return None, "overworld_observation_unavailable"

    map_id = getattr(overworld, "map_id", None)
    coordinates = getattr(overworld, "player_coordinates", None)
    if map_id is None or not isinstance(coordinates, tuple) or len(coordinates) != 2:
        return None, "player_position_unavailable"

    start = (map_id, coordinates)
    tile_at = getattr(overworld, "tile_at", None)

    def observed_tile(local_coordinates):
        if callable(tile_at):
            return tile_at(local_coordinates)
        return next(
            (
                tile
                for tile in getattr(overworld, "tiles", ())
                if getattr(tile, "location", (None, None))[0] == map_id
                and getattr(tile, "location", (None, None))[1] == local_coordinates
            ),
            None,
        )

    current_tile = observed_tile(coordinates)
    if current_tile is None or not current_tile.has_encounters:
        return None, "not_on_encounter_terrain"

    # The live tile projection is enough to identify ordinary blocked edges.
    # ``walkable_neighbors`` describes the directions from which a destination
    # tile accepts entry, which is the same directional test used by
    # NavigationWorld.neighbors().
    legal_directions: set[Direction] = set()
    local_encounter_directions: set[Direction] = set()
    local_escape_directions: set[Direction] = set()
    direction_vectors = {
        Direction.North: (0, -1),
        Direction.East: (1, 0),
        Direction.South: (0, 1),
        Direction.West: (-1, 0),
    }
    transitions = getattr(overworld, "transitions", ()) or getattr(overworld, "warps", ())
    transition_entries = {transition.entry for transition in transitions}
    facing = getattr(overworld, "facing", None)
    x, y = coordinates

    def elevation_allows_step(destination_tile) -> bool:
        source_elevation = getattr(current_tile, "elevation", None)
        if source_elevation is None:
            source_elevation = getattr(overworld, "player_elevation", None)
        destination_elevation = getattr(destination_tile, "elevation", None)
        if source_elevation is None or destination_elevation is None:
            return True
        if source_elevation == destination_elevation:
            return True
        if source_elevation in (0, 15) or destination_elevation in (0, 15):
            return True
        if source_elevation == 1 and destination_elevation == 3:
            return True
        if source_elevation == 3 and destination_elevation == 1:
            return bool(getattr(overworld, "surfing", False))
        return False

    for direction, (dx, dy) in direction_vectors.items():
        destination = (x + dx, y + dy)
        destination_tile = observed_tile(destination)
        if destination_tile is None or getattr(destination_tile, "blocked", False):
            continue
        if not elevation_allows_step(destination_tile):
            continue
        destination_location = (map_id, destination)
        if destination_location not in transition_entries and direction not in getattr(
            destination_tile, "walkable_neighbors", frozenset()
        ):
            continue
        legal_directions.add(direction)
        if destination_location not in transition_entries and destination_tile.has_encounters:
            local_encounter_directions.add(direction)
        elif destination_location not in transition_entries:
            local_escape_directions.add(direction)

    # Include field-effect transitions whose source is the current tile. The
    # ordinary local tile check cannot see these because their destination may
    # be non-adjacent or on another map.
    for transition in transitions:
        if getattr(transition, "entry", None) == start:
            direction = getattr(transition, "required_facing", None) or Direction.South
            legal_directions.add(direction)
        if start in getattr(transition, "activation_locations", ()):
            direction = (
                getattr(transition, "activation_direction", None)
                or getattr(transition, "required_facing", None)
                or Direction.South
            )
            legal_directions.add(direction)

    # Keep the established edge-roll behavior for a grass patch with a grass
    # neighbor.  A blocked directional input rolls an encounter without
    # walking across the patch.  The isolated-tile case below has no such
    # neighbor, so it can safely step out and re-enter instead.
    if local_encounter_directions:
        for direction in Direction:
            if direction is not facing and direction not in legal_directions:
                return direction, "turn_in_place_on_blocked_edge"
        direction = min(local_encounter_directions, key=lambda candidate: candidate.value)
        return direction, "move_within_encounter_terrain"

    if local_escape_directions:
        # Some route layouts contain a single grass tile embedded in a fully
        # walkable path. There is no legal stationary turn in that geometry:
        # every directional input would step off the grass. Take one bounded
        # step out and let the caller re-enter the selected encounter tile;
        # the re-entry itself is an encounter roll and avoids rebuilding a
        # costly grass traversal route.
        direction = min(
            (candidate for candidate in local_escape_directions if candidate is not facing),
            key=lambda candidate: candidate.value,
            default=min(local_escape_directions, key=lambda candidate: candidate.value),
        )
        return direction, "step_off_encounter_terrain"

    # A blocked input changes facing without changing coordinates. Only use
    # that stationary roll when there is no legal way to leave the current
    # encounter tile. Returning it before the escape branch above creates an
    # endless turn-in-place loop on one-tile grass patches.
    for direction in Direction:
        if direction is not facing and direction not in legal_directions:
            return direction, "turn_in_place_on_blocked_edge"

    # The local projection may omit a forced destination or field transition.
    # Preserve the full-world fallback for that incomplete-observation case.
    world = NavigationWorld.from_overworld(overworld)
    encounter_neighbors: list[tuple[Direction, tuple, bool]] = []
    for direction, destination, is_warp in world.neighbors(start):
        destination_tile = world.tiles.get(destination)
        if (
            not is_warp
            and destination[0] == map_id
            and destination_tile is not None
            and destination_tile.has_encounters
        ):
            encounter_neighbors.append((direction, destination, is_warp))

    if encounter_neighbors:
        direction, _destination, _is_warp = min(
            encounter_neighbors,
            key=lambda candidate: candidate[0].value,
        )
        return direction, "move_within_encounter_terrain"

    return None, "no_safe_encounter_roll_input"


def _issue_encounter_roll_input(direction: Direction) -> None:
    """Send one fresh directional edge for encounter generation."""
    emulator = context.emulator
    reset_held_buttons = getattr(emulator, "reset_held_buttons", None)
    if callable(reset_held_buttons):
        # A navigation batch may have held its final direction until the goal
        # boundary. Do not let that ownership bleed into acquisition input.
        reset_held_buttons()

    press_direction = getattr(emulator, "press_direction", None)
    if callable(press_direction):
        press_direction(direction.button_name, run=False, fresh=True)
        return

    press_button_fresh = getattr(emulator, "press_button_fresh", None)
    if callable(press_button_fresh):
        press_button_fresh(direction.button_name)
        return

    # Keep lightweight emulator adapters usable while still giving the real
    # adapter the fresh-edge behavior above.
    emulator.press_button(direction.button_name)


def _rom_backed_encounter_route(graph, source_map, target_map) -> WorldRoute:
    """Return a map route whose edges have executable ROM transitions.

    The coarse graph is useful for ranking, but a stale or incomplete map
    connection can still appear in its shortest path. Encounter execution
    needs an executable transition at every segment, so filter those edges
    lazily against the ROM transition catalog while searching.
    """
    transition_cache: dict[object, tuple[WorldTransition, ...] | None] = {}

    def edge_is_executable(edge) -> bool:
        if not getattr(edge, "source_coordinates", ()):
            return True
        if edge.source_map not in transition_cache:
            try:
                transition_cache[edge.source_map] = tuple(static_map_transitions(edge.source_map))
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # Lightweight adapters and unit tests may not expose ROM
                # metadata. Preserve the graph fallback in that environment;
                # live profiles provide the catalog and are filtered here.
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

    def filtered_edge(edge) -> bool:
        allowed = edge_is_executable(edge)
        if not allowed:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_ROUTE_FILTER: "
                    f"source_map={edge.source_map!r} "
                    f"destination_map={edge.destination_map!r} "
                    f"kind={edge.kind!r} reason='no_rom_transition'"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_ROUTE_FILTER:",
            )
        return allowed

    try:
        return route_with_edge_filter(graph, source_map, target_map, filtered_edge)
    except Exception as error:
        if isinstance(error, BotModeError):
            raise
        raise BotModeError(f"no ROM-backed route from {source_map!r} to {target_map!r}") from error


def _preparation_map_route(training_map):
    """Return the ROM-backed map corridor for trainer preparation.

    Preparation only needs to reach the selected training map.  Passing the
    map corridor into the tactical goal keeps the lower-level planner from
    exploring unrelated map detours while it is pricing live boundary
    candidates.  If the current location is not available yet, retain the
    ordinary semantic goal and let the next observation establish the route.
    """

    try:
        current_location = get_player_location()
        current_map = current_location[0]
        route = _rom_backed_encounter_route(get_world_map_graph(), current_map, training_map)
    except (AttributeError, BotModeError, RuntimeError, TypeError, ValueError):
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_ROUTE: "
                f"target={training_map!r} status='unavailable' action='use_observed_route'"
            ),
            trace=True,
            prefix="CAMPAIGN_PREPARATION_ROUTE",
        )
        return None
    diagnostic_print(
        lambda: (
            "CAMPAIGN_PREPARATION_ROUTE: "
            f"source={current_map!r} target={training_map!r} "
            f"maps={route.maps!r} status='preferred'"
        ),
        trace=True,
        prefix="CAMPAIGN_PREPARATION_ROUTE",
    )
    return route.maps


def _acquire_campaign_encounter(encounter_map) -> Iterator[object]:
    """Generate encounter rolls after navigation reaches encounter terrain."""
    rolls = 0
    executor = AgentActionExecutor()
    while True:
        observation = observe_agent()
        interaction_type = observation.interaction_type
        if interaction_type is InteractionType.BATTLE:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_ACQUISITION: " f"event=battle_started map={encounter_map!r} rolls={rolls}"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_ACQUISITION:",
            )
            # BattleListener now owns the battle. Returning lets the campaign
            # controller re-evaluate encounter completion from its projection.
            return

        if _preparation_encounter_resolved(encounter_map):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_ACQUISITION: " f"event=encounter_resolved map={encounter_map!r} rolls={rolls}"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_ACQUISITION:",
            )
            return

        if interaction_type is not InteractionType.OVERWORLD or not observation.interaction.controllable:
            decision = select_action(observation)
            executor.execute(decision.action, observation)
            yield
            continue

        overworld = observation.overworld
        if overworld is None:
            yield
            continue
        if getattr(overworld, "map_id", None) != _map_id_value(encounter_map):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_ACQUISITION: "
                    f"event=map_changed map={getattr(overworld, 'map_id', None)!r} "
                    f"expected={_map_id_value(encounter_map)!r} rolls={rolls}"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_ACQUISITION:",
            )
            # A whiteout, warp, or other external transition should be
            # handled by campaign replanning rather than by sending an input
            # on the wrong map.
            return

        if (
            getattr(overworld, "movement_state", None) is not None
            and overworld.movement_state is not MovementState.STANDING
        ):
            yield
            continue

        direction, reason = _encounter_roll_direction(overworld)
        if direction is None:
            if reason == "not_on_encounter_terrain":
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_ENCOUNTER_ACQUISITION: "
                        f"event=terrain_left map={overworld.map_id!r} "
                        f"coordinates={overworld.player_coordinates!r} rolls={rolls}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_ENCOUNTER_ACQUISITION:",
                )
                # The bounded escape step above deliberately leaves grass so
                # that the next campaign encounter mount can re-enter it and
                # produce a ROM encounter roll.
                return
            raise BotModeError(f"cannot generate an encounter roll on map {encounter_map!r}: {reason}")
        _issue_encounter_roll_input(direction)
        rolls += 1
        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_ACQUISITION: "
                f"event=roll_input map={overworld.map_id!r} "
                f"coordinates={overworld.player_coordinates!r} "
                f"direction={direction.name!r} reason={reason!r} rolls={rolls}"
            ),
            trace=True,
            prefix="CAMPAIGN_ENCOUNTER_ACQUISITION:",
        )
        yield


# After the staged map route completes, the encounter capability waits for a
# settled destination observation and for ROM-backed trainer geometry.  Those
# waits fail closed, but they must not be silent or unbounded: a post-warp
# stall otherwise looks like "no action was issued" with nothing in the log.
_ENCOUNTER_POST_TRANSITION_UNSETTLED_LIMIT = 900
_ENCOUNTER_TRAINER_GATE_WAIT_LIMIT = 600
_ENCOUNTER_WAIT_LOG_INTERVAL = 120


def _encounter_wait_should_log(waits: int) -> bool:
    return waits == 1 or waits % _ENCOUNTER_WAIT_LOG_INTERVAL == 0


def _travel_to_map_staged(map_id, graph) -> Iterator[object]:
    """Travel to ``map_id`` one ROM-backed exit at a time.

    Each stage searches the *live* observation of the current map under the
    stage's trainer policy, which sees trainer geometry accurately.  A single
    global semantic route plans through static data for the maps ahead, and its
    static trainer hazards are more conservative than a live observation, so it
    can report an avoidable corridor (for example Route 104) as unreachable.

    Returns ``True`` once the avatar is on ``map_id``, or ``False`` when the
    staging owner found no legal exit; the caller decides what that means
    (the encounter capability hands the objective back to the frontier).
    """

    stage_retry_delay = 0
    stage_failures = 0
    interaction_executor = AgentActionExecutor()

    while True:
        if stage_retry_delay:
            stage_retry_delay -= 1
            yield
            continue
        current = observe_agent()
        # A field script can own the frame before the overworld projection is
        # available.  Consume that interaction before deciding that the
        # encounter route has no usable world observation; otherwise a story
        # scene such as the Running Shoes dialogue leaves this capability
        # yielding forever.
        if current.interaction_type is InteractionType.BATTLE:
            yield
            continue
        if current.interaction_type is not InteractionType.OVERWORLD or not current.interaction.controllable:
            decision = select_action(current)
            interaction_executor.execute(decision.action, current)
            yield
            continue
        overworld = getattr(current, "overworld", None)
        if overworld is None:
            yield
            continue
        current_map = overworld.map_id
        if current_map == map_id:
            return True

        route = _rom_backed_encounter_route(graph, current_map, map_id)
        if len(route.maps) < 2:
            raise BotModeError(f"no next map while routing from {current_map!r} to {map_id!r}")
        next_map = route.maps[1]
        static_candidates = tuple(
            transition
            for transition in static_map_transitions(current_map)
            if transition.destination is not None and transition.destination[0] == next_map
        )
        if not static_candidates:
            raise BotModeError(f"no ROM-backed exit from {current_map!r} to {next_map!r}")
        transition, trainer_mode, trainer_mode_reason = _encounter_stage_trainer_mode(
            overworld,
            static_candidates,
            next_map,
            final_map=map_id,
            graph=graph,
        )
        if transition is None:
            # The staging owner has already compared every candidate under
            # AVOID and IGNORE. Returning the capability lets the campaign
            # frontier replan an alternate legal encounter or mount the
            # required preparation objective; retrying the same transition
            # would only recreate the identical failed search batch.
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_STAGE: "
                    f"event=unavailable current_map={current_map!r} next_map={next_map!r} "
                    f"reason={trainer_mode_reason!r} action='return_to_frontier'"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_STAGE",
            )
            return False
        stage_goal = NavigationGoal(
            ReachWarp(
                destination_map=next_map,
                destination=transition.destination,
                warp=transition,
            ),
            constraints=GoalConstraints(trainer_mode=trainer_mode),
            encounter_mode=EncounterMode.AVOID,
        )

        stage_last_observation = None

        def observe_stage(goal=stage_goal):
            nonlocal stage_last_observation
            observation = observe_agent(goal=goal)
            stage_last_observation = observation
            observed_world = getattr(observation, "overworld", None)
            if observed_world is None or observed_world.map_id != current_map:
                return observation
            live = tuple(getattr(observed_world, "transitions", ()) or observed_world.warps)
            if not any(transitions_match(transition, candidate) for candidate in live):
                live = live + (transition,)
            return replace(observation, overworld=replace(observed_world, transitions=live))

        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_STAGE: "
                f"current_map={current_map!r} next_map={next_map!r} "
                f"trainer_mode={trainer_mode.name!r} trainer_mode_reason={trainer_mode_reason!r} "
                f"transition={transition!r}"
            ),
            trace=True,
        )
        stage_navigation_plan = _cached_encounter_stage_plan(
            overworld,
            static_candidates,
            next_map,
            transition,
            trainer_mode,
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_STAGE: "
                f"event=execution_plan_reuse transition={transition!r} "
                f"reused={stage_navigation_plan is not None} "
                f"actions={len(stage_navigation_plan.actions) if stage_navigation_plan is not None else None}"
            ),
            trace=True,
            prefix="CAMPAIGN_ENCOUNTER_STAGE",
        )
        stage_loop = AgentControlLoop(
            observe_stage,
            goal=stage_goal,
            navigation_plan=stage_navigation_plan,
            logger=lambda message: diagnostic_print(
                lambda: f"CAMPAIGN_ENCOUNTER_STAGE: {message}",
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_STAGE:",
            ),
        )
        try:
            yield from stage_loop.run()
        except (NavigationError, PathFindingError) as error:
            # AgentControlLoop cannot safely resume a failed navigation
            # generator. Keep the campaign encounter capability mounted and
            # retry the stage with exponential backoff instead of making the
            # controller tear down and remount the entire objective every
            # frame. A trainer may have moved, a transition may still be
            # settling, or another ROM-backed exit may become usable.
            stage_failures += 1
            stage_retry_delay = min(
                _ENCOUNTER_STAGE_RETRY_DELAY_MAX,
                2 ** min(stage_failures, 6),
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_STAGE_RETRY: "
                    f"current_map={current_map!r} next_map={next_map!r} "
                    f"transition={transition!r} trainer_mode={trainer_mode.name!r} "
                    f"failure={stage_failures} delay={stage_retry_delay} "
                    f"error_type={type(error).__name__!r} error={error!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_STAGE_RETRY",
            )
            continue
        stage_failures = 0
        observed_world = getattr(stage_last_observation, "overworld", None)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_STAGE: "
                f"event=loop_return current_map={current_map!r} next_map={next_map!r} "
                f"observed_map={getattr(observed_world, 'map_id', None)!r} "
                f"observed_coordinates={getattr(observed_world, 'player_coordinates', None)!r}"
            ),
            trace=True,
        )


def execute_campaign_encounter(
    encounter_map, *, encounter_method: EncounterMethod = EncounterMethod.LAND
) -> Iterator[object]:
    """Navigate to the nearest reachable land-encounter tile on a map.

    Cross-map travel is executed one ROM-backed exit at a time.  A semantic
    destination is useful to campaign policy, but making the tactical search
    infer an entire interior-to-route chain in one state search can lose the
    first exit when live perception is partial.
    """

    if not isinstance(encounter_method, EncounterMethod):
        raise BotModeError(f"malformed campaign encounter method: {encounter_method!r}")
    if encounter_method is not EncounterMethod.LAND:
        raise BotModeError(
            f"campaign encounter method {encounter_method.value!r} is not supported; land is currently implemented"
        )

    map_id, encounter_candidates = _preparation_training_location(encounter_map)
    graph = get_world_map_graph()
    arrived = yield from _travel_to_map_staged(map_id, graph)
    if not arrived:
        return

    attempts = 0
    unsettled_observations = 0
    trainer_gate_waits = 0
    while True:
        observation = observe_agent()
        overworld = getattr(observation, "overworld", None)
        if overworld is None or getattr(overworld, "map_id", None) != map_id:
            unsettled_observations += 1
            if _encounter_wait_should_log(unsettled_observations):
                print_always(
                    lambda: (
                        "CAMPAIGN_ENCOUNTER_STAGE: event=awaiting_destination_observation "
                        f"expected_map={map_id!r} "
                        f"observed_map={getattr(overworld, 'map_id', None)!r} "
                        f"overworld_available={overworld is not None} "
                        f"waits={unsettled_observations}"
                    ),
                )
            if unsettled_observations > _ENCOUNTER_POST_TRANSITION_UNSETTLED_LIMIT:
                raise BotModeError(
                    f"encounter map transition did not settle on {map_id!r} "
                    f"after {unsettled_observations} observations"
                )
            yield
            continue
        unsettled_observations = 0
        navigation_goal, error = _preparation_navigation_goal(
            overworld,
            map_id,
            encounter_candidates,
            allow_wild=True,
            trainer_mode=TrainerMode.AVOID,
        )
        if navigation_goal is None:
            if error and "mandatory trainer" in error:
                trainer_gate_waits += 1
                exhausted = trainer_gate_waits > _ENCOUNTER_TRAINER_GATE_WAIT_LIMIT
                if exhausted or _encounter_wait_should_log(trainer_gate_waits):
                    print_always(
                        lambda: (
                            "CAMPAIGN_ENCOUNTER_TRAINER_GATE: "
                            f"map={map_id!r} "
                            f"action={'return_to_frontier' if exhausted else 'reobserve'!r} "
                            f"reason={error!r} waits={trainer_gate_waits}"
                        ),
                    )
                if exhausted:
                    # The geometry never resolved.  Fail closed by handing the
                    # objective back to the frontier instead of waiting forever.
                    return
                yield
                continue
            raise BotModeError(f"no reachable encounter tile on map {map_id!r}: {error}")
        trainer_gate_waits = 0
        attempts += 1
        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_TILE_SELECTED: "
                f"map={map_id!r} start={overworld.player_coordinates!r} "
                f"goal={navigation_goal!r} attempt={attempts}"
            ),
            trace=True,
        )
        yield from AgentControlLoop(lambda: observe_agent(goal=navigation_goal), goal=navigation_goal).run()
        # Reaching grass is only the setup boundary. The ROM does not roll
        # while the avatar is idle, so explicitly drive movement/turn inputs
        # until BattleListener observes the desired wild battle.
        yield from _acquire_campaign_encounter(map_id)
        return


def _publish_preparation_target(navigation_goal, overworld):
    """Publish the concrete target selected by the preparation navigator."""

    target = describe_preparation_target(navigation_goal, overworld)
    if target is None:
        # No published target means the wild-battle policy will not treat the
        # battles that follow as training.  Never let that be silent.
        print_always(
            lambda: (
                "CAMPAIGN_PREPARATION_TARGET_UNDESCRIBED: "
                f"goal={navigation_goal!r} action='wild_battles_will_not_be_training'"
            ),
        )
        return None
    context.campaign_preparation_target = target.description
    status = getattr(context, "campaign_status", None)
    if status is not None and not getattr(status, "complete", False):
        context.campaign_status = replace(status, target=target.semantic_target)
    diagnostic_print(
        lambda: f"CAMPAIGN_PREPARATION_TARGET: {target.description}",
        trace=False,
    )
    return target


def _campaign_preparation_center(location) -> PokemonCenter:
    """Select a registered Center for preparation's recovery owner.

    Preparation executes recovery through the campaign route planner, so the
    nested encounter loop must not reopen the legacy synchronous path search.
    The registry supplies the bounded set of possible Centers; the recovery
    executor validates the actual route when it is mounted.
    """

    candidates = pokemon_center_candidates(location)
    if not candidates:
        raise BotModeError(f"no registered recovery Center is available for preparation at {location!r}")
    return candidates[0]


def _execute_preparation_recovery(center: PokemonCenter) -> Iterator[object]:
    """Recover and return the preparation loop to an outdoor map.

    ``execute_planned_recovery`` owns the Center interior through healing and
    PC cleanup.  The generic encounter loop, however, resumes by navigating
    to its grass tile, so preparation must explicitly cross the Center exit
    before handing control back to that loop.
    """

    yield from execute_planned_recovery(center.value)
    source = emerald_healing_source_for_destination(center.value)
    if source is None:
        return
    while True:
        observation = observe_agent(require_overworld=True)
        overworld = getattr(observation, "overworld", None)
        if overworld is None or getattr(overworld, "map_id", None) != source.interior_map:
            return
        if getattr(overworld, "controllable", False):
            break
        yield
    yield from navigate_to(source.interior_map, (7, 8))


def trainer_checkpoint_ready_check(checkpoint):
    """Return a zero-argument check: does the live party have a ready line for this trainer?

    Uses the same pre-battle assessment the campaign readiness gate uses, so
    preparation stops exactly when that gate would stop asking for more.
    """

    def check() -> bool:
        readiness = assess_live_trainer_checkpoint(checkpoint, get_party())
        return getattr(getattr(readiness, "status", None), "value", None) == "ready"

    return check


def execute_campaign_preparation(
    training_map,
    *,
    target_level: int,
    ready_check=None,
) -> Iterator[object]:
    """Train the current living party to a bounded target before a boss.

    The target is checked from the live party on every loop boundary. The
    Center loop owns healing and encounter-area movement, while the campaign
    mode supplies the Nuzlocke-aware battle strategy at battle entry.

    ``target_level`` is a hint, not the only way to finish: when ``ready_check``
    is given and reports that the live party already has a ready line against
    the trainer, preparation ends without waiting for every member to reach the
    target.  The check is re-run only when the party's readiness-relevant facts
    change (species, level, max HP, or whether a member is at least half HP).
    """
    if target_level < 1:
        raise ValueError("preparation target must be positive")

    context.campaign_preparation_target = None
    context.campaign_preparation_exp_budget = None

    from modules.battle_strategies.nuzlocke_level_balancing import (
        NuzlockeLevelBalancingBattleStrategy,
    )
    loop = None
    last_preparation_roll_direction: Direction | None = None

    def preparation_activity() -> None:
        """Issue one bounded encounter input from the settled live tile.

        The generic ``spin`` helper assumes a traversable patch of terrain.
        Preparation can deliberately select a one-tile grass patch, where
        spinning only changes facing or walks away and never produces the
        re-entry roll that Gen III uses to start a wild battle. Reuse the
        campaign encounter input policy here, including its explicit
        step-off/re-entry behavior, while leaving battle and healing control
        with the existing loop.
        """

        nonlocal last_preparation_roll_direction
        observation = observe_agent()
        if observation.interaction_type is not InteractionType.OVERWORLD:
            return
        overworld = getattr(observation, "overworld", None)
        if overworld is None or not observation.interaction.controllable:
            return
        if getattr(overworld, "movement_state", None) is not None and overworld.movement_state is not MovementState.STANDING:
            return
        direction, reason = _encounter_roll_direction(overworld)
        if direction is not None:
            _issue_encounter_roll_input(direction)
            last_preparation_roll_direction = direction
            return
        if reason == "not_on_encounter_terrain" and last_preparation_roll_direction is not None:
            # The preceding bounded input stepped off a one-tile patch. The
            # opposite fresh edge returns to that tile and is itself a legal
            # encounter roll.
            direction = last_preparation_roll_direction.opposite()
            _issue_encounter_roll_input(direction)
            last_preparation_roll_direction = direction

    def recover_before_unready_trainer(goal) -> Iterator[object]:
        """Heal before a known trainer when the line is unsafe because of HP/status."""

        if not isinstance(goal, NavigationGoal) or not isinstance(goal.target, EngageTrainer):
            return False
        readiness = getattr(context, "campaign_trainer_readiness", None)
        if getattr(getattr(readiness, "status", None), "value", None) != "prepare":
            return False
        party = get_party()
        health_or_status_risk = any(
            not pokemon.is_egg
            and (
                pokemon.current_hp <= 0
                or pokemon.current_hp / max(1, pokemon.total_hp) < 0.5
                or getattr(getattr(pokemon, "status_condition", None), "value", "none") != "none"
            )
            for pokemon in party
        )
        if not health_or_status_risk:
            raise BotModeError(
                "known trainer checkpoint has no robust battle line; further preparation is required"
            )
        recovery = observe_route_recovery()
        if not (
            recovery.center_available
            and recovery.safe_to_reach_center
            and recovery.center_location is not None
            and recovery.route is not None
        ):
            raise BotModeError(
                "known trainer checkpoint is unsafe and no executable recovery route is available"
            )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_TRAINER_PREPARATION_RECOVERY: "
                f"trainer={goal.target.trainer_id!r} destination={recovery.center_location!r} "
                f"reason={getattr(readiness, 'reason', None)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_TRAINER_PREPARATION_RECOVERY",
        )
        yield from execute_planned_recovery(
            recovery.center_location,
            planned_source=recovery.healing_source,
            planned_route=recovery.route,
            planned_goal=recovery.navigation_goal,
        )
        return True

    def target_reached() -> bool:
        """Check the live living-party levels and relay battle boundaries."""

        party = get_party()
        if loop is not None and any(not pokemon.is_egg for pokemon in party):
            # PokecenterLoopController receives its battle-end notification
            # through the mode/listener boundary. Preparation owns a nested
            # controller, so relay that boundary before evaluating whether a
            # heal or another encounter is needed.
            loop.on_battle_ended()
        runtime = getattr(context, "nuzlocke_runtime", None)
        dead = frozenset(
            getattr(getattr(getattr(runtime, "rules_projection", None), "state", None), "dead_pokemon", ()) or ()
        )
        living = tuple(
            pokemon
            for pokemon in party
            if not pokemon.is_egg
            and (
                runtime is None
                or ((identity := PokemonIdentity.from_pokemon(pokemon)) is not None and identity not in dead)
            )
        )
        if bool(living) and all(pokemon.level >= target_level for pokemon in living):
            return True
        return trainer_line_ready(living)

    ready_state = {"signature": None, "ready": False}

    def trainer_line_ready(living) -> bool:
        """Whether the live party already has a ready line against the trainer."""

        if ready_check is None or not living:
            return False
        signature = tuple(
            (
                getattr(getattr(pokemon, "species", None), "name", None),
                getattr(pokemon, "level", None),
                getattr(pokemon, "total_hp", None),
                getattr(pokemon, "current_hp", 0) * 2 >= (getattr(pokemon, "total_hp", 0) or 0),
            )
            for pokemon in living
        )
        if signature != ready_state["signature"]:
            ready_state["signature"] = signature
            try:
                ready_state["ready"] = bool(ready_check())
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
                ready_state["ready"] = False
            if ready_state["ready"]:
                print_always(
                    lambda: (
                        "CAMPAIGN_PREPARATION_READY: the party already has a ready line against the trainer; "
                        f"ending preparation before every member reaches target_level={target_level} "
                        f"levels={[getattr(pokemon, 'level', None) for pokemon in living]!r}"
                    )
                )
        return ready_state["ready"]

    if target_reached():
        return

    training_map, training_candidates = _preparation_training_location(training_map)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_PREPARATION_CANDIDATES: "
            f"map={training_map!r} count={len(training_candidates)} "
            f"first={training_candidates[:10]!r}"
        ),
        trace=True,
    )
    map_navigation_attempts = 0
    while True:
        preferred_map_route = _preparation_map_route(training_map)
        # Preparation travel is optional work, so the route into the training
        # map must keep known trainer sight lines as a hard safety boundary.
        map_goal = NavigationGoal(
            SemanticTarget.map(training_map),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            encounter_mode=EncounterMode.AVOID,
            preferred_map_route=preferred_map_route,
        )
        map_navigation_attempts += 1
        try:
            yield from AgentControlLoop(
                lambda: observe_agent(goal=map_goal),
                goal=map_goal,
            ).run()
        except NavigationError as error:
            # The global route plans through static data for the maps ahead and
            # can report an avoidable corridor as unreachable.  Fall back to the
            # staged travel the encounter capability uses: each stage searches
            # the live map and still enforces the stage's trainer policy, so
            # this never becomes a permissive route.
            print_always(
                lambda: (
                    "CAMPAIGN_PREPARATION_MAP_ROUTE: global route unavailable; "
                    f"using staged trainer-safe travel map={training_map!r} "
                    f"error_type={type(error).__name__!r} error={str(error)[:200]!r}"
                )
            )
            arrived = yield from _travel_to_map_staged(training_map, get_world_map_graph())
            if not arrived:
                raise BotModeError(
                    f"preparation could not reach training map {training_map!r}: "
                    f"no trainer-safe staged route ({error})"
                ) from error
        break

    # The map-level route intentionally stops at the observed entry tile. Pick
    # a grass tile from the settled, live topology rather than assuming the
    # first ROM encounter tile is in the entry-connected component.
    navigation_attempts = 0
    while True:
        observation = observe_agent()
        overworld = getattr(observation, "overworld", None)
        start = (
            (overworld.map_id, overworld.player_coordinates)
            if overworld is not None and getattr(overworld, "player_coordinates", None) is not None
            else None
        )
        if start is None or start[0] != training_map:
            raise BotModeError("training map transition did not settle on the selected map")
        if target_reached():
            return
        encounter_resolved = _preparation_encounter_resolved(training_map)
        navigation_goal, last_error = _preparation_navigation_goal(
            overworld,
            training_map,
            training_candidates,
            allow_wild=not encounter_resolved,
            target_level=target_level,
        )
        if navigation_goal is None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PREPARATION_CANDIDATE_FAILURE: "
                    f"map={training_map!r} start={start!r} count={len(training_candidates)} "
                    f"last_error={last_error!r}"
                ),
                trace=True,
            )
            raise BotModeError(
                f"no reachable encounter tile or clearing trainer is available on training map {training_map!r}: {last_error}"
            )

        recovered = yield from recover_before_unready_trainer(navigation_goal)
        if recovered:
            continue

        _publish_preparation_target(navigation_goal, overworld)
        navigation_attempts += 1
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_NAVIGATION_SELECTED: "
                f"map={training_map!r} start={start!r} goal={navigation_goal!r} "
                f"attempt={navigation_attempts}"
            ),
            trace=True,
        )
        try:
            yield from AgentControlLoop(
                lambda: observe_agent(goal=navigation_goal),
                goal=navigation_goal,
            ).run()
        except NavigationError as error:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PREPARATION_NAVIGATION_RETRY: "
                    f"map={training_map!r} start={start!r} goal={navigation_goal!r} "
                    f"attempt={navigation_attempts} error_type={type(error).__name__!r} error={error!r}"
                ),
                trace=True,
            )
            if navigation_attempts >= _PREPARATION_NAVIGATION_RETRY_LIMIT:
                raise BotModeError(
                    f"preparation navigation did not stabilize on training map {training_map!r}: {error}"
                ) from error
            continue
        if isinstance(navigation_goal, NavigationGoal) and isinstance(navigation_goal.target, EngageTrainer):
            # Clearing a trainer is an intermediate preparation step.  Keep
            # selecting from the live topology: if the area's encounter has
            # already been resolved this remains trainer-only, while a
            # trainer that merely blocked access can be followed by grass.
            continue
        break

    # A trainer-safe grass goal is a NavigationGoal wrapping the ReachLocation.
    selected_target = (
        navigation_goal.target if isinstance(navigation_goal, NavigationGoal) else navigation_goal
    )
    training_location = getattr(selected_target, "location", None)
    if training_location is not None:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_CANDIDATE_SELECTED: "
                f"map={training_map!r} training_location={training_location!r}"
            ),
            trace=True,
        )

    loop = CampaignPreparationLoopController(
        recovery_handler=_execute_preparation_recovery,
        center_finder=_campaign_preparation_center,
        activity_handler=preparation_activity,
    )
    loop.battle_strategy = NuzlockeLevelBalancingBattleStrategy
    diagnostic_print(
        lambda: (
            "CAMPAIGN_PREPARATION_TRAINING_LOOP: "
            f"map={training_map!r} training_location={training_location!r} target_level={target_level}"
        ),
        trace=True,
    )
    loop.verify_on_start()
    target_was_reached = False

    def stop_training() -> bool:
        """Stop at the target or when the first encounter reaches a boundary."""

        nonlocal target_was_reached
        # Stop at the first terminal encounter boundary so the outer
        # preparation loop can hand off the first encounter objective. After
        # that boundary, its continuation may deliberately return to grass for
        # duplicate wild battles used only for level training.
        target_was_reached = target_reached()
        return target_was_reached or _preparation_encounter_resolved(training_map)

    yield from loop.run(stop_condition=stop_training)

    if target_was_reached:
        return
    if not _preparation_encounter_resolved(training_map):
        # Preserve the existing recovery/whiteout termination behavior.  The
        # trainer handoff is only valid when the loop stopped at a terminal
        # encounter boundary; an interrupted loop must not invent a new route.
        return

    # The first encounter loop deliberately ends after the area's encounter is
    # resolved. Continue through the same observed topology, returning to
    # grass for legal duplicate wild battles when a reachable training tile is
    # available. If a trainer partitions that grass component, the selector
    # still returns the trainer as an intermediate preparation step.
    while not target_reached():
        observation = observe_agent()
        overworld = getattr(observation, "overworld", None)
        start = (
            (overworld.map_id, overworld.player_coordinates)
            if overworld is not None and getattr(overworld, "player_coordinates", None) is not None
            else None
        )
        if start is None or start[0] != training_map:
            raise BotModeError("preparation trainer handoff did not settle on the selected map")
        trainer_goal, last_error = _preparation_navigation_goal(
            overworld,
            training_map,
            training_candidates,
            allow_wild=True,
            target_level=target_level,
        )
        if trainer_goal is None:
            raise BotModeError(
                f"no reachable trainer is available for safe preparation on training map {training_map!r}: {last_error}"
            )
        recovered = yield from recover_before_unready_trainer(trainer_goal)
        if recovered:
            continue
        _publish_preparation_target(trainer_goal, overworld)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_TRAINER_CONTINUATION: "
                f"map={training_map!r} start={start!r} goal={trainer_goal!r}"
            ),
            trace=True,
        )
        yield from AgentControlLoop(
            lambda: observe_agent(goal=trainer_goal),
            goal=trainer_goal,
        ).run()
        if isinstance(trainer_goal, NavigationGoal) and isinstance(trainer_goal.target, EngageTrainer):
            continue
        # The first encounter has already been resolved. A reachable grass
        # goal is now the owned training destination; re-enter the same loop
        # so later wild battles are fought for experience until the target is
        # reached instead of repeatedly selecting this completed location.
        break

    if not target_reached():
        yield from loop.run(stop_condition=target_reached)


def execute_planned_recovery(
    destination,
    planned_source=None,
    planned_route=None,
    planned_goal=None,
) -> Iterator[object]:
    """Execute recovery at the Center selected by campaign planning.

    The route destination is intentionally passed in from the plan.  The
    interaction affordance is still discovered from a fresh interior
    observation, but execution never reselects a different Center.  When a
    planner-composed route is supplied, it is adopted directly for the
    outdoor leg instead of being recomputed by the tactical navigator.
    """
    cleanup_marker = getattr(context, "campaign_recovery_phase", None)
    if cleanup_marker == ("cleanup", destination):
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_PHASE: phase='pc_cleanup_resume' "
                f"destination={destination!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_PHASE",
        )
        yield from _execute_recovery_pc_cleanup()
        context.campaign_recovery_phase = None
        return
    if party_is_restored():
        # Recovery can be requested for a supply/readiness boundary after the
        # party has already been restored by an intervening action. Continue
        # only when stored medicine makes the already-selected Center visit
        # useful; otherwise preserve the old no-op behavior.
        snapshot = observe_resource_snapshot()
        if not snapshot.pc_healing_items and not _remembered_fainted_party_members():
            return
    center = next((candidate for candidate in PokemonCenter if candidate.value == destination), None)
    catalog_source = planned_source or emerald_healing_source_for_destination(destination)
    interior_map = getattr(catalog_source, "interior_map", None)
    if interior_map is None:
        raise RuntimeError(f"No healing-location interior is defined for {destination!r}")
    # Catalog Center destinations name the exterior door tile, which is a
    # blocked warp entry in the ROM rather than a standable location.  Ask the
    # world navigator to execute the exact door warp so it can stop on the
    # adjacent activation tile and issue the required directional input.
    # Keep the recovery safety policy on the goal that owns the route.  A bare
    # ReachWarp carries GoalConstraints() and therefore silently changes
    # TrainerMode.AVOID back to TrainerMode.IGNORE when a supplied route is
    # invalidated (most importantly after a trainer battle).
    # Route planning may have selected a specific Center door transition.
    # Preserve that exact goal through execution; rebuilding it from only the
    # destination loses the transition and creates an unconstrained
    # ``ReachWarp(warp=None)`` search at the tactical handoff.
    navigation_goal = planned_goal or _recovery_navigation_goal(destination, catalog_source)
    safety_error = _recovery_trainer_safety_error(planned_route, navigation_goal)
    if safety_error is not None:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_TRAINER_SAFETY_BLOCKED: "
                f"destination={destination!r} reason={safety_error!r} "
                f"trainer_checkpoint={getattr(getattr(context, 'campaign_trainer_checkpoint', None), 'checkpoint_id', None)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_TRAINER_SAFETY_BLOCKED",
        )
        raise RecoveryTrainerSafetyError(safety_error)
    # Keep a durable phase marker around every route/interior/action boundary
    # so a recovery exception identifies the stage that failed. The previous
    # failure panel only exposed the final traceback line, which made a
    # transient KeyError impossible to distinguish from a stale route,
    # dialogue, or PC interaction lookup.
    context.campaign_recovery_phase = ("route_navigation", destination)
    context.campaign_status = recovery_status(
        SemanticTarget.at(destination),
        "Navigate to healing source",
    )
    if planned_route is not None and planned_goal is None:
        # A legacy recovery route includes the final door warp and therefore
        # terminates on the Center's interior map. A planner-composed
        # ReachLocation route, by contrast, ends on the selected exterior
        # destination and retains the historical goal contract.
        # A legacy route can terminate on the outdoor Center destination,
        # while a world-composed route normally terminates after the door warp
        # on the interior map. Preserve the route shape, but retain the
        # recovery constraints in either case.
        planned_goal = NavigationGoal(
            ReachLocation(destination),
            constraints=navigation_goal.constraints,
            encounter_mode=navigation_goal.encounter_mode,
            encounter_penalty=navigation_goal.encounter_penalty,
        )
        planned_destination = getattr(planned_route, "destination", None)
        if (
            isinstance(planned_destination, tuple)
            and len(planned_destination) == 2
            and planned_destination[0] != destination[0]
        ):
            planned_goal = navigation_goal
        yield from AgentControlLoop(
            lambda: observe_agent(goal=planned_goal),
            goal=planned_goal,
            navigation_plan=planned_route,
            logger=_recovery_agent_logger,
            use_movement_batch=False,
        ).run()
    elif planned_route is not None:
        yield from AgentControlLoop(
            lambda: observe_agent(goal=navigation_goal),
            goal=navigation_goal,
            navigation_plan=planned_route,
            logger=_recovery_agent_logger,
            use_movement_batch=False,
        ).run()
    else:
        yield from AgentControlLoop(
            lambda: observe_agent(goal=navigation_goal),
            goal=navigation_goal,
            logger=_recovery_agent_logger,
            use_movement_batch=False,
        ).run()
    context.campaign_recovery_phase = ("center_interior", destination)
    if center is not None:
        yield from _wait_for_center_interior(center)
    elif catalog_source is not None:
        yield from _wait_for_healing_interior(catalog_source)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_INTERIOR: "
            f"phase=entered_destination destination={destination!r} "
            f"frame={getattr(context, 'frame', None)!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_INTERIOR",
    )
    interior_map = getattr(
        catalog_source,
        "interior_map",
        getattr(center, "name", None),
    )
    context.campaign_status = recovery_status(
        SemanticTarget.map(interior_map),
        "Navigate to nurse" if center is not None else "Navigate to healing source",
    )
    executor = AgentActionExecutor()
    while True:
        context.campaign_recovery_phase = ("source_discovery", destination)
        # This loop's only completion condition is a physical map change.
        # Observe the map on every frame, including the transient CHANGE_MAP
        # classification produced by a door warp.
        observation = observe_agent(require_overworld=True)
        source = discover_healing_source(observation=observation)
        if source is not None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_SOURCE: "
                    f"source={source.source_id!r} trigger={source.interaction_trigger_id!r} "
                    f"activation_observed=True frame={getattr(context, 'frame', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_SOURCE",
            )
            context.campaign_status = recovery_status(
                SemanticTarget.interaction(
                    getattr(observation.overworld, "map_id", interior_map),
                    source.interaction_trigger_id or source.source_id,
                ),
                (
                    "Navigate to nurse"
                    if getattr(source, "source_type", None) is HealingSourceType.POKEMON_CENTER_NURSE
                    else "Navigate to healing source"
                ),
            )
            context.campaign_recovery_phase = ("interaction_navigation", destination)
            # The nurse is separated from the player by the Center counter.
            # Reaching the Center map is not the same as reaching an
            # activation position; use the observed trigger geometry so the
            # navigator can select the tile in front of (or beyond) the
            # counter, exactly as ordinary object interactions do.
            if source.interaction_trigger_id is not None:
                interaction_goal = getattr(
                    source,
                    "navigation_goal",
                    ReachInteractionPosition(source.interaction_trigger_id),
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_INTERACTION_PLAN: "
                        f"goal={interaction_goal!r} frame={getattr(context, 'frame', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_INTERACTION_PLAN",
                )
                yield from AgentControlLoop(
                    lambda: observe_agent(goal=interaction_goal),
                    goal=interaction_goal,
                ).run()
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_INTERACTION: "
                    f"phase=position_reached trigger={source.interaction_trigger_id!r} "
                    f"frame={getattr(context, 'frame', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_INTERACTION",
            )
            context.campaign_status = recovery_status(
                SemanticTarget.interaction(
                    getattr(observation.overworld, "map_id", interior_map),
                    source.interaction_trigger_id or source.source_id,
                ),
                (
                    "Interact with nurse"
                    if getattr(source, "source_type", None) is HealingSourceType.POKEMON_CENTER_NURSE
                    else "Interact with healing source"
                ),
            )
            context.campaign_recovery_phase = ("healing_interaction", destination)
            yield from _execute_healing_source_interaction(source)
            if not party_is_restored():
                raise RuntimeError("HEALING_NOT_CONFIRMED")
            context.campaign_status = recovery_status(
                SemanticTarget.interaction(
                    getattr(observation.overworld, "map_id", interior_map),
                    source.interaction_trigger_id or source.source_id,
                ),
                "Healing confirmed",
            )
            # Emerald stores the early Potion in the item PC. Once the party
            # has been healed, use the observed PC in this same Center before
            # returning to campaign travel so the medicine becomes available
            # to battle recovery and the Mart baseline does not ignore usable
            # stored stock.
            if getattr(source, "source_type", None) is HealingSourceType.POKEMON_CENTER_NURSE:
                # Healing is a confirmed phase boundary.  The PC transaction
                # is cleanup and may fail independently; a retry resumes at
                # this marker instead of travelling back through the Center
                # and replaying the nurse interaction.
                context.campaign_recovery_phase = ("cleanup", destination)
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_PHASE: phase='healing_confirmed' "
                        f"destination={destination!r} cleanup_required=True"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_PHASE",
                )
                yield from _execute_recovery_pc_cleanup()
                context.campaign_recovery_phase = None
            return
        if observation.interaction_type.name == "DIALOGUE" or not observation.interaction.controllable:
            decision = select_action(observation)
            executor.execute(decision.action, observation)
        yield


def _execute_recovery_pc_cleanup() -> Iterator[object]:
    """Withdraw stored medicine and record fainted Pokémon after healing."""

    diagnostic_print(
        lambda: "CAMPAIGN_RECOVERY_PHASE: phase='pc_cleanup_begin'",
        trace=True,
        prefix="CAMPAIGN_RECOVERY_PHASE",
    )
    yield from _withdraw_pc_healing_item_at_current_location()
    yield from _deposit_fainted_party_at_current_location()
    diagnostic_print(
        lambda: "CAMPAIGN_RECOVERY_PHASE: phase='pc_cleanup_confirmed'",
        trace=True,
        prefix="CAMPAIGN_RECOVERY_PHASE",
    )


class RecoveryTrainerSafetyError(NavigationError):
    """A recovery route would expose a trainer that readiness rejected."""


def _location_set_for_navigation_route(route) -> frozenset:
    """Return every concrete location represented by a navigation route."""

    if route is None:
        return frozenset()
    locations = set()
    for action in getattr(route, "actions", ()) or ():
        for location in (getattr(action, "source", None), getattr(action, "destination", None)):
            if isinstance(location, tuple) and len(location) == 2:
                locations.add(location)
    destination = getattr(route, "destination", None)
    if isinstance(destination, tuple) and len(destination) == 2:
        locations.add(destination)
    return frozenset(locations)


def _checkpoint_trainer_aliases(checkpoint) -> frozenset[str]:
    """Return route and ROM identities for one readiness checkpoint."""

    if checkpoint is None:
        return frozenset()
    aliases = {
        value
        for value in (
            getattr(checkpoint, "checkpoint_id", None),
            getattr(checkpoint, "trainer_affordance_id", None),
        )
        if isinstance(value, str) and value
    }
    rom_id = getattr(checkpoint, "rom_trainer_id", None)
    if rom_id is not None:
        aliases.add(f"rom_trainer:{rom_id}")
    return frozenset(aliases)


def _recovery_trainer_safety_error(route, navigation_goal) -> str | None:
    """Explain why a selected recovery route cannot be mounted safely.

    Recovery is a safety action.  A route composed while the party was in a
    different observation envelope must retain the hard AVOID contract, and a
    route that intersects the hazard geometry recorded by the readiness
    diagnostic cannot be accepted merely because its planner result was
    otherwise executable.
    """

    constraints = getattr(navigation_goal, "constraints", None)
    if getattr(getattr(constraints, "trainer_mode", None), "name", None) != "AVOID":
        return "recovery navigation goal does not enforce TrainerMode.AVOID"
    if getattr(constraints, "allow_forced_trainer_exposure", False):
        return "recovery navigation goal explicitly allows forced trainer exposure"
    if getattr(route, "forced_trainer_exposure", False):
        return "selected recovery route is marked as forced trainer exposure"

    readiness = getattr(context, "campaign_trainer_readiness", None)
    status = getattr(getattr(readiness, "status", None), "value", None)
    aliases: set[str] = set()
    if status == "prepare":
        checkpoints = [
            getattr(context, "campaign_trainer_checkpoint", None),
            getattr(getattr(context, "campaign_readiness_diagnostic", None), "trainer_checkpoint", None),
        ]
        for checkpoint in checkpoints:
            aliases.update(_checkpoint_trainer_aliases(checkpoint))
        if not aliases:
            # The readiness result is explicitly unready but has no stable owner.
            # Keep the route from being mounted under a possibly stale hazard
            # envelope; the next readiness pass must reconstruct the checkpoint.
            return "readiness marked a trainer unready but supplied no stable checkpoint identity"

    # Every recorded risk describes an undefeated trainer.  Recovery heals a
    # damaged party, so it may not expose any of them: an explicitly unready
    # checkpoint is reported by name, but an unrelated undefeated trainer on
    # the route is just as unsafe to meet while recovering.
    diagnostic = getattr(context, "campaign_readiness_diagnostic", None)
    route_locations = _location_set_for_navigation_route(route)
    for risk in tuple(getattr(diagnostic, "trainer_risks", ()) or ()):
        trainer_id = getattr(risk, "trainer_id", None)
        hazards = frozenset(getattr(risk, "hazard_locations", ()) or ())
        conflict = route_locations.intersection(hazards)
        if not conflict:
            continue
        locations = tuple(sorted(conflict, key=repr))
        if trainer_id in aliases:
            return f"route intersects explicitly unready trainer {trainer_id!r} at {locations!r}"
        return f"route intersects undefeated trainer {trainer_id!r} at {locations!r}"
    return None


def execute_trainer_roster_preparation(pc_candidate_index: int) -> Iterator[object]:
    """Reach a safe PC boundary and replace the weakest party slot.

    ``pc_candidate_index`` is the negative normalized index emitted by
    :func:`live_storage_readiness_members`.  The route is healed on arrival,
    then the selected boxed Pokémon is withdrawn. A full party deposits its
    lowest-level living member first; a smaller party simply withdraws the
    candidate.
    """

    if not isinstance(pc_candidate_index, int) or pc_candidate_index >= 0:
        raise BotModeError("trainer roster preparation requires a normalized PC candidate index")
    absolute = -pc_candidate_index - 1
    box_index, slot_index = divmod(absolute, 30)
    storage = get_pokemon_storage()
    boxes = tuple(getattr(storage, "boxes", ()) or ()) if storage is not None else ()
    if box_index >= len(boxes):
        raise BotModeError("selected trainer preparation Pokémon is no longer in PC storage")
    boxed_slot = next(
        (slot for slot in getattr(boxes[box_index], "slots", ()) if getattr(slot, "slot_index", None) == slot_index),
        None,
    )
    boxed_pokemon = getattr(boxed_slot, "pokemon", None)
    if boxed_pokemon is None:
        raise BotModeError("selected trainer preparation Pokémon is no longer in PC storage")

    route = observe_route_recovery()
    if not route.center_available or not route.safe_to_reach_center or route.route is None:
        raise BotModeError("no safe executable route to a PC is available for trainer roster preparation")
    yield from execute_campaign_recovery(route)

    party = tuple(get_party())
    deposit = None
    if len(party) >= 6:
        living = tuple(
            pokemon for pokemon in party if getattr(pokemon, "current_hp", 0) > 0 and not getattr(pokemon, "is_egg", False)
        )
        deposit = min(
            living,
            key=lambda pokemon: (
                getattr(pokemon, "level", 0),
                getattr(pokemon, "total_exp", 0),
                getattr(pokemon, "party_index", 0),
            ),
            default=None,
        )
        if deposit is None:
            raise BotModeError("cannot make room for the selected trainer preparation Pokémon")

    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation)
    if trigger is None or not getattr(trigger, "activation_locations", ()):
        used_fallback = yield from _navigate_to_known_center_pc(
            observation,
            purpose="trainer roster preparation",
        )
        if not used_fallback:
            raise BotModeError("Center PC was not observable for trainer roster preparation")
    else:
        current = (observation.overworld.map_id, observation.overworld.player_coordinates)
        target = next(
            (location for location in trigger.activation_locations if location == current),
            sorted(trigger.activation_locations, key=repr)[0],
        )
        goal = NavigationGoal(
            ReachLocation(target),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        yield from AgentControlLoop(
            lambda: observe_agent(goal=goal),
            goal=goal,
            use_movement_batch=False,
        ).run()
        observation = observe_agent(require_overworld=True)
        trigger = _observed_pc_trigger(observation) or trigger
        required_facing = dict(getattr(trigger, "activation_requirements", ())).get(
            (observation.overworld.map_id, observation.overworld.player_coordinates)
        )
        if required_facing is not None and getattr(observation.overworld, "facing", None) is not required_facing:
            press_direction = getattr(context.emulator, "press_direction", None)
            if callable(press_direction):
                press_direction(required_facing.button_name, run=False, fresh=True)
            else:
                context.emulator.press_button(required_facing.button_name)
            yield

    actions = []
    if deposit is not None:
        actions.append(PCAction.deposit_pokemon_to_box(deposit))
    actions.append(PCAction.withdraw_pokemon_from_box(boxed_pokemon))
    context.campaign_status = recovery_status(
        SemanticTarget.interaction(getattr(observation.overworld, "map_id", None), "EventScript_PC"),
        "Apply trainer roster plan",
    )
    yield from interact_with_pc(actions)
    if not any(
        getattr(pokemon, "data", b"")[:4] == getattr(boxed_pokemon, "data", b"")[:4]
        for pokemon in get_party()
    ):
        raise BotModeError("trainer roster withdrawal was not observed in the party")
    diagnostic_print(
        lambda: (
            "CAMPAIGN_TRAINER_ROSTER_PREPARATION: "
            f"pc_index={pc_candidate_index!r} deposited={getattr(deposit, 'species', None)!r} "
            f"withdrawn={getattr(boxed_pokemon, 'species', None)!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_TRAINER_ROSTER_PREPARATION",
    )


def _execute_healing_source_interaction(source: HealingSource) -> Iterator[object]:
    """Interact with a reached semantic healing source through shared control.

    This deliberately has no Center-specific dialogue loop.  Every input is
    preceded by an authoritative observation and is followed by a yield so the
    next emulator frame is reclassified by the common interaction pipeline.
    """
    executor = AgentActionExecutor()
    planned_source = source
    interaction_started = False
    interaction_start_observations = 0
    last_observation = None
    last_party_restored = False
    dialogue_input_in_flight: tuple[object, ...] | None = None
    dialogue_input_waits = 0
    stable_standing_observations = 0
    # Script activation is asynchronous: the overworld can remain apparently
    # controllable for several observations after the A input is accepted.
    # Keep ownership during that transition, but retain a bounded failure
    # path for a genuinely ignored interaction.
    interaction_start_timeout = 32
    while True:
        if (
            last_observation is not None
            and last_party_restored
            and not last_observation.interaction.script_active
            and last_observation.interaction.interaction_phase is InteractionPhase.NONE
        ):
            diagnostic_print("HEAL_PARTY_COMPLETE: party_fully_restored=True interaction_finished=True", trace=True)
            return
        observation = observe_agent()
        if interaction_started and observation.overworld is not None:
            context.campaign_status = recovery_status(
                SemanticTarget.interaction(
                    observation.overworld.map_id,
                    source.interaction_trigger_id or source.source_id,
                ),
                "Confirm healing",
            )
        observed_source = (
            discover_healing_source(observation=observation)
            if getattr(observation, "overworld", None) is not None
            else None
        )
        if observed_source is None:
            # The activation trigger can disappear from a transient Center
            # observation exactly when the avatar is standing at the counter
            # while the nurse script is being installed. The route and
            # trigger selected before navigation remain authoritative; losing
            # them here used to yield forever with no action or error.
            source = planned_source
            diagnostic_print(
                lambda: (
                    "HEALING_SOURCE_OBSERVATION_RETAINED: "
                    f"source={source.source_id!r} interaction_started={interaction_started}"
                ),
                trace=True,
                prefix="HEALING_SOURCE_OBSERVATION_RETAINED",
            )
        else:
            source = observed_source
        party_restored = party_is_restored()
        # Restored HP is not the same boundary as completion of the nurse's
        # event script. Emerald restores the party before displaying the
        # closing messages, so leave the capability mounted until the script
        # itself ends. Avatar controllability can lag that fact by one frame;
        # waiting for it gives the prior dialogue A an opportunity to be
        # consumed as a new nurse interaction.
        interaction_finished = (
            not observation.interaction.script_active
            and observation.interaction.interaction_phase is InteractionPhase.NONE
        )
        if party_restored and interaction_finished:
            diagnostic_print("HEAL_PARTY_COMPLETE: party_fully_restored=True interaction_finished=True", trace=True)
            return
        last_observation = observation
        last_party_restored = party_restored
        input_signature = (
            observation.interaction.script_active,
            observation.interaction.script_function,
            observation.interaction.native_function,
            observation.interaction.interaction_phase,
            observation.interaction.choice_menu_active,
            observation.interaction.choice_menu_input_ready,
            observation.interaction.choice_selected,
        )
        if dialogue_input_in_flight is not None:
            if input_signature == dialogue_input_in_flight:
                dialogue_input_waits += 1
                if dialogue_input_waits <= 3:
                    yield
                    continue
            dialogue_input_in_flight = None
            dialogue_input_waits = 0
        # A script may remain active while its task-owned confirmation menu
        # is already accepting input.  The task-backed interaction
        # observation is the authoritative ownership boundary for this
        # moment; do not let the broader script-active predicate hide it.
        confirmation_ready = (
            observation.interaction_type.name == "CHOICE"
            and observation.interaction.choice_menu_active
            and observation.interaction.choice_menu_input_ready
        )
        if (
            confirmation_ready
            or observation.interaction_type.name == "DIALOGUE"
            or not observation.interaction.controllable
        ):
            decision = select_action(observation)
            executor.execute(decision.action, observation)
            if decision.action.action_type is AgentActionType.ADVANCE_DIALOGUE:
                dialogue_input_in_flight = input_signature
                dialogue_input_waits = 0
            yield
            continue
        if observation.interaction.script_active:
            # The script owns the interaction even when Emerald has not yet
            # withdrawn avatar control or displayed its first message.
            yield
            continue
        if not interaction_started:
            # A navigation goal can become geometrically complete on the same
            # observation that the avatar is still finishing its last step.
            # Pressing A in that frame is lost (or, worse, consumed by the
            # next field interaction).  Require two consecutive authoritative
            # standing observations before starting a healing interaction.
            # Synthetic observations used by unit callers may omit movement
            # state; retain their existing behavior in that case.
            movement_state = getattr(observation.overworld, "movement_state", None)
            if observation.overworld is not None and movement_state is not None:
                controllable = getattr(observation.overworld, "controllable", False)
                if movement_state is not MovementState.STANDING or not controllable:
                    stable_standing_observations = 0
                    diagnostic_print(
                        lambda: (
                            "HEALING_SOURCE_INTERACTION_WAIT: "
                            f"source={source.source_id!r} movement="
                            f"{getattr(movement_state, 'name', None)!r} controllable={controllable!r}"
                        ),
                        trace=True,
                        prefix="HEALING_SOURCE_INTERACTION_WAIT",
                    )
                    yield
                    continue
                stable_standing_observations += 1
                if stable_standing_observations < 2:
                    yield
                    continue
            diagnostic_print(lambda: f"HEALING_SOURCE_INTERACTION_BEGIN: source={source.source_id!r}", trace=True)
            executor.execute(
                AgentAction(
                    AgentActionType.INTERACT,
                    option=source.source_id,
                    reason="healing source reached; begin semantic interaction",
                ),
                observation,
            )
            interaction_started = True
            yield
            continue
        interaction_start_observations += 1
        if interaction_start_observations >= interaction_start_timeout:
            raise RuntimeError("HEALING_NOT_CONFIRMED")
        diagnostic_print(
            lambda: (
                "INTERACTION_STARTING: "
                f"source={source.source_id!r} observation={interaction_start_observations} "
                f"timeout={interaction_start_timeout}"
            ),
            trace=True,
            prefix="INTERACTION_STARTING",
        )
        yield


def prepare_route_recovery(
    *, candidate_limit: int | None = None, critical: bool = False
) -> RouteRecoveryRequest | RouteRecovery:
    """Read live route inputs without executing any candidate path searches."""

    try:
        location = get_player_location()
    except (KeyError, RuntimeError) as error:
        if isinstance(error, KeyError):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_OBSERVATION_RETRY: "
                    "stage='player_location' exception_type='KeyError' "
                    f"key={error.args!r} traceback={traceback.format_exc()!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_OBSERVATION_RETRY",
            )
        return RouteRecovery(observation_available=False, observation_error=str(error))

    if (
        not isinstance(location, tuple)
        or len(location) != 2
        or location[0] is None
        or not isinstance(location[1], tuple)
        or len(location[1]) != 2
        or any(coordinate is None for coordinate in location[1])
    ):
        return RouteRecovery(observation_available=False, observation_error="player_location_unavailable")

    center = None
    center_lookup_error = None
    is_rse = bool(getattr(getattr(context, "rom", None), "is_rse", False))
    if is_rse:
        # The legacy helper performs two synchronous full-map path searches
        # per candidate. That work used to happen before the background
        # planner was even submitted and was the source of multi-second (and
        # occasionally tens-of-seconds) recovery stalls. For Emerald, the
        # catalog is already the authoritative candidate set; route distance
        # and reachability are validated by ``plan_route_recovery`` in the
        # worker below.
        map_centers = pokemon_center_candidates(location)
        center = map_centers[0] if map_centers else None
    else:
        try:
            center = find_closest_pokemon_center(location)
        except (BotModeError, PathFindingError) as error:
            center_lookup_error = error
    center_location = getattr(center, "value", None)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_ROUTE_SELECTED: "
            f"source={location!r} center={center!r} destination={center_location!r} "
            f"legacy_error={center_lookup_error!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_ROUTE_SELECTED",
    )
    if center is not None and (
        not isinstance(center_location, tuple)
        or len(center_location) != 2
        or center_location[0] is None
        or not isinstance(center_location[1], tuple)
        or len(center_location[1]) != 2
        or any(coordinate is None for coordinate in center_location[1])
    ):
        return RouteRecovery(observation_available=False, observation_error="center_location_unavailable")

    if center is not None:
        if is_rse:
            # Keep every catalog source registered for this map so a nearby
            # alternative (Route 102 -> Petalburg, for example) survives a
            # stale/failed first route search. Static graph ranking below
            # accounts for entrance/exit traversal within that bounded local
            # candidate set.
            map_centers = pokemon_center_candidates(location)
            sources = tuple(
                source
                for source in emerald_healing_sources()
                if any(
                    _normalized_location(source.outdoor_location) == _normalized_location(candidate.value)
                    for candidate in map_centers
                )
            )
            if not sources:
                # A known local candidate set is authoritative. Falling back
                # to the global catalog here makes future centers compete
                # with the actually nearby sources when a representation
                # mismatch occurs (for example Route 104 selecting Oldale).
                source_map = _map_id_value(location[0])
                sources = tuple(
                    source
                    for source in emerald_healing_sources()
                    if _normalized_location(source.outdoor_location)[0] == source_map
                    or _map_id_value(source.interior_map) == source_map
                )
                if not sources:
                    source = emerald_healing_source_for_destination(_normalized_location(center_location))
                    sources = (source,) if source is not None else ()
        else:
            source = emerald_healing_source_for_destination(center_location)
            sources = (source,) if source is not None else (None,)
    elif is_rse:
        source_map = location[0]
        map_centers = pokemon_center_candidates(location)
        mapped_sources = tuple(
            source
            for source in emerald_healing_sources()
            if any(
                _normalized_location(source.outdoor_location) == _normalized_location(candidate.value)
                for candidate in map_centers
            )
        )
        local_sources = tuple(
            source
            for source in emerald_healing_sources()
            if _normalized_location(source.outdoor_location)[0] == _map_id_value(source_map)
            or _map_id_value(source.interior_map) == _map_id_value(source_map)
        )
        # A map absent from the legacy center table still gets its local
        # catalog source first, then the bounded global catalog fallback.
        sources = local_sources or mapped_sources or emerald_healing_sources()
    else:
        sources = ()

    if not sources:
        raise center_lookup_error or BotModeError("no cataloged healing source is available")

    overworld = perceive_overworld()
    if isinstance(overworld, OverworldObservationResult):
        raise PathFindingError("overworld observation unavailable")
    world = NavigationWorld.from_overworld(overworld)
    graph = get_world_map_graph()
    start = (_map_id_value(location[0]), location[1])

    def estimated_map_cost(source):
        if source is None:
            return 0
        try:
            # Rank by estimated tile work, including the entrance-to-exit
            # traversal on every intermediate map. Exact dynamic routing is
            # still performed in the background worker.
            estimate = graph.estimate_location_cost(start, _normalized_location(source.outdoor_location))
            return float("inf") if estimate is None else estimate
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return float("inf")

    ordered_sources = tuple(
        sorted(
            enumerate(sources),
            key=lambda item: (
                estimated_map_cost(item[1]),
                item[0],
            ),
        )
    )
    limit = _RECOVERY_CATALOG_CANDIDATE_LIMIT if candidate_limit is None else max(1, candidate_limit)
    healing_warps = tuple(
        (index, warp)
        for index, source in ordered_sources[:limit]
        if (warp := _cataloged_healing_warp(source.outdoor_location, source)) is not None
    )
    static_tile_catalog: list[tuple[tuple[int, int], tuple[tuple[object, object], ...]]] = []
    static_transition_catalog: list[tuple[tuple[int, int], tuple[WorldTransition, ...]]] = []
    route_maps: set[tuple[int, int]] = {_map_id_value(start[0])}
    candidate_warps = dict(healing_warps)
    for source_index, source in ordered_sources[:limit]:
        source_location = _normalized_location(getattr(source, "outdoor_location", None))
        if not isinstance(source_location, tuple) or len(source_location) != 2:
            continue
        try:
            # Include the candidate's outdoor map. The exact worker search
            # must walk from the map-connection arrival tile to the Center
            # door; excluding the route's final map leaves that local model
            # empty and later surfaces as a misleading coordinate KeyError.
            coarse_route = graph.route(start[0], source_location[0])
            route_maps.update(_map_id_value(map_id) for map_id in getattr(coarse_route, "maps", ()) or ())
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            # The exact worker search will report an unavailable candidate;
            # do not make preparation fail merely because the coarse route is
            # temporarily incomplete.
            continue
        interior_map = getattr(source, "interior_map", None)
        if interior_map is not None:
            route_maps.add(_map_id_value(interior_map))
        warp = candidate_warps.get(source_index)
        if warp is not None and warp.destination is not None:
            route_maps.add(_map_id_value(warp.destination[0]))

    # ROM metadata and navigation caches are owned by the frame thread. Freeze
    # every map model needed by the bounded recovery candidates before handing
    # the request to the background planner. Static models do not depend on
    # the player's coordinate or dynamic object occupancy, so reuse the frozen
    # catalog while the active ROM and route-map set are unchanged. This keeps
    # a damaged party from rebuilding every large intermediate map on each
    # walking frame.
    rom = getattr(context, "rom", None)
    static_catalog_key = (
        id(rom),
        getattr(rom, "id", None),
        tuple(sorted(route_maps, key=repr)),
    )
    cached_catalog = _recovery_static_catalog_cache.get(static_catalog_key)
    if cached_catalog is not None:
        static_tile_catalog = list(cached_catalog[0])
        static_transition_catalog = list(cached_catalog[1])
        trace = getattr(context, "stutter_trace", None)
        if trace is not None:
            trace.mark("recovery_static_catalog_cache_hit", True)
    else:
        for map_id in static_catalog_key[2]:
            try:
                static_tiles = prewarm_static_map_observation(map_id)
                tile_snapshot = static_navigation_tile_snapshot(map_id, static_tiles)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            static_tile_catalog.append((map_id, tile_snapshot))
            try:
                static_transition_catalog.append((map_id, static_map_transitions(map_id)))
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # Graph transitions remain available as a coarse fallback when
                # a particular map's ROM transition table is not coherent yet.
                continue
        if len(_recovery_static_catalog_cache) >= _RECOVERY_STATIC_CATALOG_CACHE_LIMIT:
            _recovery_static_catalog_cache.pop(next(iter(_recovery_static_catalog_cache)))
        _recovery_static_catalog_cache[static_catalog_key] = (
            tuple(static_tile_catalog),
            tuple(static_transition_catalog),
        )

    frozen_world = (
        replace(
            world,
            static_tile_catalog=tuple(static_tile_catalog),
            static_transition_catalog=tuple(static_transition_catalog),
            static_metadata_frozen=True,
        )
        if isinstance(world, NavigationWorld)
        else world
    )
    return RouteRecoveryRequest(
        location=location,
        world=frozen_world,
        graph=graph,
        ordered_sources=ordered_sources[:limit],
        center_location=center_location,
        candidate_limit=limit,
        critical=critical,
        healing_warps=healing_warps,
        static_tile_catalog=tuple(static_tile_catalog),
        static_transition_catalog=tuple(static_transition_catalog),
    )


def plan_route_recovery(request: RouteRecoveryRequest, *, cancel_check=None) -> RouteRecovery:
    """Validate the prepared healing candidates using only frozen inputs."""

    route_candidates = []
    route_errors = []
    candidate_diagnostics: list[RecoveryCandidateDiagnostic] = []
    incumbent_prefix = None
    start = (_map_id_value(request.location[0]), request.location[1])
    world = request.world
    if isinstance(world, NavigationWorld):
        world = replace(
            world,
            static_tile_catalog=request.static_tile_catalog,
            static_transition_catalog=request.static_transition_catalog,
            static_metadata_frozen=True,
        )
    healing_warps = dict(request.healing_warps)

    def select_route_recovery(*, calculation_timed_out: bool = False) -> RouteRecovery | None:
        """Return the best completed candidate, including a partial search result."""

        if not route_candidates:
            return None
        incumbent, _, source, destination, navigation_goal, route = min(
            route_candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        distance = incumbent[1] if request.critical else incumbent[0]
        encounters = getattr(route.metrics, "encounter_opportunities", 0)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_CATALOG_SELECTION: "
                f"source={request.location!r} healing_source={getattr(source, 'source_id', None)!r} "
                f"destination={destination!r} distance={distance!r} "
                f"calculation_timed_out={calculation_timed_out!r} "
                f"candidates={[(getattr(item[2], 'source_id', None), item[0]) for item in route_candidates]!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_CATALOG_SELECTION",
        )
        return RouteRecovery(
            center_available=True,
            center_location=destination,
            route_origin=request.location,
            distance_to_center=distance,
            safe_to_reach_center=True,
            healing_source_available=True,
            route=route,
            navigation_goal=navigation_goal,
            healing_source=source,
            encounter_opportunities=encounters,
            calculation_timed_out=calculation_timed_out,
            candidate_diagnostics=tuple(candidate_diagnostics),
        )

    for index, source in request.ordered_sources:
        destination = request.center_location if source is None else source.outdoor_location
        navigation_goal = _recovery_navigation_goal(
            destination,
            source,
            healing_warps.get(index),
            critical=request.critical,
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_CATALOG_CANDIDATE: "
                f"source={request.location!r} healing_source={getattr(source, 'source_id', None)!r} "
                f"destination={destination!r} rank={index!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_CATALOG_CANDIDATE",
        )
        try:
            route_reason = "executable trainer-safe route"
            try:
                plan, _ = plan_with_world_navigation(
                    world,
                    start,
                    navigation_goal,
                    request.graph,
                    max_expansions=_RECOVERY_SEARCH_MAX_EXPANSIONS,
                    max_route_cost=_RECOVERY_SEARCH_MAX_ROUTE_COST,
                    cost_ceiling=(
                        incumbent_prefix
                        if request.critical
                        else (incumbent_prefix[0] if incumbent_prefix is not None else None)
                    ),
                    cancel_check=cancel_check,
                )
            except NavigationSearchLimitExceeded as strict_error:
                if not request.critical:
                    raise
                # Encounter avoidance is lexicographic in the normal planner.
                # On a large route, that can spend the entire bounded search
                # proving that every encounter-free branch is worse before it
                # ever tests the Center exit. Critical recovery must still
                # prefer the strict route, but it needs a finite way to reach
                # the Center when a small amount of encounter exposure is the
                # only available path.
                emergency_goal = replace(
                    navigation_goal,
                    encounter_mode=EncounterMode.MOSTLY_AVOID,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_EMERGENCY_SEARCH: "
                        f"source={getattr(source, 'source_id', None)!r} "
                        f"strict_error={str(strict_error)!r} "
                        f"encounter_mode={emergency_goal.encounter_mode.name!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_EMERGENCY_SEARCH",
                )
                try:
                    plan, _ = plan_with_world_navigation(
                        world,
                        start,
                        emergency_goal,
                        request.graph,
                        max_expansions=_RECOVERY_EMERGENCY_SEARCH_MAX_EXPANSIONS,
                        max_route_cost=_RECOVERY_EMERGENCY_SEARCH_MAX_ROUTE_COST,
                        cost_ceiling=None,
                        cancel_check=cancel_check,
                    )
                except (
                    BotModeError,
                    NavigationError,
                    PathFindingError,
                    TypeError,
                    ValueError,
                ) as emergency_error:
                    route_errors.append(
                        (
                            index,
                            getattr(source, "source_id", None),
                            type(emergency_error).__name__,
                            f"strict search exhausted; emergency search failed: {emergency_error}",
                        )
                    )
                    candidate_diagnostics.append(
                        RecoveryCandidateDiagnostic(
                            rank=index,
                            source_id=getattr(source, "source_id", None),
                            destination=destination,
                            outcome="failed",
                            reason=(
                                f"strict search exhausted; emergency search failed: "
                                f"{type(emergency_error).__name__}: {emergency_error}"
                            ),
                        )
                    )
                    continue
                navigation_goal = emergency_goal
                route_reason = "executable trainer-safe route after encounter-risk fallback"
            if getattr(plan, "forced_trainer_exposure", False):
                raise NavigationError("world recovery route requires trainer exposure")
            if plan.metrics is None or plan.destination is None:
                raise PathFindingError("world recovery route has no executable metrics")
            distance = plan.metrics.total_route_cost
            if distance is None:
                raise PathFindingError("world recovery route has no cost")
            # Critical recovery retains the safety ordering (encounter
            # exposure first, then route cost). Opportunistic recovery uses
            # the weighted route cost produced by MOSTLY_AVOID, then uses
            # exposure as a deterministic tie-breaker.
            encounters = getattr(plan.metrics, "encounter_opportunities", 0)
            candidate_prefix = (
                (encounters, distance)
                if request.critical
                else (distance, encounters)
            )
            route_candidates.append((candidate_prefix, index, source, destination, navigation_goal, plan))
            candidate_diagnostics.append(
                RecoveryCandidateDiagnostic(
                    rank=index,
                    source_id=getattr(source, "source_id", None),
                    destination=destination,
                    outcome="success",
                    distance=distance,
                    encounter_opportunities=encounters,
                    reason=route_reason,
                )
            )
            if request.critical and candidate_prefix[0] == 0:
                # A zero-exposure route is a complete safety result: no later
                # candidate can improve its primary recovery criterion, and
                # searching a distant fallback here was the source of long
                # Route 104 stalls. Exposed critical routes continue through
                # the bounded candidate set so encounter count can improve.
                return select_route_recovery()
            if incumbent_prefix is None or candidate_prefix < incumbent_prefix:
                incumbent_prefix = candidate_prefix
        except TimeoutError as error:
            route_errors.append((index, getattr(source, "source_id", None), "TimeoutError", str(error)))
            candidate_diagnostics.append(
                RecoveryCandidateDiagnostic(
                    rank=index,
                    source_id=getattr(source, "source_id", None),
                    destination=destination,
                    outcome="timed_out",
                    reason=f"TimeoutError: {error}",
                )
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_CANDIDATE_TIMEOUT: "
                    f"source={getattr(source, 'source_id', None)!r} rank={index!r} "
                    f"incumbent_available={bool(route_candidates)!r} reason={str(error)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_CANDIDATE_TIMEOUT",
            )
            partial = select_route_recovery(calculation_timed_out=True)
            if partial is not None:
                return partial
            raise
        except NavigationSearchLimitExceeded as error:
            route_errors.append((index, getattr(source, "source_id", None), "NavigationSearchLimitExceeded", str(error)))
            candidate_diagnostics.append(
                RecoveryCandidateDiagnostic(
                    rank=index,
                    source_id=getattr(source, "source_id", None),
                    destination=destination,
                    outcome="pruned",
                    reason=f"NavigationSearchLimitExceeded: {error}",
                )
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_CANDIDATE_PRUNED: "
                    f"source={getattr(source, 'source_id', None)!r} rank={index!r} "
                    f"incumbent={incumbent_prefix!r} reason={str(error)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_CANDIDATE_PRUNED",
            )
        except (BotModeError, NavigationError, PathFindingError, TypeError, ValueError) as error:
            route_errors.append((index, getattr(source, "source_id", None), type(error).__name__, str(error)))
            candidate_diagnostics.append(
                RecoveryCandidateDiagnostic(
                    rank=index,
                    source_id=getattr(source, "source_id", None),
                    destination=destination,
                    outcome="failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )

    if not route_candidates:
        raise RecoveryPlanningError(
            f"no safe cataloged healing route: {route_errors!r}",
            candidate_diagnostics,
        )
    selected = select_route_recovery()
    if selected is None:
        raise RecoveryPlanningError(
            f"no safe cataloged healing route: {route_errors!r}",
            candidate_diagnostics,
        )
    return selected


def observe_route_recovery(*, candidate_limit: int | None = None) -> RouteRecovery:
    """Observe and synchronously validate recovery routing for compatibility callers."""

    trace = getattr(context, "stutter_trace", None)
    started = trace.now() if trace is not None else 0
    try:
        prepared = prepare_route_recovery(candidate_limit=candidate_limit)
        if isinstance(prepared, RouteRecovery):
            return prepared
        return plan_route_recovery(prepared)
    except KeyError as error:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_OBSERVATION_RETRY: "
                "stage='route' exception_type='KeyError' "
                f"key={error.args!r} traceback={traceback.format_exc()!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_OBSERVATION_RETRY",
        )
        return RouteRecovery(observation_available=False, observation_error=f"KeyError: {error}")
    except (BotModeError, NavigationError, PathFindingError) as error:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_ROUTE_FAILURE: "
                f"error_type={type(error).__name__!r} error={str(error)!r} route_available=False"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_ROUTE_FAILURE",
        )
        # A valid world observation with no safe route is known to be
        # unavailable, not a usable recovery capability. The existence of a
        # catalog entry must not manufacture a later RECOVER decision that
        # CampaignPlan cannot execute.
        return RouteRecovery(
            observation_available=True,
            healing_source_available=False,
            observation_error=str(error),
            candidate_diagnostics=tuple(getattr(error, "candidate_diagnostics", ()) or ()),
        )
    finally:
        if trace is not None:
            trace.duration("campaign_route_recovery_observation_duration_ms", started)


def recover_at_nearest_center(current_location=None, selected_center=None) -> Iterator[object]:
    """Compatibility entry point that delegates to planned recovery.

    Older callers may still provide only a Center enum.  Convert that value
    into the Emerald healing catalog, then use the same destination, warp,
    source discovery, and dialogue executor as campaign-planned recovery.
    """
    center = selected_center or find_closest_pokemon_center(current_location or get_player_location())
    destination = getattr(center, "value", center)
    source = emerald_healing_source_for_destination(destination)
    if source is None:
        raise RuntimeError(f"No cataloged healing source is available for {destination!r}")
    yield from execute_planned_recovery(destination, source)


def _wait_for_center_interior(center) -> Iterator[object]:
    """Wait for the selected Center entry warp to produce a stable interior."""
    outdoor_map = center.value[0]
    outdoor_map_id = _map_id_value(outdoor_map)
    interior_name = (
        "PALLET_TOWN_PLAYERS_HOUSE_1F" if center.name == "PalletTown" else f"{outdoor_map.name}_POKEMON_CENTER_1F"
    )
    interior_map = getattr(MapFRLG if isinstance(outdoor_map, MapFRLG) else type(outdoor_map), interior_name, None)
    if interior_map is None:
        raise RuntimeError(f"No Pokémon Center interior map is defined for {outdoor_map!r}")
    interior_map_id = _map_id_value(interior_map)
    _prewarm_interior_map_identity(interior_map_id)
    interior_observed = False
    last_position = None
    settled_observations = 0
    unresolved_map_observations = 0
    while True:
        # A healing-source entrance has the same map-transition contract as a
        # Pokémon Center entrance; do not gate its map observation on UI state.
        observation = observe_agent(require_overworld=True)
        overworld = observation.overworld
        observed_map = getattr(overworld, "map_id", None)
        map_source = getattr(overworld, "map_identity_source", "unavailable")
        save_block_map = getattr(overworld, "save_block_map_id", None)
        live_candidates = getattr(overworld, "live_map_candidates", ())
        controllable = getattr(overworld, "controllable", False)
        position = (observed_map, getattr(overworld, "player_coordinates", None))
        if position == last_position:
            settled_observations += 1
        else:
            last_position = position
            settled_observations = 0
        interior_confirmed = observed_map == interior_map_id and map_source == "live_header"
        if interior_confirmed and not interior_observed:
            # Publish the map transition independently of the later control
            # handoff.  During a warp the map can be authoritative before the
            # avatar is actionable; hiding this phase made a stalled entry
            # look like the outdoor route was still running.
            context.campaign_status = recovery_status(
                SemanticTarget.map(interior_map_id),
                "Entering Pokémon Center",
            )
            interior_observed = True
        if not interior_confirmed:
            context.campaign_status = recovery_status(
                SemanticTarget.at(center.value),
                f"Waiting for Pokémon Center map (map={observed_map!r}, source={map_source}, "
                f"save={save_block_map!r}, candidates={live_candidates!r}, "
                f"position={getattr(overworld, 'player_coordinates', None)!r})",
            )
        elif not controllable:
            context.campaign_status = recovery_status(
                SemanticTarget.map(interior_map_id),
                "Waiting for avatar control",
            )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_HANDOFF: "
                f"observed_map={observed_map!r} expected_interior={interior_map_id!r} "
                f"source={map_source!r} controllable={controllable!r} "
                f"proceeding={interior_confirmed and controllable}"
            ),
            trace=True,
        )
        if interior_confirmed and controllable:
            return
        if interior_confirmed:
            unresolved_map_observations = 0
        else:
            unresolved_map_observations += 1
            if unresolved_map_observations >= _MAP_IDENTITY_RESOLUTION_TIMEOUT:
                raise RuntimeError(
                    "Pokémon Center map identity did not resolve after entry: "
                    f"expected={interior_map_id!r} observed={observed_map!r} source={map_source!r} "
                    f"save_block={save_block_map!r} candidates={live_candidates!r}"
                )
        # ReachLocation(center.value) terminates on the outdoor door tile.
        # Issue the one required entry step only while the *live* header says
        # that the avatar is still outside.  Once it has crossed the door,
        # this loop never starts another exterior route from a save-block
        # fallback: it only waits for the authoritative interior observation.
        if (
            overworld is not None
            and observed_map == outdoor_map_id
            and map_source == "live_header"
            and controllable
            and settled_observations == 0
        ):
            adjacent_to_entry = (
                isinstance(position[1], tuple)
                and len(position[1]) == 2
                and position[1] != center.value[1]
                and abs(position[1][0] - center.value[1][0]) + abs(position[1][1] - center.value[1][1]) == 1
            )
            if adjacent_to_entry:
                _pulse_toward_entry(observation, center.value)
        yield


def _wait_for_healing_interior(source) -> Iterator[object]:
    """Own the entrance warp for a non-Center catalog healing source."""
    interior_observed = False
    last_position = None
    settled_observations = 0
    unresolved_map_observations = 0
    interior_map_id = _map_id_value(source.interior_map)
    _prewarm_interior_map_identity(interior_map_id)
    while True:
        observation = observe_agent(require_overworld=True)
        overworld = observation.overworld
        observed_map = getattr(overworld, "map_id", None)
        map_source = getattr(overworld, "map_identity_source", "unavailable")
        save_block_map = getattr(overworld, "save_block_map_id", None)
        live_candidates = getattr(overworld, "live_map_candidates", ())
        position = (observed_map, getattr(overworld, "player_coordinates", None))
        if position == last_position:
            settled_observations += 1
        else:
            last_position = position
            settled_observations = 0
        interior_confirmed = observed_map == interior_map_id and map_source == "live_header"
        if interior_confirmed and not interior_observed:
            context.campaign_status = recovery_status(
                SemanticTarget.map(interior_map_id),
                "Entering healing location",
            )
            interior_observed = True
        if not interior_confirmed:
            context.campaign_status = recovery_status(
                SemanticTarget.at(source.outdoor_location),
                f"Waiting for healing-location map (map={observed_map!r}, source={map_source}, "
                f"save={save_block_map!r}, candidates={live_candidates!r}, "
                f"position={getattr(overworld, 'player_coordinates', None)!r})",
            )
        elif not getattr(overworld, "controllable", False):
            context.campaign_status = recovery_status(
                SemanticTarget.map(interior_map_id),
                "Waiting for avatar control",
            )
        if interior_confirmed and getattr(overworld, "controllable", False):
            return
        if interior_confirmed:
            unresolved_map_observations = 0
        else:
            unresolved_map_observations += 1
            if unresolved_map_observations >= _MAP_IDENTITY_RESOLUTION_TIMEOUT:
                raise RuntimeError(
                    "Healing-location map identity did not resolve after entry: "
                    f"expected={interior_map_id!r} observed={observed_map!r} source={map_source!r} "
                    f"save_block={save_block_map!r} candidates={live_candidates!r}"
                )
        yield


def _select_bag_recovery_item(snapshot: ResourceSnapshot) -> HealingResource | None:
    """Select a medicine that can improve the observed party.

    A Potion is not a recovery action when every living member is at full HP.
    Returning it in that state causes callers to open the bag and attempt a
    no-op use, which is especially harmful when no safe Center route exists.
    Poison is the one exception: an HP item can provide the bridge needed to
    reach a Center before overworld poison drains the affected member.
    """

    for member in snapshot.status_party:
        candidates = [
            item
            for item in snapshot.bag_status_items
            if item.cures(member.status)
        ]
        if candidates:
            return min(candidates, key=lambda item: (item.cures_all_statuses, item.reserve_tier, item.name))
    # A one-point deficit above the safety floor is not worth consuming a
    # medicine. This also prevents a Max Potion or Full Restore from being
    # selected just because it is the only HP-capable item in the bag.
    if snapshot.total_missing_hp <= 1 and not snapshot.has_poisoned_party:
        return None
    return best_hp_recovery_item(snapshot, minimum_hp_ratio=0.5)


def _recovery_target_index(item: HealingResource):
    """Choose a live party slot for a field medicine."""

    party = get_party()
    if item.has_status_effect:
        for pokemon in party:
            if pokemon.current_hp > 0 and item.cures(pokemon.status_condition.value):
                return pokemon.index
    candidates = [pokemon for pokemon in party if not pokemon.is_egg and pokemon.current_hp > 0]
    if not candidates:
        raise RuntimeError("No living party member can receive recovery item")
    return min(candidates, key=lambda pokemon: pokemon.current_hp / max(1, pokemon.total_hp)).index


_FIELD_RECOVERY_MENU_TIMEOUT = 180


def _press_field_recovery_button_fresh(button: str) -> None:
    """Send a new input edge at a field-medicine menu boundary."""

    press_button_fresh = getattr(context.emulator, "press_button_fresh", None)
    if callable(press_button_fresh):
        press_button_fresh(button)
    else:
        context.emulator.press_button(button)


def _close_field_recovery_menus(item_name: str) -> Iterator[object]:
    """Close the Bag and the Start menu reopened by Emerald's Bag callback.

    After a medicine target is confirmed, Emerald returns to the Bag and then
    re-enters the field with the Start menu open.  The two callbacks can be
    separated by several frames, and a normal repeated ``press_button`` does
    not create a new JOY_NEW edge after the first B press.  Wait for each ROM
    input task and use fresh B pulses at both boundaries.
    """

    if getattr(getattr(context, "rom", None), "is_rs", False):
        bag_input_task = "sub_80A50C8"
        start_menu_task = "sub_80712B4"
    elif getattr(getattr(context, "rom", None), "is_frlg", False):
        bag_input_task = "Task_BagMenu_HandleInput"
        start_menu_task = "Task_StartMenuHandleInput"
    else:
        bag_input_task = "Task_BagMenu_HandleInput"
        start_menu_task = "Task_ShowStartMenu"

    phase = "close_bag"
    last_phase = None
    for elapsed in range(_FIELD_RECOVERY_MENU_TIMEOUT):
        state = get_game_state()
        if phase != last_phase:
            diagnostic_print(
                lambda: (
                    "FIELD_RECOVERY_MENUS: "
                    f"item={item_name!r} phase={phase!r} "
                    f"game_state={state!r}"
                ),
                trace=True,
            )
            last_phase = phase

        if phase == "close_bag":
            if state is GameState.BAG_MENU:
                if task_is_active(bag_input_task):
                    _press_field_recovery_button_fresh("B")
                yield
                continue
            if state is GameState.OVERWORLD:
                phase = "wait_start_menu"
            else:
                yield
                continue

        if state is not GameState.OVERWORLD:
            # This should only be a transient callback boundary, but returning
            # to the Bag is safe to handle if the ROM exposes it again.
            phase = "close_bag"
            yield
            continue

        start_menu_open = task_is_active(start_menu_task)
        if not start_menu_open:
            try:
                start_menu_open = bool(parse_start_menu().get("open", False))
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # The task is the authoritative signal.  A transient parser
                # failure during the callback must not consume the watchdog.
                start_menu_open = False
        if phase == "wait_start_menu":
            if start_menu_open:
                phase = "close_start_menu"
            yield
            continue

        if not start_menu_open:
            diagnostic_print(
                lambda: (
                    "FIELD_RECOVERY_MENUS: "
                    f"item={item_name!r} completed=True elapsed_frames={elapsed!r}"
                ),
                trace=True,
            )
            return
        _press_field_recovery_button_fresh("B")
        yield

    diagnostic_print(
        lambda: (
            "FIELD_RECOVERY_MENUS: "
            f"item={item_name!r} completed=False phase={phase!r} "
            f"game_state={get_game_state()!r}"
        ),
        trace=True,
    )
    raise RuntimeError(f"Recovery item {item_name!r} did not close its menus")


def _use_field_recovery_item(item: HealingResource) -> Iterator[object]:
    """Use one medicine and drive the ROM's target-party menu to completion."""

    target_index = _recovery_target_index(item)
    before = observe_resource_snapshot()
    before_quantity = next(
        (candidate.quantity for candidate in before.bag_healing_items if candidate.name == item.name),
        0,
    )
    before_target = before.party[target_index] if target_index < len(before.party) else None
    yield from use_item_from_bag(get_item_by_name(item.name), wait_for_start_menu_to_reappear=False)

    # The item context menu ends before Emerald opens the party target menu.
    # Wait for that authoritative state instead of pressing a fixed sequence of
    # buttons across both menu transitions.
    for _ in range(180):
        if get_game_state() is GameState.PARTY_MENU:
            break
        yield
    if get_game_state() is not GameState.PARTY_MENU:
        raise RuntimeError(f"Recovery item {item.name!r} did not open the party menu")
    yield from scroll_to_party_menu_index(target_index)
    _press_field_recovery_button_fresh("A")
    yield
    for _ in range(_FIELD_RECOVERY_MENU_TIMEOUT):
        if get_game_state() is not GameState.PARTY_MENU:
            break
        _press_field_recovery_button_fresh("A")
        yield
    if get_game_state() is GameState.PARTY_MENU:
        raise RuntimeError(f"Recovery item {item.name!r} did not finish party target selection")

    # The item use returns to the Bag. Close that menu and the reopened Start
    # menu through their ROM task boundaries before handing control back to
    # campaign navigation.
    yield from _close_field_recovery_menus(item.name)

    # Verify consumption and an actual party improvement. A stale cached frame
    # is tolerated briefly, but a no-op medicine is never silently accepted.
    for _ in range(12):
        after = observe_resource_snapshot()
        after_quantity = next(
            (candidate.quantity for candidate in after.bag_healing_items if candidate.name == item.name),
            0,
        )
        after_target = after.party[target_index] if target_index < len(after.party) else None
        improved = (
            before_target is not None
            and after_target is not None
            and (
                after_target.current_hp > before_target.current_hp
                or (after_target.status or "none") != (before_target.status or "none")
            )
        )
        if after_quantity < before_quantity and improved:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_ITEM: "
                    f"item={item.name!r} target_index={target_index!r} "
                    f"quantity={before_quantity}->{after_quantity} "
                    f"hp={getattr(before_target, 'current_hp', None)!r}->{getattr(after_target, 'current_hp', None)!r} "
                    f"status={getattr(before_target, 'status', None)!r}->{getattr(after_target, 'status', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ITEM",
            )
            return
        yield
    raise RuntimeError(f"Recovery item {item.name!r} was not observed as consumed")


def use_best_bag_healing_item() -> Iterator[object]:
    """Use the most urgent useful medicine from the bag."""

    snapshot = observe_resource_snapshot()
    item = _select_bag_recovery_item(snapshot)
    if item is None:
        raise RuntimeError("No usable recovery item is available in the bag")
    yield from _use_field_recovery_item(item)


def withdraw_best_pc_healing_item() -> Iterator[object]:
    """Withdraw the most useful stored recovery item at a reached PC."""
    context.campaign_recovery_pc_status = "planning"
    snapshot = observe_resource_snapshot()
    status_candidates = [
        candidate
        for member in snapshot.status_party
        for candidate in snapshot.pc_status_items
        if candidate.cures(member.status)
    ]
    item = min(
        status_candidates,
        key=lambda candidate: (candidate.cures_all_statuses, candidate.reserve_tier, candidate.name),
        default=None,
    )
    if item is None:
        item = best_hp_recovery_item(
            ResourceSnapshot(
                party=snapshot.party,
                bag_healing_items=snapshot.pc_healing_items,
            )
        )
    if item is None:
        # The earliest convenient PC boundary can occur before the starter is
        # in the party. Still collect a stored medicine so it is available as
        # soon as the first battle route begins.
        item = min(
            snapshot.pc_healing_items,
            key=lambda candidate: (
                0 if candidate.has_hp_effect else 1,
                candidate.reserve_tier,
                candidate.name,
            ),
            default=None,
        )
    if item is None:
        context.campaign_recovery_pc_status = "no_stock"
        diagnostic_print(
            lambda: "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='no_stock'",
            trace=False,
        )
        return
    item_object = get_item_by_name(item.name)
    before = snapshot
    before_bag_quantity = next(
        (candidate.quantity for candidate in before.bag_healing_items if candidate.name == item.name),
        0,
    )
    before_pc_quantity = next(
        (candidate.quantity for candidate in before.pc_healing_items if candidate.name == item.name),
        0,
    )
    context.message = f"Withdrawing {item.name} from the item PC"
    diagnostic_print(
        lambda: (
            "CAMPAIGN_HEALING_SUPPLY: action='withdraw' "
            f"item={item.name!r} bag_before={before_bag_quantity!r} pc_before={before_pc_quantity!r}"
        ),
        trace=False,
    )
    try:
        yield from interact_with_pc([PCAction.withdraw_item(item_object, 1)])
    except Exception:
        context.campaign_recovery_pc_status = "withdrawal_failed"
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='withdrawal_failed' "
                f"item={item.name!r} bag_before={before_bag_quantity!r} pc_before={before_pc_quantity!r}"
            ),
            trace=False,
        )
        raise

    for _ in range(12):
        after = observe_resource_snapshot()
        after_bag_quantity = next(
            (candidate.quantity for candidate in after.bag_healing_items if candidate.name == item.name),
            0,
        )
        after_pc_quantity = next(
            (candidate.quantity for candidate in after.pc_healing_items if candidate.name == item.name),
            0,
        )
        if after_bag_quantity > before_bag_quantity and after_pc_quantity < before_pc_quantity:
            context.campaign_recovery_pc_status = "withdrawn"
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='withdrawn' "
                    f"item={item.name!r} quantity=1 bag_before={before_bag_quantity!r} "
                    f"bag_after={after_bag_quantity!r} pc_before={before_pc_quantity!r} "
                    f"pc_after={after_pc_quantity!r}"
                ),
                trace=False,
            )
            return
        yield
    context.campaign_recovery_pc_status = "withdrawal_failed"
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='withdrawal_failed' "
            f"item={item.name!r} bag_before={before_bag_quantity!r} pc_before={before_pc_quantity!r}"
        ),
        trace=False,
    )
    raise RuntimeError(f"PC withdrawal of {item.name!r} was not observed")


def _is_pc_trigger(trigger) -> bool:
    """Return whether an observed object trigger is a player PC."""

    if getattr(trigger, "kind", None) != "object_interaction":
        return False
    identity = (
        getattr(trigger, "affordance_id", None)
        or getattr(trigger, "script_symbol", None)
        or ""
    )
    return identity == "EventScript_PC" or identity.endswith("_EventScript_PC")


def _observed_pc_trigger(observation):
    overworld = getattr(observation, "overworld", None)
    if overworld is None:
        return None
    return next(
        (trigger for trigger in getattr(overworld, "triggers", ()) if _is_pc_trigger(trigger)),
        None,
    )


def _known_center_pc_approach(observation):
    """Return verified geometry for a Center whose PC trigger was omitted.

    The catalog is the fallback when static map reads are unavailable. When
    the map is readable, derive the approach from the ROM's actual ``PC``
    metatile so a stale catalog coordinate cannot send cleanup to an unrelated
    tile.
    """

    map_id = _map_id_value(getattr(getattr(observation, "overworld", None), "map_id", None))
    catalog_approach = None
    for source in emerald_healing_sources():
        approach = getattr(source, "center_pc_approach", None)
        if approach is not None and approach[0][0] == map_id:
            catalog_approach = approach
            break
    try:
        map_view = get_map_data(map_id, (0, 0))
        tiles = {tile.local_position: tile for tile in get_map_all_tiles(map_view)}
        direction_vectors = {
            Direction.North: (0, -1),
            Direction.East: (1, 0),
            Direction.South: (0, 1),
            Direction.West: (-1, 0),
        }
        static_approaches = []
        for pc_tile in tiles.values():
            if getattr(pc_tile, "tile_type", None) != "PC":
                continue
            pc_x, pc_y = pc_tile.local_position
            for facing, (dx, dy) in direction_vectors.items():
                approach_coordinates = (pc_x - dx, pc_y - dy)
                approach_tile = tiles.get(approach_coordinates)
                if approach_tile is None or getattr(approach_tile, "collision", 1) != 0:
                    continue
                static_approaches.append(((map_id, approach_coordinates), facing))

        if static_approaches:
            if catalog_approach is not None and catalog_approach in static_approaches:
                return catalog_approach
            current_coordinates = getattr(getattr(observation, "overworld", None), "player_coordinates", None)
            if isinstance(current_coordinates, tuple) and len(current_coordinates) == 2:
                return min(
                    static_approaches,
                    key=lambda candidate: (
                        abs(candidate[0][1][0] - current_coordinates[0])
                        + abs(candidate[0][1][1] - current_coordinates[1]),
                        candidate[0][1],
                        candidate[1].value,
                    ),
                )
            return static_approaches[0]
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        # The catalog remains an intentional fallback for a transient ROM
        # read failure. The live-avatar validation below still guards the
        # interaction itself.
        pass
    return catalog_approach


def _center_pc_avatar_state(target, required_facing):
    """Return the live avatar state needed by ``interact_with_pc``."""

    try:
        avatar = get_player_avatar()
        front = avatar.map_location_in_front
        return {
            "location": (avatar.map_group_and_number, avatar.local_coordinates),
            "facing": avatar.facing_direction,
            "front_type": getattr(front, "tile_type", None),
            "ready": (
                avatar.map_group_and_number == target[0]
                and avatar.local_coordinates == target[1]
                and avatar.facing_direction == required_facing.button_name
                and getattr(front, "tile_type", None) == "PC"
            ),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return {
            "location": None,
            "facing": None,
            "front_type": None,
            "ready": False,
        }


def _navigate_to_known_center_pc(observation, *, purpose: str) -> Iterator[object]:
    """Use Center geometry only after the live PC trigger was absent."""

    approach = _known_center_pc_approach(observation)
    if approach is None:
        return False
    target, required_facing = approach
    context.campaign_recovery_pc_target = target
    current = (
        getattr(observation.overworld, "map_id", None),
        getattr(observation.overworld, "player_coordinates", None),
    )
    navigation_goal = NavigationGoal(
        ReachLocation(target),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_PC_FALLBACK: "
            f"purpose={purpose!r} observed_trigger=False target={target!r} "
            f"current={current!r} facing={required_facing.name!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_PC_FALLBACK",
    )
    context.campaign_status = recovery_status(
        SemanticTarget.interaction(target[0], "EventScript_PC"),
        "Navigate to verified Center PC fallback",
    )
    yield from AgentControlLoop(
        lambda: observe_agent(goal=navigation_goal),
        goal=navigation_goal,
        logger=_recovery_agent_logger,
        use_movement_batch=False,
    ).run()
    settled = observe_agent(require_overworld=True)
    settled_location = (
        getattr(settled.overworld, "map_id", None),
        getattr(settled.overworld, "player_coordinates", None),
    )
    if _normalized_location(settled_location) != target:
        raise RuntimeError(
            f"verified Center PC fallback did not reach approach tile: expected {target!r}, observed {settled_location!r}"
        )
    if getattr(settled.overworld, "facing", None) is not required_facing:
        press_direction = getattr(context.emulator, "press_direction", None)
        if callable(press_direction):
            press_direction(required_facing.button_name, run=False, fresh=True)
        else:
            context.emulator.press_button(required_facing.button_name)
        yield
        verified = observe_agent(require_overworld=True)
        verified_location = (
            getattr(verified.overworld, "map_id", None),
            getattr(verified.overworld, "player_coordinates", None),
        )
        if _normalized_location(verified_location) != target or getattr(verified.overworld, "facing", None) is not required_facing:
            raise RuntimeError(
                "verified Center PC fallback did not settle on the required approach and facing: "
                f"expected location={target!r} facing={required_facing.name!r}, "
                f"observed location={verified_location!r} facing={getattr(verified.overworld, 'facing', None)!r}"
            )

    # The passive overworld observation and the PC helper read different ROM
    # views. Confirm the same live avatar state the helper will use before
    # entering storage; an observed coordinate/facing pair alone does not
    # prove that the tile in front of the avatar is actually a PC.
    avatar_state = _center_pc_avatar_state(target, required_facing)
    if not avatar_state["ready"]:
        if avatar_state["location"] != target:
            raise RuntimeError(
                "verified Center PC fallback avatar state is not at the approach tile: "
                f"expected location={target!r}, observed location={avatar_state['location']!r}"
            )
        press_direction = getattr(context.emulator, "press_direction", None)
        if callable(press_direction):
            press_direction(required_facing.button_name, run=False, fresh=True)
        else:
            context.emulator.press_button(required_facing.button_name)
        yield
        avatar_state = _center_pc_avatar_state(target, required_facing)
    if not avatar_state["ready"]:
        raise RuntimeError(
            "verified Center PC fallback is not executable: "
            f"expected location={target!r} facing={required_facing.name!r} front_type='PC', "
            f"observed location={avatar_state['location']!r} facing={avatar_state['facing']!r} "
            f"front_type={avatar_state['front_type']!r}"
        )
    return True


def _withdraw_pc_healing_item_at_current_location() -> Iterator[object]:
    """Reach an observed PC, withdraw one medicine, and verify the transfer."""

    snapshot = observe_resource_snapshot()
    if not snapshot.pc_healing_items:
        context.campaign_recovery_pc_status = "no_stock"
        diagnostic_print(
            lambda: "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='no_stock'",
            trace=False,
        )
        return
    if get_game_state() is GameState.POKE_STORAGE:
        # The PC is already open, so bypass overworld perception and let the
        # storage transaction own the current menu directly.
        yield from withdraw_best_pc_healing_item()
        return

    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation)
    context.campaign_recovery_observed_triggers = tuple(
        getattr(item, "trigger_id", None) or getattr(item, "affordance_id", None) or getattr(item, "script_symbol", None)
        for item in getattr(getattr(observation, "overworld", None), "triggers", ())
    )
    if trigger is None or not getattr(trigger, "activation_locations", ()):
        try:
            used_fallback = yield from _navigate_to_known_center_pc(
                observation,
                purpose="withdraw healing item",
            )
        except Exception:
            context.campaign_recovery_pc_status = "pc_unavailable"
            diagnostic_print(
                lambda: "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='pc_unavailable'",
                trace=False,
            )
            raise
        if used_fallback:
            yield from withdraw_best_pc_healing_item()
            return
        context.campaign_recovery_pc_status = "pc_unavailable"
        diagnostic_print(
            lambda: "CAMPAIGN_RECOVERY_PC_WITHDRAWAL: status='pc_unavailable'",
            trace=False,
        )
        raise RuntimeError("Item PC was unavailable at the healed Center interior")

    activation_locations = tuple(sorted(trigger.activation_locations, key=repr))
    current_location = (
        observation.overworld.map_id,
        observation.overworld.player_coordinates,
    )
    target = next(
        (location for location in activation_locations if location == current_location),
        activation_locations[0],
    )
    context.campaign_recovery_pc_target = target
    navigation_goal = NavigationGoal(
        ReachLocation(target),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )
    context.campaign_status = recovery_status(
        SemanticTarget.interaction(observation.overworld.map_id, trigger.trigger_id),
        "Navigate to item PC",
    )
    yield from AgentControlLoop(
        lambda: observe_agent(goal=navigation_goal),
        goal=navigation_goal,
        logger=_recovery_agent_logger,
        use_movement_batch=False,
    ).run()

    # ReachLocation deliberately stops before sending the interaction input.
    # Orient from the observed activation requirement, then let the PC helper
    # own the ROM menu transaction.
    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation) or trigger
    required_facing = dict(getattr(trigger, "activation_requirements", ())).get(
        (observation.overworld.map_id, observation.overworld.player_coordinates)
    )
    if required_facing is not None and getattr(observation.overworld, "facing", None) is not required_facing:
        press_direction = getattr(context.emulator, "press_direction", None)
        if callable(press_direction):
            press_direction(required_facing.button_name, run=False, fresh=True)
        else:
            context.emulator.press_button(required_facing.button_name)
        yield

    yield from withdraw_best_pc_healing_item()


FAINTED_POKEMON_BOX = 13


def _safe_capture_box_index(storage) -> int | None:
    """Return a non-graveyard box with room for a full-party capture."""
    boxes = tuple(getattr(storage, "boxes", ()) or ())
    active_box = getattr(storage, "active_box_index", None)
    if (
        isinstance(active_box, int)
        and active_box != FAINTED_POKEMON_BOX
        and 0 <= active_box < len(boxes)
        and boxes[active_box].first_empty_slot_index is not None
    ):
        return active_box
    for box_index, box in enumerate(boxes):
        if box_index != FAINTED_POKEMON_BOX and box.first_empty_slot_index is not None:
            return box_index
    return None


def _remembered_fainted_party_members() -> tuple[object, ...]:
    """Return campaign-dead Pokémon that are still physically in the party."""

    runtime = getattr(context, "nuzlocke_runtime", None)
    dead_identities = set(
        getattr(getattr(getattr(runtime, "rules_projection", None), "state", None), "dead_pokemon", ()) or ()
    )
    mode = getattr(context, "bot_mode_instance", None)
    dead_identities.update(getattr(mode, "_pending_fainted_identities", ()) or ())
    if not dead_identities:
        return ()
    return tuple(
        pokemon
        for pokemon in get_party()
        if not pokemon.is_egg
        and (identity := PokemonIdentity.from_pokemon(pokemon)) is not None
        and identity in dead_identities
    )


def _finalize_fainted_party_cleanup(fainted: tuple[object, ...]) -> None:
    """Verify storage and clear pending death identities after a PC action."""

    if fainted:
        remaining = _remembered_fainted_party_members()
        if remaining:
            raise RuntimeError("fainted Pokémon PC deposit was not observed")
    storage_after = get_pokemon_storage()
    if storage_after is not None and storage_after.active_box_index == FAINTED_POKEMON_BOX:
        raise RuntimeError("active capture box remains the fainted Pokémon record box")
    mode = getattr(context, "bot_mode_instance", None)
    pending = getattr(mode, "_pending_fainted_identities", None)
    identities = {
        identity
        for pokemon in fainted
        if (identity := PokemonIdentity.from_pokemon(pokemon)) is not None
    }
    if isinstance(pending, set):
        pending.difference_update(identities)
    elif pending:
        setattr(mode, "_pending_fainted_identities", frozenset(pending).difference(identities))
    diagnostic_print(
        lambda: (
            "CAMPAIGN_FAINTED_PARTY_STORED: "
            f"box={FAINTED_POKEMON_BOX + 1} count={len(fainted)} "
            f"pokemon={[getattr(pokemon, 'name', 'unknown') for pokemon in fainted]!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_FAINTED_PARTY_STORED",
    )


def _deposit_fainted_party_at_current_location() -> Iterator[object]:
    """Record deaths and leave the active capture box outside the record box.

    The Center heals the party before the PC transaction, so this uses the
    persistent Nuzlocke identities recorded by the runtime rather than the
    current HP values. The box is a record of deaths; the campaign never
    withdraws these Pokémon again. The same PC visit also selects a different
    box with room whenever the current capture box is unsafe.
    """

    fainted = _remembered_fainted_party_members()
    storage = get_pokemon_storage()
    safe_capture_box = _safe_capture_box_index(storage) if storage is not None else None
    active_box = getattr(storage, "active_box_index", None)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_FAINTED_PARTY_AUDIT: "
            f"remembered={len(fainted)!r} active_box={active_box!r} "
            f"safe_capture_box={safe_capture_box!r} storage_available={storage is not None!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_FAINTED_PARTY_AUDIT",
    )
    if not fainted and safe_capture_box == active_box:
        return
    if storage is None:
        if fainted:
            raise RuntimeError("cannot record fainted Pokémon because PC storage is unavailable")
        return
    if FAINTED_POKEMON_BOX >= len(storage.boxes):
        raise RuntimeError("cannot record fainted Pokémon because PC storage is unavailable")
    if fainted and storage.boxes[FAINTED_POKEMON_BOX].first_empty_slot_index is None:
        raise RuntimeError(f"fainted Pokémon record box {FAINTED_POKEMON_BOX + 1} is full")
    if safe_capture_box is None:
        raise RuntimeError("cannot select a safe capture box because every non-record box is full")

    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation)
    context.campaign_recovery_observed_triggers = tuple(
        getattr(item, "trigger_id", None) or getattr(item, "affordance_id", None) or getattr(item, "script_symbol", None)
        for item in getattr(getattr(observation, "overworld", None), "triggers", ())
    )
    if trigger is None or not getattr(trigger, "activation_locations", ()):
        used_fallback = yield from _navigate_to_known_center_pc(
            observation,
            purpose="record fainted Pokémon",
        )
        if used_fallback:
            actions = [PCAction.deposit_pokemon_to_box(pokemon, target_box=FAINTED_POKEMON_BOX) for pokemon in fainted]
            if safe_capture_box != active_box:
                actions.append(PCAction.select_active_box(safe_capture_box))
            yield from interact_with_pc(actions)
            _finalize_fainted_party_cleanup(fainted)
            return
        raise RuntimeError("cannot record fainted Pokémon because the Center PC was not observed")
    activation_locations = tuple(sorted(trigger.activation_locations, key=repr))
    current_location = (observation.overworld.map_id, observation.overworld.player_coordinates)
    target = next((location for location in activation_locations if location == current_location), activation_locations[0])
    context.campaign_recovery_pc_target = target
    navigation_goal = NavigationGoal(
        ReachLocation(target),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )
    capture_box_status = (
        f"Record {len(fainted)} fainted Pokémon in PC box {FAINTED_POKEMON_BOX + 1}"
        if fainted
        else f"Select PC box {safe_capture_box + 1} for new captures"
    )
    context.campaign_status = recovery_status(
        SemanticTarget.interaction(observation.overworld.map_id, trigger.trigger_id),
        capture_box_status,
    )
    yield from AgentControlLoop(
        lambda: observe_agent(goal=navigation_goal),
        goal=navigation_goal,
        logger=_recovery_agent_logger,
        use_movement_batch=False,
    ).run()

    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation) or trigger
    required_facing = dict(getattr(trigger, "activation_requirements", ())).get(
        (observation.overworld.map_id, observation.overworld.player_coordinates)
    )
    if required_facing is not None and getattr(observation.overworld, "facing", None) is not required_facing:
        press_direction = getattr(context.emulator, "press_direction", None)
        if callable(press_direction):
            press_direction(required_facing.button_name, run=False, fresh=True)
        else:
            context.emulator.press_button(required_facing.button_name)
        yield

    actions = [PCAction.deposit_pokemon_to_box(pokemon, target_box=FAINTED_POKEMON_BOX) for pokemon in fainted]
    if safe_capture_box != active_box:
        actions.append(PCAction.select_active_box(safe_capture_box))
    yield from interact_with_pc(actions)
    _finalize_fainted_party_cleanup(fainted)


def _living_storage_pokemon(storage) -> tuple[object, ...]:
    """Return readable living PC Pokémon, excluding the graveyard box."""

    result = []
    for box in getattr(storage, "boxes", ()) or ():
        if getattr(box, "number", None) == FAINTED_POKEMON_BOX:
            continue
        for slot in getattr(box, "slots", ()) or ():
            pokemon = getattr(slot, "pokemon", None)
            if (
                pokemon is not None
                and not getattr(pokemon, "is_egg", False)
                and getattr(pokemon, "current_hp", 0) > 0
            ):
                result.append(pokemon)
    return tuple(result)


def _reach_center_pc_for_action(*, purpose: str) -> Iterator[object]:
    """Navigate to an observed Center PC and leave the avatar facing it."""

    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation)
    context.campaign_recovery_observed_triggers = tuple(
        getattr(item, "trigger_id", None) or getattr(item, "affordance_id", None) or getattr(item, "script_symbol", None)
        for item in getattr(getattr(observation, "overworld", None), "triggers", ())
    )
    if trigger is None or not getattr(trigger, "activation_locations", ()):
        used_fallback = yield from _navigate_to_known_center_pc(observation, purpose=purpose)
        if used_fallback:
            return
        raise RuntimeError(f"Center PC was not observable for {purpose}")

    activation_locations = tuple(sorted(trigger.activation_locations, key=repr))
    current_location = (
        observation.overworld.map_id,
        observation.overworld.player_coordinates,
    )
    target = next((location for location in activation_locations if location == current_location), activation_locations[0])
    context.campaign_recovery_pc_target = target
    navigation_goal = NavigationGoal(
        ReachLocation(target),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )
    context.campaign_status = recovery_status(
        SemanticTarget.interaction(observation.overworld.map_id, trigger.trigger_id),
        f"Navigate to Center PC ({purpose})",
    )
    yield from AgentControlLoop(
        lambda: observe_agent(goal=navigation_goal),
        goal=navigation_goal,
        logger=_recovery_agent_logger,
        use_movement_batch=False,
    ).run()

    observation = observe_agent(require_overworld=True)
    trigger = _observed_pc_trigger(observation) or trigger
    required_facing = dict(getattr(trigger, "activation_requirements", ())).get(
        (observation.overworld.map_id, observation.overworld.player_coordinates)
    )
    if required_facing is not None and getattr(observation.overworld, "facing", None) is not required_facing:
        press_direction = getattr(context.emulator, "press_direction", None)
        if callable(press_direction):
            press_direction(required_facing.button_name, run=False, fresh=True)
        else:
            context.emulator.press_button(required_facing.button_name)
        yield


def execute_whiteout_continuation() -> Iterator[object]:
    """Replace a wiped party with a living PC Pokémon after a recoverable whiteout."""

    storage = get_pokemon_storage()
    candidates = _living_storage_pokemon(storage) if storage is not None else ()
    if not candidates:
        raise RunLostError("whiteout left no living Pokémon in the PC")

    # Prefer the strongest living candidate while preserving the stable PC
    # object for lookup after the fainted-party deposit changes storage.
    candidate = max(
        candidates,
        key=lambda pokemon: (
            getattr(pokemon, "level", 0),
            getattr(pokemon, "total_exp", 0),
        ),
    )
    candidate_identity = PokemonIdentity.from_pokemon(candidate)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_WHITEOUT_CONTINUATION: "
            f"living_pc_candidates={len(candidates)!r} "
            f"selected={getattr(candidate, 'name', getattr(candidate, 'species', None))!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_WHITEOUT_CONTINUATION",
    )

    # The native whiteout heals the party before returning to the Center. Move
    # the permanently fainted members to the record box before withdrawing a
    # living replacement, so the campaign never exposes a dead member again.
    yield from _deposit_fainted_party_at_current_location()

    storage = get_pokemon_storage()
    replacement = None
    for pokemon in _living_storage_pokemon(storage) if storage is not None else ():
        identity = PokemonIdentity.from_pokemon(pokemon)
        if candidate_identity is not None and identity == candidate_identity:
            replacement = pokemon
            break
    if replacement is None:
        raise RuntimeError("selected whiteout continuation Pokémon was not observed in PC storage")

    yield from _reach_center_pc_for_action(purpose="resume after whiteout")
    yield from interact_with_pc([PCAction.withdraw_pokemon_from_box(replacement)])
    if not any(
        candidate_identity is not None
        and PokemonIdentity.from_pokemon(pokemon) == candidate_identity
        for pokemon in get_party()
    ):
        raise RuntimeError("whiteout continuation Pokémon was not observed in the party")
    diagnostic_print(
        lambda: (
            "CAMPAIGN_WHITEOUT_CONTINUATION_COMPLETE: "
            f"selected={getattr(replacement, 'name', getattr(replacement, 'species', None))!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_WHITEOUT_CONTINUATION_COMPLETE",
    )


class RunLostError(RuntimeError):
    """The observed party has no autonomous legal continuation."""

    run_lost = True


def execute_campaign_recovery(planned_stop=None) -> Iterator[object]:
    """Execute recovery with one visible attempt/success/failure envelope."""

    destination = getattr(planned_stop, "destination", None) if planned_stop is not None else None
    reason = getattr(planned_stop, "reason", "") if planned_stop is not None else "route recovery selected by resource policy"
    execution_kind = getattr(planned_stop, "execution_kind", "center_route") if planned_stop is not None else "center_route"
    method = (
        "planned field recovery item"
        if execution_kind == "field_item"
        else "planned recovery stop"
        if planned_stop is not None
        else "route recovery"
    )
    resuming_cleanup = getattr(context, "campaign_recovery_phase", None) == ("cleanup", destination)
    if not resuming_cleanup:
        # These fields describe one live attempt. Clear them before a new
        # route or Center interaction so a later failure cannot display a PC
        # target or trigger list from an earlier map.
        context.campaign_recovery_observed_triggers = None
        context.campaign_recovery_pc_target = None
        context.campaign_recovery_pc_status = None
        context.campaign_recovery_retry_count = 0
    print_recovery_attempt(
        method=method,
        destination=destination,
        reason=reason,
        execution_kind=execution_kind,
    )
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_ATTEMPT: "
            f"method={method!r} execution_kind={execution_kind!r} "
            f"destination={destination!r} reason={reason!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_ATTEMPT:",
    )
    try:
        yield from _execute_campaign_recovery_impl(planned_stop)
        if planned_stop is not None and getattr(planned_stop, "reason", "") in {
            "FAINTED_PARTY_MEMBER",
            "NO_USABLE_POKEMON",
        }:
            snapshot = observe_resource_snapshot()
            if any(getattr(member, "fainted", False) for member in snapshot.party) or _remembered_fainted_party_members():
                raise RuntimeError("mandatory Center recovery postcondition was not observed")
    except Exception as error:
        context.campaign_recovery_retry_count = getattr(context, "campaign_recovery_retry_count", 0) + 1
        failure_phase = getattr(context, "campaign_recovery_phase", None)
        try:
            observed_game_state = get_game_state()
            failure_game_state = getattr(observed_game_state, "name", str(observed_game_state))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            failure_game_state = None
        failure_status = getattr(context, "campaign_status", None)
        failure_target = getattr(failure_status, "target", None)
        failure_location = getattr(failure_target, "location", None)
        print_recovery_failure(
            method=method,
            error=f"{type(error).__name__}: {error}",
            repeated=getattr(context, "campaign_recovery_retry_count", 0),
            phase=str(failure_phase) if failure_phase is not None else "planning/execution",
            location=failure_location,
            game_state=failure_game_state,
            observed_triggers=getattr(context, "campaign_recovery_observed_triggers", None),
            target_pc=getattr(context, "campaign_recovery_pc_target", None),
            traceback_text=traceback.format_exc() or None,
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_FAILURE: "
                f"method={method!r} destination={destination!r} exception_type={type(error).__name__!r} "
                f"exception={error!r} phase={failure_phase!r} location={failure_location!r} "
                f"game_state={failure_game_state!r} observed_triggers="
                f"{getattr(context, 'campaign_recovery_observed_triggers', None)!r} "
                f"target_pc={getattr(context, 'campaign_recovery_pc_target', None)!r} "
                f"pc_status={getattr(context, 'campaign_recovery_pc_status', None)!r} "
                f"traceback={traceback.format_exc()!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_FAILURE:",
        )
        raise
    context.campaign_recovery_retry_count = 0
    print_recovery_success(
        method=method,
        destination=destination,
        detail="recovery generator completed",
        execution_kind=execution_kind,
    )
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_SUCCESS: "
            f"method={method!r} execution_kind={execution_kind!r} destination={destination!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_SUCCESS:",
    )


def _execute_campaign_recovery_impl(planned_stop=None) -> Iterator[object]:
    """Execute the route observer's selected recovery source.

    Recovery selection and execution must agree on both destination and
    source.  In particular, do not fall back to the coordinate-based Center
    helper when the selected route is missing metadata: that helper has a
    separate hard-coded nurse dialogue path and cannot safely cross the same
    observed map/warp boundaries as planned recovery.
    """
    # The live campaign passes its planner-owned stop. The stop's execution
    # kind is authoritative: field medicine is used only when the planner
    # explicitly selected it, while a Center stop must retain its route.
    if planned_stop is not None:
        snapshot = observe_resource_snapshot()
        execution_kind = getattr(planned_stop, "execution_kind", "center_route")
        if execution_kind == "field_item":
            item = _select_bag_recovery_item(snapshot)
            if item is None:
                raise RuntimeError("planned field recovery item is no longer available")
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_EXECUTION: "
                    f"kind='field_item' item={item.name!r} destination={getattr(planned_stop, 'destination', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_EXECUTION",
            )
            yield from _use_field_recovery_item(item)
            return
        if snapshot.pc_healing_items and get_game_state() is GameState.POKE_STORAGE:
            # Item PC access is already safe and authoritative at this
            # boundary. Withdraw first; the next recovery observation will
            # consume the medicine from the bag before any Mart detour.
            yield from withdraw_best_pc_healing_item()
            return
        route = getattr(planned_stop, "route", None)
        mandatory_center = getattr(planned_stop, "reason", "") in {
            "FAINTED_PARTY_MEMBER",
            "NO_USABLE_POKEMON",
        }
        if mandatory_center and route is None:
            raise RuntimeError("mandatory Center recovery has no executable route")
        selected_destination = getattr(planned_stop, "destination", None)
        selected_source = getattr(planned_stop, "healing_source", None) or (
            emerald_healing_source_for_destination(selected_destination) if selected_destination is not None else None
        )
        if selected_destination is not None and selected_source is not None and getattr(planned_stop, "route", None) is not None:
            yield from execute_planned_recovery(
                selected_destination,
                selected_source,
                planned_route=getattr(planned_stop, "route", None),
                planned_goal=getattr(planned_stop, "navigation_goal", None),
            )
            return
        # A Center route stop is never silently converted into a field-item
        # action.  The planner must explicitly select ``field_item`` after a
        # known route-unavailable observation; a timeout or missing route is
        # retried by the campaign controller instead.
        if not snapshot.usable_party or snapshot.has_poisoned_party:
            raise RunLostError("no executable healing route remains")
        raise RuntimeError(
            "no usable campaign recovery capability: planned Center recovery has no executable route"
        )

    route = observe_route_recovery()
    diagnostic_print(
        lambda: (
            "CAMPAIGN_RECOVERY_EXECUTION: "
            f"started=True center_available={route.center_available} safe_to_reach_center={route.safe_to_reach_center} "
            f"distance_to_center={route.distance_to_center!r} selected={'center' if route.center_available and route.safe_to_reach_center else 'bag'}"
        ),
        trace=True,
    )
    if route.center_available and route.safe_to_reach_center:
        selected_destination = getattr(route, "center_location", None)
        selected_source = getattr(route, "healing_source", None) or (
            emerald_healing_source_for_destination(selected_destination) if selected_destination is not None else None
        )
        if selected_destination is None or selected_source is None:
            raise RuntimeError("selected recovery route has no executable healing source")
        # ``observe_route_recovery`` already paid for and selected this exact
        # source. Reusing its destination, source, and route keeps execution
        # aligned with readiness and avoids a second route selection.
        yield from execute_planned_recovery(
            selected_destination,
            selected_source,
            planned_route=getattr(route, "route", None),
            planned_goal=getattr(route, "navigation_goal", None),
        )
        diagnostic_print("CAMPAIGN_RECOVERY_EXECUTION: completed=True method=planned_source", trace=True)
        return
    snapshot = observe_resource_snapshot()
    item = _select_bag_recovery_item(snapshot)
    if item is not None:
        yield from _use_field_recovery_item(item)
        if not party_is_restored():
            raise RuntimeError("bag recovery did not restore the party")
        return
    raise RuntimeError("no usable campaign recovery capability is available")


@dataclass(frozen=True, slots=True)
class CampaignCapability:
    """Callable capability for the objective's tactical work.

    Resource ownership is decided by CampaignController.  Keeping a second
    readiness preflight here could transfer ownership after a battle has
    changed HP but before its overworld handoff is stable.
    """

    objective_id: str
    resource_policy: object
    tactical_goal: Goal
    delegate: object = execute_existing_tactical_goal

    def __call__(self) -> Iterator[object]:
        """Run the capability delegate against its configured tactical goal."""

        yield from self.delegate(self.tactical_goal)
