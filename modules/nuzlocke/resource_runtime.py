"""Runtime adapters for the pure campaign resource policy."""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

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
from modules.interaction_state import InteractionPhase
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
from modules.map_data import MapFRLG, PokemonCenter
from modules.map import get_map_all_tiles, get_map_data, get_map_metadata
from modules.modes.util.higher_level_actions import heal_in_pokemon_center
from modules.modes.util.items import use_item_from_bag
from modules.modes.util.map import find_closest_pokemon_center
from modules.modes.util.walking import wait_for_player_avatar_to_be_controllable
from modules.map_path import calculate_path, PathFindingError, Direction
from modules.player import get_player_location
from modules.navigation import (
    NavigationAction,
    NavigationActionType,
    NavigationError,
    NavigationPlan,
    NavigationWorld,
    plan_with_world_navigation,
)
from modules.overworld import MovementState, OverworldObservationResult, perceive_overworld
from modules.world_navigation import get_world_map_graph
from modules.modes.util.pc_interaction import PCAction, interact_with_pc
from modules.modes._interface import BotModeError
from modules.pokemon_party import get_party
from modules.console import diagnostic_print
from .campaign_status import recovery_status
from modules.goals import SemanticTarget
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
)
from .resource_policy import ResourceDecision, RouteRecovery, assess_campaign_resources


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
            "CAMPAIGN_RECOVERY_ENTRY_WARP_FALLBACK: "
            f"from={current!r} toward={target!r} direction={direction.name!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_RECOVERY_ENTRY_WARP_FALLBACK",
    )
    return True


def _map_id_value(map_id):
    """Normalize map enums to the tuple emitted by overworld perception."""
    return getattr(map_id, "value", map_id)


def _recovery_navigation_goal(destination, source=None):
    """Return the executable goal for a cataloged healing destination."""
    source = source or emerald_healing_source_for_destination(destination)
    interior_map = getattr(source, "interior_map", None)
    if interior_map is None:
        # Preserve compatibility for non-Emerald providers that expose only a
        # standable destination coordinate.
        return ReachLocation(destination)
    return ReachWarp(destination_map=_map_id_value(interior_map))


def _navigation_plan_from_legacy_path(start, waypoints) -> NavigationPlan | None:
    """Adapt an already-computed legacy path for the tactical navigator.

    ``calculate_path`` predates the goal-aware navigator and returns waypoints
    rather than ``NavigationAction`` objects. Recovery only needs to retain
    the route it just measured, so re-searching the same path merely to change
    representations would recreate the frame-time stall this adapter avoids.
    Invalid/mock waypoint collections remain deliberately unsupported and
    fall back to normal planning.
    """
    if not isinstance(start, tuple) or len(start) != 2 or not isinstance(waypoints, (tuple, list)):
        return None
    current = start
    actions = []
    for waypoint in waypoints:
        map_id = getattr(waypoint, "map", None)
        coordinates = getattr(waypoint, "coordinates", None)
        direction = getattr(waypoint, "direction", None)
        if (
            map_id is None
            or not isinstance(coordinates, tuple)
            or len(coordinates) != 2
            or direction is None
        ):
            return None
        destination = (map_id, coordinates)
        is_warp = bool(getattr(waypoint, "is_warp", False)) or destination[0] != current[0]
        actions.append(
            NavigationAction(
                NavigationActionType.WARP if is_warp else NavigationActionType.MOVE,
                direction,
                current,
                destination,
                transition_kind="warp" if is_warp else None,
            )
        )
        current = destination
    if not actions:
        return None
    return NavigationPlan(tuple(actions), current)


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
        for trigger in observation.overworld.triggers:
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


def _resolve_recovery_interaction() -> Iterator[object]:
    """Advance ordinary actionable interaction before recovery navigation.

    A non-controllable avatar is not necessarily waiting for input.  Reuse the
    observation/action boundary used by normal campaign control so dialogue is
    advanced, while passive transitions are left for the existing controllability
    wait below.
    """
    executor = AgentActionExecutor()
    iteration = 0
    previous_signature = None
    while True:
        iteration += 1
        diagnostic_print(
            lambda: f"RECOVERY_PREFLIGHT_LIFECYCLE: phase=resume iteration={iteration} frame={getattr(context, 'frame', None)!r}",
            trace=True,
            prefix="RECOVERY_PREFLIGHT_LIFECYCLE",
        )
        observation = observe_agent()
        decision = select_action(observation)
        interaction = observation.interaction
        overworld = observation.overworld
        signature = (
            getattr(interaction.game_state, "name", repr(interaction.game_state)),
            interaction.dialogue_waiting,
            interaction.field_message_lifecycle_active,
            interaction.field_message_advance_ready,
            interaction.controllable,
            getattr(overworld, "map_id", None),
            getattr(overworld, "player_coordinates", None),
        )
        diagnostic_print(
            lambda: (
                "RECOVERY_INTERACTION_PREFLIGHT: "
                f"iteration={iteration} frame={getattr(getattr(context, 'emulator', None), 'frame_count', None)!r} "
                f"location={(getattr(overworld, 'map_id', None), getattr(overworld, 'player_coordinates', None))!r} "
                f"game_state={getattr(interaction.game_state, 'name', repr(interaction.game_state))!r} "
                f"controllable={interaction.controllable!r} dialogue_waiting={interaction.dialogue_waiting!r} "
                f"field_message_lifecycle_active={interaction.field_message_lifecycle_active!r} "
                f"field_message_advance_ready={interaction.field_message_advance_ready!r} "
                f"interaction_type={observation.interaction_type.name!r} "
                f"selected_action={decision.action.action_type.name!r} "
                f"observation_changed={previous_signature is None or signature != previous_signature!r}"
            ),
            trace=True,
            prefix="RECOVERY_INTERACTION_PREFLIGHT",
        )
        previous_signature = signature
        if decision.action.action_type in (
            AgentActionType.ADVANCE_DIALOGUE,
            AgentActionType.ACCELERATE_DIALOGUE_RENDER,
        ):
            result = executor.execute(decision.action, observation)
            diagnostic_print(
                lambda: (
                    "RECOVERY_INTERACTION_ACTION: "
                    f"action={decision.action.action_type.name!r} executed={getattr(result.result_type, 'name', None)!r} "
                    f"message={getattr(result, 'message', None)!r}"
                ),
                trace=True,
                prefix="RECOVERY_INTERACTION_ACTION",
            )
            diagnostic_print(
                lambda: f"RECOVERY_PREFLIGHT_LIFECYCLE: phase=yield iteration={iteration} frame={getattr(context, 'frame', None)!r}",
                trace=True,
                prefix="RECOVERY_PREFLIGHT_LIFECYCLE",
            )
            yield
            continue
        if decision.action.action_type is AgentActionType.WAIT_REOBSERVE and not interaction.controllable:
            diagnostic_print(
                lambda: (
                    "RECOVERY_INTERACTION_PREFLIGHT: "
                    f"iteration={iteration} decision=yield_wait={decision.action.action_type.name!r}"
                ),
                trace=True,
                prefix="RECOVERY_INTERACTION_PREFLIGHT",
            )
            yield
            continue
        diagnostic_print(
            lambda: (
                "RECOVERY_INTERACTION_PREFLIGHT: "
                f"iteration={iteration} decision=return_waiting={decision.action.action_type.name!r}"
            ),
            trace=True,
            prefix="RECOVERY_INTERACTION_PREFLIGHT",
        )
        return


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
            bag_items.append(HealingResource(slot.item.name, slot.quantity, slot.item.parameter, "bag"))
    pc_items = []
    try:
        storage_items = storage.items
        for slot in storage_items:
            if slot.item.battle_use.value == "healing" or slot.item.field_use.value == "healing":
                pc_items.append(HealingResource(slot.item.name, slot.quantity, slot.item.parameter, "pc"))
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
            yield from AgentControlLoop(
                lambda: observe_agent(goal=navigation_goal), goal=navigation_goal
            ).run()
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
    candidates = tuple(sorted(
        tile.local_position
        for tile in get_map_all_tiles(map_data)
        if tile.has_encounters
    ))
    if not candidates:
        raise BotModeError(f"no land encounter tile is available on training map {map_id!r}")
    return map_id, candidates


_PREPARATION_NAVIGATION_RETRY_LIMIT = 12


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


def _preparation_navigation_goal(
    overworld,
    training_map,
    training_candidates: tuple[tuple[int, int], ...],
    *,
    allow_wild: bool = True,
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
    candidates = sorted(
        training_candidates,
        key=lambda position: (
            abs(position[0] - start[1][0]) + abs(position[1] - start[1][1]),
            position[0],
            position[1],
        ),
    )
    if allow_wild:
        reachable_positions = _reachable_local_positions(world, start, training_map)
        if reachable_positions is not None:
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
                return ReachLocation((training_map, coordinates)), None
        else:
            for coordinates in candidates:
                try:
                    plan_with_world_navigation(world, start, ReachLocation((training_map, coordinates)))
                except (AttributeError, IndexError, KeyError, NavigationError, PathFindingError, RuntimeError, TypeError, ValueError) as error:
                    last_error = error
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_PREPARATION_CANDIDATE_REJECTED: "
                            f"map={training_map!r} coordinates={coordinates!r} "
                            f"start={start!r} error_type={type(error).__name__!r} error={error!r}"
                        ),
                        trace=True,
                    )
                    continue
                return ReachLocation((training_map, coordinates)), None
    else:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_WILD_DISABLED: "
                f"map={training_map!r} start={start!r} reason='area encounter already resolved'"
            ),
            trace=True,
        )

    trainers = sorted(
        (
            obj
            for obj in getattr(overworld, "objects", ())
            if getattr(obj, "trainer_id", None) is not None
            and getattr(obj, "trainer_defeated", None) is False
        ),
        key=lambda obj: (
            abs(obj.location[1][0] - start[1][0]) + abs(obj.location[1][1] - start[1][1]),
            obj.trainer_id,
        ),
    )
    for trainer in trainers:
        trainer_goal = NavigationGoal(
            EngageTrainer(trainer.trainer_id),
            constraints=GoalConstraints(trainer_mode=TrainerMode.ENGAGE),
            encounter_mode=EncounterMode.IGNORE,
        )
        try:
            plan_with_world_navigation(world, start, trainer_goal)
        except (AttributeError, IndexError, KeyError, NavigationError, PathFindingError, RuntimeError, TypeError, ValueError) as error:
            last_error = error
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PREPARATION_TRAINER_REJECTED: "
                    f"map={training_map!r} trainer={trainer.trainer_id!r} "
                    f"start={start!r} error_type={type(error).__name__!r} error={error!r}"
                ),
                trace=True,
            )
            continue
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_TRAINER_SELECTED: "
                f"map={training_map!r} start={start!r} trainer={trainer.trainer_id!r}"
            ),
            trace=True,
        )
        return trainer_goal, None

    return None, last_error


def _preparation_encounter_resolved(training_map) -> bool:
    """Return whether the selected area's first encounter is no longer open.

    Preparation may use wild grass only until the first encounter reaches a
    terminal outcome.  Once it is captured, lost, or causes a faint, further
    grass battles would be a one-encounter violation.  Read this from the
    rules projection rather than from the event-store cache or the current
    party so the live policy and replay use the same authority.
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
    return encounter is not None and getattr(encounter, "status", None) not in {"none", "pending", "unknown"}


def execute_campaign_preparation(training_map, *, target_level: int) -> Iterator[object]:
    """Train the current living party to a bounded target before a boss.

    The target is checked from the live party on every loop boundary. The
    Center loop owns healing and encounter-area movement, while the campaign
    mode supplies the Nuzlocke-aware battle strategy at battle entry.
    """
    if target_level < 1:
        raise ValueError("preparation target must be positive")

    from modules.battle_strategies.nuzlocke_level_balancing import (
        NuzlockeLevelBalancingBattleStrategy,
    )
    from modules.modes.util.pokecenter_loop import PokecenterLoopController

    loop = None

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
                or (
                    (identity := PokemonIdentity.from_pokemon(pokemon))
                    is not None
                    and identity not in dead
                )
            )
        )
        return bool(living) and all(pokemon.level >= target_level for pokemon in living)

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
    map_goal = NavigationGoal(SemanticTarget.map(training_map), encounter_mode=EncounterMode.AVOID)
    yield from AgentControlLoop(
        lambda: observe_agent(goal=map_goal),
        goal=map_goal,
    ).run()

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

    training_location = navigation_goal.location
    if training_location is not None:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PREPARATION_CANDIDATE_SELECTED: "
                f"map={training_map!r} training_location={training_location!r}"
            ),
            trace=True,
        )

    loop = PokecenterLoopController(
        recovery_handler=lambda center: execute_planned_recovery(center.value),
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
        """Stop wild training at the target or a terminal area encounter."""

        nonlocal target_was_reached
        # Stop at the first terminal encounter boundary.  The outer
        # preparation loop will then select a trainer goal, keeping all later
        # training battles outside the consumed wild-encounter path.
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

    # The wild loop deliberately ends after the area's encounter is resolved.
    # Continue through the same observed trainer-first selection path rather
    # than returning to grass.  This also handles an encounter that was
    # resolved by a prior invocation of this capability.
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
            allow_wild=False,
        )
        if trainer_goal is None:
            raise BotModeError(
                f"no reachable trainer is available for safe preparation on training map {training_map!r}: {last_error}"
            )
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


def execute_planned_recovery(destination, planned_source=None, planned_route=None) -> Iterator[object]:
    """Execute recovery at the Center selected by campaign planning.

    The route destination is intentionally passed in from the plan.  The
    interaction affordance is still discovered from a fresh interior
    observation, but execution never reselects a different Center.  When a
    planner-composed route is supplied, it is adopted directly for the
    outdoor leg instead of being recomputed by the tactical navigator.
    """
    if party_is_restored():
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
    navigation_goal = ReachWarp(destination_map=interior_map.value if hasattr(interior_map, "value") else interior_map)
    context.campaign_status = recovery_status(
        SemanticTarget.at(destination),
        "Navigate to healing source",
    )
    if planned_route is not None:
        # A legacy recovery route includes the final door warp and therefore
        # terminates on the Center's interior map. A planner-composed
        # ReachLocation route, by contrast, ends on the selected exterior
        # destination and retains the historical goal contract.
        planned_goal = ReachLocation(destination)
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
        ).run()
    else:
        yield from AgentControlLoop(
            lambda: observe_agent(goal=navigation_goal),
            goal=navigation_goal,
        ).run()
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
                "Navigate to nurse" if getattr(source, "source_type", None) is HealingSourceType.POKEMON_CENTER_NURSE
                else "Navigate to healing source",
            )
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
                "Interact with nurse" if getattr(source, "source_type", None) is HealingSourceType.POKEMON_CENTER_NURSE
                else "Interact with healing source",
            )
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
            return
        if observation.interaction_type.name == "DIALOGUE" or not observation.interaction.controllable:
            decision = select_action(observation)
            executor.execute(decision.action, observation)
        yield
def _execute_healing_source_interaction(source: HealingSource) -> Iterator[object]:
    """Interact with a reached semantic healing source through shared control.

    This deliberately has no Center-specific dialogue loop.  Every input is
    preceded by an authoritative observation and is followed by a yield so the
    next emulator frame is reclassified by the common interaction pipeline.
    """
    executor = AgentActionExecutor()
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


def observe_route_recovery() -> RouteRecovery:
    """Observe recovery routing without turning transient avatar gaps into crashes."""
    trace = getattr(context, "stutter_trace", None)
    started = trace.now() if trace is not None else 0
    try:
        location = get_player_location()
    except RuntimeError as error:
        # get_player_location documents RuntimeError for inactive/corrupt
        # avatar data during transitions.  Do not mask pathfinding errors.
        result = RouteRecovery(observation_available=False, observation_error=str(error))
        if trace is not None:
            trace.duration("campaign_route_recovery_observation_duration_ms", started)
        return result
    try:
        if (
            not isinstance(location, tuple)
            or len(location) != 2
            or location[0] is None
            or not isinstance(location[1], tuple)
            or len(location[1]) != 2
            or any(coordinate is None for coordinate in location[1])
        ):
            result = RouteRecovery(observation_available=False, observation_error="player_location_unavailable")
            if trace is not None:
                trace.duration("campaign_route_recovery_observation_duration_ms", started)
            return result
        center = find_closest_pokemon_center(location)
        center_location = getattr(center, "value", None)
        if (
            not isinstance(center_location, tuple)
            or len(center_location) != 2
            or center_location[0] is None
            or not isinstance(center_location[1], tuple)
            or len(center_location[1]) != 2
            or any(coordinate is None for coordinate in center_location[1])
        ):
            result = RouteRecovery(observation_available=False, observation_error="center_location_unavailable")
            if trace is not None:
                trace.duration("campaign_route_recovery_observation_duration_ms", started)
            return result
        route = None
        try:
            legacy_path = calculate_path(location, center_location)
            distance = len(legacy_path)
            route = _navigation_plan_from_legacy_path(location, legacy_path)
        except PathFindingError:
            # The legacy pathfinder deliberately cannot cross map warps. Use
            # the same world planner that executes campaign navigation for
            # recovery sources inside interiors such as Birch's Lab.
            overworld = perceive_overworld()
            if isinstance(overworld, OverworldObservationResult):
                raise
            plan, _ = plan_with_world_navigation(
                NavigationWorld.from_overworld(overworld),
                location,
                _recovery_navigation_goal(center_location),
                get_world_map_graph(),
            )
            if plan.metrics is None or plan.destination is None:
                raise PathFindingError("world recovery route has no executable metrics")
            distance = plan.metrics.total_route_cost
            if distance is None:
                raise PathFindingError("world recovery route has no cost")
        result = RouteRecovery(
            center_available=True,
            center_location=center_location,
            distance_to_center=distance,
            safe_to_reach_center=True,
            route=route,
        )
        if trace is not None:
            trace.duration("campaign_route_recovery_observation_duration_ms", started)
        return result
    except (BotModeError, PathFindingError):
        # A valid observation with no usable route is known, not transient.
        local_catalog_source = next(
            (
                source
                for source in emerald_healing_sources()
                if source.outdoor_location[0] == location[0]
                or source.interior_map == location[0]
            ),
            None,
        )
        result = RouteRecovery(healing_source_available=local_catalog_source is not None)
        if trace is not None:
            trace.duration("campaign_route_recovery_observation_duration_ms", started)
        return result


def recover_at_nearest_center(current_location=None, selected_center=None) -> Iterator[object]:
    """Perform an existing Center healing flow and verify its result."""
    yield from _resolve_recovery_interaction()
    yield from wait_for_player_avatar_to_be_controllable()
    # Reuse the same current location representation required by Center
    # pathfinding; omitting it makes the helper rediscover from None and
    # causes calculate_path(None, ...).
    center = selected_center or find_closest_pokemon_center(current_location or get_player_location())
    yield from _navigate_recovery_to_center(center)
    yield from _wait_for_center_interior(center)
    diagnostic_print(
        lambda: f"CENTER_EXIT_HEAL_CALL: phase=before frame={getattr(context, 'frame', None)!r} emulator_frame={context.emulator.get_frame_count()!r} center={center!r}",
        trace=True,
        prefix="CAMPAIGN_RECOVERY_HANDOFF",
    )
    diagnostic_print(
        lambda: f"CENTER_HEAL_BEGIN: frame={getattr(context, 'frame', None)!r} emulator_frame={context.emulator.get_frame_count()!r} center={center!r}",
        trace=True,
    )
    yield from heal_in_pokemon_center(center, navigate_to_destination=False)
    diagnostic_print(
        lambda: f"CENTER_EXIT_HEAL_CALL: phase=after frame={getattr(context, 'frame', None)!r} emulator_frame={context.emulator.get_frame_count()!r} center={center!r}",
        trace=True,
    )
    diagnostic_print(
        lambda: f"CENTER_HEAL_RETURN: frame={getattr(context, 'frame', None)!r} emulator_frame={context.emulator.get_frame_count()!r} center={center!r}",
        trace=True,
    )
    if not party_is_restored():
        raise RuntimeError("Pokémon Center interaction completed without restoring the party")


def _navigate_recovery_to_center(center) -> Iterator[object]:
    """Reach a selected Center through the normal observation-driven loop."""
    location = (center.value[0], center.value[1])
    navigation_goal = _recovery_navigation_goal(location)
    loop = AgentControlLoop(lambda: observe_agent(goal=navigation_goal), goal=navigation_goal).run()
    yield from loop


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
                and abs(position[1][0] - center.value[1][0])
                + abs(position[1][1] - center.value[1][1])
                == 1
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


def use_best_bag_healing_item() -> Iterator[object]:
    """Use the strongest available healing item from the bag."""

    snapshot = observe_resource_snapshot()
    usable = [item for item in snapshot.bag_healing_items if item.quantity > 0]
    if not usable:
        raise RuntimeError("No healing item is available in the bag")
    item = max(usable, key=lambda candidate: candidate.heal_amount)
    yield from use_item_from_bag(get_item_by_name(item.name))


def withdraw_best_pc_healing_item() -> Iterator[object]:
    """Withdraw the most useful stored recovery item at an already-open PC."""
    snapshot = observe_resource_snapshot()
    usable = [item for item in snapshot.pc_healing_items if item.quantity > 0]
    if not usable:
        raise RuntimeError("No healing item is available in PC storage")
    item = max(usable, key=lambda candidate: candidate.heal_amount)
    yield from interact_with_pc([PCAction.withdraw_item(get_item_by_name(item.name), 1)])


def execute_campaign_recovery() -> Iterator[object]:
    """Use the existing recovery primitives and verify the resulting party."""
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
        # The legacy helper navigates to the nurse's object tile itself
        # (behind the counter).  That can leave the avatar stationary at the
        # Center entrance while the helper continues issuing inputs.  Use the
        # observed healing-source path so navigation terminates on the
        # counter's activation tile before interacting.
        current_location = get_player_location()
        center = find_closest_pokemon_center(current_location)
        if hasattr(center, "value"):
            yield from execute_planned_recovery(center.value)
        else:
            # Keep lightweight capability fixtures and third-party center
            # providers compatible until they expose a concrete destination.
            yield from recover_at_nearest_center(current_location, center)
        diagnostic_print("CAMPAIGN_RECOVERY_EXECUTION: completed=True method=center", trace=True)
        return
    snapshot = observe_resource_snapshot()
    if snapshot.bag_healing_items:
        yield from use_best_bag_healing_item()
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
