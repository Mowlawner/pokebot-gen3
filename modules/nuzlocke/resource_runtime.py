"""Runtime adapters for the pure campaign resource policy."""

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
from modules.goals import ActivateTrigger, EncounterMode, Goal, NavigationGoal, ReachInteractionPosition, ReachLocation
from modules.map_data import MapFRLG, PokemonCenter
from modules.map import get_map_metadata
from modules.modes.util.higher_level_actions import heal_in_pokemon_center
from modules.modes.util.items import use_item_from_bag
from modules.modes.util.map import find_closest_pokemon_center
from modules.modes.util.walking import wait_for_player_avatar_to_be_controllable
from modules.map_path import calculate_path, PathFindingError, Direction
from modules.player import get_player_location
from modules.navigation import NavigationWorld, plan_with_world_navigation
from modules.overworld import OverworldObservationResult, perceive_overworld
from modules.world_navigation import get_world_map_graph
from modules.modes.util.pc_interaction import PCAction, interact_with_pc
from modules.modes._interface import BotModeError
from modules.pokemon_party import get_party
from modules.console import diagnostic_print
from .campaign_status import recovery_status
from modules.goals import SemanticTarget
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
    FULL_PARTY_PROVIDER = "full_party_provider"
    POKEMON_CENTER_NURSE = "pokemon_center_nurse"


@dataclass(frozen=True, slots=True)
class HealingSource:
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
        if decision.action.action_type is AgentActionType.ADVANCE_DIALOGUE:
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
            yield from AgentControlLoop(
                lambda: observe_agent(goal=ReachLocation(destination)), goal=ReachLocation(destination)
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


def execute_planned_recovery(destination, planned_source=None) -> Iterator[object]:
    """Execute recovery at the Center selected by campaign planning.

    The route destination is intentionally passed in from the plan.  The
    interaction affordance is still discovered from a fresh interior
    observation, but execution never reselects a different Center.
    """
    if party_is_restored():
        return
    context.campaign_status = recovery_status(
        SemanticTarget.at(destination),
        "Navigate to healing source",
    )
    yield from AgentControlLoop(
        lambda: observe_agent(goal=ReachLocation(destination)),
        goal=ReachLocation(destination),
    ).run()
    center = next((candidate for candidate in PokemonCenter if candidate.value == destination), None)
    catalog_source = planned_source or emerald_healing_source_for_destination(destination)
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
            AgentActionExecutor().execute(decision.action, observation)
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
        try:
            distance = len(calculate_path(location, center_location))
        except PathFindingError:
            # The legacy pathfinder deliberately cannot cross map warps. Use
            # the same world planner that executes campaign navigation for
            # recovery sources inside interiors such as Birch's Lab.
            overworld = perceive_overworld()
            if isinstance(overworld, OverworldObservationResult):
                raise
            plan, route = plan_with_world_navigation(
                NavigationWorld.from_overworld(overworld),
                location,
                ReachLocation(center_location),
                get_world_map_graph(),
            )
            if plan.metrics is None or plan.destination is None:
                raise PathFindingError("world recovery route has no executable metrics")
            distance = plan.metrics.total_route_cost
            if distance is None:
                raise PathFindingError("world recovery route has no cost")
        result = RouteRecovery(center_available=True, distance_to_center=distance, safe_to_reach_center=True)
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
    loop = AgentControlLoop(lambda: observe_agent(goal=ReachLocation(location)), goal=ReachLocation(location)).run()
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
        yield from self.delegate(self.tactical_goal)
