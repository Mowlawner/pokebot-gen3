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
from modules.map_data import MapFRLG
from modules.modes.util.higher_level_actions import heal_in_pokemon_center
from modules.modes.util.items import use_item_from_bag
from modules.modes.util.map import find_closest_pokemon_center
from modules.modes.util.walking import wait_for_player_avatar_to_be_controllable
from modules.map_path import calculate_path, PathFindingError
from modules.player import get_player_location
from modules.modes.util.pc_interaction import PCAction, interact_with_pc
from modules.modes._interface import BotModeError
from modules.pokemon_party import get_party
from modules.console import diagnostic_print

from .resource_policy import (
    HealingResource,
    PartyResource,
    ResourceObservationStatus,
    ResourceSnapshot,
)
from .resource_policy import ResourceDecision, RouteRecovery, assess_campaign_resources


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


def discover_healing_source(location=None, observation=None) -> HealingSource | None:
    """Resolve a destination or an observed full-party healing affordance."""
    if observation is not None and observation.overworld is not None:
        for trigger in observation.overworld.triggers:
            identity = trigger.affordance_id or trigger.script_symbol or ""
            if (
                trigger.kind == "object_interaction"
                and identity.endswith("_PokemonCenter_1F_EventScript_Nurse")
                and trigger.activation_locations
            ):
                return HealingSource(
                    identity,
                    HealingSourceType.POKEMON_CENTER_NURSE,
                    None,
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
        "nearest_full_party_source", HealingSourceType.FULL_PARTY_PROVIDER, center, center, True, True, True
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
            )
            diagnostic_print(
                lambda: f"RECOVERY_PREFLIGHT_LIFECYCLE: phase=yield iteration={iteration} frame={getattr(context, 'frame', None)!r}",
                trace=True,
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
            )
            yield
            continue
        diagnostic_print(
            lambda: (
                "RECOVERY_INTERACTION_PREFLIGHT: "
                f"iteration={iteration} decision=return_waiting={decision.action.action_type.name!r}"
            ),
            trace=True,
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
            destination = source.interaction_target.value
            yield from AgentControlLoop(
                lambda: observe_agent(goal=ReachLocation(destination)), goal=ReachLocation(destination)
            ).run()
            # The destination route ends at the entrance/warp.  Resolve the
            # actual affordance from a fresh interior observation.
            destination_source = None
            source = None
            while source is None:
                yield
                interior_observation = observe_agent()
                source = discover_healing_source(observation=interior_observation)
        yield from AgentControlLoop(
            lambda: observe_agent(goal=ReachInteractionPosition(source.interaction_trigger_id)),
            goal=ReachInteractionPosition(source.interaction_trigger_id),
        ).run()
        yield from _execute_healing_source_interaction(source)
        return


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
    # Script activation is asynchronous: the overworld can remain apparently
    # controllable for several observations after the A input is accepted.
    # Keep ownership during that transition, but retain a bounded failure
    # path for a genuinely ignored interaction.
    interaction_start_timeout = 32
    while True:
        if (
            last_observation is not None
            and party_is_restored()
            and last_observation.interaction.controllable
            and not last_observation.interaction.script_active
            and last_observation.interaction.interaction_phase is InteractionPhase.NONE
        ):
            diagnostic_print("HEAL_PARTY_COMPLETE: party_fully_restored=True interaction_finished=True", trace=True)
            return
        observation = observe_agent()
        party_restored = party_is_restored()
        # Restored HP is not the same boundary as completion of the nurse's
        # event script.  Emerald restores the party before displaying the
        # closing messages, so leave the capability mounted until the ROM has
        # returned to ordinary overworld control.
        interaction_finished = (
            observation.interaction.controllable
            and not observation.interaction.script_active
            and observation.interaction.interaction_phase is InteractionPhase.NONE
        )
        if party_restored and interaction_finished:
            diagnostic_print("HEAL_PARTY_COMPLETE: party_fully_restored=True interaction_finished=True", trace=True)
            return
        last_observation = observation
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
        distance = len(calculate_path(location, center_location))
        result = RouteRecovery(center_available=True, distance_to_center=distance, safe_to_reach_center=True)
        if trace is not None:
            trace.duration("campaign_route_recovery_observation_duration_ms", started)
        return result
    except (BotModeError, PathFindingError):
        # A valid observation with no usable route is known, not transient.
        result = RouteRecovery()
        if trace is not None:
            trace.duration("campaign_route_recovery_observation_duration_ms", started)
        return result


def recover_at_nearest_center() -> Iterator[object]:
    """Perform an existing Center healing flow and verify its result."""
    yield from _resolve_recovery_interaction()
    yield from wait_for_player_avatar_to_be_controllable()
    # Reuse the same current location representation required by Center
    # pathfinding; omitting it makes the helper rediscover from None and
    # causes calculate_path(None, ...).
    center = find_closest_pokemon_center(get_player_location())
    yield from _navigate_recovery_to_center(center)
    yield from _wait_for_center_interior(center)
    diagnostic_print(
        lambda: f"CENTER_EXIT_HEAL_CALL: phase=before frame={getattr(context, 'frame', None)!r} emulator_frame={context.emulator.get_frame_count()!r} center={center!r}",
        trace=True,
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
    interior_name = (
        "PALLET_TOWN_PLAYERS_HOUSE_1F" if center.name == "PalletTown" else f"{outdoor_map.name}_POKEMON_CENTER_1F"
    )
    interior_map = getattr(MapFRLG if isinstance(outdoor_map, MapFRLG) else type(outdoor_map), interior_name, None)
    if interior_map is None:
        raise RuntimeError(f"No Pokémon Center interior map is defined for {outdoor_map!r}")
    while True:
        observation = observe_agent()
        overworld = observation.overworld
        observed_map = getattr(overworld, "map_id", None)
        controllable = getattr(overworld, "controllable", False)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_HANDOFF: "
                f"observed_map={observed_map!r} expected_interior={interior_map!r} "
                f"controllable={controllable!r} proceeding={observed_map == interior_map and controllable}"
            ),
            trace=True,
        )
        if observed_map == interior_map and controllable:
            return
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
        yield from recover_at_nearest_center()
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
