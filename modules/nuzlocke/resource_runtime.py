"""Runtime adapters for the pure campaign resource policy."""

from collections.abc import Iterator
from dataclasses import dataclass

from modules.items import InvalidItemIndexError, get_item_bag, get_item_storage, get_item_by_name
from modules.context import context
from modules.agent_control import AgentControlLoop, observe_agent
from modules.goals import Goal
from modules.modes.util.higher_level_actions import heal_in_pokemon_center
from modules.modes.util.items import use_item_from_bag
from modules.modes.util.map import find_closest_pokemon_center
from modules.map_path import calculate_path, PathFindingError
from modules.player import get_player_location
from modules.modes.util.pc_interaction import PCAction, interact_with_pc
from modules.modes._interface import BotModeError
from modules.pokemon_party import get_party

from .resource_policy import (
    HealingResource,
    PartyResource,
    ResourceObservationStatus,
    ResourceSnapshot,
)
from .resource_policy import ResourceDecision, RouteRecovery, assess_campaign_resources


def execute_existing_tactical_goal(goal: Goal) -> Iterator[object]:
    """Run the controller's normal tactical loop for an objective goal."""
    return AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()


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
    center = find_closest_pokemon_center()
    yield from heal_in_pokemon_center(center)
    if not party_is_restored():
        raise RuntimeError("Pokémon Center interaction completed without restoring the party")


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
    if route.center_available and route.safe_to_reach_center:
        yield from recover_at_nearest_center()
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
    """Callable capability that performs readiness preflight then resumes work."""

    objective_id: str
    resource_policy: object
    tactical_goal: Goal
    delegate: object = execute_existing_tactical_goal

    def __call__(self) -> Iterator[object]:
        if self.resource_policy is not None:
            snapshot = observe_resource_snapshot()
            decision = assess_campaign_resources(
                self.resource_policy,
                snapshot,
                RouteRecovery(),
            )
            if decision is not ResourceDecision.CONTINUE:
                route = observe_route_recovery()
                decision = assess_campaign_resources(self.resource_policy, snapshot, route)
                if decision is ResourceDecision.RECOVER_AT_CENTER and route.center_available:
                    yield from recover_at_nearest_center()
                elif decision is ResourceDecision.USE_HEALING_ITEM:
                    yield from use_best_bag_healing_item()
        yield from self.delegate(self.tactical_goal)
