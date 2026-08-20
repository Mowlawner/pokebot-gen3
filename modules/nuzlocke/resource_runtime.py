"""Runtime adapters for the pure campaign resource policy."""

from collections.abc import Iterator
from dataclasses import dataclass

from modules.items import get_item_bag, get_item_storage, get_item_by_name
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

from .resource_policy import HealingResource, PartyResource, ResourceSnapshot
from .resource_policy import ResourceDecision, RouteRecovery, assess_campaign_resources


def execute_existing_tactical_goal(goal: Goal) -> Iterator[object]:
    """Run the controller's normal tactical loop for an objective goal."""
    return AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()


def observe_resource_snapshot() -> ResourceSnapshot:
    party = get_party()
    bag_items = []
    for slot in get_item_bag().items:
        if slot.item.battle_use.value == "healing" or slot.item.field_use.value == "healing":
            bag_items.append(HealingResource(slot.item.name, slot.quantity, slot.item.parameter, "bag"))
    pc_items = []
    for slot in get_item_storage().items:
        if slot.item.battle_use.value == "healing" or slot.item.field_use.value == "healing":
            pc_items.append(HealingResource(slot.item.name, slot.quantity, slot.item.parameter, "pc"))
    return ResourceSnapshot(
        tuple(PartyResource(p.current_hp, p.total_hp, p.status_condition.value, p.current_hp <= 0) for p in party),
        tuple(bag_items),
        tuple(pc_items),
    )


def party_is_restored() -> bool:
    party = get_party()
    return bool(party) and all(
        p.current_hp == p.total_hp and p.status_condition.value == "none" for p in party if not p.is_egg
    )


def observe_route_recovery() -> RouteRecovery:
    try:
        center = find_closest_pokemon_center()
        distance = len(calculate_path(get_player_location(), center.value))
        return RouteRecovery(center_available=True, distance_to_center=distance, safe_to_reach_center=True)
    except (BotModeError, PathFindingError, RuntimeError, ValueError, AttributeError):
        return RouteRecovery()


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
