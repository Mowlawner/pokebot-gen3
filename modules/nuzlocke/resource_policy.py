"""Objective-aware campaign resource policy.

This module is deliberately pure: it contains no emulator reads and never
executes navigation, battle, item, or healing actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ReadinessImportance(Enum):
    NONE = "none"
    NORMAL = "normal"
    IMPORTANT = "important"


class EncounterPolicy(Enum):
    NORMAL = "normal"
    PRESERVE = "preserve"
    GRIND = "grind"


class ResourceDecision(Enum):
    CONTINUE = "continue"
    PRESERVE_RESOURCES = "preserve_resources"
    PREFER_RUN = "prefer_run"
    RECOVER_AT_CENTER = "recover_at_center"
    USE_HEALING_ITEM = "use_healing_item"
    WITHDRAW_HEALING_ITEM = "withdraw_healing_item"


@dataclass(frozen=True, slots=True)
class PartyResource:
    current_hp: int
    max_hp: int
    status: str | None = None
    fainted: bool = False

    @property
    def hp_ratio(self) -> float:
        return self.current_hp / self.max_hp if self.max_hp > 0 else 0.0


@dataclass(frozen=True, slots=True)
class HealingResource:
    name: str
    quantity: int
    heal_amount: int
    location: str = "bag"
    acquisition_cost: int = 0


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    party: tuple[PartyResource, ...] = ()
    bag_healing_items: tuple[HealingResource, ...] = ()
    pc_healing_items: tuple[HealingResource, ...] = ()
    current_map: object | None = None

    @property
    def usable_party(self) -> tuple[PartyResource, ...]:
        return tuple(p for p in self.party if not p.fainted and p.current_hp > 0)

    @property
    def worst_hp_ratio(self) -> float:
        usable = self.usable_party
        return min((p.hp_ratio for p in usable), default=0.0)

    @property
    def total_missing_hp(self) -> int:
        return sum(max(0, p.max_hp - p.current_hp) for p in self.usable_party)


@dataclass(frozen=True, slots=True)
class RouteRecovery:
    center_available: bool = False
    distance_to_center: int | None = None
    center_on_route: bool = False
    safe_to_reach_center: bool = True
    distance_to_objective: int | None = None
    pc_accessible: bool = False
    pc_acquisition_cost: int | None = None


@dataclass(frozen=True, slots=True)
class ResourceObjective:
    objective_id: str
    readiness: ReadinessImportance = ReadinessImportance.NONE
    encounters: EncounterPolicy = EncounterPolicy.NORMAL
    mandatory_battle: bool = False
    recover_before_completion: bool = False
    # Minimum acceptable health for an important objective. This is objective
    # metadata, not a global HP rule.
    minimum_hp_ratio: float = 0.5


def _useful_item(snapshot: ResourceSnapshot, route: RouteRecovery) -> HealingResource | None:
    candidates = [item for item in snapshot.bag_healing_items if item.quantity > 0 and item.heal_amount > 0]
    candidates = [item for item in candidates if item.heal_amount >= snapshot.total_missing_hp]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.heal_amount)


def assess_campaign_resources(
    objective: ResourceObjective,
    snapshot: ResourceSnapshot,
    route: RouteRecovery,
    *,
    wild_encounter: bool = False,
) -> ResourceDecision:
    """Return a deterministic campaign action without executing it."""
    if objective.encounters is EncounterPolicy.GRIND and wild_encounter:
        return ResourceDecision.CONTINUE
    if objective.readiness is ReadinessImportance.NONE or not objective.mandatory_battle:
        return ResourceDecision.CONTINUE
    if not snapshot.usable_party:
        return ResourceDecision.RECOVER_AT_CENTER if route.center_available else ResourceDecision.PRESERVE_RESOURCES

    healthy = snapshot.worst_hp_ratio >= objective.minimum_hp_ratio
    # A healthy party may spend some HP before a planned recovery point. The
    # objective's recovery requirement applies to completion, not to every
    # preceding optional encounter.
    if healthy:
        # A preserve objective may still spend a little HP when a natural
        # recovery point is ahead.  Once HP has actually been spent and no
        # such route margin is known, avoid turning every optional encounter
        # into an additional fight.  The objective metadata and route context
        # are the policy inputs; this is not a global HP cutoff.
        if (
            wild_encounter
            and objective.encounters is EncounterPolicy.PRESERVE
            and snapshot.total_missing_hp > 0
            and not route.center_on_route
        ):
            return ResourceDecision.PREFER_RUN
        return ResourceDecision.CONTINUE

    item = _useful_item(snapshot, route)
    pc_item = next(
        (
            item
            for item in snapshot.pc_healing_items
            if item.quantity > 0 and item.heal_amount >= snapshot.total_missing_hp
        ),
        None,
    )
    if item is None and pc_item is not None and route.pc_accessible and (route.pc_acquisition_cost or 0) <= 2:
        return ResourceDecision.WITHDRAW_HEALING_ITEM
    if item is not None and not route.center_on_route and not route.safe_to_reach_center:
        return ResourceDecision.USE_HEALING_ITEM
    if route.center_available and route.safe_to_reach_center and (route.center_on_route or not healthy):
        return ResourceDecision.RECOVER_AT_CENTER
    if item is not None and not healthy:
        return ResourceDecision.USE_HEALING_ITEM
    if wild_encounter and objective.encounters is EncounterPolicy.PRESERVE:
        return ResourceDecision.PREFER_RUN
    return ResourceDecision.PRESERVE_RESOURCES


def assess_wild_encounter(
    objective: ResourceObjective, snapshot: ResourceSnapshot, route: RouteRecovery
) -> ResourceDecision:
    return assess_campaign_resources(objective, snapshot, route, wild_encounter=True)
