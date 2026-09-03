"""Objective-aware campaign resource policy.

This module is deliberately pure: it contains no emulator reads and never
executes navigation, battle, item, or healing actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class ReadinessImportance(Enum):
    """How strongly an objective's resource condition should be enforced."""

    NONE = "none"
    NORMAL = "normal"
    IMPORTANT = "important"


class EncounterPolicy(Enum):
    """Resource posture to use while resolving optional wild encounters."""

    NORMAL = "normal"
    PRESERVE = "preserve"
    GRIND = "grind"


class ResourceDecision(Enum):
    """Pure policy result describing the next resource-related action."""

    CONTINUE = "continue"
    PRESERVE_RESOURCES = "preserve_resources"
    PREFER_RUN = "prefer_run"
    RECOVER_AT_CENTER = "recover_at_center"
    USE_HEALING_ITEM = "use_healing_item"
    WITHDRAW_HEALING_ITEM = "withdraw_healing_item"


class ResourceObservationStatus(Enum):
    """Validity state for a resource observation."""

    VALID = "valid"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class PokeballRestockPolicy:
    """Hysteresis policy for the autonomous Poké Ball reserve.

    The lower threshold controls whether a shopping detour is warranted;
    the upper target controls how many balls the detour should buy.  Keeping
    these values separate prevents the campaign from repeatedly returning to
    a shop after every small expenditure.
    """

    lower_threshold: int = 5
    upper_target: int = 10

    def __post_init__(self) -> None:
        """Reject a threshold pair that cannot express hysteresis."""

        if isinstance(self.lower_threshold, bool) or not isinstance(self.lower_threshold, int):
            raise ValueError("Poké Ball lower threshold must be an integer")
        if isinstance(self.upper_target, bool) or not isinstance(self.upper_target, int):
            raise ValueError("Poké Ball upper target must be an integer")
        if self.lower_threshold < 0:
            raise ValueError("Poké Ball lower threshold must be non-negative")
        if self.upper_target <= self.lower_threshold:
            raise ValueError("Poké Ball upper target must be greater than the lower threshold")

    def needs_restock(self, count: int) -> bool:
        """Return whether ``count`` is strictly below the lower threshold."""

        return count < self.lower_threshold

    def quantity_to_buy(self, count: int) -> int:
        """Return the number needed to reach the upper target."""

        return max(0, self.upper_target - count)

    def affordable_quantity(self, count: int, money: int, unit_price: int) -> int:
        """Cap the desired purchase at what the observed wallet can afford."""

        if money < 0:
            raise ValueError("money must be non-negative")
        if unit_price < 0:
            raise ValueError("Poké Ball price must be non-negative")
        desired = self.quantity_to_buy(count)
        if unit_price == 0:
            return desired
        return min(desired, money // unit_price)


@dataclass(frozen=True, slots=True)
class PartyResource:
    """Observed health and status information for one party member."""

    current_hp: int
    max_hp: int
    status: str | None = None
    fainted: bool = False

    @property
    def hp_ratio(self) -> float:
        """Return current HP as a safe ratio of maximum HP."""

        return self.current_hp / self.max_hp if self.max_hp > 0 else 0.0


@dataclass(frozen=True, slots=True)
class HealingResource:
    """A healing item with its observed quantity, strength, and location."""

    name: str
    quantity: int
    heal_amount: int
    location: str = "bag"
    acquisition_cost: int = 0


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Immutable party, item, and observation-quality inputs to resource policy."""

    party: tuple[PartyResource, ...] = ()
    bag_healing_items: tuple[HealingResource, ...] = ()
    pc_healing_items: tuple[HealingResource, ...] = ()
    current_map: object | None = None
    observation_status: ResourceObservationStatus = ResourceObservationStatus.VALID
    invalid_item_index: int | None = None
    invalid_item_slot: int | None = None
    invalid_item_storage: str | None = None
    observation_error: str | None = None
    unavailable_components: tuple[str, ...] = ()

    @property
    def usable_party(self) -> tuple[PartyResource, ...]:
        """Return non-fainted party members with positive HP."""

        return tuple(p for p in self.party if not p.fainted and p.current_hp > 0)

    @property
    def worst_hp_ratio(self) -> float:
        """Return the lowest HP ratio among usable party members."""

        usable = self.usable_party
        return min((p.hp_ratio for p in usable), default=0.0)

    @property
    def total_missing_hp(self) -> int:
        """Return total missing HP across usable party members."""

        return sum(max(0, p.max_hp - p.current_hp) for p in self.usable_party)


@dataclass(frozen=True, slots=True)
class RouteRecovery:
    """Known route affordances for reaching a healing source or PC."""

    center_available: bool = False
    # The concrete outdoor destination selected by route recovery.  Keeping
    # this alongside the distance prevents readiness analysis from reopening
    # the entire global healing catalog just to compare one already-selected
    # recovery route.
    center_location: object | None = None
    distance_to_center: int | None = None
    center_on_route: bool = False
    safe_to_reach_center: bool = True
    distance_to_objective: int | None = None
    pc_accessible: bool = False
    pc_acquisition_cost: int | None = None
    # False means the route observation itself was unavailable (for example
    # during a map transition), distinct from a known route with no center.
    observation_available: bool = True
    observation_error: str | None = None
    # A ROM may provide full-party healing without a Pokémon Center (for
    # example the player's mother or a rest stop).  Keep the historical
    # Center fields for compatibility, but expose the broader capability.
    healing_source_available: bool = False
    # An executable route may be retained from the same observation that
    # measured the recovery distance. CampaignPlan can adopt it directly so
    # recovery does not run a second synchronous pathfinding search.
    route: Any | None = None
    # The exact semantic target used to produce ``route``.  A Pokémon Center
    # door is represented by its outdoor destination for compatibility, but
    # executable routing must target the cataloged interior warp.  Retaining
    # this goal lets readiness compose the same route without converting a
    # blocked door tile into a false ReachLocation target.
    navigation_goal: Any | None = None


@dataclass(frozen=True, slots=True)
class ResourceObjective:
    """Objective-specific resource and encounter policy metadata."""

    objective_id: str
    readiness: ReadinessImportance = ReadinessImportance.NONE
    encounters: EncounterPolicy = EncounterPolicy.NORMAL
    mandatory_battle: bool = False
    recover_before_completion: bool = False
    # Minimum acceptable health for an important objective. This is objective
    # metadata, not a global HP rule.
    minimum_hp_ratio: float = 0.5


def _useful_item(snapshot: ResourceSnapshot, route: RouteRecovery) -> HealingResource | None:
    """Choose the strongest bag item that can cover current missing HP."""

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

    # A recover-before-completion objective must use an already-known Center
    # opportunity before the important battle, even when the party is still
    # above the ordinary minimum-health threshold.  Waiting until HP becomes
    # critical defeats the purpose of the route-level recovery affordance.
    if (
        objective.recover_before_completion
        and not wild_encounter
        and route.center_available
        and route.center_on_route
        and snapshot.total_missing_hp > 0
    ):
        return ResourceDecision.RECOVER_AT_CENTER

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
    """Assess resource policy at an optional wild-encounter boundary."""

    return assess_campaign_resources(objective, snapshot, route, wild_encounter=True)
