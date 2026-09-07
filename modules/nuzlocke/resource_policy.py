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
    RECOVER_UNDER_ATTRITION = "recover_under_attrition"
    RUN_LOST = "run_lost"


class RecoveryKind(Enum):
    """The problem a field medicine can solve."""

    HP = "hp"
    POISON = "poison"
    PARALYSIS = "paralysis"
    SLEEP = "sleep"
    BURN = "burn"
    FREEZE = "freeze"
    UNIVERSAL_STATUS = "universal_status"


def classify_recovery_item(name: str, heal_amount: int = 0) -> RecoveryKind:
    """Classify a medicine using the stable Emerald item names.

    The ROM item parameter is zero for status medicines, so it cannot be used
    as an HP/status discriminator by itself. Names are the same data already
    exposed by the item catalog and are stable across the supported RSE ROMs.
    """

    normalized = str(name).strip().casefold()
    if normalized in {"full restore", "full heal"}:
        return RecoveryKind.UNIVERSAL_STATUS
    for token, kind in (
        ("antidote", RecoveryKind.POISON),
        ("parlyz heal", RecoveryKind.PARALYSIS),
        ("paralyze heal", RecoveryKind.PARALYSIS),
        ("awakening", RecoveryKind.SLEEP),
        ("burn heal", RecoveryKind.BURN),
        ("ice heal", RecoveryKind.FREEZE),
    ):
        if normalized == token:
            return kind
    if heal_amount > 0 or normalized in {"full restore", "max potion"}:
        return RecoveryKind.HP
    # Unknown zero-parameter medicines are safer to treat as status cures than
    # as HP items. They cannot satisfy an HP restoration check accidentally.
    return RecoveryKind.UNIVERSAL_STATUS


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
class RecoverySupplyPolicy:
    """Hysteresis and cash-floor policy for early campaign medicines."""

    antidote_lower_threshold: int = 2
    antidote_target: int = 4
    potion_lower_threshold: int = 2
    potion_target: int = 5
    cash_floor: int = 200

    def __post_init__(self) -> None:
        values = (
            self.antidote_lower_threshold,
            self.antidote_target,
            self.potion_lower_threshold,
            self.potion_target,
            self.cash_floor,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("recovery supply policy values must be non-negative integers")
        if self.antidote_target <= self.antidote_lower_threshold:
            raise ValueError("Antidote target must exceed its lower threshold")
        if self.potion_target <= self.potion_lower_threshold:
            raise ValueError("Potion target must exceed its lower threshold")

    def needs_restock(self, antidotes: int, potions: int) -> bool:
        return antidotes < self.antidote_lower_threshold or potions < self.potion_lower_threshold

    def desired_quantities(self, antidotes: int, potions: int) -> dict[str, int]:
        return {
            "Antidote": max(0, self.antidote_target - antidotes),
            "Potion": max(0, self.potion_target - potions),
        }

    def affordable_quantities(
        self,
        antidotes: int,
        potions: int,
        money: int,
        prices: dict[str, int],
    ) -> dict[str, int]:
        if money < 0:
            raise ValueError("money must be non-negative")
        budget = max(0, money - self.cash_floor)
        result = {}
        for name, desired in self.desired_quantities(antidotes, potions).items():
            price = prices.get(name)
            if price is None or price < 0:
                continue
            quantity = desired if price == 0 else min(desired, budget // price)
            if quantity:
                result[name] = quantity
                budget -= quantity * price
        return result


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
    kind: RecoveryKind | None = None

    def __post_init__(self) -> None:
        if self.kind is None:
            object.__setattr__(self, "kind", classify_recovery_item(self.name, self.heal_amount))

    def cures(self, status: str | None) -> bool:
        """Return whether this item cures the supplied condition."""

        normalized = (status or "none").strip().casefold().replace(" ", "_")
        if normalized in {"none", "normal"}:
            return False
        if normalized in {"poisoned", "bad_poison", "badly_poisoned"}:
            normalized = "poison"
        if self.kind is RecoveryKind.UNIVERSAL_STATUS:
            return True
        return self.kind.value == normalized


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

    @property
    def poisoned_party(self) -> tuple[PartyResource, ...]:
        """Return living party members that will lose HP while walking."""

        poison_statuses = {"poisoned", "badly poisoned", "poison", "bad_poison"}
        return tuple(
            member
            for member in self.party
            if not member.fainted and (member.status or "none") in poison_statuses
        )

    @property
    def has_poisoned_party(self) -> bool:
        """Whether any living party member needs poison-aware recovery."""

        return bool(self.poisoned_party)

    @property
    def status_party(self) -> tuple[PartyResource, ...]:
        """Return living party members with any curable status condition."""

        return tuple(member for member in self.party if not member.fainted and member.current_hp > 0 and (member.status or "none") not in {"none", "normal"})

    @property
    def needs_status_recovery(self) -> bool:
        return bool(self.status_party)

    @property
    def bag_hp_items(self) -> tuple[HealingResource, ...]:
        return tuple(
            item
            for item in self.bag_healing_items
            if item.quantity > 0 and (item.kind is RecoveryKind.HP or item.kind is RecoveryKind.UNIVERSAL_STATUS) and item.heal_amount > 0
        )

    @property
    def bag_status_items(self) -> tuple[HealingResource, ...]:
        return tuple(item for item in self.bag_healing_items if item.quantity > 0 and item.kind is not RecoveryKind.HP)

    @property
    def pc_hp_items(self) -> tuple[HealingResource, ...]:
        return tuple(
            item
            for item in self.pc_healing_items
            if item.quantity > 0 and (item.kind is RecoveryKind.HP or item.kind is RecoveryKind.UNIVERSAL_STATUS) and item.heal_amount > 0
        )

    @property
    def pc_status_items(self) -> tuple[HealingResource, ...]:
        return tuple(item for item in self.pc_healing_items if item.quantity > 0 and item.kind is not RecoveryKind.HP)


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
    # Route validation may be running off the frame loop. This is distinct
    # from an unavailable observation: the request is valid, but no policy
    # decision may be made until its result arrives.
    calculation_pending: bool = False
    # A bounded worker may give up on a pathological route. Keep that result
    # explicit so the scheduler can retry after a cooldown instead of
    # treating the timeout as a permanent campaign truth.
    calculation_timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ResourceObjective:
    """Objective-specific resource and encounter policy metadata."""

    objective_id: str
    readiness: ReadinessImportance = ReadinessImportance.NONE
    encounters: EncounterPolicy = EncounterPolicy.NORMAL
    mandatory_battle: bool = False
    # Minimum acceptable health for an important objective. This is objective
    # metadata, not a global HP rule.
    minimum_hp_ratio: float = 0.5


def _useful_item(snapshot: ResourceSnapshot, route: RouteRecovery) -> HealingResource | None:
    """Choose the strongest bag HP item that improves the party's position."""

    candidates = [item for item in snapshot.bag_hp_items if item.heal_amount > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.heal_amount)


def _status_item(snapshot: ResourceSnapshot) -> HealingResource | None:
    """Choose the least wasteful bag item that cures an observed status."""

    for member in snapshot.status_party:
        candidates = [item for item in snapshot.bag_status_items if item.cures(member.status)]
        if candidates:
            return min(candidates, key=lambda item: (item.kind is RecoveryKind.UNIVERSAL_STATUS, item.name))
    return None


def _pc_status_item(snapshot: ResourceSnapshot) -> HealingResource | None:
    for member in snapshot.status_party:
        candidates = [item for item in snapshot.pc_status_items if item.cures(member.status)]
        if candidates:
            return min(candidates, key=lambda item: (item.kind is RecoveryKind.UNIVERSAL_STATUS, item.name))
    return None


def _pc_hp_item(snapshot: ResourceSnapshot) -> HealingResource | None:
    return max(snapshot.pc_hp_items, key=lambda item: item.heal_amount, default=None)


def _poison_bridge_is_sufficient(snapshot: ResourceSnapshot, route: RouteRecovery, item: HealingResource | None) -> bool:
    """Estimate whether one HP item buys enough steps to reach the source."""

    if item is None or route.distance_to_center is None:
        return False
    poisoned = snapshot.poisoned_party
    if not poisoned:
        return True
    # Emerald applies overworld poison damage periodically. Keep one extra HP
    # as a boundary margin because route cost is measured in inputs, not only
    # completed tile transitions.
    damage = max(1, route.distance_to_center // 4) + 1
    return min(member.current_hp + item.heal_amount for member in poisoned) > damage


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
        return ResourceDecision.RECOVER_AT_CENTER if route.center_available else ResourceDecision.RUN_LOST

    # Status damage is time-sensitive. Cure it from the bag before spending
    # turns on optional work, then use accessible PC stock before buying.
    if snapshot.needs_status_recovery:
        if _status_item(snapshot) is not None:
            return ResourceDecision.USE_HEALING_ITEM
        if _pc_status_item(snapshot) is not None and route.pc_accessible and (route.pc_acquisition_cost or 0) <= 2:
            return ResourceDecision.WITHDRAW_HEALING_ITEM

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
    pc_item = _pc_hp_item(snapshot)
    if item is None and pc_item is not None and route.pc_accessible and (route.pc_acquisition_cost or 0) <= 2:
        return ResourceDecision.WITHDRAW_HEALING_ITEM
    if item is not None and not route.center_on_route and not route.safe_to_reach_center:
        return ResourceDecision.USE_HEALING_ITEM
    if snapshot.has_poisoned_party and route.center_available and route.safe_to_reach_center and not _poison_bridge_is_sufficient(snapshot, route, item):
        # Preserve autonomous ownership even when the route may cost a life.
        return ResourceDecision.RECOVER_UNDER_ATTRITION
    if route.center_available and route.safe_to_reach_center and (route.center_on_route or not healthy):
        return ResourceDecision.RECOVER_AT_CENTER
    if item is not None and not healthy:
        return ResourceDecision.USE_HEALING_ITEM
    if wild_encounter and objective.encounters is EncounterPolicy.PRESERVE:
        return ResourceDecision.PREFER_RUN
    if snapshot.has_poisoned_party and route.observation_available:
        return ResourceDecision.RECOVER_UNDER_ATTRITION if route.center_available else ResourceDecision.RUN_LOST
    return ResourceDecision.PRESERVE_RESOURCES


def assess_wild_encounter(
    objective: ResourceObjective, snapshot: ResourceSnapshot, route: RouteRecovery
) -> ResourceDecision:
    """Assess resource policy at an optional wild-encounter boundary."""

    return assess_campaign_resources(objective, snapshot, route, wild_encounter=True)
