"""Objective-aware campaign resource policy.

This module is deliberately pure: it contains no emulator reads and never
executes navigation, battle, item, or healing actions.
"""

from dataclasses import dataclass
from enum import Enum
from math import ceil
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


class WildBattleIntent(str, Enum):
    """Campaign intent supplied to the in-battle expectiminimax planner."""

    AUTO = "auto"
    CAPTURE = "capture"
    TRAIN = "training"
    AVOID = "avoid"
    TRANSIT = "transit"
    DEFEAT = "defeat"


class DuplicateEncounterPolicy(str, Enum):
    """Policy for a duplicate wild battle that cannot be captured."""

    RUN = "run"
    TRAIN = "train"
    DEFEAT = "defeat"


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


@dataclass(frozen=True, slots=True)
class RecoveryEffect:
    """The effects a medicine has on one living party member.

    ``hp_amount`` is a fixed amount.  Full HP medicines use
    ``heals_to_full`` instead of pretending that the ROM parameter ``255`` is
    a literal number of HP.  Status coverage is kept independently so a
    targeted cure can be preferred over a broad cure with incidental HP
    restoration.
    """

    hp_amount: int = 0
    heals_to_full: bool = False
    status_cures: frozenset[str] = frozenset()
    cures_all_statuses: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.hp_amount, bool) or self.hp_amount < 0:
            raise ValueError("recovery HP amount must be a non-negative integer")
        if self.heals_to_full and self.hp_amount:
            raise ValueError("a full HP effect cannot also have a fixed HP amount")
        normalized = frozenset(_canonical_status(status) for status in self.status_cures)
        object.__setattr__(self, "status_cures", normalized - {"none", "normal"})


def classify_recovery_effect(name: str, heal_amount: int = 0) -> RecoveryEffect:
    """Translate the stable Emerald medicine names into explicit effects."""

    normalized = str(name).strip().casefold()
    if normalized == "full restore":
        return RecoveryEffect(heals_to_full=True, cures_all_statuses=True)
    if normalized == "max potion":
        return RecoveryEffect(heals_to_full=True)
    if normalized == "full heal":
        return RecoveryEffect(cures_all_statuses=True)
    status_by_name = {
        "antidote": "poison",
        "parlyz heal": "paralysis",
        "paralyze heal": "paralysis",
        "awakening": "sleep",
        "burn heal": "burn",
        "ice heal": "freeze",
    }
    status = status_by_name.get(normalized)
    if status is not None:
        return RecoveryEffect(status_cures=frozenset({status}))
    if heal_amount > 0:
        return RecoveryEffect(hp_amount=heal_amount)
    # Unknown zero-parameter medicines are safer to treat as broad status
    # cures than as HP items. They cannot satisfy an HP restoration check.
    return RecoveryEffect(cures_all_statuses=True)


def _canonical_status(status: str | None) -> str:
    """Normalize ROM status names to the recovery-kind vocabulary."""

    normalized = getattr(status, "value", status)
    normalized = str(normalized or "none").strip().casefold().replace(" ", "_")
    return {
        "poisoned": "poison",
        "bad_poison": "poison",
        "badly_poisoned": "poison",
        "paralyzed": "paralysis",
        "paralysed": "paralysis",
        "asleep": "sleep",
        "sleeping": "sleep",
        "burned": "burn",
        "burnt": "burn",
        "frozen": "freeze",
    }.get(normalized, normalized)


def classify_recovery_item(name: str, heal_amount: int = 0) -> RecoveryKind:
    """Classify a medicine using the stable Emerald item names.

    The ROM item parameter is zero for status medicines, so it cannot be used
    as an HP/status discriminator by itself. Names are the same data already
    exposed by the item catalog and are stable across the supported RSE ROMs.
    """

    effect = classify_recovery_effect(name, heal_amount)
    if effect.cures_all_statuses:
        return RecoveryKind.UNIVERSAL_STATUS
    status_kind = {
        "poison": RecoveryKind.POISON,
        "paralysis": RecoveryKind.PARALYSIS,
        "sleep": RecoveryKind.SLEEP,
        "burn": RecoveryKind.BURN,
        "freeze": RecoveryKind.FREEZE,
    }
    if len(effect.status_cures) == 1 and not effect.heals_to_full and not effect.hp_amount:
        return status_kind[next(iter(effect.status_cures))]
    if effect.hp_amount or effect.heals_to_full:
        return RecoveryKind.HP
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
    effect: RecoveryEffect | None = None

    def __post_init__(self) -> None:
        if self.effect is None:
            object.__setattr__(self, "effect", classify_recovery_effect(self.name, self.heal_amount))
        if self.kind is None:
            object.__setattr__(self, "kind", classify_recovery_item(self.name, self.heal_amount))

    @property
    def amount_healed(self) -> int:
        """Return fixed HP restored; zero denotes a full-heal effect."""

        return self.effect.hp_amount

    @property
    def status_healed(self) -> frozenset[str]:
        """Return targeted statuses covered by this item."""

        return self.effect.status_cures

    @property
    def heals_to_full(self) -> bool:
        return self.effect.heals_to_full

    @property
    def cures_all_statuses(self) -> bool:
        return self.effect.cures_all_statuses

    @property
    def has_hp_effect(self) -> bool:
        return self.amount_healed > 0 or self.heals_to_full

    @property
    def has_status_effect(self) -> bool:
        return bool(self.status_healed) or self.cures_all_statuses

    @property
    def reserve_tier(self) -> int:
        """Return the medicine tier to preserve when a smaller item suffices."""

        return {
            "potion": 0,
            "super potion": 1,
            "hyper potion": 2,
            "max potion": 3,
            "full restore": 4,
        }.get(str(self.name).strip().casefold(), 5)

    def hp_restored_for(self, current_hp: int, max_hp: int) -> int:
        """Return the actual HP this item would restore for a target."""

        missing = max(0, max_hp - current_hp)
        if self.heals_to_full:
            return missing
        return min(self.amount_healed, missing)

    def cures(self, status: str | None) -> bool:
        """Return whether this item cures the supplied condition."""

        normalized = _canonical_status(status)
        if normalized in {"none", "normal"}:
            return False
        if self.cures_all_statuses:
            return True
        return normalized in self.status_healed


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

        return tuple(
            member
            for member in self.party
            if not member.fainted and _canonical_status(member.status) == "poison"
        )

    @property
    def has_poisoned_party(self) -> bool:
        """Whether any living party member needs poison-aware recovery."""

        return bool(self.poisoned_party)

    @property
    def status_party(self) -> tuple[PartyResource, ...]:
        """Return living party members with any curable status condition."""

        return tuple(
            member
            for member in self.party
            if not member.fainted
            and member.current_hp > 0
            and _canonical_status(member.status) not in {"none", "normal"}
        )

    @property
    def needs_status_recovery(self) -> bool:
        return bool(self.status_party)

    def needs_status_recovery_at(self, hp_ratio: float) -> bool:
        """Return whether status recovery is urgent at the supplied HP floor.

        Poison is special because it continues to damage a living Pokémon in
        the overworld. Other status conditions raise the recovery floor for a
        weakened party member, but do not force an immediate detour while the
        affected member is still comfortably healthy.
        """

        if self.has_poisoned_party:
            return True
        return self.needs_status_recovery and self.worst_hp_ratio < hp_ratio

    @property
    def bag_hp_items(self) -> tuple[HealingResource, ...]:
        return tuple(
            item
            for item in self.bag_healing_items
            if item.quantity > 0 and item.has_hp_effect
        )

    @property
    def bag_status_items(self) -> tuple[HealingResource, ...]:
        return tuple(item for item in self.bag_healing_items if item.quantity > 0 and item.has_status_effect)

    @property
    def pc_hp_items(self) -> tuple[HealingResource, ...]:
        return tuple(
            item
            for item in self.pc_healing_items
            if item.quantity > 0 and item.has_hp_effect
        )

    @property
    def pc_status_items(self) -> tuple[HealingResource, ...]:
        return tuple(item for item in self.pc_healing_items if item.quantity > 0 and item.has_status_effect)


@dataclass(frozen=True, slots=True)
class RecoveryCandidateDiagnostic:
    """Outcome of one bounded recovery destination search."""

    rank: int
    source_id: str | None
    destination: object | None
    outcome: str
    distance: int | float | None = None
    encounter_opportunities: int | None = None
    reason: str = ""


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
    # Exact map/coordinate pair from which ``route`` was planned. A route is
    # only executable from this origin; the controller must discard it after
    # movement changes the starting tile.
    route_origin: Any | None = None
    # The exact semantic target used to produce ``route``.  A Pokémon Center
    # door is represented by its outdoor destination for compatibility, but
    # executable routing must target the cataloged interior warp.  Retaining
    # this goal lets readiness compose the same route without converting a
    # blocked door tile into a false ReachLocation target.
    navigation_goal: Any | None = None
    # The exact catalog source selected by route analysis. Keeping this
    # object avoids reverse-mapping a destination during execution.
    healing_source: Any | None = None
    encounter_opportunities: int | None = None
    # Route validation may be running off the frame loop. This is distinct
    # from an unavailable observation: the request is valid, but no policy
    # decision may be made until its result arrives.
    calculation_pending: bool = False
    # A bounded worker may give up on a pathological route. Keep that result
    # explicit so the scheduler can retry after a cooldown instead of
    # treating the timeout as a permanent campaign truth.
    calculation_timed_out: bool = False
    candidate_diagnostics: tuple[RecoveryCandidateDiagnostic, ...] = ()


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
    # Status conditions other than poison use a higher health floor because
    # they reduce battle reliability without draining HP during overworld
    # movement. Boss preparation may raise this through objective metadata.
    status_hp_ratio: float = 0.5
    # ``AUTO`` lets the campaign boundary choose from capture, training, and
    # avoidance facts.  The battle planner receives the resulting intent;
    # it never infers campaign purpose from HP alone.
    wild_battle_intent: WildBattleIntent = WildBattleIntent.AUTO
    duplicate_encounter_policy: DuplicateEncounterPolicy = DuplicateEncounterPolicy.RUN


@dataclass(frozen=True, slots=True)
class WildBattlePolicyInput:
    """Facts needed to assign intent to one wild battle."""

    legal_capture_target: bool = False
    duplicate_encounter: bool = False
    preparation_requested: bool = False
    transit_requested: bool = False
    policy_intent: WildBattleIntent = WildBattleIntent.AUTO
    duplicate_policy: DuplicateEncounterPolicy = DuplicateEncounterPolicy.RUN


def choose_wild_battle_intent(value: WildBattlePolicyInput) -> WildBattleIntent:
    """Choose a campaign purpose without choosing the battle turn action.

    Encounter legality and duplicate semantics are supplied by the Nuzlocke
    runtime.  This function only maps those facts to an objective that the
    normalized battle planner can evaluate.
    """

    if value.legal_capture_target:
        return WildBattleIntent.CAPTURE
    if value.policy_intent is not WildBattleIntent.AUTO:
        return value.policy_intent
    if value.duplicate_encounter:
        if value.duplicate_policy is DuplicateEncounterPolicy.TRAIN:
            return WildBattleIntent.TRAIN
        if value.duplicate_policy is DuplicateEncounterPolicy.DEFEAT:
            return WildBattleIntent.DEFEAT
        return WildBattleIntent.AVOID
    if value.preparation_requested:
        return WildBattleIntent.TRAIN
    if value.transit_requested:
        return WildBattleIntent.TRANSIT
    return WildBattleIntent.AVOID


def best_hp_recovery_item(
    snapshot: ResourceSnapshot,
    *,
    minimum_hp_ratio: float = 0.5,
) -> HealingResource | None:
    """Choose the lowest-tier HP medicine that reaches the safety floor.

    Full HP medicines are represented explicitly, so they do not win merely
    because the ROM exposes ``255`` as their parameter.  A lower-tier item is
    preferred whenever it can restore the weakest member to the requested
    floor.  If no item reaches that floor, the best available partial recovery
    remains a valid fallback.
    """

    if not 0 <= minimum_hp_ratio <= 1:
        raise ValueError("minimum_hp_ratio must be between 0 and 1")
    candidates = tuple(item for item in snapshot.bag_hp_items if item.has_hp_effect)
    if not candidates:
        return None
    target = min(
        snapshot.usable_party,
        key=lambda member: member.hp_ratio,
        default=None,
    )
    if target is None:
        return min(candidates, key=lambda item: (item.reserve_tier, -item.amount_healed))

    missing_hp = max(0, target.max_hp - target.current_hp)
    required_hp = max(0, ceil(target.max_hp * minimum_hp_ratio) - target.current_hp)
    sufficient = tuple(
        item
        for item in candidates
        if item.hp_restored_for(target.current_hp, target.max_hp) >= required_hp
    )
    if sufficient:
        # Reserve tier is the primary choice. Waste is only a tie-breaker
        # within that tier, which keeps a Hyper Potion ahead of a Max Potion
        # when both reach the same safety floor.
        return min(
            sufficient,
            key=lambda item: (
                item.reserve_tier,
                max(0, item.hp_restored_for(target.current_hp, target.max_hp) - required_hp),
                -item.hp_restored_for(target.current_hp, target.max_hp),
            ),
        )

    # No medicine reaches the floor. Use the item that restores the most HP,
    # then preserve the lower tier when two fallbacks are equally effective.
    return max(
        candidates,
        key=lambda item: (
            item.hp_restored_for(target.current_hp, target.max_hp),
            -item.reserve_tier,
            -max(0, item.amount_healed - missing_hp),
        ),
    )


def _useful_item(
    snapshot: ResourceSnapshot,
    route: RouteRecovery,
    minimum_hp_ratio: float = 0.5,
) -> HealingResource | None:
    """Choose the HP medicine with the best effective-healing score."""

    return best_hp_recovery_item(snapshot, minimum_hp_ratio=minimum_hp_ratio)


def _status_item(snapshot: ResourceSnapshot) -> HealingResource | None:
    """Choose the least wasteful bag item that cures an observed status."""

    for member in snapshot.status_party:
        candidates = [item for item in snapshot.bag_status_items if item.cures(member.status)]
        if candidates:
            return min(candidates, key=lambda item: (item.cures_all_statuses, item.reserve_tier, item.name))
    return None


def _pc_status_item(snapshot: ResourceSnapshot) -> HealingResource | None:
    for member in snapshot.status_party:
        candidates = [item for item in snapshot.pc_status_items if item.cures(member.status)]
        if candidates:
            return min(candidates, key=lambda item: (item.cures_all_statuses, item.reserve_tier, item.name))
    return None


def _pc_hp_item(snapshot: ResourceSnapshot, minimum_hp_ratio: float = 0.5) -> HealingResource | None:
    pc_snapshot = ResourceSnapshot(
        party=snapshot.party,
        bag_healing_items=snapshot.pc_healing_items,
    )
    return best_hp_recovery_item(pc_snapshot, minimum_hp_ratio=minimum_hp_ratio)


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
    return min(
        member.current_hp + item.hp_restored_for(member.current_hp, member.max_hp)
        for member in poisoned
    ) > damage


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

    # Poison damage is time-sensitive. Other statuses matter once the party
    # falls below their higher status floor. Cure an urgent condition from the
    # bag before spending turns on optional work, then use accessible PC stock
    # before buying.
    if snapshot.needs_status_recovery_at(objective.status_hp_ratio):
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

    item = _useful_item(snapshot, route, objective.minimum_hp_ratio)
    pc_item = _pc_hp_item(snapshot, objective.minimum_hp_ratio)
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
