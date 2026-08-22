"""Read-only campaign progression readiness observations.

This module deliberately reports facts only.  It does not decide whether a
route should be continued, abandoned, or healed before.
"""

from dataclasses import dataclass, asdict, replace
from enum import Enum
from typing import Any

from .resource_policy import PartyResource, ResourceSnapshot, RouteRecovery
from .snapshots import NuzlockeSnapshot


class Availability(Enum):
    KNOWN = "known"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ReadinessDecision(Enum):
    CONTINUE = "CONTINUE"
    RECOVER = "RECOVER"
    UNKNOWN = "UNKNOWN"


class ReadinessReason(Enum):
    PARTY_INFORMATION_UNKNOWN = "PARTY_INFORMATION_UNKNOWN"
    NO_USABLE_POKEMON = "NO_USABLE_POKEMON"
    CRITICAL_PARTY_HP_BEFORE_TRAINER = "CRITICAL_PARTY_HP_BEFORE_TRAINER"
    TRAINER_HAZARD_UNKNOWN = "TRAINER_HAZARD_UNKNOWN"
    RECOVERY_UNAVAILABLE = "RECOVERY_UNAVAILABLE"
    RECOVERY_CAPABILITY_UNKNOWN = "RECOVERY_CAPABILITY_UNKNOWN"
    NO_IMMINENT_TRAINER = "NO_IMMINENT_TRAINER"
    PARTY_HEALTHY = "PARTY_HEALTHY"
    OVERWORLD_UNAVAILABLE = "OVERWORLD_UNAVAILABLE"
    RESOURCE_INFORMATION_UNKNOWN = "RESOURCE_INFORMATION_UNKNOWN"


@dataclass(frozen=True, slots=True)
class PartyReadinessMember:
    party_index: int
    species: str
    current_hp: int
    max_hp: int
    hp_ratio: float
    fainted: bool
    usable: bool
    status: str


@dataclass(frozen=True, slots=True)
class TrainerHazardObservation:
    object_id: int
    trainer_type: str | None
    trainer_range: int | None
    defeated: bool | None
    distance: int | None
    relationship: str
    path_relevant: bool | None
    approach_started: bool | None = None


@dataclass(frozen=True, slots=True)
class ProgressionReadinessDiagnostic:
    objective_id: str | None
    objective_status: str | None
    destination: Any
    navigation_goal: Any
    current_map: Any
    current_coordinates: tuple[int, int] | None
    game_state: str | None
    campaign_mode: str | None
    party: tuple[PartyReadinessMember, ...]
    has_usable_pokemon: bool | None
    healing_available: bool | None
    recovery: RouteRecovery | None
    trainers: tuple[TrainerHazardObservation, ...]
    readiness: str = "NOT_EVALUATED"
    party_availability: Availability = Availability.UNAVAILABLE
    trainer_availability: Availability = Availability.UNAVAILABLE
    recovery_availability: Availability = Availability.UNAVAILABLE
    navigation_availability: Availability = Availability.UNAVAILABLE
    runtime_availability: Availability = Availability.UNAVAILABLE
    overworld_availability: Availability = Availability.UNKNOWN
    overworld_reason: str | None = None
    resource_availability: Availability = Availability.KNOWN
    resource_reason: str | None = None
    readiness_decision: ReadinessDecision | None = None
    readiness_reason: ReadinessReason | None = None

    @property
    def party_count(self) -> int | None:
        return len(self.party) if self.party_availability is Availability.KNOWN else None

    @property
    def usable_count(self) -> int | None:
        return sum(member.usable for member in self.party) if self.party_availability is Availability.KNOWN else None

    @property
    def fainted_count(self) -> int | None:
        return sum(member.fainted for member in self.party) if self.party_availability is Availability.KNOWN else None

    @property
    def lowest_hp_ratio(self) -> float | None:
        return (
            min((member.hp_ratio for member in self.party), default=None)
            if self.party_availability is Availability.KNOWN
            else None
        )

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic, emulator-independent trace data."""
        value = asdict(self)
        value["availability"] = {
            "party": self.party_availability.value,
            "trainer": self.trainer_availability.value,
            "recovery": self.recovery_availability.value,
            "navigation": self.navigation_availability.value,
            "runtime": self.runtime_availability.value,
            "overworld": self.overworld_availability.value,
        }
        for key in (
            "party_availability",
            "trainer_availability",
            "recovery_availability",
            "navigation_availability",
            "runtime_availability",
            "overworld_availability",
        ):
            value.pop(key)
        value["navigation_goal"] = repr(self.navigation_goal)
        value["recovery"] = repr(self.recovery)
        value["readiness_decision"] = self.readiness_decision.value if self.readiness_decision else None
        value["readiness_reason"] = self.readiness_reason.value if self.readiness_reason else None
        return value


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    decision: ReadinessDecision
    reason: ReadinessReason


@dataclass(frozen=True, slots=True)
class ReadinessScheduleState:
    status: str
    age_ticks: int | None
    refresh_count: int
    tick_count: int
    invalidation_reason: str | None


class ReadinessObservationScheduler:
    """Small bounded scheduler for expensive readiness observations."""

    def __init__(self, provider, cheap_context, *, max_age_ticks: int = 15):
        self._provider = provider
        self._cheap_context = cheap_context
        self._max_age_ticks = max_age_ticks
        self._cached: ProgressionReadinessDiagnostic | None = None
        self._cached_key = None
        self._age = None
        self._tick_count = 0
        self._refresh_count = 0
        self._invalidation_reason = "initial"

    def invalidate(self, reason: str) -> None:
        self._cached = None
        self._cached_key = None
        self._age = None
        self._invalidation_reason = reason

    def observe(self, objective, goal):
        self._tick_count += 1
        cheap = self._cheap_context()
        key = (getattr(objective, "objective_id", None), repr(goal), cheap)
        can_reuse = (
            self._cached is not None
            and self._cached_key == key
            and self._age is not None
            and self._age < self._max_age_ticks
        )
        if can_reuse:
            self._age += 1
            self._invalidation_reason = None
            return self._cached
        result = self._provider(objective, goal)
        self._cached = result
        self._cached_key = key
        self._age = 0
        self._refresh_count += 1
        self._invalidation_reason = None
        if (
            getattr(result, "overworld_availability", None) is Availability.UNKNOWN
            or getattr(result, "resource_availability", None) is Availability.UNKNOWN
        ):
            self._cached = None
            self._cached_key = None
            self._age = None
            self._invalidation_reason = "observation_unavailable"
        return result

    @property
    def state(self) -> ReadinessScheduleState:
        return ReadinessScheduleState(
            (
                "fresh"
                if self._cached is not None and self._age is not None and self._age < self._max_age_ticks
                else "stale"
            ),
            self._age,
            self._refresh_count,
            self._tick_count,
            self._invalidation_reason,
        )


@dataclass(frozen=True, slots=True)
class CampaignReadinessPolicy:
    """Pure first-stage progression policy; it performs no recovery action."""

    critical_hp_ratio: float = 0.20

    def evaluate(self, readiness: ProgressionReadinessDiagnostic) -> ReadinessResult:
        if readiness.overworld_availability is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.OVERWORLD_UNAVAILABLE)
        if readiness.resource_availability is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.RESOURCE_INFORMATION_UNKNOWN)
        if readiness.party_availability is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.PARTY_INFORMATION_UNKNOWN)
        if readiness.usable_count == 0:
            return self._recovery_result(readiness, ReadinessReason.NO_USABLE_POKEMON)
        if readiness.lowest_hp_ratio is None or readiness.lowest_hp_ratio > self.critical_hp_ratio:
            return ReadinessResult(ReadinessDecision.CONTINUE, ReadinessReason.PARTY_HEALTHY)

        if readiness.trainer_availability is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.TRAINER_HAZARD_UNKNOWN)
        if not _has_imminent_trainer(readiness):
            return ReadinessResult(ReadinessDecision.CONTINUE, ReadinessReason.NO_IMMINENT_TRAINER)
        return self._recovery_result(readiness, ReadinessReason.CRITICAL_PARTY_HP_BEFORE_TRAINER)

    @staticmethod
    def _recovery_result(readiness: ProgressionReadinessDiagnostic, reason: ReadinessReason) -> ReadinessResult:
        if readiness.recovery_availability is Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.RECOVER, reason)
        if readiness.recovery_availability is Availability.UNAVAILABLE:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.RECOVERY_UNAVAILABLE)
        return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN)


def _has_imminent_trainer(readiness: ProgressionReadinessDiagnostic) -> bool:
    """Require direct same-map range evidence; never infer from map presence."""
    for trainer in readiness.trainers:
        if trainer.defeated is True or trainer.defeated is None:
            continue
        if trainer.trainer_range is None or trainer.distance is None:
            continue
        if trainer.distance <= trainer.trainer_range and trainer.relationship == "nearby":
            return True
    return False


def _party(snapshot: NuzlockeSnapshot) -> tuple[PartyReadinessMember, ...]:
    if not snapshot.party_available:
        return ()
    return tuple(
        PartyReadinessMember(
            p.party_index,
            p.species,
            p.current_hp,
            p.max_hp,
            p.current_hp / p.max_hp if p.max_hp > 0 else 0.0,
            p.fainted,
            not p.fainted and p.current_hp > 0,
            p.status,
        )
        for p in snapshot.party
    )


def build_progression_readiness_diagnostic(
    snapshot: NuzlockeSnapshot,
    *,
    objective_id: str | None = None,
    objective_status: str | None = None,
    destination: Any = None,
    navigation_goal: Any = None,
    campaign_mode: str | None = None,
    overworld: Any = None,
    resource_snapshot: ResourceSnapshot | None = None,
    recovery: RouteRecovery | None = None,
    recovery_availability: Availability | None = None,
    overworld_availability: Availability = Availability.UNKNOWN,
    overworld_reason: str | None = None,
    resource_availability: Availability | None = None,
    resource_reason: str | None = None,
) -> ProgressionReadinessDiagnostic:
    """Build a point-in-time diagnostic without applying a survival policy."""
    members = _party(snapshot)
    party_availability = Availability.KNOWN if snapshot.party_available else Availability.UNAVAILABLE
    usable = None if not snapshot.party_available else any(p.usable for p in members)
    healing = (
        None
        if resource_snapshot is None
        else bool(resource_snapshot.bag_healing_items or resource_snapshot.pc_healing_items)
    )
    trainers: list[TrainerHazardObservation] = []
    if overworld is not None:
        player = overworld.player_coordinates
        for obj in overworld.objects:
            if obj.trainer_type in (None, "None"):
                continue
            distance = None
            relationship = "map_present"
            if obj.location[0] == overworld.map_id:
                distance = abs(player[0] - obj.location[1][0]) + abs(player[1] - obj.location[1][1])
                relationship = "nearby" if distance <= (obj.trainer_range or 0) else "same_map"
            trainers.append(
                TrainerHazardObservation(
                    obj.local_id,
                    obj.trainer_type,
                    obj.trainer_range,
                    obj.trainer_defeated,
                    distance,
                    relationship,
                    None,
                    None,
                )
            )
    trainer_availability = Availability.KNOWN if overworld is not None else Availability.UNAVAILABLE
    navigation_availability = (
        Availability.KNOWN
        if overworld is not None and navigation_goal is not None
        else (Availability.NOT_APPLICABLE if navigation_goal is None else Availability.UNAVAILABLE)
    )
    return ProgressionReadinessDiagnostic(
        objective_id,
        objective_status,
        destination,
        navigation_goal,
        (overworld.map_id if overworld is not None else (snapshot.player.map_group, snapshot.player.map_number)),
        (overworld.player_coordinates if overworld is not None else snapshot.player.coordinates),
        getattr(snapshot.game_state, "name", None),
        campaign_mode,
        members,
        usable,
        healing,
        recovery,
        tuple(trainers),
        "NOT_EVALUATED",
        party_availability,
        trainer_availability,
        recovery_availability or (Availability.KNOWN if recovery is not None else Availability.UNAVAILABLE),
        navigation_availability,
        Availability.KNOWN if snapshot.player_available else Availability.UNAVAILABLE,
        overworld_availability,
        overworld_reason,
        (
            resource_availability
            if resource_availability is not None
            else (
                Availability.UNKNOWN
                if resource_snapshot is not None and resource_snapshot.observation_status.value != "valid"
                else Availability.KNOWN
            )
        ),
        resource_reason,
    )


def evaluate_progression_readiness(
    readiness: ProgressionReadinessDiagnostic,
    policy: CampaignReadinessPolicy | None = None,
) -> ProgressionReadinessDiagnostic:
    """Evaluate and return a diagnostic enriched with the pure policy result."""
    result = (policy or CampaignReadinessPolicy()).evaluate(readiness)
    return replace(readiness, readiness_decision=result.decision, readiness_reason=result.reason)
