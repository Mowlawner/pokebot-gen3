"""Read-only campaign progression readiness observations.

This module deliberately reports facts only.  It does not decide whether a
route should be continued, abandoned, or healed before.
"""

from dataclasses import dataclass, asdict, replace
from enum import Enum
from typing import Any

from .resource_policy import PartyResource, ResourceObservationStatus, ResourceSnapshot, RouteRecovery
from .snapshots import NuzlockeSnapshot
from modules.console import diagnostic_print
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.navigation import RouteAnalysis


class Availability(Enum):
    """Availability state for one readiness observation component."""

    KNOWN = "known"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ReadinessDecision(Enum):
    """Pure first-stage decision for continuing or recovering."""

    CONTINUE = "CONTINUE"
    RECOVER = "RECOVER"
    UNKNOWN = "UNKNOWN"


class ReadinessReason(Enum):
    """Diagnostic reason associated with a readiness decision."""

    PARTY_INFORMATION_UNKNOWN = "PARTY_INFORMATION_UNKNOWN"
    NO_USABLE_POKEMON = "NO_USABLE_POKEMON"
    CRITICAL_PARTY_HP = "CRITICAL_PARTY_HP"
    OPPORTUNISTIC_RECOVERY = "OPPORTUNISTIC_RECOVERY"
    TRAINER_HAZARD_UNKNOWN = "TRAINER_HAZARD_UNKNOWN"
    RECOVERY_UNAVAILABLE = "RECOVERY_UNAVAILABLE"
    RECOVERY_CAPABILITY_UNKNOWN = "RECOVERY_CAPABILITY_UNKNOWN"
    NO_IMMINENT_TRAINER = "NO_IMMINENT_TRAINER"
    PARTY_HEALTHY = "PARTY_HEALTHY"
    OVERWORLD_UNAVAILABLE = "OVERWORLD_UNAVAILABLE"
    RESOURCE_INFORMATION_UNKNOWN = "RESOURCE_INFORMATION_UNKNOWN"
    OPPORTUNISTIC_ROUTE_UNAVAILABLE = "OPPORTUNISTIC_ROUTE_UNAVAILABLE"
    OPPORTUNISTIC_ROUTE_NOT_WITHIN_THRESHOLD = "OPPORTUNISTIC_ROUTE_NOT_WITHIN_THRESHOLD"


@dataclass(frozen=True, slots=True)
class PartyReadinessMember:
    """Observed health and usability facts for one party member."""

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
    """Observed trainer threat and route relationship for readiness analysis."""

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
    """Immutable evidence bundle used by the progression readiness policy."""

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
    route_analysis: "RouteAnalysis | None" = None

    @property
    def party_count(self) -> int | None:
        """Return the observed party size when party data is available."""

        return len(self.party) if self.party_availability is Availability.KNOWN else None

    @property
    def usable_count(self) -> int | None:
        """Return the number of observed usable party members."""

        return sum(member.usable for member in self.party) if self.party_availability is Availability.KNOWN else None

    @property
    def fainted_count(self) -> int | None:
        """Return the number of observed fainted party members."""

        return sum(member.fainted for member in self.party) if self.party_availability is Availability.KNOWN else None

    @property
    def lowest_hp_ratio(self) -> float | None:
        """Return the lowest HP ratio among usable observed members."""

        # Fainted party members are already represented by ``fainted_count``
        # and must not make the usable party appear critically injured.  A
        # fainted member has an HP ratio of zero, but cannot be healed or used
        # as the active readiness constraint until recovery is selected.
        usable = tuple(member for member in self.party if member.usable)
        return (
            min((member.hp_ratio for member in usable), default=None)
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
    """Pure readiness decision paired with its diagnostic reason."""

    decision: ReadinessDecision
    reason: ReadinessReason


@dataclass(frozen=True, slots=True)
class ReadinessScheduleState:
    """Observable counters and cache state for readiness scheduling."""

    status: str
    age_ticks: int | None
    refresh_count: int
    tick_count: int
    invalidation_reason: str | None


class ReadinessObservationScheduler:
    """Small bounded scheduler for expensive readiness observations."""

    def __init__(self, provider, cheap_context, *, max_age_ticks: int = 15):
        """Create a bounded scheduler around expensive readiness observations."""

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
        """Discard cached readiness and record why it became invalid."""

        self._cached = None
        self._cached_key = None
        self._age = None
        self._invalidation_reason = reason

    def observe(self, objective, goal):
        """Reuse or refresh readiness based on a cheap context key and age."""

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
            diagnostic_print(
                lambda: f"READINESS_SCHEDULER: cache=hit tick={self._tick_count} objective={getattr(objective, 'objective_id', None)!r}",
                trace=True,
            )
            return self._cached
        diagnostic_print(
            lambda: f"READINESS_SCHEDULER: cache=miss tick={self._tick_count} objective={getattr(objective, 'objective_id', None)!r}",
            trace=True,
        )
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
        """Return immutable scheduler counters and freshness state."""

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

    critical_hp_ratio: float = 0.25
    # Reuse the existing important-objective minimum-health concept for
    # opportunistic recovery, but only when a nearby Center makes recovery a
    # small tactical detour rather than a general healing mandate.
    opportunistic_hp_ratio: float = 0.75
    opportunistic_detour_threshold: int = 50

    def evaluate(self, readiness: ProgressionReadinessDiagnostic) -> ReadinessResult:
        """Choose continue, recover, or unknown from observed readiness facts."""

        evaluation_id = id(readiness)
        recovery = getattr(readiness, "recovery", None)
        lowest_hp_ratio = getattr(readiness, "lowest_hp_ratio", None)
        critical = lowest_hp_ratio is not None and lowest_hp_ratio < self.critical_hp_ratio
        opportunistic_threshold = lowest_hp_ratio is not None and lowest_hp_ratio <= self.opportunistic_hp_ratio
        recovery_known = getattr(readiness, "recovery_availability", Availability.UNKNOWN) is Availability.KNOWN
        center_available = bool(recovery and recovery.center_available)
        center_safe = bool(recovery and recovery.safe_to_reach_center)
        imminent = _has_imminent_trainer(readiness)
        diagnostic_print(
            lambda: (
                "READINESS_POLICY_BEFORE: "
                f"evaluation_id={evaluation_id} frame={getattr(readiness, 'frame', None)!r} "
                f"map={getattr(readiness, 'current_map', None)!r} location={getattr(readiness, 'current_coordinates', None)!r} "
                f"lowest_hp_ratio={lowest_hp_ratio!r} critical_threshold={self.critical_hp_ratio!r} "
                f"opportunistic_threshold={self.opportunistic_hp_ratio!r} critical={critical} "
                f"imminent_trainer={imminent} recovery_availability_known={recovery_known} "
                f"center_available={center_available} center_safe={center_safe} "
                f"route_analysis_available={getattr(readiness, 'route_analysis', None) is not None} "
                f"opportunistic_detour_threshold={self.opportunistic_detour_threshold!r}"
            ),
            trace=True,
        )
        if getattr(readiness, "overworld_availability", Availability.UNKNOWN) is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.OVERWORLD_UNAVAILABLE)
        # Coordinates can remain readable briefly while the emulator crosses
        # a battle or script boundary. Recovery requires authoritative stable
        # overworld state before it may take ownership.
        if getattr(readiness, "game_state", None) != "OVERWORLD":
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.OVERWORLD_UNAVAILABLE)
        if getattr(readiness, "resource_availability", Availability.UNKNOWN) is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.RESOURCE_INFORMATION_UNKNOWN)
        if getattr(readiness, "party_availability", Availability.UNKNOWN) is not Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.PARTY_INFORMATION_UNKNOWN)
        if getattr(readiness, "usable_count", None) == 0:
            return self._recovery_result(readiness, ReadinessReason.NO_USABLE_POKEMON)
        if getattr(readiness, "lowest_hp_ratio", None) is None:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.PARTY_INFORMATION_UNKNOWN)

        critical = readiness.lowest_hp_ratio < self.critical_hp_ratio
        if critical:
            return self._recovery_result(readiness, ReadinessReason.CRITICAL_PARTY_HP)
        # An imminent trainer interaction is part of the active campaign
        # objective.  Do not divert for optional healing immediately before
        # it; critical HP recovery above remains authoritative.
        if not imminent:
            if readiness.route_analysis is None or readiness.route_analysis.normal_cost is None:
                if readiness.lowest_hp_ratio <= self.opportunistic_hp_ratio:
                    # Some observation-driven objectives intentionally have
                    # no declarative campaign goal.  At that boundary the
                    # recovery route is still authoritative; requiring a
                    # normal-route comparison would strand the controller
                    # after a battle with a known nearby Center.
                    if (
                        readiness.objective_id == "receive_pokedex"
                        and center_available
                        and center_safe
                        and readiness.recovery.distance_to_center is not None
                        and readiness.recovery.distance_to_center <= self.opportunistic_detour_threshold
                    ):
                        return self._recovery_result(readiness, ReadinessReason.OPPORTUNISTIC_RECOVERY)
                    return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.OPPORTUNISTIC_ROUTE_UNAVAILABLE)
            elif self._opportunistic_recovery_available(readiness):
                return self._recovery_result(readiness, ReadinessReason.OPPORTUNISTIC_RECOVERY)
        if not critical:
            return ReadinessResult(ReadinessDecision.CONTINUE, ReadinessReason.PARTY_HEALTHY)

    def _opportunistic_recovery_available(self, readiness: ProgressionReadinessDiagnostic) -> bool:
        """Return whether a safe nearby healing route meets policy thresholds."""

        if readiness.lowest_hp_ratio is None or readiness.lowest_hp_ratio > self.opportunistic_hp_ratio:
            return False
        analysis = readiness.route_analysis
        if analysis is None or analysis.normal_cost is None:
            diagnostic_print(
                "OPPORTUNISTIC_HEAL_ANALYSIS: route_analysis_unavailable decision=CONTINUE",
                trace=True,
            )
            return False
        candidates = tuple(
            candidate for candidate in analysis.candidates if candidate.reachable and candidate.detour is not None
        )
        selected = (
            min(candidates, key=lambda candidate: (candidate.detour, candidate.total_cost)) if candidates else None
        )
        decision = selected is not None and selected.detour <= self.opportunistic_detour_threshold
        diagnostic_print(
            lambda: (
                "OPPORTUNISTIC_HEAL_ANALYSIS: "
                f"hp_ratio={readiness.lowest_hp_ratio!r} goal={analysis.goal!r} "
                f"normal_route_cost={analysis.normal_cost!r} "
                f"candidates={[{'source': repr(c.destination), 'route_cost': c.total_cost, 'detour': c.detour, 'reachable': c.reachable} for c in analysis.candidates]!r} "
                f"selected_source={repr(selected.destination) if selected else None!r} "
                f"opportunistic_detour_threshold={self.opportunistic_detour_threshold!r} "
                f"decision={'HEAL' if decision else 'CONTINUE'}"
            ),
            trace=True,
        )
        return decision

    @staticmethod
    def _recovery_result(readiness: ProgressionReadinessDiagnostic, reason: ReadinessReason) -> ReadinessResult:
        """Convert recovery capability availability into a policy result."""

        if getattr(readiness, "recovery_availability", Availability.UNKNOWN) is Availability.KNOWN:
            return ReadinessResult(ReadinessDecision.RECOVER, reason)
        if getattr(readiness, "recovery_availability", Availability.UNKNOWN) is Availability.UNAVAILABLE:
            return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.RECOVERY_UNAVAILABLE)
        return ReadinessResult(ReadinessDecision.UNKNOWN, ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN)


def _has_imminent_trainer(readiness: ProgressionReadinessDiagnostic) -> bool:
    """Require direct same-map range evidence; never infer from map presence."""
    for trainer in getattr(readiness, "trainers", ()):
        if trainer.defeated is True or trainer.defeated is None:
            continue
        if trainer.trainer_range is None or trainer.distance is None:
            continue
        if trainer.distance <= trainer.trainer_range and trainer.relationship == "nearby":
            return True
    return False


def _party(snapshot: NuzlockeSnapshot) -> tuple[PartyReadinessMember, ...]:
    """Normalize party observations for readiness reporting."""

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


def _party_from_resource_snapshot(resource_snapshot: ResourceSnapshot) -> tuple[PartyReadinessMember, ...]:
    """Build readiness members from a valid resource-party fallback."""

    return tuple(
        PartyReadinessMember(
            index,
            "UNKNOWN",
            member.current_hp,
            member.max_hp,
            member.hp_ratio,
            member.fainted,
            not member.fainted and member.current_hp > 0,
            member.status or "none",
        )
        for index, member in enumerate(resource_snapshot.party)
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
    route_analysis: "RouteAnalysis | None" = None,
) -> ProgressionReadinessDiagnostic:
    """Build a point-in-time diagnostic without applying a survival policy."""
    members = _party(snapshot)
    party_availability = Availability.KNOWN if snapshot.party_available else Availability.UNAVAILABLE
    # The normalized campaign snapshot and the resource reader use separate
    # ROM access paths. At a battle/script boundary the former can be
    # temporarily unavailable while the latter still has a complete party.
    # Preserve that known party rather than converting it into a false
    # no-usable-Pokémon signal.
    if (
        (not snapshot.party_available or not snapshot.party)
        and resource_snapshot is not None
        and resource_snapshot.observation_status is ResourceObservationStatus.VALID
        and resource_snapshot.party
    ):
        members = _party_from_resource_snapshot(resource_snapshot)
        party_availability = Availability.KNOWN
    usable = None if party_availability is not Availability.KNOWN else any(p.usable for p in members)
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
        None,
        None,
        route_analysis,
    )


def evaluate_progression_readiness(
    readiness: ProgressionReadinessDiagnostic,
    policy: CampaignReadinessPolicy | None = None,
) -> ProgressionReadinessDiagnostic:
    """Evaluate and return a diagnostic enriched with the pure policy result."""
    result = (policy or CampaignReadinessPolicy()).evaluate(readiness)
    active_policy = policy or CampaignReadinessPolicy()
    recovery = readiness.recovery
    diagnostic_print(
        lambda: (
            "READINESS_POLICY_AFTER: "
            f"evaluation_id={id(readiness)} map={readiness.current_map!r} "
            f"location={readiness.current_coordinates!r} lowest_hp_ratio={readiness.lowest_hp_ratio!r} "
            f"critical_threshold={active_policy.critical_hp_ratio!r} "
            f"opportunistic_threshold={active_policy.opportunistic_hp_ratio!r} "
            f"critical={readiness.lowest_hp_ratio is not None and readiness.lowest_hp_ratio < active_policy.critical_hp_ratio} "
            f"opportunistic_threshold_met={readiness.lowest_hp_ratio is not None and readiness.lowest_hp_ratio <= active_policy.opportunistic_hp_ratio} "
            f"imminent_trainer={_has_imminent_trainer(readiness)} "
            f"recovery_availability_known={readiness.recovery_availability is Availability.KNOWN} "
            f"center_available={bool(recovery and recovery.center_available)} "
            f"center_safe={bool(recovery and recovery.safe_to_reach_center)} "
            f"distance_known={bool(recovery and recovery.distance_to_center is not None)} "
            f"distance={getattr(recovery, 'distance_to_center', None)!r} "
            f"opportunistic_detour_threshold={active_policy.opportunistic_detour_threshold!r} "
            f"decision={result.decision.value} reason={result.reason.value}"
        ),
        trace=True,
    )
    return replace(readiness, readiness_decision=result.decision, readiness_reason=result.reason)
