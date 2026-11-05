"""Small, composable goal descriptions for lower-level controllers."""

from dataclasses import dataclass, field
from enum import Enum, auto

from modules.overworld import Location, MapId, WorldTransition


class SemanticTargetKind(Enum):
    MAP = auto()
    LOCATION = auto()
    INTERACTION = auto()


@dataclass(frozen=True)
class SemanticTarget:
    """A world-level destination supplied by semantic policy.

    This describes what the campaign wants, not the route used to get there.
    ``interaction_id`` is intentionally opaque to navigation.
    """

    kind: SemanticTargetKind
    target_map: MapId | None = None
    location: Location | None = None
    interaction_id: str | None = None

    @classmethod
    def map(cls, target_map: MapId) -> "SemanticTarget":
        return cls(SemanticTargetKind.MAP, target_map=target_map)

    @classmethod
    def at(cls, location: Location) -> "SemanticTarget":
        return cls(SemanticTargetKind.LOCATION, target_map=location[0], location=location)

    @classmethod
    def interaction(cls, target_map: MapId, interaction_id: str | None = None) -> "SemanticTarget":
        return cls(SemanticTargetKind.INTERACTION, target_map=target_map, interaction_id=interaction_id)


class EncounterMode(Enum):
    SEEK = auto()
    IGNORE = auto()
    AVOID = auto()
    MOSTLY_AVOID = auto()
    # Compatibility for callers that used the old default.  The old NORMAL
    # behavior was strict encounter avoidance.
    NORMAL = AVOID


class TrainerMode(Enum):
    """How navigation treats undefeated trainer sight lines."""

    IGNORE = auto()
    AVOID = auto()
    ENGAGE = auto()


class TrainerEngagementMode(Enum):
    """How a trainer battle can be started from the current world snapshot."""

    SIGHTLINE = auto()
    INTERACTION = auto()
    UNKNOWN = auto()


INTRODUCTORY_RIVAL_TRIGGER_ID = "introductory_rival"
EARLY_POKEBALL_TRIGGER_ID = "early_pokeballs"


class Goal:
    """Marker base class for objectives supplied by a higher-level planner."""


@dataclass(frozen=True)
class EngageTrainer(Goal):
    """Intentionally activate one observed trainer by map/local identity."""

    trainer_id: str


@dataclass(frozen=True)
class TrainerEngagementPlan:
    """Resolved execution contract for an ``EngageTrainer`` goal.

    A trainer may start a battle when the player enters its live sight line,
    or only after the player reaches an adjacent tile and presses A.  The
    campaign goal remains ``EngageTrainer``; this small plan records the
    observed activation mechanism so lower-level navigation does not infer
    one from an empty or overloaded geometry field.
    """

    trainer_id: str
    trigger_id: str | None
    mode: TrainerEngagementMode
    engagement_locations: frozenset[Location] = frozenset()
    interaction_locations: frozenset[Location] = frozenset()
    activation_requirements: tuple[tuple[Location, object], ...] = ()
    requires_input: bool | None = None
    reason: str = ""

    @property
    def interaction_target(self) -> "ReachInteractionPosition | None":
        """Return the lower-level target when adjacent interaction is known."""

        if self.mode is not TrainerEngagementMode.INTERACTION or self.trigger_id is None:
            return None
        return ReachInteractionPosition(self.trigger_id)


@dataclass(frozen=True)
class ReachLocation(Goal):
    location: Location


@dataclass(frozen=True)
class ReachWarp(Goal):
    destination_map: MapId | None = None
    destination: Location | None = None
    # The destination map is only a routing hint; preserve the observed
    # source record so equal-destination warps remain distinguishable.
    # Kept as ``warp`` for API compatibility; live observations may provide a
    # MapConnectionObservation here as well.  Both are locally executable
    # world transitions and are planned by the same cross-map machinery.
    warp: WorldTransition | None = None
    # Several observed boundary coordinates may describe one ROM
    # MapConnection strip.  Supplying the strip here lets the local search
    # choose its cheapest reachable member in one planning problem while
    # preserving ``warp`` for callers selecting one exact transition.
    warps: tuple[WorldTransition, ...] = ()


@dataclass(frozen=True)
class ActivateTrigger(Goal):
    trigger_id: str


def introductory_rival_goal() -> ActivateTrigger:
    """Return the generic trigger goal used by the Emerald pilot."""
    return ActivateTrigger(INTRODUCTORY_RIVAL_TRIGGER_ID)


def early_pokeball_goal() -> ActivateTrigger:
    """Return the Emerald Professor Birch Poké Ball interaction goal."""
    return ActivateTrigger(EARLY_POKEBALL_TRIGGER_ID)


@dataclass(frozen=True)
class ReachInteractionPosition(Goal):
    trigger_id: str


@dataclass(frozen=True)
class GoalConstraints:
    avoid_trigger_ids: frozenset[str] = frozenset()
    avoid_locations: frozenset[Location] = frozenset()
    trainer_mode: TrainerMode = TrainerMode.IGNORE
    # Recovery must reject a route that is forced through a trainer, while
    # ordinary campaign travel may accept that route when no trainer-free path
    # exists.  Keep this separate from trainer_mode so callers can distinguish
    # a preference from a hard safety boundary.
    allow_forced_trainer_exposure: bool = False
    # When trainer avoidance is active, an optional non-empty set narrows the
    # avoidance boundary to these trainer identities.  Route analysis uses
    # this to measure one trainer's detour without conflating it with every
    # other sight line on the map.
    avoid_trainer_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class NavigationGoal(Goal):
    target: Goal
    constraints: GoalConstraints = field(default_factory=GoalConstraints)
    encounter_mode: EncounterMode = EncounterMode.AVOID
    # Used only by MOSTLY_AVOID.  Keeping it on the goal makes the tradeoff
    # explicit and allows different traversals to choose different risk
    # tolerances without global navigator state.
    encounter_penalty: int = 8
    # An optional ROM-backed map corridor for a campaign-owned destination.
    # Navigation still chooses the executable live transition on each edge,
    # but it does not explore unrelated map detours while composing the route.
    preferred_map_route: tuple[MapId, ...] | None = None
