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
    NORMAL = auto()
    SEEK = auto()


INTRODUCTORY_RIVAL_TRIGGER_ID = "introductory_rival"
EARLY_POKEBALL_TRIGGER_ID = "early_pokeballs"


class Goal:
    """Marker base class for objectives supplied by a higher-level planner."""


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


@dataclass(frozen=True)
class NavigationGoal(Goal):
    target: Goal
    constraints: GoalConstraints = field(default_factory=GoalConstraints)
    encounter_mode: EncounterMode = EncounterMode.NORMAL
