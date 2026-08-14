"""Deterministic goal-aware navigation over a small abstract world model."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping

from modules.goals import (
    ActivateTrigger,
    Goal,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
)
from modules.map_path import Direction
from modules.overworld import Location, OverworldObservation, TriggerObservation, WarpObservation


class NavigationActionType(Enum):
    MOVE = auto()
    WARP = auto()


@dataclass(frozen=True)
class NavigationAction:
    action_type: NavigationActionType
    direction: Direction
    source: Location
    destination: Location


@dataclass(frozen=True)
class NavigableTile:
    location: Location
    blocked: bool = False
    allowed_directions: frozenset[Direction] | None = None


@dataclass(frozen=True)
class NavigationWorld:
    tiles: Mapping[Location, NavigableTile]
    warps: tuple[WarpObservation, ...] = ()
    triggers: tuple[TriggerObservation, ...] = ()

    @classmethod
    def from_overworld(cls, observation: OverworldObservation) -> "NavigationWorld":
        """Adapt the live perception model without coupling the search to emulator APIs."""
        return cls(
            tiles={tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors)
                   for tile in observation.tiles},
            warps=observation.warps,
            triggers=observation.triggers,
        )

    def neighbors(self, location: Location) -> tuple[tuple[Direction, Location, bool], ...]:
        tile = self.tiles.get(location)
        if tile is None or tile.blocked:
            return ()
        x, y = location[1]
        candidates = ((Direction.North, (x, y - 1)), (Direction.East, (x + 1, y)),
                      (Direction.South, (x, y + 1)), (Direction.West, (x - 1, y)))
        result: list[tuple[Direction, Location, bool]] = []
        for direction, coordinate in candidates:
            destination = (location[0], coordinate)
            neighbour = self.tiles.get(destination)
            if (neighbour is not None and not neighbour.blocked
                    and (neighbour.allowed_directions is None or direction in neighbour.allowed_directions)):
                result.append((direction, destination, False))
        for warp in self.warps:
            if warp.entry == location:
                result.append((self._warp_direction(location, warp), warp.destination, True))
        return tuple(result)

    @staticmethod
    def _warp_direction(location: Location, warp: WarpObservation) -> Direction:
        # Entering a warp is an input in the direction of the destination tile
        # only when the map data supplies one; otherwise any adjacent route can
        # reach the entry and the direction is a stable default.
        return warp.required_facing or Direction.South


@dataclass(frozen=True)
class NavigationPlan:
    actions: tuple[NavigationAction, ...]
    destination: Location | None


class NavigationError(RuntimeError):
    pass


class GoalAwareNavigator:
    def __init__(self, world: NavigationWorld):
        self.world = world

    def plan(self, start: Location, goal: NavigationGoal | Goal) -> NavigationPlan:
        navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
        queue = deque([start])
        came_from: dict[Location, tuple[Location, Direction, bool] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if self._satisfies(current, navigation_goal.target):
                return NavigationPlan(self._unroll(came_from, current), current)
            for direction, neighbour, is_warp in self.world.neighbors(current):
                if neighbour in came_from or self._is_undesirable(neighbour, navigation_goal.constraints):
                    continue
                came_from[neighbour] = (current, direction, is_warp)
                queue.append(neighbour)
        raise NavigationError(f"No legal route from {start} to {navigation_goal.target!r}")

    def _satisfies(self, location: Location, target: Goal) -> bool:
        if isinstance(target, ReachLocation):
            return location == target.location
        if isinstance(target, ActivateTrigger):
            return any(target.trigger_id == trigger.trigger_id and location in trigger.activation_locations
                       for trigger in self.world.triggers)
        if isinstance(target, ReachInteractionPosition):
            return any(target.trigger_id == trigger.trigger_id and location in trigger.activation_locations
                       for trigger in self.world.triggers)
        if isinstance(target, ReachWarp):
            return any(warp.entry == location and
                       (target.destination is None or warp.destination == target.destination) and
                       (target.destination_map is None or warp.destination[0] == target.destination_map)
                       for warp in self.world.warps)
        raise TypeError(f"Unsupported goal type: {type(target).__name__}")

    def _is_undesirable(self, location: Location, constraints) -> bool:
        if location in constraints.avoid_locations:
            return True
        return any(location in trigger.locations and trigger.trigger_id in constraints.avoid_trigger_ids
                   for trigger in self.world.triggers)

    @staticmethod
    def _unroll(came_from, destination: Location) -> tuple[NavigationAction, ...]:
        actions: list[NavigationAction] = []
        current = destination
        while came_from[current] is not None:
            source, direction, is_warp = came_from[current]
            actions.append(NavigationAction(
                NavigationActionType.WARP if is_warp else NavigationActionType.MOVE,
                direction, source, current,
            ))
            current = source
        actions.reverse()
        return tuple(actions)
