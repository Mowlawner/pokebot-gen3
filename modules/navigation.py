"""Deterministic goal-aware navigation over a small abstract world model."""

import heapq
from itertools import count
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
from modules.overworld import MapId, Location, OverworldObservation, TriggerObservation, WarpObservation
from modules.trigger_bindings import BindingResolution
from modules.world_navigation import WorldMapGraph, WorldNavigationError, WorldRoute, get_world_map_graph
from modules.profiler import profiled


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
    traversal_cost: int = 1


class _DynamicTileMapping(Mapping[Location, NavigableTile]):
    def __init__(self, static_tiles: Mapping[Location, NavigableTile], blocked: frozenset[tuple[int, int]]):
        self._static_tiles = static_tiles
        self._blocked = blocked

    def __getitem__(self, location: Location) -> NavigableTile:
        tile = self._static_tiles[location]
        if location[1] not in self._blocked or tile.blocked:
            return tile
        return NavigableTile(tile.location, True, tile.allowed_directions, tile.traversal_cost)

    def __iter__(self):
        return iter(self._static_tiles)

    def __len__(self):
        return len(self._static_tiles)


_static_navigation_tiles: dict[object, tuple[tuple, dict[Location, NavigableTile]]] = {}


def prewarm_navigation_tiles(map_id, tiles: tuple) -> None:
    """Populate the static navigation index for one already-known map.

    ``tiles`` must come from static overworld perception.  Dynamic blocking
    is intentionally absent here and remains an observation-time overlay.
    """
    cache = _static_navigation_tiles.get(map_id)
    if cache is None or cache[0] is not tiles:
        _static_navigation_tiles[map_id] = (
            tiles,
            {
                tile.location: NavigableTile(
                    tile.location, tile.blocked, tile.walkable_neighbors, tile.traversal_cost
                )
                for tile in tiles
            },
        )


@dataclass(frozen=True)
class NavigationWorld:
    tiles: Mapping[Location, NavigableTile]
    warps: tuple[WarpObservation, ...] = ()
    triggers: tuple[TriggerObservation, ...] = ()
    bindings: tuple[BindingResolution, ...] = ()

    @classmethod
    def from_overworld(cls, observation: OverworldObservation) -> "NavigationWorld":
        """Adapt the live perception model without coupling the search to emulator APIs."""
        cache = _static_navigation_tiles.get(observation.map_id)
        if cache is None or cache[0] is not observation.tiles:
            prewarm_navigation_tiles(observation.map_id, observation.tiles)
            static_tiles = _static_navigation_tiles[observation.map_id][1]
        else:
            static_tiles = cache[1]
        tiles: Mapping[Location, NavigableTile] = static_tiles
        if observation.dynamic_blocked_coordinates:
            tiles = _DynamicTileMapping(static_tiles, observation.dynamic_blocked_coordinates)
        return cls(
            tiles=tiles,
            warps=observation.warps,
            triggers=observation.triggers,
            bindings=observation.bindings,
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


def goal_target_map(world: NavigationWorld, goal: Goal) -> MapId | None:
    target = goal.target if isinstance(goal, NavigationGoal) else goal
    if isinstance(target, ReachLocation):
        return target.location[0]
    if isinstance(target, ActivateTrigger | ReachInteractionPosition):
        matching = tuple(trigger for trigger in world.triggers if trigger.trigger_id == target.trigger_id)
        maps = {trigger.target_map for trigger in matching if trigger.target_map is not None}
        maps.update(position[0] for trigger in matching for position in trigger.activation_locations)
        maps.update(position[0] for trigger in matching for position in trigger.locations)
        maps.update(position[0] for trigger in matching for position in trigger.navigation_locations)
        return next(iter(maps)) if len(maps) == 1 else None
    if isinstance(target, ReachWarp):
        return target.destination_map or (target.destination[0] if target.destination else None)
    return None


def _world_edge_direction(edge, fallback: Direction = Direction.South) -> Direction:
    if edge.required_facing is not None and isinstance(edge.required_facing, Direction):
        return edge.required_facing
    if edge.connection_direction is not None:
        try:
            return Direction.from_string(edge.connection_direction)
        except RuntimeError:
            pass
    return fallback


def plan_with_world_navigation(
    world: NavigationWorld,
    start: Location,
    goal: NavigationGoal | Goal,
    graph: WorldMapGraph | None = None,
) -> tuple[NavigationPlan, WorldRoute | None]:
    """Plan one local segment, appending one generic map transition if needed."""
    target_map = goal_target_map(world, goal)
    if target_map is None or target_map == start[0]:
        return GoalAwareNavigator(world).plan(start, goal), None

    graph = graph or get_world_map_graph()
    route = graph.route(start[0], target_map)
    edge = route.edges[0]
    candidates = tuple(
        (source_map, coordinates)
        for coordinates, source_map in zip(edge.source_coordinates, (edge.source_map,) * len(edge.source_coordinates))
        if source_map == start[0]
    )
    # Map warp records name the tile containing the warp, while the input
    # which activates a normal door warp is the step *onto* that tile.  Find
    # the adjacent activation positions instead of treating the warp tile as
    # the source of a second, repeated input.
    candidate_plans: list[tuple[int, NavigationPlan, tuple[int, int], Direction]] = []
    runtime_warps = {warp.entry[1]: warp for warp in world.warps
                     if warp.entry[0] == edge.source_map and warp.destination[0] == edge.destination_map}
    for _, coordinates in candidates:
        activation_candidates = ((coordinates[0], coordinates[1] - 1, Direction.South),
                                 (coordinates[0] + 1, coordinates[1], Direction.West),
                                 (coordinates[0], coordinates[1] + 1, Direction.North),
                                 (coordinates[0] - 1, coordinates[1], Direction.East))
        warp = runtime_warps.get(coordinates)
        choices = activation_candidates if warp is not None else ((coordinates[0], coordinates[1], None),)
        for x, y, activation_direction in choices:
            try:
                local_plan = GoalAwareNavigator(world).plan(
                    start, ReachLocation((start[0], (x, y)))
                )
            except NavigationError:
                continue
            candidate_plans.append((len(local_plan.actions), local_plan,
                                    coordinates, activation_direction))
            break
    if not candidate_plans:
        raise NavigationError(
            f"No reachable transition entry on {start[0]!r} for {edge.destination_map!r}"
        )
    _, local_plan, source_coordinates, activation_direction = min(candidate_plans, key=lambda item: item[0])
    destination_index = edge.source_coordinates.index(source_coordinates)
    destination_coordinates = edge.destination_coordinates[destination_index]
    transition_direction = activation_direction or _world_edge_direction(edge)
    runtime_warp = runtime_warps.get(source_coordinates)
    activation_position = None
    if activation_direction is not None:
        activation_position = (
            source_coordinates[0] - (activation_direction is Direction.East) +
            (activation_direction is Direction.West),
            source_coordinates[1] - (activation_direction is Direction.South) +
            (activation_direction is Direction.North),
        )
    if runtime_warp is not None and runtime_warp.required_facing is not None:
        # Arrow warps first step onto the warp and then accept the ROM-defined
        # direction as their activation input.
        if activation_direction is not None:
            local_plan = NavigationPlan(
                local_plan.actions + (NavigationAction(
                    NavigationActionType.MOVE, activation_direction,
                    (start[0], activation_position),
                    (start[0], source_coordinates)),),
                local_plan.destination,
            )
        transition_direction = runtime_warp.required_facing
    transition = NavigationAction(
        NavigationActionType.WARP,
        transition_direction,
        (start[0], source_coordinates) if runtime_warp is not None and runtime_warp.required_facing is not None
        else (start[0], activation_position or source_coordinates),
        (edge.destination_map, destination_coordinates),
    )
    return NavigationPlan(local_plan.actions + (transition,), local_plan.destination), route


def navigation_diagnostics(world: NavigationWorld, start: Location, goal: Goal) -> tuple[str, ...]:
    """Summarize why a navigation request has no target or route.

    This is intentionally descriptive only.  It does not add cross-map
    search, alter the graph, or inspect emulator state outside the supplied
    world snapshot.
    """
    target = goal.target if isinstance(goal, NavigationGoal) else goal
    matching_triggers = ()
    target_positions: tuple[Location, ...] = ()
    target_maps: tuple[MapId, ...] = ()
    if isinstance(target, ActivateTrigger):
        matching_triggers = tuple(
            trigger for trigger in world.triggers if trigger.trigger_id == target.trigger_id
        )
        target_positions = tuple(
            sorted({position for trigger in matching_triggers for position in trigger.activation_locations},
                   key=repr)
        )
        if not target_positions:
            target_positions = tuple(
                sorted({position for trigger in matching_triggers for position in trigger.navigation_locations},
                       key=repr)
            )
        target_maps = tuple(sorted(
            {position[0] for position in target_positions}
            | {trigger.target_map for trigger in matching_triggers if trigger.target_map is not None},
            key=repr,
        ))
    elif isinstance(target, ReachInteractionPosition):
        matching_triggers = tuple(
            trigger for trigger in world.triggers if trigger.trigger_id == target.trigger_id
        )
        target_positions = tuple(
            sorted({position for trigger in matching_triggers for position in trigger.activation_locations},
                   key=repr)
        )
        if not target_positions:
            target_positions = tuple(
                sorted({position for trigger in matching_triggers for position in trigger.navigation_locations},
                       key=repr)
            )
        target_maps = tuple(sorted(
            {position[0] for position in target_positions}
            | {trigger.target_map for trigger in matching_triggers if trigger.target_map is not None},
            key=repr,
        ))
    elif isinstance(target, ReachLocation):
        target_positions = (target.location,)
        target_maps = (target.location[0],)
    elif isinstance(target, ReachWarp):
        target_maps = tuple(
            sorted({warp.destination[0] for warp in world.warps
                    if target.destination_map is None or warp.destination[0] == target.destination_map},
                   key=repr)
        )

    current_map_goal = bool(target_maps) and set(target_maps) == {start[0]}
    available_warps = tuple(
        (warp.entry, warp.destination, warp.required_facing.name if warp.required_facing else None)
        for warp in world.warps
    )
    matching_bindings = tuple(
        binding for binding in world.bindings
        if binding.binding.trigger_id == getattr(target, "trigger_id", None)
    )
    reason = (
        "TRIGGER_NOT_PRESENT_IN_PERCEPTION" if isinstance(target, (ActivateTrigger, ReachInteractionPosition))
        and not matching_triggers else
        "STATIC_TARGET_AMBIGUOUS" if any(binding.static_ambiguous for binding in matching_bindings) else
        "STATIC_TARGET_HIDDEN" if any(
            binding.static_match and not binding.static_available
            for binding in matching_bindings
        ) else "CROSS_MAP_GOAL_NOT_SUPPORTED" if target_maps and not current_map_goal
        else "NO_ROUTE_IN_CURRENT_MAP"
    )
    return (
        f"current_map={start[0]!r}",
        f"target_map={target_maps!r}",
        f"target_positions={target_positions!r}",
        f"current_map_goal={current_map_goal}",
        f"available_warps={available_warps!r}",
        f"candidate_interaction_positions={len(target_positions)}",
        f"status=UNREACHABLE reason={reason}",
    )


class GoalAwareNavigator:
    def __init__(self, world: NavigationWorld):
        self.world = world

    @profiled("navigation_pathfinding", "pathfinding_calls")
    def plan(self, start: Location, goal: NavigationGoal | Goal) -> NavigationPlan:
        navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
        # Dijkstra's algorithm is the smallest change from the previous BFS:
        # with every traversal_cost == 1, insertion order remains the same as
        # the old FIFO queue, while positive terrain weights are minimized.
        sequence = count()
        queue = [(0, next(sequence), start)]
        costs: dict[Location, int] = {start: 0}
        came_from: dict[Location, tuple[Location, Direction, bool] | None] = {start: None}
        while queue:
            current_cost, _, current = heapq.heappop(queue)
            if current_cost != costs[current]:
                continue
            if self._satisfies(current, navigation_goal.target):
                return NavigationPlan(self._unroll(came_from, current), current)
            for direction, neighbour, is_warp in self.world.neighbors(current):
                if self._is_undesirable(neighbour, navigation_goal.constraints):
                    continue
                tile = self.world.tiles.get(neighbour)
                step_cost = 0 if is_warp else tile.traversal_cost
                new_cost = current_cost + step_cost
                if new_cost >= costs.get(neighbour, float("inf")):
                    continue
                costs[neighbour] = new_cost
                came_from[neighbour] = (current, direction, is_warp)
                heapq.heappush(queue, (new_cost, next(sequence), neighbour))
        raise NavigationError(f"No legal route from {start} to {navigation_goal.target!r}")

    def _satisfies(self, location: Location, target: Goal) -> bool:
        if isinstance(target, ReachLocation):
            return location == target.location
        if isinstance(target, ActivateTrigger):
            return any(
                target.trigger_id == trigger.trigger_id
                and location in (trigger.activation_locations or trigger.navigation_locations)
                for trigger in self.world.triggers
            )
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
