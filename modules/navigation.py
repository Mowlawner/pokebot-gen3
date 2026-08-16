"""Deterministic goal-aware navigation over a small abstract world model."""

import heapq
from time import perf_counter_ns
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
    EncounterMode,
)
from modules.map_path import Direction
from modules.overworld import (
    MapId,
    Location,
    OverworldObservation,
    TriggerObservation,
    WarpObservation,
    WarpActivation,
)
from modules.trigger_bindings import BindingResolution
from modules.world_navigation import WorldMapGraph, WorldNavigationError, WorldRoute, get_world_map_graph
from modules.stutter_trace import traced
from modules.context import context
from modules.profiler import profiled
from modules.profiler import (
    enabled as profiling_enabled,
    record_cross_map_goal_evaluation,
    record_pathfinding_search,
    pathfinding_searches,
)


class NavigationActionType(Enum):
    MOVE = auto()
    TURN = auto()
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
    has_encounters: bool = False


class _DynamicTileMapping(Mapping[Location, NavigableTile]):
    def __init__(self, static_tiles: Mapping[Location, NavigableTile], blocked: frozenset[tuple[int, int]]):
        self._static_tiles = static_tiles
        self._blocked = blocked

    def __getitem__(self, location: Location) -> NavigableTile:
        tile = self._static_tiles[location]
        if location[1] not in self._blocked or tile.blocked:
            return tile
        return NavigableTile(tile.location, True, tile.allowed_directions, tile.traversal_cost, tile.has_encounters)

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
                    tile.location, tile.blocked, tile.walkable_neighbors, tile.traversal_cost, tile.has_encounters
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
    facing: Direction | None = None

    @classmethod
    @traced("NavigationWorld_construction")
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
            facing=observation.facing,
        )

    def neighbors(self, location: Location) -> tuple[tuple[Direction, Location, bool], ...]:
        tile = self.tiles.get(location)
        if tile is None or tile.blocked:
            return ()
        x, y = location[1]
        candidates = (
            (Direction.North, (x, y - 1)),
            (Direction.East, (x + 1, y)),
            (Direction.South, (x, y + 1)),
            (Direction.West, (x - 1, y)),
        )
        result: list[tuple[Direction, Location, bool]] = []
        for direction, coordinate in candidates:
            destination = (location[0], coordinate)
            neighbour = self.tiles.get(destination)
            if (
                neighbour is not None
                and not neighbour.blocked
                and (neighbour.allowed_directions is None or direction in neighbour.allowed_directions)
            ):
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
    metrics: "NavigationMetrics | None" = None
    candidate_metrics: tuple["NavigationMetrics", ...] = ()


@dataclass(frozen=True)
class NavigationMetrics:
    movement_actions: int
    ordinary_movement_steps: int
    encounter_opportunities: int
    encounter_terrain_moves: int
    turns_on_encounter_terrain: int
    turns_on_non_encounter_terrain: int
    final_facing: Direction | None
    total_route_cost: int = 0

    @property
    def turns(self) -> int:
        return self.turns_on_encounter_terrain + self.turns_on_non_encounter_terrain

    def summary(self) -> str:
        return (
            f"{self.movement_actions} movement actions, "
            f"{self.ordinary_movement_steps} ordinary movement steps, "
            f"{self.encounter_opportunities} encounter opportunities, "
            f"{self.encounter_terrain_moves} encounter-terrain moves, "
            f"{self.turns_on_encounter_terrain} turns on encounter terrain, "
            f"{self.turns_on_non_encounter_terrain} turns on non-encounter terrain, "
            f"{self.turns} total turns, "
            f"final facing={getattr(self.final_facing, 'name', None)!r}, "
            f"total route cost={self.total_route_cost}"
        )


class NavigationError(RuntimeError):
    pass


# NORMAL is deliberately opportunity-first.  The second component keeps the
# result practical and deterministic after exposure has been minimized; it is
# not allowed to trade an extra encounter opportunity for fewer steps.
NORMAL_ENCOUNTER_OPPORTUNITY_WEIGHT = 1
NORMAL_MOVEMENT_COST_SCALE = 1


def navigation_candidate_key(metrics: NavigationMetrics, encounter_mode: EncounterMode, candidate_id: str = ""):
    """Canonical ordering for complete local candidate plans."""
    if encounter_mode is EncounterMode.NORMAL:
        return (
            metrics.encounter_opportunities,
            metrics.total_route_cost,
            metrics.movement_actions,
            metrics.encounter_terrain_moves,
            metrics.turns_on_encounter_terrain,
            metrics.turns_on_non_encounter_terrain,
            candidate_id,
        )
    return (
        metrics.total_route_cost,
        metrics.movement_actions,
        metrics.encounter_terrain_moves,
        metrics.turns_on_encounter_terrain,
        metrics.turns_on_non_encounter_terrain,
        candidate_id,
    )


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


@traced("world_map_and_candidate_planning")
def plan_with_world_navigation(
    world: NavigationWorld,
    start: Location,
    goal: NavigationGoal | Goal,
    graph: WorldMapGraph | None = None,
    *,
    algorithm: str = "dijkstra",
) -> tuple[NavigationPlan, WorldRoute | None]:
    """Plan one local segment, appending one generic map transition if needed."""
    navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
    target_map = goal_target_map(world, goal)
    if target_map is None or target_map == start[0]:
        return GoalAwareNavigator(world).plan(start, goal, algorithm=algorithm), None

    trace = getattr(context, "stutter_trace", None)
    trace_graph_start = trace.now() if trace is not None else 0
    graph = graph or get_world_map_graph()
    if trace is not None:
        trace.duration("world_map_graph_lookup_duration_ms", trace_graph_start)
    trace_route_start = trace.now() if trace is not None else 0
    route = graph.route(start[0], target_map)
    if trace is not None:
        trace.duration("world_map_route_lookup_duration_ms", trace_route_start)
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
    candidate_plans: list[tuple[NavigationPlan, tuple[int, int], Direction | None, str]] = []
    profile_start_index = len(pathfinding_searches())
    cross_map_started = perf_counter_ns() if profiling_enabled() else 0
    candidate_records: list[dict[str, object]] = []
    candidate_attempt_count = 0
    best_encounters: int | None = None
    best_route_cost: int | None = None
    runtime_warps = {
        warp.entry[1]: warp
        for warp in world.warps
        if warp.entry[0] == edge.source_map and warp.destination[0] == edge.destination_map
    }
    candidate_choices: list[tuple[str, tuple[int, int], tuple[tuple[int, int, Direction | None], ...]]] = []
    for candidate_index, (_, coordinates) in enumerate(candidates):
        activation_candidates = (
            (coordinates[0], coordinates[1] - 1, Direction.South),
            (coordinates[0] + 1, coordinates[1], Direction.West),
            (coordinates[0], coordinates[1] + 1, Direction.North),
            (coordinates[0] - 1, coordinates[1], Direction.East),
        )
        warp = runtime_warps.get(coordinates)
        if warp is not None and warp.activation_locations:
            choices = tuple(
                (source[1][0], source[1][1], warp.activation_direction) for source in warp.activation_locations
            )
        else:
            choices = activation_candidates if warp is not None else ((coordinates[0], coordinates[1], None),)
        choice_tuple = tuple(choices)
        candidate_choices.append((str(candidate_index), coordinates, choice_tuple))
        candidate_attempt_count += len(choice_tuple)

    navigator = GoalAwareNavigator(world)
    shared_target_groups = tuple(
        tuple((start[0], (choice[0], choice[1])) for choice in choices_for_candidate)
        for _, _, choices_for_candidate in candidate_choices
    )
    shared_targets = tuple(location for group in shared_target_groups for location in group)
    shared_plans = navigator.plan_many_locations(
        start,
        tuple(dict.fromkeys(shared_targets)),
        encounter_mode=navigation_goal.encounter_mode,
        algorithm=algorithm,
        target_groups=shared_target_groups,
    )
    for candidate_id, coordinates, choices_for_candidate in candidate_choices:
        selected_index = next(
            (
                index
                for index, choice in enumerate(choices_for_candidate)
                if (start[0], (choice[0], choice[1])) in shared_plans
            ),
            None,
        )
        for choice_index, choice in enumerate(choices_for_candidate):
            if selected_index is not None and choice_index > selected_index:
                break
            if selected_index is None or choice_index < selected_index:
                candidate_records.append(
                    {
                        "candidate_id": f"{candidate_id}:{choice_index}",
                        "source_coordinates": coordinates,
                        "activation_coordinates": (choice[0], choice[1]),
                        "activation_direction": getattr(choice[2], "name", None),
                        "status": "no_route",
                        "reason": "shared_local_search_exhausted",
                    }
                )
        if selected_index is None:
            continue
        selected_choice = choices_for_candidate[selected_index]
        x, y, activation_direction = selected_choice
        local_plan = shared_plans[(start[0], (x, y))]
        candidate_records.append(
            {
                "candidate_id": f"{candidate_id}:{selected_index}",
                "source_coordinates": coordinates,
                "activation_coordinates": (x, y),
                "activation_direction": getattr(activation_direction, "name", None),
                "status": "pathfound",
                "reason": "first_reachable_activation_side",
                "pathfinding_calls": 1,
                "nodes_expanded": None,
                "encounter_opportunities": local_plan.metrics.encounter_opportunities if local_plan.metrics else None,
                "route_cost": local_plan.metrics.total_route_cost if local_plan.metrics else None,
            }
        )
        candidate_plans.append((local_plan, coordinates, activation_direction, f"{candidate_id}:{selected_index}"))
        if local_plan.metrics is not None:
            if best_encounters is None or local_plan.metrics.encounter_opportunities < best_encounters:
                best_encounters = local_plan.metrics.encounter_opportunities
                best_route_cost = local_plan.metrics.total_route_cost
            elif local_plan.metrics.encounter_opportunities == best_encounters:
                best_route_cost = min(best_route_cost, local_plan.metrics.total_route_cost)
    if not candidate_plans:
        if profiling_enabled():
            group_records = pathfinding_searches()[profile_start_index:]
            record_cross_map_goal_evaluation(
                algorithm=algorithm,
                start=start,
                source_map=edge.source_map,
                destination_map=edge.destination_map,
                goal=repr(navigation_goal.target),
                goal_type=type(navigation_goal.target).__name__,
                encounter_mode=navigation_goal.encounter_mode.name,
                warp_candidates=len(candidates),
                activation_side_candidates=candidate_attempt_count,
                candidates_rejected_before_pathfinding=0,
                candidates_pruned_before_search=0,
                candidates_dominated=0,
                candidates_pathfound=0,
                candidates_no_route=len(candidate_records),
                pathfinding_calls=len(group_records),
                nodes_expanded=sum(int(record.get("nodes_expanded", 0)) for record in group_records),
                pathfinding_time_ms=round(sum(float(record.get("duration_ms", 0)) for record in group_records), 3),
                goal_planning_time_ms=round((perf_counter_ns() - cross_map_started) / 1_000_000, 3),
                best_encounter_opportunities=None,
                best_secondary_route_cost=None,
                selected_candidate_id=None,
                candidates=tuple(candidate_records),
            )
        raise NavigationError(f"No reachable transition entry on {start[0]!r} for {edge.destination_map!r}")
    # Candidate count is attached after all local searches complete.  Records
    # are mutable profiling-only dictionaries, so this does not affect plans.
    candidate_count = len(candidates)
    group_records = pathfinding_searches()[profile_start_index:]
    group_total_ms = round(sum(float(record.get("duration_ms", 0)) for record in group_records), 3)
    group_total_nodes = sum(int(record.get("nodes_expanded", 0)) for record in group_records)
    group_fastest_ms = min((float(record.get("duration_ms", 0)) for record in group_records), default=0)
    group_slowest_ms = max((float(record.get("duration_ms", 0)) for record in group_records), default=0)
    for record in group_records:
        record["candidate_searches_for_goal"] = candidate_count
        record["warp_candidates_considered"] = candidate_count
        record["cross_map_searches_performed"] = len(group_records)
        record["cross_map_total_ms"] = group_total_ms
        record["cross_map_total_nodes_expanded"] = group_total_nodes
        record["cross_map_fastest_ms"] = round(group_fastest_ms, 3)
        record["cross_map_slowest_ms"] = round(group_slowest_ms, 3)

    def completed_local_plan(item):
        local_plan, source_coordinates, activation_direction, _ = item
        if (
            activation_direction is not None
            and world.facing is not None
            and local_plan.metrics is not None
            and local_plan.metrics.final_facing is not activation_direction
        ):
            runtime_warp = runtime_warps.get(source_coordinates)
            activation_position = (
                source_coordinates[0]
                - (activation_direction is Direction.East)
                + (activation_direction is Direction.West),
                source_coordinates[1]
                - (activation_direction is Direction.South)
                + (activation_direction is Direction.North),
            )
            turn_location = (start[0], source_coordinates if runtime_warp is not None else activation_position)
            actions = local_plan.actions + (
                NavigationAction(NavigationActionType.TURN, activation_direction, turn_location, turn_location),
            )
            return NavigationPlan(
                actions, local_plan.destination, navigator._metrics(actions, navigation_goal.encounter_mode)
            )
        return local_plan

    def candidate_key(item):
        local_plan, _, _, candidate_id = item
        local_plan = completed_local_plan(item)
        metrics = local_plan.metrics
        if metrics is None:
            return (float("inf"),) * 6 + (candidate_id,)
        return navigation_candidate_key(metrics, navigation_goal.encounter_mode, candidate_id)

    local_plan, source_coordinates, activation_direction, selected_candidate_id = min(
        candidate_plans, key=candidate_key
    )
    local_plan = completed_local_plan((local_plan, source_coordinates, activation_direction, selected_candidate_id))
    for candidate_record in candidate_records:
        if candidate_record["candidate_id"] == selected_candidate_id:
            candidate_record["status"] = "selected"
            candidate_record["reason"] = "canonical_navigation_objective"
        elif candidate_record["status"] == "pathfound":
            candidate_record["reason"] = "pathfound_but_not_selected"
    if profiling_enabled():
        record_cross_map_goal_evaluation(
            algorithm=algorithm,
            start=start,
            source_map=edge.source_map,
            destination_map=edge.destination_map,
            goal=repr(navigation_goal.target),
            goal_type=type(navigation_goal.target).__name__,
            encounter_mode=navigation_goal.encounter_mode.name,
            warp_candidates=len(candidates),
            activation_side_candidates=candidate_attempt_count,
            candidates_rejected_before_pathfinding=0,
            candidates_pruned_before_search=0,
            candidates_dominated=0,
            candidates_pathfound=sum(record["status"] in ("pathfound", "selected") for record in candidate_records),
            candidates_no_route=sum(record["status"] == "no_route" for record in candidate_records),
            pathfinding_calls=len(group_records),
            nodes_expanded=group_total_nodes,
            pathfinding_time_ms=group_total_ms,
            goal_planning_time_ms=round((perf_counter_ns() - cross_map_started) / 1_000_000, 3),
            best_encounter_opportunities=best_encounters,
            best_secondary_route_cost=best_route_cost,
            selected_candidate_id=selected_candidate_id,
            candidates=tuple(candidate_records),
        )
    destination_index = edge.source_coordinates.index(source_coordinates)
    destination_coordinates = edge.destination_coordinates[destination_index]
    transition_direction = activation_direction or _world_edge_direction(edge)
    runtime_warp = runtime_warps.get(source_coordinates)
    activation_position = None
    if activation_direction is not None:
        activation_position = (
            source_coordinates[0] - (activation_direction is Direction.East) + (activation_direction is Direction.West),
            source_coordinates[1]
            - (activation_direction is Direction.South)
            + (activation_direction is Direction.North),
        )
    if runtime_warp is not None and (
        runtime_warp.activation is WarpActivation.DIRECTIONAL_STEP
        or (runtime_warp.required_facing is not None and not runtime_warp.activation_locations)
    ):
        # ROM arrow warps first step onto the warp and then accept the
        # ROM-defined direction as their activation input. Step-on warps,
        # including escalators, activate from the adjacent tile's MOVE.
        if activation_direction is not None:
            local_plan = NavigationPlan(
                local_plan.actions
                + (
                    NavigationAction(
                        NavigationActionType.MOVE,
                        activation_direction,
                        (start[0], activation_position),
                        (start[0], source_coordinates),
                    ),
                ),
                local_plan.destination,
            )
        transition_direction = runtime_warp.required_facing
    transition = NavigationAction(
        NavigationActionType.WARP,
        transition_direction,
        (
            (start[0], source_coordinates)
            if runtime_warp is not None
            and (
                runtime_warp.activation is WarpActivation.DIRECTIONAL_STEP
                or (runtime_warp.required_facing is not None and not runtime_warp.activation_locations)
            )
            else (start[0], activation_position or source_coordinates)
        ),
        (edge.destination_map, destination_coordinates),
    )
    actions = local_plan.actions + (transition,)
    return (
        NavigationPlan(
            actions, local_plan.destination, GoalAwareNavigator(world)._metrics(actions, navigation_goal.encounter_mode)
        ),
        route,
    )


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
        matching_triggers = tuple(trigger for trigger in world.triggers if trigger.trigger_id == target.trigger_id)
        target_positions = tuple(
            sorted({position for trigger in matching_triggers for position in trigger.activation_locations}, key=repr)
        )
        if not target_positions:
            target_positions = tuple(
                sorted(
                    {position for trigger in matching_triggers for position in trigger.navigation_locations}, key=repr
                )
            )
        target_maps = tuple(
            sorted(
                {position[0] for position in target_positions}
                | {trigger.target_map for trigger in matching_triggers if trigger.target_map is not None},
                key=repr,
            )
        )
    elif isinstance(target, ReachInteractionPosition):
        matching_triggers = tuple(trigger for trigger in world.triggers if trigger.trigger_id == target.trigger_id)
        target_positions = tuple(
            sorted({position for trigger in matching_triggers for position in trigger.activation_locations}, key=repr)
        )
        if not target_positions:
            target_positions = tuple(
                sorted(
                    {position for trigger in matching_triggers for position in trigger.navigation_locations}, key=repr
                )
            )
        target_maps = tuple(
            sorted(
                {position[0] for position in target_positions}
                | {trigger.target_map for trigger in matching_triggers if trigger.target_map is not None},
                key=repr,
            )
        )
    elif isinstance(target, ReachLocation):
        target_positions = (target.location,)
        target_maps = (target.location[0],)
    elif isinstance(target, ReachWarp):
        target_maps = tuple(
            sorted(
                {
                    warp.destination[0]
                    for warp in world.warps
                    if target.destination_map is None or warp.destination[0] == target.destination_map
                },
                key=repr,
            )
        )

    current_map_goal = bool(target_maps) and set(target_maps) == {start[0]}
    available_warps = tuple(
        (
            warp.entry,
            warp.destination,
            warp.required_facing.name if warp.required_facing else None,
            warp.activation.name,
            tuple(sorted(warp.activation_locations, key=repr)),
            warp.activation_direction.name if warp.activation_direction else None,
        )
        for warp in world.warps
    )
    matching_bindings = tuple(
        binding for binding in world.bindings if binding.binding.trigger_id == getattr(target, "trigger_id", None)
    )
    reason = (
        "TRIGGER_NOT_PRESENT_IN_PERCEPTION"
        if isinstance(target, (ActivateTrigger, ReachInteractionPosition)) and not matching_triggers
        else (
            "STATIC_TARGET_AMBIGUOUS"
            if any(binding.static_ambiguous for binding in matching_bindings)
            else (
                "STATIC_TARGET_HIDDEN"
                if any(binding.static_match and not binding.static_available for binding in matching_bindings)
                else (
                    "CROSS_MAP_GOAL_NOT_SUPPORTED"
                    if target_maps and not current_map_goal
                    else "NO_ROUTE_IN_CURRENT_MAP"
                )
            )
        )
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
    @traced("individual_pathfinding")
    def plan(self, start: Location, goal: NavigationGoal | Goal, *, algorithm: str = "dijkstra") -> NavigationPlan:
        navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
        # Keep ordinary location routing compatible with the established
        # movement pipeline.  Orientation becomes a search dimension when the
        # goal's semantics actually depend on it (NPC/tile interaction or a
        # directional warp).
        # Facing is a state dimension for every plan when it is observed. A
        # directional input while facing elsewhere is a TURN, not a MOVE;
        # this is required for accurate action diagnostics even for location
        # goals. Interaction and warp goals additionally use the final facing
        # to determine whether the goal is satisfied.
        target = navigation_goal.target
        orientation_required = self.world.facing is not None
        emit_orientation_actions = isinstance(target, (ActivateTrigger, ReachInteractionPosition, ReachWarp))
        executable_orientation = orientation_required
        if algorithm not in ("dijkstra", "astar"):
            raise ValueError(f"Unknown navigation search algorithm: {algorithm!r}")
        profiling = profiling_enabled()
        search_started = perf_counter_ns() if profiling else 0
        expanded = generated = frontier_max = 0
        # Dijkstra's algorithm is the smallest change from the previous BFS:
        # with every traversal_cost == 1, insertion order remains the same as
        # the old FIFO queue, while positive terrain weights are minimized.
        sequence = count()
        initial_state = (start, self.world.facing if orientation_required else None)
        # NORMAL's objective is lexicographic: exposure first, practical
        # route cost second.  Keeping the counters in the search key makes
        # pruning use the same objective as final route selection.
        # (encounter opportunities, weighted route cost, movements,
        # encounter-terrain moves, encounter-terrain turns, non-encounter
        # turns).
        initial_cost = (0, 0, 0, 0, 0, 0)
        queue = [(initial_cost, next(sequence), initial_state, initial_cost)]
        costs: dict[tuple[Location, Direction | None], tuple[int, int, int, int, int, int]] = {
            initial_state: initial_cost
        }
        came_from: dict[tuple[Location, Direction | None], tuple | None] = {initial_state: None}
        complete_candidates: list[tuple[NavigationMetrics, tuple[NavigationAction, ...], Location]] = []
        collect_candidate_diagnostics = bool(context.debug and getattr(context, "debug_trace", False))

        def priority(cost, state):
            if algorithm == "dijkstra":
                return cost
            heuristic = self._heuristic(state[0], navigation_goal.target, navigation_goal.encounter_mode)
            return tuple(cost[index] + heuristic[index] for index in range(6))

        def record(found, metrics=None):
            if profiling:
                record_pathfinding_search(
                    algorithm=algorithm,
                    map=start[0],
                    start_state=initial_state,
                    goal=repr(navigation_goal.target),
                    goal_type=type(navigation_goal.target).__name__,
                    nodes_expanded=expanded,
                    nodes_generated=generated,
                    max_frontier_size=frontier_max,
                    duration_ms=round((perf_counter_ns() - search_started) / 1_000_000, 3),
                    path_found=found,
                    final_encounter_opportunity_count=(metrics.encounter_opportunities if metrics else None),
                    final_route_cost=(metrics.total_route_cost if metrics else None),
                    candidate_searches_for_goal=1,
                )

        if algorithm == "astar":
            queue[0] = (priority(initial_cost, initial_state), queue[0][1], initial_state, initial_cost)
        generated = 1
        while queue:
            _, _, (current, facing), current_cost = heapq.heappop(queue)
            state = (current, facing)
            if current_cost != costs[state]:
                continue
            expanded += 1
            frontier_max = max(frontier_max, len(queue))
            if self._satisfies(current, facing, navigation_goal.target):
                actions = self._unroll(came_from, state, emit_orientation_actions)
                metrics = self._metrics(actions, navigation_goal.encounter_mode)
                if not collect_candidate_diagnostics:
                    record(True, metrics)
                    return NavigationPlan(actions, current, metrics)
                complete_candidates.append((metrics, actions, current))
                complete_candidates.sort(
                    key=lambda item: (
                        item[0].encounter_opportunities,
                        item[0].total_route_cost,
                        item[0].movement_actions,
                        item[0].turns,
                    )
                )
                complete_candidates = complete_candidates[:5]
                # Once a few alternatives have been captured, no route more
                # than two opportunities or twenty practical-cost units away
                # can improve the diagnostic ranking.  This bound is only for
                # collecting diagnostics; the first complete state remains
                # the selected optimum.
                if (
                    len(complete_candidates) >= 5
                    and queue
                    and queue[0][0][0] > complete_candidates[0][0].encounter_opportunities + 2
                ):
                    break
                continue
            # A directional input while standing is a real, observable input
            # event in Gen III.  Model it independently from movement so a
            # goal that is already on a valid source tile still produces the
            # required TURN before INTERACT/WARP.
            if emit_orientation_actions and executable_orientation and current in self.world.tiles:
                for turn_direction in Direction:
                    if turn_direction is facing:
                        continue
                    turn_on_encounter = int(self.world.tiles[current].has_encounters)
                    turn_on_non_encounter = int(not turn_on_encounter)
                    opportunity_cost = (
                        0
                        if navigation_goal.encounter_mode is EncounterMode.SEEK
                        else NORMAL_ENCOUNTER_OPPORTUNITY_WEIGHT
                    )
                    turn_cost = (turn_on_encounter, 1, 0, 0, turn_on_encounter, turn_on_non_encounter)
                    next_state = (current, turn_direction)
                    new_cost = tuple(current_cost[i] + turn_cost[i] for i in range(6))
                    if new_cost < costs.get(next_state, (float("inf"),) * 6):
                        costs[next_state] = new_cost
                        came_from[next_state] = (state, turn_direction, False, False, True)
                        heapq.heappush(queue, (priority(new_cost, next_state), next(sequence), next_state, new_cost))
                        generated += 1
                        frontier_max = max(frontier_max, len(queue))
            for direction, neighbour, is_warp in self.world.neighbors(current):
                if self._is_undesirable(neighbour, navigation_goal.constraints):
                    continue
                tile = self.world.tiles.get(neighbour)
                if is_warp:
                    step_cost = 0
                elif navigation_goal.encounter_mode is EncounterMode.SEEK:
                    step_cost = 1
                else:
                    # Encounter terrain is charged through the explicit
                    # opportunity policy below, rather than double-counted
                    # as an arbitrary terrain weight.
                    step_cost = tile.traversal_cost - int(tile.has_encounters)
                # A directional input while facing another way turns in place
                # in Gen III.  Interaction plans model that as an explicit
                # TURN; ordinary movement plans retain the historical MOVE
                # action contract but charge the implicit turn here.
                next_state = (neighbour, direction)
                turn_cost = int(executable_orientation and facing is not direction)
                if emit_orientation_actions and turn_cost:
                    continue
                # Encounter generation is attached to the executed action,
                # not represented as an arbitrary grass penalty. Gen III's
                # field loop checks standard encounters when the avatar
                # reaches T_TILE_CENTER. A stationary TURN reaches that
                # state too, even though it does not set tookStep.
                encounter = int(not is_warp and tile.has_encounters)
                if navigation_goal.encounter_mode is EncounterMode.SEEK:
                    # Encounter-seeking is intentionally only an architectural
                    # hook in this milestone; it must not inherit normal-mode
                    # avoidance semantics.
                    encounter = 0
                encounter_terrain_move = encounter
                turn_on_encounter = int(turn_cost and self.world.tiles[current].has_encounters)
                turn_on_non_encounter = int(turn_cost and not self.world.tiles[current].has_encounters)
                opportunity_cost = (
                    0 if navigation_goal.encounter_mode is EncounterMode.SEEK else NORMAL_ENCOUNTER_OPPORTUNITY_WEIGHT
                )
                turn_opportunity = turn_cost * int(self.world.tiles[current].has_encounters)
                new_cost = (
                    current_cost[0] + encounter + turn_opportunity,
                    current_cost[1]
                    + NORMAL_MOVEMENT_COST_SCALE * (step_cost + turn_cost)
                    + opportunity_cost * (encounter + turn_opportunity),
                    current_cost[2] + int(not is_warp),
                    current_cost[3] + encounter,
                    current_cost[4] + turn_on_encounter,
                    current_cost[5] + turn_on_non_encounter,
                )
                if new_cost >= costs.get(next_state, (float("inf"),) * 6):
                    continue
                costs[next_state] = new_cost
                came_from[next_state] = (state, direction, is_warp, turn_cost, False)
                heapq.heappush(queue, (priority(new_cost, next_state), next(sequence), next_state, new_cost))
                generated += 1
                frontier_max = max(frontier_max, len(queue))
        if complete_candidates:
            selected_metrics, selected_actions, selected_destination = complete_candidates[0]
            record(True, selected_metrics)
            return NavigationPlan(
                selected_actions, selected_destination, selected_metrics, tuple(item[0] for item in complete_candidates)
            )
        record(False)
        raise NavigationError(f"No legal route from {start} to {navigation_goal.target!r}")

    def plan_many_locations(
        self,
        start: Location,
        targets: tuple[Location, ...],
        *,
        encounter_mode: EncounterMode = EncounterMode.NORMAL,
        algorithm: str = "dijkstra",
        target_groups: tuple[tuple[Location, ...], ...] | None = None,
    ) -> dict[Location, NavigationPlan]:
        """Resolve several ReachLocation goals from one shared state search.

        Cross-map planning has several terminal activation positions, but all
        of them have the same predecessor graph and the same ``(Location,
        facing)`` state.  This deliberately searches the reachable graph once
        and reconstructs the best state for each requested coordinate.  The
        search is exhaustive: unlike arbitrary goal pruning, this is safe for
        the lexicographic objective and for candidates that have no route.

        This helper models the existing ReachLocation contract.  In
        particular, it does not emit stationary turns; the cross-map caller
        adds candidate-specific activation turns exactly as it did before.
        """
        if algorithm not in ("dijkstra", "astar"):
            raise ValueError(f"Unknown navigation search algorithm: {algorithm!r}")
        requested = frozenset(targets)
        if not requested:
            return {}
        profiling = profiling_enabled()
        search_started = perf_counter_ns() if profiling else 0
        sequence = count()
        initial_state = (start, self.world.facing if self.world.facing is not None else None)
        initial_cost = (0, 0, 0, 0, 0, 0)
        costs = {initial_state: initial_cost}
        came_from: dict[tuple[Location, Direction | None], tuple | None] = {initial_state: None}

        def priority(cost, state):
            if algorithm == "dijkstra":
                return cost
            heuristic = self._heuristic(
                state[0],
                ReachLocation(
                    next(iter(requested)),
                ),
                encounter_mode,
            )
            return tuple(cost[index] + heuristic[index] for index in range(6))

        queue = [(priority(initial_cost, initial_state), next(sequence), initial_state, initial_cost)]
        expanded = generated = frontier_max = 0
        results: dict[Location, NavigationPlan] = {}
        groups = target_groups or tuple((target,) for target in requested)
        while queue:
            _, _, state, current_cost = heapq.heappop(queue)
            if current_cost != costs.get(state):
                continue
            expanded += 1
            frontier_max = max(frontier_max, len(queue))
            current, facing = state
            if current in requested:
                # ReachLocation deliberately completes on the first state
                # popped by the ordered frontier.  This is also the old
                # single-goal planner's completion rule; candidate-specific
                # activation turns are appended by the caller afterwards.
                if current not in results:
                    actions = self._unroll(came_from, state, False)
                    results[current] = NavigationPlan(actions, current, self._metrics(actions, encounter_mode))
                    if all(any(location in results for location in group) for group in groups):
                        break
            if self._is_undesirable(current, NavigationGoal(ReachLocation(current)).constraints):
                continue
            if self.world.facing is not None and current in self.world.tiles:
                # ReachLocation plans do not emit turns, but their implicit
                # directional input costs are already represented below.
                pass
            for direction, neighbour, is_warp in self.world.neighbors(current):
                if is_warp:
                    step_cost = 0
                    encounter = 0
                elif encounter_mode is EncounterMode.SEEK:
                    step_cost = 1
                    tile = self.world.tiles[neighbour]
                    encounter = 0
                else:
                    tile = self.world.tiles[neighbour]
                    step_cost = tile.traversal_cost - int(tile.has_encounters)
                    encounter = int(tile.has_encounters)
                turn_cost = int(self.world.facing is not None and facing is not direction)
                turn_on_encounter = int(turn_cost and self.world.tiles[current].has_encounters)
                turn_on_non_encounter = int(turn_cost and not self.world.tiles[current].has_encounters)
                opportunity = encounter + turn_on_encounter
                new_cost = (
                    current_cost[0] + opportunity,
                    current_cost[1] + step_cost + turn_cost + opportunity,
                    current_cost[2] + int(not is_warp),
                    current_cost[3] + encounter,
                    current_cost[4] + turn_on_encounter,
                    current_cost[5] + turn_on_non_encounter,
                )
                next_state = (neighbour, direction)
                if new_cost >= costs.get(next_state, (float("inf"),) * 6):
                    continue
                costs[next_state] = new_cost
                came_from[next_state] = (state, direction, is_warp, turn_cost, False)
                heapq.heappush(queue, (priority(new_cost, next_state), next(sequence), next_state, new_cost))
                generated += 1
                frontier_max = max(frontier_max, len(queue))

        if profiling:
            record_pathfinding_search(
                algorithm=algorithm,
                map=start[0],
                start_state=initial_state,
                goal=f"shared ReachLocation targets={len(requested)}",
                goal_type="SharedReachLocation",
                nodes_expanded=expanded,
                nodes_generated=generated,
                max_frontier_size=frontier_max,
                duration_ms=round((perf_counter_ns() - search_started) / 1_000_000, 3),
                path_found=bool(results),
                final_encounter_opportunity_count=None,
                final_route_cost=None,
                candidate_searches_for_goal=1,
            )
        return results

    def _heuristic(self, location: Location, target: Goal, encounter_mode: EncounterMode) -> tuple[int, ...]:
        """Conservative lexicographic lower bound for experimental A*.

        NORMAL's primary component is zero because future encounter exposure
        cannot be safely bounded above zero.  The secondary lower bound is
        Manhattan distance only for a warp-free local graph: a zero-cost warp
        can invalidate it.  Every ordinary move costs at least one in both
        modes, while turns add non-negative cost.  Returning zero for
        interaction targets with no explicit positions keeps the heuristic
        admissible when the target's geometry is incomplete.
        """
        if any(warp.entry[0] == location[0] for warp in self.world.warps):
            return (0, 0, 0, 0, 0, 0)
        positions: list[Location] = []
        if isinstance(target, ReachLocation):
            positions = [target.location]
        elif isinstance(target, (ActivateTrigger, ReachInteractionPosition)):
            for trigger in self.world.triggers:
                if trigger.trigger_id == target.trigger_id:
                    positions.extend(trigger.activation_locations or trigger.navigation_locations)
        if not positions:
            return (0, 0, 0, 0, 0, 0)
        distance = (
            min(
                abs(location[1][0] - position[1][0]) + abs(location[1][1] - position[1][1])
                for position in positions
                if position[0] == location[0]
            )
            if any(position[0] == location[0] for position in positions)
            else 0
        )
        return (0, distance, 0, 0, 0, 0)

    def _satisfies(self, location: Location, facing: Direction | None, target: Goal) -> bool:
        if isinstance(target, ReachLocation):
            return location == target.location
        if isinstance(target, ActivateTrigger):
            return any(
                target.trigger_id == trigger.trigger_id and self._activation_matches(trigger, location, facing)
                for trigger in self.world.triggers
            )
        if isinstance(target, ReachInteractionPosition):
            return any(
                target.trigger_id == trigger.trigger_id and self._activation_matches(trigger, location, facing)
                for trigger in self.world.triggers
            )
        if isinstance(target, ReachWarp):
            return any(
                (location in warp.activation_locations if warp.activation_locations else warp.entry == location)
                and (
                    (warp.activation_direction is None and warp.required_facing is None)
                    or facing is None
                    or (warp.activation_direction or warp.required_facing) is facing
                )
                and (target.destination is None or warp.destination == target.destination)
                and (target.destination_map is None or warp.destination[0] == target.destination_map)
                for warp in self.world.warps
            )
        raise TypeError(f"Unsupported goal type: {type(target).__name__}")

    def satisfies(self, location: Location, facing: Direction | None, target: Goal) -> bool:
        """Check the complete executable precondition for a goal."""
        return self._satisfies(location, facing, target)

    def _activation_matches(self, trigger, location, facing) -> bool:
        requirements = dict(trigger.activation_requirements)
        if location not in (trigger.activation_locations or trigger.navigation_locations):
            return False
        # Activation positions are source states, not merely coordinates
        # adjacent to a trigger.  Revalidate the live tile here because
        # bindings/perception can describe geometry that is blocked by ROM
        # collision or a currently active object.
        tile = self.world.tiles.get(location)
        if tile is None or tile.blocked:
            return False
        required_facing = requirements.get(location)
        return required_facing is None or facing is None or required_facing is facing

    def _is_undesirable(self, location: Location, constraints) -> bool:
        if location in constraints.avoid_locations:
            return True
        return any(
            location in trigger.locations and trigger.trigger_id in constraints.avoid_trigger_ids
            for trigger in self.world.triggers
        )

    @staticmethod
    def _unroll(came_from, destination_state, emit_orientation_actions: bool) -> tuple[NavigationAction, ...]:
        actions: list[NavigationAction] = []
        current = destination_state
        while came_from[current] is not None:
            previous, direction, is_warp, turn_cost, is_turn = came_from[current]
            source, source_facing = previous
            destination_location, destination_facing = current
            if is_turn:
                actions.append(NavigationAction(NavigationActionType.TURN, direction, source, source))
                current = previous
                continue
            actions.append(
                NavigationAction(
                    NavigationActionType.WARP if is_warp else NavigationActionType.MOVE,
                    direction,
                    source,
                    destination_location,
                )
            )
            current = previous
        actions.reverse()
        return tuple(actions)

    def _metrics(
        self, actions: tuple[NavigationAction, ...], encounter_mode: EncounterMode = EncounterMode.NORMAL
    ) -> NavigationMetrics:
        movement_actions = sum(action.action_type is NavigationActionType.MOVE for action in actions)
        encounter_terrain_moves = sum(
            action.action_type is NavigationActionType.MOVE and self.world.tiles[action.destination].has_encounters
            for action in actions
        )
        turns_on_encounter = sum(
            action.action_type is NavigationActionType.TURN and self.world.tiles[action.source].has_encounters
            for action in actions
        )
        turns_on_non_encounter = sum(
            action.action_type is NavigationActionType.TURN and not self.world.tiles[action.source].has_encounters
            for action in actions
        )
        # Ordinary location plans preserve the existing MOVE-only execution
        # contract, but their diagnostics still account for the implicit turn
        # required before a direction-changing input.
        facing = self.world.facing
        if facing is not None:
            for action in actions:
                if action.action_type is NavigationActionType.TURN:
                    facing = action.direction
                elif action.action_type in (NavigationActionType.MOVE, NavigationActionType.WARP):
                    if facing is not action.direction:
                        if self.world.tiles[action.source].has_encounters:
                            turns_on_encounter += 1
                        else:
                            turns_on_non_encounter += 1
                    facing = action.direction
        encounter_opportunities = encounter_terrain_moves + turns_on_encounter
        if encounter_mode is EncounterMode.SEEK:
            encounter_opportunities = 0
        ordinary_movement_steps = movement_actions - encounter_terrain_moves
        final_facing = self.world.facing
        total_route_cost = sum(
            (
                1
                if encounter_mode is EncounterMode.SEEK
                else self.world.tiles[action.destination].traversal_cost
                - int(self.world.tiles[action.destination].has_encounters)
            )
            for action in actions
            if action.action_type is NavigationActionType.MOVE
        )
        total_route_cost += turns_on_encounter + turns_on_non_encounter
        if encounter_mode is EncounterMode.NORMAL:
            total_route_cost += encounter_opportunities * NORMAL_ENCOUNTER_OPPORTUNITY_WEIGHT
        for action in actions:
            if action.action_type in (NavigationActionType.MOVE, NavigationActionType.TURN, NavigationActionType.WARP):
                final_facing = action.direction
        return NavigationMetrics(
            movement_actions,
            ordinary_movement_steps,
            encounter_opportunities,
            encounter_terrain_moves,
            turns_on_encounter,
            turns_on_non_encounter,
            final_facing,
            total_route_cost,
        )
