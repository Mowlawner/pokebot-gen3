"""Deterministic goal-aware navigation over a small abstract world model."""

import heapq
from time import perf_counter_ns
from itertools import count
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Mapping

from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    Goal,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    SemanticTargetKind,
    EncounterMode,
)
from modules.map_path import Direction
from modules.overworld import (
    MapId,
    Location,
    OverworldObservation,
    TriggerObservation,
    WorldTransition,
    WarpObservation,
    WarpActivation,
    prewarm_static_map_observation,
    static_map_transitions,
)
from modules.trigger_bindings import BindingResolution
from modules.world_navigation import WorldMapGraph, WorldNavigationError, WorldRoute, get_world_map_graph
from modules.stutter_trace import traced
from modules.context import context
from modules.console import diagnostic_print
from modules.profiler import profiled
from modules.profiler import (
    enabled as profiling_enabled,
    count as profile_count,
    record_cross_map_goal_evaluation,
    record_pathfinding_search,
    pathfinding_searches,
)


class NavigationActionType(Enum):
    MOVE = auto()
    TURN = auto()
    WARP = auto()


class TransitionRelevance(Enum):
    RELEVANT = auto()
    IRRELEVANT = auto()
    UNKNOWN = auto()
    BLOCKED = auto()


def classify_transition_relevance(
    transition: WorldTransition,
    target: SemanticTarget | None,
    graph: WorldMapGraph | None = None,
    *,
    currently_reachable: bool | None = None,
) -> TransitionRelevance:
    """Classify a world transition before local movement cost is considered."""
    if currently_reachable is False:
        return TransitionRelevance.BLOCKED
    if target is None or target.target_map is None or transition.destination is None:
        return TransitionRelevance.UNKNOWN
    destination_map = transition.destination[0]
    if destination_map == target.target_map:
        return TransitionRelevance.RELEVANT
    try:
        (graph or get_world_map_graph()).route(destination_map, target.target_map)
    except WorldNavigationError:
        return TransitionRelevance.IRRELEVANT
    return TransitionRelevance.RELEVANT


def transition_world_route(
    transition: WorldTransition,
    target: SemanticTarget | None,
    graph: WorldMapGraph | None = None,
) -> WorldRoute | None:
    """Return the cheapest known route after an observed transition.

    ``None`` is deliberately ambiguous between no semantic target and an
    unresolved/unreachable transition; callers must retain UNKNOWN for the
    latter rather than inventing a destination.
    """
    if target is None or target.target_map is None or transition.destination is None:
        return None
    graph = graph or get_world_map_graph()
    try:
        return graph.route(
            transition.destination[0],
            target.target_map,
            prefer_interior=target.kind is SemanticTargetKind.INTERACTION,
        )
    except WorldNavigationError:
        return None


@dataclass(frozen=True)
class NavigationAction:
    action_type: NavigationActionType
    direction: Direction
    source: Location
    # Unknown destinations are valid for runtime transition execution.  The
    # world planner only emits known destinations today, but the controller
    # can still retain an observed transition until a map change confirms it.
    destination: Location | None
    transition_kind: str | None = None
    run: bool = False


@dataclass(frozen=True)
class RuntimeTransitionObservation:
    transition_id: tuple = ()
    predicted_destination: Location | None = None
    observed_destination: Location | None = None
    valid: bool = True


_runtime_transition_observations: dict[tuple, RuntimeTransitionObservation] = {}
_semantic_cross_map_plan_cache: dict[tuple, tuple["NavigationPlan", WorldRoute | None]] = {}


def clear_semantic_cross_map_plan_cache() -> None:
    """Clear memoized semantic routes at an emulator/session boundary."""
    _semantic_cross_map_plan_cache.clear()


def _semantic_plan_cache_key(world: "NavigationWorld", start: Location, goal: NavigationGoal, target: Goal) -> tuple:
    """Key only the effective inputs that can change a semantic route.

    Static tiles are ROM data and are covered by the active process/session.
    Dynamic occupancy is included because it can block the route.  Transition
    identity and effective destinations are included because runtime warp
    corrections can change executable cross-map edges.  Dialogue/script state
    is intentionally absent: it cannot change a route while the effective
    world remains unchanged, and the controller still refuses to execute
    navigation outside the overworld.
    """
    transitions = tuple(
        sorted((t.kind, t.entry, t.destination, t.required_facing) for t in (world.transitions or world.warps))
    )
    blocked = tuple(sorted(location[1] for location, tile in world.tiles.items() if tile.blocked))
    hazards = tuple(sorted(location for trigger in world.triggers for location in trigger.hazard_locations))
    return (
        start,
        world.facing,
        world.running_shoes,
        repr(target),
        goal.encounter_mode.value,
        goal.encounter_penalty,
        transitions,
        blocked,
        hazards,
    )


def canonical_transition_identity(source: Location, kind: str | None = None) -> tuple:
    return (source[0], source[1], kind or "warp")


def transition_identity(source: Location, kind: str | None = None) -> tuple:
    """Compatibility alias for the canonical activation-source identity."""
    return canonical_transition_identity(source, kind)


def record_runtime_transition_observation(
    source: Location, kind: str | None, predicted: Location | None, observed: Location
) -> RuntimeTransitionObservation:
    record = RuntimeTransitionObservation(transition_identity(source, kind), predicted, observed)
    _runtime_transition_observations[record.transition_id] = record
    return record


def runtime_transition_observation(source: Location, kind: str | None = None):
    return _runtime_transition_observations.get(transition_identity(source, kind))


def clear_runtime_transition_observations() -> None:
    """Clear session-local observations at an emulator/session boundary."""
    _runtime_transition_observations.clear()


def effective_transition(transition: WorldTransition) -> WorldTransition:
    """Return a runtime-adjusted copy, never mutating static ROM metadata."""
    source = transition_approach_position(transition) or transition.entry
    kind = getattr(transition, "kind", None)
    observation = runtime_transition_observation(source, kind)
    if observation is None or not observation.valid or observation.observed_destination is None:
        return transition
    if context.debug and getattr(context, "debug_trace", False):
        diagnostic_print(
            lambda: (
                f"TRANSITION_RUNTIME_OVERRIDE_USED source={transition.entry!r} "
                f"kind={transition.kind!r} predicted={transition.destination!r} "
                f"observed={observation.observed_destination!r}"
            ),
            trace=True,
        )
    # Reconcile only the destination learned from the live ROM.  Rebuilding
    # the object as a generic WorldTransition silently discards executable
    # metadata (notably DIRECTIONAL_STEP/arrow-warp activation), which can
    # make a north-facing arrow warp fall back to the avatar's current
    # direction and drive the player away from its source tile.
    return replace(transition, destination=observation.observed_destination)


def transitions_match(left: WorldTransition, right: WorldTransition) -> bool:
    """Compare transitions by executable identity, not transient metadata.

    Live observations can be normalized by ``effective_transition`` (for
    example after a runtime transition correction), while a tactical goal may
    still contain the raw observation.  Dataclass equality would then reject
    the same ROM transition because activation metadata differs.  The source
    activation location, kind, and destination are the stable identity used
    by planning and execution.
    """
    left_source = transition_approach_position(left) or left.entry
    right_source = transition_approach_position(right) or right.entry
    return (
        left.kind == right.kind
        and left_source == right_source
        and left.entry == right.entry
        and left.destination == right.destination
    )


@dataclass(frozen=True)
class NavigableTile:
    location: Location
    blocked: bool = False
    allowed_directions: frozenset[Direction] | None = None
    traversal_cost: int = 1
    has_encounters: bool = False
    cannot_run: bool = False
    forced_movement_to: Mapping[Direction, tuple[Location, int]] | None = None
    # RSE/FRLG elevation is part of collision semantics. In particular,
    # elevation 1 is normally water while walkable ground is elevation 3.
    # Keep it optional so synthetic worlds and incomplete observations retain
    # their historical, topology-only behavior.
    elevation: int | None = None


class _DynamicTileMapping(Mapping[Location, NavigableTile]):
    def __init__(self, static_tiles: Mapping[Location, NavigableTile], blocked: frozenset[tuple[int, int]]):
        self._static_tiles = static_tiles
        self._blocked = blocked

    def __getitem__(self, location: Location) -> NavigableTile:
        tile = self._static_tiles[location]
        if location[1] not in self._blocked or tile.blocked:
            return tile
        return NavigableTile(
            location=tile.location,
            blocked=True,
            allowed_directions=tile.allowed_directions,
            traversal_cost=tile.traversal_cost,
            has_encounters=tile.has_encounters,
            cannot_run=tile.cannot_run,
            forced_movement_to=tile.forced_movement_to,
            elevation=tile.elevation,
        )

    def __iter__(self):
        return iter(self._static_tiles)

    def __len__(self):
        return len(self._static_tiles)


class _LazyStaticTileMapping(Mapping[Location, NavigableTile]):
    """Read-through immutable per-map tile models for global search.

    The global search asks for individual locations through ``get``; avoid
    materializing every graph map up front.  Each map is still constructed at
    most once and is retained by the existing static navigation cache.
    """

    def __init__(self, initial: Mapping[Location, NavigableTile], map_ids):
        self._models = {
            map_id: dict((location, tile) for location, tile in initial.items() if location[0] == map_id)
            for map_id in {location[0] for location in initial}
        }
        self._map_ids = frozenset(map_ids)
        self._loaded = set(self._models)

    def _load(self, map_id) -> dict[Location, NavigableTile]:
        if map_id not in self._loaded:
            try:
                static_tiles = prewarm_static_map_observation(map_id)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # Synthetic worlds may intentionally omit ROM metadata for a
                # destination map.  Preserve the previous optional-cache
                # behavior and let explicit transition endpoints suffice.
                self._models.setdefault(map_id, {})
                self._loaded.add(map_id)
                return self._models[map_id]
            prewarm_navigation_tiles(map_id, static_tiles)
            self._models[map_id] = _static_navigation_tiles[map_id][1]
            self._loaded.add(map_id)
            profile_count("global_navigation_new_map_models")
            profile_count("global_navigation_materialized_tiles", len(self._models[map_id]))
        else:
            profile_count("global_navigation_cached_map_model_lookups")
        return self._models.get(map_id, {})

    def __getitem__(self, location: Location) -> NavigableTile:
        return self._load(location[0])[location]

    def get(self, location: Location, default=None):
        return self._load(location[0]).get(location, default)

    def __iter__(self):
        # Search does not iterate the complete world.  Expose already-loaded
        # models for diagnostics and Mapping compatibility without triggering
        # eager construction of unrelated maps.
        return iter(tuple(location for model in self._models.values() for location in model))

    def __len__(self):
        return sum(len(model) for model in self._models.values())


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
                    tile.location,
                    tile.blocked,
                    tile.walkable_neighbors,
                    tile.traversal_cost,
                    tile.has_encounters,
                    tile.cannot_run,
                    getattr(tile, "forced_movement_to", None),
                    getattr(tile, "elevation", None),
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
    transitions: tuple[WorldTransition, ...] = ()
    running_shoes: bool = False
    player_elevation: int | None = None
    surfing: bool = False
    _transitions_by_source: dict[Location, tuple[WorldTransition, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index executable transition sources without changing their order."""
        indexed: dict[Location, list[WorldTransition]] = {}
        for transition in self.transitions or self.warps:
            source = transition.entry
            if getattr(transition, "kind", None) == "map_connection":
                source = transition_approach_position(transition)
            if source is not None:
                indexed.setdefault(source, []).append(transition)
            # Most warps activate on their entry tile, but ROM-defined
            # field-effect transitions (for example Center escalators) expose
            # a distinct source tile.  Index those source states too so
            # ReachWarp checks do not scan every transition for every search
            # node.
            for activation_source in getattr(transition, "activation_locations", ()):
                indexed.setdefault(activation_source, []).append(transition)
        object.__setattr__(
            self,
            "_transitions_by_source",
            {source: tuple(transitions) for source, transitions in indexed.items()},
        )

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
            transitions=tuple(
                effective_transition(t) for t in (getattr(observation, "transitions", ()) or observation.warps)
            ),
            running_shoes=getattr(observation, "running_shoes", False),
            player_elevation=(
                getattr(observation, "player_current_elevation", None)
                if getattr(observation, "player_current_elevation", None) is not None
                else getattr(observation, "player_elevation", None)
            ),
            surfing=getattr(observation, "surfing", False),
        )

    def _elevation_allows_step(self, source: Location, destination: Location) -> bool:
        """Apply the coarse elevation rule used by legacy pathfinding.

        Emerald exposes water with ordinary walk collision. Without this
        guard, an early-game route can select a visually adjacent pond tile
        as though it were land. Elevation 1 -> 3 is a valid exit; 3 -> 1
        requires an already active Surf state. Unknown elevations remain
        permissive for synthetic or incomplete observations.
        """
        source_tile = self.tiles.get(source)
        destination_tile = self.tiles.get(destination)
        if source_tile is None or destination_tile is None:
            return True
        source_elevation = source_tile.elevation
        if source_elevation is None:
            source_elevation = self.player_elevation
        destination_elevation = destination_tile.elevation
        if source_elevation is None or destination_elevation is None:
            return True
        if source_elevation == destination_elevation:
            return True
        if source_elevation in (0, 15) or destination_elevation in (0, 15):
            return True
        if source_elevation == 1 and destination_elevation == 3:
            return True
        if source_elevation == 3 and destination_elevation == 1:
            return self.surfing
        return False

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
        transitions = self.transitions or self.warps
        transition_entries = {transition.entry for transition in transitions}
        for direction, coordinate in candidates:
            destination = (location[0], coordinate)
            neighbour = self.tiles.get(destination)
            if (
                neighbour is not None
                and not neighbour.blocked
                and self._elevation_allows_step(location, destination)
                and (
                    destination in transition_entries
                    or neighbour.allowed_directions is None
                    or direction in neighbour.allowed_directions
                )
            ):
                forced = (neighbour.forced_movement_to or {}).get(direction)
                result.append((direction, forced[0] if forced is not None else destination, False))
        for transition in self._transitions_by_source.get(location, ()):
            if transition.destination is not None and self._elevation_allows_step(location, transition.destination):
                result.append((self._warp_direction(location, transition), transition.destination, True))
        return tuple(result)

    @staticmethod
    def _warp_direction(location: Location, warp: WorldTransition) -> Direction:
        # Entering a warp is an input in the direction of the destination tile
        # only when the map data supplies one; otherwise any adjacent route can
        # reach the entry and the direction is a stable default.
        return warp.required_facing if warp.required_facing is not None else Direction.South


@dataclass(frozen=True)
class NavigationPlan:
    actions: tuple[NavigationAction, ...]
    destination: Location | None
    metrics: "NavigationMetrics | None" = None
    candidate_metrics: tuple["NavigationMetrics", ...] = ()
    forced_trainer_exposure: bool = False


@dataclass(frozen=True)
class RouteCheckpoint:
    """An observation boundary in a complete navigation route.

    Coordinates are deliberately optional: an interaction target can be
    resolved by fresh perception after arriving on its target map.
    """

    kind: str
    action_index: int
    expected_map: MapId | None
    expected_coordinate: tuple[int, int] | None = None
    expected_facing: Direction | None = None
    transition_kind: str | None = None


@dataclass
class RoutePlan:
    """Persistent execution state for one complete NavigationPlan."""

    actions: tuple[NavigationAction, ...]
    checkpoints: tuple[RouteCheckpoint, ...]
    target: object | None = None
    metrics: "NavigationMetrics | None" = None
    current_checkpoint: int = 0
    route_id: int = field(default_factory=lambda: next(_route_plan_ids))

    @classmethod
    def from_navigation_plan(cls, plan: NavigationPlan, target=None) -> "RoutePlan":
        checkpoints = []
        for index, action in enumerate(plan.actions):
            if action.action_type is NavigationActionType.WARP:
                checkpoints.append(
                    RouteCheckpoint(
                        "transition_precondition",
                        index,
                        action.source[0],
                        action.source[1],
                        action.direction,
                        action.transition_kind,
                    )
                )
                checkpoints.append(
                    RouteCheckpoint(
                        "transition_postcondition",
                        index + 1,
                        action.destination[0] if action.destination else None,
                        action.destination[1] if action.destination else None,
                        transition_kind=action.transition_kind,
                    )
                )
        if plan.destination is not None:
            unresolved_interaction = (
                getattr(getattr(target, "kind", None), "name", None) == "INTERACTION"
                and getattr(target, "location", None) is None
            )
            checkpoints.append(
                RouteCheckpoint(
                    "target_map",
                    len(plan.actions),
                    plan.destination[0],
                    None if unresolved_interaction else plan.destination[1],
                )
            )
        return cls(plan.actions, tuple(checkpoints), target=target, metrics=plan.metrics)

    @property
    def complete(self) -> bool:
        return self.current_checkpoint >= len(self.checkpoints) and not self.actions


_route_plan_ids = count(1)


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


@dataclass(frozen=True)
class IntermediateRouteAnalysis:
    """Cost of completing a goal after visiting one intermediate goal."""

    destination: Goal
    total_cost: int | None
    detour: int | None
    reachable: bool
    first_route: NavigationPlan | None = None
    continuation_route: NavigationPlan | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RouteAnalysis:
    """Comparable routes produced by the normal goal-aware navigator."""

    goal: Goal
    normal_route: NavigationPlan | None
    normal_cost: int | None
    candidates: tuple[IntermediateRouteAnalysis, ...] = ()


class RouteCostAnalyzer:
    """Analyze composed routes using the exact campaign navigation planner.

    This is deliberately domain-neutral.  Both the final goal and an
    intermediate destination are normal navigation goals, so semantic goals
    such as ``ReachWarp`` and ``ActivateTrigger`` retain their meaning.
    """

    def __init__(self, world: NavigationWorld, *, graph: WorldMapGraph | None = None, planner=None):
        self.world = world
        self.graph = graph
        self._planner = planner or plan_with_world_navigation

    @staticmethod
    def _cost(plan: NavigationPlan | None) -> int | None:
        return plan.metrics.total_route_cost if plan is not None and plan.metrics is not None else None

    def analyze(
        self,
        start: Location,
        goal: Goal,
        intermediate_destinations: tuple[Goal, ...] = (),
        *,
        algorithm: str = "dijkstra",
    ) -> RouteAnalysis:
        """Return the normal route and independently composed candidate routes."""

        def plan(world, origin, target):
            result = self._planner(world, origin, target, self.graph, algorithm=algorithm)
            return result[0] if isinstance(result, tuple) else result

        try:
            normal = plan(self.world, start, goal)
        except NavigationError as error:
            normal = None
            normal_cost = None
            normal_error = str(error)
        else:
            normal_cost = self._cost(normal)
            normal_error = None

        candidates = []
        for destination in intermediate_destinations:
            first = continuation = None
            try:
                first = plan(self.world, start, destination)
                if first.destination is None:
                    raise NavigationError("intermediate route has no terminal location")
                continuation_world = self.world
                if first.metrics is not None and first.metrics.final_facing is not None:
                    continuation_world = replace(self.world, facing=first.metrics.final_facing)
                continuation = plan(continuation_world, first.destination, goal)
                first_cost = self._cost(first)
                continuation_cost = self._cost(continuation)
                total = (
                    first_cost + continuation_cost if first_cost is not None and continuation_cost is not None else None
                )
                if total is None or normal_cost is None:
                    raise NavigationError("route metrics unavailable")
                candidates.append(
                    IntermediateRouteAnalysis(destination, total, total - normal_cost, True, first, continuation)
                )
            except NavigationError as error:
                candidates.append(
                    IntermediateRouteAnalysis(
                        destination, None, None, False, first, continuation, str(error) or normal_error
                    )
                )
        return RouteAnalysis(goal, normal, normal_cost, tuple(candidates))


# NORMAL is deliberately opportunity-first.  The second component keeps the
# result practical and deterministic after exposure has been minimized; it is
# not allowed to trade an extra encounter opportunity for fewer steps.
NORMAL_ENCOUNTER_OPPORTUNITY_WEIGHT = 1
NORMAL_MOVEMENT_COST_SCALE = 1


def navigation_candidate_key(metrics: NavigationMetrics, encounter_mode: EncounterMode, candidate_id: str = ""):
    """Canonical ordering for complete local candidate plans."""
    if encounter_mode in (EncounterMode.AVOID, EncounterMode.NORMAL):
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


def encounter_cost_components(mode: EncounterMode, penalty: int = 8) -> tuple[int, int]:
    """Return (lexicographic exposure cost, weighted cost) per opportunity."""
    if mode in (EncounterMode.AVOID, EncounterMode.NORMAL):
        return 1, 1
    if mode is EncounterMode.MOSTLY_AVOID:
        return 0, max(0, penalty)
    # IGNORE and SEEK intentionally do not penalize opportunities. SEEK is a
    # campaign objective signal; selecting the encounter area remains the
    # responsibility of the higher-level objective planner.
    return 0, 0


def goal_target_map(world: NavigationWorld, goal: Goal) -> MapId | None:
    if isinstance(goal, SemanticTarget):
        return goal.target_map or (goal.location[0] if goal.location else None)
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


def _direction_between(source: tuple[int, int], destination: tuple[int, int]) -> Direction | None:
    """Return the cardinal input which moves ``source`` onto ``destination``."""
    dx = destination[0] - source[0]
    dy = destination[1] - source[1]
    if (dx, dy) == (0, -1):
        return Direction.North
    if (dx, dy) == (1, 0):
        return Direction.East
    if (dx, dy) == (0, 1):
        return Direction.South
    if (dx, dy) == (-1, 0):
        return Direction.West
    return None


def map_connection_approach_position(
    boundary: tuple[int, int] | WorldTransition, direction: Direction | None = None
) -> tuple[int, int] | None:
    """Return the ordinary predecessor of a connection boundary tile.

    A ROM MapConnection describes the edge coordinate at which the field
    engine accepts the off-map crossing input.  Local A* reaches this adjacent
    walkable tile; the planner then appends the terminal boundary step and
    crossing input without making the blocked edge an ordinary search node.
    """
    if isinstance(boundary, WorldTransition):
        direction = boundary.required_facing
        boundary = boundary.entry[1]
    if direction is None:
        return None
    if direction is Direction.North:
        return boundary[0], boundary[1] + 1
    if direction is Direction.South:
        return boundary[0], boundary[1] - 1
    if direction is Direction.East:
        return boundary[0] - 1, boundary[1]
    if direction is Direction.West:
        return boundary[0] + 1, boundary[1]
    raise NavigationError("map connection has no crossing direction")


def transition_approach_position(transition: WorldTransition) -> Location | None:
    """Return the walkable predecessor of a map connection boundary."""
    if getattr(transition, "kind", None) != "map_connection":
        return None
    # An observed connection without a crossing direction is still useful as
    # topology metadata, but it is not locally executable.  Keep it out of
    # the activation index until ROM metadata or a later observation supplies
    # the direction; do not make world construction fail during that gap.
    if getattr(transition, "required_facing", None) is None:
        return None
    return (
        transition.entry[0],
        map_connection_approach_position(transition.entry[1], transition.required_facing),
    )


def _global_navigation_world(
    observation_world: NavigationWorld,
    graph: WorldMapGraph,
    *,
    enrich_maps: tuple[MapId, ...] = (),
) -> NavigationWorld:
    """Build the lazy static world overlay used by the exact-state search.

    Map metadata is materialized only for maps reachable through the coarse
    graph; dynamic occupancy remains applied only to the currently observed
    map.  ``WorldEdge`` candidates stay separate, including connection spans.
    """
    tiles: Mapping[Location, NavigableTile] = _LazyStaticTileMapping(
        observation_world.tiles,
        {edge.source_map for edge in graph.edges} | {edge.destination_map for edge in graph.edges},
    )
    transitions: list[WorldTransition] = list(observation_world.transitions or observation_world.warps)
    rom_transitions: dict[MapId, tuple[WorldTransition, ...]] = {}
    for map_id in enrich_maps:
        try:
            rom_transitions[map_id] = static_map_transitions(map_id)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            # A partially available map should not make a semantic route
            # disappear. The graph transition remains a valid fallback.
            continue
    known_entries = {(t.kind, t.entry, t.destination) for t in transitions}
    for edge in graph.edges:
        for source, destination in zip(edge.source_coordinates, edge.destination_coordinates):
            transition = WorldTransition(
                (edge.source_map, source),
                (edge.destination_map, destination),
                required_facing=(
                    edge.required_facing
                    if isinstance(edge.required_facing, Direction)
                    else _world_edge_direction(edge) if edge.kind == "connection" else None
                ),
                kind="map_connection" if edge.kind == "connection" else "warp",
            )
            if edge.source_map in rom_transitions and transition.kind == "warp":
                rom_transition = next(
                    (
                        candidate
                        for candidate in rom_transitions[edge.source_map]
                        if candidate.entry == transition.entry and candidate.destination == transition.destination
                    ),
                    None,
                )
                if rom_transition is not None:
                    transition = rom_transition
            transition = effective_transition(transition)
            if (transition.kind, transition.entry, transition.destination) not in known_entries:
                transitions.append(transition)
                known_entries.add((transition.kind, transition.entry, transition.destination))
            if transition.destination is not None:
                # Destination endpoints are known without loading the whole
                # destination map; the lazy mapping resolves their neighbors
                # only if the search actually visits that map.
                model = tiles._models.setdefault(transition.destination[0], {})
                model.setdefault(
                    transition.destination, NavigableTile(transition.destination, False, frozenset(Direction))
                )
    return NavigationWorld(
        tiles=tiles,
        warps=observation_world.warps,
        triggers=observation_world.triggers,
        bindings=observation_world.bindings,
        facing=observation_world.facing,
        transitions=tuple(transitions),
        running_shoes=observation_world.running_shoes,
        player_elevation=observation_world.player_elevation,
        surfing=observation_world.surfing,
    )


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
    # A concrete observed boundary is already the runtime transition to use.
    # Plan only its local approach here; global graph search is for unresolved
    # semantic destinations.  This also makes dynamic occupancy fail fast
    # instead of exploring every map while an NPC temporarily blocks the
    # boundary.
    if (
        isinstance(navigation_goal.target, ReachWarp)
        and navigation_goal.target.warp is not None
        and target_map != start[0]
    ):
        # ``_observed_exit_goal`` has already selected an executable ROM
        # transition from the current perception.  Searching the global world
        # again for that exact door/arrow warp is both redundant and costly:
        # every expanded state would compare its transition set against the
        # selected transition.  More importantly, a large Emerald world can
        # spend long enough in that comparison that the frame loop stops
        # advancing.  Exact observed transitions are local tactical goals;
        # reserve global search for unresolved semantic destinations.
        return plan_observed_warp_locally(world, start, navigation_goal.target), None
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
    normalized_target = (
        navigation_goal.target if isinstance(navigation_goal.target, SemanticTarget) else SemanticTarget.map(target_map)
    )
    cache_key = None
    if isinstance(normalized_target, SemanticTarget):
        # Use the observed world for the key, before lazy destination maps are
        # materialized; otherwise the cache key would change as the search
        # itself discovers more maps.
        cache_key = _semantic_plan_cache_key(world, start, navigation_goal, normalized_target)
        cached = _semantic_cross_map_plan_cache.get(cache_key)
        if cached is not None:
            profile_count("semantic_cross_map_plan_cache_hits")
            return cached
        profile_count("semantic_cross_map_plan_cache_misses")
    # Search the exact finite world graph. The old first-edge candidate code
    # remains below for compatibility with callers that explicitly require a
    # selected observed warp, but ordinary semantic cross-map goals use the
    # global state search.
    global_world = _global_navigation_world(
        world,
        graph,
        enrich_maps=tuple(route.maps[:-1]),
    )
    if isinstance(navigation_goal.target, ReachWarp):
        global_goal = navigation_goal.target
        # The static graph's connection direction is map metadata, while the
        # live observation carries the executable facing.  They can disagree
        # at edge endpoints (the metadata direction describes the adjoining
        # map, not the input direction on the current map).  Ensure the
        # selected observed transition is present in the global state model so
        # a specific ReachWarp is not rendered unreachable by that mismatch.
        selected = navigation_goal.target.warp
        if selected is not None and selected.kind == "map_connection":
            transitions = list(global_world.transitions)
            selected_source = transition_approach_position(selected)
            transitions = [
                transition
                for transition in transitions
                if not (
                    transition.kind == "map_connection"
                    and transition_approach_position(transition) == selected_source
                    and transition.destination == selected.destination
                )
            ]
            transitions.append(selected)
            global_world = replace(global_world, transitions=tuple(transitions))
    else:
        # An unresolved interaction is intentionally a map-boundary goal.
        # Runtime observation resolves its interaction after arrival.
        global_goal = normalized_target
        if normalized_target.location is not None:
            global_goal = ReachLocation(normalized_target.location)
    plan = GoalAwareNavigator(global_world).plan(
        start,
        global_goal,
        algorithm=algorithm,
        # Semantic campaign routes expose the crossing input as the action
        # from the executable approach state.  Preserve the legacy boundary
        # source representation for exact ReachLocation callers.
        connection_source_is_approach=isinstance(navigation_goal.target, SemanticTarget),
    )
    if cache_key is not None:
        _semantic_cross_map_plan_cache[cache_key] = (plan, route)
    if context.debug and getattr(context, "debug_trace", False):
        diagnostic_print(
            lambda: (
                f"GLOBAL_ROUTE_FOUND objective={getattr(context, 'campaign_objective', None)!r} "
                f"target={navigation_goal.target!r} start={start!r} target_map={target_map!r} "
                f"cost={plan.metrics.total_route_cost if plan.metrics else None!r} steps={len(plan.actions)}"
            ),
            trace=True,
        )
        for index, action in enumerate(plan.actions, 1):
            if action.action_type is NavigationActionType.WARP:
                diagnostic_print(
                    f"GLOBAL_ROUTE_STEP {index:02d} {action.transition_kind or 'WARP'} "
                    f"source={action.source!r} destination={action.destination!r}",
                    trace=True,
                )

    return plan, route
    edge = route.edges[0]
    candidates = tuple(
        (source_map, coordinates)
        for coordinates, source_map in zip(edge.source_coordinates, (edge.source_map,) * len(edge.source_coordinates))
        if source_map == start[0]
    )
    selected_warp = navigation_goal.target.warp if isinstance(navigation_goal.target, ReachWarp) else None
    if selected_warp is not None:
        candidates = tuple(candidate for candidate in candidates if candidate == selected_warp.entry)
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
    runtime_transitions = {
        transition.entry[1]: transition
        for transition in (world.transitions or world.warps)
        if transition.entry[0] == edge.source_map
        and transition.destination is not None
        and transition.destination[0] == edge.destination_map
        and (transition.kind == edge.kind or (edge.kind == "connection" and transition.kind == "map_connection"))
    }
    candidate_choices: list[tuple[str, tuple[int, int], tuple[tuple[int, int, Direction | None], ...]]] = []
    for candidate_index, (_, coordinates) in enumerate(candidates):
        activation_candidates = (
            (coordinates[0], coordinates[1] - 1, Direction.South),
            (coordinates[0] + 1, coordinates[1], Direction.West),
            (coordinates[0], coordinates[1] + 1, Direction.North),
            (coordinates[0] - 1, coordinates[1], Direction.East),
        )
        warp = runtime_transitions.get(coordinates)
        # A selected observed warp is stronger evidence than the static edge
        # metadata.  Keep it available even when the ROM edge has a different
        # destination coordinate or an unresolved/dynamic representation.
        if warp is None and selected_warp is not None and selected_warp.entry == (edge.source_map, coordinates):
            if selected_warp.destination is not None and selected_warp.destination[0] == edge.destination_map:
                warp = selected_warp
        if edge.kind == "connection":
            connection_direction = (
                warp.required_facing
                if warp is not None and warp.required_facing is not None
                else _world_edge_direction(edge)
            )
            approach = map_connection_approach_position(coordinates, connection_direction)
            choices = (
                (
                    approach[0],
                    approach[1],
                    connection_direction,
                ),
            )
        elif warp is not None and warp.activation_locations:
            choices = tuple(
                (
                    source[1][0],
                    source[1][1],
                    (
                        warp.activation_direction
                        if warp.activation_direction is not None
                        else _direction_between(source[1], coordinates)
                    ),
                )
                for source in warp.activation_locations
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
        encounter_penalty=navigation_goal.encounter_penalty,
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
            runtime_warp = runtime_transitions.get(source_coordinates)
            activation_position = (
                source_coordinates[0]
                - (activation_direction is Direction.East)
                + (activation_direction is Direction.West),
                source_coordinates[1]
                - (activation_direction is Direction.South)
                + (activation_direction is Direction.North),
            )
            turn_location = (
                (start[0], activation_position)
                if edge.kind == "connection"
                else (start[0], source_coordinates if runtime_warp is not None else activation_position)
            )
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
    from modules.route102_diagnostics import emit as route102_emit

    ranked_candidates = []
    for item in candidate_plans:
        plan = completed_local_plan(item)
        metrics = plan.metrics
        ranked_candidates.append(
            {
                "candidate_id": item[3],
                "source_coordinates": item[1],
                "destination": getattr(runtime_transitions.get(item[1]), "destination", None),
                "transition_direction": getattr(item[2], "name", None),
                "activation_state": "active",
                "local_reachability": metrics is not None,
                "local_route_cost": metrics.total_route_cost if metrics else None,
                "downstream_route_cost": None,
                "total_route_cost": metrics.total_route_cost if metrics else None,
                "ranking_tuple": candidate_key(item),
                "selected": item[3] == selected_candidate_id,
            }
        )
    route102_emit(
        "candidate_selection",
        current_map=start[0],
        current_coordinates=start[1],
        navigation_goal=repr(navigation_goal),
        candidates=tuple(candidate_records),
        ranked_candidates=ranked_candidates,
        selected_candidate=selected_candidate_id,
        anti_oscillation=None,
    )
    local_plan = completed_local_plan((local_plan, source_coordinates, activation_direction, selected_candidate_id))
    for candidate_record in candidate_records:
        if candidate_record["candidate_id"] == selected_candidate_id:
            candidate_record["status"] = "selected"
            candidate_record["reason"] = "canonical_navigation_objective"
        elif candidate_record["status"] == "pathfound":
            candidate_record["reason"] = "pathfound_but_not_selected"
    trace = getattr(context, "stutter_trace", None)
    if trace is not None:
        trace.mark(
            "navigation_candidate_selection",
            {
                "map": start[0],
                "position": start[1],
                "candidates": tuple(candidate_records),
                "selected_candidate": selected_candidate_id,
                "selection_reason": "canonical_navigation_objective",
            },
        )
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
    transition_direction = activation_direction if activation_direction is not None else _world_edge_direction(edge)
    runtime_warp = runtime_transitions.get(source_coordinates)
    if (
        runtime_warp is None
        and selected_warp is not None
        and selected_warp.entry == (edge.source_map, source_coordinates)
    ):
        if selected_warp.destination is not None and selected_warp.destination[0] == edge.destination_map:
            runtime_warp = selected_warp
    activation_position = None
    if activation_direction is not None:
        activation_position = (
            source_coordinates[0] - (activation_direction is Direction.East) + (activation_direction is Direction.West),
            source_coordinates[1]
            - (activation_direction is Direction.South)
            + (activation_direction is Direction.North),
        )
    directional_activation = runtime_warp is not None and runtime_warp.activation is WarpActivation.DIRECTIONAL_STEP
    connection_activation = edge.kind == "connection"
    if connection_activation and runtime_warp is not None and runtime_warp.required_facing is not None:
        transition_direction = runtime_warp.required_facing
    if connection_activation:
        if activation_position is None or activation_direction is None:
            raise NavigationError("map connection has no reachable boundary predecessor")
        # The edge coordinate is the final in-map tile, not the crossing
        # input's source predecessor.  Keep it outside ordinary A* expansion
        # (it may be marked blocked), but represent the observed step onto it
        # explicitly before dispatching the fresh off-map input.
        boundary_move = NavigationAction(
            NavigationActionType.MOVE,
            activation_direction,
            (start[0], activation_position),
            (start[0], source_coordinates),
        )
        local_plan = NavigationPlan(
            local_plan.actions + (boundary_move,),
            (start[0], source_coordinates),
        )
    if directional_activation:
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
        transition_direction = (
            runtime_warp.required_facing
            if runtime_warp.required_facing is not None
            else runtime_warp.activation_direction
        )
        if transition_direction is None:
            raise NavigationError("directional warp has no activation direction")
    elif runtime_warp is not None and runtime_warp.activation is WarpActivation.STEP_ON and not connection_activation:
        # STEP_ON is the movement onto the entry tile.  The synthetic WARP
        # action is retained so the executor can verify the map transition,
        # but its input must be the approach movement, never a graph-edge
        # facing hint or an arbitrary default direction.
        if activation_position is None or activation_direction is None:
            raise NavigationError("step-on warp has no reachable activation side")
        transition_direction = activation_direction
    transition = NavigationAction(
        NavigationActionType.WARP,
        transition_direction,
        (
            (start[0], activation_position)
            if connection_activation and activation_position is not None
            else (
                (start[0], source_coordinates)
                if directional_activation
                else (start[0], activation_position or source_coordinates)
            )
        ),
        (edge.destination_map, destination_coordinates),
        transition_kind="map_connection" if edge.kind == "connection" else edge.kind,
    )
    actions = local_plan.actions + (transition,)
    return (
        NavigationPlan(
            actions, local_plan.destination, GoalAwareNavigator(world)._metrics(actions, navigation_goal.encounter_mode)
        ),
        route,
    )


def plan_observed_warp_locally(
    world: NavigationWorld,
    start: Location,
    goal: ReachWarp,
) -> NavigationPlan:
    """Plan to an exact observed warp without consulting the map graph."""
    selected = goal.warp
    if selected is None or not any(
        transitions_match(selected, observed) for observed in (world.transitions or world.warps)
    ):
        raise NavigationError("observed transition is not present in the current perception")
    if selected.destination is None:
        raise NavigationError("observed warp destination is unresolved")
    if goal.destination is not None and selected.destination != goal.destination:
        raise NavigationError("observed warp destination no longer matches the goal")
    if goal.destination_map is not None and selected.destination[0] != goal.destination_map:
        raise NavigationError("observed warp map no longer matches the goal")
    # A step-on door warp exposes both its entry tile and the adjacent tiles
    # from which the field engine will accept the step.  ReachWarp's generic
    # evaluator treats those activation tiles as satisfied, but the ROM still
    # requires the player to enter the entry tile.  Keep planning until the
    # entry tile instead of returning an empty plan (which the controller
    # interprets as a completed goal and re-observes forever).
    if (
        selected.activation is WarpActivation.STEP_ON
        and start != selected.entry
        and start in selected.activation_locations
    ):
        local_plan = GoalAwareNavigator(world).plan(start, ReachLocation(selected.entry))
        return NavigationPlan(
            local_plan.actions,
            selected.entry,
            GoalAwareNavigator(world)._metrics(local_plan.actions),
        )
    # ReachWarp's local terminal condition is deliberately the entry/source
    # state.  For directional-step warps that is not the transition itself:
    # the ROM consumes one additional directional input while standing on the
    # entry tile.  The cross-map planner normally appends this action after
    # its local plan; preserve that contract for the graph-independent path.
    if selected.kind == "map_connection":
        direction = selected.required_facing
        if direction is None:
            raise NavigationError("map connection has no crossing direction")
        approach = transition_approach_position(selected)
        if approach is None:
            raise NavigationError("map connection has no approach position")
        if start == selected.entry:
            local_actions = ()
        else:
            # The local sub-plan must end at the approach tile.  Planning to
            # ReachWarp here would synthesize a second WARP action from the
            # normalized/static transition and can carry the wrong facing;
            # the observed transition is appended explicitly below.
            local_plan = GoalAwareNavigator(world).plan(start, ReachLocation(approach))
            boundary_move = NavigationAction(
                NavigationActionType.MOVE,
                direction,
                approach,
                selected.entry,
            )
            local_actions = local_plan.actions + (boundary_move,)
        crossing = NavigationAction(
            NavigationActionType.WARP,
            direction,
            selected.entry,
            selected.destination,
            transition_kind="map_connection",
        )
        actions = local_actions + (crossing,)
        return NavigationPlan(
            actions,
            selected.entry,
            GoalAwareNavigator(world)._metrics(actions),
        )
    local_plan = GoalAwareNavigator(world).plan(start, goal)
    # A recovery handoff can arrive on a step-on door tile via a
    # ReachLocation goal.  In that case the local planner may regard the
    # ReachWarp position as already satisfied and return no action, even
    # though the ROM has not crossed the door yet.  Preserve the transition
    # as an executable action whenever the observed warp is still the live
    # destination.
    if (
        selected.activation is WarpActivation.STEP_ON
        and start == selected.entry
        and not any(action.action_type is NavigationActionType.WARP for action in local_plan.actions)
    ):
        direction = selected.activation_direction
        if direction is None:
            direction = selected.required_facing
        # A plain STEP_ON warp has no distinct second directional input, so
        # carry the approach/final facing into its synthetic verification
        # action.  Explicit ROM direction metadata must still preserve North
        # (enum value 0) correctly.
        if direction is None:
            direction = world.facing
        if direction is None:
            direction = Direction.South
        crossing = NavigationAction(
            NavigationActionType.WARP,
            direction,
            selected.entry,
            selected.destination,
            transition_kind=selected.kind,
        )
        return NavigationPlan(
            local_plan.actions + (crossing,),
            selected.entry,
            GoalAwareNavigator(world)._metrics(local_plan.actions + (crossing,)),
        )
    if selected.activation is WarpActivation.DIRECTIONAL_STEP:
        direction = (
            selected.activation_direction if selected.activation_direction is not None else selected.required_facing
        )
        if direction is None:
            raise NavigationError("directional observed warp has no activation direction")
        activation_source = selected.entry
        activation = NavigationAction(
            NavigationActionType.WARP,
            direction,
            activation_source,
            selected.destination,
        )
        actions = local_plan.actions + (activation,)
        return NavigationPlan(
            actions,
            local_plan.destination,
            GoalAwareNavigator(world)._metrics(actions),
        )
    return local_plan


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
    trainer_diagnostics: tuple[str, ...] = ()
    if isinstance(target, EngageTrainer):
        matching_triggers = tuple(trigger for trigger in world.triggers if trigger.affordance_id == target.trainer_id)
        trainer_diagnostics = (
            f"trainer_mode={getattr(getattr(goal, 'constraints', None), 'trainer_mode', None)!r}",
            f"selected_trainer={target.trainer_id!r}",
            f"trainer_hazards={sum(len(trigger.hazard_locations) for trigger in matching_triggers)}",
            "policy_order=legality>trainer>encounter>movement",
        )
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
        transitions = world.transitions or world.warps
        target_maps = tuple(
            sorted(
                {
                    transition.destination[0]
                    for transition in transitions
                    if transition.destination is not None
                    if target.destination_map is None or transition.destination[0] == target.destination_map
                },
                key=repr,
            )
        )

    current_map_goal = bool(target_maps) and set(target_maps) == {start[0]}
    available_transitions = tuple(
        (
            transition.kind,
            transition.entry,
            transition.destination,
            transition.required_facing.name if transition.required_facing else None,
            transition.activation.name,
            tuple(sorted(transition.activation_locations, key=repr)),
            transition.activation_direction.name if transition.activation_direction else None,
        )
        for transition in (world.transitions or world.warps)
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
    return trainer_diagnostics + (
        f"current_map={start[0]!r}",
        f"target_map={target_maps!r}",
        f"target_positions={target_positions!r}",
        f"current_map_goal={current_map_goal}",
        f"available_transitions={available_transitions!r}",
        f"candidate_interaction_positions={len(target_positions)}",
        f"status=UNREACHABLE reason={reason}",
    )


class GoalAwareNavigator:
    def __init__(self, world: NavigationWorld):
        self.world = world

    @profiled("navigation_pathfinding", "pathfinding_calls")
    @traced("individual_pathfinding")
    def plan(
        self,
        start: Location,
        goal: NavigationGoal | Goal,
        *,
        algorithm: str = "dijkstra",
        connection_source_is_approach: bool | None = None,
    ) -> NavigationPlan:
        navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
        if context.debug and getattr(context, "debug_trace", False):
            diagnostic_print(
                lambda: (
                    "NAVIGATION_ENCOUNTER_POLICY: "
                    f"mode={navigation_goal.encounter_mode.name} "
                    f"penalty={navigation_goal.encounter_penalty} start={start!r} "
                    f"target={navigation_goal.target!r}"
                ),
                trace=True,
            )
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
        if connection_source_is_approach is None:
            connection_source_is_approach = isinstance(target, SemanticTarget)
        orientation_required = self.world.facing is not None
        emit_orientation_actions = isinstance(
            target, (ActivateTrigger, ReachInteractionPosition, EngageTrainer, ReachWarp)
        )
        executable_orientation = orientation_required
        if algorithm not in ("dijkstra", "astar"):
            raise ValueError(f"Unknown navigation search algorithm: {algorithm!r}")
        profiling = profiling_enabled()
        search_started = perf_counter_ns() if profiling else 0
        expanded = generated = frontier_max = 0
        diagnostic_search_started = perf_counter_ns()
        diagnostic_last_emit = diagnostic_search_started
        diagnostic_successors = diagnostic_movement_successors = diagnostic_warp_successors = 0
        diagnostic_goal_checks = diagnostic_goal_matches = 0
        diagnostic_maps: set[MapId] = set()
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
        # Tracing must not change planner termination semantics.  Candidate
        # collection turns a normal early-return search into an exhaustive
        # search over the global world and can make a valid route appear to
        # hang.  Keep the counters/progress diagnostics below, but always
        # return on the first valid state as production planning does.
        collect_candidate_diagnostics = False

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
            diagnostic_maps.add(current[0])
            diagnostic_goal_checks += 1
            if self._satisfies(current, facing, navigation_goal.target):
                diagnostic_goal_matches += 1
                actions = self._unroll(
                    came_from,
                    state,
                    emit_orientation_actions,
                    connection_source_is_approach=connection_source_is_approach,
                )
                if isinstance(navigation_goal.target, ReachWarp):
                    actions, current = self._append_warp_activation(actions, current, facing, navigation_goal.target)
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
                    exposure_cost, opportunity_cost = encounter_cost_components(
                        navigation_goal.encounter_mode, navigation_goal.encounter_penalty
                    )
                    turn_cost = (turn_on_encounter * exposure_cost, 1, 0, 0, turn_on_encounter, turn_on_non_encounter)
                    next_state = (current, turn_direction)
                    new_cost = tuple(current_cost[i] + turn_cost[i] for i in range(6))
                    if new_cost < costs.get(next_state, (float("inf"),) * 6):
                        costs[next_state] = new_cost
                        came_from[next_state] = (state, turn_direction, False, False, True)
                        heapq.heappush(queue, (priority(new_cost, next_state), next(sequence), next_state, new_cost))
                        generated += 1
                        frontier_max = max(frontier_max, len(queue))
            for direction, neighbour, is_warp in self.world.neighbors(current):
                diagnostic_successors += 1
                diagnostic_movement_successors += int(not is_warp)
                diagnostic_warp_successors += int(is_warp)
                if self._is_undesirable(neighbour, navigation_goal.constraints):
                    continue
                tile = self.world.tiles.get(neighbour)
                if is_warp:
                    # A transition is a real action with a small positive
                    # cost. This prevents free transition cycles while
                    # preserving the existing lexicographic priorities.
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
                encounter_terrain_move = encounter
                turn_on_encounter = int(turn_cost and self.world.tiles[current].has_encounters)
                turn_on_non_encounter = int(turn_cost and not self.world.tiles[current].has_encounters)
                exposure_cost, opportunity_cost = encounter_cost_components(
                    navigation_goal.encounter_mode, navigation_goal.encounter_penalty
                )
                turn_opportunity = turn_cost * int(self.world.tiles[current].has_encounters)
                new_cost = (
                    current_cost[0] + exposure_cost * (encounter + turn_opportunity),
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
            if collect_candidate_diagnostics or getattr(context, "debug_trace", False):
                now = perf_counter_ns()
                if now - diagnostic_last_emit >= 2_000_000_000:
                    diagnostic_last_emit = now
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_PLAN_TRACE: phase=search_progress "
                            f"algorithm={algorithm!r} goal={navigation_goal.target!r} "
                            f"source={initial_state!r} elapsed_ms={(now-diagnostic_search_started)/1_000_000:.1f} "
                            f"expanded={expanded} discovered={generated} visited={len(costs)} "
                            f"frontier={len(queue)} frontier_max={frontier_max} "
                            f"successors={diagnostic_successors} movement={diagnostic_movement_successors} "
                            f"warps={diagnostic_warp_successors} goal_checks={diagnostic_goal_checks} "
                            f"goal_matches={diagnostic_goal_matches} maps_seen={len(diagnostic_maps)} "
                            f"current={state!r}"
                        ),
                        trace=True,
                    )
        if complete_candidates:
            selected_metrics, selected_actions, selected_destination = complete_candidates[0]
            record(True, selected_metrics)
            return NavigationPlan(
                selected_actions, selected_destination, selected_metrics, tuple(item[0] for item in complete_candidates)
            )
        # Trainer avoidance is a preference, not a reason to strand the
        # controller. If every route crosses a sight line, retry once with
        # trainer hazards disabled and make that decision visible to callers.
        trainer_mode = getattr(navigation_goal.constraints, "trainer_mode", None)
        if trainer_mode is not None and trainer_mode.name == "AVOID":
            fallback_constraints = replace(navigation_goal.constraints, trainer_mode=type(trainer_mode).IGNORE)
            fallback_goal = replace(navigation_goal, constraints=fallback_constraints)
            fallback = self.plan(start, fallback_goal, algorithm=algorithm)
            record(False)
            return replace(fallback, forced_trainer_exposure=True)
        record(False)
        raise NavigationError(f"No legal route from {start} to {navigation_goal.target!r}")

    def plan_many_locations(
        self,
        start: Location,
        targets: tuple[Location, ...],
        *,
        encounter_mode: EncounterMode = EncounterMode.NORMAL,
        encounter_penalty: int = 8,
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
                    step_cost = 1
                    encounter = 0
                else:
                    tile = self.world.tiles[neighbour]
                    step_cost = tile.traversal_cost - int(tile.has_encounters)
                    encounter = int(tile.has_encounters)
                turn_cost = int(self.world.facing is not None and facing is not direction)
                turn_on_encounter = int(turn_cost and self.world.tiles[current].has_encounters)
                turn_on_non_encounter = int(turn_cost and not self.world.tiles[current].has_encounters)
                opportunity = encounter + turn_on_encounter
                exposure_cost, opportunity_cost = encounter_cost_components(encounter_mode, encounter_penalty)
                new_cost = (
                    current_cost[0] + exposure_cost * opportunity,
                    current_cost[1] + step_cost + turn_cost + opportunity_cost * opportunity,
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
        positions: list[Location] = []
        if isinstance(target, ReachLocation):
            positions = [target.location]
        elif isinstance(target, ReachWarp):
            transitions = target.warps or ((target.warp,) if target.warp is not None else self.world.transitions)
            for transition in transitions:
                if transition is None or transition.destination is None:
                    continue
                source = transition_approach_position(transition)
                if source is None:
                    activation_locations = getattr(transition, "activation_locations", ())
                    source = next(iter(activation_locations), transition.entry)
                positions.append(source)
        elif isinstance(target, (ActivateTrigger, ReachInteractionPosition, EngageTrainer)):
            for trigger in self.world.triggers:
                matches = (
                    trigger.affordance_id == target.trainer_id
                    if isinstance(target, EngageTrainer)
                    else trigger.trigger_id == target.trigger_id
                )
                if matches:
                    positions.extend(
                        trigger.hazard_locations
                        if isinstance(target, EngageTrainer)
                        else (trigger.activation_locations or trigger.navigation_locations)
                    )
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
        if isinstance(target, SemanticTarget):
            return target.location == location if target.location is not None else location[0] == target.target_map
        if isinstance(target, ReachLocation):
            return location == target.location
        if isinstance(target, EngageTrainer):
            return any(
                trigger.affordance_id == target.trainer_id
                and (
                    location in trigger.hazard_locations
                    if trigger.hazard_locations
                    else self._activation_matches(trigger, location, facing)
                )
                for trigger in self.world.triggers
            )
        if isinstance(target, ActivateTrigger):
            return any(
                target.trigger_id == trigger.trigger_id and self._activation_matches(trigger, location, facing)
                for trigger in self.world.triggers
            )
        if isinstance(target, ReachInteractionPosition):
            matched = any(
                target.trigger_id == trigger.trigger_id and self._activation_matches(trigger, location, facing)
                for trigger in self.world.triggers
            )
            return matched
        if isinstance(target, ReachWarp):
            all_transitions = self.world.transitions or self.world.warps
            allowed_transitions = target.warps or ((target.warp,) if target.warp is not None else all_transitions)
            # ReachWarp is evaluated for every search node.  Restrict the
            # candidates to transitions executable from this node before
            # comparing identities.  The previous implementation scanned all
            # boundary endpoints on every node; on Emerald's long connection
            # strips that turned one local route into a frame-scale search.
            transitions = self.world._transitions_by_source.get(location, ())
            matched = any(
                (
                    location == transition_approach_position(warp)
                    if warp.kind == "map_connection"
                    else (
                        location in warp.activation_locations if warp.activation_locations else warp.entry == location
                    )
                )
                and (
                    # A normal step-on warp is activated by the movement
                    # onto its entry tile.  Its ROM metadata may still carry
                    # a directional hint (notably stair tiles), but requiring
                    # that facing after arrival traps the tactical planner in
                    # repeated TURN actions.  Facing is a terminal condition
                    # only for warps whose activation is explicitly a second,
                    # directional input or whose observed source specifies an
                    # activation direction.
                    (warp.activation is not WarpActivation.DIRECTIONAL_STEP and warp.activation_direction is None)
                    or facing is None
                    or (
                        (warp.activation_direction if warp.activation_direction is not None else warp.required_facing)
                        is facing
                    )
                )
                and (target.destination is None or warp.destination == target.destination)
                and (
                    target.destination_map is None
                    or (warp.destination is not None and warp.destination[0] == target.destination_map)
                )
                and (target.warp is None or transitions_match(warp, target.warp))
                and (not target.warps or any(transitions_match(warp, allowed) for allowed in target.warps))
                and any(transitions_match(warp, allowed) for allowed in allowed_transitions)
                for warp in transitions
            )
            return matched
        raise TypeError(f"Unsupported goal type: {type(target).__name__}")

    def _append_warp_activation(self, actions, current, facing, target):
        """Append the executable transition after reaching its source state."""
        transitions = self.world.transitions or self.world.warps
        allowed_transitions = target.warps or ((target.warp,) if target.warp is not None else transitions)
        for transition in transitions:
            if not any(transitions_match(transition, allowed) for allowed in allowed_transitions):
                continue
            approach = transition_approach_position(transition)
            source = (
                approach
                if transition.kind == "map_connection"
                else (
                    next(iter(transition.activation_locations), transition.entry)
                    if transition.activation_locations
                    else transition.entry
                )
            )
            if current != source or transition.destination is None:
                continue
            if target.destination is not None and transition.destination != target.destination:
                continue
            if target.destination_map is not None and transition.destination[0] != target.destination_map:
                continue
            if target.warp is not None and not transitions_match(transition, target.warp):
                continue
            direction = transition.activation_direction
            if direction is None:
                direction = transition.required_facing
            if direction is None:
                direction = facing
            if direction is None:
                direction = Direction.South
            return (
                actions
                + (
                    NavigationAction(
                        NavigationActionType.WARP,
                        direction,
                        source,
                        transition.destination,
                        transition_kind=transition.kind,
                    ),
                ),
                transition.destination,
            )
        return actions, current

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
        trainer_mode = getattr(constraints, "trainer_mode", None)
        if trainer_mode is not None and trainer_mode.name == "AVOID":
            if any(location in trigger.hazard_locations for trigger in self.world.triggers):
                return True
        return any(
            location in trigger.locations and trigger.trigger_id in constraints.avoid_trigger_ids
            for trigger in self.world.triggers
        )

    def _unroll(
        self,
        came_from,
        destination_state,
        emit_orientation_actions: bool,
        *,
        connection_source_is_approach: bool = False,
    ) -> tuple[NavigationAction, ...]:
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
            transition_kind = None
            if is_warp and source[0] != destination_location[0]:
                transition = next(
                    (
                        transition
                        for transition in (self.world.transitions or self.world.warps)
                        if transition.destination == destination_location
                        and (transition.entry == source or transition_approach_position(transition) == source)
                    ),
                    None,
                )
                transition_kind = transition.kind if transition is not None else "warp"
                if transition is not None and transition.kind == "map_connection":
                    entry = transition.entry
                    approach = transition_approach_position(transition)
                    if approach == source and entry != source:
                        # Global map-connection successors originate at the
                        # executable approach tile. Reconstruct the same
                        # two-input action contract as local planning: first
                        # step onto the boundary, then cross the connection.
                        #
                        # The second action must be sourced at ``entry``.
                        # ``approach`` is where the search discovers the
                        # transition, but the first MOVE has already placed
                        # the avatar on the boundary by the time the
                        # crossing input is dispatched. Keeping the WARP
                        # source at ``approach`` makes the route controller
                        # see a divergence and replan back one tile forever
                        # at live map connections.
                        actions.append(
                            NavigationAction(
                                NavigationActionType.WARP,
                                transition.required_facing,
                                entry,
                                destination_location,
                                transition_kind="map_connection",
                            )
                        )
                        actions.append(
                            NavigationAction(
                                NavigationActionType.MOVE,
                                transition.required_facing,
                                approach,
                                entry,
                                transition_kind="map_connection",
                            )
                        )
                        current = previous
                        continue
            actions.append(
                NavigationAction(
                    NavigationActionType.WARP if is_warp else NavigationActionType.MOVE,
                    direction,
                    source,
                    destination_location,
                    transition_kind=transition_kind,
                    run=(
                        not is_warp
                        and self.world.running_shoes
                        and (destination_tile := self.world.tiles.get(destination_location)) is not None
                        and not destination_tile.cannot_run
                    ),
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
        total_route_cost += sum(1 for action in actions if action.action_type is NavigationActionType.WARP)
        total_route_cost += turns_on_encounter + turns_on_non_encounter
        _, opportunity_cost = encounter_cost_components(encounter_mode)
        total_route_cost += encounter_opportunities * opportunity_cost
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
