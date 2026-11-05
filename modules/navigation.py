"""Deterministic goal-aware navigation over a small abstract world model."""

import heapq
from collections import OrderedDict
from time import perf_counter_ns, sleep
from itertools import count
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from threading import RLock
from typing import Mapping

from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    Goal,
    GoalConstraints,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    SemanticTargetKind,
    EncounterMode,
    TrainerEngagementMode,
    TrainerEngagementPlan,
    TrainerMode,
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
    resolve_object_activation_positions,
    static_map_transitions,
    static_trainer_observations,
    trainer_engagement_locations,
    trainer_hazard_locations,
)
from modules.trigger_bindings import BindingResolution
from modules.world_navigation import WorldEdge, WorldMapGraph, WorldNavigationError, WorldRoute, get_world_map_graph
from modules.stutter_trace import background_work_suppressed, traced
from modules.context import context
from modules.console import diagnostic_print, print_navigation_pathfinding
from modules.profiler import profiled
from modules.profiler import (
    enabled as profiling_enabled,
    count as profile_count,
    record_cross_map_goal_evaluation,
    record_pathfinding_search,
    pathfinding_searches,
)


_pathfinding_call_ids = count(1)
_failed_reach_warp_diagnostics: OrderedDict[tuple, None] = OrderedDict()
_failed_reach_warp_diagnostics_lock = RLock()
_MAX_FAILED_REACH_WARP_DIAGNOSTICS = 256


def _transition_diagnostic_signature(transition: WorldTransition) -> tuple:
    """Return executable transition facts suitable for a failure log."""

    return (
        getattr(transition, "kind", None),
        getattr(transition, "entry", None),
        transition_approach_position(transition),
        getattr(transition, "destination", None),
        getattr(getattr(transition, "activation", None), "value", getattr(transition, "activation", None)),
        tuple(sorted(getattr(transition, "activation_locations", ()) or (), key=repr)),
        getattr(transition, "activation_direction", None),
        getattr(transition, "required_facing", None),
    )


def _failed_reach_warp_diagnostic_key(world: "NavigationWorld", start: Location, target: ReachWarp) -> tuple:
    """Identify one failed ReachWarp observation without retaining worlds."""

    return (
        start,
        repr(target),
        world.dynamic_blocked_revision,
        world.transition_revision,
    )


def _should_emit_failed_reach_warp_diagnostic(key: tuple) -> bool:
    """Bound repeated retry-loop diagnostics while retaining recent snapshots."""

    with _failed_reach_warp_diagnostics_lock:
        if key in _failed_reach_warp_diagnostics:
            return False
        _failed_reach_warp_diagnostics[key] = None
        while len(_failed_reach_warp_diagnostics) > _MAX_FAILED_REACH_WARP_DIAGNOSTICS:
            _failed_reach_warp_diagnostics.popitem(last=False)
        return True


def _log_failed_reach_warp_diagnostic(world: "NavigationWorld", start: Location, target: ReachWarp) -> None:
    """Explain why a reachable-looking transition was rejected by search."""

    key = _failed_reach_warp_diagnostic_key(world, start, target)
    if not _should_emit_failed_reach_warp_diagnostic(key):
        return

    transitions = tuple(world.transitions or world.warps)
    allowed = target.warps or ((target.warp,) if target.warp is not None else transitions)
    target_transition = target.warp
    target_source = (
        transition_approach_position(target_transition)
        if target_transition is not None and getattr(target_transition, "kind", None) == "map_connection"
        else (
            next(iter(getattr(target_transition, "activation_locations", ()) or ()), None)
            if target_transition is not None and getattr(target_transition, "activation_locations", ())
            else getattr(target_transition, "entry", None)
        )
    )
    target_entry = getattr(target_transition, "entry", None)
    entry_tile = world.tiles.get(target_entry) if target_entry is not None else None
    dynamic_blocked = world.dynamic_blocked_revision or ()
    indexed_target = tuple(world._transitions_by_source.get(target_source, ())) if target_source else ()
    indexed_sources = tuple(
        sorted(
            (
                (
                    source,
                    tuple(_transition_diagnostic_signature(transition) for transition in indexed),
                )
                for source, indexed in world._transitions_by_source.items()
                if source == target_source or (target_entry is not None and source == target_entry)
            ),
            key=repr,
        )
    )
    identity_matches = tuple(
        _transition_diagnostic_signature(transition)
        for transition in indexed_target
        if any(transitions_match(transition, candidate) for candidate in allowed)
    )
    diagnostic_print(
        lambda: (
            "NAVIGATION_REACH_WARP_DIAGNOSTIC: "
            f"start={start!r} target={target!r} "
            f"target_source={target_source!r} target_entry={target_entry!r} "
            f"entry_tile_present={entry_tile is not None!r} "
            f"entry_tile_blocked={getattr(entry_tile, 'blocked', None)!r} "
            f"entry_in_blocked_revision={target_entry is not None and target_entry[1] in dynamic_blocked!r} "
            f"dynamic_blocked_revision={world.dynamic_blocked_revision!r} "
            f"indexed_target_sources={indexed_sources!r} "
            f"identity_matches={identity_matches!r} "
            f"target_transition={_transition_diagnostic_signature(target_transition) if target_transition is not None else None!r} "
            f"world_transitions={tuple(_transition_diagnostic_signature(transition) for transition in transitions)!r}"
        ),
        trace=True,
        prefix="NAVIGATION_REACH_WARP_DIAGNOSTIC:",
    )


def _pokemon_center_map_ids() -> frozenset[MapId]:
    """Return Emerald Center interior maps without hard-coding map numbers."""

    from modules.map_data import MapRSE

    return frozenset(
        member.value
        for member in MapRSE
        if "POKEMON_CENTER_" in member.name
    )


def _goal_allows_pokemon_center_destination(goal: NavigationGoal) -> bool:
    """Return whether this goal explicitly requests a Center interior."""

    target = goal.target
    center_maps = _pokemon_center_map_ids()
    if isinstance(target, ReachWarp):
        destination = target.destination
        destination_map = target.destination_map or (destination[0] if destination is not None else None)
        return canonical_map_id(destination_map) in center_maps
    if isinstance(target, ReachLocation):
        return canonical_map_id(target.location[0]) in center_maps
    if isinstance(target, SemanticTarget):
        return canonical_map_id(target.target_map) in center_maps
    return False


class NavigationActionType(Enum):
    MOVE = auto()
    TURN = auto()
    WARP = auto()


class TransitionRelevance(Enum):
    RELEVANT = auto()
    IRRELEVANT = auto()
    UNKNOWN = auto()
    BLOCKED = auto()


class TrainerAvoidability(Enum):
    """How difficult it is to keep one trainer out of a route."""

    AVOIDABLE = "avoidable"
    COSTLY = "costly"
    PRACTICALLY_UNAVOIDABLE = "practically_unavoidable"
    UNAVOIDABLE = "unavoidable"
    IMMINENT = "imminent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TrainerRouteRisk:
    """Route geometry and cost evidence for one undefeated trainer."""

    trainer_id: str
    avoidability: TrainerAvoidability
    hazard_locations: frozenset[Location] = frozenset()
    exposed_on_normal_route: bool = False
    normal_cost: int | None = None
    engagement_cost: int | None = None
    avoidance_cost: int | None = None
    detour: int | None = None
    confidence: str = "known"
    reason: str = ""


class NavigationSearchCancelled(RuntimeError):
    """Raised when a background route search is superseded or times out."""


def canonical_map_id(map_id):
    """Return the tuple/value form used at navigation cache boundaries.

    ROM-facing callers commonly pass ``MapRSE`` members while live
    perception and the static navigation cache use their ``.value`` tuples.
    Keeping that conversion at the cache boundary prevents a valid map from
    being looked up once as an enum and once as its raw value.
    """

    return getattr(map_id, "value", map_id)


def canonical_location(location):
    """Normalize the map component of a navigation location."""

    if not isinstance(location, tuple) or len(location) != 2:
        return location
    return canonical_map_id(location[0]), location[1]


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
    blocked, transitions = world.revision_key()
    hazards = tuple(sorted(location for trigger in world.triggers for location in trigger.hazard_locations))
    engagement_locations = tuple(
        sorted(
            (trigger.trigger_id, location)
            for trigger in world.triggers
            for location in getattr(trigger, "engagement_locations", ())
        )
    )
    constraints = goal.constraints
    return (
        start,
        world.facing,
        world.running_shoes,
        repr(target),
        tuple(sorted(constraints.avoid_trigger_ids)),
        tuple(sorted(constraints.avoid_locations, key=repr)),
        tuple(sorted(getattr(constraints, "avoid_trainer_ids", ()))),
        getattr(getattr(constraints, "trainer_mode", None), "value", None),
        bool(getattr(constraints, "allow_forced_trainer_exposure", False)),
        goal.encounter_mode.value,
        goal.encounter_penalty,
        tuple(goal.preferred_map_route or ()),
        transitions,
        blocked,
        hazards,
        engagement_locations,
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
            prefix="CAMPAIGN_WORLD_ROUTE_DEBUG",
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


def transition_source_matches(left: WorldTransition, right: WorldTransition) -> bool:
    """Compare the stable executable identity of two transition records.

    A live transition may carry a destination learned from a previous ROM
    observation while a grouped ``ReachWarp`` still contains the static
    record that selected that source.  The source, kind, and entry identify
    the executable door; the goal separately verifies the destination map.
    Keep the relaxed comparison private to grouped observed goals so exact
    destination goals retain their stronger contract.
    """

    left_source = transition_approach_position(left) or left.entry
    right_source = transition_approach_position(right) or right.entry
    return left.kind == right.kind and left_source == right_source and left.entry == right.entry


def _reach_warp_transition_matches(transition: WorldTransition, target: ReachWarp) -> bool:
    """Return whether a live transition satisfies a ``ReachWarp`` identity."""

    if target.warp is not None:
        return transitions_match(transition, target.warp)
    if not target.warps:
        return True
    return any(
        transitions_match(transition, allowed)
        or (
            target.destination is None
            and transition_source_matches(transition, allowed)
        )
        for allowed in target.warps
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
        location = canonical_location(location)
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

    def get(self, location: Location, default=None):
        try:
            return self[location]
        except KeyError:
            return default


class _LazyStaticTileMapping(Mapping[Location, NavigableTile]):
    """Read-through immutable per-map tile models for global search.

    The global search asks for individual locations through ``get``; avoid
    materializing every graph map up front.  Each map is still constructed at
    most once and is retained by the existing static navigation cache.
    """

    def __init__(
        self,
        initial: Mapping[Location, NavigableTile],
        map_ids,
        *,
        static_maps=(),
        static_models=(),
        metadata_frozen=False,
    ):
        normalized_initial = {
            canonical_location(location): tile
            for location, tile in initial.items()
        }
        self._observed = {
            map_id: dict(
                (location, tile)
                for location, tile in normalized_initial.items()
                if location[0] == map_id
            )
            for map_id in {location[0] for location in normalized_initial}
        }
        # Live perception is often a partial window around the avatar.  Do
        # not treat those entries as a complete model: the ROM tile map must
        # still be loaded so a static transition several tiles away remains
        # reachable.  Observed tiles are overlaid afterward so dynamic
        # occupancy and live collision information retain precedence.
        self._models: dict[MapId, dict[Location, NavigableTile]] = {}
        self._map_ids = frozenset(canonical_map_id(map_id) for map_id in map_ids)
        self._static_maps = frozenset(canonical_map_id(map_id) for map_id in static_maps)
        self._static_models = {
            canonical_map_id(map_id): {
                canonical_location(location): tile for location, tile in tiles
            }
            for map_id, tiles in static_models
        }
        self._metadata_frozen = bool(metadata_frozen)
        self._loaded: set[MapId] = set()

    def _load(self, map_id) -> dict[Location, NavigableTile]:
        map_id = canonical_map_id(map_id)
        if map_id in self._static_models:
            self._models[map_id] = dict(self._static_models[map_id])
            self._loaded.add(map_id)
        elif map_id not in self._static_maps and map_id in self._observed:
            self._models[map_id] = dict(self._observed[map_id])
            self._loaded.add(map_id)
        elif map_id not in self._loaded:
            if self._metadata_frozen:
                # Background route workers must not touch ROM-backed caches.
                self._models.setdefault(map_id, {})
                self._loaded.add(map_id)
                return self._models[map_id]
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
            self._models[map_id] = dict(_static_navigation_tiles[map_id][1])
            self._loaded.add(map_id)
            profile_count("global_navigation_new_map_models")
            profile_count("global_navigation_materialized_tiles", len(self._models[map_id]))
        else:
            profile_count("global_navigation_cached_map_model_lookups")
        observed = self._observed.get(map_id)
        if observed:
            self._models.setdefault(map_id, {}).update(observed)
        return self._models.get(map_id, {})

    def __getitem__(self, location: Location) -> NavigableTile:
        location = canonical_location(location)
        return self._load(location[0])[location]

    def get(self, location: Location, default=None):
        location = canonical_location(location)
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
    map_id = canonical_map_id(map_id)
    cache = _static_navigation_tiles.get(map_id)
    if cache is None or cache[0] is not tiles:
        _static_navigation_tiles[map_id] = (
            tiles,
            {
                canonical_location(tile.location): NavigableTile(
                    canonical_location(tile.location),
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


def static_navigation_tile_snapshot(
    map_id: MapId, tiles: tuple | None = None
) -> tuple[tuple[Location, NavigableTile], ...]:
    """Return a static tile model prepared by the caller's ROM-owning thread."""

    map_id = canonical_map_id(map_id)
    if tiles is None:
        tiles = prewarm_static_map_observation(map_id)
    prewarm_navigation_tiles(map_id, tiles)
    return tuple(_static_navigation_tiles[map_id][1].items())


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
    # Background route workers consume immutable frame-thread snapshots rather
    # than reading ROM metadata or mutating navigation caches themselves.
    static_tile_catalog: tuple[tuple[MapId, tuple[tuple[Location, NavigableTile], ...]], ...] = ()
    static_transition_catalog: tuple[tuple[MapId, tuple[WorldTransition, ...]], ...] = ()
    static_metadata_frozen: bool = False
    # These revisions are produced while the live observation is assembled.
    # Keeping them on the immutable world avoids scanning a potentially large
    # static tile mapping every time a background request is polled.
    dynamic_blocked_revision: tuple | None = None
    transition_revision: tuple | None = None
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

    def revision_key(self) -> tuple[tuple, tuple]:
        """Return compact dynamic-world revisions for cache/request keys.

        Synthetic worlds constructed by tests may not provide revisions.  The
        fallback preserves their historical behavior; live worlds always set
        both values in :meth:`from_overworld`, so production polling never
        iterates the complete tile mapping.
        """
        if self.dynamic_blocked_revision is None:
            blocked = tuple(sorted((location[1] for location, tile in self.tiles.items() if tile.blocked), key=repr))
        else:
            blocked = self.dynamic_blocked_revision
        if self.transition_revision is None:
            transitions = tuple(
                sorted(
                    tuple(
                        (
                            getattr(t, "kind", None),
                            getattr(t, "entry", None),
                            getattr(t, "destination", None),
                            getattr(t, "required_facing", None),
                        )
                        for t in (self.transitions or self.warps)
                    ),
                    key=repr,
                )
            )
        else:
            transitions = self.transition_revision
        return blocked, transitions

    @classmethod
    @traced("NavigationWorld_construction")
    def from_overworld(cls, observation: OverworldObservation) -> "NavigationWorld":
        """Adapt the live perception model without coupling the search to emulator APIs."""
        map_id = canonical_map_id(observation.map_id)
        cache = _static_navigation_tiles.get(map_id)
        if cache is None or cache[0] is not observation.tiles:
            prewarm_navigation_tiles(map_id, observation.tiles)
            static_tiles = _static_navigation_tiles[map_id][1]
        else:
            static_tiles = cache[1]
        # Runtime object events are authoritative occupancy evidence even for
        # callers that provide ``objects`` without the normalized coordinate
        # overlay. This matters when a defeated trainer remains on its battle
        # tile as control returns from the battle listener.
        object_blocked_coordinates = frozenset(
            object_observation.location[1]
            for object_observation in getattr(observation, "objects", ())
            if canonical_map_id(object_observation.location[0]) == map_id
        )
        blocked_coordinates = (
            frozenset(observation.dynamic_blocked_coordinates)
            | frozenset(getattr(observation, "static_blocked_coordinates", ()))
            | object_blocked_coordinates
        )
        tiles: Mapping[Location, NavigableTile] = static_tiles
        if blocked_coordinates:
            tiles = _DynamicTileMapping(static_tiles, blocked_coordinates)
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
            dynamic_blocked_revision=tuple(sorted(blocked_coordinates, key=repr)),
            transition_revision=tuple(
                sorted(
                    tuple(
                        (
                            getattr(t, "kind", None),
                            getattr(t, "entry", None),
                            getattr(t, "destination", None),
                            getattr(t, "required_facing", None),
                        )
                        for t in tuple(
                            effective_transition(t)
                            for t in (getattr(observation, "transitions", ()) or observation.warps)
                        )
                    ),
                    key=repr,
                )
            ),
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
        # only when the map data supplies one; door warps expose that input as
        # ``activation_direction`` while their entry tile often has no
        # ``required_facing``.  Falling through to the stable South default
        # sends a Center-door approach back into the town and can create an
        # up/down route loop after recovery.
        if warp.activation_direction is not None:
            return warp.activation_direction
        if warp.activation_locations:
            # Multi-side door records do not have one global activation
            # direction.  The input is the direction from the selected
            # source tile onto the door entry.  Falling back to South here
            # sends a north-side approach back into the town and can trap
            # cross-map navigation in an up/down loop.
            source = next(
                (candidate for candidate in warp.activation_locations if candidate == location),
                None,
            )
            if source is not None:
                direction = _direction_between(source[1], warp.entry[1])
                if direction is not None:
                    return direction
        return warp.required_facing if warp.required_facing is not None else Direction.South


def resolve_trainer_engagement_plan(world: NavigationWorld, trainer_id: str) -> TrainerEngagementPlan:
    """Resolve the executable activation mode for a trainer in ``world``.

    Runtime/static trainer triggers and semantic campaign bindings can
    describe the same trainer. Prefer a semantic or live adjacent interaction
    when it is positively observed, then a live sight line, and otherwise
    preserve an explicit unknown result. In particular, an empty sight-line
    set is not permission to press A: static hazard metadata can be incomplete
    and must remain fail-closed.
    """

    matching = tuple(trigger for trigger in world.triggers if trigger.affordance_id == trainer_id)
    if not matching:
        return TrainerEngagementPlan(
            trainer_id,
            None,
            TrainerEngagementMode.UNKNOWN,
            reason="trainer affordance is not present in the current world observation",
        )

    interaction_candidates = tuple(
        trigger
        for trigger in matching
        if getattr(trigger, "requires_input", True)
        and getattr(trigger, "activation_locations", ())
        and (
            str(getattr(trigger, "kind", "")).startswith("semantic_")
            or not getattr(trigger, "engagement_locations", ())
        )
    )
    if interaction_candidates:
        trigger = interaction_candidates[0]
        return TrainerEngagementPlan(
            trainer_id,
            trigger.trigger_id,
            TrainerEngagementMode.INTERACTION,
            interaction_locations=frozenset(trigger.activation_locations),
            activation_requirements=tuple(trigger.activation_requirements),
            requires_input=True,
            reason="adjacent interaction geometry is positively observed",
        )

    sightline_candidates = tuple(
        trigger for trigger in matching if getattr(trigger, "engagement_locations", ())
    )
    if sightline_candidates:
        trigger = sightline_candidates[0]
        return TrainerEngagementPlan(
            trainer_id,
            trigger.trigger_id,
            TrainerEngagementMode.SIGHTLINE,
            engagement_locations=frozenset(trigger.engagement_locations),
            interaction_locations=frozenset(trigger.activation_locations),
            activation_requirements=tuple(trigger.activation_requirements),
            requires_input=False,
            reason="trainer sight-line geometry is positively observed",
        )

    # Retain the trigger identity for diagnostics and future re-observation,
    # but do not claim that either activation mechanism is executable.
    trigger = matching[0]
    return TrainerEngagementPlan(
        trainer_id,
        trigger.trigger_id,
        TrainerEngagementMode.UNKNOWN,
        interaction_locations=frozenset(trigger.activation_locations),
        activation_requirements=tuple(trigger.activation_requirements),
        requires_input=getattr(trigger, "requires_input", None),
        reason="trainer is known but no executable sight line or interaction geometry is observed",
    )


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


class NavigationSearchLimitExceeded(NavigationError):
    """A bounded search stopped before proving a route or no-route result."""


def _validate_search_limit(value: int | None, name: str) -> None:
    """Validate an optional non-negative navigation search limit."""

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")


def _cost_prefix(metrics: "NavigationMetrics") -> tuple[int, ...]:
    """Return the search-cost prefix used by recovery incumbent pruning."""

    return metrics.encounter_opportunities, metrics.total_route_cost


def _enforce_plan_limits(
    plan: "NavigationPlan",
    *,
    max_route_cost: int | None = None,
    cost_ceiling: tuple[int, ...] | int | None = None,
) -> None:
    """Reject a manually-composed plan that exceeds speculative limits."""

    _validate_search_limit(max_route_cost, "max_route_cost")
    if isinstance(cost_ceiling, bool) or (
        cost_ceiling is not None and not isinstance(cost_ceiling, (int, tuple))
    ):
        raise ValueError("cost_ceiling must be an integer, tuple, or None")
    if isinstance(cost_ceiling, tuple) and not all(
        isinstance(item, int) and not isinstance(item, bool) for item in cost_ceiling
    ):
        raise ValueError("cost_ceiling tuple must contain integers")
    metrics = plan.metrics
    if metrics is None:
        return
    if max_route_cost is not None and metrics.total_route_cost > max_route_cost:
        raise NavigationSearchLimitExceeded(
            f"route cost {metrics.total_route_cost} exceeds max_route_cost {max_route_cost}"
        )
    if cost_ceiling is not None:
        if isinstance(cost_ceiling, int):
            exceeded = metrics.total_route_cost > cost_ceiling
            ceiling = cost_ceiling
        else:
            ceiling = cost_ceiling
            exceeded = bool(ceiling) and _cost_prefix(metrics)[: len(ceiling)] > ceiling
        if exceeded:
            raise NavigationSearchLimitExceeded(
                f"route cost prefix {_cost_prefix(metrics)!r} exceeds cost_ceiling {ceiling!r}"
            )


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
    trainer_risks: tuple[TrainerRouteRisk, ...] = ()


class RouteCostAnalyzer:
    """Analyze composed routes using the exact campaign navigation planner.

    This is deliberately domain-neutral.  Both the final goal and an
    intermediate destination are normal navigation goals, so semantic goals
    such as ``ReachWarp`` and ``ActivateTrigger`` retain their meaning.
    """

    def __init__(
        self,
        world: NavigationWorld,
        *,
        graph: WorldMapGraph | None = None,
        planner=None,
        max_expansions: int | None = None,
        max_route_cost: int | None = None,
    ):
        self.world = world
        self.graph = graph
        self._planner = planner or plan_with_world_navigation
        self._max_expansions = max_expansions
        self._max_route_cost = max_route_cost

    @staticmethod
    def _cost(plan: NavigationPlan | None) -> int | None:
        return plan.metrics.total_route_cost if plan is not None and plan.metrics is not None else None

    @staticmethod
    def _route_locations(plan: NavigationPlan | None) -> frozenset[Location]:
        if plan is None:
            return frozenset()
        locations = set()
        for action in plan.actions:
            if action.source is not None:
                locations.add(action.source)
            if action.destination is not None:
                locations.add(action.destination)
        if plan.destination is not None:
            locations.add(plan.destination)
        return frozenset(locations)

    @staticmethod
    def _route_maps(plan: NavigationPlan | None) -> frozenset[MapId]:
        """Return maps represented by a route, including its terminal map."""

        return frozenset(
            canonical_map_id(location[0])
            for location in RouteCostAnalyzer._route_locations(plan)
            if isinstance(location, tuple) and len(location) == 2
        )

    @staticmethod
    def _trainer_goal(goal, *, mode, avoid_ids=frozenset()):
        if not isinstance(goal, NavigationGoal):
            return None
        constraints = replace(
            goal.constraints,
            trainer_mode=mode,
            avoid_trainer_ids=frozenset(avoid_ids),
        )
        return replace(goal, constraints=constraints)

    def _trainer_risks(self, start: Location, goal, normal: NavigationPlan | None, normal_cost: int | None, plan):
        """Measure per-trainer avoidance using bounded route alternatives."""

        if not isinstance(goal, NavigationGoal) or not getattr(self.world, "triggers", ()):
            return ()
        risks = []
        normal_locations = self._route_locations(normal)
        ignore_goal = self._trainer_goal(goal, mode=TrainerMode.IGNORE)
        if ignore_goal is None:
            return ()
        # A caller may deliberately request the parent route with AVOID so
        # that execution is safe.  That route cannot also be the exposure
        # baseline: by definition it has already removed the trainer hazard.
        # Compare it with the same route under IGNORE, which represents the
        # route the campaign would take if it accepted the engagement.
        try:
            engagement_route = plan(self.world, start, ignore_goal)
        except (NavigationError, RuntimeError, TypeError, ValueError):
            engagement_route = None
        engagement_cost = self._cost(engagement_route)
        engagement_locations = self._route_locations(engagement_route)
        engagement_maps = self._route_maps(engagement_route)
        requested_mode = getattr(getattr(goal, "constraints", None), "trainer_mode", None)
        exposure_locations = (
            engagement_locations
            if requested_mode is TrainerMode.AVOID and engagement_route is not None
            else normal_locations
        )
        for trigger in getattr(self.world, "triggers", ()):
            trainer_id = getattr(trigger, "affordance_id", None) or getattr(trigger, "trigger_id", None)
            if not trainer_id:
                continue
            is_trainer = getattr(trigger, "hazard_kind", None) == "trainer" or str(trainer_id).startswith(
                ("trainer:", "rom_trainer:")
            )
            if not is_trainer:
                continue
            if getattr(trigger, "trainer_defeated", None) is True:
                continue
            hazards = frozenset(getattr(trigger, "hazard_locations", ()) or ())
            if not hazards:
                # A trainer trigger with no geometry is not evidence that the
                # trainer is harmless when the selected route actually
                # reaches its object or interaction positions. It also is not
                # evidence that every trainer on the map blocks the route:
                # static Gym trainers and other unreachable objects can be
                # present in the world with an empty envelope. Keep the
                # unknown hazard only when the route has a concrete
                # intersection with the trigger's positions.
                trigger_positions = frozenset(
                    location
                    for field in ("locations", "activation_locations", "engagement_locations")
                    for location in (getattr(trigger, field, ()) or ())
                    if isinstance(location, tuple) and len(location) == 2
                )
                if trigger_positions and not trigger_positions.intersection(normal_locations):
                    continue
                risks.append(
                    TrainerRouteRisk(
                        trainer_id,
                        TrainerAvoidability.UNKNOWN,
                        exposed_on_normal_route=True,
                        confidence="unknown",
                        reason="trainer sight-line geometry is unavailable; route exposure is unknown",
                    )
                )
                continue
            exposed = bool(hazards & exposure_locations)
            if (
                not exposed
                and engagement_route is not None
                and not getattr(engagement_route, "actions", ())
                and engagement_maps
            ):
                # A lightweight planner/test double may return only a
                # terminal location and omit its movement actions.  That is
                # insufficient to prove that a trainer on the terminal map
                # is avoided, so keep the hazard in the bounded comparison.
                # Full live plans carry action locations and use the precise
                # intersection above.
                exposed = any(
                    isinstance(location, tuple)
                    and len(location) == 2
                    and canonical_map_id(location[0]) in engagement_maps
                    for location in hazards
                )
            # Avoidance cost is only relevant for a trainer that the selected
            # normal route can actually expose.  Computing an IGNORE route
            # and a separate AVOID route for every trainer on a map made the
            # campaign readiness worker spend most of its time solving
            # unrelated branches.  Static and live hazard geometry still
            # remains in the route envelope, and the tactical navigator keeps
            # validating movement against every trainer trigger.
            if not exposed:
                continue
            if start in hazards:
                risks.append(
                    TrainerRouteRisk(
                        trainer_id,
                        TrainerAvoidability.IMMINENT,
                        hazards,
                        exposed,
                        normal_cost,
                        engagement_cost,
                        0,
                        0,
                        reason="current position is inside the trainer sight line",
                    )
                )
                continue
            avoid_goal = self._trainer_goal(
                goal,
                mode=TrainerMode.AVOID,
                avoid_ids={trainer_id},
            )
            try:
                avoidance_route = plan(self.world, start, avoid_goal)
            except (NavigationError, RuntimeError, TypeError, ValueError):
                avoidance_route = None
            avoidance_cost = self._cost(avoidance_route)
            avoidance_locations = self._route_locations(avoidance_route)
            if avoidance_route is None:
                level = TrainerAvoidability.UNAVOIDABLE
                reason = "no route avoiding this trainer was found"
                detour = None
            elif getattr(avoidance_route, "forced_trainer_exposure", False) or hazards & avoidance_locations:
                level = TrainerAvoidability.UNAVOIDABLE
                reason = "every bounded route still intersects this trainer's sight line"
                detour = None if avoidance_cost is None or engagement_cost is None else avoidance_cost - engagement_cost
            else:
                detour = None if avoidance_cost is None or engagement_cost is None else avoidance_cost - engagement_cost
                if detour is None:
                    level = TrainerAvoidability.UNKNOWN
                    reason = "route costs were unavailable"
                elif detour <= 12:
                    level = TrainerAvoidability.AVOIDABLE
                    reason = "a low-cost trainer-free route exists"
                elif detour <= 40:
                    level = TrainerAvoidability.COSTLY
                    reason = "avoidance requires a measurable detour"
                else:
                    level = TrainerAvoidability.PRACTICALLY_UNAVOIDABLE
                    reason = "avoidance requires a substantial detour"
            risks.append(
                TrainerRouteRisk(
                    trainer_id,
                    level,
                    hazards,
                    exposed,
                    normal_cost,
                    engagement_cost,
                    avoidance_cost,
                    detour,
                    reason=reason,
                )
            )
        return tuple(risks)

    def analyze(
        self,
        start: Location,
        goal: Goal,
        intermediate_destinations: tuple[Goal, ...] = (),
        *,
        algorithm: str = "dijkstra",
        cancel_check=None,
    ) -> RouteAnalysis:
        """Return the normal route and independently composed candidate routes."""

        def plan(world, origin, target):
            kwargs = {"algorithm": algorithm}
            if self._max_expansions is not None:
                kwargs["max_expansions"] = self._max_expansions
            if self._max_route_cost is not None:
                kwargs["max_route_cost"] = self._max_route_cost
            if cancel_check is not None:
                kwargs["cancel_check"] = cancel_check
            result = self._planner(world, origin, target, self.graph, **kwargs)
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
        trainer_risks = self._trainer_risks(start, goal, normal, normal_cost, plan)
        return RouteAnalysis(goal, normal, normal_cost, tuple(candidates), trainer_risks)


# NORMAL is deliberately opportunity-first.  The second component keeps the
# result practical and deterministic after exposure has been minimized; it is
# not allowed to trade an extra encounter opportunity for fewer steps.
NORMAL_ENCOUNTER_OPPORTUNITY_WEIGHT = 1
NORMAL_MOVEMENT_COST_SCALE = 1


def _goal_requires_orientation(target: Goal) -> bool:
    """Return whether the search must preserve avatar facing.

    Ordinary location routing only needs a sequence of cardinal inputs.  The
    executor already handles the first input as a turn when the avatar is
    facing another direction, so carrying facing through every movement
    state only multiplies the search space.  Interaction and transition
    goals are different: their activation preconditions can depend on the
    final facing, so those searches retain the orientation dimension.
    """

    return isinstance(target, (ActivateTrigger, ReachInteractionPosition, EngageTrainer, ReachWarp))


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
    target = goal.target if isinstance(goal, NavigationGoal) else goal
    if isinstance(target, SemanticTarget):
        return target.target_map or (target.location[0] if target.location else None)
    if isinstance(target, EngageTrainer):
        matching_maps = {
            trigger_location[0]
            for trigger in world.triggers
            if trigger.affordance_id == target.trainer_id
            for trigger_location in (
                *trigger.locations,
                *trigger.navigation_locations,
                *trigger.activation_locations,
                *getattr(trigger, "engagement_locations", ()),
                *trigger.hazard_locations,
            )
        }
        if len(matching_maps) == 1:
            return next(iter(matching_maps))

        # Trainer goals are often promoted while the player is still on the
        # previous map. The runtime trigger is therefore not guaranteed to be
        # present in the current observation. Resolve the stable affordance
        # identity from the verified trainer catalog first, then from the
        # map/local-id encoding used by every live trainer observation.
        try:
            from modules.trainer_checkpoints import trainer_checkpoint_for_id

            checkpoint = trainer_checkpoint_for_id(target.trainer_id)
        except (ImportError, AttributeError, TypeError, ValueError):
            checkpoint = None
        if checkpoint is not None:
            return checkpoint.map_id

        if isinstance(target.trainer_id, str):
            parts = target.trainer_id.split(":", 2)
            if len(parts) == 3 and parts[0] == "trainer":
                try:
                    parsed_map = tuple(int(part.strip()) for part in parts[1].strip("() ").split(","))
                    int(parts[2])
                except (TypeError, ValueError):
                    parsed_map = ()
                if len(parsed_map) == 2:
                    return parsed_map
        return None
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
    allow_pokemon_center_destination: bool = False,
) -> NavigationWorld:
    """Build the lazy static world overlay used by the exact-state search.

    Map metadata is materialized only for maps reachable through the coarse
    graph; dynamic occupancy remains applied only to the currently observed
    map.  ``WorldEdge`` candidates stay separate, including connection spans.
    """
    observed_maps = {location[0] for location in observation_world.tiles}
    # Live perception is authoritative for the current map, but it is not
    # guaranteed to expose every exit (notably interior exits while an object
    # table is still settling).  Keep the observed transitions and use ROM
    # transitions as fallback avenues on every map involved in the search.
    transitions: list[WorldTransition] = list(observation_world.transitions or observation_world.warps)
    rom_transitions: dict[MapId, tuple[WorldTransition, ...]] = {}
    frozen_transition_catalog = {
        canonical_map_id(map_id): tuple(transitions_for_map)
        for map_id, transitions_for_map in observation_world.static_transition_catalog
    }
    observed_maps = {location[0] for location in observation_world.tiles}
    maps_to_enrich = tuple(dict.fromkeys((*observed_maps, *enrich_maps)))
    for map_id in maps_to_enrich:
        map_id = canonical_map_id(map_id)
        if observation_world.static_metadata_frozen:
            rom_transitions[map_id] = frozen_transition_catalog.get(map_id, ())
        else:
            try:
                rom_transitions[map_id] = static_map_transitions(map_id)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # A partially available map should not make a semantic route
                # disappear. The graph transition remains a valid fallback.
                continue
    # Expand the current map only when a ROM avenue leads beyond the live
    # tile window. Complete synthetic/live maps should continue to use their
    # observed topology without being replaced by unrelated ROM coordinates.
    current_static_needed = False
    current_map = next(iter(observed_maps), None)
    current_candidates = tuple(rom_transitions.get(current_map, ())) + tuple(
        WorldTransition(
            (edge.source_map, source),
            (edge.destination_map, destination),
            required_facing=(edge.required_facing if isinstance(edge.required_facing, Direction) else None),
            kind="map_connection" if edge.kind == "connection" else "warp",
        )
        for edge in graph.edges
        if edge.source_map == current_map
        for source, destination in zip(edge.source_coordinates, edge.destination_coordinates)
    )
    for transition in current_candidates:
        source = transition_approach_position(transition) if transition.kind == "map_connection" else transition.entry
        if source is not None and source not in observation_world.tiles:
            current_static_needed = True
            break
    tiles: Mapping[Location, NavigableTile] = _LazyStaticTileMapping(
        observation_world.tiles,
        {edge.source_map for edge in graph.edges} | {edge.destination_map for edge in graph.edges},
        static_maps=observed_maps if current_static_needed else set(),
        static_models=observation_world.static_tile_catalog,
        metadata_frozen=observation_world.static_metadata_frozen,
    )
    def transition_key(transition: WorldTransition) -> tuple:
        """Identify a transition by the state from which the ROM executes it.

        A door can have more than one ROM warp record for the same field
        interaction.  The entry tile is metadata for the warp record, while
        ``activation_locations`` is the executable source used by navigation.
        If live perception and the static overlay expose different entry
        records for the same source and destination, keeping both creates two
        competing inputs (and can select the wrong facing at a Center door).
        Live transitions are inserted first, so this key preserves the
        executable observation and treats static records as fallback data.
        """

        activation_sources = tuple(sorted(getattr(transition, "activation_locations", ()), key=repr))
        source = (
            activation_sources
            if activation_sources
            else (
                transition_approach_position(transition)
                if transition.kind == "map_connection"
                else transition.entry
            )
        )
        return transition.kind, source, transition.destination

    known_entries = {transition_key(t) for t in transitions}
    for map_id, candidates in rom_transitions.items():
        for transition in candidates:
            transition = effective_transition(transition)
            key = transition_key(transition)
            if key not in known_entries:
                transitions.append(transition)
                known_entries.add(key)
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
            key = transition_key(transition)
            if key not in known_entries:
                transitions.append(transition)
                known_entries.add(key)
            if transition.destination is not None:
                # Destination endpoints are known without loading the whole
                # destination map; the lazy mapping resolves their neighbors
                # only if the search actually visits that map.
                model = tiles._models.setdefault(transition.destination[0], {})
                model.setdefault(
                    transition.destination, NavigableTile(transition.destination, False, frozenset(Direction))
                )
    # Static trainer templates are needed on destination maps before their
    # runtime object events enter the camera. Without this overlay an
    # encounter route can cross a trainer sight line and only learn about the
    # trainer after the battle has already started. Current-map perception
    # already publishes these triggers, so enrich only unseen maps.
    enriched_triggers = list(observation_world.triggers)
    if not observation_world.static_metadata_frozen:
        for map_id in maps_to_enrich:
            map_id = canonical_map_id(map_id)
            try:
                from modules.map import get_map_metadata

                map_metadata = get_map_metadata(map_id)
                static_map_tiles = prewarm_static_map_observation(map_id)
                # Use an empty runtime set here intentionally. A runtime
                # object can exist while its facing/range fields are still
                # incomplete during a camera boundary. In that case the
                # current observation contains the trainer identity but no
                # usable sight line; the ROM template is the safe fallback.
                trainers = static_trainer_observations(map_id, map_metadata, ())
                static_objects = tuple(trainers)
                for trainer in trainers:
                    matching = tuple(
                        trigger
                        for trigger in enriched_triggers
                        if getattr(trigger, "affordance_id", None) == trainer.trainer_id
                    )
                    if any(getattr(trigger, "trainer_defeated", None) is True for trigger in matching):
                        continue
                    activation_requirements = resolve_object_activation_positions(
                        trainer,
                        static_objects,
                        static_map_tiles,
                        map_metadata.map_size,
                    )
                    hazard_locations = trainer_hazard_locations(trainer, static_map_tiles, static_objects)
                    # An observed non-empty envelope is authoritative for the
                    # live trainer orientation. Add the ROM fallback only
                    # when the current-map trigger is absent or geometrically
                    # unresolved. This also covers unseen destination maps.
                    if not hazard_locations or any(
                        getattr(trigger, "hazard_locations", ()) for trigger in matching
                    ):
                        continue
                    enriched_triggers.append(
                        TriggerObservation(
                            trigger_id=f"static_object:{trainer.local_id}:{trainer.script}",
                            locations=frozenset({trainer.location}),
                            activation_locations=frozenset(location for location, _ in activation_requirements),
                            kind="trainer_hazard",
                            affordance_id=trainer.trainer_id,
                            hazard_locations=hazard_locations,
                            hazard_kind="trainer",
                            engagement_locations=trainer_engagement_locations(trainer, static_map_tiles, static_objects),
                            trainer_defeated=trainer.trainer_defeated,
                            requires_input=False,
                        )
                    )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                # Destination-map metadata can be unavailable during a ROM
                # transition. The coarse route remains usable; readiness will
                # retry trainer discovery at the next stable boundary.
                continue

    if not allow_pokemon_center_destination:
        center_maps = _pokemon_center_map_ids()
        transitions = [
            transition
            for transition in transitions
            if transition.entry[0] in center_maps
            or transition.destination is None
            or transition.destination[0] not in center_maps
        ]
    return NavigationWorld(
        tiles=tiles,
        warps=observation_world.warps,
        triggers=tuple(enriched_triggers),
        bindings=observation_world.bindings,
        facing=observation_world.facing,
        transitions=tuple(transitions),
        running_shoes=observation_world.running_shoes,
        player_elevation=observation_world.player_elevation,
        surfing=observation_world.surfing,
        static_tile_catalog=observation_world.static_tile_catalog,
        static_transition_catalog=observation_world.static_transition_catalog,
        static_metadata_frozen=observation_world.static_metadata_frozen,
    )


def _plan_exact_cross_map_warp_chain(
    world: NavigationWorld,
    start: Location,
    navigation_goal: NavigationGoal,
    route,
    *,
    max_expansions: int | None = None,
    max_route_cost: int | None = None,
    cost_ceiling: tuple[int, ...] | int | None = None,
    cancel_check=None,
    caller: str | None = None,
) -> NavigationPlan | None:
    """Plan a selected cross-map warp one map transition at a time.

    A final Center door is a semantic recovery target with one known ROM
    transition. Searching the entire enriched Emerald world for that terminal
    state spends most of its budget exploring unrelated tiles. The coarse map
    route already supplies the map order, so each segment can terminate at the
    next exact transition and hand its destination to the following segment.
    Return ``None`` only when the transition overlay is incomplete; callers can
    then use the general state search as a compatibility fallback.
    """

    selected = navigation_goal.target.warp

    def abort(reason: str, **details):
        diagnostic_print(
            lambda: (
                "CAMPAIGN_EXACT_WARP_CHAIN: "
                f"status='fallback' reason={reason!r} "
                + " ".join(f"{key}={value!r}" for key, value in details.items())
            ),
            trace=True,
            prefix="CAMPAIGN_EXACT_WARP_CHAIN",
        )
        return None

    if selected is None or selected.destination is None:
        return abort("selected_warp_unresolved")
    edges = tuple(getattr(route, "edges", ()) or ())
    if not edges:
        return abort("map_route_empty", start=start, target=selected.destination)

    transitions = tuple(world.transitions or world.warps)
    current = start
    actions: list[NavigationAction] = []
    total_cost = 0
    navigator = GoalAwareNavigator(world)

    for edge_index, edge in enumerate(edges):
        if cancel_check is not None:
            cancel_check()
        if current[0] != edge.source_map:
            return abort(
                "map_route_discontinuity",
                edge_index=edge_index,
                current=current,
                expected=edge.source_map,
                route_maps=getattr(route, "maps", None),
            )
        candidates = tuple(
            transition
            for transition in transitions
            if transition.entry[0] == edge.source_map
            and transition.destination is not None
            and transition.destination[0] == edge.destination_map
            and (
                transition.kind == "map_connection"
                if edge.kind == "connection"
                else transition.kind == edge.kind
            )
        )
        if edge_index == len(edges) - 1:
            candidates = tuple(
                transition for transition in candidates if transitions_match(transition, selected)
            )
        if not candidates:
            return abort(
                "transition_candidates_missing",
                edge_index=edge_index,
                source_map=edge.source_map,
                destination_map=edge.destination_map,
                kind=edge.kind,
                transition_count=len(transitions),
                route_maps=getattr(route, "maps", None),
            )

        def candidate_distance(transition: WorldTransition) -> tuple[int, str]:
            approach = transition_approach_position(transition)
            coordinate = approach[1] if approach is not None else transition.entry[1]
            return (
                abs(current[1][0] - coordinate[0]) + abs(current[1][1] - coordinate[1]),
                repr(transition.entry),
            )

        # Map connection records are commonly listed in map-data order rather
        # than travel order.  Trying the nearest boundary first gives the
        # bounded search an incumbent quickly; the branch-and-bound check
        # below then avoids spending thousands of expansions on distant,
        # equivalent exits.
        candidates = tuple(sorted(candidates, key=candidate_distance))

        shared_local_plans: dict[Location, NavigationPlan] | None = None
        shared_search_attempted = False
        if len(candidates) > 1 and all(transition.kind == "map_connection" for transition in candidates):
            approaches = tuple(
                approach
                for transition in candidates
                if (approach := transition_approach_position(transition)) is not None
            )
            if len(approaches) == len(candidates):
                shared_search_attempted = True
                # All connection candidates share the same local movement
                # graph. One multi-target search is substantially cheaper
                # than running an independent A* for every boundary tile.
                # Remove the transitions from this local snapshot so the
                # search cannot cross a boundary while looking for an
                # approach position.
                local_world = replace(world, transitions=(), warps=())
                try:
                    shared_local_plans = GoalAwareNavigator(local_world).plan_many_locations(
                        current,
                        approaches,
                        target_groups=(approaches,),
                        encounter_mode=navigation_goal.encounter_mode,
                        encounter_penalty=navigation_goal.encounter_penalty,
                        constraints=navigation_goal.constraints,
                        max_expansions=max_expansions,
                        max_route_cost=(
                            None if max_route_cost is None else max_route_cost - total_cost
                        ),
                        cancel_check=cancel_check,
                    )
                    if not shared_local_plans:
                        shared_local_plans = None
                except NavigationSearchLimitExceeded:
                    # Preserve the individual-candidate fallback when a
                    # partially observed map cannot resolve every approach
                    # within the shared search budget.
                    shared_local_plans = None

        segment_plans: list[tuple[NavigationPlan, WorldTransition]] = []
        for candidate_index, transition in enumerate(candidates):
            if cancel_check is not None:
                cancel_check()
            if shared_search_attempted and shared_local_plans is None and candidate_index > 0:
                # The shared safe search already established that no
                # trainer-free approach was available within its bound. Give
                # the nearest live boundary one local compatibility attempt;
                # do not multiply the fallback cost by every map-data row.
                break
            if segment_plans:
                best_segment = min(
                    segment_plans,
                    key=lambda item: navigation_candidate_key(
                        item[0].metrics,
                        navigation_goal.encounter_mode,
                        repr(item[1].entry),
                    )
                    if item[0].metrics is not None
                    else (float("inf"),) * 6 + (repr(item[1].entry),),
                )[0]
                best_encounters = best_segment.metrics.encounter_opportunities if best_segment.metrics else None
                best_cost = best_segment.metrics.total_route_cost if best_segment.metrics else None
                lower_bound = candidate_distance(transition)[0]
                # A Manhattan lower bound is conservative even when terrain,
                # encounter weighting, or the transition itself adds cost.
                # Once the incumbent cannot be beaten, skip this boundary.
                encounter_objective_complete = (
                    navigation_goal.encounter_mode in (EncounterMode.AVOID, EncounterMode.NORMAL)
                    and best_encounters == 0
                )
                weighted_objective_complete = navigation_goal.encounter_mode is EncounterMode.MOSTLY_AVOID
                if (
                    best_cost is not None
                    and (encounter_objective_complete or weighted_objective_complete)
                    and lower_bound >= best_cost
                ):
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_EXACT_WARP_CHAIN: "
                            f"status='pruned' edge_index={edge_index!r} "
                            f"entry={transition.entry!r} lower_bound={lower_bound!r} "
                            f"incumbent_cost={best_cost!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_EXACT_WARP_CHAIN",
                    )
                    continue
            # Each segment has one executable terminal transition.  Keeping
            # unrelated world transitions out of the local search prevents a
            # Route 116 search from repeatedly considering every warp and
            # connection in the enriched Emerald world.
            segment_world = replace(world, transitions=(transition,), warps=())
            segment_goal = NavigationGoal(
                ReachWarp(
                    destination_map=transition.destination[0],
                    destination=transition.destination,
                    warp=transition,
                ),
                constraints=navigation_goal.constraints,
                encounter_mode=navigation_goal.encounter_mode,
                encounter_penalty=navigation_goal.encounter_penalty,
            )
            remaining_cost = (
                None if max_route_cost is None else max_route_cost - total_cost
            )
            if remaining_cost is not None and remaining_cost < 0:
                continue
            try:
                if shared_local_plans is not None:
                    approach = transition_approach_position(transition)
                    local_plan = shared_local_plans.get(approach) if approach is not None else None
                    if local_plan is None:
                        continue
                    boundary_move = NavigationAction(
                        NavigationActionType.MOVE,
                        transition.required_facing,
                        approach,
                        transition.entry,
                    )
                    crossing = NavigationAction(
                        NavigationActionType.WARP,
                        transition.required_facing,
                        transition.entry,
                        transition.destination,
                        transition_kind="map_connection",
                    )
                    segment_actions = local_plan.actions + (boundary_move, crossing)
                    segment = NavigationPlan(
                        segment_actions,
                        transition.entry,
                        navigator._metrics(
                            segment_actions,
                            navigation_goal.encounter_mode,
                            navigation_goal.encounter_penalty,
                        ),
                    )
                    _enforce_plan_limits(
                        segment,
                        max_route_cost=remaining_cost,
                        cost_ceiling=(cost_ceiling if edge_index == 0 and total_cost == 0 else None),
                    )
                else:
                    segment = plan_observed_warp_locally(
                        segment_world,
                        current,
                        segment_goal,
                        caller=f"{caller}.exact_warp_chain" if caller else "navigation.exact_warp_chain",
                        max_expansions=max_expansions,
                        max_route_cost=remaining_cost,
                        cost_ceiling=(cost_ceiling if edge_index == 0 and total_cost == 0 else None),
                        cancel_check=cancel_check,
                    )
            except (NavigationError, NavigationSearchLimitExceeded):
                continue
            segment_plans.append((segment, transition))

        if not segment_plans:
            return abort(
                "segment_unreachable",
                edge_index=edge_index,
                source=current,
                candidate_count=len(candidates),
                route_maps=getattr(route, "maps", None),
            )
        segment, transition = min(
            segment_plans,
            key=lambda item: navigation_candidate_key(
                item[0].metrics,
                navigation_goal.encounter_mode,
                repr(item[1].entry),
            )
            if item[0].metrics is not None
            else (float("inf"),) * 6 + (repr(item[1].entry),),
        )
        actions.extend(segment.actions)
        if segment.metrics is not None:
            total_cost += segment.metrics.total_route_cost
        current = transition.destination

    metrics = navigator._metrics(
        tuple(actions),
        navigation_goal.encounter_mode,
        navigation_goal.encounter_penalty,
    )
    result = NavigationPlan(tuple(actions), current, metrics)
    try:
        _enforce_plan_limits(result, max_route_cost=max_route_cost)
    except NavigationSearchLimitExceeded:
        return abort(
            "combined_plan_limit",
            action_count=len(actions),
            route_maps=getattr(route, "maps", None),
        )
    return result


def _semantic_interaction_local_goal(world: NavigationWorld, target: SemanticTarget):
    """Resolve a semantic interaction to the trigger visible on its map."""

    if target.location is not None:
        return ReachLocation(target.location)
    interaction_id = target.interaction_id
    if interaction_id is None:
        return None
    matching = tuple(
        trigger
        for trigger in world.triggers
        if interaction_id
        in {
            getattr(trigger, "trigger_id", None),
            getattr(trigger, "script_symbol", None),
            getattr(trigger, "affordance_id", None),
        }
        and (
            getattr(trigger, "activation_locations", ())
            or getattr(trigger, "navigation_locations", ())
        )
    )
    if not matching:
        return None
    return ReachInteractionPosition(matching[0].trigger_id)


def _semantic_transition_candidates(
    transitions: tuple[WorldTransition, ...],
    edge,
) -> tuple[WorldTransition, ...]:
    """Return executable transitions represented by one static graph edge.

    ``WorldEdge`` coordinates describe the ROM topology, while the live/static
    transition catalog carries the executable activation geometry.  Match the
    source boundary when it is available, but allow the destination coordinate
    to differ because a runtime transition observation may have corrected it.
    """

    expected_kind = "map_connection" if edge.kind == "connection" else edge.kind
    source_coordinates = frozenset(edge.source_coordinates or ())
    candidates = []
    for transition in transitions:
        if transition.entry[0] != edge.source_map:
            continue
        if transition.destination is None or transition.destination[0] != edge.destination_map:
            continue
        if transition.kind != expected_kind:
            continue
        if source_coordinates and transition.entry[1] not in source_coordinates:
            continue
        candidates.append(transition)
    return tuple(
        sorted(
            candidates,
            key=lambda transition: (
                repr(transition.entry),
                repr(transition.destination),
                repr(transition_approach_position(transition)),
            ),
        )
    )


def _semantic_portal_sources(transition: WorldTransition) -> tuple[Location, ...]:
    """Return the local states from which a transition can be executed."""

    if transition.kind == "map_connection":
        approach = transition_approach_position(transition)
        return (approach,) if approach is not None else ()
    if transition.activation_locations:
        return tuple(sorted(transition.activation_locations, key=repr))
    return (transition.entry,)


def _semantic_cost_tuple(metrics: NavigationMetrics, encounter_mode: EncounterMode) -> tuple[int, ...]:
    """Return the additive lexicographic cost used by portal routing."""

    return (
        metrics.encounter_opportunities
        if encounter_mode in (EncounterMode.AVOID, EncounterMode.NORMAL)
        else 0,
        metrics.total_route_cost,
        metrics.movement_actions,
        metrics.encounter_terrain_moves,
        metrics.turns_on_encounter_terrain,
        metrics.turns_on_non_encounter_terrain,
    )


def _add_semantic_cost(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[index] + right[index] for index in range(len(left)))


def _semantic_portal_segment_plan(
    world: NavigationWorld,
    local_plan: NavigationPlan,
    transition: WorldTransition,
    navigation_goal: NavigationGoal,
) -> NavigationPlan:
    """Add one executable transition to a local entrance-to-portal plan."""

    source = local_plan.destination
    if source is None or transition.destination is None:
        raise NavigationError("portal segment has no executable source or destination")
    if transition.kind == "map_connection":
        approach = transition_approach_position(transition)
        if approach is None or source != approach:
            raise NavigationError("portal segment does not end at the map-connection approach")
        direction = transition.required_facing
        if direction is None:
            raise NavigationError("map connection has no crossing direction")
        actions = local_plan.actions + (
            NavigationAction(
                NavigationActionType.MOVE,
                direction,
                approach,
                transition.entry,
                transition_kind="map_connection",
            ),
            NavigationAction(
                NavigationActionType.WARP,
                direction,
                transition.entry,
                transition.destination,
                transition_kind="map_connection",
            ),
        )
    else:
        direction = transition.activation_direction
        if direction is None and transition.activation_locations:
            direction = _direction_between(source[1], transition.entry[1])
        if direction is None:
            direction = transition.required_facing or Direction.South
        actions = local_plan.actions + (
            NavigationAction(
                NavigationActionType.WARP,
                direction,
                source,
                transition.destination,
                transition_kind=transition.kind,
            ),
        )
    navigator = GoalAwareNavigator(world)
    return NavigationPlan(
        actions,
        transition.entry,
        navigator._metrics(
            actions,
            navigation_goal.encounter_mode,
            navigation_goal.encounter_penalty,
        ),
    )


def _specialize_semantic_edge(edge, transition: WorldTransition):
    """Keep the selected portal pair on the route returned to callers."""

    return replace(
        edge,
        source_coordinates=(transition.entry[1],),
        destination_coordinates=(transition.destination[1],),
    )


def _route_for_map_sequence(
    graph: WorldMapGraph,
    map_sequence: tuple[MapId, ...],
    *,
    start_map: MapId,
    target_map: MapId,
) -> WorldRoute:
    """Resolve a caller-supplied map corridor into graph edges.

    Campaign preparation already has a ROM-backed corridor selected by the
    campaign route analysis.  Resolve that corridor once here and let the
    semantic planner choose only among executable transitions represented by
    each edge.  A malformed or stale corridor is rejected explicitly so the
    caller can re-observe and choose a new one instead of silently widening
    the search back to every map in the world.
    """

    maps = tuple(map_sequence)
    if len(maps) < 2 or maps[0] != start_map or maps[-1] != target_map:
        raise WorldNavigationError(
            f"preferred map route does not connect {start_map!r} to {target_map!r}: {maps!r}"
        )
    edges = []
    for source_map, destination_map in zip(maps, maps[1:]):
        candidates = tuple(
            edge
            for edge in graph.outgoing(source_map)
            if edge.destination_map == destination_map
        )
        if not candidates:
            raise WorldNavigationError(
                f"preferred map route edge is absent: {source_map!r} -> {destination_map!r}"
            )
        edges.append(
            min(
                candidates,
                key=lambda edge: (
                    max(1, edge.estimated_cost),
                    edge.kind,
                    repr(edge.source_coordinates),
                ),
            )
        )
    return WorldRoute(
        maps,
        tuple(edges),
        estimated_cost=sum(max(1, edge.estimated_cost) for edge in edges),
    )


def _weighted_semantic_world_route(
    world: NavigationWorld,
    start: Location,
    navigation_goal: NavigationGoal,
    graph: WorldMapGraph,
    *,
    max_expansions: int | None = None,
    max_route_cost: int | None = None,
    cost_ceiling: tuple[int, ...] | int | None = None,
    cancel_check=None,
    corridor: WorldRoute | None = None,
) -> WorldRoute | None:
    """Choose a cross-map route using exact local portal costs.

    The static map graph supplies directed topology.  Each search state is an
    arrival location on a map; expanding it runs one shared local search to
    every executable exit on that map.  The resulting transition destination
    becomes the next state, so an intermediate map is priced from its actual
    entrance to each possible exit instead of receiving one universal edge
    weight.
    """

    target = navigation_goal.target
    if not isinstance(target, SemanticTarget) or target.target_map is None:
        return None

    transitions = tuple(world.transitions or world.warps)
    # Portal ranking is a static movement estimate.  Dynamic blockers and
    # trainer observations remain in the supplied tile/trigger overlays, but
    # transitions are removed so a local search cannot cross two map edges at
    # once while pricing one map's entrance-to-exit movement.
    local_world = replace(world, transitions=(), warps=(), facing=None)
    local_navigator = GoalAwareNavigator(local_world)
    zero_cost = (0, 0, 0, 0, 0, 0)
    queue: list[tuple[tuple[int, ...], int, Location]] = [(zero_cost, 0, start)]
    sequence = count(1)
    best_cost: dict[Location, tuple[int, ...]] = {start: zero_cost}
    route_records: dict[Location, tuple[tuple, tuple[WorldTransition, ...]]] = {start: ((), ())}

    final_local_goal = None
    if target.kind is SemanticTargetKind.LOCATION:
        final_local_goal = ReachLocation(target.location) if target.location is not None else None
    elif target.kind is SemanticTargetKind.INTERACTION:
        final_local_goal = _semantic_interaction_local_goal(world, target)

    best_terminal: tuple[tuple[int, ...], Location, tuple, tuple[WorldTransition, ...]] | None = None

    def exceeds_cost_ceiling(cost: tuple[int, ...]) -> bool:
        if cost_ceiling is None:
            return False
        if isinstance(cost_ceiling, int):
            return cost[1] > cost_ceiling
        prefix = (cost[0], cost[1])
        return bool(cost_ceiling) and prefix[: len(cost_ceiling)] > cost_ceiling

    corridor_edges: dict[MapId, tuple[WorldEdge, ...]] | None = None
    if corridor is not None:
        corridor_edges = {}
        for edge in corridor.edges:
            corridor_edges[edge.source_map] = (edge,)

    while queue:
        if cancel_check is not None:
            cancel_check()
        current_cost, _, current = heapq.heappop(queue)
        if current_cost != best_cost.get(current):
            continue
        if best_terminal is not None and current_cost >= best_terminal[0]:
            break

        edge_path, transition_path = route_records[current]
        if current[0] == target.target_map:
            if target.kind is SemanticTargetKind.MAP or final_local_goal is None:
                best_terminal = (current_cost, current, edge_path, transition_path)
                break
            remaining_cost = (
                None if max_route_cost is None else max_route_cost - current_cost[1]
            )
            if remaining_cost is not None and remaining_cost < 0:
                continue
            try:
                final_plan = local_navigator.plan(
                    current,
                    replace(navigation_goal, target=final_local_goal),
                    max_expansions=max_expansions,
                    max_route_cost=remaining_cost,
                    cancel_check=cancel_check,
                )
            except NavigationSearchCancelled:
                raise
            except (NavigationError, NavigationSearchLimitExceeded, RuntimeError, TypeError, ValueError):
                final_plan = None
            if final_plan is not None and final_plan.metrics is not None:
                final_cost = _add_semantic_cost(
                    current_cost,
                    _semantic_cost_tuple(final_plan.metrics, navigation_goal.encounter_mode),
                )
                if max_route_cost is None or final_cost[1] <= max_route_cost:
                    if exceeds_cost_ceiling(final_cost):
                        continue
                    candidate = (final_cost, current, edge_path, transition_path)
                    if best_terminal is None or candidate[0] < best_terminal[0]:
                        best_terminal = candidate
            continue

        outgoing: list[tuple[object, WorldTransition]] = []
        candidate_edges = (
            corridor_edges.get(current[0], ())
            if corridor_edges is not None
            else graph.outgoing(current[0])
        )
        for edge in candidate_edges:
            for transition in _semantic_transition_candidates(transitions, edge):
                outgoing.append((edge, transition))
        if not outgoing:
            continue

        target_locations = tuple(
            dict.fromkeys(
                source
                for _, transition in outgoing
                for source in _semantic_portal_sources(transition)
            )
        )
        if not target_locations:
            continue
        remaining_cost = None if max_route_cost is None else max_route_cost - current_cost[1]
        if remaining_cost is not None and remaining_cost < 0:
            continue
        try:
            local_plans = local_navigator.plan_many_locations(
                current,
                target_locations,
                encounter_mode=navigation_goal.encounter_mode,
                encounter_penalty=navigation_goal.encounter_penalty,
                constraints=navigation_goal.constraints,
                max_expansions=max_expansions,
                max_route_cost=remaining_cost,
                cancel_check=cancel_check,
            )
        except NavigationSearchCancelled:
            raise
        except (NavigationSearchLimitExceeded, NavigationError, RuntimeError, TypeError, ValueError):
            continue

        best_for_transition: dict[WorldTransition, tuple[NavigationPlan, object]] = {}
        for edge, transition in outgoing:
            options = tuple(
                (local_plans[source], source)
                for source in _semantic_portal_sources(transition)
                if source in local_plans
            )
            if not options:
                continue
            selected_plan, selected_source = min(
                options,
                key=lambda item: navigation_candidate_key(
                    item[0].metrics,
                    navigation_goal.encounter_mode,
                    repr(item[1]),
                )
                if item[0].metrics is not None
                else (float("inf"),) * 6 + (repr(item[1]),),
            )
            best_for_transition[transition] = (selected_plan, edge)

        for transition, (local_plan, edge) in best_for_transition.items():
            try:
                segment = _semantic_portal_segment_plan(
                    local_world,
                    local_plan,
                    transition,
                    navigation_goal,
                )
            except (NavigationError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            if segment.metrics is None:
                continue
            segment_cost = _semantic_cost_tuple(segment.metrics, navigation_goal.encounter_mode)
            next_cost = _add_semantic_cost(current_cost, segment_cost)
            if max_route_cost is not None and next_cost[1] > max_route_cost:
                continue
            if exceeds_cost_ceiling(next_cost):
                continue
            destination = transition.destination
            specialized_edge = _specialize_semantic_edge(edge, transition)
            previous = best_cost.get(destination)
            if previous is not None and next_cost >= previous:
                continue
            best_cost[destination] = next_cost
            route_records[destination] = (
                edge_path + (specialized_edge,),
                transition_path + (transition,),
            )
            heapq.heappush(queue, (next_cost, next(sequence), destination))

    if best_terminal is None:
        return None
    final_cost, destination, edge_path, _ = best_terminal
    maps = [start[0]]
    for edge in edge_path:
        if not maps or maps[-1] != edge.destination_map:
            maps.append(edge.destination_map)
    return WorldRoute(
        tuple(maps),
        tuple(edge_path),
        estimated_cost=final_cost[1],
        selected_transitions=best_terminal[3],
    )


def _plan_semantic_cross_map_chain(
    world: NavigationWorld,
    start: Location,
    navigation_goal: NavigationGoal,
    route,
    graph: WorldMapGraph | None = None,
    *,
    max_expansions: int | None = None,
    max_route_cost: int | None = None,
    cost_ceiling: tuple[int, ...] | int | None = None,
    cancel_check=None,
    caller: str | None = None,
) -> tuple[NavigationPlan, WorldRoute] | None:
    """Plan a semantic destination by map route, then solve its final map locally.

    The ROM map route already fixes the map order, so solve each boundary as
    an exact local transition instead of searching unrelated tiles in the
    enriched world. Map goals are complete after the final boundary; location
    goals search only their destination map; interaction goals search only for
    the resolved trigger position there.
    """

    target = navigation_goal.target
    if not isinstance(target, SemanticTarget) or target.kind not in {
        SemanticTargetKind.MAP,
        SemanticTargetKind.LOCATION,
        SemanticTargetKind.INTERACTION,
    }:
        return None
    if graph is not None:
        weighted_route = _weighted_semantic_world_route(
            world,
            start,
            navigation_goal,
            graph,
            max_expansions=max_expansions,
            max_route_cost=max_route_cost,
            cost_ceiling=cost_ceiling,
            cancel_check=cancel_check,
            corridor=route if navigation_goal.preferred_map_route is not None else None,
        )
        if weighted_route is None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_SEMANTIC_ROUTE: "
                    "status='fallback' reason='no_weighted_portal_route'"
                ),
                trace=True,
                prefix="CAMPAIGN_SEMANTIC_ROUTE",
            )
            return None
        route = weighted_route
    edges = tuple(getattr(route, "edges", ()) or ())
    if not edges:
        return None

    transitions = tuple(world.transitions or world.warps)
    current = start
    actions: list[NavigationAction] = []
    total_cost = 0
    navigator = GoalAwareNavigator(world)

    for edge_index, edge in enumerate(edges):
        if cancel_check is not None:
            cancel_check()
        if current[0] != edge.source_map:
            return None
        candidates = _semantic_transition_candidates(transitions, edge)
        selected_transitions = tuple(getattr(route, "selected_transitions", ()) or ())
        if edge_index < len(selected_transitions):
            selected = selected_transitions[edge_index]
            candidates = tuple(
                transition
                for transition in candidates
                if transitions_match(transition, selected)
            )
        if not candidates:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_SEMANTIC_ROUTE: "
                    f"status='fallback' reason='transition_candidates_missing' edge_index={edge_index} "
                    f"source_map={edge.source_map!r} destination_map={edge.destination_map!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_SEMANTIC_ROUTE",
            )
            return None

        def candidate_distance(transition):
            approach = transition_approach_position(transition)
            coordinate = approach[1] if approach is not None else transition.entry[1]
            return abs(current[1][0] - coordinate[0]) + abs(current[1][1] - coordinate[1]), repr(transition.entry)

        segment_plans = []
        for transition in sorted(candidates, key=candidate_distance):
            if cancel_check is not None:
                cancel_check()
            remaining_cost = None if max_route_cost is None else max_route_cost - total_cost
            if remaining_cost is not None and remaining_cost < 0:
                continue
            segment_world = replace(world, transitions=(transition,), warps=())
            segment_goal = NavigationGoal(
                ReachWarp(
                    destination_map=transition.destination[0],
                    destination=transition.destination,
                    warp=transition,
                ),
                constraints=navigation_goal.constraints,
                encounter_mode=navigation_goal.encounter_mode,
                encounter_penalty=navigation_goal.encounter_penalty,
            )
            try:
                segment = plan_observed_warp_locally(
                    segment_world,
                    current,
                    segment_goal,
                    caller=f"{caller}.semantic_warp_chain" if caller else "navigation.semantic_warp_chain",
                    max_expansions=max_expansions,
                    max_route_cost=remaining_cost,
                    cost_ceiling=(cost_ceiling if edge_index == 0 and total_cost == 0 else None),
                    cancel_check=cancel_check,
                )
            except NavigationSearchCancelled:
                raise
            except (NavigationError, NavigationSearchLimitExceeded, RuntimeError, TypeError, ValueError) as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_SEMANTIC_ROUTE: "
                        f"status='segment_candidate_rejected' edge_index={edge_index} "
                        f"transition={transition!r} error_type={type(error).__name__!r} error={str(error)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_SEMANTIC_ROUTE",
                )
                continue
            segment_plans.append((segment, transition))
        if not segment_plans:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_SEMANTIC_ROUTE: "
                    f"status='fallback' reason='segment_unreachable' edge_index={edge_index} "
                    f"source={current!r} candidate_count={len(candidates)}"
                ),
                trace=True,
                prefix="CAMPAIGN_SEMANTIC_ROUTE",
            )
            return None
        segment, transition = min(
            segment_plans,
            key=lambda item: (
                getattr(getattr(item[0], "metrics", None), "encounter_opportunities", float("inf")),
                getattr(getattr(item[0], "metrics", None), "total_route_cost", float("inf")),
                repr(item[1].entry),
            ),
        )
        actions.extend(segment.actions)
        total_cost += getattr(getattr(segment, "metrics", None), "total_route_cost", 0) or 0
        current = transition.destination

    if current[0] != target.target_map:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_SEMANTIC_ROUTE: "
                f"status='fallback' reason='destination_map_not_reached' "
                f"map={current[0]!r} target_map={target.target_map!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_SEMANTIC_ROUTE",
        )
        return None

    if target.kind is SemanticTargetKind.MAP:
        result = NavigationPlan(
            tuple(actions),
            current,
            navigator._metrics(
                tuple(actions),
                navigation_goal.encounter_mode,
                navigation_goal.encounter_penalty,
            ),
        )
        _enforce_plan_limits(result, max_route_cost=max_route_cost, cost_ceiling=cost_ceiling)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_SEMANTIC_ROUTE: "
                f"status='composed' maps={getattr(route, 'maps', None)!r} "
                f"target_kind='MAP' actions={len(result.actions)} "
                f"cost={getattr(result.metrics, 'total_route_cost', None)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_SEMANTIC_ROUTE",
        )
        return result, route

    local_target = _semantic_interaction_local_goal(world, target)
    if local_target is None:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_SEMANTIC_ROUTE: "
                f"status='fallback' reason='destination_interaction_geometry_unresolved' "
                f"map={current[0]!r} interaction={target.interaction_id!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_SEMANTIC_ROUTE",
        )
        return None
    local_goal = replace(navigation_goal, target=local_target)
    try:
        final_plan = navigator.plan(
            current,
            local_goal,
            max_expansions=max_expansions,
            max_route_cost=(None if max_route_cost is None else max_route_cost - total_cost),
            cancel_check=cancel_check,
            caller=caller,
        )
    except NavigationSearchCancelled:
        raise
    except (NavigationError, NavigationSearchLimitExceeded, RuntimeError, TypeError, ValueError) as error:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_SEMANTIC_ROUTE: "
                f"status='final_interaction_rejected' error_type={type(error).__name__!r} error={str(error)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_SEMANTIC_ROUTE",
        )
        return None
    actions.extend(final_plan.actions)
    result = NavigationPlan(
        tuple(actions),
        final_plan.destination,
        navigator._metrics(tuple(actions), navigation_goal.encounter_mode, navigation_goal.encounter_penalty),
    )
    _enforce_plan_limits(result, max_route_cost=max_route_cost, cost_ceiling=cost_ceiling)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_SEMANTIC_ROUTE: "
            f"status='composed' maps={getattr(route, 'maps', None)!r} "
            f"interaction={target.interaction_id!r} actions={len(result.actions)} "
            f"cost={getattr(result.metrics, 'total_route_cost', None)!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_SEMANTIC_ROUTE",
    )
    return result, route


@traced("world_map_and_candidate_planning")
def plan_with_world_navigation(
    world: NavigationWorld,
    start: Location,
    goal: NavigationGoal | Goal,
    graph: WorldMapGraph | None = None,
    *,
    algorithm: str = "dijkstra",
    caller: str | None = None,
    max_expansions: int | None = None,
    max_route_cost: int | None = None,
    cost_ceiling: tuple[int, ...] | int | None = None,
    cancel_check=None,
) -> tuple[NavigationPlan, WorldRoute | None]:
    """Plan one local segment, appending one generic map transition if needed.

    The optional limits are intended for speculative route comparisons.  A
    bounded caller must be able to abandon an unreachable candidate without
    monopolizing the emulator loop. ``cost_ceiling`` is a lexicographic
    prefix of the search cost and is useful when an incumbent candidate has
    already been found; ``max_route_cost`` is an absolute safety budget.
    """
    navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
    diagnostic_print(
        lambda: (
            "NAVIGATION_REQUEST: "
            f"caller={caller or 'unspecified'!r} start={start!r} "
            f"goal={navigation_goal.target!r} "
            f"trainer_mode={getattr(navigation_goal.constraints.trainer_mode, 'name', None)!r} "
            f"encounter_mode={getattr(navigation_goal.encounter_mode, 'name', None)!r}"
        ),
        trace=True,
        prefix="NAVIGATION_REQUEST",
    )
    target_map = goal_target_map(world, goal)
    # A concrete observed boundary is already the runtime transition to use.
    # Plan only its local approach here; global graph search is for unresolved
    # semantic destinations.  This also makes dynamic occupancy fail fast
    # instead of exploring every map while an NPC temporarily blocks the
    # boundary.
    if isinstance(navigation_goal.target, ReachWarp) and target_map != start[0]:
        target = navigation_goal.target
        observed_warps = target.warps or ((target.warp,) if target.warp is not None else ())
        local_observed_warp = bool(observed_warps) and all(
            warp is not None
            and warp.entry[0] == start[0]
            and warp.destination is not None
            for warp in observed_warps
        )
        if local_observed_warp:
            # ``_observed_exit_goal`` has already selected one or more
            # executable ROM transitions from the current perception.
            # Searching the global world again for an exact door/arrow warp is
            # redundant and can lose the live transition identity when a
            # grouped goal contains equivalent exits.  Keep both single and
            # grouped observed exits in the local tactical planner; reserve
            # global search for unresolved destinations.
            local_plan = (
                plan_observed_warp_locally(
                    world,
                    start,
                    navigation_goal,
                    max_expansions=max_expansions,
                    max_route_cost=max_route_cost,
                    cost_ceiling=cost_ceiling,
                    cancel_check=cancel_check,
                    caller=caller,
                )
                if target.warp is not None
                else GoalAwareNavigator(world).plan(
                    start,
                    navigation_goal,
                    algorithm=algorithm,
                    caller=caller,
                    max_expansions=max_expansions,
                    max_route_cost=max_route_cost,
                    cost_ceiling=cost_ceiling,
                    cancel_check=cancel_check,
                )
            )
            return local_plan, None
    if target_map is None or target_map == start[0]:
        return (
            GoalAwareNavigator(world).plan(
                start,
                goal,
                algorithm=algorithm,
                caller=caller,
                max_expansions=max_expansions,
                max_route_cost=max_route_cost,
                cost_ceiling=cost_ceiling,
                cancel_check=cancel_check,
            ),
            None,
        )

    trace = getattr(context, "stutter_trace", None)
    trace_graph_start = trace.now() if trace is not None else 0
    graph = graph or get_world_map_graph()
    if trace is not None:
        trace.duration("world_map_graph_lookup_duration_ms", trace_graph_start)
    trace_route_start = trace.now() if trace is not None else 0
    try:
        if navigation_goal.preferred_map_route is not None:
            route = _route_for_map_sequence(
                graph,
                navigation_goal.preferred_map_route,
                start_map=start[0],
                target_map=target_map,
            )
        else:
            route = graph.route(start[0], target_map)
    except WorldNavigationError as error:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_WORLD_ROUTE_DEBUG: "
                f"start_map={start[0]!r} target_map={target_map!r} "
                f"edge_count={len(graph.edges)!r} "
                f"outgoing_start={graph.outgoing(start[0])!r} "
                f"error={str(error)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_WORLD_ROUTE_DEBUG",
        )
        raise NavigationError(str(error)) from error
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
            try:
                _enforce_plan_limits(
                    cached[0],
                    max_route_cost=max_route_cost,
                    cost_ceiling=cost_ceiling,
                )
            except NavigationSearchLimitExceeded:
                # A cached route may have been produced for a different
                # incumbent candidate. It is still useful as a cache entry,
                # but it cannot satisfy the tighter speculative bound.
                pass
            else:
                return cached
        profile_count("semantic_cross_map_plan_cache_misses")
    # Search the exact finite world graph. The old first-edge candidate code
    # remains below for compatibility with callers that explicitly require a
    # selected observed warp, but ordinary semantic cross-map goals use the
    # global state search.
    global_world = _global_navigation_world(
        world,
        graph,
        # Include the terminal map as well: a destination encounter area can
        # contain a trainer before the selected grass tile.
        enrich_maps=tuple(route.maps),
        allow_pokemon_center_destination=_goal_allows_pokemon_center_destination(navigation_goal),
    )
    observed_current_connections = tuple(
        transition
        for transition in (world.transitions or world.warps)
        if transition.entry[0] == start[0] and transition.kind == "map_connection"
    )
    if observed_current_connections:
        # The live overlay has already filtered map-boundary endpoints for
        # collision, objects, and the current map state. Static metadata can
        # contain every geometric boundary row, including endpoints that are
        # not executable from this observation. Keep those static fallbacks on
        # destination maps, but do not make them compete with the live source
        # map candidates.
        global_world = replace(
            global_world,
            transitions=tuple(
                transition
                for transition in global_world.transitions
                if transition.entry[0] != start[0]
                or transition.kind != "map_connection"
                or transition in observed_current_connections
            ),
        )
    if isinstance(navigation_goal.target, SemanticTarget):
        composed = _plan_semantic_cross_map_chain(
            global_world,
            start,
            navigation_goal,
            route,
            graph,
            max_expansions=max_expansions,
            max_route_cost=max_route_cost,
            cost_ceiling=cost_ceiling,
            cancel_check=cancel_check,
            caller=caller,
        )
        if composed is not None:
            composed_plan, composed_route = composed
            if cache_key is not None:
                _semantic_cross_map_plan_cache[cache_key] = (composed_plan, composed_route)
            return composed_plan, composed_route
        if navigation_goal.preferred_map_route is not None:
            raise NavigationError(
                f"preferred map route is currently unreachable: {navigation_goal.preferred_map_route!r}"
            )
    if isinstance(navigation_goal.target, ReachWarp):
        # Keep the complete NavigationGoal wrapper.  Unwrapping the target
        # here silently restored the default TrainerMode.IGNORE, which made
        # planner-composed recovery routes cross live trainer sight lines
        # despite being requested with TrainerMode.AVOID.
        global_goal = navigation_goal
        # The static graph's transition metadata may omit executable details
        # from the live catalog (for example the activation side of a Center
        # door). Ensure the selected transition replaces its graph-generated
        # equivalent so a specific ReachWarp retains the exact ROM contract.
        selected = navigation_goal.target.warp
        if selected is not None:
            transitions = list(global_world.transitions)
            transitions = [
                transition
                for transition in transitions
                if not transitions_match(transition, selected)
            ]
            transitions.append(selected)
            global_world = replace(global_world, transitions=tuple(transitions))
        if navigation_goal.target.warp is not None:
            chained = _plan_exact_cross_map_warp_chain(
                global_world,
                start,
                navigation_goal,
                route,
                max_expansions=max_expansions,
                max_route_cost=max_route_cost,
                cost_ceiling=cost_ceiling,
                cancel_check=cancel_check,
                caller=caller,
            )
            if chained is not None:
                return chained, route
    else:
        # An unresolved interaction is intentionally a map-boundary goal.
        # Runtime observation resolves its interaction after arrival.
        global_goal = replace(navigation_goal, target=normalized_target)
        if normalized_target.location is not None:
            global_goal = replace(navigation_goal, target=ReachLocation(normalized_target.location))
    # Keep the caller's algorithm authoritative. Production callers use the
    # default Dijkstra search so one lexicographic cost model applies to local
    # routes, shared objectives, and cross-map transition chains. Explicit A*
    # remains available to benchmarks and diagnostic callers.
    search_algorithm = algorithm
    try:
        plan = GoalAwareNavigator(global_world).plan(
            start,
            global_goal,
            algorithm=search_algorithm,
            caller=caller,
            max_expansions=max_expansions,
            max_route_cost=max_route_cost,
            cost_ceiling=cost_ceiling,
            cancel_check=cancel_check,
            # Semantic campaign routes expose the crossing input as the action
            # from the executable approach state.  Preserve the legacy boundary
            # source representation for exact ReachLocation callers.
            connection_source_is_approach=isinstance(navigation_goal.target, SemanticTarget),
        )
    except Exception as error:
        current_tiles = tuple(
            global_world.tiles.get(location) for location in global_world.tiles if location[0] == start[0]
        )
        current_transitions = tuple(
            transition for transition in global_world.transitions if transition.entry[0] == start[0]
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_WORLD_ROUTE_DEBUG: "
                f"phase=state_search_failed start={start!r} target={target_map!r} "
                f"route_maps={route.maps!r} enriched_maps={tuple(route.maps[:-1])!r} "
                f"loaded_maps={tuple(sorted(getattr(global_world.tiles, '_models', {}).keys(), key=repr))!r} "
                f"current_tile_count={len(current_tiles)!r} current_transition_count={len(current_transitions)!r} "
                f"current_transitions={current_transitions!r} error={str(error)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_WORLD_ROUTE_DEBUG",
        )
        if isinstance(error, NavigationSearchLimitExceeded):
            # Preserve the specific failure class through the diagnostic
            # context added here. Recovery uses this distinction to make one
            # bounded emergency search with a less rigid encounter policy;
            # converting it to NavigationError makes an exhausted search
            # indistinguishable from a proven unreachable route.
            raise NavigationSearchLimitExceeded(
                f"{error}; current tile count={len(current_tiles)}, "
                f"current transition count={len(current_transitions)}"
            ) from error
        if isinstance(error, NavigationError):
            raise NavigationError(
                f"{error}; current tile count={len(current_tiles)}, "
                f"current transition count={len(current_transitions)}"
            ) from error
        raise
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
                actions,
                local_plan.destination,
                navigator._metrics(
                    actions,
                    navigation_goal.encounter_mode,
                    navigation_goal.encounter_penalty,
                ),
            )
        return local_plan

    def activation_candidate_key(item):
        completed = completed_local_plan(item)
        metrics = completed.metrics
        if metrics is None:
            return (float("inf"),) * 6 + (item[3],)
        return navigation_candidate_key(metrics, navigation_goal.encounter_mode, item[3])

    shared_targets = tuple(
        (start[0], (choice[0], choice[1]))
        for _, _, choices_for_candidate in candidate_choices
        for choice in choices_for_candidate
    )
    shared_plans = navigator.plan_many_locations(
        start,
        tuple(dict.fromkeys(shared_targets)),
        encounter_mode=navigation_goal.encounter_mode,
        encounter_penalty=navigation_goal.encounter_penalty,
        algorithm=algorithm,
        max_expansions=max_expansions,
        max_route_cost=max_route_cost,
    )
    for candidate_id, coordinates, choices_for_candidate in candidate_choices:
        reachable_choices = []
        for choice_index, choice in enumerate(choices_for_candidate):
            local_plan = shared_plans.get((start[0], (choice[0], choice[1])))
            if local_plan is None:
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
                continue
            activation_direction = choice[2]
            candidate_item = (
                local_plan,
                coordinates,
                activation_direction,
                f"{candidate_id}:{choice_index}",
            )
            completed = completed_local_plan(candidate_item)
            metrics = completed.metrics
            candidate_records.append(
                {
                    "candidate_id": candidate_item[3],
                    "source_coordinates": coordinates,
                    "activation_coordinates": (choice[0], choice[1]),
                    "activation_direction": getattr(activation_direction, "name", None),
                    "status": "pathfound",
                    "reason": "reachable_activation_side",
                    "pathfinding_calls": 1,
                    "nodes_expanded": None,
                    "encounter_opportunities": metrics.encounter_opportunities if metrics else None,
                    "route_cost": metrics.total_route_cost if metrics else None,
                }
            )
            reachable_choices.append(candidate_item)
        if not reachable_choices:
            continue
        selected_item = min(reachable_choices, key=activation_candidate_key)
        candidate_plans.append(selected_item)
        selected_activation = selected_item[3]
        for candidate_record in candidate_records:
            if candidate_record["candidate_id"] == selected_activation:
                candidate_record["reason"] = "cheapest_reachable_activation_side"
        selected_metrics = completed_local_plan(selected_item).metrics
        if selected_metrics is not None:
            if best_encounters is None or selected_metrics.encounter_opportunities < best_encounters:
                best_encounters = selected_metrics.encounter_opportunities
                best_route_cost = selected_metrics.total_route_cost
            elif selected_metrics.encounter_opportunities == best_encounters:
                best_route_cost = min(best_route_cost, selected_metrics.total_route_cost)
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
            actions,
            local_plan.destination,
            GoalAwareNavigator(world)._metrics(
                actions,
                navigation_goal.encounter_mode,
                navigation_goal.encounter_penalty,
            ),
        ),
        route,
    )


def plan_observed_warp_locally(
    world: NavigationWorld,
    start: Location,
    goal: ReachWarp | NavigationGoal,
    *,
    max_expansions: int | None = None,
    max_route_cost: int | None = None,
    cost_ceiling: tuple[int, ...] | int | None = None,
    cancel_check=None,
    caller: str | None = None,
) -> NavigationPlan:
    """Plan to an exact observed warp without consulting the map graph."""
    navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
    goal = navigation_goal.target
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
        local_plan = GoalAwareNavigator(world).plan(
            start,
            replace(navigation_goal, target=ReachLocation(selected.entry)),
            max_expansions=max_expansions,
            max_route_cost=max_route_cost,
            cost_ceiling=cost_ceiling,
            cancel_check=cancel_check,
            caller=f"{caller}.observed_warp" if caller else "navigation.observed_warp",
        )
        result = NavigationPlan(
            local_plan.actions,
            selected.entry,
            GoalAwareNavigator(world)._metrics(
                local_plan.actions,
                navigation_goal.encounter_mode,
                navigation_goal.encounter_penalty,
            ),
        )
        _enforce_plan_limits(result, max_route_cost=max_route_cost, cost_ceiling=cost_ceiling)
        return result
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
            local_plan = GoalAwareNavigator(world).plan(
                start,
                replace(navigation_goal, target=ReachLocation(approach)),
                max_expansions=max_expansions,
                max_route_cost=max_route_cost,
                cost_ceiling=cost_ceiling,
                cancel_check=cancel_check,
                caller=f"{caller}.observed_warp" if caller else "navigation.observed_warp",
            )
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
        result = NavigationPlan(
            actions,
            selected.entry,
            GoalAwareNavigator(world)._metrics(
                actions,
                navigation_goal.encounter_mode,
                navigation_goal.encounter_penalty,
            ),
        )
        _enforce_plan_limits(result, max_route_cost=max_route_cost, cost_ceiling=cost_ceiling)
        return result
    local_plan = GoalAwareNavigator(world).plan(
        start,
        navigation_goal,
        max_expansions=max_expansions,
        max_route_cost=max_route_cost,
        cost_ceiling=cost_ceiling,
        cancel_check=cancel_check,
        caller=f"{caller}.observed_warp" if caller else "navigation.observed_warp",
    )
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
        result = NavigationPlan(
            local_plan.actions + (crossing,),
            selected.entry,
            GoalAwareNavigator(world)._metrics(
                local_plan.actions + (crossing,),
                navigation_goal.encounter_mode,
                navigation_goal.encounter_penalty,
            ),
        )
        _enforce_plan_limits(result, max_route_cost=max_route_cost, cost_ceiling=cost_ceiling)
        return result
    if selected.activation is WarpActivation.DIRECTIONAL_STEP:
        direction = (
            selected.activation_direction if selected.activation_direction is not None else selected.required_facing
        )
        if direction is None:
            raise NavigationError("directional observed warp has no activation direction")
        # GoalAwareNavigator.plan() already appends the executable WARP when
        # its ReachWarp terminal state is reached.  The observed-warp path
        # historically appended the directional activation a second time,
        # producing two identical WARP actions at arrow warps.  Keep the
        # planner-produced action when it is already the selected transition;
        # only append here for a local plan that stopped at the entry tile.
        already_appended = bool(
            local_plan.actions
            and local_plan.actions[-1].action_type is NavigationActionType.WARP
            and local_plan.actions[-1].source == selected.entry
            and local_plan.actions[-1].destination == selected.destination
        )
        activation_source = selected.entry
        activation = NavigationAction(
            NavigationActionType.WARP,
            direction,
            activation_source,
            selected.destination,
        )
        actions = local_plan.actions if already_appended else local_plan.actions + (activation,)
        result = NavigationPlan(
            actions,
            local_plan.destination,
            GoalAwareNavigator(world)._metrics(
                actions,
                navigation_goal.encounter_mode,
                navigation_goal.encounter_penalty,
            ),
        )
        _enforce_plan_limits(result, max_route_cost=max_route_cost, cost_ceiling=cost_ceiling)
        return result
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
                "STATIC_TARGET_UNAVAILABLE"
                if any(
                    binding.static_match
                    and not binding.static_available
                    and binding.static_availability is None
                    for binding in matching_bindings
                )
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
        # Cross-map semantic goals used to have a zero heuristic because a
        # warp can invalidate a local Manhattan estimate.  That is safe but
        # makes Dijkstra explore every equally encounter-free branch before
        # it reaches the requested map.  Cache the map-hop lower bounds used
        # by the conservative cross-map heuristic below.
        self._map_hops_to_target: dict[MapId, tuple[tuple[MapId, int], ...]] = {}

    @staticmethod
    def _yield_to_frame_loop(expanded: int) -> None:
        """Give the emulator thread a scheduling window during worker A*."""

        if background_work_suppressed() and expanded > 0 and expanded % 4 == 0:
            # Route searches are intentionally off the frame loop, but they
            # still share a Python process with mGBA. A short cooperative
            # yield prevents a CPU-bound speculative search from starving the
            # emulator core on low-core machines.
            sleep(0.001)

    @profiled("navigation_pathfinding", "pathfinding_calls")
    @traced("individual_pathfinding")
    def plan(
        self,
        start: Location,
        goal: NavigationGoal | Goal,
        *,
        algorithm: str = "dijkstra",
        caller: str | None = None,
        connection_source_is_approach: bool | None = None,
        max_expansions: int | None = None,
        max_route_cost: int | None = None,
        cost_ceiling: tuple[int, ...] | int | None = None,
        cancel_check=None,
    ) -> NavigationPlan:
        navigation_goal = goal if isinstance(goal, NavigationGoal) else NavigationGoal(goal)
        target = navigation_goal.target
        orientation_required = self.world.facing is not None and _goal_requires_orientation(target)
        pathfinding_call_id = next(_pathfinding_call_ids)
        print_navigation_pathfinding(
            phase="start",
            call_id=pathfinding_call_id,
            algorithm=algorithm,
            start=start,
            goal=repr(navigation_goal.target),
            goal_type=type(navigation_goal.target).__name__,
            orientation_required=orientation_required,
            caller=caller,
        )
        _validate_search_limit(max_expansions, "max_expansions")
        _validate_search_limit(max_route_cost, "max_route_cost")
        if isinstance(cost_ceiling, bool) or (
            cost_ceiling is not None and not isinstance(cost_ceiling, (int, tuple))
        ):
            raise ValueError("cost_ceiling must be an integer, tuple, or None")
        if isinstance(cost_ceiling, tuple) and not all(
            isinstance(item, int) and not isinstance(item, bool) for item in cost_ceiling
        ):
            raise ValueError("cost_ceiling tuple must contain integers")
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
        # Keep ordinary location routing compact. A directional input while
        # facing elsewhere is still handled by the execution layer as a
        # turn-only input followed by a retry; it does not need to become a
        # graph state for a location goal. Interaction and transition goals
        # retain facing because their activation preconditions depend on it.
        if connection_source_is_approach is None:
            connection_source_is_approach = isinstance(target, SemanticTarget)
        emit_orientation_actions = _goal_requires_orientation(target)
        executable_orientation = orientation_required
        if algorithm not in ("dijkstra", "astar"):
            raise ValueError(f"Unknown navigation search algorithm: {algorithm!r}")
        profiling = profiling_enabled()
        # Keep timing the search even when the optional profiler is disabled.
        # The Rich boundary records are used to explain visible frame-loop
        # stalls, so their elapsed time must not depend on profiler settings.
        search_started = perf_counter_ns()
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
        limit_pruned = False

        def priority(cost, state):
            if algorithm == "dijkstra":
                return cost
            heuristic = self._heuristic(state[0], navigation_goal.target, navigation_goal.encounter_mode)
            return tuple(cost[index] + heuristic[index] for index in range(6))

        def record(found, metrics=None, error=None):
            duration_ms = round((perf_counter_ns() - search_started) / 1_000_000, 3)
            print_navigation_pathfinding(
                phase="complete",
                call_id=pathfinding_call_id,
                algorithm=algorithm,
                start=start,
                goal=repr(navigation_goal.target),
                goal_type=type(navigation_goal.target).__name__,
                orientation_required=orientation_required,
                nodes_expanded=expanded,
                nodes_generated=generated,
                duration_ms=duration_ms,
                path_found=found,
                action_count=getattr(metrics, "movement_actions", None) if metrics is not None else None,
                route_cost=getattr(metrics, "total_route_cost", None) if metrics is not None else None,
                error=(f"{type(error).__name__}: {error}" if error is not None else None),
                caller=caller,
            )
            if not found and isinstance(navigation_goal.target, ReachWarp):
                _log_failed_reach_warp_diagnostic(self.world, start, navigation_goal.target)
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
                    duration_ms=duration_ms,
                    path_found=found,
                    final_encounter_opportunity_count=(metrics.encounter_opportunities if metrics else None),
                    final_route_cost=(metrics.total_route_cost if metrics else None),
                    candidate_searches_for_goal=1,
                )

        if algorithm == "astar":
            queue[0] = (priority(initial_cost, initial_state), queue[0][1], initial_state, initial_cost)
        generated = 1
        def check_cancel():
            if cancel_check is None:
                return
            try:
                cancel_check()
            except Exception as error:
                # A superseded or timed-out worker used to leave only a
                # ``phase=start`` panel behind. Emit the matching terminal
                # record before propagating the cancellation so every
                # pathfinding call remains explainable in the Rich console.
                record(False, error=error)
                raise

        while queue:
            check_cancel()
            if max_expansions is not None and expanded >= max_expansions:
                record(False)
                raise NavigationSearchLimitExceeded(
                    f"navigation search exceeded max_expansions {max_expansions}"
                )
            _, _, (current, facing), current_cost = heapq.heappop(queue)
            state = (current, facing)
            if current_cost != costs[state]:
                continue
            if max_route_cost is not None and current_cost[1] > max_route_cost:
                limit_pruned = True
                continue
            if isinstance(cost_ceiling, int) and current_cost[1] > cost_ceiling:
                limit_pruned = True
                continue
            if isinstance(cost_ceiling, tuple) and cost_ceiling and current_cost[: len(cost_ceiling)] > cost_ceiling:
                limit_pruned = True
                continue
            expanded += 1
            self._yield_to_frame_loop(expanded)
            frontier_max = max(frontier_max, len(queue))
            diagnostic_maps.add(current[0])
            diagnostic_goal_checks += 1
            # A target tile can itself be part of an undefeated trainer's
            # sight line (most notably a map-connection approach tile).  Do
            # not let goal completion bypass the AVOID boundary: reaching
            # that tile is already exposure even if no successor needs to
            # enter it.
            if self._satisfies(current, facing, navigation_goal.target) and not self._is_undesirable(
                current, navigation_goal.constraints
            ):
                diagnostic_goal_matches += 1
                actions = self._unroll(
                    came_from,
                    state,
                    emit_orientation_actions,
                    connection_source_is_approach=connection_source_is_approach,
                )
                if isinstance(navigation_goal.target, ReachWarp):
                    actions, current = self._append_warp_activation(actions, current, facing, navigation_goal.target)
                metrics = self._metrics(
                    actions,
                    navigation_goal.encounter_mode,
                    navigation_goal.encounter_penalty,
                )
                if max_route_cost is not None and metrics.total_route_cost > max_route_cost:
                    limit_pruned = True
                    continue
                if isinstance(cost_ceiling, int) and metrics.total_route_cost > cost_ceiling:
                    limit_pruned = True
                    continue
                if isinstance(cost_ceiling, tuple) and cost_ceiling and _cost_prefix(metrics)[: len(cost_ceiling)] > cost_ceiling:
                    limit_pruned = True
                    continue
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
        if limit_pruned:
            raise NavigationSearchLimitExceeded("navigation search reached its configured cost bound")
        # Trainer avoidance is a safety boundary for campaign navigation. A
        # caller must explicitly opt in before accepting a forced battle;
        # otherwise the failure belongs to readiness, which can identify and
        # prepare for the specific ROM-backed trainer checkpoint.
        trainer_mode = getattr(navigation_goal.constraints, "trainer_mode", None)
        if (
            trainer_mode is not None
            and trainer_mode.name == "AVOID"
            and getattr(navigation_goal.constraints, "allow_forced_trainer_exposure", False)
        ):
            fallback_constraints = replace(navigation_goal.constraints, trainer_mode=type(trainer_mode).IGNORE)
            fallback_goal = replace(navigation_goal, constraints=fallback_constraints)
            fallback = self.plan(
                start,
                fallback_goal,
                algorithm=algorithm,
                connection_source_is_approach=connection_source_is_approach,
                max_expansions=max_expansions,
                max_route_cost=max_route_cost,
                cost_ceiling=cost_ceiling,
                cancel_check=cancel_check,
            )
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
        constraints: GoalConstraints | None = None,
        algorithm: str = "dijkstra",
        target_groups: tuple[tuple[Location, ...], ...] | None = None,
        max_expansions: int | None = None,
        max_route_cost: int | None = None,
        cancel_check=None,
    ) -> dict[Location, NavigationPlan]:
        """Resolve several ReachLocation goals from one shared state search.

        Cross-map planning has several terminal activation positions, but all
        of them have the same predecessor graph and the same ``(Location,
        None)`` state.  Facing is intentionally excluded because these are
        ordinary ReachLocation goals; candidate-specific activation turns
        are added by the caller afterwards.
        This deliberately searches the reachable graph once
        and reconstructs the best state for each requested coordinate.  The
        search is exhaustive: unlike arbitrary goal pruning, this is safe for
        the lexicographic objective and for candidates that have no route.

        This helper models the existing ReachLocation contract.  In
        particular, it does not emit stationary turns; the cross-map caller
        adds candidate-specific activation turns exactly as it did before.
        """
        if algorithm not in ("dijkstra", "astar"):
            raise ValueError(f"Unknown navigation search algorithm: {algorithm!r}")
        _validate_search_limit(max_expansions, "max_expansions")
        _validate_search_limit(max_route_cost, "max_route_cost")
        requested = frozenset(targets)
        pathfinding_call_id = next(_pathfinding_call_ids)
        print_navigation_pathfinding(
            phase="start",
            call_id=pathfinding_call_id,
            algorithm=algorithm,
            start=start,
            goal=f"shared ReachLocation targets={len(requested)}",
            goal_type="SharedReachLocation",
            orientation_required=False,
        )
        if not requested:
            print_navigation_pathfinding(
                phase="complete",
                call_id=pathfinding_call_id,
                algorithm=algorithm,
                start=start,
                goal="shared ReachLocation targets=0",
                goal_type="SharedReachLocation",
                orientation_required=False,
                nodes_expanded=0,
                nodes_generated=0,
                path_found=False,
            )
            return {}
        profiling = profiling_enabled()
        search_started = perf_counter_ns()
        sequence = count()
        # Shared searches resolve ordinary ReachLocation targets.  Keep one
        # graph state per tile and let execution perform any required first
        # turn.  This is the high-volume cross-map path where four facing
        # variants per tile are particularly expensive.
        initial_state = (start, None)
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
        target_constraints = constraints or GoalConstraints()
        while queue:
            if cancel_check is not None:
                cancel_check()
            if max_expansions is not None and expanded >= max_expansions:
                raise NavigationSearchLimitExceeded(
                    f"navigation search exceeded max_expansions {max_expansions}"
                )
            _, _, state, current_cost = heapq.heappop(queue)
            if current_cost != costs.get(state):
                continue
            if max_route_cost is not None and current_cost[1] > max_route_cost:
                continue
            expanded += 1
            self._yield_to_frame_loop(expanded)
            frontier_max = max(frontier_max, len(queue))
            current, facing = state
            # Shared ReachLocation searches are used for cross-map approach
            # candidates.  Apply the same safety check before accepting an
            # approach tile, otherwise a trainer sight-line endpoint can be
            # returned as a supposedly safe candidate.
            if current in requested and not self._is_undesirable(
                current,
                target_constraints,
            ):
                # ReachLocation deliberately completes on the first state
                # popped by the ordered frontier.  This is also the old
                # single-goal planner's completion rule; candidate-specific
                # activation turns are appended by the caller afterwards.
                if current not in results:
                    actions = self._unroll(came_from, state, False)
                    results[current] = NavigationPlan(
                        actions,
                        current,
                        self._metrics(actions, encounter_mode, encounter_penalty),
                    )
                    if all(any(location in results for location in group) for group in groups):
                        break
            if self._is_undesirable(current, NavigationGoal(ReachLocation(current), constraints=target_constraints).constraints):
                continue
            for direction, neighbour, is_warp in self.world.neighbors(current):
                if is_warp:
                    step_cost = 1
                    encounter = 0
                else:
                    tile = self.world.tiles[neighbour]
                    step_cost = tile.traversal_cost - int(tile.has_encounters)
                    encounter = int(tile.has_encounters)
                # ReachLocation searches deliberately ignore turn costs and
                # facing.  The resulting MOVE action is still executable:
                # the controller observes and retries a turn-only first
                # input before advancing.
                turn_cost = 0
                turn_on_encounter = 0
                turn_on_non_encounter = 0
                opportunity = encounter
                exposure_cost, opportunity_cost = encounter_cost_components(encounter_mode, encounter_penalty)
                new_cost = (
                    current_cost[0] + exposure_cost * opportunity,
                    current_cost[1] + step_cost + turn_cost + opportunity_cost * opportunity,
                    current_cost[2] + int(not is_warp),
                    current_cost[3] + encounter,
                    current_cost[4] + turn_on_encounter,
                    current_cost[5] + turn_on_non_encounter,
                )
                next_state = (neighbour, None)
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
        print_navigation_pathfinding(
            phase="complete",
            call_id=pathfinding_call_id,
            algorithm=algorithm,
            start=start,
            goal=f"shared ReachLocation targets={len(requested)}",
            goal_type="SharedReachLocation",
            orientation_required=False,
            nodes_expanded=expanded,
            nodes_generated=generated,
            duration_ms=round((perf_counter_ns() - search_started) / 1_000_000, 3),
            path_found=bool(results),
        )
        return results

    def _map_hop_distances_to(self, target_map: MapId) -> dict[MapId, int]:
        """Return a reverse map-hop lower bound for a semantic target.

        Every executable transition costs at least one route-cost unit and
        ordinary movement/turns add non-negative cost.  Counting transitions
        therefore remains a lower bound even when a transition teleports
        across the map graph.  The table is built from the already-materialized
        transition overlay and never loads additional ROM map data.
        """

        cached = self._map_hops_to_target.get(target_map)
        if cached is not None:
            return dict(cached)

        incoming: dict[MapId, set[MapId]] = {}
        for transition in self.world.transitions or self.world.warps:
            destination = getattr(transition, "destination", None)
            entry = getattr(transition, "entry", None)
            if destination is None or entry is None:
                continue
            incoming.setdefault(destination[0], set()).add(entry[0])

        distances = {target_map: 0}
        queue = [target_map]
        for current in queue:
            for previous in incoming.get(current, ()):
                if previous in distances:
                    continue
                distances[previous] = distances[current] + 1
                queue.append(previous)
        self._map_hops_to_target[target_map] = tuple(distances.items())
        return distances

    @staticmethod
    def _transition_source_locations(transition: WorldTransition) -> tuple[tuple[int, int], ...]:
        """Return all local source coordinates an observed transition accepts."""

        sources = getattr(transition, "activation_locations", ())
        if sources:
            return tuple(source[1] for source in sources)
        if transition.kind == "map_connection":
            approach = transition_approach_position(transition)
            if approach is not None:
                return (approach[1],)
        return (transition.entry[1],)

    def _cross_map_heuristic(
        self,
        location: Location,
        target_map: MapId,
        target: Goal,
    ) -> int:
        """Return a conservative movement-cost bound to a cross-map goal.

        The bound is the Manhattan distance to an executable transition plus
        one for that transition and the minimum number of remaining map
        transitions.  Manhattan is only used within the current map, so a
        warp cannot make it overestimate the cost.  If the transition overlay
        is incomplete, return zero rather than guessing.
        """

        if location[0] == target_map:
            return 0
        map_hops = self._map_hop_distances_to(target_map)
        if not map_hops:
            return 0

        allowed_transitions = self.world.transitions or self.world.warps
        if isinstance(target, ReachWarp):
            if target.warps:
                allowed_transitions = tuple(
                    transition
                    for transition in allowed_transitions
                    if any(transitions_match(transition, allowed) for allowed in target.warps)
                )
            elif target.warp is not None:
                allowed_transitions = tuple(
                    transition for transition in allowed_transitions if transitions_match(transition, target.warp)
                )

        best: int | None = None
        for transition in allowed_transitions:
            destination = getattr(transition, "destination", None)
            entry = getattr(transition, "entry", None)
            if destination is None or entry is None or entry[0] != location[0]:
                continue
            remaining_hops = map_hops.get(destination[0])
            if remaining_hops is None:
                continue
            for source in self._transition_source_locations(transition):
                bound = abs(location[1][0] - source[0]) + abs(location[1][1] - source[1])
                bound += 1 + remaining_hops
                if best is None or bound < best:
                    best = bound
        return best or 0

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
        target_map = None
        if isinstance(target, SemanticTarget):
            target_map = target.target_map
        elif isinstance(target, ReachWarp):
            target_map = target.destination_map
        if target_map is not None and location[0] != target_map:
            return (0, self._cross_map_heuristic(location, target_map, target), 0, 0, 0, 0)

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
                    positions_for_engagement = (
                        getattr(trigger, "engagement_locations", ())
                    )
                    positions.extend(
                        positions_for_engagement
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
                and location in getattr(trigger, "engagement_locations", ())
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
                and _reach_warp_transition_matches(warp, target)
                and any(
                    transitions_match(warp, allowed)
                    or (
                        target.destination is None
                        and target.warps
                        and transition_source_matches(warp, allowed)
                    )
                    for allowed in allowed_transitions
                )
                for warp in transitions
            )
            return matched
        raise TypeError(f"Unsupported goal type: {type(target).__name__}")

    def _append_warp_activation(self, actions, current, facing, target):
        """Append the executable transition after reaching its source state."""
        transitions = self.world.transitions or self.world.warps
        allowed_transitions = target.warps or ((target.warp,) if target.warp is not None else transitions)
        for transition in transitions:
            if not any(
                transitions_match(transition, allowed)
                or (
                    target.destination is None
                    and target.warps
                    and transition_source_matches(transition, allowed)
                )
                for allowed in allowed_transitions
            ):
                continue
            approach = transition_approach_position(transition)
            source = (
                approach
                if transition.kind == "map_connection"
                else (
                    current
                    if current in transition.activation_locations
                    else next(iter(transition.activation_locations), transition.entry)
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
            if direction is None and transition.activation_locations:
                direction = _direction_between(source[1], transition.entry[1])
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
            if any(
                (
                    not getattr(constraints, "avoid_trainer_ids", ())
                    or getattr(trigger, "affordance_id", None) in getattr(constraints, "avoid_trainer_ids", ())
                )
                and (
                    getattr(trigger, "hazard_kind", None) == "trainer"
                    or str(getattr(trigger, "affordance_id", "")).startswith(("trainer:", "rom_trainer:"))
                )
                and (
                    location in trigger.hazard_locations
                    # An adjacent interaction position is only a movement
                    # hazard while the trainer's sight-line geometry is
                    # unresolved.  Once a positive sight-line envelope is
                    # available, walking past an NPC without pressing A is
                    # safe; treating every adjacent tile as hazardous can
                    # close the only trainer-free corridor on a map. Keep
                    # the empty-envelope case fail-closed so unknown
                    # trainer geometry never becomes implicit permission.
                    or (
                        getattr(trigger, "hazard_kind", None) == "trainer"
                        and not getattr(trigger, "hazard_locations", ())
                        and location in trigger.activation_locations
                    )
                )
                for trigger in self.world.triggers
            ):
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
        self,
        actions: tuple[NavigationAction, ...],
        encounter_mode: EncounterMode = EncounterMode.NORMAL,
        encounter_penalty: int = 8,
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
        _, opportunity_cost = encounter_cost_components(encounter_mode, encounter_penalty)
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
