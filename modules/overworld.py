"""Explicit overworld perception models and the live emulator adapter."""

from dataclasses import dataclass, replace
from enum import Enum, auto
from modules.context import context
from modules.map import get_map_metadata, get_map_objects
from modules.map_path import Direction, _get_map_metadata
from modules.player import get_player_avatar, player_avatar_is_controllable
from modules.trigger_bindings import BindingResolution, TRIGGER_BINDINGS, resolve_trigger_binding
from modules.console import profile_print
from modules.profiler import count, enabled, format_snapshot, now, timing
from modules.stutter_trace import traced

MapId = tuple[int, int]
Coordinate = tuple[int, int]
Location = tuple[MapId, Coordinate]


class MovementState(Enum):
    TURNING = auto()
    MOVING = auto()
    STANDING = auto()


class WarpActivation(Enum):
    STEP_ON = auto()
    DIRECTIONAL_STEP = auto()


@dataclass(frozen=True)
class WarpObservation:
    entry: Location
    destination: Location
    required_facing: Direction | None = None
    activation: WarpActivation = WarpActivation.STEP_ON
    # A warp entry is not necessarily the tile from which the activating
    # input is issued.  ROM terrain such as escalators has one valid source
    # tile, even though the entry tile is geometrically reachable from more
    # than one side.
    activation_locations: frozenset[Location] = frozenset()
    activation_direction: Direction | None = None


@dataclass(frozen=True)
class TriggerObservation:
    trigger_id: str
    locations: frozenset[Location]
    activation_locations: frozenset[Location] = frozenset()
    kind: str = "script"
    target_map: MapId | None = None
    navigation_locations: frozenset[Location] = frozenset()
    # Per activation-position orientation requirements.  This is a mapping
    # rather than one direction because an object can be approached from any
    # of its four adjacent tiles, with a different required facing at each.
    activation_requirements: tuple[tuple[Location, Direction], ...] = ()


@dataclass(frozen=True)
class ObjectObservation:
    local_id: int
    location: Location
    # ObjectEvent has no category/kind field.  Keep this nullable until a
    # domain-level category can be derived without guessing from graphics IDs.
    kind: str | None = None
    script: str = ""


@dataclass(frozen=True)
class TileObservation:
    location: Location
    blocked: bool
    walkable_neighbors: frozenset[Direction]
    warp: WarpObservation | None = None
    trigger_ids: frozenset[str] = frozenset()
    # Static map data. Dynamic object blocking is represented separately on
    # OverworldObservation and must not invalidate this value.
    traversal_cost: int = 1
    # ROM-derived encounter terrain property. This is deliberately separate
    # from traversal_cost: encounter generation is an action property.
    has_encounters: bool = False


@dataclass(frozen=True)
class OverworldObservation:
    map_id: MapId
    player_coordinates: Coordinate
    facing: Direction | None
    controllable: bool
    tiles: tuple[TileObservation, ...]
    warps: tuple[WarpObservation, ...]
    objects: tuple[ObjectObservation, ...]
    triggers: tuple[TriggerObservation, ...]
    bindings: tuple[BindingResolution, ...] = ()
    movement_state: MovementState | None = None
    dynamic_blocked_coordinates: frozenset[Coordinate] = frozenset()

    def tile_at(self, coordinates: Coordinate) -> TileObservation | None:
        tile = next((tile for tile in self.tiles if tile.location[1] == coordinates), None)
        if tile is not None and coordinates in self.dynamic_blocked_coordinates and not tile.blocked:
            return replace(tile, blocked=True)
        return tile


def _direction(value: str) -> Direction | None:
    try:
        return Direction.from_string(value)
    except (RuntimeError, AttributeError):
        return None


@dataclass(frozen=True)
class _StaticMapObservation:
    path_tiles: object
    map_data: object
    by_coordinate: dict[Coordinate, object]
    warps: tuple[WarpObservation, ...]
    tiles: tuple[TileObservation, ...]
    triggers: tuple[TriggerObservation, ...]


_static_map_observations: dict[MapId, _StaticMapObservation] = {}


@traced("static_tile_preparation")
def prewarm_static_map_observation(map_id: MapId) -> tuple[TileObservation, ...]:
    """Populate and return static perception data for one known map.

    This deliberately does not read the player, object events, or interaction
    state.  It is safe to call before a warp because it only materializes data
    derived from the destination map's ROM metadata.
    """
    path_map = _get_map_metadata(map_id)
    path_tiles = path_map.tiles
    map_data = get_map_metadata(map_id)
    map_width, map_height = map_data.map_size
    static = _static_map_observations.get(map_id)
    if static is None or static.path_tiles is not path_tiles or static.map_data is not map_data:
        by_coordinate = {tile.local_coordinates: tile for tile in path_tiles}
        warps: list[WarpObservation] = []
        for warp in map_data.warps:
            destination = (
                (warp.destination_location.map_group, warp.destination_location.map_number),
                warp.destination_location.local_position,
            )
            path_tile = by_coordinate.get(warp.local_coordinates)
            required_facing = None
            activation = WarpActivation.STEP_ON
            activation_locations: frozenset[Location] = frozenset()
            activation_direction = None
            if path_tile is not None and path_tile.warps_to is not None:
                required_facing = path_tile.warps_to[2]
                if getattr(path_tile, "tile_type", "").endswith("Arrow Warp"):
                    activation = WarpActivation.DIRECTIONAL_STEP
            tile_type = getattr(path_tile, "tile_type", "") if path_tile is not None else ""
            x, y = warp.local_coordinates
            if tile_type in ("Escalator Up", "Escalator Down"):
                # Up/Down describes the destination transition, not the
                # side from which the field interaction is started.  The
                # Gen III Pokémon Center escalator rail is activated from
                # its right-hand source tile with a LEFT input.  Keep this
                # in activation_locations so the navigator can represent
                # source-state semantics independently of the warp tile.
                activation_direction = Direction.West
                required_facing = Direction.West
                source = (x + 1, y)
                if source in by_coordinate:
                    activation_locations = frozenset({(map_id, source)})
            warps.append(
                WarpObservation(
                    (map_id, warp.local_coordinates),
                    destination,
                    required_facing=required_facing,
                    activation=activation,
                    activation_locations=activation_locations,
                    activation_direction=activation_direction,
                )
            )
        warps_by_entry = {warp.entry[1]: warp for warp in warps}
        static_trigger_ids_by_location: dict[Coordinate, set[str]] = {}
        for index, event in enumerate(map_data.coord_events):
            if event.type != "weather":
                static_trigger_ids_by_location.setdefault(event.local_coordinates, set()).add(
                    f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}"
                )
        for index, event in enumerate(map_data.bg_events):
            script = event.script_symbol if event.kind == "Script" else ""
            static_trigger_ids_by_location.setdefault(event.local_coordinates, set()).add(
                f"bg:{index}:{event.kind}:{script}"
            )
        static_tiles: list[TileObservation] = []
        for coordinate, tile in by_coordinate.items():
            walkable = frozenset(
                direction for direction in Direction if tile.accessible_from_direction[direction.value]
            )
            static_tiles.append(
                TileObservation(
                    location=(map_id, coordinate),
                    blocked=not walkable and tile.warps_to is None,
                    walkable_neighbors=walkable,
                    warp=warps_by_entry.get(coordinate),
                    trigger_ids=frozenset(static_trigger_ids_by_location.get(coordinate, set())),
                    traversal_cost=getattr(tile, "traversal_cost", 1),
                    has_encounters=getattr(tile, "has_encounters", False),
                )
            )
        static_triggers: list[TriggerObservation] = []
        for index, event in enumerate(map_data.coord_events):
            if event.type != "weather":
                location = (map_id, event.local_coordinates)
                static_triggers.append(
                    TriggerObservation(
                        trigger_id=f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}",
                        locations=frozenset({location}),
                        activation_locations=frozenset({location}),
                        kind=event.type,
                    )
                )
        for index, event in enumerate(map_data.bg_events):
            location = (map_id, event.local_coordinates)
            activation_locations = frozenset()
            direction = None
            if event.player_facing_direction == "Any":
                activation_locations = frozenset(
                    (map_id, (candidate_x, candidate_y))
                    for candidate_x, candidate_y in (
                        (event.local_coordinates[0], event.local_coordinates[1] - 1),
                        (event.local_coordinates[0] + 1, event.local_coordinates[1]),
                        (event.local_coordinates[0], event.local_coordinates[1] + 1),
                        (event.local_coordinates[0] - 1, event.local_coordinates[1]),
                    )
                    if 0 <= candidate_x < map_width and 0 <= candidate_y < map_height
                )
            else:
                direction = _direction(event.player_facing_direction)
                if direction is not None:
                    offsets = {
                        Direction.North: (0, 1),
                        Direction.East: (-1, 0),
                        Direction.South: (0, -1),
                        Direction.West: (1, 0),
                    }
                    offset_x, offset_y = offsets[direction]
                    activation = (
                        event.local_coordinates[0] + offset_x,
                        event.local_coordinates[1] + offset_y,
                    )
                    if 0 <= activation[0] < map_width and 0 <= activation[1] < map_height:
                        activation_locations = frozenset({(map_id, activation)})
            script = event.script_symbol if event.kind == "Script" else ""
            static_triggers.append(
                TriggerObservation(
                    trigger_id=f"bg:{index}:{event.kind}:{script}",
                    locations=frozenset({location}),
                    activation_locations=activation_locations,
                    kind=f"bg_{event.kind.lower().replace(' ', '_')}",
                    activation_requirements=(
                        tuple(((map_id, activation), direction) for activation in activation_locations)
                        if event.player_facing_direction != "Any" and direction is not None
                        else ()
                    ),
                )
            )
        tiles = tuple(static_tiles)
        static = _StaticMapObservation(path_tiles, map_data, by_coordinate, tuple(warps), tiles, tuple(static_triggers))
        _static_map_observations[map_id] = static
    return static.tiles


@traced("overworld_perception")
def perceive_overworld() -> OverworldObservation:
    """Read the current map and avatar without changing emulator state.

    Collision and directional access come from the existing PathMap metadata;
    active object positions are reported separately so a planner can choose to
    treat them as dynamic obstacles.
    """

    profiling = enabled()
    trace = getattr(context, "stutter_trace", None)
    profile_start = now()
    stage_start = now()
    avatar = get_player_avatar()
    map_id = avatar.map_group_and_number
    player_coordinates = avatar.local_coordinates
    timing("perception_player_avatar", stage_start)
    count("player_avatar_observations")
    stage_start = now()
    path_metadata_start = now()
    trace_path_metadata_start = trace.now() if trace is not None else 0
    if trace is not None:
        trace.call("map_metadata_path_lookup", _get_map_metadata, map_id).tiles
    else:
        _get_map_metadata(map_id).tiles
    if trace is not None:
        trace.duration("overworld_map_metadata_lookup_duration_ms", trace_path_metadata_start)
    timing("perception_path_map_lookup", path_metadata_start)
    map_data_start = now()
    map_data = (
        trace.call("map_metadata_rom_lookup", get_map_metadata, map_id)
        if trace is not None
        else get_map_metadata(map_id)
    )
    map_width, map_height = map_data.map_size
    timing("perception_rom_map_lookup", map_data_start)
    timing("perception_map_data_lookup", stage_start)
    count("map_data_lookups")

    static_start = now()
    trace_static_start = trace.now() if trace is not None else 0
    tiles = prewarm_static_map_observation(map_id)
    if trace is not None:
        trace.duration("overworld_static_tile_preparation_duration_ms", trace_static_start)
    timing("perception_static_map_observation", static_start)
    static = _static_map_observations[map_id]
    warps = static.warps

    triggers: list[TriggerObservation] = list(static.triggers)
    runtime_start = now()
    trace_runtime_start = trace.now() if trace is not None else 0

    def scan_runtime_objects():
        return tuple(
            ObjectObservation(
                local_id=object_event.local_id,
                location=(object_event.map_group_and_number, object_event.current_coords),
                # ObjectEvent itself does not expose either of these attributes.
                # Its map template owns the interaction script; template.kind is
                # only "normal"/"clone", not an object category, so do not expose
                # it as one here.
                script=object_event.object_event_template.script_symbol,
            )
            for object_event in get_map_objects()
            if object_event.map_group_and_number == map_id and "isPlayer" not in object_event.flags
        )

    objects = trace.call("runtime_object_scan", scan_runtime_objects) if trace is not None else scan_runtime_objects()
    object_locations = {object_observation.location[1] for object_observation in objects}

    # Object templates/scripts are the reliable data available for NPC
    # interactions.  The four adjacent tiles are candidate activation tiles;
    # the planner can refine these when a game-specific interaction reports
    # more restrictive geometry.
    for object_observation in objects:
        x, y = object_observation.location[1]
        activation_locations = frozenset(
            (map_id, (candidate_x, candidate_y))
            for candidate_x, candidate_y in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
            if 0 <= candidate_x < map_width and 0 <= candidate_y < map_height
        )
        triggers.append(
            TriggerObservation(
                trigger_id=f"object:{object_observation.local_id}:{object_observation.script}",
                locations=frozenset({object_observation.location}),
                activation_locations=activation_locations,
                kind="object_interaction",
                activation_requirements=tuple(
                    (
                        (map_id, (candidate_x, candidate_y)),
                        {
                            (x, y - 1): Direction.South,
                            (x + 1, y): Direction.West,
                            (x, y + 1): Direction.North,
                            (x - 1, y): Direction.East,
                        }[(candidate_x, candidate_y)],
                    )
                    for candidate_x, candidate_y in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
                    if 0 <= candidate_x < map_width and 0 <= candidate_y < map_height
                ),
            )
        )

    timing("perception_active_runtime_objects", runtime_start)
    if trace is not None:
        trace.duration("overworld_runtime_object_scan_duration_ms", trace_runtime_start)
    count("runtime_object_scans")
    static_start = now()
    count("static_template_lookups")
    binding_start = now()
    trace_binding_start = trace.now() if trace is not None else 0

    def resolve_bindings():
        return tuple(
            resolve_trigger_binding(
                binding,
                objects,
                (map_width, map_height),
                static_objects=getattr(map_data, "objects", ()) if map_id == binding.map_id else (),
            )
            for binding in TRIGGER_BINDINGS
        )

    bindings = (
        trace.call("trigger_discovery_and_binding", resolve_bindings) if trace is not None else resolve_bindings()
    )
    timing("perception_static_object_template_lookup", static_start)
    timing("trigger_binding_resolution", binding_start)
    if trace is not None:
        trace.duration("overworld_trigger_binding_duration_ms", trace_binding_start)
    for resolution in bindings:
        binding = resolution.binding
        activation_requirements = tuple(
            (
                (map_id, (candidate_x, candidate_y)),
                {
                    (object_x, object_y - 1): Direction.South,
                    (object_x + 1, object_y): Direction.West,
                    (object_x, object_y + 1): Direction.North,
                    (object_x - 1, object_y): Direction.East,
                }[(candidate_x, candidate_y)],
            )
            for object_x, object_y in resolution.runtime_locations
            for candidate_x, candidate_y in (
                (object_x, object_y - 1),
                (object_x + 1, object_y),
                (object_x, object_y + 1),
                (object_x - 1, object_y),
            )
            if (candidate_x, candidate_y) in resolution.interaction_positions
        )
        triggers.append(
            TriggerObservation(
                trigger_id=binding.trigger_id,
                locations=frozenset((map_id, coordinates) for coordinates in resolution.runtime_locations),
                activation_locations=frozenset(
                    (map_id, coordinates) for coordinates in resolution.interaction_positions
                ),
                kind=f"semantic_{binding.event_type}",
                target_map=binding.map_id,
                navigation_locations=frozenset(
                    {resolution.static_location} if resolution.static_location is not None else set()
                ),
                activation_requirements=activation_requirements,
            )
        )

    # Static topology is reused. Object positions remain a per-observation
    # overlay; they must never be folded into the cached tile objects.
    dynamic_blocked_coordinates = frozenset(object_locations)

    running_state = getattr(avatar, "running_state", None)
    tile_transition_state = getattr(avatar, "tile_transition_state", None)
    if getattr(running_state, "name", None) == "TURN_DIRECTION":
        movement_state = MovementState.TURNING
    elif (
        getattr(running_state, "name", None) == "MOVING"
        or getattr(tile_transition_state, "name", "NOT_MOVING") != "NOT_MOVING"
    ):
        movement_state = MovementState.MOVING
    else:
        movement_state = MovementState.STANDING

    if profiling:
        timing("perception_total", profile_start)
        profile_print(
            lambda: f"OVERWORLD_PROFILE: {format_snapshot()} "
            f"map={map_id!r} position={player_coordinates!r} "
            f"movement={movement_state.name!r}",
            every=60,
        )

    return OverworldObservation(
        map_id=map_id,
        player_coordinates=player_coordinates,
        facing=_direction(avatar.facing_direction),
        controllable=player_avatar_is_controllable(),
        tiles=static.tiles,
        warps=tuple(warps),
        objects=objects,
        triggers=tuple(triggers),
        bindings=bindings,
        movement_state=movement_state,
        dynamic_blocked_coordinates=dynamic_blocked_coordinates,
    )
