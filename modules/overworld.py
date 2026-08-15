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

MapId = tuple[int, int]
Coordinate = tuple[int, int]
Location = tuple[MapId, Coordinate]


class MovementState(Enum):
    TURNING = auto()
    MOVING = auto()
    STANDING = auto()


@dataclass(frozen=True)
class WarpObservation:
    entry: Location
    destination: Location
    required_facing: Direction | None = None


@dataclass(frozen=True)
class TriggerObservation:
    trigger_id: str
    locations: frozenset[Location]
    activation_locations: frozenset[Location] = frozenset()
    kind: str = "script"
    target_map: MapId | None = None
    navigation_locations: frozenset[Location] = frozenset()


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


_static_map_observations: dict[MapId, _StaticMapObservation] = {}


def prewarm_static_map_observation(map_id: MapId) -> tuple[TileObservation, ...]:
    """Populate and return static perception data for one known map.

    This deliberately does not read the player, object events, or interaction
    state.  It is safe to call before a warp because it only materializes data
    derived from the destination map's ROM metadata.
    """
    path_map = _get_map_metadata(map_id)
    path_tiles = path_map.tiles
    map_data = get_map_metadata(map_id)
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
            if path_tile is not None and path_tile.warps_to is not None:
                required_facing = path_tile.warps_to[2]
            warps.append(
                WarpObservation((map_id, warp.local_coordinates), destination, required_facing=required_facing)
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
                )
            )
        tiles = tuple(static_tiles)
        static = _StaticMapObservation(path_tiles, map_data, by_coordinate, tuple(warps), tiles)
        _static_map_observations[map_id] = static
    return static.tiles


def perceive_overworld() -> OverworldObservation:
    """Read the current map and avatar without changing emulator state.

    Collision and directional access come from the existing PathMap metadata;
    active object positions are reported separately so a planner can choose to
    treat them as dynamic obstacles.
    """

    profiling = enabled()
    profile_start = now()
    stage_start = now()
    avatar = get_player_avatar()
    map_id = avatar.map_group_and_number
    player_coordinates = avatar.local_coordinates
    timing("perception_player_avatar", stage_start)
    count("player_avatar_observations")
    stage_start = now()
    _get_map_metadata(map_id).tiles
    map_data = get_map_metadata(map_id)
    map_width, map_height = map_data.map_size
    timing("perception_map_data_lookup", stage_start)
    count("map_data_lookups")

    tiles = prewarm_static_map_observation(map_id)
    static = _static_map_observations[map_id]
    warps = static.warps

    triggers: list[TriggerObservation] = []
    for index, event in enumerate(map_data.coord_events):
        if event.type == "weather":
            continue
        location = (map_id, event.local_coordinates)
        triggers.append(
            TriggerObservation(
                trigger_id=f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}",
                locations=frozenset({location}),
                activation_locations=frozenset({location}),
                kind=event.type,
            )
        )
    runtime_start = now()
    objects = tuple(
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
            )
        )

    timing("perception_active_runtime_objects", runtime_start)
    count("runtime_object_scans")
    static_start = now()
    count("static_template_lookups")
    binding_start = now()
    bindings = tuple(
        resolve_trigger_binding(
            binding,
            objects,
            (map_width, map_height),
            static_objects=getattr(map_data, "objects", ()) if map_id == binding.map_id else (),
        )
        for binding in TRIGGER_BINDINGS
    )
    timing("perception_static_object_template_lookup", static_start)
    timing("trigger_binding_resolution", binding_start)
    for resolution in bindings:
        binding = resolution.binding
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
            )
        )

    # Tile interaction events (signs, scripted tiles, hidden items, and
    # secret-base entrances) are MapBgEvents, not ObjectEvents.  Preserve
    # their real kind/script data as interaction triggers.
    for index, event in enumerate(map_data.bg_events):
        location = (map_id, event.local_coordinates)
        activation_locations = frozenset()
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
        triggers.append(
            TriggerObservation(
                trigger_id=f"bg:{index}:{event.kind}:{script}",
                locations=frozenset({location}),
                activation_locations=activation_locations,
                kind=f"bg_{event.kind.lower().replace(' ', '_')}",
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
            f"movement={movement_state.name!r}"
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
