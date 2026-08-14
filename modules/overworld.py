"""Explicit overworld perception models and the live emulator adapter."""

from dataclasses import dataclass
from modules.map import get_map_data, get_map_objects
from modules.map_path import Direction, _get_map_metadata
from modules.player import get_player_avatar, player_avatar_is_controllable

MapId = tuple[int, int]
Coordinate = tuple[int, int]
Location = tuple[MapId, Coordinate]


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


@dataclass(frozen=True)
class ObjectObservation:
    local_id: int
    location: Location
    kind: str
    script: str = ""


@dataclass(frozen=True)
class TileObservation:
    location: Location
    blocked: bool
    walkable_neighbors: frozenset[Direction]
    warp: WarpObservation | None = None
    trigger_ids: frozenset[str] = frozenset()


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

    def tile_at(self, coordinates: Coordinate) -> TileObservation | None:
        return next((tile for tile in self.tiles if tile.location[1] == coordinates), None)


def _direction(value: str) -> Direction | None:
    try:
        return Direction.from_string(value)
    except (RuntimeError, AttributeError):
        return None


def perceive_overworld() -> OverworldObservation:
    """Read the current map and avatar without changing emulator state.

    Collision and directional access come from the existing PathMap metadata;
    active object positions are reported separately so a planner can choose to
    treat them as dynamic obstacles.
    """

    avatar = get_player_avatar()
    map_id = avatar.map_group_and_number
    player_coordinates = avatar.local_coordinates
    path_map = _get_map_metadata(map_id)
    path_tiles = path_map.tiles
    map_data = get_map_data(map_id, (0, 0))
    by_coordinate = {tile.local_coordinates: tile for tile in path_tiles}

    warps: list[WarpObservation] = []
    for warp in map_data.warps:
        destination = ((warp.destination_location.map_group, warp.destination_location.map_number),
                       warp.destination_location.local_position)
        warps.append(WarpObservation((map_id, warp.local_coordinates), destination))
    warps_by_entry = {warp.entry[1]: warp for warp in warps}

    triggers: list[TriggerObservation] = []
    for index, event in enumerate(map_data.coord_events):
        if event.type == "weather":
            continue
        location = (map_id, event.local_coordinates)
        triggers.append(TriggerObservation(
            trigger_id=f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}",
            locations=frozenset({location}),
            activation_locations=frozenset({location}),
            kind=event.type,
        ))
    trigger_ids_by_location: dict[Coordinate, set[str]] = {}
    for trigger in triggers:
        for location in trigger.locations:
            trigger_ids_by_location.setdefault(location[1], set()).add(trigger.trigger_id)

    objects = tuple(
        ObjectObservation(
            local_id=object_event.local_id,
            location=(object_event.map_group_and_number, object_event.current_coords),
            kind=object_event.kind,
            script=object_event.script_symbol,
        )
        for object_event in get_map_objects()
        if object_event.map_group_and_number == map_id and "isPlayer" not in object_event.flags
    )
    object_locations = {object_observation.location[1] for object_observation in objects}

    # Object templates/scripts are the reliable data available for NPC
    # interactions.  The four adjacent tiles are candidate activation tiles;
    # the planner can refine these when a game-specific interaction reports
    # more restrictive geometry.
    map_width, map_height = map_data.map_size
    for object_observation in objects:
        x, y = object_observation.location[1]
        activation_locations = frozenset(
            (map_id, (candidate_x, candidate_y))
            for candidate_x, candidate_y in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
            if 0 <= candidate_x < map_width and 0 <= candidate_y < map_height
        )
        triggers.append(TriggerObservation(
            trigger_id=f"object:{object_observation.local_id}:{object_observation.script}",
            locations=frozenset({object_observation.location}),
            activation_locations=activation_locations,
            kind="object_interaction",
        ))

    trigger_ids_by_location = {}
    for trigger in triggers:
        for location in trigger.locations:
            trigger_ids_by_location.setdefault(location[1], set()).add(trigger.trigger_id)

    tiles: list[TileObservation] = []
    for coordinate, tile in by_coordinate.items():
        walkable = frozenset(direction for direction in Direction if tile.accessible_from_direction[direction.value])
        tiles.append(TileObservation(
            location=(map_id, coordinate),
            blocked=coordinate in object_locations or (not walkable and tile.warps_to is None),
            walkable_neighbors=walkable,
            warp=warps_by_entry.get(coordinate),
            trigger_ids=frozenset(trigger_ids_by_location.get(coordinate, set())),
        ))

    return OverworldObservation(
        map_id=map_id,
        player_coordinates=player_coordinates,
        facing=_direction(avatar.facing_direction),
        controllable=player_avatar_is_controllable(),
        tiles=tuple(tiles),
        warps=tuple(warps),
        objects=objects,
        triggers=tuple(triggers),
    )
