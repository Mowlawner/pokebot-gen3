"""Static map-level routing built from the repository's ROM map metadata."""

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from modules.context import context
from modules.map import get_map_data
from modules.map_data import MapFRLG, MapRSE
from modules.stutter_trace import traced

MapId = tuple[int, int]
Coordinate = tuple[int, int]


@dataclass(frozen=True)
class WorldEdge:
    source_map: MapId
    destination_map: MapId
    kind: str
    source_coordinates: tuple[Coordinate, ...]
    destination_coordinates: tuple[Coordinate, ...]
    warp_index: int | None = None
    required_facing: object | None = None
    connection_direction: str | None = None


@dataclass(frozen=True)
class WorldRoute:
    maps: tuple[MapId, ...]
    edges: tuple[WorldEdge, ...]


class WorldNavigationError(RuntimeError):
    pass


def _connection_coordinates(
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
    direction: str,
    offset: int,
) -> tuple[tuple[Coordinate, Coordinate], ...]:
    """Return boundary coordinate pairs described by a MapConnection.

    MapConnection does not contain one singular warp coordinate.  Its
    direction/offset describe the overlapping boundary, so retain every
    geometrically valid pair and let local BFS choose a usable one.
    """
    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    pairs: list[tuple[Coordinate, Coordinate]] = []
    if direction in ("North", "South"):
        first = max(0, offset)
        last = min(source_width, offset + destination_width)
        for source_x in range(first, last):
            destination_x = source_x - offset
            pairs.append(
                (
                    (source_x, 0 if direction == "North" else source_height - 1),
                    (destination_x, destination_height - 1 if direction == "North" else 0),
                )
            )
    elif direction in ("East", "West"):
        first = max(0, offset)
        last = min(source_height, offset + destination_height)
        for source_y in range(first, last):
            destination_y = source_y - offset
            pairs.append(
                (
                    (source_width - 1 if direction == "East" else 0, source_y),
                    (0 if direction == "East" else destination_width - 1, destination_y),
                )
            )
    return tuple(pairs)


class WorldMapGraph:
    def __init__(self, edges: tuple[WorldEdge, ...] = ()):
        self.edges = edges
        self._outgoing: dict[MapId, tuple[WorldEdge, ...]] = {}
        for edge in edges:
            self._outgoing.setdefault(edge.source_map, ())
            self._outgoing[edge.source_map] += (edge,)

    @classmethod
    def from_map_data(cls, maps: Mapping[MapId, object]) -> "WorldMapGraph":
        edges: list[WorldEdge] = []
        for source_map, map_data in maps.items():
            for warp_index, warp in enumerate(map_data.warps):
                destination = warp.destination_location
                destination_map = destination.map_group_and_number
                edges.append(
                    WorldEdge(
                        source_map=source_map,
                        destination_map=destination_map,
                        kind="warp",
                        source_coordinates=(warp.local_coordinates,),
                        destination_coordinates=(destination.local_position,),
                        warp_index=warp_index,
                    )
                )
            for connection in map_data.connections:
                destination_map = (connection.destination_map_group, connection.destination_map_number)
                destination_data = maps.get(destination_map)
                if destination_data is None:
                    continue
                pairs = _connection_coordinates(
                    map_data.map_size,
                    destination_data.map_size,
                    connection.direction,
                    connection.offset,
                )
                if not pairs:
                    continue
                edges.append(
                    WorldEdge(
                        source_map=source_map,
                        destination_map=destination_map,
                        kind="connection",
                        source_coordinates=tuple(source for source, _ in pairs),
                        destination_coordinates=tuple(destination for _, destination in pairs),
                        connection_direction=connection.direction,
                    )
                )
        return cls(tuple(edges))

    def outgoing(self, map_id: MapId) -> tuple[WorldEdge, ...]:
        return self._outgoing.get(map_id, ())

    @traced("world_map_route_calculation")
    def route(self, source_map: MapId, target_map: MapId) -> WorldRoute:
        if source_map == target_map:
            return WorldRoute((source_map,), ())
        queue = deque([source_map])
        came_from: dict[MapId, tuple[MapId, WorldEdge] | None] = {source_map: None}
        while queue:
            current = queue.popleft()
            for edge in self.outgoing(current):
                if edge.destination_map in came_from:
                    continue
                came_from[edge.destination_map] = (current, edge)
                if edge.destination_map == target_map:
                    return self._unroll(came_from, target_map)
                queue.append(edge.destination_map)
        raise WorldNavigationError(f"No map route from {source_map!r} to {target_map!r}")

    @staticmethod
    def _unroll(came_from: Mapping[MapId, tuple[MapId, WorldEdge] | None], destination: MapId) -> WorldRoute:
        maps = [destination]
        edges: list[WorldEdge] = []
        current = destination
        while came_from[current] is not None:
            previous, edge = came_from[current]
            maps.append(previous)
            edges.append(edge)
            current = previous
        maps.reverse()
        edges.reverse()
        return WorldRoute(tuple(maps), tuple(edges))


_world_graph_cache: dict[str, WorldMapGraph] = {}


@traced("WorldMapGraph_lookup_or_construction")
def get_world_map_graph() -> WorldMapGraph:
    """Build/cache the static graph from the active ROM's map metadata."""
    game_key = context.rom.id
    if game_key not in _world_graph_cache:
        map_enum = MapRSE if context.rom.is_rse else MapFRLG
        maps = {
            map_address.value: get_map_data(map_address, (0, 0))
            for map_address in map_enum
            if not context.rom.is_rs or map_address.exists_on_rs
        }
        _world_graph_cache[game_key] = WorldMapGraph.from_map_data(maps)
    return _world_graph_cache[game_key]
