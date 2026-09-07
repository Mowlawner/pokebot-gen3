"""Static map-level routing built from the repository's ROM map metadata."""

from collections import deque
import heapq
from dataclasses import dataclass
from functools import lru_cache
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
    # A conservative map-level estimate.  Callers that have better movement
    # information (for example a test graph or a cached map planner) can
    # supply it without changing the graph topology.
    # One transition is deliberately more expensive than a handful of local
    # inputs.  This keeps a direct goal advance ahead of a cheap exit, while
    # callers can provide measured estimates when they have them.
    estimated_cost: int = 10


@dataclass(frozen=True)
class WorldRoute:
    maps: tuple[MapId, ...]
    edges: tuple[WorldEdge, ...]
    estimated_cost: int = 0


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
        self._incoming: dict[MapId, tuple[WorldEdge, ...]] = {}
        for edge in edges:
            self._outgoing.setdefault(edge.source_map, ())
            self._outgoing[edge.source_map] += (edge,)
            self._incoming.setdefault(edge.destination_map, ())
            self._incoming[edge.destination_map] += (edge,)

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

    @lru_cache(maxsize=8192)
    @traced("world_map_costs_calculation")
    def map_costs(self, source_map: MapId) -> tuple[tuple[MapId, int], ...]:
        """Return static shortest-path costs from one map to every reachable map.

        Campaign planning often needs to compare one current map with many
        encounter maps. Calling :meth:`route` once per destination reruns the
        same Dijkstra search each time. This immutable distance table keeps
        the ROM/static graph as the source of routing avenues while avoiding
        that repeated work.
        """

        return self._costs(source_map, reverse=False)

    @lru_cache(maxsize=8192)
    @traced("world_map_reverse_costs_calculation")
    def map_costs_to(self, target_map: MapId) -> tuple[tuple[MapId, int], ...]:
        """Return static shortest-path costs from every map to ``target_map``."""

        return self._costs(target_map, reverse=True)

    @staticmethod
    def _manhattan(left: Coordinate, right: Coordinate) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    @lru_cache(maxsize=16384)
    @traced("world_location_route_estimate")
    def estimate_location_cost(
        self,
        source: tuple[MapId, Coordinate],
        target: tuple[MapId, Coordinate],
    ) -> int | None:
        """Estimate tile work between two locations using the static graph.

        This is deliberately a cheap ranking heuristic, not a replacement for
        live collision-aware navigation.  It accounts for the walk from the
        current position to the chosen map exit, the walk from each map
        entrance to its next exit, and the final walk to the destination.
        Consequently a geographically nearby center behind a long dungeon is
        not ranked as though the dungeon were one free warp.
        """
        source_map, source_coordinate = source
        target_map, target_coordinate = target
        if source_map == target_map:
            return self._manhattan(source_coordinate, target_coordinate)
        try:
            route = self.route(source_map, target_map)
        except WorldNavigationError:
            return None
        frontier: dict[Coordinate, int] = {source_coordinate: 0}
        for edge in route.edges:
            pairs = tuple(zip(edge.source_coordinates, edge.destination_coordinates))
            if not pairs:
                return None
            next_frontier: dict[Coordinate, int] = {}
            for current, current_cost in frontier.items():
                for entry, destination in pairs:
                    candidate = current_cost + self._manhattan(current, entry) + 1
                    previous = next_frontier.get(destination)
                    if previous is None or candidate < previous:
                        next_frontier[destination] = candidate
            if not next_frontier:
                return None
            frontier = next_frontier
        return min(
            (cost + self._manhattan(coordinate, target_coordinate) for coordinate, cost in frontier.items()),
            default=None,
        )

    def _costs(self, source_map: MapId, *, reverse: bool) -> tuple[tuple[MapId, int], ...]:
        """Run one Dijkstra search over outgoing or reversed static edges."""

        queue: list[tuple[int, int, MapId]] = []
        order = 0
        distances: dict[MapId, int] = {source_map: 0}
        heapq.heappush(queue, (0, order, source_map))
        while queue:
            distance, _, current = heapq.heappop(queue)
            if distance != distances[current]:
                continue
            edges = self._incoming.get(current, ()) if reverse else self.outgoing(current)
            for edge in edges:
                edge_cost = max(1, edge.estimated_cost)
                next_map = edge.source_map if reverse else edge.destination_map
                next_distance = distance + edge_cost
                if next_distance >= distances.get(next_map, float("inf")):
                    continue
                distances[next_map] = next_distance
                order += 1
                heapq.heappush(queue, (next_distance, order, next_map))
        return tuple(sorted(distances.items()))

    @lru_cache(maxsize=8192)
    @traced("world_map_route_calculation")
    def route(self, source_map: MapId, target_map: MapId, *, prefer_interior: bool = False) -> WorldRoute:
        """Return the cheapest route, optionally preferring interior warps.

        The ordinary route remains a weighted shortest path.  Interaction
        targets may opt into a large boundary penalty so an all-warp route
        through a building (for example, first floor to second floor) beats
        an outdoor detour that only happens to be globally reachable.
        """
        if source_map == target_map:
            return WorldRoute((source_map,), ())
        # Dijkstra retains the old insertion-order behavior for equal-cost
        # edges, while allowing a longer warp sequence to win when its
        # estimated movement cost is lower.
        queue: list[tuple[int, int, MapId]] = []
        order = 0
        heapq.heappush(queue, (0, order, source_map))
        distances: dict[MapId, int] = {source_map: 0}
        came_from: dict[MapId, tuple[MapId, WorldEdge] | None] = {source_map: None}
        while queue:
            distance, _, current = heapq.heappop(queue)
            if distance != distances[current]:
                continue
            if current == target_map:
                return self._unroll(came_from, target_map, distance)
            for edge in self.outgoing(current):
                edge_cost = max(1, edge.estimated_cost)
                if prefer_interior and edge.kind == "connection":
                    edge_cost += 1_000_000
                next_distance = distance + edge_cost
                if next_distance >= distances.get(edge.destination_map, float("inf")):
                    continue
                distances[edge.destination_map] = next_distance
                came_from[edge.destination_map] = (current, edge)
                order += 1
                heapq.heappush(queue, (next_distance, order, edge.destination_map))
        raise WorldNavigationError(f"No map route from {source_map!r} to {target_map!r}")

    @staticmethod
    def _unroll(
        came_from: Mapping[MapId, tuple[MapId, WorldEdge] | None],
        destination: MapId,
        estimated_cost: int = 0,
    ) -> WorldRoute:
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
        return WorldRoute(tuple(maps), tuple(edges), estimated_cost)


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
