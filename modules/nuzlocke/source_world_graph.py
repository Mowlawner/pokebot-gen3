"""Map-level topology extracted directly from a pret-style source tree."""

from __future__ import annotations

import json
from pathlib import Path

from modules.world_navigation import WorldEdge, WorldMapGraph


def source_world_graph(source_root: str | Path) -> WorldMapGraph:
    """Build a map-level graph from ``data/maps/*/map.json``.

    The graph intentionally contains only static warp and connection topology.
    Collision, dynamic object visibility, and script guards belong to the
    route-impact layer and must be supplied as explicit edge effects.
    """

    root = Path(source_root).expanduser().resolve()
    maps: dict[str, dict] = {}
    for map_path in sorted(root.glob("data/maps/*/map.json"), key=lambda path: path.as_posix()):
        try:
            data = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        map_id = str(data.get("id", map_path.parent.name))
        maps[map_id] = data

    edges: list[WorldEdge] = []
    for source_map in sorted(maps):
        data = maps[source_map]
        for warp_index, warp in enumerate(data.get("warp_events") or ()):
            destination = str(warp.get("dest_map", ""))
            if destination not in maps:
                continue
            edges.append(
                WorldEdge(
                    source_map=source_map,
                    destination_map=destination,
                    kind="warp",
                    source_coordinates=(),
                    destination_coordinates=(),
                    warp_index=warp_index,
                )
            )
        for connection in data.get("connections") or ():
            destination = str(connection.get("map", ""))
            if destination not in maps:
                continue
            edges.append(
                WorldEdge(
                    source_map=source_map,
                    destination_map=destination,
                    kind="connection",
                    source_coordinates=(),
                    destination_coordinates=(),
                    connection_direction=str(connection.get("direction", "")),
                )
            )
    return WorldMapGraph(tuple(edges))


__all__ = ["source_world_graph"]
