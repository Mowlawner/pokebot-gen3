from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.source_world_graph import source_world_graph


class SourceWorldGraphTests(unittest.TestCase):
    def test_builds_static_warp_and_connection_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, data in {
                "Start": {
                    "id": "MAP_START",
                    "connections": [{"map": "MAP_MIDDLE", "direction": "East"}],
                    "warp_events": [{"dest_map": "MAP_TARGET"}],
                },
                "Middle": {"id": "MAP_MIDDLE", "connections": [], "warp_events": []},
                "Target": {"id": "MAP_TARGET", "connections": [], "warp_events": []},
            }.items():
                directory = root / "data" / "maps" / name
                directory.mkdir(parents=True)
                (directory / "map.json").write_text(json.dumps(data), encoding="utf-8")
            graph = source_world_graph(root)

        route = graph.route("MAP_START", "MAP_TARGET")
        self.assertEqual(route.maps, ("MAP_START", "MAP_TARGET"))
        self.assertEqual(route.edges[0].kind, "warp")
        self.assertEqual({edge.kind for edge in graph.outgoing("MAP_START")}, {"warp", "connection"})

    def test_ignores_warps_to_unlisted_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "data" / "maps" / "Start"
            directory.mkdir(parents=True)
            (directory / "map.json").write_text(
                json.dumps({"id": "MAP_START", "warp_events": [{"dest_map": "MAP_MISSING"}]}),
                encoding="utf-8",
            )
            graph = source_world_graph(root)
        self.assertEqual(graph.outgoing("MAP_START"), ())


if __name__ == "__main__":
    unittest.main()
