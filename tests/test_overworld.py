import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.map import MapMetadata, ObjectEvent
from modules.map_path import Direction
from modules.overworld import perceive_overworld


def _object_event() -> ObjectEvent:
    data = bytearray(0x24)
    data[0] = 0x01  # active
    data[8] = 7  # local ID
    data[9] = 2  # map number
    data[10] = 1  # map group
    data[0x10:0x12] = (8).to_bytes(2, "little")  # x = 1
    data[0x12:0x14] = (9).to_bytes(2, "little")  # y = 2
    data[0x18:0x1A] = (1).to_bytes(2, "little")  # Down
    return ObjectEvent(bytes(data))


class TestOverworldPerception(unittest.TestCase):
    def test_reuses_static_tiles_when_only_avatar_state_changes(self):
        map_id = (91, 7)
        map_data = MapMetadata(
            map_id, b"header", (2).to_bytes(4, "little") + (1).to_bytes(4, "little"),
            b"events", (), (), (), (), (),
        )
        path_tiles = [
            SimpleNamespace(local_coordinates=(x, 0), accessible_from_direction=[True] * 4,
                             warps_to=None, traversal_cost=2 if x == 1 else 1)
            for x in range(2)
        ]
        avatar = SimpleNamespace(
            map_group_and_number=map_id, local_coordinates=(0, 0), facing_direction="Down",
        )
        with (
            patch("modules.overworld.get_player_avatar", return_value=avatar),
            patch("modules.overworld._get_map_metadata", return_value=SimpleNamespace(tiles=path_tiles)),
            patch("modules.overworld.get_map_metadata", return_value=map_data),
            patch("modules.overworld.get_map_objects", return_value=[]),
            patch("modules.overworld.player_avatar_is_controllable", return_value=True),
        ):
            first = perceive_overworld()
            avatar.local_coordinates = (1, 0)
            second = perceive_overworld()

        self.assertIs(first.tiles, second.tiles)
        self.assertIs(first.tiles[0], second.tiles[0])
        self.assertEqual(first.tiles[1].traversal_cost, 2)
        self.assertIs(first.tiles[1], second.tiles[1])
        self.assertEqual(first.player_coordinates, (0, 0))
        self.assertEqual(second.player_coordinates, (1, 0))

    def test_processes_real_object_event_without_kind_or_script_attributes(self):
        object_event = _object_event()
        map_id = (1, 2)
        map_data = MapMetadata(
            map_id, b"header", (3).to_bytes(4, "little") + (4).to_bytes(4, "little"),
            b"events", (), (), (), (), (),
        )
        template_map = SimpleNamespace(
            objects=[SimpleNamespace(local_id=7, script_symbol="Route101_EventScript_NPC")]
        )
        path_tile = SimpleNamespace(
            local_coordinates=(1, 2),
            accessible_from_direction=[True, True, True, True],
            warps_to=None,
        )
        avatar = SimpleNamespace(
            map_group_and_number=map_id,
            local_coordinates=(0, 0),
            facing_direction="Down",
        )

        with (
            patch("modules.overworld.get_player_avatar", return_value=avatar),
            patch("modules.overworld._get_map_metadata", return_value=SimpleNamespace(tiles=[path_tile])),
            patch("modules.overworld.get_map_metadata", return_value=map_data),
            patch("modules.overworld.get_map_objects", return_value=[object_event]),
            patch("modules.overworld.player_avatar_is_controllable", return_value=True),
            patch("modules.map.get_map_metadata", return_value=SimpleNamespace(
                object_template=lambda local_id: template_map.objects[0]
                if local_id == template_map.objects[0].local_id else None,
            )),
        ):
            observation = perceive_overworld()

        self.assertEqual(len(observation.objects), 1)
        self.assertEqual(observation.objects[0].local_id, 7)
        self.assertEqual(observation.objects[0].script, "Route101_EventScript_NPC")
        self.assertIsNone(observation.objects[0].kind)
        self.assertEqual(observation.facing, Direction.South)


if __name__ == "__main__":
    unittest.main()
