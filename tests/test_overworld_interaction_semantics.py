import unittest

from modules.map_path import Direction
from modules.overworld import (
    EMERALD_MB_COUNTER,
    ObjectObservation,
    TileObservation,
    resolve_object_activation_positions,
)


class TestEmeraldObjectInteractionSemantics(unittest.TestCase):
    def test_emerald_counter_behavior_matches_rom_table(self):
        self.assertEqual(EMERALD_MB_COUNTER, 0x80)

    def _resolve(self, object_position, counter_position=None, behavior=None, elevation=3):
        map_id = (2, 2)
        obj = ObjectObservation(1, (map_id, object_position), elevation=elevation)
        tiles = []
        if counter_position is not None:
            tiles.append(
                TileObservation(
                    (map_id, counter_position),
                    blocked=True,
                    walkable_neighbors=frozenset(),
                    metatile_behavior=behavior,
                )
            )
        return resolve_object_activation_positions(obj, (obj,), tiles, (20, 20))

    def test_ordinary_object_has_adjacent_activation_positions(self):
        result = self._resolve((7, 2))
        self.assertEqual(
            {position for position, _ in result},
            {
                ((2, 2), (7, 3)),
                ((2, 2), (8, 2)),
                ((2, 2), (7, 1)),
                ((2, 2), (6, 2)),
            },
        )

    def test_counter_extends_activation_one_tile_in_each_direction(self):
        cases = (
            (Direction.North, (7, 3), (7, 4)),
            (Direction.East, (6, 2), (5, 2)),
            (Direction.South, (7, 1), (7, 0)),
            (Direction.West, (8, 2), (9, 2)),
        )
        for facing, counter, activation in cases:
            with self.subTest(facing=facing):
                result = self._resolve((7, 2), counter, EMERALD_MB_COUNTER)
                self.assertIn(((2, 2), activation), {position for position, _ in result})
                self.assertEqual(dict(result)[((2, 2), activation)], facing)

    def test_blocked_non_counter_does_not_extend_activation(self):
        result = self._resolve((7, 2), (7, 3), behavior=None)
        self.assertIn(((2, 2), (7, 3)), {position for position, _ in result})
        self.assertNotIn(((2, 2), (7, 4)), {position for position, _ in result})

    def test_oldale_nurse_counter_geometry(self):
        result = self._resolve((7, 2), (7, 3), EMERALD_MB_COUNTER)
        self.assertEqual(dict(result)[((2, 2), (7, 4))], Direction.North)

    def test_activation_tile_elevation_does_not_reject_matching_object(self):
        map_id = (2, 2)
        target = ObjectObservation(1, (map_id, (7, 2)), elevation=3)
        counter = TileObservation(
            (map_id, (7, 3)), True, frozenset(), metatile_behavior=EMERALD_MB_COUNTER, elevation=0
        )
        player_tile = TileObservation((map_id, (7, 4)), True, frozenset(), elevation=0)
        result = resolve_object_activation_positions(target, (target,), (counter, player_tile), (20, 20))
        self.assertIn(((2, 2), (7, 4)), {position for position, _ in result})

    def test_counter_object_at_other_elevation_does_not_block_resolution(self):
        map_id = (2, 2)
        target = ObjectObservation(1, (map_id, (7, 2)), elevation=3)
        other_level = ObjectObservation(2, (map_id, (7, 3)), elevation=2)
        counter = TileObservation((map_id, (7, 3)), True, frozenset(), metatile_behavior=EMERALD_MB_COUNTER)
        result = resolve_object_activation_positions(target, (target, other_level), (counter,), (20, 20))
        self.assertIn(((2, 2), (7, 4)), {position for position, _ in result})

    def test_counter_is_not_extended_when_counter_contains_object(self):
        map_id = (2, 2)
        target = ObjectObservation(1, (map_id, (7, 2)), elevation=3)
        counter_object = ObjectObservation(2, (map_id, (7, 3)), elevation=3)
        counter = TileObservation((map_id, (7, 3)), True, frozenset(), metatile_behavior=EMERALD_MB_COUNTER)
        result = resolve_object_activation_positions(target, (target, counter_object), (counter,), (20, 20))
        self.assertIn(((2, 2), (7, 3)), {position for position, _ in result})
        self.assertNotIn(((2, 2), (7, 4)), {position for position, _ in result})


if __name__ == "__main__":
    unittest.main()
