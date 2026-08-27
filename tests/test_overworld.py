import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from modules.map import MapMetadata, ObjectEvent
from modules.map_path import Direction
from modules.overworld import (
    ObjectObservation,
    TileObservation,
    trainer_hazard_locations,
    TriggerObservation,
    evaluate_trigger_condition,
    prewarm_static_map_observation,
    perceive_overworld,
    OverworldObservationResult,
    OverworldObservationStatus,
)


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
    def test_trainer_hazards_follow_facing_and_range(self):
        map_id = (1, 2)
        tiles = tuple(TileObservation((map_id, (x, 0)), False, frozenset(Direction), elevation=0) for x in range(5))
        trainer = ObjectObservation(
            1, (map_id, (1, 0)), facing="East", trainer_type="Normal", trainer_range=3, trainer_defeated=False
        )
        self.assertEqual(
            trainer_hazard_locations(trainer, tiles),
            frozenset({(map_id, (2, 0)), (map_id, (3, 0)), (map_id, (4, 0))}),
        )

    def test_defeated_trainer_has_no_hazards_and_objects_block_sight(self):
        map_id = (1, 2)
        tiles = tuple(TileObservation((map_id, (x, 0)), False, frozenset(Direction)) for x in range(5))
        trainer = ObjectObservation(
            1, (map_id, (1, 0)), facing="East", trainer_type="Normal", trainer_range=3, trainer_defeated=False
        )
        blocker = ObjectObservation(2, (map_id, (3, 0)))
        self.assertEqual(trainer_hazard_locations(trainer, tiles, (blocker,)), frozenset({(map_id, (2, 0))}))
        self.assertEqual(trainer_hazard_locations(replace(trainer, trainer_defeated=True), tiles), frozenset())

    def test_missing_avatar_is_explicitly_unavailable(self):
        with patch("modules.overworld.get_player_avatar", return_value=None):
            result = perceive_overworld()
        self.assertIsInstance(result, OverworldObservationResult)
        self.assertEqual(result.status, OverworldObservationStatus.UNAVAILABLE)
        self.assertEqual(result.reason, "player_avatar_unavailable")

    def test_malformed_avatar_coordinates_are_explicitly_malformed(self):
        avatar = SimpleNamespace(map_group_and_number=(1, 2), local_coordinates=None)
        with patch("modules.overworld.get_player_avatar", return_value=avatar):
            result = perceive_overworld()
        self.assertEqual(result.status, OverworldObservationStatus.MALFORMED)

    def test_unexpected_avatar_error_propagates(self):
        with patch("modules.overworld.get_player_avatar", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                perceive_overworld()

    def test_coordinate_trigger_condition_is_observable_without_script_interpretation(self):
        trigger = TriggerObservation(
            "coord:1:123:4",
            frozenset({((1, 2), (2, 8))}),
            condition_variable_number=123,
            condition_variable="LITTLEROOT_INTRO_STATE",
            condition_required_value=4,
            script_symbol="LittlerootTown_MaysHouse_1F_EventScript_GoSeeRoom",
        )
        self.assertTrue(evaluate_trigger_condition(trigger, 4))
        self.assertFalse(evaluate_trigger_condition(trigger, 3))
        self.assertIsNone(evaluate_trigger_condition(trigger, None))

    def test_object_observation_preserves_dynamic_and_interaction_metadata(self):
        object_observation = ObjectObservation(
            2,
            ((1, 2), (5, 5)),
            script="TrainerScript",
            previous_location=((1, 2), (5, 4)),
            movement_type="Walk Around",
            movement_action="Walk Normal Down",
            script_controlled=True,
            trainer_type="Normal",
            trainer_range=3,
            trainer_defeated=False,
        )
        self.assertNotEqual(object_observation.location, object_observation.previous_location)
        self.assertTrue(object_observation.script_controlled)
        self.assertEqual(object_observation.trainer_type, "Normal")
        self.assertFalse(object_observation.trainer_defeated)

    def test_any_facing_script_exposes_per_side_facing_requirements(self):
        map_id = (92, 7)
        map_data = MapMetadata(
            map_id,
            b"header",
            (3).to_bytes(4, "little") + (3).to_bytes(4, "little"),
            b"events",
            (),
            (),
            (),
            (
                SimpleNamespace(
                    local_coordinates=(1, 1),
                    kind="Script",
                    player_facing_direction="Any",
                    script_symbol="Generic_EventScript_Interact",
                ),
            ),
            (),
        )
        path_tiles = [
            SimpleNamespace(
                local_coordinates=(x, y),
                accessible_from_direction=[True] * 4,
                warps_to=None,
                traversal_cost=1,
            )
            for y in range(3)
            for x in range(3)
        ]
        with (
            patch("modules.overworld._get_map_metadata", return_value=SimpleNamespace(tiles=path_tiles)),
            patch("modules.overworld.get_map_metadata", return_value=map_data),
        ):
            prewarm_static_map_observation(map_id)

        trigger = next(
            trigger
            for trigger in __import__("modules.overworld", fromlist=["_static_map_observations"])
            ._static_map_observations[map_id]
            .triggers
            if trigger.script_symbol == "Generic_EventScript_Interact"
        )
        requirements = dict(trigger.activation_requirements)
        self.assertEqual(requirements[((map_id, (1, 2)))], Direction.North)
        self.assertEqual(requirements[((map_id, (1, 0)))], Direction.South)

    def test_map_connection_observation_uses_full_endpoint_coordinates(self):
        source_id, destination_id = (93, 1), (93, 2)
        tile = lambda coordinate: SimpleNamespace(
            local_coordinates=coordinate,
            accessible_from_direction=[True] * 4,
            warps_to=None,
            traversal_cost=1,
        )
        source = SimpleNamespace(
            map_size=(3, 2),
            warps=(),
            connections=(
                SimpleNamespace(
                    destination_map_group=destination_id[0],
                    destination_map_number=destination_id[1],
                    direction="North",
                    offset=0,
                ),
            ),
            objects=(),
            bg_events=(),
            coord_events=(),
        )
        destination = SimpleNamespace(
            map_size=(3, 2),
            warps=(),
            connections=(),
            objects=(),
            bg_events=(),
            coord_events=(),
        )
        with patch(
            "modules.overworld._get_map_metadata",
            side_effect=lambda map_id: SimpleNamespace(tiles=tuple(tile((x, y)) for y in range(2) for x in range(3))),
        ), patch(
            "modules.overworld.get_map_metadata",
            side_effect=lambda map_id: source if map_id == source_id else destination,
        ):
            prewarm_static_map_observation(source_id)
        static = __import__("modules.overworld", fromlist=["_static_map_observations"])._static_map_observations[
            source_id
        ]
        self.assertTrue(any(t.kind == "map_connection" for t in static.transitions))

    def test_reuses_static_tiles_when_only_avatar_state_changes(self):
        map_id = (91, 7)
        map_data = MapMetadata(
            map_id,
            b"header",
            (2).to_bytes(4, "little") + (1).to_bytes(4, "little"),
            b"events",
            (),
            (),
            (),
            (),
            (),
        )
        path_tiles = [
            SimpleNamespace(
                local_coordinates=(x, 0),
                accessible_from_direction=[True] * 4,
                warps_to=None,
                traversal_cost=2 if x == 1 else 1,
            )
            for x in range(2)
        ]
        avatar = SimpleNamespace(
            map_group_and_number=map_id,
            local_coordinates=(0, 0),
            facing_direction="Down",
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
            map_id,
            b"header",
            (3).to_bytes(4, "little") + (4).to_bytes(4, "little"),
            b"events",
            (),
            (),
            (),
            (),
            (),
        )
        template_map = SimpleNamespace(objects=[SimpleNamespace(local_id=7, script_symbol="Route101_EventScript_NPC")])
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
            patch(
                "modules.map.get_map_metadata",
                return_value=SimpleNamespace(
                    object_template=lambda local_id: (
                        template_map.objects[0] if local_id == template_map.objects[0].local_id else None
                    ),
                ),
            ),
        ):
            observation = perceive_overworld()

        self.assertEqual(len(observation.objects), 1)
        self.assertEqual(observation.objects[0].local_id, 7)
        self.assertEqual(observation.objects[0].script, "Route101_EventScript_NPC")
        self.assertIsNone(observation.objects[0].kind)
        self.assertEqual(observation.facing, Direction.South)


if __name__ == "__main__":
    unittest.main()
