import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.map import ObjectEvent
from modules.map_data import MapRSE
from modules.overworld import ObjectObservation, perceive_overworld
from modules.trigger_bindings import (
    get_trigger_binding,
    resolve_trigger_binding,
    TriggerBinding,
)


def _route103_object_event() -> ObjectEvent:
    data = bytearray(0x24)
    data[0] = 0x01  # active
    data[8] = 2  # local ID from Route 103's rival object template
    data[9] = MapRSE.ROUTE103.value[1]
    data[10] = MapRSE.ROUTE103.value[0]
    data[0x10:0x12] = (17).to_bytes(2, "little")  # current x = 10
    data[0x12:0x14] = (10).to_bytes(2, "little")  # current y = 3
    return ObjectEvent(bytes(data))


class TestTriggerBindings(unittest.TestCase):
    def test_introductory_rival_uses_authoritative_identity(self):
        binding = get_trigger_binding("introductory_rival")

        self.assertIsNotNone(binding)
        self.assertEqual(binding.map_id, MapRSE.ROUTE103.value)
        self.assertEqual(binding.script_symbol, "Route103_EventScript_Rival")
        self.assertEqual(binding.event_type, "object")

    def test_real_object_event_template_script_matches_binding(self):
        object_event = _route103_object_event()
        template_map = SimpleNamespace(
            objects=[
                SimpleNamespace(
                    local_id=object_event.local_id,
                    script_symbol="Route103_EventScript_Rival",
                )
            ]
        )
        with patch(
            "modules.map.get_map_metadata",
            return_value=SimpleNamespace(
                object_template=lambda local_id: (
                    template_map.objects[0] if local_id == template_map.objects[0].local_id else None
                ),
            ),
        ):
            runtime_object = ObjectObservation(
                local_id=object_event.local_id,
                location=(object_event.map_group_and_number, object_event.current_coords),
                script=object_event.object_event_template.script_symbol,
            )

        resolution = resolve_trigger_binding(
            get_trigger_binding("introductory_rival"),
            (runtime_object,),
            map_size=(30, 20),
        )

        self.assertTrue(resolution.runtime_match)
        self.assertEqual(resolution.object_ids, (2,))
        self.assertEqual(resolution.runtime_locations, ((10, 3),))
        self.assertEqual(
            set(resolution.interaction_positions),
            {(10, 2), (11, 3), (10, 4), (9, 3)},
        )

    def test_static_template_resolves_without_runtime_object(self):
        binding = get_trigger_binding("introductory_rival")
        static_template = SimpleNamespace(
            local_id=2,
            local_coordinates=(10, 3),
            script_symbol="Route103_EventScript_Rival",
            flag_id=0,
        )

        resolution = resolve_trigger_binding(binding, (), map_size=(30, 30), static_objects=(static_template,))

        self.assertTrue(resolution.static_match)
        self.assertTrue(resolution.static_available)
        self.assertEqual(resolution.static_location, (MapRSE.ROUTE103.value, (10, 3)))
        self.assertFalse(resolution.runtime_match)
        self.assertEqual(
            set(resolution.interaction_positions),
            {(10, 2), (11, 3), (10, 4), (9, 3)},
        )

    def test_runtime_position_overrides_static_position(self):
        binding = get_trigger_binding("introductory_rival")
        static_template = SimpleNamespace(
            local_id=2,
            local_coordinates=(10, 3),
            script_symbol="Route103_EventScript_Rival",
            flag_id=0,
        )
        runtime_object = ObjectObservation(
            local_id=2,
            location=(MapRSE.ROUTE103.value, (11, 4)),
            script="Route103_EventScript_Rival",
        )

        resolution = resolve_trigger_binding(
            binding, (runtime_object,), map_size=(30, 30), static_objects=(static_template,)
        )

        self.assertEqual(resolution.static_location, (MapRSE.ROUTE103.value, (10, 3)))
        self.assertEqual(
            set(resolution.interaction_positions),
            {(11, 3), (12, 4), (11, 5), (10, 4)},
        )

    def test_hidden_static_template_is_not_available(self):
        binding = get_trigger_binding("introductory_rival")
        static_template = SimpleNamespace(
            local_id=2,
            local_coordinates=(10, 3),
            script_symbol="Route103_EventScript_Rival",
            flag_id=123,
        )

        with patch("modules.trigger_bindings.get_event_flag_by_number", return_value=True):
            resolution = resolve_trigger_binding(
                binding,
                (
                    ObjectObservation(
                        local_id=2,
                        location=(MapRSE.ROUTE103.value, (10, 3)),
                        script="Route103_EventScript_Rival",
                    ),
                ),
                map_size=(30, 30),
                static_objects=(static_template,),
            )

        self.assertTrue(resolution.static_match)
        self.assertFalse(resolution.static_available)
        self.assertIsNone(resolution.static_location)
        self.assertFalse(resolution.runtime_match)

    def test_local_id_disambiguates_static_templates(self):
        binding = get_trigger_binding("introductory_rival")
        templates = (
            SimpleNamespace(
                local_id=8, local_coordinates=(2, 2), script_symbol="Route103_EventScript_Rival", flag_id=0
            ),
            SimpleNamespace(
                local_id=2, local_coordinates=(10, 3), script_symbol="Route103_EventScript_Rival", flag_id=0
            ),
        )

        resolution = resolve_trigger_binding(binding, (), map_size=(30, 30), static_objects=templates)

        self.assertEqual(resolution.static_location, (MapRSE.ROUTE103.value, (10, 3)))

    def test_script_only_ambiguous_static_templates_are_not_selected(self):
        binding = TriggerBinding("rival", MapRSE.ROUTE103.value, "Route103_EventScript_Rival")
        templates = (
            SimpleNamespace(local_id=2, local_coordinates=(10, 3), script_symbol=binding.script_symbol, flag_id=0),
            SimpleNamespace(local_id=3, local_coordinates=(11, 3), script_symbol=binding.script_symbol, flag_id=0),
        )

        resolution = resolve_trigger_binding(binding, (), map_size=(30, 30), static_objects=templates)

        self.assertTrue(resolution.static_match)
        self.assertTrue(resolution.static_ambiguous)
        self.assertIsNone(resolution.static_location)

    def test_perception_emits_semantic_trigger_for_runtime_object(self):
        object_event = _route103_object_event()
        avatar = SimpleNamespace(
            map_group_and_number=MapRSE.ROUTE103.value,
            local_coordinates=(5, 5),
            facing_direction="North",
        )
        map_data = SimpleNamespace(
            map_size=(30, 20),
            warps=[],
            coord_events=[],
            bg_events=[],
            objects=[
                SimpleNamespace(
                    local_id=2,
                    local_coordinates=(10, 3),
                    script_symbol="Route103_EventScript_Rival",
                    flag_id=0,
                )
            ],
        )
        path_tile = SimpleNamespace(
            local_coordinates=(10, 3),
            accessible_from_direction=[True, True, True, True],
            warps_to=None,
        )
        template_map = SimpleNamespace(
            objects=[
                SimpleNamespace(
                    local_id=object_event.local_id,
                    script_symbol="Route103_EventScript_Rival",
                )
            ]
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

        semantic_trigger = next(
            trigger for trigger in observation.triggers if trigger.trigger_id == "introductory_rival"
        )
        self.assertEqual(semantic_trigger.target_map, MapRSE.ROUTE103.value)
        self.assertEqual(
            semantic_trigger.activation_locations,
            frozenset(
                {
                    (MapRSE.ROUTE103.value, (10, 2)),
                    (MapRSE.ROUTE103.value, (11, 3)),
                    (MapRSE.ROUTE103.value, (10, 4)),
                    (MapRSE.ROUTE103.value, (9, 3)),
                }
            ),
        )
        self.assertEqual(
            semantic_trigger.navigation_locations,
            frozenset({(MapRSE.ROUTE103.value, (10, 3))}),
        )

    def test_unrelated_object_does_not_match(self):
        binding = get_trigger_binding("introductory_rival")
        unrelated = ObjectObservation(
            local_id=8,
            location=(MapRSE.ROUTE103.value, (10, 3)),
            script="Route103_EventScript_Boy",
        )

        resolution = resolve_trigger_binding(binding, (unrelated,), map_size=(30, 20))

        self.assertFalse(resolution.runtime_match)
        self.assertEqual(resolution.object_ids, ())
        self.assertEqual(resolution.interaction_positions, ())

    def test_unknown_semantic_goal_has_no_arbitrary_binding(self):
        self.assertIsNone(get_trigger_binding("not_in_rom"))

    def test_early_pokeball_binding_matches_both_gender_scripts(self):
        binding = get_trigger_binding("early_pokeballs")
        self.assertIsNotNone(binding)
        self.assertEqual(binding.map_id, MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value)
        self.assertEqual(binding.local_id, 2)
        self.assertEqual(binding.script_symbol, "LittlerootTown_ProfessorBirchsLab_EventScript_Birch")
        self.assertIn(binding.script_symbol, binding.script_symbols)
        self.assertIn("LittlerootTown_ProfessorBirchsLab_EventScript_BrendanGivePokeBalls", binding.script_symbols)
        self.assertIn("LittlerootTown_ProfessorBirchsLab_EventScript_MayGivePokeBalls", binding.script_symbols)

        for script in binding.script_symbols:
            resolution = resolve_trigger_binding(
                binding,
                (ObjectObservation(2, (binding.map_id, (5, 5)), script=script),),
                map_size=(20, 20),
            )
            self.assertTrue(resolution.runtime_match)

    def test_early_pokeball_birch_runtime_position_and_facing(self):
        binding = get_trigger_binding("early_pokeballs")
        resolution = resolve_trigger_binding(
            binding,
            (ObjectObservation(2, (binding.map_id, (6, 4)), script=binding.script_symbol),),
            map_size=(20, 20),
        )
        self.assertTrue(resolution.runtime_match)
        self.assertEqual(resolution.runtime_locations, ((6, 4),))
        self.assertEqual(set(resolution.interaction_positions), {(6, 3), (7, 4), (6, 5), (5, 4)})


if __name__ == "__main__":
    unittest.main()
