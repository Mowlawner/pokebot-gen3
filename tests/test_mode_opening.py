import types
import unittest
from unittest.mock import patch


class TestEmeraldOpeningState(unittest.TestCase):
    def test_littleroot_and_house_maps_have_distinct_progression_states(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.get_event_flag", return_value=True),
        ):
            for map_id, expected in (
                ((0, 9), OpeningSequenceState.LITTLEROOT_TOWN),
                ((1, 0), OpeningSequenceState.PLAYER_HOUSE_1F),
                ((1, 1), OpeningSequenceState.PLAYER_HOUSE_2F),
                ((1, 2), OpeningSequenceState.BIRCH_HOUSE_1F),
                ((1, 3), OpeningSequenceState.BIRCH_HOUSE_2F),
                ((0, 16), OpeningSequenceState.ROUTE_101),
            ):
                with patch(
                    "modules.modes.opening.get_map_data_for_current_position",
                    return_value=types.SimpleNamespace(map_group_and_number=map_id),
                ):
                    self.assertIs(get_opening_sequence_state(), expected)

    def test_clock_task_is_explicitly_detected(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.get_map_data_for_current_position", return_value=None),
            patch(
                "modules.modes.opening.task_is_active",
                side_effect=lambda task: task == "Task_SetClock_HandleInput",
            ),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.CLOCK_SETTING)

    def test_transitional_frame_is_retryable_and_not_manual(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.get_map_data_for_current_position", return_value=None),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.TRUCK)

    def test_unavailable_game_state_is_unknown_and_retryable(self):
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with patch("modules.modes.opening.get_game_state", side_effect=RuntimeError("transition")):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.UNKNOWN)

    def test_littleroot_map_is_authoritative_after_truck_transition(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                return_value=types.SimpleNamespace(map_group_and_number=(0, 9)),
            ),
            patch("modules.modes.opening.get_event_flag", return_value=False),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.LITTLEROOT_TOWN)

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                return_value=types.SimpleNamespace(map_group_and_number=(0, 9)),
            ),
            patch("modules.modes.opening.get_event_flag", return_value=True),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.LITTLEROOT_TOWN)

    def test_dedicated_truck_map_is_truck_even_when_littleroot_is_reported_during_transition(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                return_value=types.SimpleNamespace(map_group_and_number=MapRSE.INSIDE_OF_TRUCK.value),
            ),
            patch("modules.modes.opening.task_is_active", return_value=False),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.TRUCK)

    def test_truck_task_is_waited_for_without_pressing_a(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.TRUCK
        with (
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.INSIDE_OF_TRUCK.value),
            patch("modules.modes.opening.task_is_active", return_value=True),
            patch("modules.modes.opening.get_global_script_context", return_value=None),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch.object(
                __import__("modules.modes.opening", fromlist=["context"]).context,
                "emulator",
                types.SimpleNamespace(press_button=unittest.mock.Mock()),
            ) as mocked_emulator,
        ):
            generator = mode._advance_phase(OpeningSequenceState.TRUCK)
            next(generator)
            self.assertIs(mode.phase, OpeningSequenceState.TRUCK)
            mocked_emulator.press_button.assert_not_called()

    def test_controllable_truck_navigates_to_rom_defined_exit_tile(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        resolved_destination = types.SimpleNamespace(
            map_group_and_number=MapRSE.LITTLEROOT_TOWN.value,
        )
        location = types.SimpleNamespace(
            map_group_and_number=MapRSE.INSIDE_OF_TRUCK.value,
            warps=[types.SimpleNamespace(
                destination_map_group=127,
                destination_map_number=127,
                destination_location=resolved_destination,
                local_coordinates=(4, 4),
            )],
        )
        with (
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.INSIDE_OF_TRUCK.value),
            patch("modules.modes.opening.task_is_active", return_value=False),
            patch("modules.modes.opening.get_global_script_context", return_value=None),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_map_data_for_current_position", return_value=location),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
        ):
            generator = mode._advance_phase(OpeningSequenceState.TRUCK)
            list(generator)
            navigate.assert_called_once_with(
                MapRSE.INSIDE_OF_TRUCK.value,
                (4, 1),
                avoid_scripted_events=False,
                expecting_script=True,
            )

    def test_observed_controllable_truck_frame_navigates_instead_of_waiting(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import (
            EmeraldOpeningMode,
            OpeningDiagnostics,
            OpeningSequenceState,
        )

        mode = EmeraldOpeningMode()
        location = types.SimpleNamespace(
            map_group_and_number=MapRSE.INSIDE_OF_TRUCK.value,
            warps=[types.SimpleNamespace(
                destination_map_group=127,
                destination_map_number=127,
                destination_location=property(lambda _: (_ for _ in ()).throw(AssertionError("dynamic destination must not be read"))),
                local_coordinates=(4, 1),
            )],
        )
        diagnostics = OpeningDiagnostics(
            phase=OpeningSequenceState.TRUCK,
            observed=OpeningSequenceState.TRUCK,
            map_id=MapRSE.INSIDE_OF_TRUCK.value,
            coordinates=(2, 2),
            game_state=GameState.OVERWORLD,
            controllable=True,
            script_active=False,
            waiting_for_input=0,
            facing="Down",
            truck_flag=False,
            truck_sequence_active=False,
        )
        with (
            patch("modules.modes.opening.get_map_data_for_current_position", return_value=location),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
        ):
            list(mode._advance_phase(OpeningSequenceState.TRUCK, diagnostics))

        navigate.assert_called_once_with(
            MapRSE.INSIDE_OF_TRUCK.value,
            (4, 1),
            avoid_scripted_events=False,
            expecting_script=True,
        )
        self.assertEqual(
            mode._last_truck_navigation_target,
            (MapRSE.INSIDE_OF_TRUCK.value, (4, 1)),
        )
        self.assertIn("ROM truck exit tile", mode._last_truck_decision)

    def test_truck_exit_discards_target_and_waits_for_expected_mom_script(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._last_truck_navigation_target = (MapRSE.INSIDE_OF_TRUCK.value, (4, 1))
        mode.phase = OpeningSequenceState.TRUCK

        with (
            patch("modules.modes.opening._truck_has_left", return_value=True),
            patch("modules.modes.opening._normal_littleroot_overworld", return_value=False),
            patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance,
        ):
            list(mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN))

        self.assertIs(mode.phase, OpeningSequenceState.LITTLEROOT_TOWN)
        self.assertIsNone(mode._last_truck_navigation_target)
        advance.assert_not_called()

        with (
            patch("modules.modes.opening._littleroot_arrival_ready", return_value=False),
            patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance,
        ):
            list(mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN))

        advance.assert_called_once_with()

    def test_littleroot_map_exits_truck_phase_during_mom_dialogue(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._last_truck_navigation_target = (MapRSE.INSIDE_OF_TRUCK.value, (4, 1))
        mode._last_truck_exit_warp = (4, 1)
        mode._last_truck_exit_destination = MapRSE.LITTLEROOT_TOWN.value
        mode.phase = OpeningSequenceState.TRUCK

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN.value),
            patch("modules.modes.opening._truck_has_left", return_value=False),
            patch("modules.modes.opening._truck_sequence_active", return_value=True),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=False),
        ):
            list(mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN))

        self.assertIs(mode.phase, OpeningSequenceState.LITTLEROOT_TOWN)
        self.assertIsNone(mode._last_truck_navigation_target)
        self.assertIsNone(mode._last_truck_exit_warp)
        self.assertIsNone(mode._last_truck_exit_destination)

        with (
            patch("modules.modes.opening._littleroot_arrival_ready", return_value=False),
            patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance,
        ):
            list(mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN))
        advance.assert_called_once_with()

    def test_littleroot_phase_reconciles_observed_player_house_and_navigates_stairs(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.LITTLEROOT_TOWN
        with (
            patch("modules.modes.opening._littleroot_arrival_ready", return_value=True),
            patch("modules.modes.opening._player_house_1f_ready_for_navigation", return_value=True),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
            patch("modules.modes.opening.get_event_flag", return_value=False),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_1F))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)
        self.assertIs(
            mode._pending_house_warp_destination,
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
        )
        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F)

        with patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))
        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_2F)
        self.assertIsNone(mode._pending_house_warp_destination)

    def test_player_house_1f_initial_dialogue_is_advanced_before_stair_navigation(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_1F
        with (
            patch("modules.modes.opening._player_house_1f_ready_for_navigation", return_value=False),
            patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance,
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
            patch("modules.modes.opening.get_event_flag", return_value=False),
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_1F))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)
        self.assertIsNone(mode._pending_house_warp_destination)
        self.assertEqual(mode._last_truck_decision, "wait: player's house 1F introductory dialogue")
        advance.assert_called_once_with()
        warp.assert_not_called()

    def test_player_house_1f_ready_state_starts_stair_navigation(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_1F
        with (
            patch("modules.modes.opening._player_house_1f_ready_for_navigation", return_value=True),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
            patch("modules.modes.opening.get_event_flag", return_value=False),
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_1F))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)
        self.assertIs(mode._pending_house_warp_destination, MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F)
        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F)

    def test_upstairs_navigation_transitions_to_clock_setting_phase(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_2F
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())) as face,
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(
                __import__("modules.modes.opening", fromlist=["context"]).context,
                "emulator",
                types.SimpleNamespace(press_button=unittest.mock.Mock()),
            ) as emulator,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        self.assertIs(mode.phase, OpeningSequenceState.CLOCK_SETTING)
        navigate.assert_called_once_with(
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
            (7, 2),
            avoid_scripted_events=False,
        )
        face.assert_called_once_with("Up")
        emulator.press_button.assert_called_once_with("A")

    def test_mom_dialogue_task_advances_when_script_context_is_unavailable(self):
        from modules.modes.opening import _advance_scripted_input

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        with (
            patch("modules.modes.opening.context") as opening_context,
            patch("modules.modes.opening.is_waiting_for_input", return_value=False),
            patch("modules.modes.opening.task_is_active", side_effect=lambda task: task == "Task_DrawFieldMessage"),
        ):
            opening_context.rom.is_emerald = True
            opening_context.emulator = emulator
            list(_advance_scripted_input())

        emulator.press_button.assert_called_once_with("A")

    def test_truck_phase_waits_for_observed_littleroot_transition(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import (
            EmeraldOpeningMode,
            OpeningDiagnostics,
            OpeningSequenceState,
        )

        mode = EmeraldOpeningMode()
        diagnostics = OpeningDiagnostics(
            OpeningSequenceState.TRUCK,
            OpeningSequenceState.TRUCK,
            MapRSE.INSIDE_OF_TRUCK.value,
            (2, 2),
            GameState.OVERWORLD,
            True,
            False,
            False,
            "Down",
            False,
            False,
        )
        with patch("modules.modes.opening._normal_littleroot_overworld", return_value=True):
            list(mode._advance_phase(OpeningSequenceState.TRUCK, diagnostics))
        self.assertIs(mode.phase, OpeningSequenceState.LITTLEROOT_TOWN)

    def test_truck_phase_requires_normal_littleroot_overworld_after_warp(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.TRUCK
        with (
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN.value),
            patch("modules.modes.opening._truck_has_left", return_value=True),
            patch("modules.modes.opening._truck_sequence_active", return_value=False),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
        ):
            generator = mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN)
            with patch("modules.modes.opening._warp_to", return_value=iter(())):
                list(generator)
            self.assertIs(mode.phase, OpeningSequenceState.LITTLEROOT_TOWN)

    def test_transitional_truck_observation_remains_retryable(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.TRUCK
        with patch("modules.modes.opening._current_map_id", return_value=None):
            generator = mode._advance_phase(OpeningSequenceState.UNKNOWN)
            next(generator)
        self.assertIs(mode.phase, OpeningSequenceState.TRUCK)

    def test_naming_waits_for_keyboard_data(self):
        from modules.memory import GameState
        from modules.modes.opening import _enter_player_name

        naming_data = types.SimpleNamespace()
        with (
            patch(
                "modules.modes.opening.get_game_state",
                side_effect=[GameState.NAMING_SCREEN, GameState.NAMING_SCREEN],
            ),
            patch(
                "modules.modes.opening.get_naming_screen_data",
                side_effect=[None, naming_data],
            ),
            patch("modules.modes.opening.type_in_naming_screen", return_value=iter(())) as type_name,
        ):
            generator = _enter_player_name()
            next(generator)
            with self.assertRaises(StopIteration):
                next(generator)
            type_name.assert_called_once_with("RED")

    def test_item_ball_detection_returns_object_coordinate(self):
        from modules.modes.opening import _item_ball_coordinates

        objects = [
            types.SimpleNamespace(graphics_id=0x01, local_coordinates=(4, 4)),
            types.SimpleNamespace(graphics_id=0x3C, local_coordinates=(6, 3)),
        ]
        with patch(
            "modules.modes.opening.get_map_data_for_current_position",
            return_value=types.SimpleNamespace(objects=objects),
        ):
            self.assertEqual(_item_ball_coordinates(), (6, 3))

    def test_choose_starter_callback_on_route_101_is_explicit_handoff(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        location = types.SimpleNamespace(map_group_and_number=(0, 16))
        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.CHOOSE_STARTER),
            patch("modules.modes.opening.get_map_data_for_current_position", return_value=location),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.STARTER_SELECTION)

    def test_choose_starter_callback_on_another_map_is_not_handoff(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        location = types.SimpleNamespace(map_group_and_number=(1, 4))
        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.CHOOSE_STARTER),
            patch("modules.modes.opening.get_map_data_for_current_position", return_value=location),
        ):
            self.assertIsNot(get_opening_sequence_state(), OpeningSequenceState.STARTER_SELECTION)

    def test_controller_hands_off_without_pressing_a(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        with patch("modules.modes.opening.context") as mocked_context:
            mocked_context.rom = types.SimpleNamespace(is_emerald=True)
            mocked_context.bot_mode = "Start New Game"
            mocked_context.debug = False
            mocked_context.emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
            with patch(
                "modules.modes.opening.get_opening_sequence_state",
                return_value=OpeningSequenceState.STARTER_SELECTION,
            ):
                with self.assertRaises(StopIteration):
                    next(EmeraldOpeningMode().run())
            self.assertEqual(mocked_context.bot_mode, "Starters")
            mocked_context.emulator.press_button.assert_not_called()
            # The real main loop would immediately construct StartersMode and
            # consume this handoff marker; this unit test stops at the boundary.
            from modules.modes.opening import consume_starter_handoff

            self.assertTrue(consume_starter_handoff())
