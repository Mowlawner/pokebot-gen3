import types
import unittest
from unittest.mock import patch


class TestEmeraldOpeningState(unittest.TestCase):
    def test_run_rebases_legacy_phase_before_dispatch(self):
        """A capability mounted with a stale phase follows the ROM frame."""
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningCapability, OpeningSequenceState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            debug=False,
            debug_trace=False,
            rom=types.SimpleNamespace(is_emerald=True),
            emulator=emulator,
            bot_mode="Start New Game",
        )
        mode = EmeraldOpeningCapability(campaign_owned=True)
        mode.phase = OpeningSequenceState.BIRCH_HOUSE_2F
        with (
            patch("modules.modes.opening.context", opening_context),
            patch(
                "modules.modes.opening.get_opening_sequence_state", return_value=OpeningSequenceState.PLAYER_HOUSE_1F
            ),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(
                mode,
                "_dialogue_state_snapshot",
                return_value=(True, 0, False, "WaitForAorBPress", None, True, True, True, None, (), None, None),
            ),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            execution = mode.run()
            next(execution)

        emulator.press_button.assert_called_once_with("B")
        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)

    def test_navigation_transaction_is_abandoned_after_map_observation_changes(self):
        from modules.modes.opening import EmeraldOpeningCapability, OpeningSequenceState

        mode = EmeraldOpeningCapability(campaign_owned=True)
        inputs = []

        def navigation():
            inputs.append("Up")
            yield
            inputs.append("Up")
            yield

        with (
            patch(
                "modules.modes.opening.get_opening_sequence_state",
                side_effect=[
                    OpeningSequenceState.PLAYER_HOUSE_1F,
                    OpeningSequenceState.PLAYER_HOUSE_1F,
                    OpeningSequenceState.BIRCH_HOUSE_1F,
                ],
            ),
            patch.object(
                mode,
                "_dialogue_state_snapshot",
                return_value=(False, None, False, None, None, False, True, False, None, (), None, None),
            ),
            patch.object(mode, "_dialogue_detection", return_value=(False, "no dialogue")),
        ):
            transaction = mode._run_navigation_transaction(navigation())
            next(transaction)
            with self.assertRaises(StopIteration):
                next(transaction)

        self.assertEqual(inputs, ["Up"])

    def test_littleroot_house_maps_are_resolved_by_player_gender(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        cases = (
            (
                "male",
                MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F,
                OpeningSequenceState.PLAYER_HOUSE_1F,
                MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
                OpeningSequenceState.BIRCH_HOUSE_1F,
            ),
            (
                "female",
                MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
                OpeningSequenceState.PLAYER_HOUSE_1F,
                MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F,
                OpeningSequenceState.BIRCH_HOUSE_1F,
            ),
        )
        with patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD):
            for gender, player_map, player_state, rival_map, rival_state in cases:
                for map_id, expected in ((player_map, player_state), (rival_map, rival_state)):
                    with patch(
                        "modules.modes.opening.get_map_data_for_current_position",
                        return_value=types.SimpleNamespace(map_group_and_number=map_id.value),
                    ):
                        self.assertIs(get_opening_sequence_state(gender), expected)

    def test_options_menu_is_recognized_as_a_pre_game_state(self):
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with patch("modules.modes.opening.get_game_state", return_value=GameState.OPTIONS_MENU):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.OPTIONS_MENU)

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

    def test_birch_lab_with_a_party_is_the_post_starter_completion_state(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import OpeningSequenceState, get_opening_sequence_state

        with (
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                return_value=types.SimpleNamespace(
                    map_group_and_number=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value,
                ),
            ),
            patch("modules.modes.opening.get_party_size", return_value=1),
        ):
            self.assertIs(get_opening_sequence_state(), OpeningSequenceState.COMPLETE)

    def test_completed_opening_returns_without_input_or_navigation(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            bot_mode="Start New Game",
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            debug=False,
            config=types.SimpleNamespace(
                start_game=types.SimpleNamespace(player_name="gibberish", player_gender="random")
            ),
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch(
                "modules.modes.opening.get_opening_sequence_state",
                return_value=OpeningSequenceState.COMPLETE,
            ),
        ):
            with self.assertRaises(StopIteration):
                next(EmeraldOpeningMode().run())

        self.assertEqual(opening_context.bot_mode, "Start New Game")
        emulator.press_button.assert_not_called()

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
        mode._resolved_player_gender = "male"
        mode.phase = OpeningSequenceState.TRUCK
        resolved_destination = types.SimpleNamespace(
            map_group_and_number=MapRSE.LITTLEROOT_TOWN.value,
        )
        location = types.SimpleNamespace(
            map_group_and_number=MapRSE.INSIDE_OF_TRUCK.value,
            warps=[
                types.SimpleNamespace(
                    destination_map_group=127,
                    destination_map_number=127,
                    destination_location=resolved_destination,
                    local_coordinates=(4, 4),
                )
            ],
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

    def test_actionable_dialogue_preempts_suspended_navigation_transaction(self):
        """A higher-priority observation abandons movement before the next input."""
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState
        from modules.nuzlocke.emerald_capabilities import (
            EmeraldCampaignAction,
            EmeraldCampaignObservation,
            choose_emerald_campaign_action,
        )

        mode = EmeraldOpeningMode()
        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        navigation_inputs = []

        def navigation():
            while True:
                emulator.press_button("Up")
                navigation_inputs.append("Up")
                yield

        with (
            patch("modules.modes.opening.context.emulator", emulator),
            patch.object(
                mode,
                "_dialogue_state_snapshot",
                return_value=(False, None, True, "WaitForAorBPress", "Std_MsgboxDefault", True, False, False),
            ),
            patch.object(
                mode,
                "_dialogue_detection",
                side_effect=[(False, "navigation still owns execution"), (True, "dialogue is actionable")],
            ),
            patch("modules.modes.opening.get_opening_sequence_state", return_value=OpeningSequenceState.SCRIPTED_INTRO),
        ):
            transaction = mode._run_navigation_transaction(navigation())
            next(transaction)
            with self.assertRaises(StopIteration):
                next(transaction)

        observation = EmeraldCampaignObservation(
            OpeningSequenceState.SCRIPTED_INTRO,
            False,
            False,
            dialogue_waiting=True,
        )
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.ADVANCE_DIALOGUE)
        self.assertEqual(navigation_inputs, ["Up"])

        def advance_dialogue():
            emulator.press_button("A")
            yield

        dialogue_execution = advance_dialogue()
        next(dialogue_execution)
        self.assertEqual(
            [call.args[0] for call in emulator.press_button.call_args_list],
            ["Up", "A"],
        )

    def test_actionable_dialogue_preempts_facing_transaction_before_input(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        facing_inputs = []

        def facing():
            facing_inputs.append("Left")
            yield

        with (
            patch.object(
                mode,
                "_dialogue_state_snapshot",
                return_value=(False, None, True, "WaitForAorBPress", "Std_MsgboxDefault", True, False, False),
            ),
            patch.object(mode, "_dialogue_detection", return_value=(True, "dialogue is actionable")),
            patch(
                "modules.modes.opening.get_opening_sequence_state", return_value=OpeningSequenceState.PLAYER_HOUSE_1F
            ),
        ):
            list(mode._run_preemptible_transaction(facing()))

        self.assertEqual(facing_inputs, [])

    def test_observed_controllable_truck_frame_navigates_instead_of_waiting(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import (
            EmeraldOpeningMode,
            OpeningDiagnostics,
            OpeningSequenceState,
        )

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.TRUCK
        location = types.SimpleNamespace(
            map_group_and_number=MapRSE.INSIDE_OF_TRUCK.value,
            warps=[
                types.SimpleNamespace(
                    destination_map_group=127,
                    destination_map_number=127,
                    destination_location=property(
                        lambda _: (_ for _ in ()).throw(AssertionError("dynamic destination must not be read"))
                    ),
                    local_coordinates=(4, 1),
                )
            ],
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
        mode._resolved_player_gender = "male"
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

        with patch(
            "modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value
        ):
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
        mode._resolved_player_gender = "male"
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

    def test_post_clock_2f_does_not_stick_on_pending_1f_warp(self):
        """2F is temporary while the 1F phase waits for Mom's sequence."""
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "male"
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_1F
        mode._pending_house_warp_destination = MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F
        with (
            patch(
                "modules.modes.opening._current_map_id",
                return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value,
            ),
            patch("modules.modes.opening.get_event_flag", return_value=True),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.is_waiting_for_input", return_value=False),
            patch("modules.modes.opening.get_global_script_context", return_value=None),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)
        self.assertEqual(
            mode._last_truck_decision,
            "navigate: ROM-defined staircase warp back to player's house 1F",
        )
        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F)

    def test_house_1f_waits_for_non_animated_door_exit_before_navigation(self):
        """The 1F map is visible before the ROM's post-warp door task ends."""
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_1F
        with (
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F.value),
            patch(
                "modules.modes.opening.get_player_avatar",
                return_value=types.SimpleNamespace(local_coordinates=(8, 3)),
            ),
            patch("modules.modes.opening.get_event_flag", return_value=True),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.task_is_active", side_effect=lambda task: task == "Task_ExitNonAnimDoor"),
            # This is the result of the shared player controllability guard
            # while Task_ExitNonAnimDoor is active.
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=False),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_1F))

        navigate.assert_not_called()
        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)

    def test_post_clock_1f_arrival_waits_for_rom_tv_event_completion(self):
        """VAR_LITTLEROOT_INTRO_STATE 6 starts the expected TV event."""
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL
        with (
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F.value),
            patch("modules.modes.opening.get_event_var", return_value=6),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_global_script_context", return_value=None),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_1F))

        navigate.assert_not_called()
        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL)
        self.assertEqual(mode._last_truck_decision, "wait: expected post-clock 1F TV event")

    def test_post_clock_1f_arrival_returns_to_navigation_after_rom_completion_marker(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "male"
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL
        with (
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F.value),
            patch("modules.modes.opening.get_event_var", return_value=7),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_global_script_context", return_value=None),
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_1F))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)
        self.assertEqual(mode._last_truck_decision, "complete: post-clock 1F TV event")

    def test_post_clock_town_accepts_expected_birch_house_arrival_script(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "female"
        mode.phase = OpeningSequenceState.POST_CLOCK_TOWN
        with patch("modules.modes.opening._warp_to", return_value=iter(())) as warp:
            list(mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN))

        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F, expecting_script=True)
        self.assertIs(mode.phase, OpeningSequenceState.BIRCH_HOUSE_1F)

    def test_birch_house_waits_for_arrival_script_before_navigating_upstairs(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "female"
        mode.phase = OpeningSequenceState.BIRCH_HOUSE_1F
        with (
            patch("modules.modes.opening._birch_house_1f_ready_for_navigation", return_value=False),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
        ):
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_1F))

        warp.assert_not_called()
        self.assertEqual(mode._last_truck_decision, "wait: Birch's house arrival event")

        with (
            patch("modules.modes.opening._birch_house_1f_ready_for_navigation", return_value=True),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
        ):
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_1F))

        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F)
        self.assertIs(mode.phase, OpeningSequenceState.BIRCH_HOUSE_2F)

    def test_birch_house_reconciles_observed_second_floor_before_warp_returns(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.BIRCH_HOUSE_1F
        with patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance:
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_2F))

        self.assertIs(mode.phase, OpeningSequenceState.BIRCH_HOUSE_2F)
        advance.assert_not_called()

    def test_birch_second_floor_navigates_to_ball_and_starts_rival_event(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "female"
        mode.phase = OpeningSequenceState.BIRCH_HOUSE_2F
        emulator = unittest.mock.Mock()
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch(
                "modules.modes.opening._rival_pokeball_interaction",
                return_value=((6, 4), (6, 3)),
            ),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())) as face,
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_2F))

        navigate.assert_called_once_with(
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
            (6, 4),
            avoid_scripted_events=False,
        )
        face.assert_called_once_with((6, 3))
        emulator.press_button.assert_called_once_with("A")
        self.assertIs(mode.phase, OpeningSequenceState.MAY_SEQUENCE)

    def test_expected_rival_event_returns_to_opening_control(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.MAY_SEQUENCE
        with (
            patch("modules.modes.opening._may_sequence_event_complete", return_value=False),
            patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance,
        ):
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_2F))

        advance.assert_not_called()
        self.assertIs(mode.phase, OpeningSequenceState.MAY_SEQUENCE)

        list(mode._advance_phase(OpeningSequenceState.ROUTE_101))
        self.assertIs(mode.phase, OpeningSequenceState.ROUTE_101)

    def test_may_sequence_descends_after_rival_event_completes(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "female"
        mode.phase = OpeningSequenceState.MAY_SEQUENCE
        with (
            patch("modules.modes.opening._may_sequence_event_complete", return_value=True),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
        ):
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_2F))

        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F, expecting_script=True)
        self.assertIs(mode.phase, OpeningSequenceState.MAY_SEQUENCE)

    def test_may_sequence_leaves_house_after_descending(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.MAY_SEQUENCE
        with (
            patch("modules.modes.opening._may_sequence_event_complete", return_value=True),
            patch("modules.modes.opening._warp_to", return_value=iter(())) as warp,
        ):
            list(mode._advance_phase(OpeningSequenceState.BIRCH_HOUSE_1F))

        warp.assert_called_once_with(MapRSE.LITTLEROOT_TOWN, expecting_script=True)
        self.assertIs(mode.phase, OpeningSequenceState.ROUTE_101)

    def test_observed_second_floor_at_start_position_starts_clock_interaction(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "male"
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_2F
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening.get_event_flag", return_value=False),
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())) as face,
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
            (5, 2),
            avoid_scripted_events=False,
        )
        face.assert_called_once_with("Up")
        emulator.press_button.assert_called_once_with("A")

    def test_observed_second_floor_with_completed_clock_skips_clock_phase(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_2F
        with (
            patch("modules.modes.opening.get_event_flag", return_value=True),
            patch.object(mode, "_start_clock_interaction", return_value=iter(())) as start_clock,
            patch.object(mode, "_run_navigation_transaction", return_value=iter(())) as navigation,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)
        self.assertEqual(
            mode._last_truck_decision,
            "complete: wall clock already set; resume post-clock house progression",
        )
        start_clock.assert_not_called()
        navigation.assert_not_called()

    def test_completed_clock_dialogue_owns_frame_before_post_clock_navigation(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            debug=False,
            rom=types.SimpleNamespace(is_emerald=True),
            emulator=emulator,
        )
        actionable_message = (
            True,
            0,
            False,
            "WaitForAorBPress",
            None,
            True,
            True,
            True,
            None,
            (),
            None,
            None,
        )
        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.PLAYER_HOUSE_2F
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(mode, "_dialogue_state_snapshot", return_value=actionable_message),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
            patch.object(mode, "_run_navigation_transaction", return_value=iter(())) as navigation,
        ):
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_2F))

        self.assertEqual(result, [True])
        emulator.press_button.assert_called_once_with("B")
        navigation.assert_not_called()

    def test_clock_setting_observation_starts_interaction_without_event_lookup(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode._resolved_player_gender = "male"
        mode.phase = OpeningSequenceState.CLOCK_SETTING
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening._clock_task_active", return_value=False),
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())) as face,
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                side_effect=AssertionError("clock event lookup must not be required"),
            ),
            patch.object(
                __import__("modules.modes.opening", fromlist=["context"]).context,
                "emulator",
                types.SimpleNamespace(press_button=unittest.mock.Mock()),
            ),
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        self.assertIs(mode.phase, OpeningSequenceState.CLOCK_SETTING)
        self.assertTrue(mode._clock_interaction_started)
        navigate.assert_called_once_with(
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
            (5, 2),
            avoid_scripted_events=False,
        )
        face.assert_called_once_with("Up")

    def test_clock_ui_is_left_to_existing_clock_handler_after_interaction(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.CLOCK_SETTING
        mode._clock_interaction_started = True
        with (
            patch("modules.modes.opening._clock_task_active", return_value=True),
            patch.object(mode, "_set_clock", return_value=iter(())) as set_clock,
            patch("modules.modes.opening.navigate_to") as navigate,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        set_clock.assert_called_once_with()
        navigate.assert_not_called()

    def test_clock_handle_input_is_advanced_once_per_task_phase(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = object.__new__(EmeraldOpeningMode)
        mode._clock_target = (10, 0)
        mode._last_clock_task = None
        mode._clock_a_sent = False
        mode._clock_confirm_yes_prepared = False
        emulator = unittest.mock.Mock()
        with (
            patch(
                "modules.modes.opening._active_clock_task",
                side_effect=[
                    "Task_SetClock_HandleInput",
                    "Task_SetClock_HandleInput",
                    "Task_SetClock_AskConfirm",
                    "Task_SetClock_HandleConfirmInput",
                    "Task_SetClock_HandleConfirmInput",
                ],
            ),
            patch(
                "modules.modes.opening.get_task",
                return_value=types.SimpleNamespace(
                    data_value=lambda index: {0: 0, 2: 10, 3: 0}[index],
                ),
            ),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            for _ in range(5):
                list(mode._set_clock())

        self.assertEqual(
            emulator.press_button.call_args_list,
            [unittest.mock.call("A"), unittest.mock.call("Up"), unittest.mock.call("A")],
        )

    def test_clock_selection_uses_directional_input_until_target_time(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = object.__new__(EmeraldOpeningMode)
        mode._clock_target = (10, 0)
        mode._last_clock_task = None
        mode._clock_a_sent = False
        mode._clock_confirm_yes_prepared = False
        emulator = unittest.mock.Mock()
        task = types.SimpleNamespace(
            data_value=lambda index: {0: 0, 2: 9, 3: 59}[index],
        )
        with (
            patch("modules.modes.opening._active_clock_task", return_value="Task_SetClock_HandleInput"),
            patch("modules.modes.opening.get_task", return_value=task),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            list(mode._set_clock())

        emulator.hold_button.assert_called_once_with("Right")
        emulator.press_button.assert_not_called()

    def test_clock_direction_remains_held_across_input_frames(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = object.__new__(EmeraldOpeningMode)
        mode._clock_target = (10, 0)
        mode._last_clock_task = None
        mode._clock_a_sent = False
        mode._clock_confirm_yes_prepared = False
        emulator = unittest.mock.Mock()
        state = {0: 0, 2: 9, 3: 59}
        task = types.SimpleNamespace(data_value=state.__getitem__)
        with (
            patch("modules.modes.opening._active_clock_task", return_value="Task_SetClock_HandleInput"),
            patch("modules.modes.opening.get_task", return_value=task),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            list(mode._set_clock())
            list(mode._set_clock())
            state[2], state[3] = mode._clock_target
            list(mode._set_clock())

        emulator.hold_button.assert_called_once_with("Right")
        emulator.release_button.assert_called_once_with("Right")
        emulator.press_button.assert_called_once_with("A")

    def test_clock_confirmed_and_exit_tasks_require_no_additional_input(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = EmeraldOpeningMode()
        emulator = unittest.mock.Mock()
        with (
            patch(
                "modules.modes.opening._active_clock_task",
                side_effect=[
                    "Task_SetClock_Confirmed",
                    "Task_SetClock_Exit",
                ],
            ),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            list(mode._set_clock())
            list(mode._set_clock())

        emulator.press_button.assert_not_called()

    def test_clock_setting_completion_returns_to_house_progression(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.CLOCK_SETTING
        mode._clock_interaction_started = True
        with (
            patch("modules.modes.opening._clock_task_active", return_value=False),
            patch.object(mode, "_set_clock", return_value=iter(())),
            patch("modules.modes.opening.get_event_flag", return_value=True),
        ):
            list(mode._advance_phase(OpeningSequenceState.CLOCK_SETTING))

        self.assertIs(mode.phase, OpeningSequenceState.PLAYER_HOUSE_1F)

    def test_clock_setting_waits_for_rom_exit_after_completion_flag(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.CLOCK_SETTING
        mode._clock_interaction_started = True
        with (
            patch("modules.modes.opening._clock_task_active", return_value=True),
            patch.object(mode, "_set_clock", return_value=iter(())),
            patch("modules.modes.opening.get_event_flag", return_value=True),
        ):
            list(mode._advance_phase(OpeningSequenceState.CLOCK_SETTING))

        self.assertIs(mode.phase, OpeningSequenceState.CLOCK_SETTING)

    def test_clock_setting_phase_does_not_resume_staircase_navigation(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.CLOCK_SETTING
        mode._clock_interaction_started = True
        with (
            patch("modules.modes.opening._clock_task_active", return_value=False),
            patch("modules.modes.opening._scripted_input_waiting", return_value=False),
            patch.object(mode, "_set_clock", return_value=iter(())) as set_clock,
            patch("modules.modes.opening.get_event_flag", return_value=False),
            patch("modules.modes.opening.navigate_to") as navigate,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        set_clock.assert_called_once_with()
        navigate.assert_not_called()
        self.assertNotEqual(mode._last_truck_decision, "navigate: ROM-defined staircase warp to player's house 2F")

    def test_clock_popup_advances_before_clock_task_starts(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.CLOCK_SETTING
        mode._clock_interaction_started = True
        with (
            patch("modules.modes.opening._clock_task_active", return_value=False),
            patch("modules.modes.opening._scripted_input_waiting", return_value=True),
            patch("modules.modes.opening._advance_scripted_input", return_value=iter(())) as advance,
            patch.object(mode, "_set_clock", return_value=iter(())) as set_clock,
        ):
            list(mode._advance_phase(OpeningSequenceState.PLAYER_HOUSE_2F))

        advance.assert_called_once_with()
        set_clock.assert_not_called()

    def test_wall_clock_target_uses_rom_known_fallback_for_brendan_house(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import _wall_clock_interaction

        with patch(
            "modules.modes.opening.get_map_data_for_current_position",
            side_effect=AssertionError("clock interaction must not resolve ROM events"),
        ):
            self.assertEqual(
                _wall_clock_interaction(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value, "male"),
                ((5, 2), "Up"),
            )

    def test_system_time_converts_local_24_hour_time_to_emerald_clock(self):
        from datetime import datetime

        from modules.config.schemas_v1 import WallClockTimeMode
        from modules.modes.opening import _emerald_clock_time

        cases = {
            (0, 5): (0, 5),
            (11, 12): (11, 12),
            (12, 34): (12, 34),
            (16, 32): (16, 32),
            (23, 59): (23, 59),
        }
        for (hour, minute), expected in cases.items():
            with self.subTest(hour=hour, minute=minute):
                self.assertEqual(
                    _emerald_clock_time(
                        WallClockTimeMode.SYSTEM_TIME,
                        now=datetime(2026, 1, 1, hour, minute),
                    ),
                    expected,
                )

    def test_random_clock_time_uses_injected_rng_boundaries(self):
        from modules.config.schemas_v1 import WallClockTimeMode
        from modules.modes.opening import _emerald_clock_time

        class BoundaryRng:
            def __init__(self):
                self.calls = []

            def randint(self, start, end):
                self.calls.append((start, end))
                return start if len(self.calls) == 1 else end

        rng = BoundaryRng()
        self.assertEqual(_emerald_clock_time(WallClockTimeMode.RANDOM, rng=rng), (0, 59))
        self.assertEqual(rng.calls, [(0, 23), (0, 59)])

    def test_system_time_target_reaches_clock_controller(self):
        from datetime import datetime

        from modules.config.schemas_v1 import WallClockTimeMode
        from modules.modes.opening import EmeraldOpeningMode

        mode = object.__new__(EmeraldOpeningMode)
        mode._clock_target = None
        mode._rng = None
        mode._last_clock_task = None
        mode._clock_a_sent = False
        mode._clock_confirm_yes_prepared = False
        emulator = unittest.mock.Mock()
        task = types.SimpleNamespace(data_value=lambda index: {0: 0, 2: 3, 3: 27}[index])
        with (
            patch("modules.modes.opening._clock_time_mode", return_value=WallClockTimeMode.SYSTEM_TIME),
            patch("modules.modes.opening.datetime") as clock,
            patch("modules.modes.opening._active_clock_task", return_value="Task_SetClock_HandleInput"),
            patch("modules.modes.opening.get_task", return_value=task),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            clock.now.return_value = datetime(2026, 1, 1, 16, 32)
            list(mode._set_clock())

        self.assertEqual(mode._clock_target, (16, 32))
        emulator.hold_button.assert_called_once_with("Left")
        emulator.press_button.assert_not_called()

    def test_random_target_reaches_clock_controller(self):
        from modules.config.schemas_v1 import WallClockTimeMode
        from modules.modes.opening import EmeraldOpeningMode

        class BoundaryRng:
            def randint(self, start, end):
                return end

        mode = object.__new__(EmeraldOpeningMode)
        mode._clock_target = None
        mode._rng = BoundaryRng()
        mode._last_clock_task = None
        mode._clock_a_sent = False
        mode._clock_confirm_yes_prepared = False
        emulator = unittest.mock.Mock()
        task = types.SimpleNamespace(data_value=lambda index: {0: 0, 2: 23, 3: 59}[index])
        with (
            patch("modules.modes.opening._clock_time_mode", return_value=WallClockTimeMode.RANDOM),
            patch("modules.modes.opening._active_clock_task", return_value="Task_SetClock_HandleInput"),
            patch("modules.modes.opening.get_task", return_value=task),
            patch.object(__import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator),
        ):
            list(mode._set_clock())

        self.assertEqual(mode._clock_target, (23, 59))
        emulator.press_button.assert_called_once_with("A")

    def test_clock_selection_uses_shortest_24_hour_direction(self):
        from modules.modes.opening import EmeraldOpeningMode

        cases = (
            ((10, 0), (16, 32), "Right"),
            ((10, 0), (4, 32), "Left"),
            ((23, 50), (0, 10), "Right"),
            ((0, 10), (23, 50), "Left"),
        )
        for current, target, first_button in cases:
            with self.subTest(current=current, target=target):
                mode = object.__new__(EmeraldOpeningMode)
                mode._clock_target = target
                mode._last_clock_task = None
                mode._clock_a_sent = False
                mode._clock_confirm_yes_prepared = False
                emulator = unittest.mock.Mock()
                state = {0: 0, 2: current[0], 3: current[1]}
                task = types.SimpleNamespace(data_value=state.__getitem__)
                with (
                    patch("modules.modes.opening._active_clock_task", return_value="Task_SetClock_HandleInput"),
                    patch("modules.modes.opening.get_task", return_value=task),
                    patch.object(
                        __import__("modules.modes.opening", fromlist=["context"]).context, "emulator", emulator
                    ),
                ):
                    list(mode._set_clock())
                    self.assertEqual(emulator.hold_button.call_args.args, (first_button,))

                    state[2], state[3] = target
                    list(mode._set_clock())

                self.assertEqual(emulator.press_button.call_args_list[-1].args, ("A",))

    def test_unknown_protagonist_house_does_not_reuse_brendan_clock_target(self):
        from modules.modes.opening import _wall_clock_interaction

        with patch("modules.modes.opening.get_map_data_for_current_position", return_value=None):
            self.assertIsNone(_wall_clock_interaction((1, 3)))

    def test_may_house_clock_target_comes_from_its_map_event_data(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import _wall_clock_interaction

        clock = types.SimpleNamespace(
            kind="Script",
            script_symbol="LittlerootTown_MaysHouse_2F_EventScript_WallClock",
            local_coordinates=(3, 1),
            player_facing_direction="Up",
        )
        with patch(
            "modules.modes.opening.get_map_data_for_current_position",
            return_value=types.SimpleNamespace(bg_events=[clock]),
        ):
            self.assertEqual(
                _wall_clock_interaction(MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F.value, "female"),
                ((3, 2), "Up"),
            )

    def test_female_clock_controller_resolves_target_without_wall_clock_symbol(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import EmeraldOpeningMode
        from modules.start_game import PlayerGender

        mode = EmeraldOpeningMode.__new__(EmeraldOpeningMode)
        mode._resolved_player_gender = PlayerGender.FEMALE
        mode._last_truck_decision = None
        mode._clock_interaction_started = False
        mode._last_clock_task = None
        mode._clock_a_sent = False
        mode._clock_confirm_yes_prepared = False
        opening_context = types.SimpleNamespace(
            emulator=types.SimpleNamespace(press_button=unittest.mock.Mock()),
            debug=False,
        )
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening._current_map_id", return_value=MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F.value),
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                return_value=types.SimpleNamespace(bg_events=[]),
            ),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())) as face,
            patch("modules.modes.opening.context", opening_context),
        ):
            list(mode._start_clock_interaction())

        navigate.assert_called_once_with((1, 3), (3, 2), avoid_scripted_events=False)
        face.assert_called_once_with("Up")
        opening_context.emulator.press_button.assert_called_once_with("A")

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

        emulator.press_button.assert_not_called()

    def test_scripted_input_preserves_field_message_lifecycle_at_native_wait(self):
        from modules.modes.opening import EmeraldOpeningCapability, _advance_scripted_input

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            debug=False,
            debug_trace=False,
            rom=types.SimpleNamespace(is_emerald=True),
            emulator=emulator,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(
                EmeraldOpeningCapability,
                "_dialogue_state_snapshot",
                return_value=(
                    False,
                    None,
                    True,
                    "WaitForAorBPress",
                    "Std_MsgboxDefault",
                    True,
                    True,
                    True,
                    None,
                    (),
                    None,
                    None,
                ),
            ),
            patch("modules.modes.opening.is_emerald_field_dialogue_advanceable", return_value=True) as advanceable,
        ):
            list(_advance_scripted_input(field_message_lifecycle_active=True))

        self.assertTrue(advanceable.call_args.kwargs["field_message_lifecycle_active"])
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
        mode.phase = OpeningSequenceState.TRUCK
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
            generator = _enter_player_name("TEST")
            next(generator)
            with self.assertRaises(StopIteration):
                next(generator)
            type_name.assert_called_once_with("TEST")

    def test_rival_pokeball_object_lookup_selects_may_script(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import _rival_pokeball_object_template

        rival = types.SimpleNamespace(
            local_id=15,
            local_coordinates=(5, 4),
            script_symbol="LittlerootTown_MaysHouse_2F_EventScript_RivalsPokeBall",
        )
        unrelated = types.SimpleNamespace(local_id=2, local_coordinates=(5, 4), script_symbol="Unrelated")
        with (
            patch(
                "modules.modes.opening.get_map_data_for_current_position",
                return_value=types.SimpleNamespace(
                    map_group_and_number=MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F.value,
                    objects=[unrelated, rival],
                ),
            ),
        ):
            self.assertIs(_rival_pokeball_object_template(), rival)

    def test_rival_pokeball_object_lookup_selects_brendan_script(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import _rival_pokeball_object_template

        rival = types.SimpleNamespace(
            local_id=15,
            local_coordinates=(3, 4),
            script_symbol="LittlerootTown_BrendansHouse_2F_EventScript_RivalsPokeBall",
        )
        with patch(
            "modules.modes.opening.get_map_data_for_current_position",
            return_value=types.SimpleNamespace(
                map_group_and_number=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value,
                objects=[
                    types.SimpleNamespace(script_symbol="LittlerootTown_MaysHouse_2F_EventScript_RivalsPokeBall"),
                    rival,
                ],
            ),
        ):
            self.assertIs(_rival_pokeball_object_template(), rival)

    def test_rival_pokeball_object_lookup_rejects_unrelated_objects(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import _rival_pokeball_object_template

        with patch(
            "modules.modes.opening.get_map_data_for_current_position",
            return_value=types.SimpleNamespace(
                map_group_and_number=MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F.value,
                objects=[
                    types.SimpleNamespace(local_id=15, script_symbol="LittlerootTown_MaysHouse_2F_EventScript_PC")
                ],
            ),
        ):
            self.assertIsNone(_rival_pokeball_object_template())

    def test_rival_pokeball_interaction_uses_object_coordinates(self):
        from modules.modes.opening import _rival_pokeball_interaction

        object_template = types.SimpleNamespace(local_coordinates=(5, 4), script_symbol="Rival")
        with (
            patch("modules.modes.opening._rival_pokeball_object_template", return_value=object_template),
            patch(
                "modules.modes.opening.get_player_avatar", return_value=types.SimpleNamespace(local_coordinates=(7, 2))
            ),
        ):
            self.assertEqual(_rival_pokeball_interaction(), ((5, 5), (5, 4)))

    def test_route101_starter_bag_target_comes_from_scripted_object(self):
        from modules.map_data import MapRSE
        from modules.modes.opening import _starter_bag_interaction

        bag = types.SimpleNamespace(
            local_coordinates=(7, 14),
            script_symbol="Route101_EventScript_BirchsBag",
        )
        with patch(
            "modules.modes.opening.get_map_data_for_current_position",
            return_value=types.SimpleNamespace(
                map_group_and_number=MapRSE.ROUTE101.value,
                objects=[types.SimpleNamespace(local_coordinates=(10, 6), script_symbol="Unrelated"), bag],
            ),
        ):
            self.assertEqual(_starter_bag_interaction(), ((7, 15), (7, 14)))

    def test_route101_handoff_navigates_below_bag_and_hands_off(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState, consume_starter_handoff

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.ROUTE_101
        emulator = unittest.mock.Mock()
        opening_context = types.SimpleNamespace(bot_mode="Start New Game", emulator=emulator, debug=False)
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening._route101_ready_for_navigation", return_value=True),
            patch("modules.modes.opening._starter_bag_interaction", return_value=((7, 15), (7, 14))),
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())) as face,
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.get_player_avatar",
                return_value=types.SimpleNamespace(
                    local_coordinates=(7, 15),
                ),
            ),
            patch("modules.modes.opening.context", opening_context),
        ):
            handoff = mode._advance_phase(OpeningSequenceState.ROUTE_101)
            with self.assertRaises(StopIteration) as stopped:
                next(handoff)
            self.assertTrue(stopped.exception.value)

        navigate.assert_called_once_with(
            MapRSE.ROUTE101,
            (7, 15),
            avoid_encounters=False,
            avoid_scripted_events=False,
            expecting_script=True,
        )
        face.assert_called_once_with((7, 14))
        self.assertIs(mode.phase, OpeningSequenceState.ROUTE_101)
        self.assertEqual(opening_context.bot_mode, "Starters")
        self.assertTrue(consume_starter_handoff())

    def test_route101_handoff_completes_opening_mode_generator(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.ROUTE_101
        opening_context = types.SimpleNamespace(
            bot_mode="Start New Game",
            emulator=unittest.mock.Mock(),
            debug=False,
            rom=types.SimpleNamespace(is_emerald=True),
        )
        with (
            patch.object(mode, "_report_diagnostics"),
            patch.object(mode, "_report_dialogue_detection"),
            patch.object(mode, "_report_route101_lifecycle"),
            patch.object(mode, "_report_route101_runtime"),
            patch.object(mode, "_report_phase_dispatch"),
            patch.object(mode, "_advance_startup_dialogue", return_value=iter(())),
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening.get_opening_sequence_state", return_value=OpeningSequenceState.ROUTE_101),
            patch("modules.modes.opening._route101_ready_for_navigation", return_value=True),
            patch("modules.modes.opening._starter_bag_interaction", return_value=((7, 15), (7, 14))),
            patch("modules.modes.opening.navigate_to", return_value=iter(())),
            patch("modules.modes.opening.ensure_facing_direction", return_value=iter(())),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.get_player_avatar",
                return_value=types.SimpleNamespace(local_coordinates=(7, 15)),
            ),
            patch("modules.modes.opening.context", opening_context),
        ):
            with self.assertRaises(StopIteration):
                next(mode.run())

        self.assertEqual(opening_context.bot_mode, "Starters")
        opening_context.emulator.press_button.assert_not_called()

    def test_route101_waits_for_printing_arrival_script_before_navigation(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.ROUTE_101
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening._route101_ready_for_navigation", return_value=False),
            patch("modules.modes.opening.navigate_to") as navigate,
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
        ):
            list(mode._advance_phase(OpeningSequenceState.ROUTE_101))

        navigate.assert_not_called()

    def test_route101_phase_navigates_from_littleroot_before_checking_arrival_state(self):
        from modules.map_data import MapRSE
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.ROUTE_101
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening._route101_ready_for_navigation") as ready,
            patch("modules.modes.opening.navigate_to", return_value=iter(())) as navigate,
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
        ):
            list(mode._advance_phase(OpeningSequenceState.LITTLEROOT_TOWN))

        ready.assert_not_called()
        navigate.assert_called_once_with(
            MapRSE.ROUTE101,
            (10, 6),
            avoid_encounters=False,
            avoid_scripted_events=False,
            expecting_script=True,
        )

    def test_route101_readiness_rejects_active_printing_field_message(self):
        from modules.memory import GameState
        from modules.modes.opening import _route101_ready_for_navigation

        with (
            patch("modules.modes.opening.get_event_var", return_value=1),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_waiting_for_input", return_value=False),
            patch("modules.modes.opening.task_is_active", side_effect=lambda task: task == "Task_DrawFieldMessage"),
            patch(
                "modules.modes.opening.get_global_script_context",
                return_value=types.SimpleNamespace(
                    is_active=True,
                ),
            ),
        ):
            self.assertFalse(_route101_ready_for_navigation())

    def test_route101_readiness_uses_rescue_completion_marker_not_stale_message_task(self):
        from modules.memory import GameState
        from modules.modes.opening import _route101_ready_for_navigation

        with (
            patch("modules.modes.opening.get_event_var", return_value=2),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.task_is_active", return_value=True),
        ):
            self.assertTrue(_route101_ready_for_navigation())

    def test_route101_observation_stage_labels_only_proven_raw_states(self):
        from modules.modes.opening import _route101_observation_stage

        def state(*, script=False, visible=False, controllable=False):
            values = [False, None, script, None, None, False, controllable, visible, None, (), None, None]
            return tuple(values)

        self.assertEqual(
            _route101_observation_stage(0, state()),
            "A_arrival_before_rescue",
        )
        self.assertEqual(
            _route101_observation_stage(1, state(script=True)),
            "B_or_D_rescue_script_no_message",
        )
        self.assertEqual(
            _route101_observation_stage(1, state(script=True, visible=True)),
            "C_message_visible",
        )
        self.assertEqual(
            _route101_observation_stage(2, state(controllable=True)),
            "E_or_F_rescue_complete_control_restored",
        )

    def test_route101_readiness_waits_until_rescue_completion_marker(self):
        from modules.memory import GameState
        from modules.modes.opening import _route101_ready_for_navigation

        with (
            patch("modules.modes.opening.get_event_var", return_value=1),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.task_is_active", return_value=False),
        ):
            self.assertFalse(_route101_ready_for_navigation())

    def test_route101_readiness_ignores_stale_native_wait_after_rescue(self):
        from modules.memory import GameState
        from modules.modes.opening import _route101_ready_for_navigation

        # The ROM has released control and removed the field-message task, but
        # the global context still reports WaitForAorBPress for this frame.
        with (
            patch("modules.modes.opening.get_event_var", return_value=2),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_waiting_for_input", return_value=True),
            patch("modules.modes.opening.task_is_active", return_value=False),
            patch(
                "modules.modes.opening.get_global_script_context",
                return_value=types.SimpleNamespace(
                    is_active=True,
                    native_function_name="WaitForAorBPress",
                ),
            ),
        ):
            self.assertTrue(_route101_ready_for_navigation())

    def test_route101_arrival_dialogue_still_owns_input_before_completion_marker(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.ROUTE_101
        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        with (
            patch("modules.modes.opening.get_event_var", return_value=1),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.player_avatar_is_controllable", return_value=True),
            patch("modules.modes.opening.task_is_active", side_effect=lambda task: task == "Task_DrawFieldMessage"),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
            patch(
                "modules.modes.opening.context",
                types.SimpleNamespace(
                    rom=types.SimpleNamespace(is_emerald=True),
                    emulator=emulator,
                    debug=False,
                ),
            ),
        ):
            action = list(mode._advance_startup_dialogue(OpeningSequenceState.ROUTE_101))

        self.assertEqual(action, [True])
        emulator.press_button.assert_called_once_with("B")

    def test_route101_does_not_face_or_save_if_navigation_returns_early(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        mode = EmeraldOpeningMode()
        mode.phase = OpeningSequenceState.ROUTE_101
        with (
            patch.object(mode, "_can_navigate", return_value=True),
            patch("modules.modes.opening._route101_ready_for_navigation", return_value=True),
            patch("modules.modes.opening._starter_bag_interaction", return_value=((7, 15), (7, 14))),
            patch("modules.modes.opening.navigate_to", return_value=iter(())),
            patch("modules.modes.opening.ensure_facing_direction") as face,
            patch(
                "modules.modes.opening.get_player_avatar",
                return_value=types.SimpleNamespace(
                    local_coordinates=(11, 19),
                ),
            ),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
        ):
            list(mode._advance_phase(OpeningSequenceState.ROUTE_101))

        face.assert_not_called()

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
            mocked_context.config = types.SimpleNamespace(
                start_game=types.SimpleNamespace(player_name="gibberish", player_gender="random")
            )
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

    def test_birch_gender_task_precedes_main_menu_new_game_branch(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState
        from modules.start_game import PlayerGender

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            bot_mode="Start New Game",
            debug=False,
            config=types.SimpleNamespace(
                start_game=types.SimpleNamespace(player_name="gibberish", player_gender="female")
            ),
        )
        choose_gender_task = types.SimpleNamespace(symbol="Task_NewGameBirchSpeech_ChooseGender")
        with (
            patch("modules.modes.opening.context", opening_context),
            patch(
                "modules.modes.opening.resolve_start_game_initialization",
                return_value=types.SimpleNamespace(gender=PlayerGender.FEMALE, name="MAY"),
            ),
            patch(
                "modules.modes.opening.get_opening_sequence_state",
                return_value=OpeningSequenceState.MAIN_MENU,
            ),
            patch("modules.modes.opening.get_tasks", return_value=[choose_gender_task]),
            patch(
                "modules.modes.opening.get_task",
                return_value=types.SimpleNamespace(data_value=lambda index: 0),
            ),
        ):
            mode = EmeraldOpeningMode()
            next(mode.run())

        emulator.press_button.assert_called_once_with("Down")

    def test_initial_settings_enter_options_set_fast_and_return_to_main_menu(self):
        from modules.modes.opening import EmeraldOpeningMode
        from modules.memory import GameState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            debug=False,
        )
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(
                mode,
                "_message_speed_observation",
                side_effect=[
                    (1, 1, None, "Task_OptionMenuProcessInput.data[1]"),
                    (1, 1, None, "Task_OptionMenuProcessInput.data[1]"),
                    (2, 1, None, "Task_OptionMenuProcessInput.data[1]"),
                ],
            ),
            patch("modules.modes.opening.task_is_active", side_effect=[True, False, True, True]),
            patch("modules.modes.opening.get_game_state", return_value=GameState.MAIN_MENU),
        ):
            list(mode._configure_initial_game_settings())

        self.assertEqual(
            emulator.press_button.call_args_list,
            [
                unittest.mock.call("Down"),
                unittest.mock.call("A"),
                unittest.mock.call("Right"),
                unittest.mock.call("B"),
            ],
        )
        self.assertTrue(mode._initial_options_entered)
        self.assertTrue(mode._initial_options_cursor_positioned)

    def test_initial_settings_skip_options_when_fast_is_already_selected(self):
        from modules.modes.opening import EmeraldOpeningMode

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            debug=False,
        )
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(
                mode,
                "_message_speed_observation",
                return_value=(2, 1, None, "Task_OptionMenuProcessInput.data[1]"),
            ),
        ):
            list(mode._configure_initial_game_settings())

        emulator.press_button.assert_not_called()
        self.assertFalse(mode._initial_options_entered)

    def test_message_speed_reads_emerald_save_block2_option(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = EmeraldOpeningMode()
        with patch("modules.modes.opening.get_save_block", return_value=b"\x02\x00") as read_save_block:
            self.assertEqual(mode._message_speed(), 2)
        read_save_block.assert_called_once_with(2, offset=0x14, size=2)

    def test_options_task_data_is_authoritative_while_menu_is_open(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = EmeraldOpeningMode()
        task = types.SimpleNamespace(data_value=lambda index: 2)
        with (
            patch("modules.modes.opening.get_task", return_value=task),
            patch("modules.modes.opening.get_save_block", return_value=b"\x01\x00"),
            patch("modules.modes.opening.read_symbol", return_value=b"\x00\x00\x00\x02"),
        ):
            self.assertEqual(
                mode._message_speed_observation(),
                (2, 1, 0x02000000, "Task_OptionMenuProcessInput.data[1]"),
            )

    def test_runtime_save_block_value_change_from_medium_to_fast_completes_configuration(self):
        from modules.modes.opening import EmeraldOpeningMode
        from modules.memory import GameState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            debug=False,
        )
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch(
                "modules.modes.opening.get_save_block",
                side_effect=[b"\x01\x00", b"\x01\x00", b"\x02\x00"],
            ),
            patch("modules.modes.opening.read_symbol", return_value=b"\x00\x00\x00\x02"),
            patch("modules.modes.opening.task_is_active", side_effect=[True, False, True, True]),
            patch("modules.modes.opening.get_game_state", return_value=GameState.MAIN_MENU),
        ):
            list(mode._configure_initial_game_settings())

        self.assertEqual(
            emulator.press_button.call_args_list,
            [
                unittest.mock.call("Down"),
                unittest.mock.call("A"),
                unittest.mock.call("Right"),
                unittest.mock.call("B"),
            ],
        )

    def test_text_speed_configuration_diagnostics_are_state_change_deduplicated(self):
        from modules.modes.opening import EmeraldOpeningMode

        mode = EmeraldOpeningMode()
        with patch("modules.modes.opening.diagnostic_print") as diagnostic:
            observation = (1, 1, 0x1234, "Task_OptionMenuProcessInput.data[1]")
            mode._report_text_speed_configuration(observation, "press Right: advance text speed")
            mode._report_text_speed_configuration(observation, "press Right: advance text speed")
            mode._report_text_speed_configuration(
                (2, 1, 0x1234, "Task_OptionMenuProcessInput.data[1]"),
                "complete: FAST observed",
            )

        self.assertEqual(diagnostic.call_count, 2)

    def test_initial_settings_advance_from_slow_until_fast_without_wrapping(self):
        from modules.modes.opening import EmeraldOpeningMode
        from modules.memory import GameState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            debug=False,
        )
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(
                mode,
                "_message_speed_observation",
                side_effect=[
                    (0, 0, None, "Task_OptionMenuProcessInput.data[1]"),
                    (0, 0, None, "Task_OptionMenuProcessInput.data[1]"),
                    (1, 0, None, "Task_OptionMenuProcessInput.data[1]"),
                    (2, 0, None, "Task_OptionMenuProcessInput.data[1]"),
                ],
            ),
            patch("modules.modes.opening.task_is_active", side_effect=[True, True, True, True]),
            patch("modules.modes.opening.get_game_state", return_value=GameState.MAIN_MENU),
        ):
            list(mode._configure_initial_game_settings())

        self.assertEqual(
            emulator.press_button.call_args_list,
            [
                unittest.mock.call("Down"),
                unittest.mock.call("A"),
                unittest.mock.call("Right"),
                unittest.mock.call("Right"),
                unittest.mock.call("B"),
            ],
        )

    def test_main_menu_repositions_to_new_game_after_options(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState
        from modules.memory import GameState

        emulator = types.SimpleNamespace(press_button=unittest.mock.Mock())
        opening_context = types.SimpleNamespace(
            emulator=emulator,
            rom=types.SimpleNamespace(is_emerald=True),
            debug=False,
            bot_mode="Start New Game",
        )
        mode = EmeraldOpeningMode()
        mode._initial_settings_configured = True
        mode._initial_options_entered = True
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_opening_sequence_state", return_value=OpeningSequenceState.MAIN_MENU),
            patch("modules.modes.opening.get_opening_diagnostics", return_value=None),
            patch("modules.modes.opening.get_game_state", return_value=GameState.MAIN_MENU),
            patch("modules.modes.opening.task_is_active", return_value=True),
        ):
            generator = mode.run()
            next(generator)
            next(generator)

        self.assertEqual(
            emulator.press_button.call_args_list,
            [
                unittest.mock.call("Up"),
                unittest.mock.call("A"),
            ],
        )
        self.assertTrue(mode._initial_menu_repositioned)
