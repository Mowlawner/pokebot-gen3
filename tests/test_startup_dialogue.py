import types
import unittest
from unittest.mock import patch


class TestStartupDialogue(unittest.TestCase):
    def _mode_context(self):
        return types.SimpleNamespace(
            rom=types.SimpleNamespace(is_emerald=True),
            emulator=types.SimpleNamespace(press_button=unittest.mock.Mock()),
            debug=False,
        )

    def test_normal_field_dialogue_sends_b(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            list(EmeraldOpeningMode()._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_same_field_wait_state_can_advance_multiple_pages(self):
        """A repeated native/task snapshot can represent another text page."""
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        state = (
            True,
            2,
            True,
            "IsFieldMessageBoxHidden",
            "Std_MsgboxDefault",
            True,
            True,
            True,
            "Clear",
            ("Task_DrawFieldMessage",),
            134841149,
            136284591,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", return_value=state),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.BIRCH_HOUSE_1F)),
                [True],
            )
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.BIRCH_HOUSE_1F)),
                [True],
            )

        self.assertEqual(
            opening_context.emulator.press_button.call_args_list,
            [
                unittest.mock.call("B"),
                unittest.mock.call("B"),
            ],
        )

    def test_multi_page_dialogue_advances_until_complete(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        first_page = (
            True,
            2,
            True,
            "IsFieldMessageBoxHidden",
            "Std_MsgboxDefault",
            True,
            True,
            True,
            "Clear",
            (),
            1,
            10,
        )
        second_page = (
            True,
            3,
            True,
            "IsFieldMessageBoxHidden",
            "Std_MsgboxDefault",
            True,
            True,
            True,
            "Clear",
            (),
            1,
            11,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.is_field_message_waiting_for_input",
                side_effect=[True, True, False],
            ),
            patch.object(
                EmeraldOpeningMode,
                "_dialogue_state_snapshot",
                side_effect=[first_page, second_page, second_page, second_page],
            ),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
        self.assertEqual(opening_context.emulator.press_button.call_count, 2)

    def test_final_message_task_still_active_waits_without_advancing_phase(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_field_message_waiting_for_input", side_effect=[True, False]),
            patch(
                "modules.modes.opening.get_task",
                side_effect=[types.SimpleNamespace(data_value=lambda _: 7), None, None],
            ),
            patch(
                "modules.modes.opening.get_global_script_context",
                return_value=types.SimpleNamespace(
                    is_active=True,
                    native_function_name="IsFieldMessageBoxHidden",
                    script_function_name="Std_MsgboxDefault",
                ),
            ),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
        self.assertEqual(result, [True])
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_final_message_dismissal_allows_arrival_script_to_resume(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_field_message_waiting_for_input", side_effect=[True, False]),
            patch(
                "modules.modes.opening.get_task",
                side_effect=[types.SimpleNamespace(data_value=lambda _: 7), None, None],
            ),
            patch(
                "modules.modes.opening.get_global_script_context",
                return_value=types.SimpleNamespace(
                    is_active=False,
                    native_function_name="",
                    script_function_name="",
                ),
            ),
            patch(
                "modules.modes.opening.get_text_printer",
                return_value=types.SimpleNamespace(
                    active=False,
                    raw_state="End",
                ),
            ),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
        self.assertEqual(result, [])
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_task_ended_wait_for_a_or_b_is_still_active_dialogue(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        task = types.SimpleNamespace(data_value=lambda _: 2)
        script_context = types.SimpleNamespace(
            is_active=True,
            native_function_name="WaitForAorBPress",
            script_function_name="Std_MsgboxDefault",
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening._startup_dialogue_waiting", side_effect=[True, False, False, False]),
            patch("modules.modes.opening.is_waiting_for_input", return_value=True),
            patch("modules.modes.opening.get_task", side_effect=[task, None, None, None]),
            patch("modules.modes.opening.get_global_script_context", return_value=script_context),
        ):
            # The first call establishes the ordinary field-message phase.
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
            # The draw task has ended, but the script-native wait is still
            # part of the same dialogue and must receive input.
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN)),
                [True],
            )
            # A repeated native wait is still an actionable dialogue page.
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN)),
                [True],
            )
        self.assertEqual(opening_context.emulator.press_button.call_count, 3)

    def test_script_owned_standard_message_without_draw_task_sends_b(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        state = (
            False,
            None,
            True,
            "WaitForAorBPress",
            "Std_MsgboxDefault",
            True,
            True,
            False,
            "HandleCharacter",
            (),
            134851737,
            136778545,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", return_value=state),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=False),
        ):
            detected, reason = mode._dialogue_detection(OpeningSequenceState.ROUTE_101, state)
            self.assertTrue(detected)
            self.assertEqual(reason, "Std_MsgboxDefault is waiting for A/B input")
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.ROUTE_101))

        self.assertEqual(result, [True])
        opening_context.emulator.press_button.assert_called_once_with("B")
        self.assertTrue(mode._dialogue_active)

    def test_wait_for_a_or_b_without_standard_message_script_is_not_dialogue(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        state = (
            False,
            None,
            True,
            "WaitForAorBPress",
            "SomeOtherScript",
            True,
            True,
            False,
            "HandleCharacter",
            (),
            1,
            2,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", return_value=state),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=False),
        ):
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.ROUTE_101))

        self.assertEqual(result, [])
        opening_context.emulator.press_button.assert_not_called()

    def test_stale_native_wait_without_input_readiness_does_not_dismiss_script(self):
        """A scripted transition must not treat a stale native name as input."""
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        mode._dialogue_active = True
        state = (
            False,
            None,
            True,
            "WaitForAorBPress",
            "Std_MsgboxDefault",
            False,
            False,
            False,
            "HandleCharacter",
            (),
            1,
            10,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", return_value=state),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=False),
        ):
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.ROUTE_101))

        self.assertEqual(result, [])
        opening_context.emulator.press_button.assert_not_called()
        self.assertFalse(mode._dialogue_active)

    def test_task_ended_inactive_printer_wait_for_a_or_b_generates_a(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        mode._dialogue_active = True
        state = (
            False,
            None,
            True,
            "WaitForAorBPress",
            "Std_MsgboxDefault",
            True,
            False,
            False,
            "HandleCharacter",
            (),
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", return_value=state),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            result = list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F))
        self.assertEqual(result, [True])
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_repeated_script_wait_gets_another_dialogue_dismissal(self):
        """A final native wait remains actionable even with the same snapshot."""
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        opening_context.emulator.get_inputs = unittest.mock.Mock(side_effect=[0x1, 0x0])
        mode = EmeraldOpeningMode()
        mode._dialogue_active = True
        mode._dialogue_before_input = ("previous",)
        state = (
            False,
            None,
            True,
            "WaitForAorBPress",
            "Std_MsgboxDefault",
            True,
            False,
            False,
            "HandleCharacter",
            (),
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", return_value=state),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F)),
                [True],
            )
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F)),
                [True],
            )

        self.assertEqual(
            opening_context.emulator.press_button.call_args_list,
            [
                unittest.mock.call("B"),
                unittest.mock.call("B"),
            ],
        )

    def test_new_script_wait_after_printer_state_gets_input(self):
        """A printer transition must not suppress the newly exposed native wait."""
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        mode._dialogue_active = True
        mode._dialogue_wait_input_state = ("stale native wait",)
        printer_state = (
            True,
            2,
            True,
            "IsFieldMessageBoxHidden",
            "Std_MsgboxDefault",
            True,
            False,
            True,
            "Clear",
            (),
            134851737,
            136778545,
        )
        native_wait_state = (
            False,
            None,
            True,
            "WaitForAorBPress",
            "Std_MsgboxDefault",
            True,
            False,
            False,
            "HandleCharacter",
            (),
            134851737,
            136778545,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(
                EmeraldOpeningMode,
                "_dialogue_state_snapshot",
                side_effect=[printer_state, native_wait_state, native_wait_state],
            ),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F))
            list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F))

        self.assertEqual(opening_context.emulator.press_button.call_count, 2)

    def test_unavailable_dialogue_context_is_retryable(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        with patch(
            "modules.modes.opening.is_field_message_waiting_for_input",
            side_effect=RuntimeError("transition"),
        ):
            # The production predicate catches emulator-state failures.  This
            # verifies the controller retries without input or mode failure.
            list(EmeraldOpeningMode()._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))

    def test_non_dialogue_activity_does_not_send_a(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening._startup_dialogue_waiting", return_value=False),
        ):
            list(EmeraldOpeningMode()._advance_startup_dialogue(OpeningSequenceState.SCRIPTED_INTRO))
        opening_context.emulator.press_button.assert_not_called()

    def test_specialized_ui_states_are_excluded(self):
        from modules.modes.opening import OpeningSequenceState, _startup_dialogue_waiting

        with patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True):
            for state in (
                OpeningSequenceState.CLOCK_SETTING,
                OpeningSequenceState.PLAYER_NAMING,
                OpeningSequenceState.STARTER_SELECTION,
            ):
                self.assertFalse(_startup_dialogue_waiting(state))

    def test_dialogue_after_map_transition_is_phase_independent(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.POST_CLOCK_TOWN))
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_player_house_intro_dialogue_is_handled_before_phase_navigation(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            list(EmeraldOpeningMode()._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F))
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_dialogue_remains_cross_phase_when_opening_state_changes(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.opening.is_field_message_waiting_for_input",
                side_effect=[True, True],
            ),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
            list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_1F))
        self.assertEqual(opening_context.emulator.press_button.call_count, 2)

    def test_wait_for_movement_finish_is_not_dialogue(self):
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening._startup_dialogue_waiting", return_value=False),
            patch.object(
                EmeraldOpeningMode,
                "_dialogue_state_snapshot",
                return_value=(
                    False,
                    None,
                    True,
                    "WaitForMovementFinish",
                    "OpeningScript",
                    False,
                    True,
                    False,
                    "HandleCharacter",
                    (),
                ),
            ),
        ):
            result = list(EmeraldOpeningMode()._advance_startup_dialogue(OpeningSequenceState.LITTLEROOT_TOWN))
        self.assertEqual(result, [])
        opening_context.emulator.press_button.assert_not_called()

    def test_mom_after_clock_regression_uses_general_handler(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.opening.is_field_message_waiting_for_input", return_value=True),
        ):
            list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_2F))
        opening_context.emulator.press_button.assert_called_once_with("B")

    def test_final_dialogue_dismissal_does_not_restart_conversation(self):
        from modules.memory import GameState
        from modules.modes.opening import EmeraldOpeningMode, OpeningSequenceState

        opening_context = self._mode_context()
        mode = EmeraldOpeningMode()
        active = (
            True,
            2,
            True,
            "IsFieldMessageBoxHidden",
            "Std_MsgboxDefault",
            True,
            False,
            True,
            "Clear",
            (),
            134149,
            136778544,
        )
        ended = (False, None, False, "", "", False, True, False, "End", (), None, None)
        with (
            patch("modules.modes.opening.context", opening_context),
            patch("modules.modes.opening.get_game_state", return_value=GameState.OVERWORLD),
            patch.object(EmeraldOpeningMode, "_dialogue_state_snapshot", side_effect=[active, ended, ended, ended]),
            patch("modules.modes.opening.is_field_message_waiting_for_input", side_effect=[True, False, False]),
        ):
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_2F)),
                [True],
            )
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_2F)),
                [],
            )
            self.assertEqual(
                list(mode._advance_startup_dialogue(OpeningSequenceState.PLAYER_HOUSE_2F)),
                [],
            )
        opening_context.emulator.press_button.assert_called_once_with("B")
