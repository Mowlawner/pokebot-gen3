import types
import unittest
from unittest.mock import patch

from modules.interaction_state import observe_interaction
from modules.memory import GameState


class TestIsWaitingForInput(unittest.TestCase):
    def test_emerald_dialogue_predicate_accepts_native_message_handoff(self):
        from modules.tasks import is_emerald_field_dialogue_advanceable

        with patch("modules.tasks.is_field_message_waiting_for_input", return_value=False):
            self.assertTrue(
                is_emerald_field_dialogue_advanceable(
                    task_active=False,
                    script_active=True,
                    native_function_name="WaitForAorBPress",
                    script_function_name="Std_MsgboxDefault",
                    input_waiting=True,
                )
            )

    def test_emerald_dialogue_predicate_accepts_birch_speech_printer_pause(self):
        from modules.tasks import is_emerald_field_dialogue_advanceable

        with patch("modules.tasks.is_field_message_waiting_for_input", return_value=False):
            self.assertTrue(
                is_emerald_field_dialogue_advanceable(
                    task_active=True,
                    task_name="Task_NewGameBirchSpeech_ThisIsAPokemon",
                    script_active=False,
                    native_function_name=None,
                    script_function_name=None,
                    input_waiting=True,
                )
            )

    def test_emerald_dialogue_predicate_rejects_printing_and_other_native_waits(self):
        from modules.tasks import is_emerald_field_dialogue_advanceable

        with patch("modules.tasks.is_field_message_waiting_for_input", return_value=False):
            for native, script, ready in (
                ("WaitForAorBPress", "Std_MsgboxDefault", False),
                ("WaitForMovementFinish", "Std_MsgboxDefault", True),
                ("WaitForAorBPress", "SomeScript", True),
            ):
                self.assertFalse(
                    is_emerald_field_dialogue_advanceable(
                        task_active=True,
                        script_active=True,
                        native_function_name=native,
                        script_function_name=script,
                        input_waiting=ready,
                    )
                )

    def test_emerald_field_message_requires_task_and_input_wait(self):
        from modules.tasks import is_field_message_waiting_for_input

        with (
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.task_is_active", return_value=True),
            patch("modules.tasks.is_waiting_for_input", return_value=True),
        ):
            self.assertTrue(is_field_message_waiting_for_input())

        with (
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.task_is_active", return_value=False),
            patch("modules.tasks.is_waiting_for_input", return_value=True),
        ):
            self.assertFalse(is_field_message_waiting_for_input())

    def test_emerald_wait_for_a_or_b_with_active_printer_is_field_dialogue(self):
        from modules.tasks import is_field_message_waiting_for_input

        with (
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.task_is_active", return_value=False),
            patch("modules.tasks.is_waiting_for_input", return_value=True),
            patch(
                "modules.tasks.get_global_script_context",
                return_value=types.SimpleNamespace(
                    native_function_name="WaitForAorBPress",
                ),
            ),
            patch("modules.tasks.get_text_printer", return_value=types.SimpleNamespace(active=True)),
        ):
            self.assertTrue(is_field_message_waiting_for_input())

    def test_emerald_wait_for_a_or_b_after_draw_task_is_still_field_dialogue(self):
        from modules.tasks import is_field_message_waiting_for_input

        with (
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.task_is_active", return_value=False),
            patch("modules.tasks.is_waiting_for_input", return_value=True),
            patch(
                "modules.tasks.get_global_script_context",
                return_value=types.SimpleNamespace(
                    native_function_name="WaitForAorBPress",
                ),
            ),
            patch("modules.tasks.get_text_printer", return_value=types.SimpleNamespace(active=False)),
        ):
            self.assertTrue(is_field_message_waiting_for_input(field_message_lifecycle_active=True))

    def test_emerald_movement_wait_is_not_field_dialogue(self):
        from modules.tasks import is_field_message_waiting_for_input

        with (
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.task_is_active", return_value=False),
            patch("modules.tasks.is_waiting_for_input", return_value=False),
            patch(
                "modules.tasks.get_global_script_context",
                return_value=types.SimpleNamespace(
                    native_function_name="WaitForMovementFinish",
                ),
            ),
        ):
            self.assertFalse(is_field_message_waiting_for_input())

    def test_missing_global_script_context_is_not_waiting(self):
        from modules.tasks import is_waiting_for_input

        with (
            patch("modules.tasks.get_global_script_context", return_value=None),
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.read_symbol", return_value=bytes((0, 0))),
        ):
            self.assertFalse(is_waiting_for_input())

    def test_wait_for_a_or_b_native_context_is_waiting(self):
        from modules.tasks import is_waiting_for_input

        script_context = types.SimpleNamespace(native_function_name="WaitForAorBPress")
        with patch("modules.tasks.get_global_script_context", return_value=script_context):
            self.assertTrue(is_waiting_for_input())

    def test_other_native_context_uses_text_printer_detection(self):
        from modules.tasks import is_waiting_for_input

        script_context = types.SimpleNamespace(native_function_name="SomeOtherFunction")
        with (
            patch("modules.tasks.get_global_script_context", return_value=script_context),
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.read_symbol", return_value=bytes((1, 2))),
        ):
            self.assertTrue(is_waiting_for_input())

    def test_emerald_inactive_text_printer_returns_bool_false(self):
        from modules.tasks import is_waiting_for_input

        script_context = types.SimpleNamespace(native_function_name="SomeOtherFunction")
        with (
            patch("modules.tasks.get_global_script_context", return_value=script_context),
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=False))),
            patch("modules.tasks.read_symbol", return_value=bytes((0, 2))),
        ):
            waiting = is_waiting_for_input()

        self.assertIs(waiting, False)

    def test_rs_field_message_box_states_are_preserved(self):
        from modules.tasks import is_waiting_for_input

        script_context = types.SimpleNamespace(native_function_name="SomeOtherFunction")
        with (
            patch("modules.tasks.get_global_script_context", return_value=script_context),
            patch("modules.tasks.context", types.SimpleNamespace(rom=types.SimpleNamespace(is_rs=True))),
            patch("modules.tasks.task_is_active", return_value=True),
            patch("modules.tasks.read_symbol", return_value=bytes([8])),
        ):
            self.assertTrue(is_waiting_for_input())

    def test_interaction_observer_carries_field_dialogue_through_native_wait(self):
        import modules.interaction_state as interaction_state

        interaction_state._field_message_lifecycle_active = False
        interaction_state._field_message_advance_ready = False
        with (
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=False),
            patch("modules.interaction_state.task_is_active", side_effect=[True, False]),
            patch("modules.interaction_state.is_field_message_waiting_for_input", side_effect=[False, True]),
            patch(
                "modules.interaction_state.get_global_script_context",
                return_value=types.SimpleNamespace(native_function_name="WaitForAorBPress"),
            ),
        ):
            first = observe_interaction()
            second = observe_interaction()

        self.assertFalse(first.dialogue_waiting)
        self.assertTrue(second.dialogue_waiting)

    def test_interaction_observer_preserves_ordinary_dialogue(self):
        import modules.interaction_state as interaction_state

        interaction_state._field_message_lifecycle_active = False
        interaction_state._field_message_advance_ready = False
        with (
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=False),
            patch("modules.interaction_state.task_is_active", return_value=True),
            patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=True),
        ):
            observation = observe_interaction()

        self.assertTrue(observation.dialogue_waiting)

    def test_interaction_observer_marks_hidden_box_transition_as_dialogue_advance(self):
        import modules.interaction_state as interaction_state

        interaction_state._field_message_lifecycle_active = False
        interaction_state._field_message_advance_ready = False
        with (
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=False),
            patch("modules.interaction_state.task_is_active", return_value=True),
            patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=False),
            patch(
                "modules.interaction_state.get_global_script_context",
                return_value=types.SimpleNamespace(native_function_name="IsFieldMessageBoxHidden"),
            ),
        ):
            observation = observe_interaction()

        self.assertFalse(observation.dialogue_waiting)
        self.assertTrue(observation.field_message_lifecycle_active)
        self.assertTrue(observation.field_message_advance_ready)

    def test_active_hidden_box_is_dialogue_lifecycle_but_not_actionable(self):
        import modules.interaction_state as interaction_state

        interaction_state._field_message_lifecycle_active = True
        interaction_state._field_message_advance_ready = False
        with (
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=True),
            patch("modules.interaction_state.task_is_active", return_value=False),
            patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=True),
            patch(
                "modules.interaction_state.get_global_script_context",
                return_value=types.SimpleNamespace(
                    is_active=True,
                    script_function_name="Std_MsgboxYesNo",
                    native_function_name="IsFieldMessageBoxHidden",
                ),
            ),
        ):
            observation = observe_interaction()

        self.assertFalse(observation.dialogue_waiting)
        self.assertTrue(observation.field_message_lifecycle_active)
        self.assertFalse(observation.field_message_advance_ready)
        interaction_state._field_message_lifecycle_active = False
