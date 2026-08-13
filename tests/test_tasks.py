import types
import unittest
from unittest.mock import patch


class TestIsWaitingForInput(unittest.TestCase):
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
