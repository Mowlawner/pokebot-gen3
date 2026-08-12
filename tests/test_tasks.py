import types
import unittest
from unittest.mock import patch


class TestIsWaitingForInput(unittest.TestCase):
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
