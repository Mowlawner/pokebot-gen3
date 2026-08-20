import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.interaction_state import _observe_field_message_waiting
from modules.memory import GameState


class InteractionStateTests(unittest.TestCase):
    def test_unknown_coarse_state_still_recognizes_field_dialogue_wait(self):
        script = SimpleNamespace(
            is_active=True,
            native_function_name="WaitForAorBPress",
            script_function_name="Std_MsgboxDefault",
        )
        with patch("modules.interaction_state.get_global_script_context", return_value=script), patch(
            "modules.interaction_state.is_waiting_for_input", return_value=True
        ):
            self.assertTrue(_observe_field_message_waiting(GameState.UNKNOWN))

    def test_unknown_state_without_active_message_is_not_sticky(self):
        script = SimpleNamespace(
            is_active=False,
            native_function_name=None,
            script_function_name=None,
        )
        with patch("modules.interaction_state.get_global_script_context", return_value=script), patch(
            "modules.interaction_state.is_waiting_for_input", return_value=False
        ):
            self.assertFalse(_observe_field_message_waiting(GameState.UNKNOWN))


if __name__ == "__main__":
    unittest.main()
