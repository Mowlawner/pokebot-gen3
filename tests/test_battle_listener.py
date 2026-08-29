import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.modes._listeners import _battle_return_to_field_complete


class BattleListenerLifecycleTests(unittest.TestCase):
    def test_return_to_field_gate_does_not_require_avatar_to_be_standing_still(self):
        frame = SimpleNamespace(active_tasks=[])

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_OVERWORLD"),
            patch("modules.modes._listeners.get_map_objects", return_value=[object()]),
            patch("modules.modes._listeners.get_global_script_context", return_value=None),
        ):
            self.assertTrue(_battle_return_to_field_complete(frame))

    def test_return_to_field_gate_waits_for_return_callback(self):
        frame = SimpleNamespace(active_tasks=[])

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_RETURNTOFIELD"),
            patch("modules.modes._listeners.get_map_objects", return_value=[object()]),
            patch("modules.modes._listeners.get_global_script_context", return_value=None),
        ):
            self.assertFalse(_battle_return_to_field_complete(frame))

    def test_return_to_field_gate_waits_for_return_task(self):
        frame = SimpleNamespace(active_tasks=["Task_ReturnToFieldNoScript"])

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_OVERWORLD"),
            patch("modules.modes._listeners.get_map_objects", return_value=[object()]),
            patch("modules.modes._listeners.get_global_script_context", return_value=None),
        ):
            self.assertFalse(_battle_return_to_field_complete(frame))

    def test_return_to_field_gate_waits_for_post_battle_script(self):
        frame = SimpleNamespace(active_tasks=[])
        script = SimpleNamespace(is_active=True, native_function_name="WaitForMovementFinish")

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_OVERWORLD"),
            patch("modules.modes._listeners.get_map_objects", return_value=[object()]),
            patch("modules.modes._listeners.get_global_script_context", return_value=script),
        ):
            self.assertFalse(_battle_return_to_field_complete(frame))


if __name__ == "__main__":
    unittest.main()
