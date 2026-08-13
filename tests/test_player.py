import types
import unittest
from unittest.mock import patch


class TestPlayerControllability(unittest.TestCase):
    def test_non_animated_door_exit_is_not_navigation_ready(self):
        from modules.memory import GameState
        from modules.player import player_avatar_is_controllable

        with (
            patch("modules.player.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.player.get_player_map_object",
                return_value=types.SimpleNamespace(flags={"heldMovementActive"}),
            ),
            patch("modules.player.get_player_avatar", return_value=types.SimpleNamespace(flags=set())),
            patch(
                "modules.player.task_is_active",
                side_effect=lambda task: task == "Task_ExitNonAnimDoor",
            ),
        ):
            self.assertFalse(player_avatar_is_controllable())

    def test_non_animated_door_exit_becomes_navigation_ready_when_task_finishes(self):
        from modules.memory import GameState
        from modules.player import player_avatar_is_controllable

        with (
            patch("modules.player.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.player.get_player_map_object",
                return_value=types.SimpleNamespace(flags={"heldMovementActive"}),
            ),
            patch("modules.player.get_player_avatar", return_value=types.SimpleNamespace(flags=set())),
            patch("modules.player.task_is_active", return_value=False),
        ):
            self.assertTrue(player_avatar_is_controllable())
