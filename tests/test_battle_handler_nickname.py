import types
import unittest
from unittest.mock import patch

from modules.battle_handler import handle_nickname_caught_pokemon
from modules.context import context


class TestCaughtPokemonNicknameBoundary(unittest.TestCase):
    def test_nuzlocke_policy_name_is_entered_for_captured_pokemon(self):
        encounter = types.SimpleNamespace(
            pokemon=types.SimpleNamespace(species=types.SimpleNamespace(name="Wurmple"), gender="female")
        )
        old_runtime = context.nuzlocke_runtime
        context.nuzlocke_runtime = object()
        try:
            with patch("modules.battle_handler.plugin_should_nickname_pokemon", return_value=None), patch(
                "modules.battle_handler.generate_pokemon_nickname", return_value="MAY"
            ) as generate, patch("modules.battle_handler.handle_naming_screen", return_value=iter(())) as naming, patch(
                "modules.battle_handler.get_current_battle_script_instruction", return_value="Other"
            ):
                list(handle_nickname_caught_pokemon(encounter))
            generate.assert_called_once_with("Wurmple", "female")
            naming.assert_called_once_with("MAY")
        finally:
            context.nuzlocke_runtime = old_runtime

    def test_missing_nuzlocke_name_preserves_decline_fallback(self):
        encounter = types.SimpleNamespace(
            pokemon=types.SimpleNamespace(species=types.SimpleNamespace(name="Wurmple"), gender="female")
        )
        old_runtime = context.nuzlocke_runtime
        context.nuzlocke_runtime = object()
        try:
            with patch("modules.battle_handler.plugin_should_nickname_pokemon", return_value=None), patch(
                "modules.battle_handler.generate_pokemon_nickname", side_effect=ValueError
            ), patch(
                "modules.battle_handler.get_current_battle_script_instruction",
                side_effect=["BattleScript_TryNicknameCaughtMon", "Other"],
            ), patch.object(
                context, "emulator", types.SimpleNamespace(press_button=lambda _: None)
            ):
                list(handle_nickname_caught_pokemon(encounter))
        finally:
            context.nuzlocke_runtime = old_runtime
