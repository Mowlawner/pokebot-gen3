import types
import unittest
from unittest.mock import Mock, patch

from modules.memory import GameState
from modules.nuzlocke.emerald_confirmation import (
    EmeraldConfirmationChoice,
    EmeraldConfirmationContext,
    observe_emerald_confirmation,
)
from modules.nuzlocke.emerald_naming import (
    EmeraldNamingTarget,
    observe_emerald_naming,
)


class EmeraldNamingObservationTests(unittest.TestCase):
    def observe(self, template):
        base = 0x02010000
        addresses = []

        def read_bytes(address, size):
            addresses.append(address)
            return bytes([template]) if address == base + 0x1E2C else bytes(size)

        emulator = types.SimpleNamespace(read_bytes=read_bytes)
        with (
            patch("modules.nuzlocke.emerald_naming.get_game_state", return_value=GameState.NAMING_SCREEN),
            patch("modules.nuzlocke.emerald_naming.read_symbol", return_value=(0x02010000).to_bytes(4, "little")),
            patch("modules.context.context", types.SimpleNamespace(emulator=emulator)),
            patch("modules.nuzlocke.emerald_naming.get_naming_screen_data", return_value=object()),
        ):
            observation = observe_emerald_naming()
        self.assertIn(base + 0x1E2C, addresses)
        return observation

    def test_player_template_is_player_name(self):
        self.assertEqual(self.observe(0).target, EmeraldNamingTarget.PLAYER_NAME)

    def test_nickname_template_is_pokemon_nickname(self):
        self.assertEqual(self.observe(3).target, EmeraldNamingTarget.POKEMON_NICKNAME)

    def test_caught_mon_template_is_distinct_nickname_context(self):
        self.assertEqual(self.observe(2).target, EmeraldNamingTarget.CAUGHT_POKEMON_NICKNAME)

    def test_unknown_template_is_explicit_and_safe(self):
        self.assertEqual(self.observe(99).target, EmeraldNamingTarget.UNKNOWN)

    def test_missing_live_pointer_is_unknown(self):
        with (
            patch("modules.nuzlocke.emerald_naming.get_game_state", return_value=GameState.NAMING_SCREEN),
            patch("modules.nuzlocke.emerald_naming.read_symbol", return_value=b"\0\0\0\0"),
        ):
            observation = observe_emerald_naming()
        self.assertEqual(observation.target, EmeraldNamingTarget.UNKNOWN)

    def test_supplied_player_naming_state_has_authoritative_template_zero(self):
        from pathlib import Path

        state = Path(__file__).parents[1] / "profiles" / "test_player_naming" / "current_state.ss1"
        if not state.exists():
            self.skipTest("isolated naming-screen fixture is not present")
        from tests.utility import _load_test_state

        _load_test_state(state)
        observation = observe_emerald_naming()
        self.assertEqual(observation.template_number, 0)
        self.assertEqual(observation.target, EmeraldNamingTarget.PLAYER_NAME)


class NamingCapabilityPolicyTests(unittest.TestCase):
    def test_player_policy_does_not_supply_nickname_policy(self):
        from modules.nuzlocke.emerald_capabilities import _configured_name

        config = types.SimpleNamespace(start_game=types.SimpleNamespace(player_name="ASH"))
        with patch("modules.nuzlocke.emerald_capabilities.context", types.SimpleNamespace(config=config)):
            self.assertEqual(_configured_name(EmeraldNamingTarget.PLAYER_NAME), "ASH")
            self.assertIsNone(_configured_name(EmeraldNamingTarget.POKEMON_NICKNAME))

    def test_nickname_policy_is_used_only_for_nickname_target(self):
        from modules.nuzlocke.emerald_capabilities import _configured_name

        config = types.SimpleNamespace(
            start_game=types.SimpleNamespace(player_name="ASH"),
            naming=types.SimpleNamespace(pokemon_nickname="SPARK"),
        )
        with patch("modules.nuzlocke.emerald_capabilities.context", types.SimpleNamespace(config=config)):
            self.assertEqual(_configured_name(EmeraldNamingTarget.POKEMON_NICKNAME), "SPARK")
            self.assertEqual(_configured_name(EmeraldNamingTarget.PLAYER_NAME), "ASH")

    def test_random_player_policy_is_resolved_once_with_observed_gender(self):
        from modules.nuzlocke.emerald_capabilities import _resolve_player_campaign_name

        config = types.SimpleNamespace(start_game=types.SimpleNamespace(player_name="random", player_gender="random"))
        player = types.SimpleNamespace(gender="female")
        initialization = types.SimpleNamespace(name="MAY")
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", types.SimpleNamespace(config=config)),
            patch("modules.nuzlocke.emerald_capabilities.get_player", return_value=player),
            patch(
                "modules.nuzlocke.emerald_capabilities.resolve_start_game_initialization", return_value=initialization
            ) as resolve,
        ):
            self.assertEqual(_resolve_player_campaign_name(), "MAY")
        self.assertEqual(resolve.call_args.args[:2], ("random", "female"))

    def test_keyboard_helper_is_one_bounded_transaction(self):
        from modules.nuzlocke.emerald_capabilities import _naming_input

        press = Mock()

        def keyboard(_name):
            press()
            yield

        with patch("modules.nuzlocke.emerald_capabilities.type_in_naming_screen", keyboard):
            _naming_input("ASH")
        press.assert_called_once_with()


class EmeraldConfirmationObservationTests(unittest.TestCase):
    def test_player_confirmation_cursor_and_ready_state_are_task_backed(self):
        task = types.SimpleNamespace(symbol="Task_NewGameBirchSpeech_ProcessNameYesNoMenu")
        with (
            patch("modules.nuzlocke.emerald_confirmation.get_tasks", return_value=[task]),
            patch("modules.nuzlocke.emerald_confirmation.read_symbol", return_value=b"\0\0\0") as read,
        ):
            observation = observe_emerald_confirmation()
        self.assertTrue(observation.active)
        self.assertTrue(observation.input_ready)
        self.assertIs(observation.selected, EmeraldConfirmationChoice.YES)
        self.assertIs(observation.context, EmeraldConfirmationContext.PLAYER_NAME)
        read.assert_called_once_with("sMenu", offset=0x02, size=1)

    def test_no_row_is_observed_without_assuming_yes(self):
        task = types.SimpleNamespace(symbol="Task_NewGameBirchSpeech_ProcessNameYesNoMenu")
        with (
            patch("modules.nuzlocke.emerald_confirmation.get_tasks", return_value=[task]),
            patch("modules.nuzlocke.emerald_confirmation.read_symbol", return_value=b"\1"),
        ):
            observation = observe_emerald_confirmation()
        self.assertIs(observation.selected, EmeraldConfirmationChoice.NO)

    def test_create_task_is_active_but_not_input_ready(self):
        task = types.SimpleNamespace(symbol="Task_NewGameBirchSpeech_CreateNameYesNo")
        with (
            patch("modules.nuzlocke.emerald_confirmation.get_tasks", return_value=[task]),
            patch("modules.nuzlocke.emerald_confirmation.read_symbol", return_value=b"\0"),
        ):
            observation = observe_emerald_confirmation()
        self.assertTrue(observation.active)
        self.assertFalse(observation.input_ready)

    def test_generic_yes_no_handler_has_no_invented_naming_context(self):
        task = types.SimpleNamespace(symbol="Task_HandleYesNoInput")
        with (
            patch("modules.nuzlocke.emerald_confirmation.get_tasks", return_value=[task]),
            patch("modules.nuzlocke.emerald_confirmation.read_symbol", return_value=b"\0"),
        ):
            observation = observe_emerald_confirmation()
        self.assertIs(observation.context, EmeraldConfirmationContext.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
