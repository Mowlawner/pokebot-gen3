import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from modules.config.schemas_v1 import StartGame
from modules.map_data import MapRSE
from modules.memory import GameState
from modules.nuzlocke.emerald_capabilities import (
    EmeraldCampaignAction,
    choose_emerald_observation_action,
    observation_driven_emerald_campaign,
)
from modules.nuzlocke.emerald_observation import EmeraldObservation
from modules.nuzlocke.emerald_starter_selection import (
    HOENN_STARTER_CHOICES,
    EmeraldStarterSelectionObservation,
    EmeraldStarterSelectionPhase,
    configured_emerald_starter,
    observe_emerald_starter_selection,
    resolve_emerald_starter,
)


class EmeraldStarterSelectionTests(unittest.TestCase):
    @staticmethod
    def observation(starter=None):
        return EmeraldObservation(
            game_state=GameState.CHOOSE_STARTER,
            ui_mode="CHOOSE_STARTER",
            map_id=MapRSE.ROUTE101.value,
            coordinates=None,
            controllable=False,
            dialogue_actionable=False,
            dialogue_lifecycle_active=False,
            menu=None,
            naming=None,
            confirmation=None,
            gender_task=None,
            rom_owned_movement=False,
            starter_selection=starter,
        )

    def test_configuration_accepts_each_hoenn_starter_and_rejects_invalid(self):
        for species in HOENN_STARTER_CHOICES:
            self.assertEqual(StartGame(starter=species).starter, species)
        with self.assertRaises(ValidationError):
            StartGame(starter="Bulbasaur")

    def test_campaign_configuration_is_required_and_clear(self):
        for species in HOENN_STARTER_CHOICES:
            fake_context = SimpleNamespace(config=SimpleNamespace(start_game=SimpleNamespace(starter=species)))
            with patch("modules.nuzlocke.emerald_starter_selection.context", fake_context):
                self.assertEqual(configured_emerald_starter(), species)
        fake_context = SimpleNamespace(config=SimpleNamespace(start_game=SimpleNamespace(starter=None)))
        with patch("modules.nuzlocke.emerald_starter_selection.context", fake_context):
            with self.assertRaisesRegex(ValueError, "start_game.starter"):
                configured_emerald_starter()

    def test_observes_rom_task_cursor_and_expected_choices(self):
        task = Mock()
        task.data_value.return_value = 1
        with (
            patch("modules.nuzlocke.emerald_starter_selection.get_game_state", return_value=GameState.CHOOSE_STARTER),
            patch(
                "modules.nuzlocke.emerald_starter_selection.get_player_avatar",
                return_value=SimpleNamespace(map_group_and_number=MapRSE.ROUTE101.value),
            ),
            patch(
                "modules.nuzlocke.emerald_starter_selection.get_task",
                side_effect=lambda name: task if name == "Task_HandleStarterChooseInput" else None,
            ),
        ):
            observed = observe_emerald_starter_selection()
        self.assertEqual(observed.choices, HOENN_STARTER_CHOICES)
        self.assertEqual(observed.selected_species, "Torchic")
        self.assertTrue(observed.input_ready)

    def test_does_not_observe_before_ui_exists(self):
        with patch("modules.nuzlocke.emerald_starter_selection.get_game_state", return_value=GameState.OVERWORLD):
            self.assertIsNone(observe_emerald_starter_selection())

    def test_random_resolves_once_from_exact_hoenn_pool_without_defaulting(self):
        rng = Mock()
        rng.choice.return_value = "Mudkip"
        self.assertEqual(resolve_emerald_starter("Random", rng), "Mudkip")
        rng.choice.assert_called_once_with(HOENN_STARTER_CHOICES)

    def test_each_configured_starter_maps_to_bounded_navigation(self):
        expected = {
            "Treecko": EmeraldCampaignAction.MOVE_STARTER_LEFT,
            "Torchic": EmeraldCampaignAction.CHOOSE_STARTER,
            "Mudkip": EmeraldCampaignAction.MOVE_STARTER_RIGHT,
        }
        starter = EmeraldStarterSelectionObservation(
            HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.CHOOSING, True
        )
        for configured, action in expected.items():
            with (
                self.subTest(configured=configured),
                patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value=configured),
            ):
                self.assertIs(choose_emerald_observation_action(self.observation(starter)), action)

    def test_animation_waits_and_confirmation_uses_rom_yes_default(self):
        animating = EmeraldStarterSelectionObservation(
            HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.ANIMATING, False
        )
        confirming = EmeraldStarterSelectionObservation(
            HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.CONFIRMING, True
        )
        self.assertIs(choose_emerald_observation_action(self.observation(animating)), EmeraldCampaignAction.WAIT)
        self.assertIs(
            choose_emerald_observation_action(self.observation(confirming)), EmeraldCampaignAction.CONFIRM_STARTER
        )

    def test_executor_emits_one_selection_input_then_reobserves_without_movement(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Torchic")),
            nuzlocke_runtime=SimpleNamespace(session_id="test"),
        )
        choosing = self.observation(
            EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.CHOOSING, True)
        )
        animating = self.observation(
            EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.ANIMATING, False)
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Torchic"),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", side_effect=[choosing, animating]),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            next(execution)
            next(execution)
        emulator.press_button.assert_called_once_with("A")

    def test_stale_cursor_observation_does_not_repeat_directional_input(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Mudkip")),
            nuzlocke_runtime=SimpleNamespace(session_id="stable"),
        )
        choosing = self.observation(
            EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.CHOOSING, True)
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Mudkip"),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=choosing),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            next(execution)
            next(execution)
        emulator.press_button.assert_called_once_with("Right")

    def test_stale_cursor_retries_after_two_quiet_observations(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Mudkip")),
            nuzlocke_runtime=SimpleNamespace(session_id="retry"),
        )
        choosing = self.observation(
            EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.CHOOSING, True)
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Mudkip"),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=choosing),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            for _ in range(4):
                next(execution)
        self.assertEqual(
            [call.args[0] for call in emulator.press_button.call_args_list],
            ["Right", "Right"],
        )

    def test_random_resolution_is_stable_for_mounted_session(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Random")),
            nuzlocke_runtime=SimpleNamespace(session_id="stable"),
        )
        treecko_cursor = self.observation(
            EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, 1, EmeraldStarterSelectionPhase.CHOOSING, True)
        )
        moved = self.observation(
            EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, 0, EmeraldStarterSelectionPhase.CHOOSING, True)
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Random"),
            patch("modules.nuzlocke.emerald_capabilities.resolve_emerald_starter", return_value="Treecko") as resolve,
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", side_effect=[treecko_cursor, moved]),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            next(execution)
            next(execution)
        resolve.assert_called_once()
        self.assertEqual([call.args[0] for call in emulator.press_button.call_args_list], ["Left", "A"])


if __name__ == "__main__":
    unittest.main()
