import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.memory import GameState
from modules.nuzlocke.emerald_capabilities import (
    EmeraldCampaignAction,
    emerald_campaign_capability,
    observation_driven_emerald_campaign,
    choose_emerald_observation_action,
)
from modules.nuzlocke.emerald_dialogue import advance_dialogue
from modules.nuzlocke.emerald_observation import EmeraldObservation


class CampaignObservationMigrationTests(unittest.TestCase):
    @staticmethod
    def observation(**changes):
        values = dict(
            game_state=GameState.OVERWORLD,
            ui_mode="OVERWORLD",
            map_id=(1, 1),
            coordinates=(4, 4),
            controllable=True,
            dialogue_actionable=False,
            dialogue_lifecycle_active=False,
            menu=None,
            naming=None,
            confirmation=None,
            gender_task=None,
            rom_owned_movement=False,
            campaign_facts=(("wall_clock_set", False),),
            overworld=SimpleNamespace(facing="Up"),
            objective_id="set_wall_clock",
            clock_target=((4, 4), "Up"),
        )
        values.update(changes)
        return EmeraldObservation(**values)

    def test_post_startup_capabilities_are_observation_driven_and_not_legacy(self):
        emulator = SimpleNamespace(press_button=lambda _: None)
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace(), debug=False)
        wait = self.observation(coordinates=(9, 9), clock_target=None)
        for objective_id in (
            "set_wall_clock",
            "meet_rival",
            "rescue_birch",
            "obtain_starter",
            "receive_pokedex",
        ):
            with self.subTest(objective_id=objective_id):
                with (
                    patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
                    patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=wait),
                    patch("modules.modes.opening.EmeraldOpeningCapability") as legacy,
                ):
                    execution = emerald_campaign_capability(objective_id)
                    next(execution)
            legacy.assert_not_called()
            self.assertEqual(execution.gi_yieldfrom.gi_code.co_name, "observation_driven_emerald_campaign")

    def test_clock_interaction_is_selected_from_current_observation(self):
        observation = self.observation(clock_target=((4, 4), "Up"))
        self.assertIs(
            choose_emerald_observation_action(observation),
            EmeraldCampaignAction.INTERACT_CLOCK,
        )

    def test_dialogue_preempts_clock_navigation_and_rom_movement_waits(self):
        dialogue = self.observation(dialogue_actionable=True, rom_owned_movement=True)
        self.assertIs(
            choose_emerald_observation_action(dialogue),
            EmeraldCampaignAction.ADVANCE_DIALOGUE,
        )
        scripted = self.observation(rom_owned_movement=True)
        self.assertIs(
            choose_emerald_observation_action(scripted),
            EmeraldCampaignAction.WAIT,
        )

    def test_executor_reobserves_after_each_bounded_action(self):
        observations = iter(
            (
                self.observation(),
                self.observation(coordinates=(4, 4)),
            )
        )
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace(), debug=False)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._emerald_observation",
                side_effect=lambda *_: next(observations),
            ) as mock_observation,
        ):
            execution = observation_driven_emerald_campaign("set_wall_clock")
            next(execution)
            next(execution)
        self.assertEqual(emulator.press_button.call_count, 2)
        self.assertEqual(mock_observation.call_count, 2)

    def test_actionable_dialogue_requests_one_fresh_a_edge(self):
        emulator = SimpleNamespace(press_button=Mock(), press_button_fresh=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace(), debug=False)
        with (
            patch("modules.nuzlocke.emerald_dialogue.context", fake_context),
            patch("modules.nuzlocke.emerald_dialogue.observe_dialogue", return_value=(True, True)),
        ):
            next(advance_dialogue(actionable=True))
        emulator.press_button_fresh.assert_called_once_with("A")
        emulator.press_button.assert_not_called()


if __name__ == "__main__":
    unittest.main()
