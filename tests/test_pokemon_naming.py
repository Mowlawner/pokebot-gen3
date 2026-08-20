import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.memory import GameState
from modules.nuzlocke.emerald_capabilities import (
    EmeraldCampaignAction,
    choose_emerald_observation_action,
    observation_driven_emerald_campaign,
)
from modules.nuzlocke.emerald_confirmation import (
    EmeraldConfirmationChoice,
    EmeraldConfirmationContext,
    EmeraldConfirmationObservation,
)
from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget
from modules.nuzlocke.emerald_observation import EmeraldObservation
from modules.nuzlocke.pokemon_naming import generate_pokemon_nickname


class PokemonNamingTests(unittest.TestCase):
    @staticmethod
    def observation(**changes):
        values = dict(
            game_state=GameState.OVERWORLD,
            ui_mode="OVERWORLD",
            map_id=None,
            coordinates=None,
            controllable=False,
            dialogue_actionable=False,
            dialogue_lifecycle_active=False,
            menu=None,
            naming=None,
            confirmation=None,
            gender_task=None,
            rom_owned_movement=False,
            overworld=None,
        )
        values.update(changes)
        return EmeraldObservation(**values)

    def test_gender_aware_names_and_genderless_request(self):
        for gender in ("male", "female", None):
            with (
                self.subTest(gender=gender),
                patch("modules.nuzlocke.pokemon_naming._human_name", return_value="Alex") as source,
            ):
                self.assertEqual(generate_pokemon_nickname("Treecko", gender), "Alex")
                source.assert_called_once_with(gender)

    def test_name_length_is_preserved_or_safely_bounded(self):
        with patch("modules.nuzlocke.pokemon_naming._human_name", return_value="May"):
            self.assertEqual(generate_pokemon_nickname("Mudkip", "female"), "May")
        with patch("modules.nuzlocke.pokemon_naming._human_name", return_value="Alexanderson"):
            self.assertEqual(generate_pokemon_nickname("Mudkip", "male"), "Alexanders")

    def test_nickname_prompt_chooses_yes_and_preempts_dialogue(self):
        prompt = EmeraldConfirmationObservation(
            True, EmeraldConfirmationChoice.YES, True, EmeraldConfirmationContext.POKEMON_NICKNAME
        )
        action = choose_emerald_observation_action(self.observation(confirmation=prompt, dialogue_actionable=True))
        self.assertIs(action, EmeraldCampaignAction.CHOOSE_POKEMON_NICKNAME)

    def test_nickname_screen_owns_campaign_and_prevents_navigation(self):
        naming = EmeraldNamingObservation(
            EmeraldNamingTarget.POKEMON_NICKNAME,
            3,
            0x02010000,
            True,
            277,
            "Treecko",
            "female",
            1234,
        )
        action = choose_emerald_observation_action(
            self.observation(
                game_state=GameState.NAMING_SCREEN,
                naming=naming,
                controllable=True,
                overworld=SimpleNamespace(),
            )
        )
        self.assertIs(action, EmeraldCampaignAction.ENTER_POKEMON_NICKNAME)

    def test_nonready_naming_screen_waits(self):
        naming = EmeraldNamingObservation(
            EmeraldNamingTarget.POKEMON_NICKNAME,
            3,
            0x02010000,
            False,
            277,
            "Treecko",
            "male",
            1234,
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(naming=naming)),
            EmeraldCampaignAction.WAIT,
        )

    def test_stale_nickname_confirmation_is_bounded(self):
        emulator = SimpleNamespace(press_button=Mock())
        context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Treecko")),
            nuzlocke_runtime=SimpleNamespace(session_id="nickname"),
        )
        prompt = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.POKEMON_NICKNAME,
            "Task_HandleYesNoInput",
        )
        observed = self.observation(confirmation=prompt)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Treecko"),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=observed),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            next(execution)
            next(execution)
        emulator.press_button.assert_called_once_with("A")

    def test_stale_nickname_confirmation_retries_after_two_quiet_observations(self):
        emulator = SimpleNamespace(press_button=Mock())
        context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Treecko")),
            nuzlocke_runtime=SimpleNamespace(session_id="nickname-retry"),
        )
        prompt = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.POKEMON_NICKNAME,
            "Task_HandleYesNoInput",
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Treecko"),
            patch(
                "modules.nuzlocke.emerald_capabilities._emerald_observation",
                return_value=self.observation(confirmation=prompt),
            ),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            for _ in range(4):
                next(execution)
        self.assertEqual(
            [call.args[0] for call in emulator.press_button.call_args_list],
            ["A", "A"],
        )

    def test_confirmed_nickname_waits_for_naming_screen_without_navigation(self):
        emulator = SimpleNamespace(press_button=Mock())
        context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Treecko")),
            nuzlocke_runtime=SimpleNamespace(session_id="nickname-transition"),
        )
        prompt = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.POKEMON_NICKNAME,
            "Task_HandleYesNoInput",
        )
        transition = self.observation(controllable=True, overworld=SimpleNamespace())
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Treecko"),
            patch(
                "modules.nuzlocke.emerald_capabilities._emerald_observation",
                side_effect=[self.observation(confirmation=prompt), transition],
            ),
            patch("modules.nuzlocke.emerald_capabilities.observation_driven_overworld_progression") as navigation,
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            next(execution)
            next(execution)
        emulator.press_button.assert_called_once_with("A")
        navigation.assert_not_called()

    def test_unrelated_yes_no_prompt_is_not_claimed(self):
        prompt = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.UNKNOWN,
            "Task_HandleYesNoInput",
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(confirmation=prompt)),
            EmeraldCampaignAction.WAIT,
        )

    def test_go_see_rival_prompt_chooses_yes(self):
        prompt = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.GO_SEE_RIVAL,
            "Task_HandleYesNoInput",
            script_identity="LittlerootTown_ProfessorBirchsLab_EventScript_GoSeeRival",
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(confirmation=prompt)),
            EmeraldCampaignAction.CONFIRM_GO_SEE_RIVAL,
        )

    def test_go_see_rival_confirmation_is_bounded_and_retries(self):
        emulator = SimpleNamespace(press_button=Mock())
        context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Treecko")),
            nuzlocke_runtime=SimpleNamespace(session_id="go-see-rival"),
        )
        prompt = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.GO_SEE_RIVAL,
            "Task_HandleYesNoInput",
            script_identity="LittlerootTown_ProfessorBirchsLab_EventScript_GoSeeRival",
        )
        observed = self.observation(confirmation=prompt)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Treecko"),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=observed),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            for _ in range(4):
                next(execution)
        self.assertEqual(
            [call.args[0] for call in emulator.press_button.call_args_list],
            ["A", "A"],
        )

    def test_go_see_rival_script_owns_non_actionable_transition(self):
        script = "LittlerootTown_ProfessorBirchsLab_EventScript_GoSeeRival"
        owned = self.observation(
            script_stack=(script,),
            controllable=True,
            overworld=SimpleNamespace(),
        )
        self.assertIs(
            choose_emerald_observation_action(owned),
            EmeraldCampaignAction.WAIT,
        )

    def test_go_see_rival_script_allows_dialogue_then_releases(self):
        script = "LittlerootTown_ProfessorBirchsLab_EventScript_GoSeeRival"
        self.assertIs(
            choose_emerald_observation_action(self.observation(script_stack=(script,), dialogue_actionable=True)),
            EmeraldCampaignAction.ADVANCE_DIALOGUE,
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(controllable=True, overworld=SimpleNamespace())),
            EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD,
        )

    def test_nickname_debounce_does_not_claim_wall_clock_confirmation(self):
        emulator = SimpleNamespace(press_button=Mock())
        context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Treecko")),
            nuzlocke_runtime=SimpleNamespace(session_id="clock"),
        )
        clock = self.observation(
            campaign_facts=(("wall_clock_set", False),),
            clock_interaction=("Task_SetClock_HandleConfirmInput", 0, 12, 0, 1),
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", context),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=clock),
        ):
            execution = observation_driven_emerald_campaign("set_wall_clock")
            next(execution)
        emulator.press_button.assert_called_once_with("Up")

    def test_observed_gender_drives_one_stable_nickname_keyboard_session(self):
        emulator = SimpleNamespace(press_button=Mock())
        context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(starter="Treecko")),
            nuzlocke_runtime=SimpleNamespace(session_id="nickname"),
        )
        naming = EmeraldNamingObservation(
            EmeraldNamingTarget.POKEMON_NICKNAME,
            3,
            0x02010000,
            True,
            277,
            "Treecko",
            "female",
            1234,
        )
        observed = self.observation(game_state=GameState.NAMING_SCREEN, naming=naming)
        keyboard_steps = Mock()

        def keyboard(name, max_length):
            keyboard_steps(name, max_length)
            yield
            return

        with (
            patch("modules.nuzlocke.emerald_capabilities.context", context),
            patch("modules.nuzlocke.emerald_capabilities.configured_emerald_starter", return_value="Treecko"),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", return_value=observed),
            patch("modules.nuzlocke.emerald_capabilities.generate_pokemon_nickname", return_value="Alice") as generate,
            patch("modules.nuzlocke.emerald_capabilities.type_in_naming_screen", side_effect=keyboard),
        ):
            execution = observation_driven_emerald_campaign("obtain_starter")
            next(execution)
            next(execution)
        generate.assert_called_once_with("Treecko", "female")
        keyboard_steps.assert_called_once_with("Alice", 10)


if __name__ == "__main__":
    unittest.main()
