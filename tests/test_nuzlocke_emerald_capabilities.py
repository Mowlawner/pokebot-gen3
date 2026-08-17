import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.modes.opening import OpeningSequenceState
from modules.memory import GameState
from modules.nuzlocke.emerald_capabilities import (
    EmeraldCampaignAction,
    EmeraldCampaignObservation,
    choose_emerald_campaign_action,
    observation_driven_emerald_campaign,
)
from modules.nuzlocke.emerald_confirmation import (
    EmeraldConfirmationChoice,
    EmeraldConfirmationContext,
    EmeraldConfirmationObservation,
)
from modules.nuzlocke.emerald_menu_observation import (
    EmeraldMainMenuItem,
    EmeraldMenuKind,
    EmeraldMenuObservation,
    EmeraldOptionsItem,
)


class EmeraldCampaignCapabilityTests(unittest.TestCase):
    def observation(self, state, *, fast=False, setup=False, dialogue=False, gender_task=None):
        return EmeraldCampaignObservation(
            state,
            text_speed_fast=fast,
            new_game_setup_complete=setup,
            dialogue_waiting=dialogue,
            gender_task=gender_task,
        )

    def test_options_with_fast_text_speed_exits_without_opening_flow_state(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.OPTIONS_MENU, fast=True))
        self.assertIs(action, EmeraldCampaignAction.EXIT_OPTIONS)

    def test_main_menu_with_fast_text_speed_starts_new_game(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU, fast=True))
        self.assertIs(action, EmeraldCampaignAction.START_NEW_GAME)

    def test_main_menu_without_fast_text_speed_enters_options(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU))
        self.assertIs(action, EmeraldCampaignAction.ENTER_OPTIONS)

    def test_immediate_states_take_priority_over_campaign_menu_action(self):
        dialogue = choose_emerald_campaign_action(
            self.observation(OpeningSequenceState.MAIN_MENU, fast=True, dialogue=True)
        )
        gender = choose_emerald_campaign_action(
            self.observation(
                OpeningSequenceState.MAIN_MENU,
                fast=True,
                gender_task="Task_NewGameBirchSpeech_ChooseGender",
            )
        )
        self.assertIs(dialogue, EmeraldCampaignAction.ADVANCE_DIALOGUE)
        self.assertIs(gender, EmeraldCampaignAction.CHOOSE_GENDER)

    def test_post_gender_name_prompt_is_a_single_transition_input(self):
        action = choose_emerald_campaign_action(
            self.observation(
                OpeningSequenceState.SCRIPTED_INTRO,
                gender_task="Task_NewGameBirchSpeech_WaitPressBeforeNameChoice",
            )
        )
        self.assertIs(action, EmeraldCampaignAction.ADVANCE_NAME_PROMPT)

    def test_configured_gender_selection_issues_one_input(self):
        from modules.nuzlocke.emerald_capabilities import _gender_input

        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_gender="male")),
        )
        task = SimpleNamespace(data_value=lambda index: 0)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities.get_task", return_value=task),
        ):
            list(_gender_input("male"))
        emulator.press_button.assert_called_once_with("A")

    def test_completed_setup_does_not_start_new_game_again(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU, fast=True, setup=True))
        self.assertIs(action, EmeraldCampaignAction.WAIT)


class EmeraldCampaignLifecycleTests(unittest.TestCase):
    def observation(
        self,
        state,
        *,
        fast=False,
        setup=False,
        dialogue=False,
        cursor=None,
        menu=None,
        option_value=None,
        gender_task=None,
    ):
        if menu is None and state is OpeningSequenceState.MAIN_MENU and cursor is not None:
            items = (EmeraldMainMenuItem.NEW_GAME, EmeraldMainMenuItem.OPTIONS)
            menu = EmeraldMenuObservation(
                EmeraldMenuKind.MAIN_MENU,
                cursor,
                items[cursor] if 0 <= cursor < len(items) else None,
                True,
                "Task_HandleMainMenuInput",
                cursor,
                0,
            )
        if menu is None and state is OpeningSequenceState.OPTIONS_MENU:
            row = 0 if cursor is None else cursor
            value = 2 if fast else (1 if option_value is None else option_value)
            menu = EmeraldMenuObservation(
                EmeraldMenuKind.OPTIONS_MENU,
                row,
                tuple(EmeraldOptionsItem)[row] if 0 <= row < len(EmeraldOptionsItem) else None,
                True,
                "Task_OptionMenuProcessInput",
                row,
                None,
                (value, 0, 0, 0, 0, 0),
            )
        return EmeraldCampaignObservation(
            state,
            fast,
            setup,
            dialogue,
            gender_task,
            cursor,
            False,
            menu,
        )

    def run_generator(self, observations, *, task_active=True, dialogue_step=None):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        generator = None
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=observations),
            patch("modules.nuzlocke.emerald_capabilities._advance_scripted_input", side_effect=dialogue_step),
        ):
            generator = observation_driven_emerald_campaign()
            for _ in observations:
                next(generator)
        return emulator.press_button

    def test_main_menu_options_is_two_observed_transactions(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=1),
                self.observation(OpeningSequenceState.OPTIONS_MENU),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Down", "A", "Right"])

    def test_options_changes_speed_reobserves_fact_then_exits(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.OPTIONS_MENU),
                self.observation(OpeningSequenceState.OPTIONS_MENU),
                self.observation(OpeningSequenceState.OPTIONS_MENU, fast=True),
                self.observation(OpeningSequenceState.MAIN_MENU, fast=True, setup=True, cursor=1),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Right", "B"])

    def test_options_moves_toward_text_speed_one_row_per_observation(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=3),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=2),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=1),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=0),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=0, fast=True),
            ]
        )
        self.assertEqual(
            [call.args[0] for call in presses.call_args_list],
            ["Up", "Up", "Up", "Right", "B"],
        )

    def test_options_exit_is_not_followed_by_reopening_options(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.OPTIONS_MENU, fast=True),
                self.observation(OpeningSequenceState.MAIN_MENU, fast=True, cursor=1),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["B", "Up"])
        self.assertNotIn("Down", [call.args[0] for call in presses.call_args_list])

    def test_dialogue_input_is_one_transaction_even_when_uncontrollable(self):
        # The executor only relies on the observation's dialogue predicate;
        # the legacy helper remains responsible for the native wait details.
        with patch(
            "modules.nuzlocke.emerald_capabilities._advance_scripted_input",
            return_value=iter([None]),
        ) as advance_input:
            emulator = SimpleNamespace(press_button=Mock())
            fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
            with (
                patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
                patch(
                    "modules.nuzlocke.emerald_capabilities._campaign_observation",
                    side_effect=[
                        self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                        self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
                    ],
                ),
            ):
                generator = observation_driven_emerald_campaign()
                next(generator)
                next(generator)
        advance_input.assert_called_once()

    def test_dialogue_transition_does_not_repeat_input(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
            ],
            dialogue_step=iter([iter([None])]),
        )
        self.assertEqual(presses.call_count, 0)

    def test_next_dialogue_message_is_reobserved_and_advanced_once(self):
        advance = Mock(return_value=iter([None]))
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=[
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                ],
            ),
            patch("modules.nuzlocke.emerald_capabilities._advance_scripted_input", advance),
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
            next(generator)
            next(generator)
        self.assertEqual(advance.call_count, 2)

    def test_dialogue_end_is_not_treated_as_dialogue(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
            ],
            dialogue_step=iter([iter([None])]),
        )
        self.assertEqual(presses.call_count, 0)

    def test_failed_menu_transition_does_not_continue_to_a_without_cursor_observation(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Down"])

    def test_unexpected_menu_transition_recalculates_from_new_observation(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
                self.observation(
                    OpeningSequenceState.MAIN_MENU,
                    cursor=2,
                    menu=EmeraldMenuObservation(
                        EmeraldMenuKind.MAIN_MENU,
                        2,
                        EmeraldMainMenuItem.OPTIONS,
                        True,
                        "Task_HandleMainMenuInput",
                        2,
                        1,
                    ),
                ),
                self.observation(
                    OpeningSequenceState.MAIN_MENU,
                    cursor=1,
                    menu=EmeraldMenuObservation(
                        EmeraldMenuKind.MAIN_MENU,
                        1,
                        EmeraldMainMenuItem.NEW_GAME,
                        True,
                        "Task_HandleMainMenuInput",
                        1,
                        1,
                    ),
                ),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Down", "A", "Down"])

    def test_native_standard_message_wait_is_observed_without_controllability(self):
        from modules.nuzlocke.emerald_capabilities import _campaign_observation

        facts = SimpleNamespace(
            text_speed_fast=SimpleNamespace(value=False),
            new_game_setup_complete=SimpleNamespace(value=False),
        )
        campaign_state = SimpleNamespace(campaign_facts=facts)
        native_message_wait = (False, None, True, "WaitForAorBPress", "Std_MsgboxDefault", True, False, False)
        with (
            patch("modules.nuzlocke.campaign_controller.runtime_campaign_state", return_value=campaign_state),
            patch(
                "modules.nuzlocke.emerald_capabilities.EmeraldOpeningCapability._dialogue_state_snapshot",
                return_value=native_message_wait,
            ),
            patch("modules.nuzlocke.emerald_capabilities.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.nuzlocke.emerald_capabilities.get_opening_sequence_state",
                return_value=OpeningSequenceState.SCRIPTED_INTRO,
            ),
            patch("modules.nuzlocke.emerald_capabilities._active_gender_task", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities._main_menu_cursor", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities.is_field_message_waiting_for_input", return_value=False),
        ):
            observation = _campaign_observation()
        self.assertTrue(observation.dialogue_waiting)

    def test_birch_speech_printer_pause_is_observed_outside_overworld(self):
        from modules.nuzlocke.emerald_capabilities import _campaign_observation

        facts = SimpleNamespace(
            text_speed_fast=SimpleNamespace(value=False),
            new_game_setup_complete=SimpleNamespace(value=False),
        )
        campaign_state = SimpleNamespace(campaign_facts=facts)
        birch_message = (
            True,
            0,
            False,
            None,
            None,
            True,
            False,
            True,
            "Pause",
            ("Task_NewGameBirchSpeech_ThisIsAPokemon",),
            None,
            None,
            None,
            None,
        )
        with (
            patch("modules.nuzlocke.campaign_controller.runtime_campaign_state", return_value=campaign_state),
            patch(
                "modules.nuzlocke.emerald_capabilities.EmeraldOpeningCapability._dialogue_state_snapshot",
                return_value=birch_message,
            ),
            patch("modules.nuzlocke.emerald_capabilities.get_game_state", return_value=GameState.UNKNOWN),
            patch(
                "modules.nuzlocke.emerald_capabilities.get_opening_sequence_state",
                return_value=OpeningSequenceState.SCRIPTED_INTRO,
            ),
            patch("modules.nuzlocke.emerald_capabilities._active_gender_task", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities.observe_emerald_menu", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities.is_emerald_field_dialogue_advanceable", return_value=True),
        ):
            observation = _campaign_observation()
        self.assertTrue(observation.dialogue_waiting)
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.ADVANCE_DIALOGUE)

    def test_birch_speech_runtime_shape_issues_one_input_then_yields(self):
        from modules.modes.opening import _advance_scripted_input, EmeraldOpeningCapability

        emulator = SimpleNamespace(press_button=Mock())
        opening_context = SimpleNamespace(
            rom=SimpleNamespace(is_emerald=True),
            emulator=emulator,
            debug=False,
        )
        birch_message = (
            True,
            0,
            False,
            None,
            None,
            True,
            False,
            True,
            "Pause",
            ("Task_NewGameBirchSpeech_ThisIsAPokemon",),
            None,
            None,
            None,
            None,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(EmeraldOpeningCapability, "_dialogue_state_snapshot", return_value=birch_message),
            patch("modules.modes.opening.is_emerald_field_dialogue_advanceable", return_value=True),
        ):
            list(_advance_scripted_input())
        emulator.press_button.assert_called_once_with("A")

    def test_name_prompt_transition_issues_one_input_and_reobserves(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        prompt = self.observation(
            OpeningSequenceState.SCRIPTED_INTRO,
            gender_task="Task_NewGameBirchSpeech_WaitPressBeforeNameChoice",
        )
        naming = self.observation(OpeningSequenceState.PLAYER_NAMING)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=[prompt, naming]),
            patch("modules.nuzlocke.emerald_capabilities._advance_name_prompt", return_value=iter([None])) as advance,
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
        advance.assert_called_once()

    def test_naming_screen_is_recognized_without_entering_name(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.PLAYER_NAMING))
        self.assertIs(action, EmeraldCampaignAction.WAIT)

    def test_player_naming_target_selects_reusable_name_capability(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        observation = self.observation(OpeningSequenceState.PLAYER_NAMING)
        observation = replace(
            observation,
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, True),
        )
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.ENTER_NAME)

    def test_naming_capability_waits_until_keyboard_is_ready(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        observation = self.observation(OpeningSequenceState.PLAYER_NAMING)
        observation = replace(
            observation,
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, False),
        )
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.WAIT)

    def test_player_confirmation_yes_and_no_are_observed_actions(self):
        for choice, expected in (
            (EmeraldConfirmationChoice.YES, EmeraldCampaignAction.CONFIRM_PLAYER_NAME),
            (EmeraldConfirmationChoice.NO, EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES),
        ):
            confirmation = EmeraldConfirmationObservation(
                True,
                choice,
                True,
                EmeraldConfirmationContext.PLAYER_NAME,
            )
            observation = replace(
                self.observation(OpeningSequenceState.SCRIPTED_INTRO),
                confirmation_observation=confirmation,
            )
            self.assertIs(choose_emerald_campaign_action(observation), expected)

    def test_unready_or_unknown_selected_confirmation_waits(self):
        for confirmation in (
            EmeraldConfirmationObservation(
                True,
                EmeraldConfirmationChoice.NO,
                False,
                EmeraldConfirmationContext.PLAYER_NAME,
            ),
            EmeraldConfirmationObservation(
                True,
                None,
                True,
                EmeraldConfirmationContext.PLAYER_NAME,
            ),
        ):
            observation = replace(
                self.observation(OpeningSequenceState.SCRIPTED_INTRO),
                confirmation_observation=confirmation,
            )
            self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.WAIT)

    def test_naming_capability_reobserves_from_transition_to_ready_keyboard(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        not_ready = replace(
            self.observation(OpeningSequenceState.PLAYER_NAMING),
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, False),
        )
        ready = replace(
            not_ready,
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, True),
        )
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=[not_ready, ready]),
            patch("modules.nuzlocke.emerald_capabilities._naming_input") as naming_input,
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
        naming_input.assert_called_once_with("ASH")

    def test_naming_input_reobserves_before_repeating(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        naming = EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 1, True)
        first = self.observation(OpeningSequenceState.PLAYER_NAMING)
        first = replace(first, naming_observation=naming)
        same = replace(first, naming_observation=naming)
        ended = self.observation(OpeningSequenceState.SCRIPTED_INTRO)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=[first, same, ended]),
            patch("modules.nuzlocke.emerald_capabilities._naming_input") as naming_input,
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
            next(generator)
        naming_input.assert_called_once_with("ASH")

    def test_no_to_yes_then_confirm_uses_one_input_per_observation(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        no = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.NO,
            True,
            EmeraldConfirmationContext.PLAYER_NAME,
        )
        yes = replace(no, selected=EmeraldConfirmationChoice.YES)
        first = replace(self.observation(OpeningSequenceState.SCRIPTED_INTRO), confirmation_observation=no)
        second = replace(first, confirmation_observation=yes)
        exited = self.observation(OpeningSequenceState.SCRIPTED_INTRO)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=[first, second, exited],
            ),
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            self.assertEqual(emulator.press_button.call_args_list, [unittest.mock.call("Up")])
            next(generator)
            self.assertEqual(
                emulator.press_button.call_args_list,
                [unittest.mock.call("Up"), unittest.mock.call("A")],
            )
            next(generator)
        self.assertEqual(emulator.press_button.call_count, 2)

    def test_confirmation_exit_accepts_real_birch_post_name_task(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        confirmation = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.PLAYER_NAME,
        )
        before = replace(
            self.observation(OpeningSequenceState.MAIN_MENU),
            confirmation_observation=confirmation,
        )
        after = self.observation(
            OpeningSequenceState.MAIN_MENU,
            gender_task="Task_NewGameBirchSpeech_SlidePlatformAway2",
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=[before, after],
            ),
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
        emulator.press_button.assert_called_once_with("A")

    def test_unexpected_post_gender_state_waits_for_reobservation(self):
        action = choose_emerald_campaign_action(
            self.observation(OpeningSequenceState.SCRIPTED_INTRO, gender_task=None, dialogue=False)
        )
        self.assertIs(action, EmeraldCampaignAction.WAIT)


if __name__ == "__main__":
    unittest.main()
