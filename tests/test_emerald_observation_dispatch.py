import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.memory import GameState
from modules.goals import SemanticTarget
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
from modules.nuzlocke.emerald_menu_observation import (
    EmeraldMainMenuItem,
    EmeraldMenuKind,
    EmeraldMenuObservation,
)
from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget
from modules.nuzlocke.emerald_observation import EmeraldObservation


class EmeraldObservationDispatchTests(unittest.TestCase):
    def observation(self, **changes):
        values = dict(
            game_state=GameState.OVERWORLD,
            ui_mode="OVERWORLD",
            map_id=((1, 1)),
            coordinates=(3, 4),
            controllable=True,
            dialogue_actionable=False,
            dialogue_lifecycle_active=False,
            menu=None,
            naming=None,
            confirmation=None,
            gender_task=None,
            rom_owned_movement=False,
            campaign_facts=(
                ("text_speed_fast", True),
                ("new_game_setup_complete", False),
                ("wall_clock_set", False),
            ),
            overworld=object(),
        )
        values.update(changes)
        return EmeraldObservation(**values)

    def test_dialogue_is_global_highest_priority(self):
        self.assertIs(
            choose_emerald_observation_action(self.observation(dialogue_actionable=True, rom_owned_movement=True)),
            EmeraldCampaignAction.ADVANCE_DIALOGUE,
        )

    def test_ui_actions_preempt_navigation_and_rom_movement(self):
        cases = (
            (
                dict(naming=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 1, True)),
                EmeraldCampaignAction.ENTER_NAME,
            ),
            (
                dict(
                    confirmation=EmeraldConfirmationObservation(
                        True, EmeraldConfirmationChoice.YES, True, EmeraldConfirmationContext.PLAYER_NAME
                    )
                ),
                EmeraldCampaignAction.CONFIRM_PLAYER_NAME,
            ),
            (
                dict(
                    menu=EmeraldMenuObservation(
                        EmeraldMenuKind.MAIN_MENU, 0, EmeraldMainMenuItem.NEW_GAME, True, "Task_HandleMainMenuInput", 0
                    )
                ),
                EmeraldCampaignAction.START_NEW_GAME,
            ),
        )
        for fields, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(choose_emerald_observation_action(self.observation(**fields)), expected)

    def test_scripted_movement_waits(self):
        self.assertIs(
            choose_emerald_observation_action(self.observation(rom_owned_movement=True)),
            EmeraldCampaignAction.WAIT,
        )

    def test_clock_interaction_is_currently_actionable(self):
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    campaign_facts=(("wall_clock_set", False),),
                    clock_interaction=("Task_SetClock_HandleInput", 0, 12, 30, 0),
                )
            ),
            EmeraldCampaignAction.ADVANCE_CLOCK,
        )

    def test_wall_clock_target_preempts_generic_overworld_dispatch(self):
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    objective_id="set_wall_clock",
                    semantic_target=SemanticTarget.interaction((1, 3), "WallClockScript"),
                    clock_target=((4, 4), "Up"),
                )
            ),
            EmeraldCampaignAction.NAVIGATE_TO_CLOCK,
        )

    def test_observed_wall_clock_affordance_uses_generic_interaction_dispatch(self):
        trigger = SimpleNamespace(
            trigger_id="bg:clock",
            affordance_id="WallClockScript",
            script_symbol="WallClockScript",
            condition_active=True,
            locations=frozenset({((1, 3), (3, 4))}),
            activation_locations=frozenset({((1, 3), (3, 4))}),
        )
        overworld = SimpleNamespace(map_id=(1, 3), triggers=(trigger,))
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    objective_id="set_wall_clock",
                    semantic_target=SemanticTarget.interaction((1, 3), "WallClockScript"),
                    clock_target=((4, 4), "Up"),
                    overworld=overworld,
                )
            ),
            EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD,
        )

    def test_clock_confirmation_is_actionable_while_completion_flag_is_false(self):
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    campaign_facts=(("wall_clock_set", False),),
                    clock_interaction=("Task_SetClock_HandleConfirmInput", 0, 12, 30, 1),
                )
            ),
            EmeraldCampaignAction.CONFIRM_CLOCK,
        )

    def test_completed_clock_flag_prevents_revisiting_clock_task(self):
        observation = self.observation(
            campaign_facts=(("wall_clock_set", True),),
            objective_id="set_wall_clock",
            clock_interaction=("Task_SetClock_HandleInput", 0, 12, 30, 0),
            clock_target=((4, 4), "Up"),
        )
        self.assertIs(
            choose_emerald_observation_action(observation),
            EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD,
        )

    def test_completed_clock_flag_prevents_revisiting_clock_tile(self):
        observation = self.observation(
            campaign_facts=(("wall_clock_set", True),),
            objective_id="set_wall_clock",
            clock_target=((3, 4), "Up"),
        )
        self.assertIs(
            choose_emerald_observation_action(observation),
            EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD,
        )

    def test_dialogue_preempts_clock_action(self):
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    dialogue_actionable=True,
                    clock_interaction=("Task_SetClock_HandleInput", 0, 12, 30, 0),
                )
            ),
            EmeraldCampaignAction.ADVANCE_DIALOGUE,
        )

    def test_observed_clock_position_selects_interaction_or_facing(self):
        exact = self.observation(
            objective_id="set_wall_clock",
            coordinates=(4, 4),
            clock_target=((4, 4), "Up"),
            overworld=SimpleNamespace(facing="Up"),
        )
        self.assertIs(choose_emerald_observation_action(exact), EmeraldCampaignAction.INTERACT_CLOCK)
        facing = self.observation(
            objective_id="set_wall_clock",
            coordinates=(4, 4),
            clock_target=((4, 4), "Up"),
            overworld=SimpleNamespace(facing="Left"),
        )
        self.assertIs(choose_emerald_observation_action(facing), EmeraldCampaignAction.FACE_CLOCK)

    def test_gender_selection_is_observed_without_opening_history(self):
        self.assertIs(
            choose_emerald_observation_action(self.observation(gender_task="Task_NewGameBirchSpeech_ChooseGender")),
            EmeraldCampaignAction.CHOOSE_GENDER,
        )

    def test_naming_target_and_keyboard_readiness_are_current_facts(self):
        naming = EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 1, False)
        self.assertIs(
            choose_emerald_observation_action(self.observation(naming=naming)),
            EmeraldCampaignAction.WAIT,
        )

    def test_confirmation_no_is_current_action(self):
        confirmation = EmeraldConfirmationObservation(
            True, EmeraldConfirmationChoice.NO, True, EmeraldConfirmationContext.PLAYER_NAME
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(confirmation=confirmation)),
            EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES,
        )

    def test_consequence_policy_moves_to_objective_advancing_choice(self):
        from modules.semantic_choices import ChoiceConsequence, ConsequenceKnowledge

        confirmation = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.NO,
            True,
            EmeraldConfirmationContext.UNKNOWN,
            "Task_SetClock_HandleConfirmInput",
            "Is this the correct time?",
            ("YES", "NO"),
            "PlayersHouse_2F_EventScript_SetWallClock",
            "PlayersHouse_2F_EventScript_SetWallClock",
            (
                ChoiceConsequence("YES", ConsequenceKnowledge.KNOWN, {"wall_clock_set": True}),
                ChoiceConsequence("NO", ConsequenceKnowledge.KNOWN, {"wall_clock_set": False}),
            ),
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(confirmation=confirmation)),
            EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES,
        )

    def test_map_change_is_a_new_selection(self):
        first = self.observation(map_id=(1, 1))
        second = self.observation(map_id=(2, 3), coordinates=(8, 2), rom_owned_movement=True)
        self.assertIs(choose_emerald_observation_action(first), EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD)
        self.assertIs(choose_emerald_observation_action(second), EmeraldCampaignAction.WAIT)

    def test_one_dialogue_input_then_yield_and_reobserve(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(),
            rom=SimpleNamespace(is_emerald=True),
            debug=False,
        )
        dialogue = self.observation(dialogue_actionable=True)
        after = self.observation(dialogue_actionable=False, overworld=None)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", side_effect=[dialogue, after]),
            patch(
                "modules.nuzlocke.emerald_capabilities._advance_scripted_input",
                side_effect=lambda _lifecycle_active: (emulator.press_button("A") for _ in [0]),
            ),
        ):
            execution = observation_driven_emerald_campaign()
            next(execution)
            self.assertEqual(emulator.press_button.call_count, 1)
            next(execution)

    def test_dialogue_executor_contract_issues_a_once(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(),
            rom=SimpleNamespace(is_emerald=True),
            debug=False,
        )
        with patch("modules.modes.opening.context", fake_context):
            from modules.modes.opening import _advance_scripted_input

            with patch.object(
                __import__("modules.modes.opening", fromlist=["EmeraldOpeningCapability"]).EmeraldOpeningCapability,
                "_dialogue_state_snapshot",
                return_value=(
                    False,
                    None,
                    True,
                    "WaitForAorBPress",
                    "Std_MsgboxDefault",
                    True,
                    False,
                    False,
                    None,
                    (),
                    None,
                    None,
                    None,
                    None,
                ),
            ), patch("modules.modes.opening.is_emerald_field_dialogue_advanceable", return_value=True):
                step = _advance_scripted_input()
                next(step)
        emulator.press_button.assert_called_once_with("A")

    def test_title_action_requires_current_actionable_title_observation(self):
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(game_state=GameState.TITLE_SCREEN, title_actionable=True, overworld=None)
            ),
            EmeraldCampaignAction.ADVANCE_TITLE,
        )
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(game_state=GameState.TITLE_SCREEN, title_actionable=False, overworld=None)
            ),
            EmeraldCampaignAction.WAIT,
        )

    def test_main_menu_policy_uses_current_cursor_and_selected_item(self):
        options = EmeraldMenuObservation(
            EmeraldMenuKind.MAIN_MENU, 0, EmeraldMainMenuItem.NEW_GAME, True, "Task_HandleMainMenuInput", 0
        )
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(menu=options, campaign_facts=(("text_speed_fast", False),))
            ),
            EmeraldCampaignAction.ENTER_OPTIONS,
        )
        new_game = EmeraldMenuObservation(
            EmeraldMenuKind.MAIN_MENU, 1, EmeraldMainMenuItem.NEW_GAME, True, "Task_HandleMainMenuInput", 1
        )
        self.assertIs(
            choose_emerald_observation_action(self.observation(menu=new_game)),
            EmeraldCampaignAction.START_NEW_GAME,
        )
        options_selected = EmeraldMenuObservation(
            EmeraldMenuKind.MAIN_MENU, 1, EmeraldMainMenuItem.OPTIONS, True, "Task_HandleMainMenuInput", 1
        )
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(menu=options_selected, campaign_facts=(("text_speed_fast", False),))
            ),
            EmeraldCampaignAction.ENTER_OPTIONS,
        )

    def test_options_policy_starts_from_any_cursor_position(self):
        menu = EmeraldMenuObservation(
            EmeraldMenuKind.OPTIONS_MENU,
            3,
            None,
            True,
            "Task_OptionMenuProcessInput",
            3,
            raw_option_values=(2, 0, 0, 0, 0, 0),
        )
        observation = self.observation(
            game_state=GameState.OPTIONS_MENU,
            menu=menu,
            campaign_facts=(("text_speed_fast", False),),
        )
        self.assertIs(choose_emerald_observation_action(observation), EmeraldCampaignAction.ADVANCE_TEXT_SPEED)

    def test_set_battle_style_rule_advances_style_after_fast_text(self):
        menu = EmeraldMenuObservation(
            EmeraldMenuKind.OPTIONS_MENU,
            3,
            None,
            True,
            "Task_OptionMenuProcessInput",
            3,
            raw_option_values=(2, 0, 0, 0, 0, 0),
        )
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    game_state=GameState.OPTIONS_MENU,
                    menu=menu,
                    campaign_facts=(
                        ("text_speed_fast", False),
                        ("set_battle_style_required", True),
                    ),
                )
            ),
            EmeraldCampaignAction.ADVANCE_BATTLE_STYLE,
        )

        configured_menu = EmeraldMenuObservation(
            EmeraldMenuKind.OPTIONS_MENU,
            3,
            None,
            True,
            "Task_OptionMenuProcessInput",
            3,
            raw_option_values=(2, 0, 1, 0, 0, 0),
        )
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(
                    game_state=GameState.OPTIONS_MENU,
                    menu=configured_menu,
                    campaign_facts=(
                        ("text_speed_fast", True),
                        ("set_battle_style_required", True),
                    ),
                )
            ),
            EmeraldCampaignAction.EXIT_OPTIONS,
        )

    def test_options_ready_configuration_returns_to_menu_from_fresh_observation(self):
        menu = EmeraldMenuObservation(
            EmeraldMenuKind.OPTIONS_MENU,
            0,
            None,
            True,
            "Task_OptionMenuProcessInput",
            0,
            raw_option_values=(2, 0, 0, 0, 0, 0),
        )
        observation = self.observation(
            game_state=GameState.OPTIONS_MENU,
            menu=menu,
            campaign_facts=(("text_speed_fast", True),),
        )
        self.assertIs(choose_emerald_observation_action(observation), EmeraldCampaignAction.EXIT_OPTIONS)

    def test_naming_observation_takes_ownership_after_new_game_transition(self):
        naming = EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 1, True)
        self.assertIs(
            choose_emerald_observation_action(
                self.observation(game_state=GameState.NAMING_SCREEN, naming=naming, overworld=None)
            ),
            EmeraldCampaignAction.ENTER_NAME,
        )

    def test_title_executor_issues_one_input_then_observes_again(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        title = self.observation(game_state=GameState.TITLE_SCREEN, title_actionable=True, overworld=None)
        menu = self.observation(game_state=GameState.MAIN_MENU, overworld=None)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._emerald_observation", side_effect=[title, menu]),
        ):
            execution = observation_driven_emerald_campaign()
            next(execution)
            self.assertEqual([call.args[0] for call in emulator.press_button.call_args_list], ["A"])
            next(execution)
        self.assertEqual(emulator.press_button.call_count, 1)


if __name__ == "__main__":
    unittest.main()
