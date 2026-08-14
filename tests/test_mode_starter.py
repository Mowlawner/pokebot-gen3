from tests.utility import (
    BotTestCase,
    with_save_state,
    with_frame_timeout,
    set_next_choice,
    set_next_rng_seed,
)
from unittest.mock import ANY, Mock, patch


class TestStarter(BotTestCase):
    def test_configured_specific_starter_bypasses_interactive_prompt(self):
        from modules.modes.starters import _resolve_hoenn_starter, run_rse_hoenn

        class BagTaskReached(Exception):
            pass

        starter_context = Mock()
        starter_context.bot_mode = "Starters"
        starter_context.rom.is_rs = False
        starter_context.config.start_game.starter = "Torchic"
        with (
            patch("modules.modes.starters.context", starter_context),
            patch("modules.modes.starters.ask_for_choice") as ask_for_choice,
            patch("modules.modes.starters.get_player_avatar", return_value=Mock(local_coordinates=(7, 15))),
            patch("modules.modes.starters.wait_until_task_is_active", side_effect=BagTaskReached),
        ):
            with self.assertRaises(BagTaskReached):
                next(run_rse_hoenn(lambda: None, reset_before_selection=False))

        self.assertEqual(_resolve_hoenn_starter("Torchic"), "Torchic")
        ask_for_choice.assert_not_called()

    def test_configured_random_starter_bypasses_prompt_and_uses_hoenn_pool(self):
        from modules.modes.starters import _resolve_hoenn_starter, run_rse_hoenn

        class BagTaskReached(Exception):
            pass

        starter_context = Mock()
        starter_context.bot_mode = "Starters"
        starter_context.rom.is_rs = False
        starter_context.config.start_game.starter = "Random"
        with (
            patch("modules.modes.starters.context", starter_context),
            patch("modules.modes.starters.ask_for_choice") as ask_for_choice,
            patch("modules.modes.starters.random.choice", return_value="Mudkip") as random_choice,
            patch("modules.modes.starters.get_player_avatar", return_value=Mock(local_coordinates=(7, 15))),
            patch("modules.modes.starters.wait_until_task_is_active", side_effect=BagTaskReached),
        ):
            with self.assertRaises(BagTaskReached):
                next(run_rse_hoenn(lambda: None, reset_before_selection=False))

            self.assertEqual(_resolve_hoenn_starter("Random"), "Mudkip")
            random_choice.assert_called_with(("Treecko", "Torchic", "Mudkip"))

        ask_for_choice.assert_not_called()

    def test_opening_handoff_uses_normal_bag_task_for_configured_and_random_choices(self):
        from modules.modes.starters import run_rse_hoenn

        class BagTaskReached(Exception):
            pass

        for choice in ("Treecko", "Random"):
            with self.subTest(choice=choice):
                starter_context = Mock()
                starter_context.bot_mode = "Starters"
                starter_context.rom.is_rs = False
                with (
                    patch("modules.modes.starters.context", starter_context),
                    patch("modules.modes.starters.ask_for_choice", return_value=choice) as ask_for_choice,
                    patch(
                        "modules.modes.starters.get_player_avatar",
                        return_value=Mock(local_coordinates=(7, 15)),
                    ),
                    patch("modules.modes.starters.ensure_facing_direction", return_value=iter(())),
                    patch(
                        "modules.modes.starters.wait_until_task_is_active",
                        side_effect=BagTaskReached,
                    ) as wait_for_bag,
                ):
                    with self.assertRaises(BagTaskReached):
                        next(run_rse_hoenn(lambda: None, reset_before_selection=False))

                    wait_for_bag.assert_called_once_with("Task_HandleStarterChooseInput", "A")
                    ask_for_choice.assert_called_once()

    def test_opening_handoff_returns_to_opening_controller_after_starter_sequence(self):
        from modules.modes.starters import StartersMode

        starter_context = Mock()
        starter_context.bot_mode = "Starters"
        with (
            patch("modules.modes.opening.consume_starter_handoff", return_value=True),
            patch("modules.modes.starters.context", starter_context),
            patch("modules.modes.starters.run_rse_hoenn", return_value=iter(())) as run_starter,
        ):
            list(StartersMode().run())

        run_starter.assert_called_once_with(
            ANY,
            reset_before_selection=False,
        )
        self.assertEqual(starter_context.bot_mode, "Start New Game")
        starter_context.set_manual_mode.assert_not_called()

    def test_opening_handoff_delegates_battle_to_default_handler(self):
        from modules.modes import BattleAction
        from modules.modes.starters import StartersMode

        starters_mode = StartersMode()
        starters_mode._opening_handoff_active = True

        self.assertEqual(starters_mode.on_battle_started(None), BattleAction.Fight)

    def test_opening_hoenn_generator_exits_visibility_loop_after_delegated_battle(self):
        from modules.memory import GameState
        from modules.modes.starters import run_rse_hoenn

        starter_context = Mock()
        starter_context.bot_mode = "Starters"
        starter_context.rom.is_emerald = True
        starter_context.rom.is_rs = False
        starter_context.config.start_game.starter = "Torchic"
        with (
            patch("modules.modes.starters.context", starter_context),
            patch("modules.modes.starters.get_player_avatar", return_value=Mock(local_coordinates=(7, 15))),
            patch("modules.modes.starters.ensure_facing_direction", return_value=iter(())),
            patch("modules.modes.starters.wait_until_task_is_active", return_value=iter(())),
            patch("modules.modes.starters.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.starters.get_main_battle_callback",
                return_value="ReturnFromBattleToOverworld",
            ),
            patch("modules.modes.starters.handle_encounter") as handle_encounter,
            patch("modules.modes.starters._report_starters_state"),
            patch("modules.modes.starters.diagnostic_print"),
        ):
            generator = run_rse_hoenn(lambda: Mock(), reset_before_selection=False)
            with self.assertRaises(StopIteration):
                next(generator)

        handle_encounter.assert_called_once()

    def test_saved_game_starter_mode_keeps_custom_battle_handler(self):
        from modules.modes import BattleAction
        from modules.modes.starters import StartersMode

        starters_mode = StartersMode()

        self.assertEqual(starters_mode.on_battle_started(None), BattleAction.CustomAction)

    def test_battle_listener_delegates_opening_battle_to_fight_controller(self):
        from modules.memory import GameState
        from modules.modes import BattleAction, FrameInfo
        from modules.modes._listeners import BattleListener
        from modules.battle_state import EncounterType

        listener = BattleListener()
        bot_mode = Mock()
        bot_mode.on_battle_started.return_value = BattleAction.Fight
        fake_context = Mock(bot_mode="Starters", controller_stack=[])
        frame = FrameInfo(1, GameState.BATTLE, [], [], [], None)

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_encounter_type", return_value=EncounterType.Trainer),
            patch("modules.modes._listeners.get_battle_state", return_value=Mock(type=0)),
            patch("modules.modes._listeners.get_game_state", return_value=GameState.BATTLE),
            patch.object(listener, "fight", return_value=iter(())) as fight,
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            listener.handle_frame(bot_mode, frame)
            listener.handle_frame(bot_mode, frame)

        fight.assert_called_once()
        self.assertEqual(len(fake_context.controller_stack), 1)

    @with_save_state(
        [
            "emerald/in_front_of_starter_pokemon_bag.ss1",
            "ruby/in_front_of_starter_pokemon_bag.ss1",
            "firered/in_front_of_starter_pokemon_table.ss1",
        ]
    )
    @with_frame_timeout(10000)
    def test_it_runs_three_times(self):
        from modules.modes import BattleAction
        from modules.modes.starters import StartersMode

        if self.rom.is_rse:
            expected_species = "Mudkip"
            set_next_choice(expected_species)
        else:
            expected_species = "Squirtle"

        starters_mode = StartersMode()
        self.bot_mode.set_on_battle_started(lambda e: BattleAction.CustomAction)

        for _ in starters_mode.run():
            if len(self.stats.logged_encounters) > 3:
                for encounter in self.stats.logged_encounters:
                    self.assertEqual(expected_species, encounter.species_name)
                return None
            else:
                yield

    @with_save_state(
        [
            "emerald/in_front_of_starter_pokemon_bag.ss1",
            "ruby/in_front_of_starter_pokemon_bag.ss1",
            "firered/in_front_of_starter_pokemon_table.ss1",
        ]
    )
    @with_frame_timeout(10000)
    def test_it_stops_when_encountering_shiny(self):
        from modules.context import context
        from modules.modes import BattleAction
        from modules.modes.starters import StartersMode

        if self.rom.is_emerald:
            rng_seed = 0x2FAB1EEB
            expected_species = "Treecko"
        elif self.rom.is_ruby:
            rng_seed = 0x327B87DF
            expected_species = "Treecko"
        elif self.rom.is_fr:
            rng_seed = 0x2031F790
            expected_species = "Squirtle"
        else:
            self.fail("No pre-calculated shiny seed available for this game.")

        set_next_choice(expected_species)
        set_next_rng_seed(rng_seed)

        starters_mode = StartersMode()
        self.bot_mode.set_on_battle_started(lambda e: BattleAction.CustomAction)
        self.bot_mode.allow_ending_on_manual_mode = True

        for _ in starters_mode.run():
            if self.stats.last_encounter is not None:
                self.assertEqual(expected_species, self.stats.last_encounter.species_name)
                self.assertTrue(
                    self.stats.last_encounter.is_shiny,
                    "Encountered starter Pokémon was not shiny.",
                )
                self.assertIsInManualMode()
                return
            yield
