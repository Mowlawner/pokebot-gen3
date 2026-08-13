from tests.utility import (
    BotTestCase,
    with_save_state,
    with_frame_timeout,
    set_next_choice,
    set_next_rng_seed,
)
from unittest.mock import ANY, Mock, patch


class TestStarter(BotTestCase):
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
                    patch("modules.modes.starters.ask_for_choice", return_value=choice),
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

    def test_opening_handoff_returns_to_manual_after_starter_sequence(self):
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
        starter_context.set_manual_mode.assert_called_once_with()

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
