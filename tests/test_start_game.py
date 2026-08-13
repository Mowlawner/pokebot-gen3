import random
import unittest
from pathlib import Path


class TestStartGameNameResolution(unittest.TestCase):
    def test_random_generation_is_deterministic_with_seeded_rng(self):
        from modules.start_game import generate_gibberish_player_name

        first = generate_gibberish_player_name(random.Random(1234))
        second = generate_gibberish_player_name(random.Random(1234))

        self.assertEqual(first, second)

    def test_random_generation_uses_permitted_keyboard_characters(self):
        from modules.keyboard import get_current_keyboard_layout
        from modules.start_game import generate_gibberish_player_name

        name = generate_gibberish_player_name(random.Random(1))
        permitted = {character for character in get_current_keyboard_layout().valid_characters if character.isalpha()}

        self.assertTrue(name)
        self.assertTrue(set(name) <= permitted)

    def test_random_generation_enforces_explicit_boundary(self):
        from modules.start_game import generate_gibberish_player_name

        name = generate_gibberish_player_name(random.Random(2), max_length=3)

        self.assertLessEqual(len(name), 3)

    def test_gibberish_names_never_exceed_seven_characters(self):
        from modules.start_game import PLAYER_NAME_MAX_LENGTH, generate_gibberish_player_name

        for seed in range(20):
            self.assertLessEqual(
                len(generate_gibberish_player_name(random.Random(seed))),
                PLAYER_NAME_MAX_LENGTH,
            )

    def test_human_name_accepts_seven_characters(self):
        from modules.start_game import PlayerGender, generate_human_player_name

        with unittest.mock.patch("modules.start_game._get_human_first_name", return_value="Rachel"):
            self.assertEqual(generate_human_player_name(PlayerGender.FEMALE), "Rachel")

    def test_human_name_retries_long_names_without_truncating(self):
        from modules.start_game import PlayerGender, generate_human_player_name

        with unittest.mock.patch(
            "modules.start_game._get_human_first_name", side_effect=["Christopher", "Andrew"]
        ) as get_name:
            self.assertEqual(generate_human_player_name(PlayerGender.MALE), "Andrew")
            self.assertEqual(get_name.call_count, 2)

    def test_explicit_name_longer_than_seven_is_rejected(self):
        from modules.start_game import resolve_start_game_initialization

        with self.assertRaises(ValueError):
            resolve_start_game_initialization("ChristoXX", "male", random.Random(1))

    def test_human_name_resolution_uses_resolved_gender(self):
        from modules.start_game import PlayerGender, resolve_start_game_initialization

        with unittest.mock.patch("modules.start_game._get_human_first_name", return_value="Rachel") as get_name:
            resolved = resolve_start_game_initialization("random", "female", random.Random(1))

        self.assertEqual(resolved.gender, PlayerGender.FEMALE)
        self.assertEqual(resolved.name, "Rachel")
        get_name.assert_called_once_with(PlayerGender.FEMALE)

    def test_random_gender_resolves_to_one_concrete_gender(self):
        from modules.start_game import PlayerGender, resolve_start_game_initialization

        with unittest.mock.patch("modules.start_game._get_human_first_name", return_value="Andrew"):
            resolved = resolve_start_game_initialization("random", "random", random.Random(1))

        self.assertIn(resolved.gender, (PlayerGender.MALE, PlayerGender.FEMALE))
        self.assertEqual(resolved.name, "Andrew")

    def test_explicit_name_policy_is_preserved(self):
        from modules.start_game import PlayerNamePolicy, player_name_policy, resolve_start_game_initialization

        self.assertIs(player_name_policy("ASH"), PlayerNamePolicy.EXPLICIT)
        self.assertEqual(
            resolve_start_game_initialization("ASH", "female", random.Random(1)).name,
            "ASH",
        )

    def test_random_name_policy_is_resolved(self):
        from modules.start_game import PlayerNamePolicy, player_name_policy, resolve_start_game_initialization

        self.assertIs(player_name_policy("random"), PlayerNamePolicy.HUMAN_RANDOM)
        with unittest.mock.patch("modules.start_game._get_human_first_name", return_value="Andrew"):
            self.assertEqual(
                resolve_start_game_initialization("random", "male", random.Random(1)).name,
                "Andrew",
            )

    def test_gibberish_policy_remains_separate(self):
        from modules.start_game import PlayerNamePolicy, player_name_policy, resolve_start_game_initialization

        self.assertIs(player_name_policy("gibberish"), PlayerNamePolicy.GIBBERISH)
        resolved = resolve_start_game_initialization("gibberish", "male", random.Random(1))
        self.assertLessEqual(len(resolved.name), 7)

    def test_configuration_resolves_a_name_once_for_a_mode_instance(self):
        import types
        from unittest.mock import patch

        from modules.modes.opening import EmeraldOpeningMode

        opening_context = types.SimpleNamespace(
            config=types.SimpleNamespace(
                start_game=types.SimpleNamespace(player_name="random", player_gender="random")
            ),
            rom=types.SimpleNamespace(is_emerald=True),
        )
        with patch("modules.modes.opening.context", opening_context), patch(
            "modules.modes.opening.resolve_start_game_initialization",
            return_value=types.SimpleNamespace(gender="male", name="ONCE"),
        ) as resolve:
            mode = EmeraldOpeningMode()
            self.assertEqual(mode._resolved_player_name, "ONCE")
            self.assertEqual(mode._resolved_player_gender, "male")
            self.assertEqual(mode._resolved_player_name, "ONCE")

        self.assertEqual(resolve.call_args.args[:2], ("random", "random"))
        self.assertIsNotNone(resolve.call_args.args[2])

    def test_configuration_defaults_and_parsing(self):
        from modules.config import Config
        from modules.config.schemas_v1 import StartGame

        config = Config(Path(__file__).parent.parent / "modules" / "config" / "templates")

        self.assertEqual(config.start_game.player_name, "random")
        self.assertEqual(config.start_game.player_gender, "random")
        self.assertEqual(config.start_game.clock_time_mode.value, "system_time")
        self.assertEqual(StartGame(player_name="ASH").player_name, "ASH")
