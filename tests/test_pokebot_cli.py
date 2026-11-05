import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPokebotCli(unittest.TestCase):
    def test_debug_log_path_is_parsed_without_enabling_stutter_trace(self):
        import pokebot

        path = Path("profiles") / "run.log"
        with patch.object(sys, "argv", ["pokebot.py", "--debug-log", str(path), "--debug-trace"]):
            settings = pokebot.parse_arguments(("Manual",))

        self.assertEqual(settings.debug_log_path, path)
        self.assertTrue(settings.debug)
        self.assertTrue(settings.debug_trace)
        self.assertFalse(settings.stutter_trace)


if __name__ == "__main__":
    unittest.main()
