import os
import tempfile
import unittest
from unittest.mock import patch

from modules import stall_dump


class StallDumpTests(unittest.TestCase):
    def tearDown(self):
        stall_dump.disable()

    def test_disabled_without_the_environment_variable(self):
        with patch("modules.stall_dump.faulthandler.dump_traceback_later") as dump:
            self.assertIsNone(stall_dump.enable_from_environment({}))
        dump.assert_not_called()

    def test_enabled_dumps_repeatedly_to_the_requested_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stalls.txt")
            with patch("modules.stall_dump.faulthandler.dump_traceback_later") as dump:
                result = stall_dump.enable_from_environment(
                    {"POKEBOT_STALL_DUMP": path, "POKEBOT_STALL_DUMP_SECONDS": "30"}
                )
                stall_dump.disable()
            self.assertEqual(result, path)
            self.assertTrue(os.path.exists(path))
        args, kwargs = dump.call_args
        self.assertEqual(args, (30.0,))
        self.assertTrue(kwargs["repeat"])

    def test_invalid_interval_falls_back_to_the_default(self):
        for value in ("abc", "0", "-5"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory, patch(
                "modules.stall_dump.faulthandler.dump_traceback_later"
            ) as dump:
                stall_dump.enable_from_environment(
                    {
                        "POKEBOT_STALL_DUMP": os.path.join(directory, "s.txt"),
                        "POKEBOT_STALL_DUMP_SECONDS": value,
                    }
                )
                stall_dump.disable()
                self.assertEqual(dump.call_args.args, (stall_dump.DEFAULT_INTERVAL_SECONDS,))

    def test_unwritable_path_disables_quietly(self):
        with patch("modules.stall_dump.faulthandler.dump_traceback_later") as dump:
            result = stall_dump.enable_from_environment({"POKEBOT_STALL_DUMP": "/proc/not/writable/stalls.txt"})
        self.assertIsNone(result)
        dump.assert_not_called()
