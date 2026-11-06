import os
import tempfile
import threading
import time
import unittest

from modules import stall_dump


class StallDumpTests(unittest.TestCase):
    def tearDown(self):
        stall_dump.disable()

    def test_disabled_without_the_environment_variable(self):
        self.assertIsNone(stall_dump.enable_from_environment({}))
        self.assertIsNone(stall_dump._thread)

    def test_enabled_dumps_repeatedly_to_the_requested_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stalls.txt")
            result = stall_dump.enable_from_environment(
                {"POKEBOT_STALL_DUMP": path, "POKEBOT_STALL_DUMP_SECONDS": "0.05"}
            )
            self.assertEqual(result, path)
            deadline = time.monotonic() + 5
            text = ""
            while time.monotonic() < deadline:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
                if text.count("=== stack dump") >= 2:
                    break
                time.sleep(0.05)
            stall_dump.disable()
        self.assertGreaterEqual(text.count("=== stack dump"), 2)
        # The main thread's stack is included, so a stall shows where it is.
        self.assertIn("test_enabled_dumps_repeatedly_to_the_requested_file", text)
        self.assertNotIn("pokebot-stall-dump", text)

    def test_invalid_interval_falls_back_to_the_default(self):
        for value in ("abc", "0", "-5"):
            with self.subTest(value=value):
                self.assertEqual(
                    stall_dump._parse_interval({"POKEBOT_STALL_DUMP_SECONDS": value}),
                    stall_dump.DEFAULT_INTERVAL_SECONDS,
                )
        self.assertEqual(stall_dump._parse_interval({"POKEBOT_STALL_DUMP_SECONDS": "30"}), 30.0)
        self.assertEqual(stall_dump._parse_interval({}), stall_dump.DEFAULT_INTERVAL_SECONDS)

    def test_unwritable_path_disables_quietly(self):
        result = stall_dump.enable_from_environment(
            {"POKEBOT_STALL_DUMP": os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "stalls.txt")}
        )
        self.assertIsNone(result)
        self.assertIsNone(stall_dump._thread)

    def test_disable_stops_the_watchdog_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            stall_dump.enable_from_environment(
                {"POKEBOT_STALL_DUMP": os.path.join(directory, "s.txt"), "POKEBOT_STALL_DUMP_SECONDS": "0.05"}
            )
            stall_dump.disable()
        self.assertFalse(any(thread.name == "pokebot-stall-dump" for thread in threading.enumerate()))

    def test_a_python_busy_loop_in_another_thread_is_visible(self):
        stop = threading.Event()

        def spin_in_busy_loop():
            while not stop.is_set():
                pass

        worker = threading.Thread(target=spin_in_busy_loop, name="spinner", daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "s.txt")
                with open(path, "w", encoding="utf-8") as handle:
                    stall_dump.write_stacks(handle)
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
        finally:
            stop.set()
            worker.join(timeout=5)
        self.assertIn("spin_in_busy_loop", text)
        self.assertIn("Thread spinner", text)
