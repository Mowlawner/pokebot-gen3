import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.profiles import ProfileLock


class TestHeadlessLifecycle(unittest.TestCase):
    def test_keyboard_interrupt_saves_state_and_releases_profile(self):
        from modules.gui.headless import PokebotHeadless

        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile"
            profile = SimpleNamespace(path=profile_path, rom=object())
            emulator = Mock()
            fake_context = SimpleNamespace(
                emulator=None,
                profile=None,
                config=SimpleNamespace(load=Mock()),
                audio=None,
                video=None,
                emulation_speed=None,
                debug=None,
                debug_trace=None,
                debug_profile=None,
                debug_stutter_trace=None,
                stutter_trace_threshold_ms=None,
                bot_mode=None,
            )
            settings = SimpleNamespace(
                profile=profile,
                profile_source="test",
                no_audio=True,
                no_video=True,
                emulation_speed=1,
                debug=False,
                debug_trace=False,
                debug_profile=False,
                stutter_trace=False,
                stutter_threshold_ms=50.0,
                bot_mode="Manual",
                no_save_state=False,
            )

            def stop():
                raise KeyboardInterrupt

            with (
                patch("modules.gui.headless.context", fake_context),
                patch("modules.gui.headless.set_rom"),
                patch("modules.gui.headless.LibmgbaEmulator", return_value=emulator),
            ):
                PokebotHeadless(stop, Mock()).run(settings)

            emulator.shutdown.assert_called_once()
            with ProfileLock(profile_path):
                pass


if __name__ == "__main__":
    unittest.main()
