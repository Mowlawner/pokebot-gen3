import tempfile
import unittest
from pathlib import Path

from modules.profiles import ProfileInUseError, ProfileLock


class TestProfileLock(unittest.TestCase):
    def test_lock_is_exclusive_and_reusable_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile"
            first = ProfileLock(profile).acquire()
            second = ProfileLock(profile)

            with self.assertRaises(ProfileInUseError):
                second.acquire()

            first.release()
            second.acquire()
            second.release()

    def test_context_manager_releases_lock_on_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile"
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with ProfileLock(profile):
                    raise RuntimeError("stop")

            with ProfileLock(profile):
                pass


if __name__ == "__main__":
    unittest.main()
