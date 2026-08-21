import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.fixture_state import (
    FixtureStateUnavailable,
    audit_fixture,
    load_campaign_state_from_fixture,
    load_offline_normalized_snapshot,
)


class FixtureStateAdapterTests(unittest.TestCase):
    fixture = Path(__file__).parents[1] / "profiles" / "test_begin_nuzlocke"

    def test_known_good_save_produces_existing_normalized_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile"
            profile.mkdir()
            shutil.copy(self.fixture / "metadata.yml", profile / "metadata.yml")
            shutil.copy(
                Path(__file__).parents[1] / "profiles" / "Eli_emerald" / "current_save.sav",
                profile / "current_save.sav",
            )
            snapshot = load_offline_normalized_snapshot(profile)
            self.assertTrue(snapshot.player_available)
            self.assertTrue(snapshot.party_available)
            self.assertTrue(snapshot.inventory_available)
            self.assertFalse(snapshot.battle_available)
            self.assertIsNone(snapshot.game_state)

    def test_blank_save_fails_explicitly_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile"
            profile.mkdir()
            shutil.copy(self.fixture / "metadata.yml", profile / "metadata.yml")
            save = profile / "current_save.sav"
            original = b"\xff" * 131072
            save.write_bytes(original)
            with self.assertRaises(FixtureStateUnavailable):
                load_offline_normalized_snapshot(profile)
            self.assertEqual(save.read_bytes(), original)

    def test_adapter_api_is_explicitly_read_only_at_missing_snapshot_boundary(self):
        event_file = self.fixture / "nuzlocke_events.json"
        before = hashlib.sha256(event_file.read_bytes()).digest()
        state = load_campaign_state_from_fixture(self.fixture)
        self.assertTrue(state.campaign_facts.nuzlocke_started.is_known)
        self.assertEqual(state.raw_map.value, (1, 4))
        self.assertEqual(event_file.read_bytes() and hashlib.sha256(event_file.read_bytes()).digest(), before)

    def test_offline_audit_reports_events_without_inventing_live_state(self):
        report = audit_fixture(self.fixture)
        self.assertEqual(report["state_status"], "available")
        self.assertGreater(report["persisted_event_count"], 0)
        self.assertIn("current battle", report["missing"])


if __name__ == "__main__":
    unittest.main()
