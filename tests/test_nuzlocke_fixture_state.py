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

    def usable_fixture(self):
        """Stage a contiguous prefix without touching the ignored live profile."""
        directory = tempfile.TemporaryDirectory()
        profile = Path(directory.name) / "profile"
        profile.mkdir()
        shutil.copy(self.fixture / "metadata.yml", profile / "metadata.yml")
        shutil.copy(self.fixture / "current_save.sav", profile / "current_save.sav")
        records = (self.fixture / "nuzlocke_events.json").read_text().splitlines()
        (profile / "nuzlocke_events.json").write_text("\n".join(records[:15]) + "\n")
        return directory, profile

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
            self.assertEqual(
                {flag.name for flag in snapshot.campaign_observation.flags},
                {
                    "SET_WALL_CLOCK",
                    "RESCUED_BIRCH",
                    "DEFEATED_RIVAL_ROUTE103",
                    "HIDE_ROUTE_103_RIVAL",
                    "SYS_POKEMON_GET",
                    "SYS_POKEDEX_GET",
                    "RECEIVED_POKEDEX_FROM_BIRCH",
                    "VISITED_PETALBURG_CITY",
                    "DEVON_GOODS_STOLEN",
                    "RECOVERED_DEVON_GOODS",
                    "RETURNED_DEVON_GOODS",
                    "DELIVERED_DEVON_GOODS",
                    "INTERACTED_WITH_DEVON_EMPLOYEE_GOODS_STOLEN",
                    "VISITED_RUSTBORO_CITY",
                    "DEFEATED_RUSTBORO_GYM",
                },
            )
            self.assertEqual(
                {variable.name for variable in snapshot.campaign_observation.variables},
                {
                    "LITTLEROOT_INTRO_STATE",
                    "LITTLEROOT_RIVAL_STATE",
                    "BIRCH_LAB_STATE",
                    "PETALBURG_CITY_STATE",
                    "PETALBURG_GYM_STATE",
                    "PETALBURG_WOODS_STATE",
                    "RUSTBORO_CITY_STATE",
                    "RUSTURF_TUNNEL_STATE",
                    "DEVON_CORP_3F_STATE",
                },
            )

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
        directory, profile = self.usable_fixture()
        try:
            state = load_campaign_state_from_fixture(profile)
            self.assertTrue(state.campaign_facts.nuzlocke_started.is_known)
            self.assertEqual(state.raw_map.value, (1, 4))
        finally:
            directory.cleanup()
        self.assertEqual(event_file.read_bytes() and hashlib.sha256(event_file.read_bytes()).digest(), before)

    def test_offline_audit_reports_events_without_inventing_live_state(self):
        directory, profile = self.usable_fixture()
        try:
            report = audit_fixture(profile)
            self.assertEqual(report["state_status"], "available")
            self.assertGreater(report["persisted_event_count"], 0)
            self.assertIn("current battle", report["missing"])
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
