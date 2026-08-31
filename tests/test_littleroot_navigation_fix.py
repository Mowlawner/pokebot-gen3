import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from modules.map import MapLocation, MapMetadata, get_wild_encounters_for_map, WildEncounterList, WildEncounter
from modules.map_data import MapRSE


class TestLittlerootNavigationFix(unittest.TestCase):
    def setUp(self):
        self.mock_rom = MagicMock()
        self.mock_rom.is_frlg = False
        self.context = SimpleNamespace(rom=self.mock_rom)

    def test_littleroot_has_no_encounters(self):
        # Littleroot Town is (0, 9)
        # Coordinates (6,9), (7,9), (8,9) should be perceived as has_encounters=False

        map_id = (0, 9)
        # Mock encounter list to have no land encounters
        with patch(
            "modules.map.get_wild_encounters_for_map",
            return_value=WildEncounterList(
                land_encounter_rate=0,
                surf_encounter_rate=0,
                rock_smash_encounter_rate=0,
                fishing_encounter_rate=0,
                land_encounters=[],
                surf_encounters=[],
                rock_smash_encounters=[],
                old_rod_encounters=[],
                good_rod_encounters=[],
                super_rod_encounters=[],
            ),
        ), patch("modules.map.context", self.context):
            # Create a mock map location for (6,9) in Littleroot
            mock_loc = MapLocation(b"header", 0, 9, (6, 9))
            # Emerald Tall Grass is tile type 2 / behavior 2.
            mock_loc.__dict__["_metatile_attributes"] = (2, 0, 0)
            mock_loc.__dict__["_tile_behaviour"] = 2

            self.assertFalse(mock_loc.has_encounters)

    def test_route101_has_encounters(self):
        # Route 101 is (0, 1) or similar. Let's assume it has encounters.
        map_id = (0, 1)
        # Mock encounter list to have land encounters
        mock_encounters = WildEncounterList(
            land_encounter_rate=10,
            surf_encounter_rate=0,
            rock_smash_encounter_rate=0,
            fishing_encounter_rate=0,
            land_encounters=[MagicMock(spec=WildEncounter)],
            surf_encounters=[],
            rock_smash_encounters=[],
            old_rod_encounters=[],
            good_rod_encounters=[],
            super_rod_encounters=[],
        )
        with patch("modules.map.get_wild_encounters_for_map", return_value=mock_encounters), patch(
            "modules.map.context", self.context
        ):
            mock_loc = MapLocation(b"header", 0, 1, (6, 9))
            # Route grass is tile type 2 even though its behavior value is 2.
            mock_loc.__dict__["_metatile_attributes"] = (2, 0, 0)
            mock_loc.__dict__["_tile_behaviour"] = 2

            self.assertTrue(mock_loc.has_encounters)

    def test_emerald_tall_grass_behavior_two_is_encounterable(self):
        mock_encounters = WildEncounterList(
            land_encounter_rate=10,
            surf_encounter_rate=0,
            rock_smash_encounter_rate=0,
            fishing_encounter_rate=0,
            land_encounters=[MagicMock(spec=WildEncounter)],
            surf_encounters=[],
            rock_smash_encounters=[],
            old_rod_encounters=[],
            good_rod_encounters=[],
            super_rod_encounters=[],
        )
        with patch("modules.map.get_wild_encounters_for_map", return_value=mock_encounters), patch(
            "modules.map.context", self.context
        ):
            mock_loc = MapLocation(b"header", 0, 18, (14, 11))
            mock_loc.__dict__["_metatile_attributes"] = (2, 0, 0)
            mock_loc.__dict__["_tile_behaviour"] = 2

            self.assertTrue(mock_loc.has_encounters)

    def test_emerald_jump_south_behavior_fifty_nine_is_not_encounterable(self):
        mock_encounters = WildEncounterList(
            land_encounter_rate=10,
            surf_encounter_rate=0,
            rock_smash_encounter_rate=0,
            fishing_encounter_rate=0,
            land_encounters=[MagicMock(spec=WildEncounter)],
            surf_encounters=[],
            rock_smash_encounters=[],
            old_rod_encounters=[],
            good_rod_encounters=[],
            super_rod_encounters=[],
        )
        with patch("modules.map.get_wild_encounters_for_map", return_value=mock_encounters), patch(
            "modules.map.context", self.context
        ):
            mock_loc = MapLocation(b"header", 0, 18, (10, 12))
            # Jump South is a normal-looking tile type with behavior 59.
            mock_loc.__dict__["_metatile_attributes"] = (0, 0, 0)
            mock_loc.__dict__["_tile_behaviour"] = 59

            self.assertFalse(mock_loc.has_encounters)


if __name__ == "__main__":
    unittest.main()
