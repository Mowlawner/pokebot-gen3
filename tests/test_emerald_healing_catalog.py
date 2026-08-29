import unittest

from modules.map_data import MapRSE
from modules.nuzlocke.emerald_healing_catalog import (
    HealingSourceRSE,
    emerald_healing_source_for_destination,
    emerald_healing_sources_for_map,
)


class EmeraldHealingCatalogTests(unittest.TestCase):
    def test_catalog_contains_centers_and_non_center_sources(self):
        self.assertIn(HealingSourceRSE.OLDALE_POKEMON_CENTER, tuple(HealingSourceRSE))
        self.assertEqual(
            HealingSourceRSE.PLAYERS_MOTHER.script_symbol,
            "PlayersHouse_1F_EventScript_Mom",
        )
        self.assertEqual(
            HealingSourceRSE.ROUTE111_REST_STOP.script_symbol,
            "Route111_OldLadysRestStop_EventScript_OldLady",
        )

    def test_special_sources_are_indexed_by_their_outdoor_map(self):
        littleroot = emerald_healing_sources_for_map(MapRSE.LITTLEROOT_TOWN)
        route111 = emerald_healing_sources_for_map(MapRSE.ROUTE111)
        self.assertIn(HealingSourceRSE.PLAYERS_MOTHER, littleroot)
        self.assertIn(HealingSourceRSE.ROUTE111_REST_STOP, route111)

    def test_destination_resolves_to_catalog_source(self):
        source = emerald_healing_source_for_destination(
            (MapRSE.ROUTE111, (26, 18))
        )
        self.assertIs(source, HealingSourceRSE.ROUTE111_REST_STOP)


if __name__ == "__main__":
    unittest.main()
