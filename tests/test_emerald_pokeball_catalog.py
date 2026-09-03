import unittest

from modules.map_data import MapRSE
from modules.nuzlocke.emerald_pokeball_catalog import (
    PokeballSourceRSE,
    emerald_pokeball_sources,
    emerald_pokeball_sources_for_map,
)


class EmeraldPokeballCatalogTests(unittest.TestCase):
    def test_catalog_contains_only_marts_that_sell_ordinary_pokeballs(self):
        sources = tuple(emerald_pokeball_sources())

        self.assertEqual(len(sources), 12)
        self.assertEqual(
            {source.source_id for source in sources},
            {
                "pokemart:oldale",
                "pokemart:petalburg",
                "pokemart:rustboro",
                "pokemart:slateport",
                "pokemart:mauville",
                "department_store:lilycove:2f",
                "pokemart:lavaridge",
                "pokemart:fallarbor",
                "pokemart:verdanturf",
                "pokemart:fortree",
                "pokemart:mossdeep",
                "pokemart:sootopolis",
            },
        )
        self.assertIn(PokeballSourceRSE.OLDALE_POKEMART, sources)
        self.assertIn(PokeballSourceRSE.LILYCOVE_DEPARTMENT_STORE_2F, sources)
        self.assertNotIn(
            MapRSE.LILYCOVE_CITY_UNUSED_MART,
            tuple(source.interior_map for source in sources),
        )

    def test_catalog_indexes_exterior_and_interior_maps(self):
        sources = emerald_pokeball_sources_for_map(MapRSE.RUSTBORO_CITY.value)

        self.assertEqual(sources, (PokeballSourceRSE.RUSTBORO_POKEMART,))
        self.assertEqual(
            emerald_pokeball_sources_for_map(MapRSE.LILYCOVE_CITY_DEPARTMENT_STORE_2F.value),
            (PokeballSourceRSE.LILYCOVE_DEPARTMENT_STORE_2F,),
        )

    def test_department_store_has_both_clerk_symbols(self):
        self.assertEqual(
            len(PokeballSourceRSE.LILYCOVE_DEPARTMENT_STORE_2F.clerk_symbols),
            2,
        )

    def test_source_entrances_use_direct_or_first_world_route_edge(self):
        class Edge:
            source_map = MapRSE.LILYCOVE_CITY.value
            destination_map = MapRSE.LILYCOVE_CITY_DEPARTMENT_STORE_1F.value
            source_coordinates = ((12, 8),)

        class Route:
            edges = (Edge(),)

        class Graph:
            edges = ()

            @staticmethod
            def route(_source, _target):
                return Route()

        from modules.nuzlocke.emerald_pokeball_catalog import emerald_pokeball_source_entrances

        self.assertEqual(
            emerald_pokeball_source_entrances(PokeballSourceRSE.LILYCOVE_DEPARTMENT_STORE_2F, Graph()),
            ((MapRSE.LILYCOVE_CITY.value, (12, 8)),),
        )


if __name__ == "__main__":
    unittest.main()
