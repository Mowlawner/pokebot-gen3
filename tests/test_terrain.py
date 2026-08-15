import unittest

from modules.terrain import (
    ENCOUNTER_TERRAIN_COST,
    NORMAL_TERRAIN_COST,
    terrain_cost,
)


class TestTerrainCosts(unittest.TestCase):
    def test_normal_tiles_have_baseline_cost(self):
        self.assertEqual(terrain_cost("Normal"), NORMAL_TERRAIN_COST)

    def test_tall_grass_tiles_have_encounter_penalty(self):
        self.assertEqual(terrain_cost("Long Grass"), ENCOUNTER_TERRAIN_COST)
        self.assertEqual(terrain_cost("Tall Grass"), ENCOUNTER_TERRAIN_COST)


if __name__ == "__main__":
    unittest.main()
