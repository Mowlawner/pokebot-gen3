import unittest

from modules.goals import ActivateTrigger
from modules.map_path import Direction
from modules.navigation import (
    NavigationWorld,
    NavigableTile,
    navigation_diagnostics,
)


class TestNavigationDiagnostics(unittest.TestCase):
    def test_missing_trigger_reports_snapshot_and_binding_failure(self):
        current = ((1, 4), (6, 5))
        world = NavigationWorld(
            tiles={current: NavigableTile(current)},
            warps=(),
            triggers=(),
        )

        diagnostics = navigation_diagnostics(world, current, ActivateTrigger("introductory_rival"))

        self.assertIn("current_map=(1, 4)", diagnostics)
        self.assertIn("target_map=()", diagnostics)
        self.assertIn("target_positions=()", diagnostics)
        self.assertIn("candidate_interaction_positions=0", diagnostics)
        self.assertIn("status=UNREACHABLE reason=TRIGGER_NOT_PRESENT_IN_PERCEPTION", diagnostics)


if __name__ == "__main__":
    unittest.main()
