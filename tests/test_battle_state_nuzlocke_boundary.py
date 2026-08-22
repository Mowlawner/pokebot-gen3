import types
import unittest
from unittest.mock import patch

from modules.battle_state import BattleState
from modules.context import context
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.rules import LocationEncounter, NuzlockeCampaignState, PENDING


class TestBattleStateNuzlockeBoundary(unittest.TestCase):
    def test_route_101_pending_wild_encounter_is_capture_target(self):
        runtime = NuzlockeRuntime()
        runtime.rules_projection._state = NuzlockeCampaignState(
            encounters=(LocationEncounter((0, 16), PENDING, eligible=True),)
        )

        battle_state = BattleState.__new__(BattleState)
        battle_state._battle_type = (0).to_bytes(4, "little")

        avatar = types.SimpleNamespace(map_group_and_number=(0, 16))
        old_runtime = context.nuzlocke_runtime
        context.nuzlocke_runtime = runtime
        try:
            with patch("modules.battle_state.get_player_avatar", return_value=avatar):
                self.assertTrue(battle_state.nuzlocke_capture_target)
        finally:
            context.nuzlocke_runtime = old_runtime


if __name__ == "__main__":
    unittest.main()
