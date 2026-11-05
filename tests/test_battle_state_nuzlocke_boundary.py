import types
import unittest
from unittest.mock import patch

from modules.battle_state import BattlePokemon, BattleState
from modules.context import context
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.rules import LocationEncounter, NuzlockeCampaignState, PENDING


class TestBattleStateNuzlockeBoundary(unittest.TestCase):
    def test_battle_pokemon_exposes_live_bide_timer_and_stored_damage(self):
        data = bytearray(0x58)
        data[0x50:0x54] = (2 << 8).to_bytes(4, "little")
        battler = BattlePokemon(bytes(data), b"\x00" * 4, b"\x00" * 0x1C, 0, (7).to_bytes(4, "little"))

        self.assertEqual(battler.bide_turns_remaining, 2)
        self.assertEqual(battler.bide_damage, 7)
        self.assertTrue(battler.bide_damage_known)

    def test_unreadable_bide_damage_is_unknown_not_zero(self):
        data = bytearray(0x58)
        data[0x50:0x54] = (1 << 8).to_bytes(4, "little")
        battler = BattlePokemon(bytes(data), b"\x00" * 4, b"\x00" * 0x1C, 0, b"")

        self.assertEqual(battler.bide_turns_remaining, 1)
        self.assertEqual(battler.bide_damage, 0)
        self.assertFalse(battler.bide_damage_known)

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
            with patch("modules.battle_state.get_player_avatar", return_value=avatar), patch.object(
                runtime, "capture_target_for", return_value=True
            ) as capture_target:
                self.assertTrue(battle_state.nuzlocke_capture_target)
            capture_target.assert_called_once_with((0, 16), is_wild=True, is_trainer=False)
        finally:
            context.nuzlocke_runtime = old_runtime


if __name__ == "__main__":
    unittest.main()
