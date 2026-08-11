import sys
import types
import unittest
from dataclasses import FrozenInstanceError
from enum import Enum
from unittest.mock import Mock


class TestNuzlockeSnapshots(unittest.TestCase):
    def test_snapshot_structures_are_immutable(self):
        from modules.nuzlocke.snapshots import NuzlockeSnapshot, PlayerSnapshot

        snapshot = PlayerSnapshot("May", 1, 2, "Route", (3, 4), "Down", True)
        with self.assertRaises(FrozenInstanceError):
            snapshot.name = "Brendan"
        self.assertTrue(hasattr(NuzlockeSnapshot, "__dataclass_fields__"))

    def test_party_and_player_data_are_normalized(self):
        from modules.nuzlocke.snapshots import MoveSnapshot, PartyPokemonSnapshot, PlayerSnapshot

        pokemon = PartyPokemonSnapshot(
            party_index=0,
            species="Treecko",
            nickname="Sprig",
            level=12,
            current_hp=20,
            max_hp=25,
            status="none",
            personality_value=123,
            original_trainer_id=10,
            original_trainer_secret_id=20,
            original_trainer_name="May",
            moves=(MoveSnapshot("Pound", 35, 35, 0),),
            held_item=None,
            fainted=False,
            egg=False,
        )
        player = PlayerSnapshot("May", 3, 4, "ROUTE_101", (8, 9), "Up", True)
        self.assertEqual((pokemon.species, pokemon.party_index, pokemon.moves[0].pp), ("Treecko", 0, 35))
        self.assertEqual((player.map_group, player.map_number, player.coordinates), (3, 4, (8, 9)))

    def test_battle_absent_and_present_are_explicit(self):
        from modules.nuzlocke.snapshots import BattleSnapshot

        self.assertIsNone(None)  # the top-level field uses None when no battle is active
        battle = BattleSnapshot(("Trainer",), True, False, False, (), (), "InProgress")
        self.assertTrue(battle.is_trainer)
        self.assertFalse(battle.is_double)

    def test_assembly_calls_readers_without_a_rom(self):
        import modules.nuzlocke.snapshots as snapshots

        class FakeState(Enum):
            OVERWORLD = 1
            BATTLE = 2
            BATTLE_STARTING = 3
            BATTLE_ENDING = 4

        class FakePokemon:
            index = 0
            species = types.SimpleNamespace(name="Poochyena")
            nickname = "Pup"
            level = 3
            current_hp = 12
            total_hp = 12
            status_condition = types.SimpleNamespace(value="none")
            personality_value = 1
            original_trainer = types.SimpleNamespace(id=2, secret_id=3, name="May")
            moves = ()
            held_item = None
            is_egg = False

        player = Mock(name="player")
        player.name = "May"
        avatar = Mock(name="avatar")
        avatar.map_group_and_number = (1, 2)
        avatar.local_coordinates = (5, 6)
        avatar.facing_direction = "Down"
        bag = types.SimpleNamespace(items=(), poke_balls=(), key_items=())
        box = types.SimpleNamespace(number=0, slots=())
        storage = types.SimpleNamespace(active_box_index=0, boxes=(box,))
        fake_context = types.SimpleNamespace(
            emulator=types.SimpleNamespace(get_frame_count=lambda: 42),
            rom=types.SimpleNamespace(game_name="Pokémon Emerald"),
        )
        fake_map = types.SimpleNamespace(name="ROUTE_101")
        readers = {
            "modules.context": types.SimpleNamespace(context=fake_context),
            "modules.items": types.SimpleNamespace(get_item_bag=Mock(return_value=bag)),
            "modules.memory": types.SimpleNamespace(
                GameState=FakeState,
                get_game_state=Mock(return_value=FakeState.OVERWORLD),
                get_event_flag=Mock(return_value=False),
            ),
            "modules.pokemon_party": types.SimpleNamespace(get_party=Mock(return_value=(FakePokemon(),))),
            "modules.pokemon_storage": types.SimpleNamespace(get_pokemon_storage=Mock(return_value=storage)),
            "modules.player": types.SimpleNamespace(
                get_player=Mock(return_value=player),
                get_player_avatar=Mock(return_value=avatar),
                get_player_location=Mock(return_value=(fake_map, (5, 6))),
                player_avatar_is_controllable=Mock(return_value=True),
            ),
        }
        old_modules = {name: sys.modules.get(name) for name in readers}
        try:
            sys.modules.update(readers)
            result = snapshots.get_nuzlocke_snapshot()
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result.frame, 42)
        self.assertEqual(result.player.coordinates, (5, 6))
        self.assertEqual(result.party[0].species, "Poochyena")
        self.assertIsNone(result.battle)
        readers["modules.memory"].get_game_state.assert_called_once()
        readers["modules.items"].get_item_bag.assert_called_once()


if __name__ == "__main__":
    unittest.main()
