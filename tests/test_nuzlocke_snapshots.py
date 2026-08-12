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

        class FakeMove:
            name = "Pound"

        class FakeLearnedMove:
            move = FakeMove()
            pp = 35
            total_pp = 35
            pp_ups = 0

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
            moves = (FakeLearnedMove(),)
            held_item = None
            is_egg = False
            is_empty = False
            is_valid = True

        player = Mock(name="player")
        player.name = "May"
        avatar = Mock(name="avatar")
        avatar.map_group_and_number = (1, 2)
        avatar.local_coordinates = (5, 6)
        avatar.facing_direction = "Down"
        bag = types.SimpleNamespace(items=(), poke_balls=(), key_items=())
        storage_pokemon = FakePokemon()
        storage_slot = types.SimpleNamespace(slot_index=4, pokemon=storage_pokemon)
        box = types.SimpleNamespace(number=0, slots=(storage_slot,))
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
            readers["modules.pokemon_storage"].get_pokemon_storage.return_value = types.SimpleNamespace(
                active_box_index=0,
                boxes=(types.SimpleNamespace(number=0, slots=()),),
            )
            empty_storage_result = snapshots.get_nuzlocke_snapshot()
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result.frame, 42)
        self.assertEqual(result.player.coordinates, (5, 6))
        self.assertEqual(result.party[0].species, "Poochyena")
        self.assertEqual(result.pc.pokemon[0].slot, 4)
        self.assertEqual(result.pc.pokemon[0].pokemon.moves[0].name, "Pound")
        self.assertTrue(result.pc_available)
        self.assertEqual(empty_storage_result.pc.pokemon, ())
        self.assertTrue(empty_storage_result.pc_available)
        self.assertIsNone(result.battle)
        readers["modules.memory"].get_game_state.assert_called()
        readers["modules.items"].get_item_bag.assert_called()

    def test_unavailable_readers_produce_an_unavailable_snapshot(self):
        import modules.nuzlocke.snapshots as snapshots

        class FakeState(Enum):
            TITLE_SCREEN = 1

        fake_context = types.SimpleNamespace(
            emulator=types.SimpleNamespace(get_frame_count=lambda: 7),
            rom=types.SimpleNamespace(game_name=None),
        )
        readers = {
            "modules.context": types.SimpleNamespace(context=fake_context),
            "modules.items": types.SimpleNamespace(get_item_bag=Mock(return_value=None)),
            "modules.memory": types.SimpleNamespace(
                get_game_state=Mock(return_value=None),
                get_event_flag=Mock(return_value=False),
            ),
            "modules.pokemon_party": types.SimpleNamespace(get_party=Mock(return_value=None)),
            "modules.pokemon_storage": types.SimpleNamespace(get_pokemon_storage=Mock(return_value=None)),
            "modules.player": types.SimpleNamespace(
                get_player=Mock(return_value=None),
                get_player_avatar=Mock(return_value=None),
                get_player_location=Mock(),
                player_avatar_is_controllable=Mock(return_value=False),
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

        self.assertEqual(result.party, ())
        self.assertEqual(result.inventory.items, ())
        self.assertEqual(result.pc.pokemon, ())
        self.assertIsNone(result.game_state)
        self.assertFalse(result.game_state_available)
        self.assertFalse(result.player_available)
        self.assertFalse(result.party_available)
        self.assertFalse(result.inventory_available)
        self.assertFalse(result.pc_available)

    def test_malformed_storage_pokemon_is_not_normalized(self):
        import modules.nuzlocke.snapshots as snapshots

        class FakeState(Enum):
            OVERWORLD = 1

        malformed_pokemon = types.SimpleNamespace(is_empty=False, is_valid=False)
        storage = types.SimpleNamespace(
            active_box_index=13,
            boxes=(
                types.SimpleNamespace(
                    number=13,
                    slots=(types.SimpleNamespace(slot_index=29, pokemon=malformed_pokemon),),
                ),
            ),
        )
        fake_context = types.SimpleNamespace(
            emulator=types.SimpleNamespace(get_frame_count=lambda: 7),
            rom=types.SimpleNamespace(game_name="Pokémon Emerald"),
        )
        readers = {
            "modules.context": types.SimpleNamespace(context=fake_context),
            "modules.items": types.SimpleNamespace(get_item_bag=Mock(return_value=None)),
            "modules.memory": types.SimpleNamespace(
                get_game_state=Mock(return_value=FakeState.OVERWORLD),
                get_event_flag=Mock(return_value=False),
            ),
            "modules.pokemon_party": types.SimpleNamespace(get_party=Mock(return_value=None)),
            "modules.pokemon_storage": types.SimpleNamespace(get_pokemon_storage=Mock(return_value=storage)),
            "modules.player": types.SimpleNamespace(
                get_player=Mock(return_value=None),
                get_player_avatar=Mock(return_value=None),
                get_player_location=Mock(),
                player_avatar_is_controllable=Mock(return_value=False),
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

        self.assertTrue(result.pc_available)
        self.assertEqual(result.pc.active_box, 13)
        self.assertEqual(result.pc.pokemon, ())

    def test_battle_starting_with_battlers_normalizes_battle_pokemon(self):
        import modules.nuzlocke.snapshots as snapshots

        class FakeState(Enum):
            BATTLE = 0
            BATTLE_STARTING = 1
            BATTLE_ENDING = 2

        class FakeBattleType(Enum):
            FirstBattle = 1

        # This is the shape exposed by BattlePokemon.  In particular, it does
        # not have the party model's is_fainted property.
        class FakeBattlePokemon:
            def __init__(self, party_index):
                self.party_index = party_index
                self.species = types.SimpleNamespace(name="Treecko")
                self.current_hp = 15
                self.total_hp = 15
                self.status_permanent = types.SimpleNamespace(value="none")
                self.is_egg = False
                self.moves = ()

        own = FakeBattlePokemon(0)
        opponent = FakeBattlePokemon(0)

        class BattleSide:
            def __init__(self, battler):
                self.active_battlers = (battler,)

        class BattleStartingState:
            type = {FakeBattleType.FirstBattle}
            is_trainer_battle = False
            is_double_battle = False

            def __init__(self):
                self.battling_pokemon = (own, opponent)
                self.own_side = BattleSide(own)
                self.opponent = BattleSide(opponent)

        fake_context = types.SimpleNamespace(
            emulator=types.SimpleNamespace(get_frame_count=lambda: 10),
            rom=types.SimpleNamespace(game_name="Pokémon Emerald"),
        )
        readers = {
            "modules.context": types.SimpleNamespace(context=fake_context),
            "modules.items": types.SimpleNamespace(get_item_bag=Mock(return_value=None)),
            "modules.memory": types.SimpleNamespace(
                GameState=FakeState,
                get_game_state=Mock(return_value=FakeState.BATTLE_STARTING),
                get_event_flag=Mock(return_value=False),
            ),
            "modules.pokemon_party": types.SimpleNamespace(get_party=Mock(return_value=None)),
            "modules.pokemon_storage": types.SimpleNamespace(get_pokemon_storage=Mock(return_value=None)),
            "modules.player": types.SimpleNamespace(
                get_player=Mock(return_value=None),
                get_player_avatar=Mock(return_value=None),
                get_player_location=Mock(),
                player_avatar_is_controllable=Mock(return_value=False),
            ),
            "modules.battle_state": types.SimpleNamespace(
                BattleState=BattleStartingState,
                BattleType=FakeBattleType,
                get_battle_state=Mock(return_value=BattleStartingState()),
                get_last_battle_outcome=Mock(return_value=types.SimpleNamespace(name="InProgress")),
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

        self.assertTrue(result.battle.ready)
        self.assertEqual(result.battle.own_active[0].current_hp, 15)
        self.assertEqual(result.battle.own_active[0].status, "none")
        self.assertIsNone(result.battle.own_active[0].fainted)

    def test_battle_starting_with_no_battlers_is_not_ready(self):
        import modules.nuzlocke.snapshots as snapshots

        class FakeState(Enum):
            BATTLE = 0
            BATTLE_STARTING = 1
            BATTLE_ENDING = 2

        class FakeBattleType(Enum):
            FirstBattle = 1

        class IncompleteBattleState:
            type = {FakeBattleType.FirstBattle}
            is_trainer_battle = False
            is_double_battle = False
            battling_pokemon = ()

            @property
            def own_side(self):
                raise AssertionError("active battlers must not be read before battle initialization")

            @property
            def opponent(self):
                raise AssertionError("active battlers must not be read before battle initialization")

        fake_context = types.SimpleNamespace(
            emulator=types.SimpleNamespace(get_frame_count=lambda: 10),
            rom=types.SimpleNamespace(game_name="Pokémon Emerald"),
        )
        readers = {
            "modules.context": types.SimpleNamespace(context=fake_context),
            "modules.items": types.SimpleNamespace(get_item_bag=Mock(return_value=None)),
            "modules.memory": types.SimpleNamespace(
                GameState=FakeState,
                get_game_state=Mock(return_value=FakeState.BATTLE_STARTING),
                get_event_flag=Mock(return_value=False),
            ),
            "modules.pokemon_party": types.SimpleNamespace(get_party=Mock(return_value=None)),
            "modules.pokemon_storage": types.SimpleNamespace(get_pokemon_storage=Mock(return_value=None)),
            "modules.player": types.SimpleNamespace(
                get_player=Mock(return_value=None),
                get_player_avatar=Mock(return_value=None),
                get_player_location=Mock(),
                player_avatar_is_controllable=Mock(return_value=False),
            ),
            "modules.battle_state": types.SimpleNamespace(
                BattleState=IncompleteBattleState,
                BattleType=FakeBattleType,
                get_battle_state=Mock(return_value=IncompleteBattleState()),
                get_last_battle_outcome=Mock(return_value=types.SimpleNamespace(name="InProgress")),
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

        self.assertIsNotNone(result.battle)
        self.assertFalse(result.battle.ready)
        self.assertEqual(result.battle.own_active, ())
        self.assertEqual(result.battle.opponent_active, ())
        self.assertTrue(result.battle_available)


if __name__ == "__main__":
    unittest.main()
