import unittest
from dataclasses import FrozenInstanceError
from enum import Enum


class State(Enum):
    OVERWORLD = 1
    BATTLE = 2
    WHITEOUT = 3


class TestNuzlockeEvents(unittest.TestCase):
    @staticmethod
    def snapshot(*, frame=1, state=State.OVERWORLD, map_id=(1, 2), party=(), battle=None):
        from modules.nuzlocke.snapshots import (
            InventorySnapshot,
            NuzlockeSnapshot,
            PlayerSnapshot,
            ProgressionSnapshot,
            StorageSnapshot,
        )

        return NuzlockeSnapshot(
            frame,
            "test",
            state,
            PlayerSnapshot("May", map_id[0], map_id[1], "MAP", (1, 1), "Down", True),
            tuple(party),
            InventorySnapshot((), (), ()),
            battle,
            StorageSnapshot(0, ()),
            ProgressionSnapshot(()),
        )

    @staticmethod
    def pokemon(index=0, *, hp=10, fainted=False, species="Poochyena", pv=1, status="none"):
        from modules.nuzlocke.snapshots import PartyPokemonSnapshot

        return PartyPokemonSnapshot(
            species=species,
            nickname="Pup",
            level=5,
            current_hp=hp,
            max_hp=10,
            status=status,
            personality_value=pv,
            original_trainer_id=2,
            original_trainer_secret_id=3,
            original_trainer_name="May",
            moves=(),
            held_item=None,
            fainted=fainted,
            egg=False,
            party_index=index,
        )

    @staticmethod
    def battle(outcome="InProgress"):
        from modules.nuzlocke.snapshots import BattleSnapshot

        return BattleSnapshot(("WILD",), False, True, False, (), (), outcome)

    def test_battle_start_and_end_are_one_shot(self):
        from modules.nuzlocke.events import BattleEnded, BattleStarted, GameStateChanged, NuzlockeEventObserver

        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        started = observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=self.battle()))
        self.assertEqual([type(e) for e in started], [BattleStarted, GameStateChanged])
        self.assertEqual(observer.observe(self.snapshot(frame=3, state=State.BATTLE, battle=self.battle())), ())
        ended = observer.observe(self.snapshot(frame=4, battle=None))
        self.assertIsInstance(ended[0], BattleEnded)
        self.assertEqual(sum(isinstance(e, BattleEnded) for e in observer.observe(self.snapshot(frame=5))), 0)

    def test_map_party_faint_and_whiteout_transitions(self):
        from modules.nuzlocke.events import MapChanged, NuzlockeEventObserver, PartyChanged, PokemonFainted, WhiteoutOccurred

        observer = NuzlockeEventObserver()
        party = (self.pokemon(),)
        observer.observe(self.snapshot(party=party))
        changed = observer.observe(self.snapshot(frame=2, map_id=(1, 3), party=(self.pokemon(hp=9),)))
        self.assertEqual(sum(isinstance(e, MapChanged) for e in changed), 1)
        self.assertEqual(sum(isinstance(e, PartyChanged) for e in changed), 0)
        fainted = observer.observe(self.snapshot(frame=3, party=(self.pokemon(hp=0, fainted=True),)))
        self.assertEqual(sum(isinstance(e, PokemonFainted) for e in fainted), 1)
        self.assertEqual(observer.observe(self.snapshot(frame=4, party=(self.pokemon(hp=0, fainted=True),))), ())
        whiteout = observer.observe(self.snapshot(frame=5, state=State.WHITEOUT, party=(self.pokemon(hp=0, fainted=True),)))
        self.assertEqual(sum(isinstance(e, WhiteoutOccurred) for e in whiteout), 1)
        self.assertEqual(sum(isinstance(e, WhiteoutOccurred) for e in observer.observe(self.snapshot(frame=6, state=State.WHITEOUT))), 0)

    def test_party_add_remove_reorder_and_state_change(self):
        from modules.nuzlocke.events import NuzlockeEventObserver, PartyChanged

        first, second = self.pokemon(pv=1), self.pokemon(index=1, species="Zigzagoon", pv=2)
        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot(party=(first,)))
        event = observer.observe(self.snapshot(frame=2, party=(first, second)))[0]
        self.assertIsInstance(event, PartyChanged)
        self.assertEqual(event.entered_party_indices, (1,))
        event = observer.observe(self.snapshot(frame=3, party=(second, first)))[0]
        self.assertTrue(event.reordered)
        observer.observe(self.snapshot(frame=4, party=(second, self.pokemon(pv=1, hp=9))))
        self.assertEqual(observer.observe(self.snapshot(frame=5, party=(second, self.pokemon(pv=1, hp=9)))), ())

    def test_records_are_immutable_and_sequence_is_deterministic(self):
        from modules.nuzlocke.events import MapChanged, NuzlockeEventObserver

        event = MapChanged(1, (1, 1), (1, 2))
        with self.assertRaises(FrozenInstanceError):
            event.frame = 2
        snapshots = [self.snapshot(frame=i, map_id=(1, i)) for i in range(3)]
        observer = NuzlockeEventObserver()
        first = [observer.observe(snapshot) for snapshot in snapshots]
        observer = NuzlockeEventObserver()
        second = [observer.observe(snapshot) for snapshot in snapshots]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
