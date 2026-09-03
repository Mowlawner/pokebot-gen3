import unittest
from dataclasses import FrozenInstanceError
from enum import Enum


class State(Enum):
    OVERWORLD = 1
    BATTLE = 2
    WHITEOUT = 3


class TestNuzlockeEvents(unittest.TestCase):
    @staticmethod
    def snapshot(*, frame=1, state=State.OVERWORLD, map_id=(1, 2), party=(), battle=None, **availability):
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
            **availability,
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
    def battle(outcome="InProgress", ready=True):
        from modules.nuzlocke.snapshots import BattleSnapshot

        return BattleSnapshot(("WILD",), False, True, False, (), (), outcome, ready)

    def test_battle_start_and_end_are_one_shot(self):
        from modules.nuzlocke.events import (
            BattleEnded,
            BattleStarted,
            GameStateChanged,
            NuzlockeEventObserver,
        )

        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        started = observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=self.battle()))
        self.assertEqual([type(e) for e in started], [BattleStarted, GameStateChanged])
        self.assertEqual(
            observer.observe(self.snapshot(frame=3, state=State.BATTLE, battle=self.battle())),
            (),
        )
        ended = observer.observe(self.snapshot(frame=4, battle=None))
        self.assertIsInstance(ended[0], BattleEnded)
        self.assertEqual(
            sum(isinstance(e, BattleEnded) for e in observer.observe(self.snapshot(frame=5))),
            0,
        )

    def test_authoritative_caught_outcome_emits_capture_before_battle_end(self):
        from modules.nuzlocke.events import BattleEnded, BattleStarted, NuzlockeEventObserver, PokemonCaptured
        from modules.nuzlocke.snapshots import BattlePokemonSnapshot
        from modules.nuzlocke.identity import PokemonIdentity

        identity = PokemonIdentity(7, 2, 3)
        opponent = BattlePokemonSnapshot(0, "Poochyena", 5, 10, "none", False, False, (), identity)
        battle = self.battle(outcome="Caught")
        battle = battle.__class__(battle.battle_type, False, True, False, (), (opponent,), battle.outcome, True)
        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        self.assertTrue(
            any(
                isinstance(e, BattleStarted)
                for e in observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=battle))
            )
        )
        ended = observer.observe(self.snapshot(frame=3, battle=None))
        self.assertEqual(
            [type(e) for e in ended if isinstance(e, (PokemonCaptured, BattleEnded))], [PokemonCaptured, BattleEnded]
        )

    def test_failed_capture_does_not_emit_capture_event(self):
        from modules.nuzlocke.events import NuzlockeEventObserver, PokemonCaptured

        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=self.battle(outcome="InProgress")))
        ended = observer.observe(self.snapshot(frame=3, battle=None))
        self.assertFalse(any(isinstance(e, PokemonCaptured) for e in ended))

    def test_map_party_faint_and_whiteout_transitions(self):
        from modules.nuzlocke.events import (
            MapChanged,
            NuzlockeEventObserver,
            PartyChanged,
            PokemonFainted,
            WhiteoutOccurred,
        )

        observer = NuzlockeEventObserver()
        party = (self.pokemon(),)
        observer.observe(self.snapshot(party=party))
        changed = observer.observe(self.snapshot(frame=2, map_id=(1, 3), party=(self.pokemon(hp=9),)))
        self.assertEqual(sum(isinstance(e, MapChanged) for e in changed), 1)
        self.assertEqual(sum(isinstance(e, PartyChanged) for e in changed), 0)
        fainted = observer.observe(self.snapshot(frame=3, party=(self.pokemon(hp=0, fainted=True),)))
        self.assertEqual(sum(isinstance(e, PokemonFainted) for e in fainted), 1)
        self.assertEqual(
            observer.observe(self.snapshot(frame=4, party=(self.pokemon(hp=0, fainted=True),))),
            (),
        )
        whiteout = observer.observe(
            self.snapshot(frame=5, state=State.WHITEOUT, party=(self.pokemon(hp=0, fainted=True),))
        )
        self.assertEqual(sum(isinstance(e, WhiteoutOccurred) for e in whiteout), 1)
        self.assertEqual(
            sum(
                isinstance(e, WhiteoutOccurred) for e in observer.observe(self.snapshot(frame=6, state=State.WHITEOUT))
            ),
            0,
        )

    def test_party_add_remove_reorder_and_state_change(self):
        from modules.nuzlocke.events import NuzlockeEventObserver, PartyChanged

        first, second = (
            self.pokemon(pv=1),
            self.pokemon(index=1, species="Zigzagoon", pv=2),
        )
        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot(party=(first,)))
        event = observer.observe(self.snapshot(frame=2, party=(first, second)))[0]
        self.assertIsInstance(event, PartyChanged)
        self.assertEqual(event.entered_party_indices, (1,))
        event = observer.observe(self.snapshot(frame=3, party=(second, first)))[0]
        self.assertTrue(event.reordered)
        observer.observe(self.snapshot(frame=4, party=(second, self.pokemon(pv=1, hp=9))))
        self.assertEqual(
            observer.observe(self.snapshot(frame=5, party=(second, self.pokemon(pv=1, hp=9)))),
            (),
        )

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

    def test_temporary_unavailable_party_does_not_emit_party_change(self):
        from modules.nuzlocke.events import NuzlockeEventObserver, PartyChanged

        party = (self.pokemon(),)
        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot(party=party))
        self.assertEqual(
            observer.observe(self.snapshot(frame=2, party=(), party_available=False)),
            (),
        )
        self.assertEqual(
            observer.observe(self.snapshot(frame=3, party=party)),
            (),
        )
        changed = observer.observe(self.snapshot(frame=4, party=()))
        self.assertTrue(any(isinstance(event, PartyChanged) for event in changed))

    def test_unavailable_interval_does_not_end_battle_or_change_map(self):
        from modules.nuzlocke.events import (
            BattleEnded,
            MapChanged,
            NuzlockeEventObserver,
        )

        observer = NuzlockeEventObserver()
        battle = self.battle()
        observer.observe(self.snapshot(battle=battle, state=State.BATTLE))
        unavailable = self.snapshot(
            frame=2,
            state=State.OVERWORLD,
            map_id=(9, 9),
            battle=None,
            game_state_available=False,
            player_available=False,
            battle_available=False,
        )
        self.assertEqual(observer.observe(unavailable), ())
        events = observer.observe(self.snapshot(frame=3, state=State.BATTLE, battle=battle))
        self.assertFalse(any(isinstance(event, BattleEnded) for event in events))
        self.assertFalse(any(isinstance(event, MapChanged) for event in events))

    def test_storage_availability_interval_does_not_create_events(self):
        from modules.nuzlocke.events import NuzlockeEventObserver

        observer = NuzlockeEventObserver()
        available = self.snapshot()
        unavailable = self.snapshot(frame=2, pc_available=False)
        self.assertEqual(observer.observe(available), ())
        self.assertEqual(observer.observe(unavailable), ())
        self.assertEqual(observer.observe(self.snapshot(frame=3)), ())
        self.assertEqual(observer.observe(self.snapshot(frame=4, pc_available=False)), ())
        self.assertEqual(observer.observe(self.snapshot(frame=5)), ())

    def test_battle_starting_snapshot_does_not_start_battle_twice(self):
        from modules.nuzlocke.events import BattleStarted, NuzlockeEventObserver

        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        self.assertFalse(
            any(
                isinstance(event, BattleStarted)
                for event in observer.observe(
                    self.snapshot(frame=2, state=State.BATTLE, battle=self.battle(ready=False))
                )
            )
        )
        started = observer.observe(self.snapshot(frame=3, state=State.BATTLE, battle=self.battle()))
        self.assertEqual(sum(isinstance(event, BattleStarted) for event in started), 1)
        self.assertEqual(
            observer.observe(self.snapshot(frame=4, state=State.BATTLE, battle=self.battle())),
            (),
        )

    def test_initial_ready_battle_is_a_start_boundary_after_restore(self):
        from modules.nuzlocke.events import BattleStarted, NuzlockeEventObserver

        observer = NuzlockeEventObserver()
        events = observer.observe(self.snapshot(state=State.BATTLE, battle=self.battle()))
        self.assertEqual(sum(isinstance(event, BattleStarted) for event in events), 1)
        self.assertEqual(
            observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=self.battle())),
            (),
        )

    def test_zero_type_starting_frame_waits_for_real_battle_boundary(self):
        from modules.nuzlocke.events import BattleStarted, NuzlockeEventObserver

        observer = NuzlockeEventObserver()
        starting = self.battle().__class__((), False, True, False, (), (), "InProgress", True)
        self.assertEqual(
            observer.observe(self.snapshot(state=State.BATTLE, battle=starting)),
            (),
        )
        events = observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=self.battle()))
        self.assertEqual(sum(isinstance(event, BattleStarted) for event in events), 1)

    def test_stale_opponent_buffer_is_not_accepted_as_new_battle(self):
        from modules.nuzlocke.events import BattleEnded, BattleStarted, NuzlockeEventObserver
        from modules.nuzlocke.identity import PokemonIdentity
        from modules.nuzlocke.snapshots import BattlePokemonSnapshot, BattleSnapshot

        def battle(species: str, personality_value: int) -> BattleSnapshot:
            own = BattlePokemonSnapshot(
                0,
                "Treecko",
                10,
                10,
                "none",
                False,
                False,
                (),
                PokemonIdentity(100, 2, 3),
            )
            opponent = BattlePokemonSnapshot(
                1,
                species,
                5,
                10,
                "none",
                False,
                False,
                (),
                PokemonIdentity(personality_value, 4, 5),
            )
            return BattleSnapshot(("WILD",), False, True, False, (own,), (opponent,), "InProgress")

        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        first = observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=battle("Poochyena", 10)))
        self.assertTrue(any(isinstance(event, BattleStarted) for event in first))
        self.assertTrue(any(isinstance(event, BattleEnded) for event in observer.observe(self.snapshot(frame=3))))

        # The first ready frames of the next battle still contain the prior
        # Poochyena in Emerald's shared opponent buffer.
        for frame in (4, 5):
            self.assertFalse(
                any(
                    isinstance(event, BattleStarted)
                    for event in observer.observe(
                        self.snapshot(frame=frame, state=State.BATTLE, battle=battle("Poochyena", 10))
                    )
                )
            )

        # A new species/identity is first treated as a candidate, then
        # accepted only after it remains stable for another observation.
        self.assertFalse(
            any(
                isinstance(event, BattleStarted)
                for event in observer.observe(
                    self.snapshot(frame=6, state=State.BATTLE, battle=battle("Zigzagoon", 20))
                )
            )
        )
        started = observer.observe(self.snapshot(frame=7, state=State.BATTLE, battle=battle("Zigzagoon", 20)))
        battle_started = next(event for event in started if isinstance(event, BattleStarted))
        self.assertEqual(battle_started.opponent_species, ("Zigzagoon",))

    def test_same_species_new_identity_is_a_new_battle(self):
        from modules.nuzlocke.events import BattleStarted, NuzlockeEventObserver
        from modules.nuzlocke.identity import PokemonIdentity
        from modules.nuzlocke.snapshots import BattlePokemonSnapshot, BattleSnapshot

        def battle(personality_value: int) -> BattleSnapshot:
            opponent = BattlePokemonSnapshot(
                1,
                "Poochyena",
                5,
                10,
                "none",
                False,
                False,
                (),
                PokemonIdentity(personality_value, 4, 5),
            )
            return BattleSnapshot(
                ("WILD",),
                False,
                True,
                False,
                (),
                (opponent,),
                "InProgress",
            )

        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        self.assertTrue(
            any(
                isinstance(event, BattleStarted)
                for event in observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=battle(10)))
            )
        )
        observer.observe(self.snapshot(frame=3))
        observer.observe(self.snapshot(frame=4, state=State.BATTLE, battle=battle(10)))
        self.assertFalse(
            any(
                isinstance(event, BattleStarted)
                for event in observer.observe(self.snapshot(frame=5, state=State.BATTLE, battle=battle(20)))
            )
        )
        started = observer.observe(self.snapshot(frame=6, state=State.BATTLE, battle=battle(20)))
        self.assertEqual(sum(isinstance(event, BattleStarted) for event in started), 1)

    def test_missing_opponent_identity_preserves_legacy_boundary_behavior(self):
        from modules.nuzlocke.events import BattleEnded, BattleStarted, NuzlockeEventObserver
        from modules.nuzlocke.snapshots import BattlePokemonSnapshot, BattleSnapshot

        battle = BattleSnapshot(
            ("WILD",),
            False,
            True,
            False,
            (),
            (BattlePokemonSnapshot(1, "Poochyena", 5, 10, "none", False, False, (), None),),
            "InProgress",
        )
        observer = NuzlockeEventObserver()
        observer.observe(self.snapshot())
        self.assertTrue(
            any(isinstance(event, BattleStarted) for event in observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=battle)))
        )
        self.assertTrue(any(isinstance(event, BattleEnded) for event in observer.observe(self.snapshot(frame=3))))

        # Snapshot producers that cannot provide stable opponent identities
        # cannot safely detect the stale shared buffer, so retain the previous
        # one-shot boundary behavior for compatibility.
        next_battle = observer.observe(self.snapshot(frame=4, state=State.BATTLE, battle=battle))
        self.assertEqual(sum(isinstance(event, BattleStarted) for event in next_battle), 1)

    def test_partial_battle_teardown_does_not_end_battle_early(self):
        from modules.nuzlocke.events import (
            BattleEnded,
            BattleStarted,
            NuzlockeEventObserver,
        )

        observer = NuzlockeEventObserver()
        battle = self.battle()
        observer.observe(self.snapshot(state=State.BATTLE, battle=battle))
        self.assertFalse(
            any(
                isinstance(event, BattleEnded)
                for event in observer.observe(
                    self.snapshot(frame=2, state=State.BATTLE, battle=self.battle(ready=False))
                )
            )
        )
        self.assertFalse(
            any(
                isinstance(event, BattleStarted)
                for event in observer.observe(self.snapshot(frame=3, state=State.BATTLE, battle=battle))
            )
        )
        ended = observer.observe(self.snapshot(frame=4, state=State.OVERWORLD))
        self.assertEqual(sum(isinstance(event, BattleEnded) for event in ended), 1)
        self.assertEqual(observer.observe(self.snapshot(frame=5, state=State.OVERWORLD)), ())

    def test_missing_battle_snapshot_during_switch_does_not_end_or_restart_battle(self):
        from modules.nuzlocke.events import BattleEnded, BattleStarted, NuzlockeEventObserver

        observer = NuzlockeEventObserver()
        battle = self.battle()
        observer.observe(self.snapshot())
        started = observer.observe(self.snapshot(frame=2, state=State.BATTLE, battle=battle))
        self.assertEqual(sum(isinstance(event, BattleStarted) for event in started), 1)

        # The game remains in the battle lifecycle while Emerald briefly
        # cannot materialize the battle object during a party switch.
        transient = observer.observe(self.snapshot(frame=3, state=State.BATTLE, battle=None))
        self.assertFalse(any(isinstance(event, (BattleStarted, BattleEnded)) for event in transient))

        resumed = observer.observe(self.snapshot(frame=4, state=State.BATTLE, battle=battle))
        self.assertFalse(any(isinstance(event, (BattleStarted, BattleEnded)) for event in resumed))

        ended = observer.observe(self.snapshot(frame=5, state=State.OVERWORLD, battle=None))
        self.assertEqual(sum(isinstance(event, BattleEnded) for event in ended), 1)

    def test_unknown_state_during_battle_read_gap_does_not_end_battle(self):
        from modules.nuzlocke.events import BattleEnded, BattleStarted, NuzlockeEventObserver

        class UnknownState(Enum):
            UNKNOWN = 1

        observer = NuzlockeEventObserver()
        battle = self.battle()
        observer.observe(self.snapshot(state=State.BATTLE, battle=battle))
        transient = observer.observe(self.snapshot(frame=2, state=UnknownState.UNKNOWN, battle=None))
        self.assertFalse(any(isinstance(event, (BattleStarted, BattleEnded)) for event in transient))
        self.assertTrue(
            any(
                isinstance(event, BattleEnded)
                for event in observer.observe(self.snapshot(frame=3, state=State.OVERWORLD, battle=None))
            )
        )


if __name__ == "__main__":
    unittest.main()
