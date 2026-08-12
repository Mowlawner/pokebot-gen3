import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.events import BattleEnded, BattleStarted, PokemonCaptured, PokemonFainted, PartyChanged
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.persistence import JsonEventStore
from modules.nuzlocke.rules import (
    CAPTURED,
    FAINTED,
    LOST,
    PENDING,
    UNKNOWN,
    load_rules,
    reduce_rules,
)


class TestNuzlockeRules(unittest.TestCase):
    a = (1, 2)
    b = (1, 3)
    wild = PokemonIdentity(10, 20, 30)
    player = PokemonIdentity(11, 20, 30)

    def start(self, frame=1, location=None, *, trainer=False, wild=True, identity=None):
        location = self.a if location is None else location
        identity = self.wild if identity is None else identity
        return BattleStarted(
            frame, ("TRAINER",) if trainer else ("WILD",), trainer, wild, False, (self.player,), (identity,), location
        )

    def end(self, frame=2, location=None, outcome="RanAway", *, trainer=False, wild=True):
        location = self.a if location is None else location
        return BattleEnded(
            frame,
            outcome,
            ("TRAINER",) if trainer else ("WILD",),
            trainer,
            wild,
            False,
            (self.player,),
            (self.wild,),
            location,
        )

    def test_first_encounter_consumes_location_and_capture_associates_identity(self):
        captured = PokemonIdentity(99, 20, 30)
        state = reduce_rules((self.start(), PokemonCaptured(2, captured, self.a), self.end()))
        self.assertEqual(state.encounters[0].status, CAPTURED)
        self.assertEqual(state.encounters[0].pokemon_identity, captured)

    def test_subsequent_wild_encounter_is_ineligible_and_illegal(self):
        state = reduce_rules((self.start(), self.end(), self.start(3)))
        self.assertFalse(state.legal)
        self.assertIn("first encounter", state.violations[0].reason)
        self.assertFalse(state.encounters[-1].eligible)

    def test_trainer_battle_does_not_consume_encounter(self):
        state = reduce_rules((self.start(trainer=True, wild=False), self.end(trainer=True, wild=False)))
        self.assertEqual(state.encounters, ())

    def test_failed_and_incomplete_encounters(self):
        state = reduce_rules((self.start(), self.end(), self.start(3, self.b), self.end(4, self.b, "Unknown")))
        self.assertEqual([e.status for e in state.encounters], [LOST, UNKNOWN])

    def test_faint_death_survives_movement_and_unknown_faint_is_not_death(self):
        faint = PokemonFainted(3, 0, "Poochyena", "Pup", 10, "battle", self.wild)
        moved = PartyChanged(4, (), (0,), False, (), (self.wild,), ())
        state = reduce_rules((faint, moved))
        self.assertEqual(state.dead_pokemon, (self.wild,))
        unknown = reduce_rules((PokemonFainted(1, 0, "Poochyena", "Pup", 10, "unknown"),))
        self.assertEqual(unknown.dead_pokemon, ())
        self.assertEqual(len(unknown.unknown_faints), 1)

    def test_locations_are_independent_duplicate_and_replay_are_deterministic(self):
        events = (self.start(), self.end(), self.start(3, self.b), self.end(4, self.b, "Fainted"))
        self.assertEqual(reduce_rules(events), reduce_rules(events + (events[0], events[1])))
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json", session_id="run")
            store.append_many(events)
            self.assertEqual(load_rules(JsonEventStore(store.path)), reduce_rules(events))

    def test_pending_unknown_does_not_create_violation(self):
        state = reduce_rules((self.start(),))
        self.assertTrue(state.legal)
        self.assertEqual(state.encounters[0].status, PENDING)
        self.assertEqual(state.unresolved_encounters[0].location, self.a)

    def test_violation_reason_is_deterministic(self):
        events = (self.start(), self.end(), self.start(3))
        self.assertEqual(reduce_rules(events).violations, reduce_rules(events).violations)


if __name__ == "__main__":
    unittest.main()
