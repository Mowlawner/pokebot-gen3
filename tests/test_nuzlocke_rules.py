import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.events import BattleEnded, BattleStarted, PokemonCaptured, PokemonFainted, PartyChanged
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.persistence import JsonEventStore
from modules.nuzlocke.rule_config import CampaignRuleId, CampaignRulesConfig
from modules.nuzlocke.rules import (
    CAPTURED,
    FAINTED,
    LOST,
    PENDING,
    UNKNOWN,
    NuzlockeRulesProjection,
    species_clause_conflict_reason,
    load_rules,
    reduce_rules,
)


class TestNuzlockeRules(unittest.TestCase):
    a = (1, 2)
    b = (1, 3)
    wild = PokemonIdentity(10, 20, 30)
    player = PokemonIdentity(11, 20, 30)

    def start(self, frame=1, location=None, *, trainer=False, wild=True, identity=None, species=()):
        location = self.a if location is None else location
        identity = self.wild if identity is None else identity
        return BattleStarted(
            frame,
            ("TRAINER",) if trainer else ("WILD",),
            trainer,
            wild,
            False,
            (self.player,),
            (identity,),
            location,
            opponent_species=species,
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

    def test_subsequent_wild_encounter_is_ineligible_but_legal_to_fight(self):
        captured = PokemonIdentity(99, 20, 30)
        state = reduce_rules(
            (
                self.start(),
                PokemonCaptured(2, captured, self.a),
                self.end(),
                self.start(3),
                PokemonCaptured(4, self.wild, self.a),
            )
        )
        self.assertTrue(state.legal)
        self.assertEqual(state.violations, ())
        self.assertFalse(state.encounters[-1].eligible)
        self.assertEqual(state.encounters[0].pokemon_identity, captured)

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

    def test_species_clause_skips_duplicate_without_consuming_new_location(self):
        captured = PokemonIdentity(99, 20, 30)
        other_wild = PokemonIdentity(12, 20, 30)
        events = (
            self.start(species=("Poochyena",)),
            PokemonCaptured(2, captured, self.a, "Poochyena"),
            self.end(),
            self.start(3, self.b, identity=other_wild, species=("poochyena",)),
            self.end(4, self.b),
        )

        state = reduce_rules(events)

        self.assertEqual(state.captured_species, ("Poochyena",))
        self.assertEqual(len(state.encounters), 1)
        self.assertEqual(state.encounters[0].location, self.a)
        self.assertEqual(state.encounters[0].status, CAPTURED)

    def test_species_clause_matching_is_case_insensitive_and_new_species_is_recorded(self):
        captured = PokemonIdentity(99, 20, 30)
        state = reduce_rules(
            (
                self.start(species=("ZIGZAGOON",)),
                PokemonCaptured(2, captured, self.a, "zigzagoon"),
            )
        )

        self.assertEqual(state.captured_species, ("zigzagoon",))
        self.assertEqual(state.encounters[0].status, CAPTURED)

    def test_species_clause_can_be_disabled(self):
        captured = PokemonIdentity(99, 20, 30)
        projection = NuzlockeRulesProjection(
            rule_config=CampaignRulesConfig(frozenset({CampaignRuleId.ONE_ENCOUNTER_PER_AREA}))
        )
        projection.apply(self.start(species=("Poochyena",)), sequence=1)
        projection.apply(PokemonCaptured(2, captured, self.a, "Poochyena"), sequence=2)
        projection.apply(self.end(), sequence=3)
        projection.apply(self.start(4, self.b, species=("Poochyena",)), sequence=4)

        self.assertEqual(len(projection.state.encounters), 2)
        self.assertEqual(projection.state.encounters[-1].status, PENDING)

    def test_species_clause_treats_evolution_line_as_one_species(self):
        captured = PokemonIdentity(99, 20, 30)
        state = reduce_rules(
            (
                self.start(species=("Poochyena",)),
                PokemonCaptured(2, captured, self.a, "Poochyena"),
            )
        )

        self.assertIsNotNone(species_clause_conflict_reason(state, ("Mightyena",)))
        self.assertIsNone(species_clause_conflict_reason(state, ("Zigzagoon",)))

    def test_species_clause_preserves_wurmple_branch_exception(self):
        captured = PokemonIdentity(99 << 16, 20, 30)
        state = reduce_rules(
            (
                self.start(species=("Wurmple",)),
                PokemonCaptured(2, captured, self.a, "Wurmple"),
            )
        )

        self.assertEqual(state.captured_evolution_branches, (("wurmple", "cascoon"),))
        self.assertIsNotNone(species_clause_conflict_reason(state, ("Wurmple",)))
        self.assertIsNotNone(species_clause_conflict_reason(state, ("Cascoon",)))
        self.assertIsNotNone(species_clause_conflict_reason(state, ("Dustox",)))
        self.assertIsNone(species_clause_conflict_reason(state, ("Silcoon",)))
        self.assertIsNone(species_clause_conflict_reason(state, ("Beautifly",)))

    def test_species_clause_matches_lowercase_evolution_names(self):
        captured = PokemonIdentity(99, 20, 30)
        state = reduce_rules(
            (
                self.start(species=("wurmple",)),
                PokemonCaptured(2, captured, self.a, "wurmple", "cascoon"),
            )
        )

        self.assertIsNotNone(species_clause_conflict_reason(state, ("dustox",)))
        self.assertIsNone(species_clause_conflict_reason(state, ("silcoon",)))

    def test_violation_reason_is_deterministic(self):
        events = (self.start(), self.end(), self.start(3))
        self.assertEqual(reduce_rules(events).violations, reduce_rules(events).violations)


if __name__ == "__main__":
    unittest.main()
