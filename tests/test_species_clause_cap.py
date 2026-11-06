import unittest
from dataclasses import replace

from modules.nuzlocke.events import BattleStarted
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.persistence import deserialize_event, serialize_event
from modules.nuzlocke.rule_config import CampaignRuleId, CampaignRulesConfig
from modules.nuzlocke.rules import (
    PENDING,
    LocationEncounter,
    NuzlockeCampaignState,
    NuzlockeRulesProjection,
    reduce_rules,
    species_clause_conflict_reason,
)

AREA = (1, 2)
OTHER = (1, 3)


def started(frame=1, location=AREA, species=("Poochyena",), *, duplicate=False):
    return BattleStarted(
        frame,
        ("WILD",),
        False,
        True,
        False,
        (PokemonIdentity(1, 2, 3),),
        (PokemonIdentity(10 + frame, 20, 30),),
        location,
        opponent_species=species,
        species_clause_duplicate=duplicate,
    )


def state_with(captured=("Poochyena",), duplicates=()):
    return NuzlockeCampaignState(captured_species=captured, duplicate_encounters=duplicates)


class SpeciesClauseCapConfigTests(unittest.TestCase):
    def test_the_default_is_unlimited(self):
        self.assertIsNone(CampaignRulesConfig().species_clause_max_duplicates_per_area)
        self.assertIsNone(
            CampaignRulesConfig.from_names(["species_clause"]).species_clause_max_duplicates_per_area
        )

    def test_a_positive_cap_is_accepted_and_passed_through(self):
        config = CampaignRulesConfig.from_names(
            ["species_clause"], species_clause_max_duplicates_per_area=3
        )

        self.assertEqual(config.species_clause_max_duplicates_per_area, 3)

    def test_an_invalid_cap_is_rejected(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CampaignRulesConfig(species_clause_max_duplicates_per_area=value)

    def test_the_config_schema_defaults_to_unlimited_and_validates(self):
        from pydantic import ValidationError

        from modules.config.schemas_v1 import NuzlockeRules

        self.assertIsNone(NuzlockeRules().species_clause_max_duplicates_per_area)
        self.assertEqual(NuzlockeRules(species_clause_max_duplicates_per_area=4).species_clause_max_duplicates_per_area, 4)
        with self.assertRaises(ValidationError):
            NuzlockeRules(species_clause_max_duplicates_per_area=0)


class SpeciesClauseCapDecisionTests(unittest.TestCase):
    def test_without_a_cap_a_duplicate_is_always_a_conflict(self):
        state = state_with(duplicates=((AREA, 50),))

        self.assertIsNotNone(species_clause_conflict_reason(state, ("Poochyena",), location=AREA))

    def test_below_the_cap_a_duplicate_is_still_skipped(self):
        state = state_with(duplicates=((AREA, 2),))

        reason = species_clause_conflict_reason(state, ("Poochyena",), location=AREA, max_duplicates=3)

        self.assertIsNotNone(reason)

    def test_at_the_cap_the_clause_is_waived_for_that_area(self):
        state = state_with(duplicates=((AREA, 3),))

        self.assertIsNone(
            species_clause_conflict_reason(state, ("Poochyena",), location=AREA, max_duplicates=3)
        )

    def test_the_cap_is_counted_per_area(self):
        state = state_with(duplicates=((AREA, 3),))

        self.assertIsNotNone(
            species_clause_conflict_reason(state, ("Poochyena",), location=OTHER, max_duplicates=3)
        )

    def test_a_new_species_is_never_a_conflict_regardless_of_the_cap(self):
        state = state_with()

        self.assertIsNone(species_clause_conflict_reason(state, ("Wurmple",), location=AREA, max_duplicates=1))


class SpeciesClauseCapCountingTests(unittest.TestCase):
    def test_skipped_duplicates_are_counted_per_area(self):
        state = reduce_rules(
            (
                started(1, AREA, duplicate=True),
                started(2, AREA, duplicate=True),
                started(3, OTHER, duplicate=True),
                started(4, AREA, species=("Wurmple",)),  # not a skipped duplicate
            )
        )

        self.assertEqual(state.duplicates_at(AREA), 2)
        self.assertEqual(state.duplicates_at(OTHER), 1)
        self.assertEqual(state.duplicates_at((9, 9)), 0)

    def test_the_count_is_deterministic_across_replay(self):
        events = tuple(started(frame, AREA, duplicate=True) for frame in range(1, 5))

        first = reduce_rules(events)
        second = reduce_rules(events)

        self.assertEqual(first.duplicate_encounters, second.duplicate_encounters)
        self.assertEqual(first.duplicates_at(AREA), 4)

    def test_nothing_is_counted_when_species_clause_is_disabled(self):
        projection = NuzlockeRulesProjection(
            rule_config=CampaignRulesConfig(frozenset({CampaignRuleId.ONE_ENCOUNTER_PER_AREA}))
        )
        projection.apply(started(1, AREA, duplicate=True), sequence=1)

        self.assertEqual(projection.state.duplicates_at(AREA), 0)

    def test_the_flag_survives_persistence_and_old_events_default_to_false(self):
        event = started(1, AREA, duplicate=True)

        restored = deserialize_event(serialize_event(event))
        self.assertTrue(restored.species_clause_duplicate)

        payload = serialize_event(event)
        del payload["payload"]["species_clause_duplicate"]  # written before this field existed
        self.assertFalse(deserialize_event(payload).species_clause_duplicate)


class SpeciesClauseCapRuntimeTests(unittest.TestCase):
    """The runtime must apply the configured cap when deciding capture eligibility."""

    def _runtime(self, cap):
        from tests.test_nuzlocke_runtime import InventorySnapshot, ItemQuantity, snapshot
        from modules.nuzlocke.runtime import NuzlockeRuntime

        runtime = NuzlockeRuntime(
            rule_config=CampaignRulesConfig.from_names(
                ["one_encounter_per_area", "species_clause"],
                species_clause_max_duplicates_per_area=cap,
            )
        )
        balls = InventorySnapshot((), (ItemQuantity("Poké Ball", 3),), ())
        runtime.update(replace(snapshot(1, pokedex_received=True), inventory=balls))
        return runtime

    def _eligibility(self, runtime, duplicates):
        projection = runtime.rules_projection
        projection._state = NuzlockeCampaignState(
            encounters=(LocationEncounter(AREA, PENDING, eligible=True),),
            captured_species=("Poochyena",),
            duplicate_encounters=duplicates,
        )
        return runtime.capture_eligibility_for(
            AREA, is_wild=True, is_trainer=False, species=("Poochyena",)
        )

    def test_with_no_cap_a_duplicate_is_never_a_capture_target(self):
        runtime = self._runtime(None)

        result = self._eligibility(runtime, ((AREA, 99),))

        self.assertFalse(result.eligible)
        self.assertTrue(result.duplicate)

    def test_below_the_cap_the_duplicate_is_still_skipped(self):
        runtime = self._runtime(2)

        result = self._eligibility(runtime, ((AREA, 1),))

        self.assertFalse(result.eligible)
        self.assertTrue(result.duplicate)

    def test_at_the_cap_the_duplicate_becomes_the_areas_capture_target(self):
        runtime = self._runtime(2)

        result = self._eligibility(runtime, ((AREA, 2),))

        self.assertTrue(result.eligible)
        self.assertFalse(result.duplicate)


if __name__ == "__main__":
    unittest.main()
