import unittest
from dataclasses import dataclass
from enum import Enum

from pydantic import ValidationError

from modules.config.schemas_v1 import NuzlockeRules
from modules.nuzlocke.events import BattleStarted, PokemonFainted, WhiteoutOccurred
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.rule_config import CampaignRule, CampaignRuleId, CampaignRulesConfig
from modules.nuzlocke.rules import NuzlockeRulesProjection
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.snapshots import (
    CampaignObservationSnapshot,
    InventorySnapshot,
    NuzlockeSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    StorageSnapshot,
)


class FixtureGameState(Enum):
    OVERWORLD = 1


def runtime_snapshot(frame: int) -> NuzlockeSnapshot:
    return NuzlockeSnapshot(
        frame,
        "test",
        FixtureGameState.OVERWORLD,
        PlayerSnapshot("May", 1, 2, "MAP", (1, 1), "Down", True),
        (),
        InventorySnapshot((), (), ()),
        None,
        StorageSnapshot(0, ()),
        ProgressionSnapshot(()),
        campaign_observation=CampaignObservationSnapshot(),
    )


@dataclass(frozen=True)
class Candidate:
    allowed: bool


class StubRule:
    rule_id = CampaignRuleId.LEVEL_CAP

    def evaluate(self, state):
        return Candidate(state is not None)

    def constrain(self, state, candidate):
        return Candidate(candidate.allowed and state is not None)


class CampaignRulesConfigTests(unittest.TestCase):
    identity = PokemonIdentity(10, 20, 30)

    def test_default_preserves_existing_nuzlocke_rules(self):
        config = CampaignRulesConfig()

        self.assertTrue(config.is_enabled(CampaignRuleId.ONE_ENCOUNTER_PER_AREA))
        self.assertTrue(config.is_enabled(CampaignRuleId.FAINTING))
        self.assertTrue(config.is_enabled(CampaignRuleId.LEVEL_CAP))

    def test_unrestricted_config_disables_baseline_rule_reduction(self):
        projection = NuzlockeRulesProjection(rule_config=CampaignRulesConfig.unrestricted())
        events = (
            BattleStarted(1, ("WILD",), False, True, False, (), (self.identity,), (1, 2)),
            PokemonFainted(2, 0, "Treecko", "Sprig", 10, "battle", self.identity),
            WhiteoutOccurred(3),
        )

        for sequence, event in enumerate(events, 1):
            projection.apply(event, sequence=sequence)

        self.assertEqual(projection.state.encounters, ())
        self.assertEqual(projection.state.dead_pokemon, ())
        self.assertFalse(projection.state.run_lost)
        self.assertTrue(projection.state.legal)

    def test_runtime_retains_rule_configuration_when_reset(self):
        config = CampaignRulesConfig.unrestricted()
        runtime = NuzlockeRuntime(rule_config=config)

        runtime.update(runtime_snapshot(5))
        runtime.update(runtime_snapshot(1))

        self.assertIs(runtime.rule_config, config)
        self.assertIs(runtime.rules_projection.rule_config, config)

    def test_profile_schema_rejects_unknown_rule_ids(self):
        with self.assertRaises(ValidationError):
            NuzlockeRules(enabled_rules=["not_a_rule"])

    def test_rule_contract_is_pure_structural_extension_point(self):
        rule = StubRule()

        self.assertIsInstance(rule, CampaignRule)
        self.assertEqual(rule.evaluate(object()).allowed, True)
        self.assertTrue(rule.constrain(object(), Candidate(True)).allowed)


if __name__ == "__main__":
    unittest.main()
