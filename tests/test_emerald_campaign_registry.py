import unittest

from modules.nuzlocke.campaign_state import CampaignFacts, Fact, FactStatus
from modules.nuzlocke.emerald_campaign_registry import (
    active_emerald_boss,
    emerald_bosses,
    emerald_capability_definition,
    emerald_objectives,
)
from modules.goals import SemanticTargetKind, TrainerMode
from modules.map_data import MapRSE


def facts(*, first_badge=Fact.known(False)):
    values = [Fact.unavailable() for _ in range(16)]
    values[-2] = first_badge
    return CampaignFacts(*values)


class EmeraldCampaignRegistryTests(unittest.TestCase):
    def test_implemented_boss_data_is_ordered_and_rom_owned(self):
        bosses = emerald_bosses()

        self.assertEqual(len(bosses), 1)
        self.assertEqual(bosses[0].boss_id, "roxanne")
        self.assertEqual(bosses[0].level_cap, 15)

    def test_active_boss_uses_authoritative_completion_fact(self):
        progression = active_emerald_boss(facts())

        self.assertEqual(progression.status, FactStatus.KNOWN)
        self.assertEqual(progression.active_boss.boss_id, "roxanne")

    def test_unavailable_completion_does_not_guess_boss_progression(self):
        progression = active_emerald_boss(facts(first_badge=Fact.unavailable()))

        self.assertEqual(progression.status, FactStatus.UNAVAILABLE)
        self.assertIsNone(progression.active_boss)

    def test_completed_implemented_slice_has_no_active_boss(self):
        progression = active_emerald_boss(facts(first_badge=Fact.known(True)))

        self.assertEqual(progression.status, FactStatus.KNOWN)
        self.assertIsNone(progression.active_boss)

    def test_first_badge_capability_definition_owns_rom_target(self):
        definition = emerald_capability_definition("defeat_roxanne")

        self.assertIsNotNone(definition)
        self.assertEqual(definition.capability_id, "emerald_observation")
        self.assertEqual(definition.semantic_target.kind, SemanticTargetKind.INTERACTION)
        self.assertEqual(definition.semantic_target.target_map, MapRSE.RUSTBORO_CITY_GYM.value)
        self.assertEqual(definition.semantic_target.interaction_id, "roxanne")
        self.assertEqual(definition.readiness_goal.constraints.trainer_mode, TrainerMode.AVOID)

    def test_preparation_capability_definition_owns_target_level(self):
        definition = emerald_capability_definition("prepare_roxanne")

        self.assertIsNotNone(definition)
        self.assertEqual(definition.capability_id, "emerald_preparation")
        self.assertEqual(definition.semantic_target.target_map, MapRSE.ROUTE116.value)
        self.assertEqual(definition.preparation_level, 14)

    def test_petalburg_capability_definition_owns_norman_target(self):
        definition = emerald_capability_definition("complete_petalburg_wally")

        self.assertIsNotNone(definition)
        self.assertEqual(definition.semantic_target.kind, SemanticTargetKind.INTERACTION)
        self.assertEqual(definition.semantic_target.target_map, MapRSE.PETALBURG_CITY_GYM.value)
        self.assertEqual(
            definition.semantic_target.interaction_id,
            "PetalburgCity_Gym_EventScript_Norman",
        )

    def test_devon_progression_facts_are_registered(self):
        registered = {item.completion_fact for item in emerald_objectives()}

        self.assertTrue(
            {
                "petalburg_woods_scene_complete",
                "devon_goods_stolen",
                "devon_goods_reported",
                "devon_goods_recovered",
                "devon_goods_returned",
                "devon_corp_3f_scene_complete",
                "visited_rustboro",
                "first_badge_obtained",
            }
            <= registered
        )

    def test_devon_capabilities_are_registry_owned(self):
        expected = {
            "complete_petalburg_woods": (MapRSE.PETALBURG_WOODS.value, "devon_goods_researcher"),
            "complete_rustboro_goods_stolen": (MapRSE.RUSTBORO_CITY.value, "rustboro_goods_stolen"),
            "report_devon_goods": (MapRSE.RUSTBORO_CITY.value, "rustboro_goods_report"),
            "recover_devon_goods": (MapRSE.RUSTURF_TUNNEL.value, "rusturf_tunnel_goods"),
            "return_devon_goods": (MapRSE.RUSTBORO_CITY.value, "rustboro_return_devon_goods"),
        }

        for objective_id, (map_id, interaction_id) in expected.items():
            definition = emerald_capability_definition(objective_id)
            self.assertIsNotNone(definition, objective_id)
            self.assertEqual(definition.semantic_target.kind, SemanticTargetKind.INTERACTION)
            self.assertEqual(definition.semantic_target.target_map, map_id)
            self.assertEqual(definition.semantic_target.interaction_id, interaction_id)


if __name__ == "__main__":
    unittest.main()
