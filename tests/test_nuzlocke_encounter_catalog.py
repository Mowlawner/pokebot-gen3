import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from modules.map import WildEncounter, WildEncounterList
from modules.modes._interface import BotModeError
from modules.nuzlocke.capture_policy import EncounterCandidate, EncounterMethod
from modules.nuzlocke.campaign_execution import (
    CampaignExecutionStatus,
    adapt_campaign_execution,
)
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    EncounterClassification,
    EncounterEvaluation,
    EncounterRecommendation,
    ObjectiveSelection,
    ObjectiveStatus,
    _select_available_campaign_tasks,
    encounter_task,
)
from modules.nuzlocke.campaign_state import Fact
from modules.nuzlocke.encounter_catalog import (
    available_encounter_methods,
    campaign_encounter_methods,
    encounter_candidates_for_location,
)
from modules.nuzlocke.snapshots import InventorySnapshot, ItemQuantity, NamedFlag


class _CapabilityState:
    """Small state double for pure encounter-capability tests."""

    def __init__(self, *, badges=(), moves=(), items=()):
        self.badges = Fact.known(tuple(NamedFlag(name, True) for name in badges))
        self.party = Fact.known(
            (
                SimpleNamespace(
                    egg=False,
                    moves=tuple(SimpleNamespace(name=name) for name in moves),
                ),
            )
        )
        self.inventory = Fact.known(
            InventorySnapshot(
                (),
                (),
                tuple(ItemQuantity(name, 1) for name in items),
            )
        )

    def has_badge(self, name):
        return Fact.known(any(flag.name == name and flag.value for flag in self.badges.value))

    def item_quantity(self, name):
        return Fact.known(next((item.quantity for item in self.inventory.value.key_items if item.name == name), 0))


def _water_tables():
    water = SimpleNamespace(name="Water", types=(SimpleNamespace(name="Water"),))
    return WildEncounterList(
        land_encounter_rate=0,
        surf_encounter_rate=60,
        rock_smash_encounter_rate=0,
        fishing_encounter_rate=20,
        land_encounters=[],
        surf_encounters=[WildEncounter(water, 5, 5, 60)],
        rock_smash_encounters=[],
        old_rod_encounters=[WildEncounter(water, 5, 5, 70)],
        good_rod_encounters=[WildEncounter(water, 5, 5, 60)],
        super_rod_encounters=[WildEncounter(water, 5, 5, 40)],
    )


class EncounterCatalogTests(unittest.TestCase):
    def test_method_capabilities_require_their_save_resources(self):
        early = _CapabilityState()
        self.assertEqual(available_encounter_methods(early), frozenset({EncounterMethod.LAND}))

        capable = _CapabilityState(
            badges=("BADGE03_GET", "BADGE05_GET"),
            moves=("Rock Smash", "Surf"),
            items=("Old Rod", "Good Rod"),
        )
        self.assertEqual(
            available_encounter_methods(capable),
            frozenset(
                {
                    EncounterMethod.LAND,
                    EncounterMethod.ROCK_SMASH,
                    EncounterMethod.SURF,
                    EncounterMethod.FISHING_OLD_ROD,
                    EncounterMethod.FISHING_GOOD_ROD,
                }
            ),
        )

    def test_catalog_can_show_all_tables_but_filter_to_current_methods(self):
        with patch(
            "modules.nuzlocke.encounter_catalog.get_wild_encounters_for_map",
            return_value=_water_tables(),
        ):
            all_candidates = encounter_candidates_for_location((0, 0))
            self.assertEqual(
                {candidate.method for candidate in all_candidates},
                {
                    EncounterMethod.SURF,
                    EncounterMethod.FISHING_OLD_ROD,
                    EncounterMethod.FISHING_GOOD_ROD,
                    EncounterMethod.FISHING_SUPER_ROD,
                },
            )

            early_candidates = encounter_candidates_for_location(
                (0, 0),
                available_methods=campaign_encounter_methods(_CapabilityState()),
            )
            self.assertEqual(early_candidates, ())

            old_rod_candidates = encounter_candidates_for_location(
                (0, 0),
                available_methods=frozenset({EncounterMethod.FISHING_OLD_ROD}),
            )
            self.assertEqual(
                tuple(candidate.method for candidate in old_rod_candidates),
                (EncounterMethod.FISHING_OLD_ROD,),
            )

    def test_campaign_method_set_stays_land_only_until_non_land_executor_exists(self):
        capable = _CapabilityState(badges=("BADGE05_GET",), moves=("Surf",))
        self.assertEqual(campaign_encounter_methods(capable), frozenset({EncounterMethod.LAND}))

    def test_selected_non_land_method_is_rejected_before_mounting(self):
        objective = encounter_task((0, 0))
        objective = replace(objective, encounter_method=EncounterMethod.SURF)
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "test")

        result = adapt_campaign_execution(selection)

        self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED)
        self.assertIn("surf", result.reason)

    def test_selector_carries_the_winning_method_into_the_objective(self):
        optional = replace(
            encounter_task((0, 0)),
            encounter_evaluation=EncounterEvaluation(
                task_id="obtain_encounter:0:0",
                location=(0, 0),
                progression_destination=(0, 1),
                classification=EncounterClassification.ON_ROUTE,
                recommendation=EncounterRecommendation.PREFER,
                direct_progression_cost=1,
                encounter_cost=1,
                encounter_to_progression_cost=0,
                via_encounter_cost=1,
                detour_cost=0,
                detour_ratio=0.0,
            ),
        )
        required = CampaignObjective(
            "required",
            "required",
            (),
            CampaignPredicate("required_done", "required", lambda _: Fact.known(False)),
        )
        with patch(
            "modules.nuzlocke.campaign_objectives.encounter_candidates_for_location",
            return_value=(EncounterCandidate((0, 0), EncounterMethod.LAND, type_pool=("water",)),),
        ):
            selection = _select_available_campaign_tasks((optional, required), _CapabilityState())

        self.assertEqual(selection.objective.objective_id, optional.objective_id)
        self.assertEqual(selection.objective.encounter_method, EncounterMethod.LAND)

    def test_runtime_rejects_non_land_method_without_reading_the_emulator(self):
        from modules.nuzlocke.resource_runtime import execute_campaign_encounter

        with self.assertRaises(BotModeError):
            next(execute_campaign_encounter((0, 0), encounter_method=EncounterMethod.FISHING_OLD_ROD))


if __name__ == "__main__":
    unittest.main()
