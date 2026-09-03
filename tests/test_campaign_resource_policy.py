from modules.nuzlocke.resource_policy import (
    EncounterPolicy,
    HealingResource,
    PokeballRestockPolicy,
    PartyResource,
    ReadinessImportance,
    ResourceDecision,
    ResourceObjective,
    ResourceSnapshot,
    RouteRecovery,
    assess_campaign_resources,
    assess_wild_encounter,
)
from modules.nuzlocke.resource_runtime import CampaignCapability
from modules.goals import ActivateTrigger
from unittest.mock import patch

IMPORTANT = ResourceObjective(
    "dangerous_trainer",
    readiness=ReadinessImportance.IMPORTANT,
    encounters=EncounterPolicy.PRESERVE,
    mandatory_battle=True,
)


def test_pokeball_restock_policy_uses_lower_threshold_for_detours():
    policy = PokeballRestockPolicy(lower_threshold=5, upper_target=10)

    assert policy.needs_restock(4)
    assert not policy.needs_restock(5)
    assert not policy.needs_restock(8)
    assert policy.quantity_to_buy(4) == 6
    assert policy.quantity_to_buy(10) == 0
    assert policy.quantity_to_buy(12) == 0


def test_pokeball_restock_policy_caps_purchase_at_available_money():
    policy = PokeballRestockPolicy(lower_threshold=5, upper_target=10)

    assert policy.affordable_quantity(2, money=700, unit_price=200) == 3
    assert policy.affordable_quantity(2, money=0, unit_price=200) == 0


def test_pokeball_restock_policy_rejects_malformed_ranges():
    import pytest

    with pytest.raises(ValueError):
        PokeballRestockPolicy(lower_threshold=-1, upper_target=10)
    with pytest.raises(ValueError):
        PokeballRestockPolicy(lower_threshold=10, upper_target=10)


def party(hp, maximum=20):
    return ResourceSnapshot((PartyResource(hp, maximum),))


def test_healthy_party_can_continue_without_immediate_danger():
    objective = ResourceObjective("walk")
    assert assess_campaign_resources(objective, party(20), RouteRecovery()) is ResourceDecision.CONTINUE


def test_healthy_party_does_not_force_center_before_important_battle():
    assert (
        assess_campaign_resources(IMPORTANT, party(18), RouteRecovery(center_available=True, center_on_route=True))
        is ResourceDecision.CONTINUE
    )


def test_damaged_party_uses_center_when_safe_and_available():
    assert (
        assess_campaign_resources(IMPORTANT, party(6), RouteRecovery(center_available=True, safe_to_reach_center=True))
        is ResourceDecision.RECOVER_AT_CENTER
    )


def test_damaged_party_uses_bag_item_when_center_is_unsafe():
    snapshot = ResourceSnapshot((PartyResource(8, 20),), (HealingResource("Potion", 1, 20),))
    route = RouteRecovery(center_available=True, safe_to_reach_center=False)
    assert assess_campaign_resources(IMPORTANT, snapshot, route) is ResourceDecision.USE_HEALING_ITEM


def test_damaged_party_can_withdraw_accessible_pc_item():
    snapshot = ResourceSnapshot((PartyResource(8, 20),), pc_healing_items=(HealingResource("Potion", 1, 20, "pc"),))
    route = RouteRecovery(pc_accessible=True, pc_acquisition_cost=1)
    assert assess_campaign_resources(IMPORTANT, snapshot, route) is ResourceDecision.WITHDRAW_HEALING_ITEM


def test_inaccessible_pc_item_is_not_selected():
    snapshot = ResourceSnapshot((PartyResource(8, 20),), pc_healing_items=(HealingResource("Potion", 1, 20, "pc"),))
    route = RouteRecovery(pc_accessible=False)
    assert assess_campaign_resources(IMPORTANT, snapshot, route) is ResourceDecision.PRESERVE_RESOURCES


def test_preserve_policy_prefers_run_for_wild_encounter():
    assert assess_wild_encounter(IMPORTANT, party(4), RouteRecovery()) is ResourceDecision.PREFER_RUN


def test_grinding_objective_allows_wild_battle():
    objective = ResourceObjective("grind", encounters=EncounterPolicy.GRIND)
    assert assess_wild_encounter(objective, party(4), RouteRecovery()) is ResourceDecision.CONTINUE


def test_important_objective_preserves_spent_hp_before_optional_wild_battle():
    assert assess_wild_encounter(IMPORTANT, party(18), RouteRecovery()) is ResourceDecision.PREFER_RUN


def test_important_objective_can_spend_healthy_margin_before_natural_recovery():
    route = RouteRecovery(center_available=True, center_on_route=True)
    assert assess_wild_encounter(IMPORTANT, party(18), route) is ResourceDecision.CONTINUE


def test_important_objective_at_full_hp_does_not_run_from_first_wild_battle():
    assert assess_wild_encounter(IMPORTANT, party(20), RouteRecovery()) is ResourceDecision.CONTINUE


def test_normal_encounter_policy_does_not_preserve_optional_wild_battle():
    objective = ResourceObjective(
        "important",
        readiness=ReadinessImportance.IMPORTANT,
        mandatory_battle=True,
    )
    assert assess_wild_encounter(objective, party(4), RouteRecovery()) is ResourceDecision.PRESERVE_RESOURCES


def test_non_wild_battle_never_becomes_run_decision():
    assert (
        assess_campaign_resources(IMPORTANT, party(4), RouteRecovery(), wild_encounter=False)
        is ResourceDecision.PRESERVE_RESOURCES
    )


def test_healthy_capability_does_not_resolve_recovery_context():
    calls = []

    def delegate(_objective_id):
        calls.append("delegate")
        yield

    capability = CampaignCapability("complete_intro_rival", IMPORTANT, ActivateTrigger("test"), delegate=delegate)
    with patch(
        "modules.nuzlocke.resource_runtime.observe_resource_snapshot", side_effect=AssertionError("unexpected lookup")
    ):
        next(capability())
    assert calls == ["delegate"]


def test_damaged_capability_resolves_recovery_context_lazily():
    calls = []

    def delegate(_objective_id):
        calls.append("delegate")
        yield

    capability = CampaignCapability("complete_intro_rival", IMPORTANT, ActivateTrigger("test"), delegate=delegate)
    with patch(
        "modules.nuzlocke.resource_runtime.observe_resource_snapshot", side_effect=AssertionError("unexpected lookup")
    ), patch(
        "modules.nuzlocke.resource_runtime.observe_route_recovery", side_effect=AssertionError("unexpected lookup")
    ):
        next(capability())
    assert calls == ["delegate"]


def test_damaged_capability_can_recover_at_center_then_resume():
    events = []

    def delegate(_objective_id):
        events.append("delegate")
        yield

    def recovery():
        events.append("recovery")
        yield

    capability = CampaignCapability("complete_intro_rival", IMPORTANT, ActivateTrigger("test"), delegate=delegate)
    list(capability())
    assert events == ["delegate"]
