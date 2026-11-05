from modules.battle_expectiminimax import SearchResult, SearchResultCategory
from modules.battle_readiness import BattleReadinessStatus, assess_battle_readiness
from modules.nuzlocke.trainer_policy import TrainerPolicyInput, choose_trainer_mode
from modules.goals import TrainerMode
from modules.goals import EngageTrainer, NavigationGoal, SemanticTarget
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    CampaignReadinessPolicy,
    ReadinessDecision,
    ReadinessReason,
)


def test_readiness_turns_a_safe_search_into_ready_checkpoint():
    result = SearchResult(0, "Tackle", 100000, 1.0, 0.0, 0.0, ("Tackle", "KO"))
    readiness = assess_battle_readiness(type("Checkpoint", (), {"checkpoint_id": "route102:first_poochyena"})(), result)

    assert readiness.status is BattleReadinessStatus.READY
    assert readiness.ready is True
    assert readiness.recommended_action == "Tackle"


def test_unresolved_search_requires_preparation_without_claiming_failure():
    readiness = assess_battle_readiness(type("Checkpoint", (), {"checkpoint_id": "unknown"})(), None)

    assert readiness.status is BattleReadinessStatus.UNKNOWN
    assert readiness.ready is None


def test_autonomous_horizon_root_is_ready_but_keeps_unproven_metrics_unknown():
    result = SearchResult(
        0,
        "Pound",
        0.0,
        0.0,
        0.0,
        0.0,
        category=SearchResultCategory.HORIZON_PROJECTION,
    )

    readiness = assess_battle_readiness(
        type("Checkpoint", (), {"checkpoint_id": "rom:BPEE0:532"})(),
        result,
    )

    assert readiness.status is BattleReadinessStatus.READY
    assert readiness.ready is True
    assert readiness.win_probability is None
    assert readiness.catastrophic_probability is None
    assert readiness.run_loss_probability is None


def test_unready_required_trainer_is_not_engaged_by_navigation_policy():
    result = SearchResult(0, "Tackle", 0, 0.4, 0.6, 0.6, ("Tackle", "FAINT"))
    readiness = assess_battle_readiness(type("Checkpoint", (), {"checkpoint_id": "route102:first_poochyena"})(), result)

    assert choose_trainer_mode(
        TrainerPolicyInput(trainer_required=True, battle_readiness=readiness)
    ) is TrainerMode.AVOID


def test_encounter_seeking_never_silently_accepts_a_blocking_trainer():
    assert choose_trainer_mode(
        TrainerPolicyInput(seeking_encounter=True, trainer_blocks_route=True)
    ) is TrainerMode.AVOID


def test_route_exposed_known_trainer_uses_the_same_preparation_gate_as_explicit_goal():
    checkpoint = type("Checkpoint", (), {"checkpoint_id": "route102:first_poochyena"})()
    trainer_readiness = assess_battle_readiness(
        checkpoint,
        SearchResult(0, "Tackle", 0, 0.4, 0.6, 0.6, ("Tackle", "FAINT")),
    )
    readiness = type(
        "Readiness",
        (),
        {
            "overworld_availability": Availability.KNOWN,
            "game_state": "OVERWORLD",
            "resource_availability": Availability.KNOWN,
            "party_availability": Availability.KNOWN,
            "usable_count": 1,
            "fainted_count": 0,
            "trainer_readiness": trainer_readiness,
            "trainer_engagement_required": True,
            "navigation_goal": NavigationGoal(SemanticTarget.map((0, 17))),
            "needs_status_recovery": False,
            "lowest_hp_ratio": 0.9,
            "recovery_availability": Availability.KNOWN,
            "route_analysis": type("Route", (), {"normal_cost": 1})(),
        },
    )()

    result = CampaignReadinessPolicy().evaluate(readiness)

    assert result.decision is ReadinessDecision.PREPARE
    assert result.reason is ReadinessReason.TRAINER_BATTLE_UNREADY


def test_unready_known_trainer_checkpoint_requests_recovery_before_engagement():
    checkpoint = type("Checkpoint", (), {"checkpoint_id": "route102:first_poochyena"})()
    trainer_readiness = assess_battle_readiness(
        checkpoint,
        SearchResult(0, "Tackle", 0, 0.4, 0.6, 0.6, ("Tackle", "FAINT")),
    )
    readiness = type(
        "Readiness",
        (),
        {
            "overworld_availability": Availability.KNOWN,
            "game_state": "OVERWORLD",
            "resource_availability": Availability.KNOWN,
            "party_availability": Availability.KNOWN,
            "usable_count": 1,
            "fainted_count": 0,
            "trainer_readiness": trainer_readiness,
            "navigation_goal": NavigationGoal(EngageTrainer("trainer:route102")),
            "needs_status_recovery": False,
            "lowest_hp_ratio": 0.9,
            "recovery_availability": Availability.KNOWN,
        },
    )()

    result = CampaignReadinessPolicy().evaluate(readiness)

    assert result.decision is ReadinessDecision.PREPARE
    assert result.reason is ReadinessReason.TRAINER_BATTLE_UNREADY
