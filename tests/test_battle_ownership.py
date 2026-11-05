from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_planner import (
    PlannerAction,
    PlannerConfidence,
    PlannerDecision,
    PlannerDecisionClass,
    PlannerSafety,
)
from modules.battle_strategies import DefaultBattleStrategy, TurnAction
from modules.context import context


def test_default_strategy_executes_an_incomplete_expectiminimax_projection_directly():
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        confidence=PlannerConfidence.MEDIUM,
        rationale="expectiminimax projected the least-risk legal move",
        safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SimpleNamespace(complete=False),
        evaluation_method="expectiminimax",
    )
    strategy = DefaultBattleStrategy.__new__(DefaultBattleStrategy)

    with patch("modules.battle_strategies.default.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.use_move(0)

    assert context.battle_decision_source == "PLANNER BEST_AVAILABLE"
    assert context.battle_evaluation_method == "expectiminimax"


def test_default_strategy_keeps_legacy_fallback_when_expectiminimax_aborts():
    decision = PlannerDecision(
        PlannerAction.AbortForSafety,
        confidence=PlannerConfidence.INSUFFICIENT,
        rationale="no complete legal search result",
        safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.ABORT,
        evaluation_method="expectiminimax",
    )
    strategy = DefaultBattleStrategy.__new__(DefaultBattleStrategy)
    strategy._capture_strategy = None
    active = SimpleNamespace(
        current_hp_percentage=80,
        moves=(SimpleNamespace(move=SimpleNamespace(name="Tackle")),),
    )
    battle_state = SimpleNamespace(
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace()),
    )

    with (
        patch("modules.battle_strategies.default.plan_battle_state", return_value=decision),
        patch("modules.battle_strategies.default.BattleStrategyUtil") as util,
    ):
        util.return_value.pokemon_has_enough_hp.return_value = True
        util.return_value.get_strongest_move_against.return_value = 0
        assert strategy.decide_turn(battle_state) == TurnAction.use_move(0)

    assert context.battle_decision_source == "LEGACY FALLBACK"


def test_capture_strategy_remains_a_legacy_capture_policy():
    from modules.battle_strategies.catch import CatchStrategy

    strategy = CatchStrategy.__new__(CatchStrategy)
    ball = SimpleNamespace(name="Poké Ball")
    battle_state = SimpleNamespace(
        opponent=SimpleNamespace(active_battler=SimpleNamespace(status_permanent="healthy")),
        own_side=SimpleNamespace(active_battler=SimpleNamespace(moves=())),
    )
    with (
        patch.object(strategy, "capture_target_is_authorized", return_value=True),
        patch.object(strategy, "_get_best_poke_ball", return_value=ball),
        patch.object(strategy, "_get_safe_weakening_move", return_value=None),
    ):
        assert strategy.decide_turn(battle_state) == TurnAction.use_item(ball)


def test_nuzlocke_training_strategy_publishes_facts_then_delegates_to_campaign_planner():
    from modules.battle_strategies.nuzlocke_level_balancing import NuzlockeLevelBalancingBattleStrategy
    from modules.battle_strategies.campaign import CampaignExpectiminimaxStrategy

    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    expected = TurnAction.use_move(2)
    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts") as publish,
        patch.object(CampaignExpectiminimaxStrategy, "decide_turn", return_value=expected) as planner,
    ):
        assert strategy.decide_turn(SimpleNamespace()) == expected
    publish.assert_called_once()
    planner.assert_called_once()
