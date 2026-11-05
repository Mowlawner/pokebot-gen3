from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from modules.battle_expectiminimax import (
    Expectiminimax,
    SearchConfig,
    SearchItem,
    SearchMove,
    SearchResult,
    SearchResultCategory,
    SearchState,
)
from modules.battle_menuing import scroll_to_battle_action
from modules.battle_planner import (
    PlannerAction,
    PlannerConfidence,
    PlannerDecision,
    PlannerDecisionClass,
    PlannerSafety,
)
from modules.battle_strategies.campaign import CampaignExpectiminimaxStrategy
from modules.battle_strategies.default import DefaultBattleStrategy
from modules.battle_strategies import TurnAction
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeCaptureStrategy,
    NuzlockeLevelBalancingBattleStrategy,
    NuzlockeRunAwayStrategy,
    RoxanneBattleStrategy,
)
from modules.context import context


def test_campaign_strategies_do_not_reach_the_legacy_strategy_chain():
    campaign_classes = (
        NuzlockeLevelBalancingBattleStrategy,
        NuzlockeCaptureStrategy,
        NuzlockeRunAwayStrategy,
        EmeraldIntroRivalBattleStrategy,
        RoxanneBattleStrategy,
    )

    assert all(DefaultBattleStrategy not in strategy.__mro__ for strategy in campaign_classes)
    assert all(CampaignExpectiminimaxStrategy in strategy.__mro__ for strategy in campaign_classes)


def test_campaign_base_returns_the_expectiminimax_root_action():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.HIGH,
        "expectiminimax selected Tackle",
        PlannerSafety.SAFE,
        classification=PlannerDecisionClass.SAFE,
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision) as planner:
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.use_move(0)

    planner.assert_called_once()
    assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX"


def test_campaign_rejects_an_unsafe_incomplete_projection_instead_of_substituting_a_move():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.MEDIUM,
        "expectiminimax returned an incomplete line",
        PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.INCOMPLETE,
            projection_safe=False,
            is_executable_projection=False,
        ),
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.switch_to_manual()


def test_campaign_executes_a_concrete_unresolved_root_autonomously():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.MEDIUM,
        "expectiminimax returned the best available Pound root",
        PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SearchResult(
            0,
            "Pound",
            0.0,
            0.0,
            0.0,
            0.0,
            category=SearchResultCategory.HORIZON_PROJECTION,
            projection_safe=False,
        ),
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.use_move(0)

    assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX BEST_AVAILABLE"
    assert "category='horizon_projection'" in context.battle_decision_detail


def test_campaign_translates_a_reserve_healing_root_to_a_targeted_item_action():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    potion = SimpleNamespace(name="Potion", battle_use=SimpleNamespace(value="healing"))
    decision = PlannerDecision(
        PlannerAction.UseItem,
        (potion, 1),
        PlannerConfidence.MEDIUM,
        "expectiminimax selected Potion for reserve slot 1",
        PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SearchResult(
            0,
            "Potion",
            0.0,
            0.0,
            0.0,
            0.0,
            action_kind="item",
            category=SearchResultCategory.INCOMPLETE,
            target_party_index=1,
        ),
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.use_item_on(
            potion, 1
        )

    assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX BEST_AVAILABLE"


def test_campaign_executes_a_concrete_complete_best_available_root_without_guaranteed_loss():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.MEDIUM,
        "expectiminimax selected Scratch; the complete line remains probabilistically exposed",
        PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SearchResult(
            0,
            "Scratch",
            0.0,
            0.875,
            0.125,
            0.0,
            category=SearchResultCategory.TERMINAL_COMPLETE,
        ),
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=True)) == TurnAction.use_move(0)

    assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX BEST_AVAILABLE"
    assert "category='terminal_complete'" in context.battle_decision_detail


def test_campaign_executes_a_low_confidence_defeat_root_to_resolve_loss_autonomously():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.INSUFFICIENT,
        "the only legal defeat root leads to an active faint",
        PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SearchResult(
            0,
            "Tackle",
            -100000.0,
            0.0,
            1.0,
            1.0,
            category=SearchResultCategory.TERMINAL_COMPLETE,
            objective="defeat",
        ),
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=True)) == TurnAction.use_move(0)

    assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX BEST_AVAILABLE"
    assert "category='terminal_complete'" in context.battle_decision_detail


def test_campaign_rejects_a_terminal_objective_failure():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.RunAway,
        confidence=PlannerConfidence.MEDIUM,
        rationale="the legal capture target would be lost",
        safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.TERMINAL_COMPLETE,
            win_probability=0.0,
            is_executable_projection=False,
        ),
        evaluation_method="expectiminimax",
    )

    with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=decision):
        assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.switch_to_manual()

    assert "safety rejection" in context.battle_decision_detail
    assert "no executable expectiminimax root action" not in context.battle_decision_detail


def test_campaign_switch_mismatch_forces_fresh_observation_before_replan():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    decision = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.HIGH,
        "expectiminimax selected Tackle after switch reconciliation",
        PlannerSafety.SAFE,
        classification=PlannerDecisionClass.SAFE,
        evaluation_method="expectiminimax",
    )
    battle_state = SimpleNamespace(is_trainer_battle=False)
    observation = object()
    previous_replan = context.battle_switch_replan_required
    previous_status = context.battle_switch_reconciliation_status
    context.battle_switch_replan_required = True
    context.battle_switch_reconciliation_status = "mismatch"
    try:
        with (
            patch(
                "modules.battle_strategies.campaign.observe_current_battle",
                return_value=observation,
            ) as observe,
            patch(
                "modules.battle_strategies.campaign.default_battle_recorder.record"
            ) as record,
            patch(
                "modules.battle_strategies.campaign.plan_battle_state",
                return_value=decision,
            ) as planner,
        ):
            assert strategy.decide_turn(battle_state) == TurnAction.use_move(0)

        observe.assert_called_once_with()
        record.assert_called_once_with(observation)
        planner.assert_called_once()
        assert context.battle_switch_replan_required is False
        assert context.battle_switch_reconciliation_status == "replanning"
    finally:
        context.battle_switch_replan_required = previous_replan
        context.battle_switch_reconciliation_status = previous_status


def test_campaign_empty_root_refreshes_battle_facts_once_before_replanning():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    empty_root = PlannerDecision(
        PlannerAction.AbortForSafety,
        confidence=PlannerConfidence.INSUFFICIENT,
        rationale="Expectiminimax could not construct a legal root",
        safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
    )
    valid_root = PlannerDecision(
        PlannerAction.UseMove,
        0,
        PlannerConfidence.HIGH,
        "expectiminimax selected Tackle after refreshing the battle boundary",
        PlannerSafety.SAFE,
        classification=PlannerDecisionClass.SAFE,
    )
    initial_state = SimpleNamespace(is_trainer_battle=False)
    refreshed_state = SimpleNamespace(
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=1)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=16)),
    )
    previous_refresh = getattr(context, "battle_empty_root_replan_used", False)
    context.battle_empty_root_replan_used = False
    try:
        with (
            patch(
                "modules.battle_strategies.campaign.plan_battle_state",
                side_effect=(empty_root, valid_root),
            ) as planner,
            patch(
                "modules.battle_strategies.campaign.observe_current_battle",
                return_value=object(),
            ) as observe,
            patch(
                "modules.battle_strategies.campaign.get_battle_state",
                return_value=refreshed_state,
            ) as get_state,
            patch("modules.battle_strategies.campaign.state_cache.invalidate_runtime_observations") as invalidate,
        ):
            assert strategy.decide_turn(initial_state) == TurnAction.use_move(0)

        assert planner.call_count == 2
        observe.assert_called_once_with()
        invalidate.assert_called_once_with()
        get_state.assert_called_once_with()
        assert context.battle_empty_root_replan_used is True
    finally:
        context.battle_empty_root_replan_used = previous_refresh


def test_campaign_empty_root_requests_a_frame_retry_without_inventing_an_action():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    empty_root = PlannerDecision(
        PlannerAction.AbortForSafety,
        confidence=PlannerConfidence.INSUFFICIENT,
        rationale="Expectiminimax could not construct a legal root",
        safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
    )
    initial_state = SimpleNamespace(is_trainer_battle=False)
    refreshed_state = SimpleNamespace(
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=1)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=16)),
    )
    previous_refresh = getattr(context, "battle_empty_root_replan_used", False)
    previous_retry_requested = getattr(context, "battle_planner_retry_requested", False)
    previous_retry_count = getattr(context, "battle_planner_retry_count", 0)
    context.battle_empty_root_replan_used = False
    context.battle_planner_retry_requested = False
    context.battle_planner_retry_count = 0
    try:
        with (
            patch(
                "modules.battle_strategies.campaign.plan_battle_state",
                side_effect=(empty_root, empty_root),
            ) as planner,
            patch(
                "modules.battle_strategies.campaign.observe_current_battle",
                return_value=object(),
            ) as observe,
            patch(
                "modules.battle_strategies.campaign.get_battle_state",
                return_value=refreshed_state,
            ),
            patch("modules.battle_strategies.campaign.state_cache.invalidate_runtime_observations"),
        ):
            assert strategy.decide_turn(initial_state) == TurnAction.switch_to_manual()

        assert planner.call_count == 2
        observe.assert_called_once_with()
        assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX RETRY"
        assert "waiting for a fresh battle snapshot" in context.battle_decision_detail
        assert context.battle_planner_retry_requested is True
        assert context.battle_planner_retry_count == 1
    finally:
        context.battle_empty_root_replan_used = previous_refresh
        context.battle_planner_retry_requested = previous_retry_requested
        context.battle_planner_retry_count = previous_retry_count


def test_campaign_empty_root_stops_only_after_retry_budget_is_exhausted():
    strategy = CampaignExpectiminimaxStrategy.__new__(CampaignExpectiminimaxStrategy)
    empty_root = PlannerDecision(
        PlannerAction.AbortForSafety,
        confidence=PlannerConfidence.INSUFFICIENT,
        rationale="Expectiminimax could not construct a legal root",
        safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
    )
    previous_refresh = getattr(context, "battle_empty_root_replan_used", False)
    previous_retry_requested = getattr(context, "battle_planner_retry_requested", False)
    previous_retry_count = getattr(context, "battle_planner_retry_count", 0)
    context.battle_empty_root_replan_used = True
    context.battle_planner_retry_requested = False
    context.battle_planner_retry_count = 10_000
    try:
        with patch("modules.battle_strategies.campaign.plan_battle_state", return_value=empty_root):
            assert strategy.decide_turn(SimpleNamespace(is_trainer_battle=False)) == TurnAction.switch_to_manual()

        assert context.battle_decision_source == "CAMPAIGN EXPECTIMINIMAX STOP"
        assert context.battle_decision_detail.startswith("no normalized legal planner root was returned;")
        assert "planner_rationale='Expectiminimax could not construct a legal root'" in context.battle_decision_detail
        assert "candidate_inventory=()" in context.battle_decision_detail
    finally:
        context.battle_empty_root_replan_used = previous_refresh
        context.battle_planner_retry_requested = previous_retry_requested
        context.battle_planner_retry_count = previous_retry_count


def test_battle_menu_cursor_timeout_reports_the_selected_boundary():
    emulator = Mock()
    previous_emulator = context.emulator
    context.emulator = emulator
    try:
        with patch("modules.battle_menuing.read_symbol", return_value=bytes((0, 0, 0, 0))):
            with pytest.raises(RuntimeError, match="gActionSelectionCursor.*target=1"):
                list(scroll_to_battle_action(1))
    finally:
        context.emulator = previous_emulator


def test_ready_capture_ball_wins_a_tie_against_a_setup_move():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=2,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),),
        opponent_moves=(SearchMove(0, "Splash", 0, 0),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 5, is_capture_ball=True, capture_probability=0.5),),
    )

    result = Expectiminimax(SearchConfig(max_turns=3)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"


def test_potion_loses_when_the_active_pokemon_already_survives_and_can_ko():
    state = SearchState(
        player_hp=18,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 5, 5),),
        player_speed=20,
        opponent_speed=10,
        player_items=(SearchItem(0, "Potion", 1, heal_amount=20),),
    )

    assert Expectiminimax(SearchConfig(max_turns=3)).evaluate(state).action_name == "Tackle"


def test_potion_wins_when_it_is_the_survivable_line():
    state = SearchState(
        player_hp=2,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 4, 4),),
        opponent_moves=(SearchMove(0, "Tackle", 5, 5),),
        player_speed=10,
        opponent_speed=20,
        player_items=(SearchItem(0, "Potion", 1, heal_amount=20),),
    )

    assert Expectiminimax(SearchConfig(max_turns=3)).evaluate(state).action_name == "Potion"
