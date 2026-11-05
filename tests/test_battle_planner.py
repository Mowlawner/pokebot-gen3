import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from modules.battle_observation import (
    BattleKnowledge,
    DamageOccurred,
    KnowledgePolicy,
    PokemonSwitched,
)
from modules.battle_expectiminimax import SearchResult, SearchRootEvaluation, SearchResultCategory
from modules.battle_planner import (
    BattlePlanner,
    PlannerAction,
    PlannerConfidence,
    PlannerContext,
    PlannerDecision,
    PlannerDecisionClass,
    PlannerEvidenceKind,
    PlannerSafety,
    PlannerMove,
    PlannerItem,
    PlannerOpponentMove,
    PlannerRisk,
    PlannerSwitch,
    RecoveryItem,
    _battle_hp_facts,
    _battle_moves,
    _knowledge_hp_facts,
    knowledge_for_planner,
    plan_battle_state,
    format_planner_decision_diagnostic,
    format_search_principal_variation,
    normalized_battle_objective,
    _planner_stat_stages,
    _observed_switch_pressure,
)
from modules.battle_state import BattleType
from modules.battle_strategies.default import DefaultBattleStrategy
from modules.battle_strategies.catch import CatchStrategy
from modules.battle_strategies import TurnAction
from modules.context import context as shared_context


def context(**changes):
    values = dict(
        active_name="Treecko",
        active_hp=20,
        active_max_hp=20,
        opponent_name="Wurmple",
        opponent_hp=12,
        moves=(PlannerMove(0, "Scratch", 12, 15),),
    )
    values.update(changes)
    return PlannerContext(**values)


class BattlePlannerTests(unittest.TestCase):
    def test_observed_switch_pressure_survives_new_planner_roots_until_damage(self):
        knowledge = BattleKnowledge(
            "battle",
            KnowledgePolicy.HUMAN_KNOWLEDGE,
            events=(
                DamageOccurred(1, "opponent", 0, 2, 20, 18),
                PokemonSwitched(2, "player", 0, 1),
                PokemonSwitched(3, "player", 1, 2),
                PokemonSwitched(4, "player", 2, 0),
            ),
        )

        self.assertEqual(_observed_switch_pressure(knowledge), (3, 1))

    def test_battle_boundary_falls_back_only_to_matching_party_slot(self):
        party = (
            SimpleNamespace(index=0, current_hp=18, total_hp=20),
            SimpleNamespace(index=1, current_hp=12, total_hp=20),
        )
        transient_active = SimpleNamespace(party_index=0, current_hp=0, total_hp=0, moves=())
        unrelated_active = SimpleNamespace(party_index=2, current_hp=0, total_hp=0, moves=())
        move = SimpleNamespace(name="Tackle")
        party[0].moves = (move,)

        self.assertEqual(_battle_hp_facts(transient_active, party), (18, 20))
        self.assertEqual(_battle_moves(transient_active, party), (move,))
        self.assertEqual(_battle_hp_facts(unrelated_active, party), (0, 0))
        self.assertEqual(_battle_moves(unrelated_active, party), ())

    def test_battle_boundary_uses_latest_same_battle_observation_for_zero_zero_hp(self):
        observation = SimpleNamespace(
            player_active=(SimpleNamespace(party_index=0, species="Wurmple", current_hp=13, max_hp=13),),
            opponent_active=(SimpleNamespace(party_index=0, species="Lotad", current_hp=18, max_hp=18),),
        )
        knowledge = SimpleNamespace(observations=(observation,))

        self.assertEqual(
            _knowledge_hp_facts(
                knowledge,
                SimpleNamespace(party_index=0, species=SimpleNamespace(name="Wurmple")),
                side="player",
            ),
            (13, 13),
        )
        self.assertEqual(
            _knowledge_hp_facts(
                knowledge,
                SimpleNamespace(party_index=0, species=SimpleNamespace(name="Lotad")),
                side="opponent",
            ),
            (18, 18),
        )

    def test_battle_boundary_does_not_restore_an_explicit_faint(self):
        observation = SimpleNamespace(
            player_active=(SimpleNamespace(party_index=0, species="Wurmple", current_hp=0, max_hp=13),),
        )
        knowledge = SimpleNamespace(observations=(observation,))

        self.assertIsNone(
            _knowledge_hp_facts(
                knowledge,
                SimpleNamespace(party_index=0, species=SimpleNamespace(name="Wurmple")),
                side="player",
            )
        )

    def test_first_battle_is_always_a_defeat_objective(self):
        previous_intent = shared_context.campaign_wild_battle_intent
        shared_context.campaign_wild_battle_intent = "transit"
        try:
            state = SimpleNamespace(type=BattleType.FirstBattle, is_trainer_battle=False)
            self.assertEqual(normalized_battle_objective(state), "defeat")
        finally:
            shared_context.campaign_wild_battle_intent = previous_intent

    def test_unavailable_escape_is_not_presented_as_a_legal_candidate(self):
        decision = BattlePlanner().plan(
            context(
                moves=(PlannerMove(0, "Tackle", 10, 10),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
                is_wild=True,
                can_run=False,
            )
        )

        self.assertFalse(any(candidate.action is PlannerAction.RunAway for candidate in decision.candidate_evaluations))

    def test_principal_variation_labels_search_escape_as_player_action(self):
        result = SimpleNamespace(
            principal_variation=("Tackle", "Tackle", "Run Away", "FAILED", "Tackle"),
            action_kind="move",
            category=None,
        )

        formatted = format_search_principal_variation(result, context(active_speed=10, opponent_speed=20))

        self.assertIn("player:Run Away", formatted)
        self.assertNotIn("opponent:Run Away", formatted)

    def test_decision_diagnostic_logs_the_effective_battle_intent(self):
        decision = BattlePlanner().plan(
            context(
                battle_objective="avoid",
                is_wild=True,
                can_run=True,
                escape_probability=1.0,
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )

        diagnostic = format_planner_decision_diagnostic(decision, context(
            battle_objective="avoid",
            is_wild=True,
            can_run=True,
            escape_probability=1.0,
            opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        ))

        self.assertIn("effective_intent='avoid'", diagnostic)

    def test_live_stage_note_uses_literal_delimiters_for_rich(self):
        note = BattlePlanner._live_stage_note(
            context(
                active_stat_stages=(("attack", 2),),
                opponent_stat_stages=(("defense", -1),),
            )
        )

        self.assertIn("Live stages: active (attack=+2); opponent (defense=-1).", note)

    def test_live_stat_stage_diagnostics_keep_readable_fields_when_one_stage_is_unreadable(self):
        class PartialModifiers:
            attack = 2
            speed = -1

            @property
            def defence(self):
                raise RuntimeError("transient stage read")

        stages = dict(_planner_stat_stages(SimpleNamespace(stats_modifiers=PartialModifiers())))

        self.assertEqual(stages, {"attack": 2, "speed": -1})

    def test_recovery_item_exposes_unified_hp_effect_name(self):
        potion = SimpleNamespace(name="Potion")
        item = RecoveryItem(potion, 2, 20)
        self.assertEqual(item.amount_healed, 20)
        self.assertFalse(item.status_healed)

    def test_full_heal_is_status_only_and_cures_all_statuses(self):
        item = RecoveryItem(SimpleNamespace(name="Full Heal"), 1, 0, status_healed=frozenset({"ALL"}))
        self.assertEqual(item.amount_healed, 0)
        self.assertTrue(item.cures_all_status)

    def test_abort_retains_ranked_legal_candidate_for_executor_policy(self):
        decision = BattlePlanner().plan(
            context(
                is_trainer=True,
                moves=(PlannerMove(0, "Unsupported", 1, 2, mechanics_supported=False),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)
        self.assertIsNotNone(decision.best_available_candidate)
        self.assertEqual(decision.best_available_candidate.target, 0)

    def test_catch_strategy_ignores_empty_ball_slots(self):
        empty = SimpleNamespace(quantity=0, item=SimpleNamespace(name="Poké Ball", index=4))
        usable = SimpleNamespace(quantity=2, item=SimpleNamespace(name="Great Ball", index=3))
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        with patch(
            "modules.battle_strategies.catch.get_item_bag",
            return_value=SimpleNamespace(poke_balls=(empty, usable)),
        ), patch("modules.battle_strategies.default.get_party", return_value=party), patch.object(
            CatchStrategy, "_get_poke_ball_catch_rate_multiplier", return_value=1
        ):
            selected = CatchStrategy()._get_best_poke_ball(SimpleNamespace())
        self.assertIs(selected, usable.item)

    def test_catch_strategy_delegates_when_no_ball_exists(self):
        expected = (TurnAction.UseMove, 0)
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        with patch(
            "modules.battle_strategies.catch.get_item_bag",
            return_value=SimpleNamespace(poke_balls=(SimpleNamespace(quantity=0, item=SimpleNamespace(index=4)),)),
        ), patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.DefaultBattleStrategy.decide_turn",
            return_value=expected,
        ) as ordinary:
            result = CatchStrategy().decide_turn(SimpleNamespace())
        self.assertEqual(result, expected)
        ordinary.assert_called_once()

    def test_capture_target_prefers_a_ball_when_weakening_does_not_change_odds(self):
        ball = SimpleNamespace(name="Poké Ball")
        decision = BattlePlanner().plan(
            context(
                opponent_hp=10,
                moves=(PlannerMove(0, "Strong", 10, 14), PlannerMove(1, "Weak", 3, 5)),
                capture_target=True,
                capture_available=True,
                capture_probability_for_state=lambda _state, _item: 0.5,
                items=(PlannerItem(ball, 3, 0, True),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseItem)
        self.assertEqual(decision.target, (ball, 0))

    def test_capture_target_prefers_ball_at_capture_hp(self):
        ball = SimpleNamespace(name="Poké Ball")
        decision = BattlePlanner().plan(
            context(
                opponent_hp=2,
                capture_target=True,
                capture_available=True,
                capture_probability_for_state=lambda _state, _item: 1.0,
                items=(PlannerItem(ball, 5, 0, True),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseItem)
        self.assertEqual(decision.target, (ball, 0))

    def test_capture_target_with_only_lethal_attacks_uses_ball(self):
        ball = SimpleNamespace(name="Poké Ball")
        decision = BattlePlanner().plan(
            context(
                opponent_hp=10,
                moves=(PlannerMove(0, "Strong", 10, 14),),
                capture_target=True,
                capture_available=True,
                capture_probability_for_state=lambda _state, _item: 1.0,
                items=(PlannerItem(ball, 1, 0, True),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseItem)

    def test_capture_target_without_capture_resources_keeps_attack_policy(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=10,
                moves=(PlannerMove(0, "Strong", 10, 14),),
                capture_target=True,
                capture_available=False,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)

    def test_healing_is_not_selected_when_active_pokemon_is_at_full_hp(self):
        potion = SimpleNamespace(name="Potion")
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                active_max_hp=20,
                moves=(),
                opponent_damage_max=5,
                items=(PlannerItem(potion, 1, 20),),
            )
        )

        self.assertNotEqual(decision.action, PlannerAction.UseItem)
        self.assertIsNone(decision.best_safe_candidate)
        healing_candidate = next(
            candidate
            for candidate in decision.candidate_evaluations
            if candidate.action is PlannerAction.UseItem
        )
        self.assertFalse(healing_candidate.is_safe)
        self.assertIn("full HP", healing_candidate.reason)

    def test_complete_expectiminimax_can_own_a_healing_item_decision(self):
        potion = SimpleNamespace(name="Potion")
        decision = BattlePlanner().plan(
            context(
                active_hp=5,
                active_max_hp=20,
                opponent_hp=11,
                moves=(PlannerMove(0, "Tackle", 10, 10),),
                opponent_moves=(PlannerOpponentMove("Tackle", 5, 5),),
                active_speed=10,
                opponent_speed=20,
                is_trainer=True,
                opponent_response_facts_complete=True,
                items=(PlannerItem(potion, 1, 20),),
            )
        )

        self.assertEqual(decision.action, PlannerAction.UseItem)
        self.assertEqual(decision.target, (potion, 0))
        self.assertEqual(decision.evaluation_method, "expectiminimax")
        self.assertEqual(decision.stochastic_result.action_kind, "item")

    def test_healing_is_not_selected_when_active_already_clears_response_ceiling(self):
        potion = SimpleNamespace(name="Potion")
        decision = BattlePlanner().plan(
            context(
                active_hp=17,
                active_max_hp=18,
                moves=(),
                opponent_damage_max=5,
                items=(PlannerItem(potion, 1, 20),),
            )
        )

        self.assertNotEqual(decision.action, PlannerAction.UseItem)
        healing_candidate = next(
            candidate
            for candidate in decision.candidate_evaluations
            if candidate.action is PlannerAction.UseItem
        )
        self.assertFalse(healing_candidate.is_safe)
        self.assertIn("above the known response ceiling", healing_candidate.reason)

    def test_safe_item_action_can_target_a_benched_party_member(self):
        potion = SimpleNamespace(name="Potion")
        decision = BattlePlanner().plan(
            context(
                active_hp=17,
                active_max_hp=18,
                moves=(),
                switches=(PlannerSwitch(1, "Mudkip", 2, 20, None),),
                opponent_damage_max=5,
                items=(PlannerItem(potion, 1, 20),),
            )
        )

        self.assertEqual(decision.action, PlannerAction.UseItem)
        self.assertEqual(decision.target, (potion, 1))

    def test_best_available_diagnostic_explains_unknown_response_and_rejected_move(self):
        battle_context = context(
            opponent_hp=30,
            moves=(PlannerMove(0, "Pound", 8, 10), PlannerMove(1, "Tail Whip", 0, 0, effect="DEFENSE_DOWN")),
        )
        decision = BattlePlanner().plan(battle_context)
        diagnostic = format_planner_decision_diagnostic(decision, battle_context)
        self.assertIn("BEST_AVAILABLE", diagnostic)
        self.assertIn("No fully safe line established", diagnostic)
        self.assertIn("Pound", diagnostic)
        self.assertIn("opponent_response_unknown=True", diagnostic)
        self.assertIn("setup_line=False", diagnostic)

    def test_default_strategy_consults_planner_above_and_below_hp_threshold(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        active = SimpleNamespace(current_hp_percentage=80)
        battle_state = SimpleNamespace(own_side=SimpleNamespace(active_battler=active))
        safe_attack = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=2,
            confidence=PlannerConfidence.HIGH,
            rationale="safe",
            is_safe_to_execute=True,
        )
        safe_switch = SimpleNamespace(
            action=PlannerAction.SwitchPokemon,
            target=1,
            confidence=PlannerConfidence.MEDIUM,
            rationale="switch",
            is_safe_to_execute=True,
        )
        util = SimpleNamespace()
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch(
            "modules.battle_strategies.default.plan_battle_state", side_effect=[safe_attack, safe_switch]
        ) as planner:
            strategy = DefaultBattleStrategy()
            self.assertEqual(strategy.decide_turn(battle_state), (TurnAction.UseMove, 2))

            active.current_hp_percentage = 5
            self.assertEqual(strategy.decide_turn(battle_state), (TurnAction.RotateLead, 1))

        self.assertEqual(planner.call_count, 2)

    def test_ready_legal_nuzlocke_battle_delegates_to_existing_capture_strategy(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=(SimpleNamespace(quantity=1),))
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True,
            own_side=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        expected = (TurnAction.UseItem, SimpleNamespace(name="Poké Ball"))
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn", return_value=expected
        ) as capture_turn, patch(
            "modules.battle_strategies.default.plan_battle_state",
            return_value=SimpleNamespace(
                action=PlannerAction.UseItem, classification=PlannerDecisionClass.SAFE, is_safe_to_execute=True
            ),
        ):
            result = DefaultBattleStrategy().decide_turn(battle_state)
        self.assertIs(result[0], TurnAction.UseItem)
        self.assertEqual(result[1].name, "Poké Ball")
        capture_turn.assert_called_once_with(battle_state)

    def test_legal_target_uses_safe_planner_move_before_capture(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=(SimpleNamespace(quantity=1),))
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True, own_side=SimpleNamespace(active_battler=SimpleNamespace())
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove, target=1, classification=PlannerDecisionClass.SAFE, is_safe_to_execute=True
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=decision), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn"
        ) as catch:
            catch.return_value = (TurnAction.UseMove, 1)
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 1))
        catch.assert_called_once_with(battle_state)

    def test_legal_target_falls_back_to_capture_when_planner_is_not_safe(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=(SimpleNamespace(quantity=1),))
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True, own_side=SimpleNamespace(active_battler=SimpleNamespace())
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=1,
            classification=PlannerDecisionClass.BEST_AVAILABLE,
            is_safe_to_execute=False,
        )
        expected = (TurnAction.UseItem, SimpleNamespace(name="Poké Ball"))
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=decision), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn", return_value=expected
        ):
            result = DefaultBattleStrategy().decide_turn(battle_state)
        self.assertIs(result[0], TurnAction.UseItem)

    def test_legal_target_without_balls_returns_to_normal_battle_policy(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=())
        active = SimpleNamespace(current_hp_percentage=80)
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True,
            own_side=SimpleNamespace(active_battler=active),
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=1,
            confidence=PlannerConfidence.HIGH,
            classification=PlannerDecisionClass.SAFE,
            is_safe_to_execute=True,
            rationale="ordinary battle action",
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=decision), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn"
        ) as catch:
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 1))
        catch.assert_not_called()

    def test_ready_non_target_does_not_delegate_to_capture_strategy(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        active = SimpleNamespace(current_hp_percentage=80)
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=False,
            own_side=SimpleNamespace(active_battler=active),
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=0,
            confidence=PlannerConfidence.HIGH,
            rationale="safe",
            is_safe_to_execute=True,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.plan_battle_state", return_value=decision
        ), patch("modules.battle_strategies.catch.CatchStrategy.decide_turn") as capture_turn:
            result = DefaultBattleStrategy().decide_turn(battle_state)
        self.assertEqual(result, (TurnAction.UseMove, 0))
        capture_turn.assert_not_called()

    def test_safe_planner_actions_keep_existing_turn_action_shapes(self):
        item = SimpleNamespace(name="Potion")
        cases = (
            (PlannerAction.UseMove, 0, (TurnAction.UseMove, 0)),
            (PlannerAction.SwitchPokemon, 2, (TurnAction.RotateLead, 2)),
            (PlannerAction.UseItem, (item, 0), (TurnAction.UseItem, (item, 0))),
            (PlannerAction.RunAway, None, (TurnAction.RunAway, None)),
        )
        for action, target, expected in cases:
            decision = SimpleNamespace(action=action, target=target)
            self.assertEqual(DefaultBattleStrategy._turn_action_from_planner_decision(decision), expected)

    def test_best_available_planner_action_is_accepted_without_safety_flag(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp_percentage=80)),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        best = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=1,
            confidence=PlannerConfidence.MEDIUM,
            rationale="unknown response",
            is_safe_to_execute=False,
            classification=PlannerDecisionClass.BEST_AVAILABLE,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil"
        ) as util, patch("modules.battle_strategies.default.plan_battle_state", return_value=best):
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 1))
            util.return_value.pokemon_has_enough_hp.assert_not_called()

    def test_default_strategy_uses_legacy_move_after_planner_abort(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp_percentage=80)),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        candidate = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=2,
            label="use Unsupported",
            reason="unsupported mechanics",
        )
        aborted = SimpleNamespace(
            action=PlannerAction.AbortForSafety,
            target=None,
            confidence=PlannerConfidence.INSUFFICIENT,
            rationale="no safe line",
            is_safe_to_execute=False,
            classification=PlannerDecisionClass.ABORT,
            best_available_candidate=candidate,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil"
        ) as util, patch("modules.battle_strategies.default.plan_battle_state", return_value=aborted):
            util.return_value.pokemon_has_enough_hp.return_value = True
            util.return_value.get_strongest_move_against.return_value = 2
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 2))
            util.return_value.pokemon_has_enough_hp.assert_called_once_with(
                battle_state.own_side.active_battler
            )

    def test_unsafe_planner_result_keeps_legacy_attack_fallback_above_threshold(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp_percentage=80)),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        util = SimpleNamespace(
            pokemon_has_enough_hp=lambda pokemon: True,
            get_strongest_move_against=lambda active, opponent: 3,
        )
        unsafe = SimpleNamespace(
            action=PlannerAction.AbortForSafety,
            target=None,
            confidence=PlannerConfidence.INSUFFICIENT,
            rationale="unknown response",
            is_safe_to_execute=False,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=unsafe):
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 3))

    def test_strict_strategy_does_not_use_legacy_attack_fallback(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp_percentage=80)),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        util = SimpleNamespace(
            pokemon_has_enough_hp=lambda pokemon: True,
            get_strongest_move_against=Mock(return_value=3),
        )
        unsafe = SimpleNamespace(
            action=PlannerAction.AbortForSafety,
            target=None,
            confidence=PlannerConfidence.INSUFFICIENT,
            rationale="unknown response",
            is_safe_to_execute=False,
        )

        class StrictStrategy(DefaultBattleStrategy):
            def allows_legacy_fallback(self):
                return False

        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=unsafe):
            self.assertEqual(StrictStrategy().decide_turn(battle_state), (TurnAction.SwitchToManual, None))

        util.get_strongest_move_against.assert_not_called()

    def test_live_party_switch_candidates_use_ordered_slot_not_party_index(self):
        class PartyPokemonWithoutPartyIndex:
            def __init__(self, name, hp):
                self.name = name
                self.current_hp = hp
                self.total_hp = 20
                self.is_egg = False
                self.moves = ()

        move = SimpleNamespace(name="Scratch", base_power=40)
        active_move = SimpleNamespace(move=move, pp=1)
        active = SimpleNamespace(
            party_index=0,
            species=SimpleNamespace(name="Treecko"),
            current_hp=2,
            total_hp=20,
            moves=(active_move,),
            disabled_move=None,
        )
        opponent = SimpleNamespace(
            species=SimpleNamespace(name="Wurmple"),
            current_hp=12,
            moves=(),
        )
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=active),
            opponent=SimpleNamespace(active_battler=opponent),
            is_trainer_battle=False,
        )
        alternate = PartyPokemonWithoutPartyIndex("Torchic", 18)
        alternate.moves = (SimpleNamespace(move=move, pp=1),)
        party = [PartyPokemonWithoutPartyIndex("Treecko", 2), alternate]
        damage_range = SimpleNamespace(min=3, max=4)
        util = SimpleNamespace(
            calculate_move_damage_range=lambda *args: damage_range,
            can_switch=lambda: True,
            get_escape_chance=lambda: 0,
        )
        knowledge = BattleKnowledge(
            "battle-1",
            KnowledgePolicy.HUMAN_KNOWLEDGE,
            revealed_opponent_moves=((("opponent", None, None), "Tackle"),),
        )

        with patch("modules.pokemon_party.get_party", return_value=party), patch(
            "modules.battle_strategies._util.BattleStrategyUtil", return_value=util
        ), patch("modules.pokemon.get_move_by_name", return_value=move):
            decision = plan_battle_state(battle_state, knowledge)

        self.assertEqual(decision.action, PlannerAction.SwitchPokemon)
        self.assertEqual(decision.target, 1)

    def test_live_planner_adapter_passes_battle_stat_stages(self):
        move = SimpleNamespace(
            name="Pound",
            base_power=40,
            accuracy=1.0,
            priority=0,
            effect="HIT",
            type=SimpleNamespace(kind="Physical"),
        )
        learned = SimpleNamespace(move=move, pp=1)
        modifiers = SimpleNamespace(
            attack=1,
            defence=-2,
            speed=0,
            special_attack=0,
            special_defence=0,
            accuracy=0,
            evasion=0,
        )
        species = SimpleNamespace(name="Treecko")
        active = SimpleNamespace(
            party_index=0,
            species=species,
            current_hp=10,
            total_hp=20,
            moves=(learned,),
            disabled_move=None,
            stats=SimpleNamespace(speed=10),
            stats_modifiers=modifiers,
            status_permanent=None,
        )
        opponent = SimpleNamespace(
            party_index=0,
            species=SimpleNamespace(name="Zigzagoon"),
            current_hp=10,
            total_hp=15,
            moves=(),
            stats=SimpleNamespace(speed=8),
            stats_modifiers=SimpleNamespace(
                attack=0,
                defence=-6,
                speed=0,
                special_attack=0,
                special_defence=0,
                accuracy=0,
                evasion=0,
            ),
        )
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=active),
            opponent=SimpleNamespace(active_battler=opponent),
            is_trainer_battle=True,
        )
        party_member = SimpleNamespace(
            is_egg=False,
            current_hp=20,
            total_hp=20,
            name="Treecko",
            moves=(learned,),
            status_condition=None,
        )
        damage_range = SimpleNamespace(min=2, max=3)
        util = SimpleNamespace(
            calculate_move_damage_range=lambda *args, **kwargs: damage_range,
            can_switch=lambda: False,
        )
        captured = {}

        def capture_plan(_planner, planner_context):
            captured["context"] = planner_context
            return PlannerDecision(PlannerAction.AbortForSafety)

        with (
            patch("modules.pokemon_party.get_party", return_value=(party_member,)),
            patch("modules.battle_strategies._util.BattleStrategyUtil", return_value=util),
            patch("modules.battle_planner.opponent_response_move_names", return_value=("Tackle",)),
            patch("modules.pokemon.get_move_by_name", return_value=move),
            patch("modules.items.get_item_bag", return_value=SimpleNamespace(items=(), poke_balls=())),
            patch.object(BattlePlanner, "plan", autospec=True, side_effect=capture_plan),
        ):
            plan_battle_state(battle_state, BattleKnowledge("battle", KnowledgePolicy.HUMAN_KNOWLEDGE))

        planner_context = captured["context"]
        self.assertEqual(dict(planner_context.active_stat_stages)["defense"], -2)
        self.assertEqual(dict(planner_context.opponent_stat_stages)["defense"], -6)

    def _planner_context_for_bide_opponent(self, **opponent_bide):
        move = SimpleNamespace(
            name="Pound",
            base_power=40,
            accuracy=1.0,
            priority=0,
            effect="HIT",
            type=SimpleNamespace(kind="Physical"),
        )
        learned = SimpleNamespace(move=move, pp=5)
        modifiers = SimpleNamespace(
            attack=0, defence=0, speed=0, special_attack=0, special_defence=0, accuracy=0, evasion=0
        )
        active = SimpleNamespace(
            party_index=0,
            species=SimpleNamespace(name="Torchic"),
            current_hp=20,
            total_hp=20,
            moves=(learned,),
            disabled_move=None,
            stats=SimpleNamespace(speed=10),
            stats_modifiers=modifiers,
            status_permanent=None,
        )
        opponent = SimpleNamespace(
            party_index=0,
            species=SimpleNamespace(name="Seedot"),
            current_hp=30,
            total_hp=30,
            moves=(),
            stats=SimpleNamespace(speed=8),
            stats_modifiers=modifiers,
            **opponent_bide,
        )
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=active),
            opponent=SimpleNamespace(active_battler=opponent),
            is_trainer_battle=True,
        )
        party_member = SimpleNamespace(
            is_egg=False,
            current_hp=20,
            total_hp=20,
            name="Torchic",
            moves=(learned,),
            status_condition=None,
        )
        util = SimpleNamespace(
            calculate_move_damage_range=lambda *args, **kwargs: SimpleNamespace(min=2, max=3),
            can_switch=lambda: False,
        )
        captured = {}

        def capture_plan(_planner, planner_context):
            captured["context"] = planner_context
            return PlannerDecision(PlannerAction.AbortForSafety)

        with (
            patch("modules.pokemon_party.get_party", return_value=(party_member,)),
            patch("modules.battle_strategies._util.BattleStrategyUtil", return_value=util),
            patch("modules.battle_planner.opponent_response_move_names", return_value=("Tackle",)),
            patch("modules.pokemon.get_move_by_name", return_value=move),
            patch("modules.items.get_item_bag", return_value=SimpleNamespace(items=(), poke_balls=())),
            patch.object(BattlePlanner, "plan", autospec=True, side_effect=capture_plan),
        ):
            plan_battle_state(battle_state, BattleKnowledge("battle", KnowledgePolicy.HUMAN_KNOWLEDGE))
        return captured["context"]

    def test_unreadable_bide_damage_on_release_turn_is_assumed_lethal_not_zero(self):
        planner_context = self._planner_context_for_bide_opponent(
            bide_turns_remaining=1,
            bide_damage=0,
            bide_damage_known=False,
        )

        self.assertEqual(planner_context.opponent_bide_turns, 1)
        # Twice the stored damage must be able to faint the 20 HP active Pokemon.
        self.assertGreaterEqual(planner_context.opponent_bide_damage * 2, planner_context.active_hp)

    def test_readable_zero_bide_damage_is_not_inflated(self):
        planner_context = self._planner_context_for_bide_opponent(
            bide_turns_remaining=1,
            bide_damage=0,
            bide_damage_known=True,
        )

        self.assertEqual(planner_context.opponent_bide_damage, 0)

    def test_unreadable_bide_damage_while_storing_is_not_inflated(self):
        planner_context = self._planner_context_for_bide_opponent(
            bide_turns_remaining=2,
            bide_damage=0,
            bide_damage_known=False,
        )

        self.assertEqual(planner_context.opponent_bide_damage, 0)

    def test_safe_attack_is_high_confidence(self):
        decision = BattlePlanner().plan(context())
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.confidence, PlannerConfidence.HIGH)

    def test_complete_wild_response_envelope_uses_expectiminimax(self):
        decision = BattlePlanner().plan(
            context(
                is_wild=True,
                opponent_hp=10,
                moves=(PlannerMove(0, "Pound", 5, 5),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 2),),
                opponent_response_facts_complete=True,
            )
        )

        self.assertEqual(decision.evaluation_method, "expectiminimax")
        self.assertIsNotNone(decision.stochastic_result)

    def test_partial_response_facts_cannot_be_called_safe(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 2, response_facts_complete=False),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertFalse(decision.is_safe_to_execute)
        self.assertIn("incomplete", decision.rationale)

    def test_paralysis_reduces_action_reliability(self):
        decision = BattlePlanner().plan(
            context(
                active_status="paralysed",
                opponent_hp=4,
                moves=(PlannerMove(0, "Pound", 4, 4),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertFalse(decision.is_safe_to_execute)

    def test_targeted_status_cure_beats_full_restore_when_hp_is_safe(self):
        antidote = SimpleNamespace(name="Antidote")
        full_restore = SimpleNamespace(name="Full Restore")
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                active_max_hp=20,
                active_status="poison",
                moves=(),
                opponent_damage_max=5,
                items=(
                    PlannerItem(antidote, 1, 0, False, frozenset({"poison"})),
                    PlannerItem(full_restore, 1, 0, False, frozenset({"ALL"}), True, 5),
                ),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseItem)
        self.assertIs(decision.target[0], antidote)

    def test_small_hp_gap_does_not_spend_full_restore_when_response_is_cleared(self):
        full_restore = SimpleNamespace(name="Full Restore")
        decision = BattlePlanner().plan(
            context(
                active_hp=19,
                active_max_hp=20,
                moves=(),
                opponent_damage_max=5,
                items=(PlannerItem(full_restore, 1, 0, False, frozenset({"ALL"}), True, 5),),
            )
        )
        self.assertNotEqual(decision.action, PlannerAction.UseItem)

    def test_immediate_ko_rationale_contains_known_and_derived_evidence(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                active_max_hp=20,
                opponent_name="Torchic",
                opponent_hp=4,
                moves=(PlannerMove(0, "Pound", 4, 5),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.safety, PlannerSafety.SAFE)
        self.assertIn("Pound", decision.rationale)
        self.assertIn("4-5 damage", decision.rationale)
        self.assertIn("8/20", decision.rationale)
        self.assertIn("guaranteed KO", decision.rationale)
        self.assertIn(PlannerEvidenceKind.KNOWN, {item.kind for item in decision.evidence})
        self.assertIn(PlannerEvidenceKind.DERIVED, {item.kind for item in decision.evidence})

    def test_guaranteed_immediate_ko_is_low_risk_safe(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=4,
                moves=(PlannerMove(0, "Pound", 4, 5),),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.SAFE)
        self.assertEqual(decision.risk, PlannerRisk.LOW)

    def test_risky_attack_is_ranked_against_a_worse_legal_attack(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=5,
                opponent_hp=12,
                moves=(
                    PlannerMove(0, "Pound", 4, 5),
                    PlannerMove(1, "Scratch", 2, 3),
                ),
                opponent_moves=(PlannerOpponentMove("Tackle", 3, 3),),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.target, 0)

    def test_critical_hit_ceiling_makes_winning_line_risky(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Tackle", 2, 2, critical_damage_max=10),),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.risk, PlannerRisk.CRITICAL)
        self.assertIn("critical-hit ceiling", decision.rationale)

    def test_safe_multi_turn_attack_is_established_from_ranges(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5, accuracy=1.0),),
                opponent_moves=(PlannerOpponentMove("Ember", 3, 4),),
                active_speed=30,
                opponent_speed=20,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertIn("estimated 3 hits", decision.rationale)
        self.assertIn("safe sequence established", decision.rationale)

    def test_semi_invulnerable_move_models_charge_before_strike(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=10,
                opponent_hp=10,
                moves=(PlannerMove(0, "Fly", 10, 10, turns_required=2),),
                opponent_moves=(PlannerOpponentMove("Tackle", 10, 10),),
                active_speed=30,
                opponent_speed=20,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 0)
        self.assertEqual(decision.classification, PlannerDecisionClass.SAFE)
        self.assertIn("Fly (charge) → Fly", decision.rationale)
        self.assertIn("2 turn(s)", decision.rationale)
        self.assertIn("projected incoming damage 0", decision.rationale)

    def test_semi_invulnerable_move_can_still_be_rejected_when_strike_is_exposed(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=10,
                opponent_hp=10,
                moves=(PlannerMove(0, "Dig", 10, 10, turns_required=2),),
                opponent_moves=(PlannerOpponentMove("Tackle", 10, 10),),
                active_speed=20,
                opponent_speed=30,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.safety, PlannerSafety.UNSAFE_OR_UNCERTAIN)
        self.assertIn("Two-turn moves", decision.rationale)

    def test_multi_turn_attack_is_rejected_when_exchange_can_faint_player(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Ember", 4, 5),),
                active_speed=20,
                opponent_speed=30,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.safety, PlannerSafety.UNSAFE_OR_UNCERTAIN)
        self.assertIn("8/20 HP", decision.rationale)
        self.assertIn("projected incoming damage 10", decision.rationale)
        self.assertTrue(decision.rejected_candidates)

    def test_multi_turn_damage_range_uses_minimum_for_hit_count(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=13,
                moves=(PlannerMove(0, "Pound", 4, 9),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertIn("estimated 4 hits", decision.rationale)

    def test_accuracy_can_make_a_multi_turn_sequence_unsafe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=12,
                moves=(PlannerMove(0, "Low Accuracy", 4, 5, accuracy=0.9),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertIn("accuracy", decision.rationale)

    def test_unknown_opponent_response_is_not_collapsed_into_generic_failure(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertFalse(decision.is_safe_to_execute)
        self.assertIn("opponent response is unknown", decision.rationale)
        self.assertIn(PlannerEvidenceKind.UNKNOWN, {item.kind for item in decision.evidence})

    def test_priority_and_speed_change_whether_final_hit_gets_a_response(self):
        first = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=8,
                moves=(PlannerMove(0, "Quick Attack", 4, 4, priority=1),),
                opponent_moves=(PlannerOpponentMove("Tackle", 4, 4, priority=0),),
                active_speed=1,
                opponent_speed=99,
            )
        )
        second = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=8,
                moves=(PlannerMove(0, "Pound", 4, 4),),
                opponent_moves=(PlannerOpponentMove("Tackle", 4, 4),),
                active_speed=1,
                opponent_speed=99,
            )
        )
        self.assertEqual(first.action, PlannerAction.UseMove)
        self.assertEqual(second.action, PlannerAction.UseMove)
        self.assertEqual(second.classification, PlannerDecisionClass.BEST_AVAILABLE)

    def test_multiple_known_opponent_moves_use_the_most_dangerous_one(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(
                    PlannerOpponentMove("Growl", 0, 0),
                    PlannerOpponentMove("Ember", 4, 6),
                ),
                active_speed=30,
                opponent_speed=20,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)

    def test_unsupported_mechanics_do_not_look_like_safe_damage(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=12,
                moves=(PlannerMove(0, "Counter", 4, 8, mechanics_supported=False),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)
        self.assertIn("unsupported", decision.rationale)

    def test_bide_is_an_executable_stateful_move(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=20,
                moves=(PlannerMove(0, "Bide", 0, 0, effect="BIDE"),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
                active_speed=20,
                opponent_speed=10,
            )
        )

        assert decision.action is PlannerAction.UseMove
        assert decision.target == 0
        assert decision.evaluation_method == "expectiminimax"

    def test_stale_selected_item_root_falls_back_to_a_live_move_root(self):
        potion = RecoveryItem(SimpleNamespace(name="Potion"), 1, 20)
        battle_context = context(
            active_hp=20,
            active_max_hp=20,
            moves=(PlannerMove(0, "Scratch", 3, 3),),
            items=(potion,),
            opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        )
        roots = (
            SearchRootEvaluation(
                "item", 0, "Potion", 100.0, 1.0, 0.0, 0.0,
                SearchResultCategory.TERMINAL_COMPLETE,
            ),
            SearchRootEvaluation(
                "move", 0, "Scratch", 10.0, 1.0, 0.0, 0.0,
                SearchResultCategory.TERMINAL_COMPLETE,
            ),
        )
        stale = SearchResult(
            0,
            "Potion",
            100.0,
            1.0,
            0.0,
            0.0,
            action_kind="item",
            category=SearchResultCategory.TERMINAL_COMPLETE,
            root_evaluations=roots,
        )

        with patch("modules.battle_planner.search_planner_context", return_value=stale):
            decision = BattlePlanner().plan(battle_context)

        assert decision.action is PlannerAction.UseMove
        assert decision.target == 0
        assert decision.stochastic_result.action_name == "Scratch"

    def test_incomplete_capture_root_translates_a_reserve_item_target(self):
        potion = RecoveryItem(SimpleNamespace(name="Potion"), 6, heal_amount=20)
        projected_reserve_item = SearchResult(
            1,
            "Potion",
            0.0,
            0.0,
            0.0,
            0.0,
            action_kind="item",
            category=SearchResultCategory.INCOMPLETE,
            objective="capture",
            target_party_index=1,
        )
        with patch(
            "modules.battle_planner.search_planner_context",
            return_value=projected_reserve_item,
        ):
            decision = BattlePlanner().plan(
                context(
                    active_name="Wurmple",
                    active_hp=18,
                    active_max_hp=18,
                    opponent_name="Silcoon",
                    opponent_hp=21,
                    opponent_max_hp=21,
                    moves=(PlannerMove(0, "Tackle", 2, 3),),
                    switches=(PlannerSwitch(1, "Samuel", 5, 20, None, moves=()),),
                    items=(
                        RecoveryItem(
                            SimpleNamespace(name="Antidote"),
                            4,
                            status_healed=frozenset({"poison"}),
                        ),
                        potion,
                        RecoveryItem(SimpleNamespace(name="Poké Ball"), 10, is_capture_ball=True),
                    ),
                    is_wild=True,
                    can_switch=True,
                    can_run=True,
                    escape_confidence=PlannerConfidence.HIGH,
                    capture_target=True,
                    capture_available=True,
                    battle_objective="capture",
                    campaign_owned=True,
                    opponent_response_facts_complete=False,
                )
            )

        assert decision.action is PlannerAction.UseItem
        assert decision.target == (potion.item, 1)
        assert decision.stochastic_result is not None

    def test_incomplete_capture_root_recovers_target_from_matching_root_evaluation(self):
        potion = RecoveryItem(SimpleNamespace(name="Potion"), 6, heal_amount=20)
        projected_reserve_item = SearchResult(
            1,
            "Potion",
            0.0,
            0.0,
            0.0,
            0.0,
            action_kind="item",
            category=SearchResultCategory.INCOMPLETE,
            objective="capture",
            root_evaluations=(
                SearchRootEvaluation(
                    "item",
                    1,
                    "Potion",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    SearchResultCategory.INCOMPLETE,
                    target_party_index=1,
                ),
            ),
        )
        with patch(
            "modules.battle_planner.search_planner_context",
            return_value=projected_reserve_item,
        ):
            decision = BattlePlanner().plan(
                context(
                    active_name="Wurmple",
                    active_hp=18,
                    active_max_hp=18,
                    opponent_name="Silcoon",
                    opponent_hp=21,
                    moves=(PlannerMove(0, "Tackle", 2, 3),),
                    switches=(PlannerSwitch(1, "Samuel", 5, 20, None, moves=()),),
                    items=(
                        RecoveryItem(
                            SimpleNamespace(name="Antidote"),
                            4,
                            status_healed=frozenset({"poison"}),
                        ),
                        potion,
                    ),
                    is_wild=True,
                    can_switch=True,
                    can_run=True,
                    escape_confidence=PlannerConfidence.HIGH,
                    capture_target=True,
                    capture_available=True,
                    battle_objective="capture",
                    campaign_owned=True,
                    opponent_response_facts_complete=False,
                )
            )

        assert decision.action is PlannerAction.UseItem
        assert decision.target == (potion.item, 1)

    def test_low_hp_prefers_a_clearly_safer_switch(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=2,
                active_max_hp=20,
                opponent_damage_max=8,
                moves=(PlannerMove(0, "Scratch", 3, 5),),
                can_switch=True,
                switches=(PlannerSwitch(1, "Torchic", 18, 20, 5, 10),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.SwitchPokemon)
        self.assertEqual(decision.target, 1)

    def test_safe_switch_requires_matchup_damage(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=2,
                active_max_hp=20,
                opponent_damage_max=8,
                moves=(),
                can_switch=True,
                switches=(PlannerSwitch(1, "Lotad", 18, 20, 5, 0),),
            )
        )
        self.assertNotEqual(decision.action, PlannerAction.SwitchPokemon)
        self.assertNotEqual(decision.target, 1)

    def test_safe_switch_prefers_the_matchup_capable_member_over_zero_damage_member(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=2,
                active_max_hp=20,
                opponent_damage_max=8,
                moves=(),
                can_switch=True,
                switches=(
                    PlannerSwitch(1, "Lotad", 18, 20, 5, 0),
                    PlannerSwitch(2, "Treecko", 15, 20, 5, 6),
                ),
            )
        )
        self.assertEqual(decision.action, PlannerAction.SwitchPokemon)
        self.assertEqual(decision.target, 2)

    def test_action_report_separates_safe_switch_from_unsafe_zero_damage_switch(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=2,
                active_max_hp=20,
                opponent_hp=12,
                opponent_damage_max=8,
                moves=(PlannerMove(0, "Astonish", 0, 0),),
                can_switch=True,
                switches=(
                    PlannerSwitch(1, "Lotad", 18, 20, 5, 0),
                    PlannerSwitch(2, "Treecko", 15, 20, 5, 6),
                ),
            )
        )
        candidates = {candidate.label: candidate for candidate in decision.candidate_evaluations}

        self.assertFalse(candidates["Astonish"].is_safe)
        self.assertIn("no effective damaging move", candidates["Astonish"].reason)
        self.assertFalse(candidates["switch to Lotad (slot 1)"].is_safe)
        self.assertIn("no effective damaging move", candidates["switch to Lotad (slot 1)"].reason)
        self.assertTrue(candidates["switch to Treecko (slot 2)"].is_safe)
        self.assertEqual(decision.best_safe_candidate.label, "switch to Treecko (slot 2)")

        diagnostic = format_planner_decision_diagnostic(decision, context(
            active_hp=2,
            active_max_hp=20,
            opponent_hp=12,
            opponent_damage_max=8,
            moves=(PlannerMove(0, "Astonish", 0, 0),),
            can_switch=True,
            switches=(
                PlannerSwitch(1, "Lotad", 18, 20, 5, 0),
                PlannerSwitch(2, "Treecko", 15, 20, 5, 6),
            ),
        ))
        self.assertIn("BATTLE_ACTION_EVALUATION", diagnostic)
        self.assertIn("switch to Lotad (slot 1)=unsafe/uncertain", diagnostic)
        self.assertIn("switch to Treecko (slot 2)=safe", diagnostic)

    def test_action_report_keeps_best_available_separate_from_best_safe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=5,
                active_max_hp=20,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Tackle", 4, 5),),
            )
        )

        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertIsNone(decision.best_safe_candidate)
        self.assertEqual(decision.best_available_candidate.label, "Pound")
        self.assertFalse(decision.best_available_candidate.is_safe)
        self.assertIn("No guaranteed-safe sequence", decision.best_available_candidate.reason)

    def test_low_hp_still_attacks_when_ko_is_guaranteed(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=4,
                active_max_hp=20,
                opponent_hp=8,
                moves=(PlannerMove(0, "Scratch", 8, 11),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)

    def test_wild_escape_is_used_when_other_actions_are_not_safe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=1,
                active_max_hp=20,
                moves=(),
                is_wild=True,
                can_run=True,
                escape_confidence=PlannerConfidence.HIGH,
            )
        )
        self.assertEqual(decision.action, PlannerAction.RunAway)

    def test_expectiminimax_can_use_wild_escape_for_zero_progress_conservation(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                active_max_hp=20,
                opponent_hp=20,
                moves=(PlannerMove(0, "Growl", 0, 0),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
                is_wild=True,
                can_run=True,
                escape_confidence=PlannerConfidence.HIGH,
                escape_probability=1.0,
            )
        )

        self.assertEqual(decision.action, PlannerAction.RunAway)
        self.assertEqual(decision.evaluation_method, "expectiminimax")

    def test_trainer_battle_never_runs(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=1,
                active_max_hp=20,
                moves=(),
                is_trainer=True,
                can_run=True,
                escape_confidence=PlannerConfidence.HIGH,
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)

    def test_no_safe_action_is_explicit(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=1,
                active_max_hp=20,
                moves=(),
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)
        self.assertEqual(decision.confidence, PlannerConfidence.INSUFFICIENT)

    def test_best_available_damaging_line_is_explicit_when_worst_case_is_unsafe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=5,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Ember", 4, 5),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertIn("No guaranteed-safe sequence established", decision.rationale)

    def test_setup_move_can_remain_in_a_shorter_damage_line(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=10,
                moves=(
                    PlannerMove(0, "Pound", 2, 2),
                    PlannerMove(1, "Tail Whip", 0, 0, effect="DEFENSE_DOWN_2"),
                ),
                opponent_moves=(PlannerOpponentMove("Ember", 2, 2),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        # Expectiminimax evaluates the setup root as well, then selects the
        # direct damaging line because it wins within the configured horizon.
        self.assertEqual(decision.target, 0)
        self.assertIn("expectiminimax", decision.rationale)
        self.assertIn("effective stat changes", decision.rationale)

    def test_setup_move_is_selected_when_direct_damage_cannot_finish_in_horizon(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=10,
                moves=(
                    PlannerMove(0, "Pound", 1, 1),
                    PlannerMove(1, "Tail Whip", 0, 0, effect="DEFENSE_DOWN_2"),
                ),
                opponent_moves=(PlannerOpponentMove("Ember", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 0)
        self.assertIn("expectiminimax", decision.rationale)
        self.assertIn("Tail Whip", decision.rationale)

    def test_stat_move_at_live_lower_bound_is_rejected_as_ineffective(self):
        battle_context = context(
            opponent_hp=10,
            opponent_stat_stages=(("defense", -6),),
            moves=(
                PlannerMove(0, "Pound", 2, 2),
                PlannerMove(1, "Leer", 0, 0, effect="DEFENSE_DOWN"),
            ),
            opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        )
        decision = BattlePlanner().plan(battle_context)
        self.assertEqual(decision.target, 0)
        leer = next(candidate for candidate in decision.candidate_evaluations if candidate.label == "Leer")
        self.assertFalse(leer.is_safe)
        self.assertIn("no battle effect", leer.reason)

    def test_stat_move_can_improve_from_the_opposite_bound(self):
        battle_context = context(
            opponent_hp=10,
            opponent_stat_stages=(("defense", 6),),
            moves=(
                PlannerMove(0, "Pound", 2, 2),
                PlannerMove(1, "Leer", 0, 0, effect="DEFENSE_DOWN"),
            ),
            opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        )
        decision = BattlePlanner().plan(battle_context)
        leer = next(candidate for candidate in decision.candidate_evaluations if candidate.label == "Leer")
        self.assertIn("can improve the line", leer.reason)

    def test_diagnostic_includes_live_stat_stages_and_move_effectiveness(self):
        battle_context = context(
            opponent_stat_stages=(("defense", -6),),
            moves=(PlannerMove(0, "Leer", 0, 0, effect="DEFENSE_DOWN"),),
        )
        decision = BattlePlanner().plan(battle_context)
        diagnostic = format_planner_decision_diagnostic(decision, battle_context)
        self.assertIn("opponent_stat_stages", diagnostic)
        self.assertIn("defense=-6", diagnostic)
        self.assertIn("selected_move_effectiveness", diagnostic)
        self.assertIn("effective=False", diagnostic)

    def test_full_emulator_facts_are_not_forwarded_as_planner_knowledge(self):
        knowledge = BattleKnowledge("b", KnowledgePolicy.FULL_EMULATOR, known_opponent_moves=(("id", "Thunder"),))
        safe = knowledge_for_planner(knowledge)
        self.assertEqual(safe.policy, KnowledgePolicy.HUMAN_KNOWLEDGE)
        self.assertEqual(safe.known_opponent_moves, ())

    def test_fallback_rationale_identifies_legacy_path_and_does_not_claim_ko(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        active = SimpleNamespace(current_hp_percentage=80, moves=())
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=active),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        util = SimpleNamespace(
            pokemon_has_enough_hp=lambda pokemon: True,
            get_strongest_move_against=lambda active, opponent: 3,
        )
        unsafe = SimpleNamespace(
            action=PlannerAction.AbortForSafety,
            target=None,
            confidence=PlannerConfidence.INSUFFICIENT,
            rationale="safe sequence cannot be established",
            is_safe_to_execute=False,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=unsafe):
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 3))
        self.assertIn("LEGACY FALLBACK", __import__("modules.context", fromlist=["context"]).context.message)
        self.assertIn("legacy selector chose", __import__("modules.context", fromlist=["context"]).context.message)

    def test_unknown_response_gets_best_available_attack(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=30,
                moves=(PlannerMove(0, "Pound", 8, 10), PlannerMove(1, "Scratch", 5, 12)),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 0)
        self.assertFalse(decision.is_safe_to_execute)


if __name__ == "__main__":
    unittest.main()
