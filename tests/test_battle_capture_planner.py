import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from modules.battle_capture_planner import (
    CaptureConfig,
    choose_capture_plan,
    evaluate_capture_plans,
)
from modules.battle_planner import (
    BattlePlanner,
    PlannerAction,
    PlannerConfidence,
    PlannerContext,
    PlannerDecisionClass,
    PlannerMove,
    PlannerOpponentMove,
    PlannerSafety,
    PlannerSwitch,
    RecoveryItem,
)


def catch_probability(hp: int, max_hp: int, rate: int = 255, ball: float = 1.0) -> float:
    """The Gen III catch formula, so these tests use realistic odds."""

    a = (3 * max_hp - 2 * hp) * rate * ball / (3 * max_hp)
    if a >= 255:
        return 1.0
    b = 1048560 / math.sqrt(math.sqrt(16711680 / a))
    return (b / 65536) ** 4


def ball(quantity: int):
    return RecoveryItem(SimpleNamespace(name="Poke Ball"), quantity, 0, True)


def capture_context(**changes) -> PlannerContext:
    max_hp = changes.pop("target_max_hp", 19)
    values = dict(
        active_name="Lead",
        active_hp=40,
        active_max_hp=40,
        opponent_name="Wurmple",
        opponent_hp=19,
        opponent_max_hp=max_hp,
        moves=(PlannerMove(0, "Tackle", 3, 4),),
        items=(ball(5),),
        opponent_moves=(PlannerOpponentMove("Tackle", 3, 5, accuracy=0.95, critical_damage_max=10),),
        active_speed=20,
        opponent_speed=10,
        is_wild=True,
        capture_target=True,
        capture_available=True,
        capture_ball_count=5,
        battle_objective="capture",
        campaign_owned=True,
        opponent_response_facts_complete=True,
        capture_probability_for_state=lambda state, item: catch_probability(state.opponent_hp, max_hp),
    )
    values.update(changes)
    return PlannerContext(**values)


class CapturePlanChoiceTests(unittest.TestCase):
    def test_a_sturdy_lead_weakens_the_target_before_throwing(self):
        choice = choose_capture_plan(capture_context())

        self.assertIsNotNone(choice)
        self.assertGreater(choice.plan.weaken_turns, 0)
        self.assertEqual(choice.plan.move_name, "Tackle")
        self.assertGreater(choice.plan.capture_probability, 0.95)
        self.assertEqual(choice.plan.target_ko_probability, 0.0)

    def test_weakening_beats_throwing_at_full_hp(self):
        plans, _ = evaluate_capture_plans(capture_context())
        throw_now = next(plan for plan in plans if plan.weaken_turns == 0)
        best = max(plans, key=lambda plan: plan.capture_probability)

        self.assertGreater(best.capture_probability, throw_now.capture_probability + 0.1)

    def test_a_target_already_low_is_thrown_at_not_attacked(self):
        choice = choose_capture_plan(capture_context(opponent_hp=3))

        self.assertEqual(choice.plan.weaken_turns, 0)
        self.assertEqual(choice.plan.first_action, "throw")

    def test_a_move_that_would_knock_out_the_target_is_not_used(self):
        # 8-12 damage against 10 HP knocks the target out most of the time.
        choice = choose_capture_plan(
            capture_context(opponent_hp=10, moves=(PlannerMove(0, "Slam", 8, 12),))
        )

        self.assertEqual(choice.plan.weaken_turns, 0)

    def test_false_swipe_is_used_because_it_cannot_knock_out(self):
        choice = choose_capture_plan(
            capture_context(
                opponent_hp=10,
                moves=(PlannerMove(0, "False Swipe", 8, 12, effect="FALSE_SWIPE"),),
            )
        )

        self.assertGreater(choice.plan.weaken_turns, 0)
        self.assertEqual(choice.plan.target_ko_probability, 0.0)

    def test_fewer_balls_make_weakening_more_valuable(self):
        many = choose_capture_plan(capture_context(items=(ball(10),), capture_ball_count=10)).plan
        one_ball = choose_capture_plan(capture_context(items=(ball(1),), capture_ball_count=1)).plan

        self.assertGreaterEqual(one_ball.weaken_turns, many.weaken_turns)
        self.assertGreater(one_ball.weaken_turns, 0)

    def test_when_no_plan_is_safe_the_least_risky_one_is_chosen_and_marked(self):
        # A 5 HP lead against 6-7 damage cannot be kept safe by any plan.
        context = capture_context(
            active_hp=5,
            active_max_hp=5,
            opponent_moves=(PlannerOpponentMove("Tackle", 6, 7),),
        )
        plans, _ = evaluate_capture_plans(context)

        choice = choose_capture_plan(context)

        self.assertTrue(choice.risk_accepted)
        self.assertEqual(
            choice.plan.active_faint_probability,
            min(plan.active_faint_probability for plan in plans),
        )

    def test_a_fragile_lead_with_no_one_to_switch_to_takes_the_least_risky_plan(self):
        # The live Woods case: a 13 HP lead against a Tackle that can crit for 14.
        context = capture_context(
            active_hp=13,
            active_max_hp=13,
            opponent_moves=(PlannerOpponentMove("Tackle", 5, 7, accuracy=0.95, critical_damage_max=14),),
        )
        plans, _ = evaluate_capture_plans(context)
        choice = choose_capture_plan(context)

        self.assertTrue(all(plan.active_faint_probability > 0.03 for plan in plans))
        self.assertTrue(choice.risk_accepted)
        # Throwing now exposes the lead to the fewest opponent actions.
        self.assertEqual(choice.plan.weaken_turns, 0)

    def test_a_safe_choice_is_not_marked_as_risk_accepted(self):
        self.assertFalse(choose_capture_plan(capture_context()).risk_accepted)

    def test_probabilities_are_consistent(self):
        plans, _ = evaluate_capture_plans(capture_context(active_hp=14, active_max_hp=14))

        for plan in plans:
            with self.subTest(plan=plan.summary()):
                total = plan.capture_probability + plan.active_faint_probability + plan.target_ko_probability
                self.assertLessEqual(total, 1.0 + 1e-9)
                for value in (plan.capture_probability, plan.active_faint_probability, plan.target_ko_probability):
                    self.assertGreaterEqual(value, 0.0)

    def test_the_tradeoff_between_ko_risk_and_running_out_of_balls_is_weighed(self):
        # With one ball, taking some KO risk to land the throw at low HP wins.
        plans, _ = evaluate_capture_plans(capture_context(items=(ball(1),), capture_ball_count=1))
        throw_now = next(plan for plan in plans if plan.weaken_turns == 0)
        risky = max(plans, key=lambda plan: plan.capture_probability)

        self.assertGreater(risky.capture_probability, throw_now.capture_probability)
        self.assertGreaterEqual(risky.target_ko_probability, 0.0)


class CaptureCriticalHitTests(unittest.TestCase):
    def test_our_critical_hit_is_a_knockout_risk_even_when_normal_damage_is_not(self):
        # Tackle 3-4 cannot knock out a 7 HP target, but a critical hit (double
        # the maximum, 8) can: exactly the 1-in-16 critical chance.
        plans, _ = evaluate_capture_plans(
            capture_context(opponent_hp=7, moves=(PlannerMove(0, "Tackle", 3, 4),))
        )
        one_move = next(plan for plan in plans if plan.weaken_turns == 1)

        self.assertAlmostEqual(one_move.target_ko_probability, 1 / 16, places=4)

    def test_an_opponent_critical_that_exceeds_the_leads_hp_raises_the_faint_risk(self):
        def throw_now_faint(critical_damage):
            context = capture_context(
                active_hp=12,
                active_max_hp=12,
                opponent_moves=(
                    PlannerOpponentMove(
                        "Tackle", 5, 7, accuracy=1.0, critical_damage_max=critical_damage
                    ),
                ),
            )
            plans, _ = evaluate_capture_plans(context)
            return next(plan for plan in plans if plan.weaken_turns == 0).active_faint_probability

        # 14 can one-shot a 12 HP lead; 11 cannot.
        self.assertGreater(throw_now_faint(14), throw_now_faint(11))

    def test_a_replan_after_a_critical_uses_the_live_target_hp(self):
        # The plan is recomputed from the observed state each turn: after a
        # critical drops the target to 3 HP the planner throws instead of
        # attacking, because another hit would knock it out.
        before = choose_capture_plan(capture_context(opponent_hp=19))
        after_crit = choose_capture_plan(capture_context(opponent_hp=3))

        self.assertGreater(before.plan.weaken_turns, 0)
        self.assertEqual(after_crit.plan.weaken_turns, 0)


class CapturePreconditionTests(unittest.TestCase):
    def assertDefers(self, **changes):
        self.assertIsNone(choose_capture_plan(capture_context(**changes)))

    def test_not_a_capture_objective(self):
        self.assertDefers(battle_objective="defeat")

    def test_trainer_battle(self):
        self.assertDefers(is_trainer=True)

    def test_not_a_legal_capture_target(self):
        self.assertDefers(capture_target=False)

    def test_no_usable_ball(self):
        self.assertDefers(capture_available=False)
        self.assertDefers(items=(ball(0),))

    def test_no_capture_probability_model(self):
        self.assertDefers(capture_probability_for_state=None)

    def test_incomplete_opponent_facts_cannot_support_a_safety_claim(self):
        self.assertDefers(opponent_response_facts_complete=False)

    def test_unsupported_opponent_moves(self):
        self.assertDefers(unsupported_opponent_moves=("Counter",))

    def test_unknown_opponent_damage(self):
        self.assertDefers(opponent_moves=(), opponent_damage_max=None)

    def test_a_bare_damage_ceiling_is_enough_to_model_the_response(self):
        self.assertIsNotNone(
            choose_capture_plan(capture_context(opponent_moves=(), opponent_damage_max=4))
        )

    def test_status_and_unsupported_moves_are_never_used_to_weaken(self):
        context = capture_context(
            opponent_hp=19,
            moves=(
                PlannerMove(0, "Growl", 0, 0, effect="ATTACK_DOWN"),
                PlannerMove(1, "Counter", 5, 6, mechanics_supported=False),
                PlannerMove(2, "Explosion", 100, 120, effect="EXPLOSION"),
                PlannerMove(3, "Bide", 0, 0, effect="BIDE"),
            ),
        )
        plans, _ = evaluate_capture_plans(context)

        self.assertEqual([plan.weaken_turns for plan in plans], [0])

    def test_an_exhausted_move_is_not_used(self):
        plans, _ = evaluate_capture_plans(capture_context(moves=(PlannerMove(0, "Tackle", 3, 4, pp=0),)))

        self.assertEqual([plan.weaken_turns for plan in plans], [0])

    def test_planning_is_fast_even_with_many_balls_and_moves(self):
        import time

        context = capture_context(
            items=(ball(99),),
            capture_ball_count=99,
            moves=tuple(PlannerMove(index, f"Move{index}", 2 + index, 5 + index) for index in range(4)),
        )
        started = time.perf_counter()
        choose_capture_plan(context)

        self.assertLess(time.perf_counter() - started, 1.0)


class CapturePlannerIntegrationTests(unittest.TestCase):
    def test_the_planner_uses_the_capture_solver_for_a_wild_capture(self):
        decision = BattlePlanner().plan(capture_context())

        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertIs(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 0)
        self.assertIs(decision.safety, PlannerSafety.SAFE)
        self.assertIs(decision.classification, PlannerDecisionClass.SAFE)
        self.assertTrue(decision.is_safe_to_execute)
        self.assertIn("capture=", decision.rationale)

    def test_a_ball_decision_targets_the_ball_and_the_active_slot(self):
        context = capture_context(opponent_hp=3, active_party_index=2)
        decision = BattlePlanner().plan(context)

        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertIs(decision.action, PlannerAction.UseItem)
        item, party_index = decision.target
        self.assertEqual(item.name, "Poke Ball")
        self.assertEqual(party_index, 2)

    def test_the_generic_search_still_owns_trainer_and_defeat_battles(self):
        for changes in (dict(is_trainer=True), dict(battle_objective="defeat")):
            with self.subTest(changes=changes):
                decision = BattlePlanner().plan(capture_context(**changes))
                self.assertNotEqual(decision.evaluation_method, "capture_planner")

    def test_a_risk_accepted_capture_is_not_labeled_safe_but_the_campaign_may_proceed(self):
        context = capture_context(
            active_hp=5,
            active_max_hp=5,
            opponent_moves=(PlannerOpponentMove("Tackle", 6, 7),),
        )
        decision = BattlePlanner().plan(context)

        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertIs(decision.safety, PlannerSafety.UNSAFE_OR_UNCERTAIN)
        self.assertIs(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertFalse(decision.is_safe_to_execute)
        self.assertIn("RISK ACCEPTED", decision.rationale)
        # The campaign gate proceeds only for a result that declares itself an
        # autonomous root; it must not claim the line is safe.
        self.assertTrue(decision.stochastic_result.is_autonomous_root)
        self.assertFalse(decision.stochastic_result.is_safe)

    def test_an_unmodelable_capture_still_falls_back_to_the_generic_search(self):
        decision = BattlePlanner().plan(capture_context(opponent_response_facts_complete=False))

        self.assertNotEqual(decision.evaluation_method, "capture_planner")


def switch_candidate(party_index=1, name="Sturdy", hp=60, moves=(PlannerMove(0, "Tackle", 3, 4),)):
    return PlannerSwitch(party_index, name, hp, hp, 5, moves=moves, speed=15)


def fragile_context(**changes):
    values = dict(
        active_hp=13,
        active_max_hp=13,
        opponent_moves=(PlannerOpponentMove("Tackle", 5, 7, accuracy=0.95, critical_damage_max=14),),
        can_switch=True,
        switches=(switch_candidate(),),
    )
    values.update(changes)
    return capture_context(**values)


class CaptureSwitchTests(unittest.TestCase):
    def test_a_fragile_lead_switches_to_a_member_who_can_take_two_hits(self):
        choice = choose_capture_plan(fragile_context())

        self.assertIsNotNone(choice.plan.switch_party_index)
        self.assertEqual(choice.plan.switch_party_index, 1)
        self.assertFalse(choice.risk_accepted)
        self.assertLessEqual(choice.plan.active_faint_probability, 0.03)

    def test_a_merely_sturdier_member_is_chosen_but_the_risk_is_still_flagged(self):
        # 40 HP is far safer than the 13 HP lead (about 4% against about 36%
        # chance of fainting) but not within the 3% bound, so the switch is the
        # least risky plan and is labeled as risk accepted rather than safe.
        choice = choose_capture_plan(fragile_context(switches=(switch_candidate(hp=40),)))
        lead_plans, _ = evaluate_capture_plans(fragile_context())

        self.assertEqual(choice.plan.switch_party_index, 1)
        self.assertTrue(choice.risk_accepted)
        self.assertLess(
            choice.plan.active_faint_probability,
            min(plan.active_faint_probability for plan in lead_plans) / 4,
        )

    def test_the_switch_becomes_a_planner_switch_decision(self):
        decision = BattlePlanner().plan(fragile_context())

        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertIs(decision.action, PlannerAction.SwitchPokemon)
        self.assertEqual(decision.target, 1)
        self.assertIs(decision.safety, PlannerSafety.SAFE)

    def test_a_member_who_cannot_take_two_worst_case_hits_is_not_eligible(self):
        # The worst single attack is a 14 damage critical, so two need > 28 HP.
        marginal = switch_candidate(hp=27)
        choice = choose_capture_plan(fragile_context(switches=(marginal,)))

        self.assertIsNone(choice.plan.switch_party_index)
        self.assertTrue(choice.risk_accepted)

    def test_a_member_exactly_at_the_threshold_is_eligible(self):
        choice = choose_capture_plan(fragile_context(switches=(switch_candidate(hp=29),)))

        self.assertEqual(choice.plan.switch_party_index, 1)

    def test_no_switch_when_switching_is_not_legal(self):
        choice = choose_capture_plan(fragile_context(can_switch=False))

        self.assertIsNone(choice.plan.switch_party_index)

    def test_no_second_switch_in_the_same_stretch_of_battle(self):
        choice = choose_capture_plan(fragile_context(switch_streak=1))

        self.assertIsNone(choice.plan.switch_party_index)

    def test_a_member_just_switched_out_is_not_switched_back_in(self):
        choice = choose_capture_plan(fragile_context(switch_history=(1,)))

        self.assertIsNone(choice.plan.switch_party_index)

    def test_a_healthy_lead_never_considers_switching(self):
        choice = choose_capture_plan(capture_context(can_switch=True, switches=(switch_candidate(),)))

        self.assertIsNone(choice.plan.switch_party_index)
        self.assertFalse(choice.risk_accepted)

    def test_a_switch_costs_an_opponent_action(self):
        context = fragile_context()
        lead_plans, balls = evaluate_capture_plans(context)
        from modules.battle_capture_planner import CaptureConfig as Config, evaluate_switch_plans

        switch_plans = evaluate_switch_plans(context, Config(), balls)
        throw_after_switch = next(
            plan for plan in switch_plans if plan.weaken_turns == 0 and plan.switch_party_index == 1
        )
        # The same sturdy member in the lead faces one fewer opponent action.
        hp = switch_candidate().hp
        sturdy_lead = capture_context(active_hp=hp, active_max_hp=hp, opponent_moves=context.opponent_moves)
        direct, _ = evaluate_capture_plans(sturdy_lead)
        throw_now = next(plan for plan in direct if plan.weaken_turns == 0)

        self.assertGreaterEqual(
            throw_after_switch.active_faint_probability, throw_now.active_faint_probability
        )


if __name__ == "__main__":
    unittest.main()


class CaptureFleeTests(unittest.TestCase):
    """Abandon the encounter only when neither the lead nor a safe switch can continue."""

    def hopeless(self, **changes):
        values = dict(
            active_hp=5,
            active_max_hp=5,
            opponent_moves=(PlannerOpponentMove("Tackle", 6, 7),),
            can_run=True,
            escape_confidence=PlannerConfidence.HIGH,
        )
        values.update(changes)
        return capture_context(**values)

    def test_a_hopeless_capture_flees_when_escape_is_legal(self):
        decision = BattlePlanner().plan(self.hopeless())

        self.assertIs(decision.action, PlannerAction.RunAway)
        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertTrue(decision.is_safe_to_execute)
        self.assertIn("abandoned", decision.rationale)

    def test_it_still_takes_the_least_risky_plan_when_escape_is_not_legal(self):
        decision = BattlePlanner().plan(self.hopeless(can_run=False))

        self.assertIsNot(decision.action, PlannerAction.RunAway)
        self.assertIn("RISK ACCEPTED", decision.rationale)

    def test_it_does_not_flee_on_an_unknown_escape_chance(self):
        decision = BattlePlanner().plan(self.hopeless(escape_confidence=PlannerConfidence.INSUFFICIENT))

        self.assertIsNot(decision.action, PlannerAction.RunAway)

    def test_a_safe_switch_is_preferred_to_fleeing(self):
        decision = BattlePlanner().plan(
            fragile_context(can_run=True, escape_confidence=PlannerConfidence.HIGH)
        )

        self.assertIs(decision.action, PlannerAction.SwitchPokemon)

    def test_a_healthy_lead_captures_rather_than_flees(self):
        decision = BattlePlanner().plan(
            capture_context(can_run=True, escape_confidence=PlannerConfidence.HIGH)
        )

        self.assertIsNot(decision.action, PlannerAction.RunAway)
