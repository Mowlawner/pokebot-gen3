import unittest
from dataclasses import replace
from types import SimpleNamespace

from modules.goals import Goal, ReachLocation
from modules.navigation import IntermediateRouteAnalysis, RouteAnalysis
from modules.nuzlocke.campaign_objectives import CampaignObjective
from modules.nuzlocke.campaign_planner import CampaignPlan, RecoveryUrgency, build_campaign_plan
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    ReadinessReason,
)


class CampaignPlannerTests(unittest.TestCase):
    def setUp(self):
        self.parent = CampaignObjective("reach_goal", "reach goal", (), SimpleNamespace())

    def test_recovery_becomes_an_explicit_first_stop(self):
        center = ReachLocation(("OLDale", (5, 6)))
        candidate = IntermediateRouteAnalysis(center, 48, 8, True, first_route="center-route")
        readiness = SimpleNamespace(
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
            route_analysis=RouteAnalysis(Goal(), "normal-route", 40, (candidate,)),
        )

        plan = build_campaign_plan(self.parent, readiness)

        self.assertIsInstance(plan, CampaignPlan)
        self.assertEqual(plan.parent_objective_id, "reach_goal")
        self.assertEqual(plan.recovery_stop.destination, ("OLDale", (5, 6)))
        self.assertEqual(plan.recovery_stop.route, "center-route")
        self.assertEqual(plan.recovery_stop.parent_objective_id, "reach_goal")

    def test_non_recovery_readiness_keeps_parent_plan_unmodified(self):
        readiness = SimpleNamespace(readiness_decision=ReadinessDecision.CONTINUE)
        plan = build_campaign_plan(self.parent, readiness)
        self.assertEqual(plan, CampaignPlan("reach_goal"))

    def test_unreachable_candidates_are_not_planned(self):
        center = ReachLocation(("OLDale", (5, 6)))
        candidate = IntermediateRouteAnalysis(center, None, None, False)
        readiness = SimpleNamespace(
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.CRITICAL_PARTY_HP,
            route_analysis=RouteAnalysis(Goal(), "normal-route", 40, (candidate,)),
        )
        self.assertEqual(build_campaign_plan(self.parent, readiness), CampaignPlan("reach_goal"))

    def test_critical_recovery_uses_source_route_when_parent_route_is_unavailable(self):
        center = ReachLocation(("OLDale", (5, 6)))
        source_route = SimpleNamespace(metrics=SimpleNamespace(total_route_cost=12))
        candidate = SimpleNamespace(
            destination=center,
            total_cost=None,
            detour=None,
            reachable=True,
            first_route=source_route,
            continuation_route=None,
        )
        readiness = SimpleNamespace(
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.CRITICAL_PARTY_HP,
            route_analysis=RouteAnalysis(Goal(), None, None, (candidate,)),
        )

        plan = build_campaign_plan(self.parent, readiness)

        self.assertIsNotNone(plan.recovery_stop)
        self.assertEqual(plan.recovery_stop.urgency, RecoveryUrgency.CRITICAL)
        self.assertIs(plan.recovery_stop.route, source_route)
        self.assertIsNone(plan.recovery_stop.continuation_route)

    def test_targetless_recovery_uses_observed_center_when_parent_route_is_absent(self):
        destination = ("LittlerootTown", (5, 6))
        observed_route = object()
        readiness = SimpleNamespace(
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.CRITICAL_PARTY_HP,
            route_analysis=None,
            recovery=SimpleNamespace(
                center_available=True,
                center_location=destination,
                safe_to_reach_center=True,
                route=observed_route,
            ),
        )

        plan = build_campaign_plan(self.parent, readiness)

        self.assertEqual(plan.recovery_stop.destination, destination)
        self.assertIsInstance(plan.recovery_stop.source, ReachLocation)
        self.assertIs(plan.recovery_stop.route, observed_route)
        self.assertEqual(plan.recovery_stop.urgency, RecoveryUrgency.CRITICAL)

    def test_targetless_opportunistic_recovery_uses_observed_center_when_parent_route_is_absent(self):
        destination = ("PetalburgCity", (20, 16))
        observed_route = object()
        readiness = SimpleNamespace(
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
            route_analysis=None,
            recovery=SimpleNamespace(
                center_available=True,
                center_location=destination,
                safe_to_reach_center=True,
                route=observed_route,
            ),
        )

        plan = build_campaign_plan(self.parent, readiness)

        self.assertIsNotNone(plan.recovery_stop)
        self.assertEqual(plan.recovery_stop.destination, destination)
        self.assertIs(plan.recovery_stop.route, observed_route)
        self.assertEqual(plan.recovery_stop.urgency, RecoveryUrgency.OPPORTUNISTIC)

    def test_poisoned_party_recovery_is_an_executable_critical_stop(self):
        center = ReachLocation(("Petalburg", (5, 6)))
        source_route = SimpleNamespace(metrics=SimpleNamespace(total_route_cost=24))
        candidate = SimpleNamespace(
            destination=center,
            total_cost=None,
            detour=None,
            reachable=True,
            first_route=source_route,
            continuation_route=None,
        )
        readiness = SimpleNamespace(
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.POISONED_PARTY,
            route_analysis=RouteAnalysis(Goal(), None, None, (candidate,)),
        )

        plan = build_campaign_plan(self.parent, readiness)

        self.assertIsNotNone(plan.recovery_stop)
        self.assertEqual(plan.recovery_stop.urgency, RecoveryUrgency.CRITICAL)
        self.assertIs(plan.recovery_stop.route, source_route)

    def test_planner_evaluates_raw_readiness_observations(self):
        center = ReachLocation(("OLDale", (5, 6)))
        candidate = IntermediateRouteAnalysis(
            center,
            48,
            8,
            True,
            first_route=SimpleNamespace(metrics=SimpleNamespace(total_route_cost=8)),
        )
        readiness = ProgressionReadinessDiagnostic(
            "reach_goal",
            "ready",
            None,
            Goal(),
            (1, 2),
            (3, 4),
            "OVERWORLD",
            "test",
            (SimpleNamespace(usable=True, hp_ratio=0.1, fainted=False),),
            True,
            None,
            None,
            (),
            party_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            route_analysis=RouteAnalysis(Goal(), None, None, (candidate,)),
        )

        plan = build_campaign_plan(self.parent, readiness)

        self.assertEqual(plan.readiness_decision, ReadinessDecision.RECOVER)
        self.assertEqual(plan.readiness_reason, ReadinessReason.CRITICAL_PARTY_HP)
        self.assertIs(plan.recovery_stop.route, candidate.first_route)


if __name__ == "__main__":
    unittest.main()
