import unittest
from types import SimpleNamespace

from modules.goals import Goal, ReachLocation
from modules.navigation import IntermediateRouteAnalysis, RouteAnalysis
from modules.nuzlocke.campaign_objectives import CampaignObjective
from modules.nuzlocke.campaign_planner import CampaignPlan, build_campaign_plan
from modules.nuzlocke.readiness_diagnostics import ReadinessDecision, ReadinessReason


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


if __name__ == "__main__":
    unittest.main()
