from types import SimpleNamespace
from unittest.mock import patch

from modules.context import context
from modules.goals import GoalConstraints, NavigationGoal, ReachLocation, ReachWarp, TrainerMode
from modules.navigation import (
    IntermediateRouteAnalysis,
    NavigationMetrics,
    NavigationError,
    NavigationPlan,
    RouteCostAnalyzer,
    TrainerAvoidability,
    TrainerRouteRisk,
)
from modules.modes.campaign import CampaignProgressionMode


def plan(cost, destination):
    return NavigationPlan((), destination, NavigationMetrics(0, 0, 0, 0, 0, 0, None, cost))


def test_campaign_route_analysis_defers_while_tactical_search_is_active():
    mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
    mode._background_route_planner = object()

    with (
        patch("modules.modes.campaign.navigation_background_search_active", return_value=True),
        patch("modules.modes.campaign.diagnostic_print"),
    ):
        result = mode._analyze_route_async(None, None, None, None, ())

    assert result == (None, True, False)


def test_campaign_route_analysis_defers_before_tactical_loop_mount():
    mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
    mode._background_route_planner = object()

    with (
        patch("modules.modes.campaign.navigation_background_search_active", return_value=False),
        patch.object(context, "active_agent_control_loop", None, create=True),
        patch("modules.modes.campaign.diagnostic_print"),
    ):
        result = mode._analyze_route_async(None, None, None, None, (), tactical_goal=object())

    assert result == (None, False, False)


def test_campaign_route_analysis_runs_for_completed_recovery_without_tactical_owner():
    mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
    analysis = object()

    class Planner:
        def poll(self, key, operation, **_kwargs):
            return "complete", analysis, None

    mode._background_route_planner = Planner()
    mode._pending_route_analysis = None
    world = SimpleNamespace(
        running_shoes=False,
        player_elevation=None,
        surfing=False,
        revision_key=lambda: ((), ()),
        triggers=(),
    )

    with (
        patch("modules.modes.campaign.navigation_background_search_active", return_value=False),
        patch.object(context, "active_agent_control_loop", None, create=True),
        patch("modules.modes.campaign.diagnostic_print"),
    ):
        result = mode._analyze_route_async(
            world,
            object(),
            ((1, 2), (1, 1)),
            object(),
            (),
            tactical_goal=object(),
            allow_without_tactical_owner=True,
        )

    assert result == (analysis, False, False)


class FakeNavigator:
    def __init__(self):
        self.routes = {}

    def __call__(self, world, start, goal, graph=None, *, algorithm="dijkstra"):
        result = self.routes[(start, goal)]
        if isinstance(result, Exception):
            raise result
        return result


def test_route_analysis_composes_costs_and_detour_for_semantic_goal():
    start = (0, (0, 0))
    destination = (0, (5, 5))
    intermediate = (0, (2, 2))
    goal = ReachWarp(destination_map=(0, 1))
    fake = FakeNavigator()
    fake.routes.update(
        {
            (start, goal): plan(10, destination),
            (start, ReachLocation(intermediate)): plan(4, intermediate),
            (intermediate, goal): plan(9, destination),
        }
    )

    result = RouteCostAnalyzer(SimpleNamespace(), planner=fake).analyze(start, goal, (ReachLocation(intermediate),))

    assert result.normal_cost == 10
    candidate = result.candidates[0]
    assert candidate.reachable
    assert candidate.total_cost == 13
    assert candidate.detour == 3
    assert isinstance(result.goal, ReachWarp)


def test_route_analysis_keeps_multiple_and_unreachable_candidates_structured():
    start = (0, (0, 0))
    goal = ReachLocation((0, (5, 5)))
    first = ReachLocation((0, (1, 1)))
    second = ReachLocation((0, (9, 9)))
    fake = FakeNavigator()
    fake.routes.update(
        {
            (start, goal): plan(8, goal.location),
            (start, first): plan(3, first.location),
            (first.location, goal): plan(7, goal.location),
            (start, second): NavigationError("unreachable"),
        }
    )

    result = RouteCostAnalyzer(SimpleNamespace(), planner=fake).analyze(start, goal, (first, second))

    assert [(c.reachable, c.total_cost, c.detour) for c in result.candidates] == [
        (True, 10, 2),
        (False, None, None),
    ]


def test_route_analysis_reports_per_trainer_avoidance_cost():
    start = (0, (0, 0))
    destination = (0, (5, 5))
    trainer_id = "trainer:(0, 17):2"
    goal = NavigationGoal(
        ReachLocation(destination),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )
    trigger = SimpleNamespace(
        affordance_id=trainer_id,
        trigger_id=trainer_id,
        hazard_kind="trainer",
        hazard_locations=frozenset({(0, (1, 0))}),
        trainer_defeated=False,
    )

    def fake(world, origin, requested, graph=None, *, algorithm="dijkstra"):
        mode = getattr(getattr(requested, "constraints", None), "trainer_mode", None)
        avoided = getattr(getattr(requested, "constraints", None), "avoid_trainer_ids", frozenset())
        if origin == start and requested.target == goal.target:
            if mode is TrainerMode.IGNORE:
                return plan(10, destination)
            if avoided == {trainer_id}:
                return plan(25, destination)
            return plan(25, destination)
        raise NavigationError("unhandled route")

    result = RouteCostAnalyzer(
        SimpleNamespace(triggers=(trigger,)),
        planner=fake,
    ).analyze(start, goal)

    assert len(result.trainer_risks) == 1
    risk = result.trainer_risks[0]
    assert isinstance(risk, TrainerRouteRisk)
    assert risk.avoidability is TrainerAvoidability.COSTLY
    assert risk.engagement_cost == 10
    assert risk.avoidance_cost == 25
    assert risk.detour == 15


def test_route_analysis_ignores_a_defeated_trainer_trigger_even_without_hazard_geometry():
    trainer_id = "trainer:(0, 17):2"
    trigger = SimpleNamespace(
        affordance_id=trainer_id,
        trigger_id=trainer_id,
        hazard_kind=None,
        hazard_locations=frozenset(),
        trainer_defeated=True,
    )
    goal = NavigationGoal(
        ReachLocation((0, (5, 5))),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )

    result = RouteCostAnalyzer(
        SimpleNamespace(triggers=(trigger,)),
        planner=lambda *_args, **_kwargs: plan(10, goal.target.location),
    ).analyze((0, (0, 0)), goal)

    assert result.trainer_risks == ()


def test_route_analysis_ignores_unreachable_trainer_with_unknown_hazard_geometry():
    trainer_id = "trainer:(0, 8):3"
    trigger = SimpleNamespace(
        affordance_id=trainer_id,
        trigger_id=trainer_id,
        hazard_kind="trainer",
        locations=frozenset({(0, (20, 20))}),
        activation_locations=frozenset({(0, (20, 19))}),
        engagement_locations=frozenset(),
        hazard_locations=frozenset(),
        trainer_defeated=False,
    )
    goal = NavigationGoal(
        ReachLocation((0, (5, 5))),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )

    result = RouteCostAnalyzer(
        SimpleNamespace(triggers=(trigger,)),
        planner=lambda *_args, **_kwargs: plan(10, goal.target.location),
    ).analyze((0, (0, 0)), goal)

    assert result.trainer_risks == ()
