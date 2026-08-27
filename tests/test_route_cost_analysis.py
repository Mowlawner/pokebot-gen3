from types import SimpleNamespace

from modules.goals import ReachLocation, ReachWarp
from modules.navigation import (
    IntermediateRouteAnalysis,
    NavigationMetrics,
    NavigationError,
    NavigationPlan,
    RouteCostAnalyzer,
)


def plan(cost, destination):
    return NavigationPlan((), destination, NavigationMetrics(0, 0, 0, 0, 0, 0, None, cost))


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
