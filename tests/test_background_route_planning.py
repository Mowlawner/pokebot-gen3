import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE, PokemonCenter
from modules.modes.campaign import _SingleFlightBackgroundPlanner
from modules.navigation import NavigationWorld
from modules.nuzlocke.resource_runtime import (
    RouteRecoveryRequest,
    prepare_route_recovery,
)


def test_single_flight_does_not_submit_duplicate_work_and_reuses_completion():
    planner = _SingleFlightBackgroundPlanner()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def work():
        calls.append(1)
        started.set()
        release.wait(1)
        return "complete"

    try:
        assert planner.poll("same", work)[0] == "pending"
        assert started.wait(1)
        assert planner.poll("same", work)[0] == "pending"
        assert calls == [1]
        release.set()
        for _ in range(100):
            result = planner.poll("same", work)
            if result[0] == "complete":
                break
            time.sleep(0.001)
        assert result == ("complete", "complete", None)
        assert planner.poll("same", work) == ("complete", "complete", None)
        assert calls == [1]
    finally:
        planner.close()


def test_single_flight_discards_stale_result_before_submitting_new_key():
    planner = _SingleFlightBackgroundPlanner()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def work(key):
        calls.append(key)
        started.set()
        release.wait(1)
        return key

    try:
        assert planner.poll("old", lambda: work("old"))[0] == "pending"
        assert started.wait(1)
        assert planner.poll("new", lambda: work("new"))[0] == "pending"
        release.set()
        for _ in range(100):
            result = planner.poll("new", lambda: work("new"))
            if result[0] == "pending" and calls == ["old", "new"]:
                break
            time.sleep(0.001)
        assert calls == ["old", "new"]
        release.set()
        for _ in range(100):
            result = planner.poll("new", lambda: work("new"))
            if result[0] == "complete":
                break
            time.sleep(0.001)
        assert result == ("complete", "new", None)
    finally:
        planner.close()


def test_single_flight_preserves_completed_result_for_its_domain():
    """A completed analysis must not be consumed by a recovery poll, or vice versa."""

    for owner_domain, other_domain in (("recovery", "analysis"), ("analysis", "recovery")):
        planner = _SingleFlightBackgroundPlanner()
        started = threading.Event()
        release = threading.Event()
        calls = []
        owner_key = (owner_domain, "owner")
        other_key = (other_domain, "other")

        def owner_work():
            calls.append(owner_domain)
            started.set()
            release.wait(1)
            return owner_domain

        try:
            assert planner.poll(owner_key, owner_work)[0] == "pending"
            assert started.wait(1)
            release.set()

            owner_result = None
            for _ in range(100):
                # Polling the other domain must leave the completed future
                # available to its owner instead of discarding it.
                other_result = planner.poll(other_key, lambda: "other")
                assert other_result[0] in {"pending", "complete"}
                owner_result = planner.poll(owner_key, owner_work)
                if owner_result[0] == "complete":
                    break
                time.sleep(0.001)

            assert owner_result == ("complete", owner_domain, None)
            assert calls == [owner_domain]
        finally:
            planner.close()


def test_recovery_preparation_does_not_run_candidate_path_search():
    location = (MapRSE.ROUTE101, (3, 4))
    with (
        patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(rom=SimpleNamespace(is_rse=False), stutter_trace=None),
        ),
        patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location),
        patch("modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown),
        patch("modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()),
        patch("modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()),
        patch("modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=SimpleNamespace()),
        patch("modules.nuzlocke.resource_runtime.plan_with_world_navigation") as plan,
    ):
        prepared = prepare_route_recovery(candidate_limit=1)

    assert isinstance(prepared, RouteRecoveryRequest)
    plan.assert_not_called()


def test_background_planner_enforces_a_configurable_wall_clock_budget():
    planner = _SingleFlightBackgroundPlanner()

    def never_finishes(budget):
        while True:
            budget.check()

    try:
        assert planner.poll("budget", never_finishes, timeout_seconds=0.01)[0] == "pending"
        result = None
        for _ in range(100):
            result = planner.poll("budget", never_finishes, timeout_seconds=0.01)
            if result[0] == "error":
                break
            time.sleep(0.001)
        assert result[0] == "error"
        assert isinstance(result[2], TimeoutError)
    finally:
        planner.close()


def test_critical_background_work_supersedes_speculative_work():
    planner = _SingleFlightBackgroundPlanner()
    started = threading.Event()

    def speculative(budget):
        started.set()
        while True:
            budget.check()

    try:
        assert planner.poll("speculative", speculative, priority=2, timeout_seconds=1)[0] == "pending"
        assert started.wait(1)
        assert planner.poll("critical", lambda: "safe", priority=0)[0] == "pending"
        result = None
        for _ in range(100):
            result = planner.poll("critical", lambda: "safe", priority=0)
            if result[0] == "complete":
                break
            time.sleep(0.001)
        assert result == ("complete", "safe", None)
    finally:
        planner.close()


def test_recovery_request_key_does_not_scan_static_tiles():
    class ExplodingTiles(dict):
        def items(self):
            raise AssertionError("request key scanned static tiles")

    world = NavigationWorld(
        tiles=ExplodingTiles(),
        dynamic_blocked_revision=((1, 2),),
        transition_revision=(),
    )
    request = RouteRecoveryRequest(
        location=(MapRSE.ROUTE101, (3, 4)),
        world=world,
        graph=object(),
        ordered_sources=(),
        center_location=None,
        candidate_limit=1,
    )
    assert request.key[0] == (MapRSE.ROUTE101, (3, 4))
