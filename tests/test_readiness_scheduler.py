import unittest
from types import SimpleNamespace

from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    CampaignReadinessPolicy,
    ReadinessDecision,
    ReadinessReason,
    ReadinessObservationScheduler,
)
from modules.nuzlocke.resource_policy import RouteRecovery


class ReadinessSchedulerTests(unittest.TestCase):
    def test_initial_refresh_then_indefinite_semantic_reuse(self):
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda objective, goal: calls.append((objective, goal)) or "observation",
            lambda: ("OVERWORLD", (1, 2), (3, 4)),
            max_age_ticks=2,
        )
        self.assertEqual(scheduler.observe("objective", "goal"), "observation")
        self.assertFalse(scheduler.last_observation_was_cache_hit)
        self.assertEqual(scheduler.observe("objective", "goal"), "observation")
        self.assertTrue(scheduler.last_observation_was_cache_hit)
        for _ in range(10):
            self.assertEqual(scheduler.observe("objective", "goal"), "observation")
        self.assertEqual(len(calls), 1)
        self.assertEqual(scheduler.state.refresh_count, 1)
        self.assertEqual(scheduler.state.tick_count, 12)
        self.assertEqual(scheduler.state.status, "fresh")

    def test_coordinate_only_change_does_not_refresh(self):
        context = [("OVERWORLD", (1, 2), (3, 4))]
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or "observation", lambda: context[0][:2], max_age_ticks=30
        )
        scheduler.observe("objective", "goal")
        context[0] = ("OVERWORLD", (1, 2), (0, 1))
        scheduler.observe("objective", "goal")
        self.assertEqual(len(calls), 1)

    def test_map_or_game_state_change_refreshes_immediately(self):
        context = [("OVERWORLD", (1, 2))]
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or "observation", lambda: context[0], max_age_ticks=30
        )
        scheduler.observe("objective", "goal")
        context[0] = ("OVERWORLD", (1, 3))
        scheduler.observe("objective", "goal")
        context[0] = ("CHANGE_MAP", (1, 3))
        scheduler.observe("objective", "goal")
        self.assertEqual(len(calls), 3)

    def test_battle_end_invalidation_refreshes_even_when_boundary_context_is_unchanged(self):
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(len(calls)) or calls[-1],
            lambda: ("OVERWORLD", (1, 2)),
            max_age_ticks=30,
        )
        self.assertEqual(scheduler.observe("objective", "goal"), 0)
        scheduler.invalidate("battle_ended")
        self.assertEqual(scheduler.observe("objective", "goal"), 1)
        self.assertEqual(len(calls), 2)

    def test_invalidation_refreshes_and_does_not_reuse_stale_data(self):
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(len(calls)) or calls[-1], lambda: "same", max_age_ticks=30
        )
        self.assertEqual(scheduler.observe("objective", "goal"), 0)
        scheduler.invalidate("battle_end")
        self.assertEqual(scheduler.observe("objective", "goal"), 1)
        self.assertEqual(scheduler.state.invalidation_reason, None)
        self.assertEqual(scheduler.state.refresh_count, 2)

    def test_unavailable_observation_is_cached_until_explicit_invalidation(self):
        calls = []
        unavailable = SimpleNamespace(overworld_availability=Availability.UNKNOWN)
        valid = SimpleNamespace(overworld_availability=Availability.KNOWN)
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or (unavailable if len(calls) == 1 else valid),
            lambda: "same",
            max_age_ticks=2,
        )
        self.assertIs(scheduler.observe("objective", "goal"), unavailable)
        self.assertFalse(scheduler.last_observation_was_cache_hit)
        self.assertIs(scheduler.observe("objective", "goal"), unavailable)
        self.assertTrue(scheduler.last_observation_was_cache_hit)
        for _ in range(10):
            self.assertIs(scheduler.observe("objective", "goal"), unavailable)
        scheduler.invalidate("observation_boundary")
        self.assertIs(scheduler.observe("objective", "goal"), valid)
        self.assertEqual(len(calls), 2)
        self.assertEqual(scheduler.state.status, "fresh")

    def test_unknown_overworld_blocks_readiness(self):
        readiness = SimpleNamespace(
            overworld_availability=Availability.UNKNOWN,
            party_availability=Availability.KNOWN,
        )
        result = CampaignReadinessPolicy().evaluate(readiness)
        self.assertEqual(result.decision, ReadinessDecision.UNKNOWN)
        self.assertEqual(result.reason, ReadinessReason.OVERWORLD_UNAVAILABLE)

    def test_unknown_resources_block_readiness(self):
        readiness = SimpleNamespace(
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.UNKNOWN,
            party_availability=Availability.KNOWN,
            game_state="OVERWORLD",
        )
        result = CampaignReadinessPolicy().evaluate(readiness)
        self.assertEqual(result.decision, ReadinessDecision.UNKNOWN)
        self.assertEqual(result.reason, ReadinessReason.RESOURCE_INFORMATION_UNKNOWN)

    def test_pending_route_observation_is_polled_until_completion(self):
        calls = []
        pending = SimpleNamespace(route_analysis_pending=True)
        complete = SimpleNamespace(route_analysis_pending=False)
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or (pending if len(calls) == 1 else complete),
            lambda: "same",
        )

        self.assertIs(scheduler.observe("objective", "goal"), pending)
        self.assertIs(scheduler.observe("objective", "goal"), complete)
        self.assertEqual(len(calls), 2)
        self.assertEqual(scheduler.state.status, "fresh")

    def test_timed_out_route_is_retried_after_cooldown(self):
        calls = []
        timed_out = SimpleNamespace(
            recovery=RouteRecovery(calculation_timed_out=True),
            route_analysis_pending=False,
        )
        complete = SimpleNamespace(
            recovery=RouteRecovery(),
            route_analysis_pending=False,
        )
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or (timed_out if len(calls) == 1 else complete),
            lambda: "same",
        )

        self.assertIs(scheduler.observe("objective", "goal"), timed_out)
        for _ in range(59):
            self.assertIs(scheduler.observe("objective", "goal"), timed_out)
        self.assertIs(scheduler.observe("objective", "goal"), complete)
        self.assertEqual(len(calls), 2)

    def test_timed_out_route_analysis_is_retried_after_cooldown(self):
        calls = []
        timed_out = SimpleNamespace(
            recovery=RouteRecovery(),
            route_analysis_timed_out=True,
            route_analysis_pending=False,
        )
        complete = SimpleNamespace(
            recovery=RouteRecovery(),
            route_analysis_timed_out=False,
            route_analysis_pending=False,
        )
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or (timed_out if len(calls) == 1 else complete),
            lambda: "same",
        )

        self.assertIs(scheduler.observe("objective", "goal"), timed_out)
        for _ in range(59):
            self.assertIs(scheduler.observe("objective", "goal"), timed_out)
        self.assertIs(scheduler.observe("objective", "goal"), complete)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
