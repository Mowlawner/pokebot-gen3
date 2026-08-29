import unittest
from types import SimpleNamespace

from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    CampaignReadinessPolicy,
    ReadinessDecision,
    ReadinessReason,
    ReadinessObservationScheduler,
)


class ReadinessSchedulerTests(unittest.TestCase):
    def test_initial_refresh_then_bounded_reuse(self):
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda objective, goal: calls.append((objective, goal)) or "observation",
            lambda: ("OVERWORLD", (1, 2), (3, 4)),
            max_age_ticks=2,
        )
        self.assertEqual(scheduler.observe("objective", "goal"), "observation")
        self.assertEqual(scheduler.observe("objective", "goal"), "observation")
        self.assertEqual(scheduler.observe("objective", "goal"), "observation")
        self.assertEqual(len(calls), 1)
        self.assertEqual(scheduler.state.refresh_count, 1)
        self.assertEqual(scheduler.state.tick_count, 3)

    def test_map_or_context_change_refreshes_immediately(self):
        context = [("OVERWORLD", (1, 2), (3, 4))]
        calls = []
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or "observation", lambda: context[0], max_age_ticks=30
        )
        scheduler.observe("objective", "goal")
        context[0] = ("OVERWORLD", (1, 3), (0, 1))
        scheduler.observe("objective", "goal")
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

    def test_unavailable_observation_is_not_cached(self):
        calls = []
        unavailable = SimpleNamespace(overworld_availability=Availability.UNKNOWN)
        valid = SimpleNamespace(overworld_availability=Availability.KNOWN)
        scheduler = ReadinessObservationScheduler(
            lambda *_: calls.append(1) or (unavailable if len(calls) == 1 else valid),
            lambda: "same",
            max_age_ticks=30,
        )
        self.assertIs(scheduler.observe("objective", "goal"), unavailable)
        self.assertIs(scheduler.observe("objective", "goal"), valid)
        self.assertEqual(len(calls), 2)

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


if __name__ == "__main__":
    unittest.main()
