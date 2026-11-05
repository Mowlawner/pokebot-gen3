from __future__ import annotations

import unittest

from modules.nuzlocke.gate_discovery import GateCandidate, SourceLocation
from modules.nuzlocke.gate_observation import resolve_candidate_observation, resolve_guard_state
from modules.nuzlocke.snapshots import (
    CampaignObservationLifecycle,
    CampaignObservationSnapshot,
    NamedFlag,
    NamedVariable,
)


def _candidate(expression: str, symbol: str, value: str | None = None) -> GateCandidate:
    return GateCandidate(
        candidate_id="emerald:script_event:observation",
        game="emerald",
        source_revision="fixture-1",
        kind="script_event",
        source=SourceLocation("fixture/scripts.inc", 1, "Fixture"),
        guard_expression=expression,
        guard_symbol=symbol,
        guard_value=value,
    )


def _snapshot(*, flags=(), variables=(), available=True, lifecycle=CampaignObservationLifecycle.ACTIVE):
    return CampaignObservationSnapshot(
        flags=tuple(NamedFlag(name, value) for name, value in flags),
        variables=tuple(NamedVariable(name, value) for name, value in variables),
        available=available,
        lifecycle=lifecycle,
    )


class GateObservationTests(unittest.TestCase):
    def test_resolves_flags_and_variable_comparisons(self):
        snapshot = _snapshot(flags=(("FLAG_OPEN", True),), variables=(("VAR_STATE", 2),))
        self.assertEqual(resolve_guard_state(_candidate("goto_if_set FLAG_OPEN, Done", "FLAG_OPEN"), snapshot), "satisfied")
        self.assertEqual(resolve_guard_state(_candidate("FLAG_OPEN is unset", "FLAG_OPEN"), snapshot), "unsatisfied")
        self.assertEqual(
            resolve_guard_state(_candidate("map_script_2 VAR_STATE, 2, Done", "VAR_STATE", "2"), snapshot),
            "satisfied",
        )

    def test_resolves_source_flag_symbol_against_runtime_flag_name(self):
        candidate = _candidate(
            "call_if_unset FLAG_ADVENTURE_STARTED, OldaleTown_EventScript_BlockWestEntrance",
            "FLAG_ADVENTURE_STARTED",
        )

        self.assertEqual(
            resolve_guard_state(candidate, _snapshot(flags=(("ADVENTURE_STARTED", True),))),
            "unsatisfied",
        )
        self.assertEqual(
            resolve_guard_state(candidate, _snapshot(flags=(("ADVENTURE_STARTED", False),))),
            "satisfied",
        )

    def test_missing_and_unavailable_observations_stay_distinct(self):
        candidate = _candidate("goto_if_set FLAG_OPEN, Done", "FLAG_OPEN")
        self.assertEqual(resolve_guard_state(candidate, _snapshot()), "unknown")
        self.assertEqual(
            resolve_guard_state(candidate, _snapshot(available=False, lifecycle=CampaignObservationLifecycle.UNAVAILABLE)),
            "unavailable",
        )
        self.assertEqual(
            resolve_guard_state(candidate, _snapshot(available=True, lifecycle=CampaignObservationLifecycle.FRESH_START)),
            "unavailable",
        )

    def test_trainer_clearance_is_not_invented_from_a_named_value(self):
        candidate = _candidate("goto_if_defeated TRAINER_CALVIN_1, Done", "TRAINER_CALVIN_1")
        snapshot = _snapshot(variables=(("TRAINER_CALVIN_1", 1),))
        self.assertEqual(resolve_guard_state(candidate, snapshot), "unknown")

    def test_resolution_returns_a_new_immutable_candidate(self):
        candidate = _candidate("goto_if_set FLAG_OPEN, Done", "FLAG_OPEN")
        resolved = resolve_candidate_observation(candidate, _snapshot(flags=(("FLAG_OPEN", True),)))
        self.assertIsNot(resolved, candidate)
        self.assertEqual(candidate.current_state, "unknown")
        self.assertEqual(resolved.current_state, "satisfied")


if __name__ == "__main__":
    unittest.main()
