from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from modules.map_path import Direction
from modules.map_data import MapRSE
from modules.navigation import NavigationAction, NavigationActionType
from modules.nuzlocke.gate_impact import EdgeKey
from modules.nuzlocke.gate_live_observation import (
    GateGuardPredicate,
    GateLiveObservationRecorder,
    load_gate_live_observer,
)


def _edge() -> EdgeKey:
    return EdgeKey("MAP_START", "MAP_TARGET", "walk")


def _navigation() -> NavigationAction:
    return NavigationAction(
        NavigationActionType.MOVE,
        Direction.East,
        ("MAP_START", (1, 1)),
        ("MAP_TARGET", (0, 1)),
    )


def _guard() -> GateGuardPredicate:
    return GateGuardPredicate("TEST_STATE", "equals", 1)


def _recorder() -> GateLiveObservationRecorder:
    return GateLiveObservationRecorder("session-1", guard_reader=lambda _symbol: 0)


class GateLiveObservationTests(unittest.TestCase):
    def test_explicit_blocked_and_open_boundaries_are_recorded(self):
        recorder = _recorder()
        recorder.arm("candidate-1", _edge(), guard=_guard())
        navigation = _navigation()

        self.assertEqual(recorder.record_navigation_issued(navigation, action_frame=10), 1)
        self.assertEqual(
            recorder.record_navigation_outcome(navigation, "blocked", observation_frame=12),
            1,
        )
        self.assertEqual(recorder.record_navigation_issued(navigation, action_frame=20), 1)
        self.assertEqual(
            recorder.record_navigation_outcome(navigation, "open", observation_frame=22),
            1,
        )

        self.assertEqual(len(recorder.observations), 2)
        self.assertEqual({item.blocked for item in recorder.observations}, {True, False})
        self.assertEqual(recorder.observations[0].action_frame, 10)
        self.assertEqual(recorder.observations[0].observation_frame, 12)
        self.assertEqual(recorder.observations[0].session_id, "session-1")
        self.assertEqual(recorder.observations[0].guard_symbol, "TEST_STATE")
        self.assertEqual(recorder.observations[0].guard_value, 0)
        self.assertEqual(recorder.observations[0].guard_expression, "TEST_STATE == 1")

    def test_missing_action_or_terminal_outcome_is_rejected(self):
        recorder = _recorder()

        self.assertEqual(
            recorder.record_attempt(
                candidate_id="candidate-1",
                edge=_edge(),
                source_kind="movement",
                action_kind="move",
                guard_state="unsatisfied",
                action_issued=False,
                outcome="blocked",
            ),
            0,
        )
        self.assertEqual(
            recorder.record_attempt(
                candidate_id="candidate-1",
                edge=_edge(),
                source_kind="movement",
                action_kind="move",
                guard_state="unsatisfied",
                action_issued=True,
                outcome=None,
            ),
            0,
        )
        self.assertEqual(recorder.observations, ())
        self.assertEqual(len(recorder.diagnostics), 2)

    def test_unknown_guard_and_unissued_navigation_do_not_classify(self):
        recorder = GateLiveObservationRecorder("session-1", guard_reader=lambda _symbol: (_ for _ in ()).throw(ValueError("unavailable")))
        recorder.arm("candidate-1", _edge(), guard=_guard())
        navigation = _navigation()

        recorder.record_navigation_issued(navigation, action_frame=10)
        self.assertEqual(recorder.record_navigation_outcome(navigation, "blocked", observation_frame=12), 0)
        self.assertEqual(recorder.record_navigation_outcome(navigation, "open", observation_frame=14), 0)
        self.assertEqual(recorder.observations, ())
        self.assertTrue(any("could not read" in item for item in recorder.diagnostics))
        self.assertTrue(any("no issued action" in item for item in recorder.diagnostics))

    def test_satisfied_guard_can_record_only_an_explicit_open_boundary(self):
        recorder = _recorder()
        self.assertEqual(
            recorder.record_attempt(
                candidate_id="candidate-1",
                edge=_edge(),
                source_kind="transition",
                action_kind="warp",
                guard_state="satisfied",
                action_issued=True,
                outcome="blocked",
            ),
            0,
        )
        self.assertEqual(
            recorder.record_attempt(
                candidate_id="candidate-1",
                edge=_edge(),
                source_kind="transition",
                action_kind="warp",
                guard_state="satisfied",
                action_issued=True,
                outcome="open",
            ),
            1,
        )
        self.assertEqual(recorder.observations[0].guard_state, "satisfied")

    def test_edge_and_candidate_must_be_armed(self):
        recorder = _recorder()
        recorder.arm("candidate-1", EdgeKey("OTHER", "MAP_TARGET", "walk"), guard=_guard())
        navigation = _navigation()

        self.assertEqual(recorder.record_navigation_issued(navigation, action_frame=10), 0)
        self.assertEqual(recorder.record_navigation_outcome(navigation, "blocked", observation_frame=12), 0)
        self.assertEqual(recorder.observations, ())

    def test_serialization_is_deterministic(self):
        def build() -> GateLiveObservationRecorder:
            recorder = _recorder()
            recorder.arm("candidate-1", _edge(), guard=_guard())
            navigation = _navigation()
            recorder.record_navigation_issued(navigation, action_frame=10)
            recorder.record_navigation_outcome(navigation, "blocked", observation_frame=12)
            return recorder

        self.assertEqual(build().to_json(), build().to_json())

    def test_enum_backed_navigation_maps_match_tuple_bindings(self):
        recorder = _recorder()
        recorder.arm(
            "candidate-1",
            EdgeKey((0, 0), (0, 19), "map_connection"),
            guard=_guard(),
            source_kind="transition",
        )
        navigation = NavigationAction(
            NavigationActionType.WARP,
            Direction.West,
            (MapRSE.PETALBURG_CITY, (0, 12)),
            (MapRSE.ROUTE104, (39, 62)),
            transition_kind="map_connection",
        )

        self.assertEqual(recorder.record_navigation_issued(navigation, action_frame=10), 1)
        self.assertEqual(recorder.record_navigation_outcome(navigation, "open", observation_frame=12), 1)

    def test_terminal_failure_records_blocked_edge_after_controller_disposal(self):
        recorder = _recorder()
        recorder.arm(
            "candidate-1",
            EdgeKey("MAP_START", "MAP_TARGET", "warp"),
            guard=GateGuardPredicate("TEST_STATE", "equals", 1),
            source_kind="transition",
        )
        navigation = NavigationAction(
            NavigationActionType.WARP,
            Direction.East,
            ("MAP_START", (1, 1)),
            ("MAP_TARGET", (0, 1)),
            transition_kind="warp",
        )

        self.assertEqual(
            recorder.record_navigation_terminal_outcome(
                navigation,
                "blocked",
                action_frame=10,
                observation_frame=12,
                source_kind="transition",
                dynamic_objects=(),
            ),
            1,
        )
        self.assertTrue(recorder.observations[0].blocked)
        self.assertEqual(recorder.to_dict()["pending"], [])

    def test_dynamic_object_positions_are_retained_with_live_evidence(self):
        recorder = _recorder()
        recorder.arm("candidate-1", _edge(), guard=_guard())
        navigation = _navigation()
        object_observation = type(
            "Object",
            (),
            {"local_id": 8, "location": ((0, 0), (5, 11))},
        )()

        self.assertEqual(
            recorder.record_navigation_issued(
                navigation,
                action_frame=10,
                dynamic_objects=(object_observation,),
            ),
            1,
        )
        self.assertEqual(recorder.record_navigation_outcome(navigation, "blocked", observation_frame=12), 1)
        self.assertEqual(recorder.observations[0].dynamic_object_positions, ((8, (0, 0), (5, 11)),))
        self.assertIn('"local_id": 8', recorder.to_json())

    def test_disposable_fixture_configuration_round_trips(self):
        payload = {
            "session_id": "fixture-session",
            "output": "/tmp/fixture-session.json",
            "fixture": {
                "source_map": [0, 0],
                "event_vars": {"PETALBURG_CITY_STATE": 0},
                "player_coordinates": [29, 12],
                "object_positions": {"8": [5, 11]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "fixture.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            recorder = load_gate_live_observer(config)

        self.assertIsNotNone(recorder.fixture)
        self.assertEqual(recorder.fixture.source_map, (0, 0))
        self.assertEqual(recorder.fixture.event_vars, (("PETALBURG_CITY_STATE", 0),))
        self.assertEqual(recorder.fixture.player_coordinates, (29, 12))
        self.assertEqual(recorder.fixture.object_positions, ((8, (5, 11)),))
        self.assertEqual(json.loads(recorder.to_json())["fixture"]["object_positions"]["8"], [5, 11])


if __name__ == "__main__":
    unittest.main()
