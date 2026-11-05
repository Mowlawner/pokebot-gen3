from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.gate_discovery import scan_source_tree
from modules.overworld import TriggerObservation, evaluate_trigger_condition


SOURCE_REVISION = "5eff78649e7170a877b961ef0b3da13b81a16038"


def _write_early_state_fixture(root: Path) -> None:
    maps = {
        "LittlerootTown": {
            "id": "MAP_LITTLEROOT_TOWN",
            "name": "LittlerootTown",
            "object_events": [],
            "coord_events": [
                {
                    "type": "trigger",
                    "x": 10,
                    "y": 1,
                    "var": "VAR_LITTLEROOT_TOWN_STATE",
                    "var_value": "0",
                    "script": "LittlerootTown_EventScript_NeedPokemonTriggerLeft",
                },
                {
                    "type": "trigger",
                    "x": 11,
                    "y": 1,
                    "var": "VAR_LITTLEROOT_TOWN_STATE",
                    "var_value": "0",
                    "script": "LittlerootTown_EventScript_NeedPokemonTriggerRight",
                },
                {
                    "type": "trigger",
                    "x": 11,
                    "y": 1,
                    "var": "VAR_LITTLEROOT_TOWN_STATE",
                    "var_value": "1",
                    "script": "LittlerootTown_EventScript_GoSaveBirchTrigger",
                },
                {
                    "type": "trigger",
                    "x": 8,
                    "y": 9,
                    "var": "VAR_LITTLEROOT_TOWN_STATE",
                    "var_value": "3",
                    "script": "LittlerootTown_EventScript_GiveRunningShoesTrigger4",
                },
            ],
            "warp_events": [],
        },
        "Route101": {
            "id": "MAP_ROUTE101",
            "name": "Route101",
            "object_events": [],
            "coord_events": [
                {
                    "type": "trigger",
                    "x": 10,
                    "y": 19,
                    "var": "VAR_ROUTE101_STATE",
                    "var_value": "1",
                    "script": "Route101_EventScript_StartBirchRescue",
                },
                {
                    "type": "trigger",
                    "x": 10,
                    "y": 18,
                    "var": "VAR_ROUTE101_STATE",
                    "var_value": "2",
                    "script": "Route101_EventScript_PreventExitSouth",
                },
                {
                    "type": "trigger",
                    "x": 6,
                    "y": 16,
                    "var": "VAR_ROUTE101_STATE",
                    "var_value": "2",
                    "script": "Route101_EventScript_PreventExitWest",
                },
                {
                    "type": "trigger",
                    "x": 7,
                    "y": 13,
                    "var": "VAR_ROUTE101_STATE",
                    "var_value": "2",
                    "script": "Route101_EventScript_PreventExitNorth",
                },
            ],
            "warp_events": [],
        },
    }
    scripts = {
        "LittlerootTown": """LittlerootTown_EventScript_GoSaveBirchTrigger::
    setvar VAR_LITTLEROOT_TOWN_STATE, 2
    end

LittlerootTown_EventScript_SetReceivedRunningShoes::
    setvar VAR_LITTLEROOT_TOWN_STATE, 4
    end
""",
        "Route101": """Route101_EventScript_BirchsBag::
    setvar VAR_BIRCH_LAB_STATE, 2
    setvar VAR_ROUTE101_STATE, 3
    end
""",
    }
    for map_name, data in maps.items():
        directory = root / "data" / "maps" / map_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "map.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        (directory / "scripts.inc").write_text(scripts[map_name], encoding="utf-8")


class EmeraldStateGuardFixtureTests(unittest.TestCase):
    def _report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_early_state_fixture(root)
            return scan_source_tree(root, source_revision=SOURCE_REVISION)

    @staticmethod
    def _coordinate_candidates(report, map_name: str):
        map_id = {
            "LittlerootTown": "MAP_LITTLEROOT_TOWN",
            "Route101": "MAP_ROUTE101",
        }[map_name]
        return tuple(
            candidate
            for candidate in report.candidates
            if candidate.affected_maps == (map_id,)
            and dict(candidate.metadata).get("event_type") == "coordinate_trigger"
        )

    def test_littleroot_fixture_retains_scene_state_boundaries_and_writers(self):
        report = self._report()
        candidates = self._coordinate_candidates(report, "LittlerootTown")

        self.assertEqual(
            {
                (candidate.guard_value, dict(candidate.metadata)["x"], dict(candidate.metadata)["y"], dict(candidate.metadata)["script"])
                for candidate in candidates
            },
            {
                ("0", "10", "1", "LittlerootTown_EventScript_NeedPokemonTriggerLeft"),
                ("0", "11", "1", "LittlerootTown_EventScript_NeedPokemonTriggerRight"),
                ("1", "11", "1", "LittlerootTown_EventScript_GoSaveBirchTrigger"),
                ("3", "8", "9", "LittlerootTown_EventScript_GiveRunningShoesTrigger4"),
            },
        )
        save_birch = next(candidate for candidate in candidates if candidate.guard_value == "1")
        self.assertTrue(
            any(action.symbol == "LittlerootTown_EventScript_GoSaveBirchTrigger" for action in save_birch.producer_actions)
        )
        self.assertTrue(any(action.value == "2" for action in save_birch.producer_actions))

    def test_route101_fixture_distinguishes_blocked_exit_scene_from_open_post_scene(self):
        report = self._report()
        candidates = self._coordinate_candidates(report, "Route101")
        exit_candidates = tuple(candidate for candidate in candidates if candidate.guard_value == "2")

        self.assertEqual(
            {
                (dict(candidate.metadata)["x"], dict(candidate.metadata)["y"], dict(candidate.metadata)["script"])
                for candidate in exit_candidates
            },
            {
                ("10", "18", "Route101_EventScript_PreventExitSouth"),
                ("6", "16", "Route101_EventScript_PreventExitWest"),
                ("7", "13", "Route101_EventScript_PreventExitNorth"),
            },
        )
        for candidate in exit_candidates:
            trigger = TriggerObservation(
                trigger_id=candidate.candidate_id,
                locations=frozenset(),
                condition_required_value=int(candidate.guard_value),
            )
            self.assertTrue(evaluate_trigger_condition(trigger, 2))
            self.assertFalse(evaluate_trigger_condition(trigger, 3))

        rescue_scene = next(
            candidate
            for candidate in candidates
            if dict(candidate.metadata)["script"] == "Route101_EventScript_StartBirchRescue"
        )
        self.assertEqual(rescue_scene.guard_value, "1")
        self.assertTrue(any(action.value == "3" for action in rescue_scene.producer_actions))


if __name__ == "__main__":
    unittest.main()
