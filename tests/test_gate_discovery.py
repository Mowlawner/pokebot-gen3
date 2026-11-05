from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.gate_discovery import scan_source_tree


def _fixture_source(root: Path, *, reverse_maps: bool = False) -> None:
    maps = ["Route102", "PetalburgCity"]
    if reverse_maps:
        maps.reverse()
    for name in maps:
        directory = root / "data" / "maps" / name
        directory.mkdir(parents=True, exist_ok=True)
        map_id = f"MAP_{name.upper()}"
        if name == "Route102":
            data = {
                "id": map_id,
                "name": name,
                "object_events": [
                    {
                        "local_id": "LOCALID_CALVIN",
                        "trainer_type": "TRAINER_TYPE_NORMAL",
                        "script": "Route102_EventScript_Calvin",
                        "flag": "0",
                    },
                    {
                        "local_id": "LOCALID_BLOCKER",
                        "script": "0x0",
                        "flag": "FLAG_HIDE_ROUTE102_BLOCKER",
                    },
                ],
                "coord_events": [
                    {
                        "var": "VAR_ROUTE102_STATE",
                        "var_value": "2",
                        "script": "Route102_EventScript_Open",
                    }
                ],
                "warp_events": [
                    {"dest_map": "MAP_OLDALE_TOWN"},
                ],
            }
            script = """Route102_MapScripts::
    map_script_2 VAR_ROUTE102_STATE, 2, Route102_EventScript_Open
    .2byte 0

Route102_EventScript_Calvin::
    trainerbattle_single TRAINER_CALVIN_1, Intro, Defeat
    end

Route102_EventScript_Open::
    goto_if_set FLAG_ROUTE102_OPEN, Route102_EventScript_Done
    setflag FLAG_ROUTE102_OPEN
    setvar VAR_ROUTE102_STATE, 3
    end

Route102_EventScript_Done::
    compare VAR_ROUTE102_STATE, 3
    specialvar VAR_RESULT, UnsupportedComputedValue
    end
"""
        else:
            data = {
                "id": map_id,
                "name": name,
                "object_events": [],
                "coord_events": [],
                "warp_events": [],
            }
            script = "PetalburgCity_MapScripts::\n    setflag FLAG_VISITED_PETALBURG_CITY\n    end\n"
        (directory / "map.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        (directory / "scripts.inc").write_text(script, encoding="utf-8")


class GateDiscoveryTests(unittest.TestCase):
    def test_fixture_extracts_guards_writers_and_trainers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture_source(root)
            report = scan_source_tree(root, source_revision="fixture-1")

        self.assertEqual(report.source_revision, "fixture-1")
        self.assertTrue(any(candidate.kind == "trainer_checkpoint" for candidate in report.candidates))
        blocker = next(candidate for candidate in report.candidates if candidate.kind == "object_visibility")
        self.assertEqual(blocker.guard_symbol, "FLAG_HIDE_ROUTE102_BLOCKER")
        self.assertEqual(blocker.current_state, "unknown")
        open_guard = next(
            candidate
            for candidate in report.candidates
            if candidate.guard_symbol == "FLAG_ROUTE102_OPEN" and candidate.source.symbol == "Route102_EventScript_Open"
        )
        self.assertEqual(open_guard.producer_actions[0].action, "setflag")
        self.assertEqual(open_guard.producer_actions[0].symbol, "Route102_EventScript_Open")

    def test_unsupported_expression_is_explicit_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture_source(root)
            report = scan_source_tree(root, source_revision="fixture-1")
        unknown = [candidate for candidate in report.candidates if candidate.kind == "unknown_expression"]
        self.assertEqual(len(unknown), 2)
        self.assertTrue(all(candidate.current_state == "unknown" for candidate in unknown))
        self.assertTrue(all("unsupported" in candidate.diagnostics[0] for candidate in unknown))

    def test_report_is_canonical_and_independent_of_file_creation_order(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _fixture_source(Path(first), reverse_maps=False)
            _fixture_source(Path(second), reverse_maps=True)
            first_report = scan_source_tree(first, source_revision="fixture-1")
            second_report = scan_source_tree(second, source_revision="fixture-1")
        first_dict = first_report.to_dict()
        second_dict = second_report.to_dict()
        first_dict["source_root"] = "<fixture>"
        second_dict["source_root"] = "<fixture>"
        self.assertEqual(first_dict, second_dict)
        self.assertEqual(first_report.to_json().count("candidate_id"), len(first_report.candidates))

    def test_candidate_ids_include_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture_source(root)
            first = scan_source_tree(root, source_revision="fixture-1")
            second = scan_source_tree(root, source_revision="fixture-2")
        self.assertNotEqual(
            [candidate.candidate_id for candidate in first.candidates],
            [candidate.candidate_id for candidate in second.candidates],
        )


if __name__ == "__main__":
    unittest.main()
