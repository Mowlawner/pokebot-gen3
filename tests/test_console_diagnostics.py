import types
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console


class TestDiagnosticPrint(unittest.TestCase):
    def test_console_log_is_not_created_without_a_path(self):
        from modules.console import configure_console_log

        with tempfile.TemporaryDirectory() as directory:
            configure_console_log(None)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_console_log_tees_rich_output_to_plain_text_file(self):
        from modules.console import close_console_log, configure_console_log

        output = StringIO()
        sink = Console(file=output, color_system="standard", force_terminal=True, width=100)
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            with patch("modules.console.console", sink):
                configure_console_log(log_path)
                sink.print("[red]hello[/]")
                close_console_log()

            self.assertIn("hello", output.getvalue())
            logged = log_path.read_text(encoding="utf-8")
            self.assertIn("hello", logged)
            self.assertNotIn("\x1b[", logged)

    def test_campaign_objective_ids_show_map_names(self):
        from modules.console import format_campaign_objective_id

        self.assertEqual(format_campaign_objective_id("obtain_encounter:0:17"), "obtain_encounter:Route 102")
        self.assertEqual(format_campaign_objective_id("restock_recovery_items"), "restock_recovery_items")

    def test_campaign_frontier_is_normal_rich_output(self):
        from modules.console import print_campaign_frontier

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        with patch("modules.console.console", sink):
            print_campaign_frontier(
                ("receive_pokedex",),
                ("obtain_encounter:0:17",),
                selected="obtain_encounter:0:17",
                status="ready",
                reason="nearest eligible encounter",
            )

        sink.print.assert_called_once()
        rendered = sink.print.call_args.args[0]
        self.assertIn("Campaign Objective Frontier", rendered.title)

    def test_campaign_encounter_stage_panel_reports_safe_and_permissive_candidates(self):
        from modules.console import print_campaign_encounter_stage

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        evidence = types.SimpleNamespace(
            current_map=(0, 19),
            next_map=(0, 24),
            blocked=True,
            selected_candidate=1,
            readiness=None,
            known_hazard_ids=("trainer:(0, 19):3",),
            unresolved_hazard_ids=(),
            trainer_evidence=(
                types.SimpleNamespace(
                    trainer_id="trainer:(0, 19):28",
                    object_id=28,
                    location=((0, 19), (18, 67)),
                    trainer_type="Normal",
                    trainer_range=3,
                    facing="South",
                    movement_type="WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT",
                    movement_range=(6, 5),
                    movement_origin=((0, 19), (18, 67)),
                    defeated=False,
                    hazard_locations=(((0, 19), (18, 66)),),
                    engagement_locations=(((0, 19), (18, 66)),),
                    activation_locations=(((0, 19), (18, 68)),),
                ),
            ),
            reason="stage_unavailable:optional_or_unknown_trainer_exposure_requires_explicit_promotion",
            shared_search=True,
            safe_candidates=(
                types.SimpleNamespace(
                    index=0,
                    entry=((0, 19), (10, 30)),
                    approach=((0, 19), (10, 29)),
                    destination=((0, 24), (14, 5)),
                    status="not_found",
                    route_cost=None,
                    trainer_ids=(),
                    trainer_free=None,
                    encounter_opportunities=None,
                ),
            ),
            permissive_candidates=(
                types.SimpleNamespace(
                    index=0,
                    entry=((0, 19), (10, 30)),
                    approach=((0, 19), (10, 29)),
                    destination=((0, 24), (14, 5)),
                    status="found",
                    route_cost=84,
                    trainer_ids=("trainer:Route 104 (0, 19):28",),
                    trainer_free=False,
                    encounter_opportunities=3,
                ),
            ),
        )

        with patch("modules.console.console", sink):
            print_campaign_encounter_stage(evidence)

        rendered = output.getvalue()
        self.assertIn("Campaign Encounter Stage", rendered)
        self.assertIn("BLOCKED", rendered)
        self.assertIn("trainer:Route 104 (0, 19):3", rendered)
        self.assertIn("permissive=trainer:Route", rendered)
        self.assertIn("Grass", rendered)
        self.assertIn("84", rendered)
        self.assertIn("WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT", rendered)
        self.assertIn("movement_range", rendered)

    def test_print_always_prints_without_debug_but_diagnostic_print_does_not(self):
        from modules.console import diagnostic_print, print_always

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink), patch("modules.context.context.debug", False), patch(
            "modules.context.context.debug_trace", False
        ):
            diagnostic_print("DEBUG_ONLY_RECORD", trace=False)
            print_always("ALWAYS_RECORD")

        rendered = output.getvalue()
        self.assertIn("ALWAYS_RECORD", rendered)
        self.assertNotIn("DEBUG_ONLY_RECORD", rendered)

    def test_campaign_snapshot_party_rows_show_where_each_pokemon_was_met(self):
        from modules.console import print_campaign_snapshot

        output = StringIO()
        sink = Console(file=output, color_system=None, width=160)
        member = lambda species, nickname, met: types.SimpleNamespace(
            species=species,
            nickname=nickname,
            level=5,
            current_hp=20,
            max_hp=20,
            fainted=False,
            status="healthy",
            met_location=met,
        )
        snapshot = types.SimpleNamespace(
            frame=12,
            player=types.SimpleNamespace(name="May", map_name="Route 104"),
            party=(member("Marill", "David", "Route 104"), member("Mudkip", "William", None)),
        )
        with patch("modules.console.console", sink):
            print_campaign_snapshot(snapshot=snapshot)

        rendered = " ".join(output.getvalue().split())
        self.assertIn('Marill "David" Lv5 HP 20/20 (healthy) [met: Route 104]', rendered)
        # An unknown origin is simply omitted rather than shown as a guess.
        self.assertIn('Mudkip "William" Lv5 HP 20/20 (healthy)', rendered)
        self.assertNotIn('Mudkip "William" Lv5 HP 20/20 (healthy) [met', rendered)

    def test_campaign_snapshot_reports_party_cap_boss_and_exp_budget(self):
        from modules.console import print_campaign_snapshot

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        snapshot = types.SimpleNamespace(
            frame=12,
            player=types.SimpleNamespace(name="May", map_name="Route 116"),
                party=(
                    types.SimpleNamespace(
                        species="Treecko",
                        nickname="Sprig",
                    level=14,
                    current_hp=22,
                    max_hp=30,
                    fainted=False,
                    status="healthy",
                ),
            ),
        )
        budget = types.SimpleNamespace(
            cap_headroom=120,
            required_exp=80,
            unavoidable_exp=35,
            remaining_deficit=45,
        )
        status = types.SimpleNamespace(objective="Prepare Roxanne", intent="Train")
        boss = types.SimpleNamespace(boss_id="roxanne")
        checkpoint = types.SimpleNamespace(checkpoint_id="rom:emerald:12", source="active_rom")
        with patch("modules.console.console", sink):
            print_campaign_snapshot(
                snapshot=snapshot,
                status=status,
                level_cap=14,
                boss=boss,
                exp_budget=budget,
                trainer_checkpoint=checkpoint,
            )

        rendered = output.getvalue()
        self.assertIn("Campaign Snapshot", rendered)
        self.assertIn("May", rendered)
        self.assertIn('Treecko "Sprig" Lv14 HP 22/30', rendered)
        self.assertIn("roxanne / 14", rendered)
        self.assertIn("headroom=120", rendered)
        self.assertIn("rom:emerald:12 (active_rom)", rendered)

    def test_campaign_snapshot_formats_map_names_in_trainer_fact_ids(self):
        from modules.console import print_campaign_snapshot

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        snapshot = types.SimpleNamespace(
            frame=12,
            player=types.SimpleNamespace(name="May", map_name="Route 102"),
            party=(),
        )
        fact = types.SimpleNamespace(
            display_name="Calvin",
            trainer_affordance_id="trainer:(0, 17):1",
            source_records=(("active_rom", "rom:BPEE0:318"),),
        )
        with patch("modules.console.console", sink):
            print_campaign_snapshot(snapshot=snapshot, trainer_facts=(fact,))

        rendered = output.getvalue()
        self.assertIn("Calvin", rendered)
        self.assertIn("Route 102 (0, 17)", rendered)

    def test_navigation_cancellation_reports_reason_and_request_change(self):
        from modules.console import print_navigation_cancellation

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        request = (((0, 17), (6, 5)), "Reach Route 103", (("warp",),))
        replacement = (((0, 9), (7, 17)), "Reach Route 103", (("warp",),))
        with patch("modules.console.console", sink):
            print_navigation_cancellation(
                phase="requested",
                reason="newer navigation request superseded the active request",
                request_key=request,
                replacement_key=replacement,
                changed=("start",),
                duration_ms=123.4,
            )

        rendered = output.getvalue()
        self.assertIn("NAVIGATION_PATHFINDING_CANCELLED", rendered)
        self.assertIn("newer navigation request superseded", rendered)
        self.assertIn("Route 102 (0, 17)", rendered)
        self.assertIn("Littleroot Town (0, 9)", rendered)
        self.assertIn("start", rendered)

    def test_navigation_invalidation_reports_cached_route_context(self):
        from modules.console import print_navigation_invalidation

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        cached_action = types.SimpleNamespace(
            action_type=types.SimpleNamespace(name="MOVE"),
            source=((0, 9), (7, 17)),
            destination=((0, 9), (8, 17)),
        )
        with patch("modules.console.console", sink):
            print_navigation_invalidation(
                reason="world_signature_changed",
                route_id=12,
                current=((0, 9), (7, 17)),
                goal="SemanticTarget(target_map=(0, 18))",
                action_index=6,
                remaining_actions=76,
                total_actions=82,
                cached_action=cached_action,
                detail="changed=('objects',) relevant=route_object affects_route=True",
            )

        rendered = output.getvalue()
        self.assertIn("NAVIGATION_PATH_INVALIDATED", rendered)
        self.assertIn("world_signature_changed", rendered)
        self.assertIn("Littleroot Town (0, 9)", rendered)
        self.assertIn("Route 103 (0, 18)", rendered)
        self.assertIn("remaining=76", rendered)
        self.assertIn("route_object", rendered)

    def test_recovery_handoff_reports_calculation_mount_and_action_fields(self):
        from modules.console import print_recovery_handoff

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        with patch("modules.console.console", sink):
            print_recovery_handoff(
                phase="first_action_emitted",
                objective_id="complete_intro_rival",
                execution_phase="RECOVERY",
                destination=((0, 10), (6, 16)),
                action="NAVIGATE_TOWARD_GOAL",
                action_source=((0, 18), (10, 21)),
                action_destination=((0, 10), (10, 0)),
                route_actions=21,
                route_cost=25,
                detail="first recovery action executed",
            )

        rendered = output.getvalue()
        self.assertIn("CAMPAIGN_RECOVERY_HANDOFF phase=first_action_emitted", rendered)
        self.assertIn("complete_intro_rival", rendered)
        self.assertIn("Oldale Town", rendered)
        self.assertIn("first recovery action executed", rendered)

    def test_recovery_plan_reports_stop_composition(self):
        from modules.console import print_recovery_plan

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        with patch("modules.console.console", sink):
            print_recovery_plan(
                objective_id="complete_intro_rival",
                decision="RECOVER",
                reason="OPPORTUNISTIC_RECOVERY",
                route_analysis_candidates=1,
                route_analysis_eligible=0,
                observed_route_available=True,
                observed_center_available=False,
                observed_healing_source_available=True,
                observed_safe_to_reach=True,
                observed_destination=((0, 10), (6, 16)),
                stop_present=True,
                stop_destination=((0, 10), (6, 16)),
                stop_route_actions=35,
                stop_route_cost=38,
                detail="using completed recovery route fallback",
            )

        rendered = output.getvalue()
        self.assertIn("Recovery Plan", rendered)
        self.assertIn("CAMPAIGN_RECOVERY_PLAN", rendered)
        self.assertIn("eligible=0", rendered)
        self.assertIn("Recovery stop", rendered)
        self.assertIn("present", rendered)
        self.assertIn("using completed recovery route fallback", rendered)

    def test_campaign_field_lead_reports_pokemon_for_each_party_slot(self):
        from modules.console import print_campaign_field_lead

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        party = (
            types.SimpleNamespace(species="Wurmple", nickname="Donna", level=3),
            types.SimpleNamespace(species="Mudkip", nickname="Henry", level=7),
        )
        with patch("modules.console.console", sink):
            print_campaign_field_lead(
                objective_id="obtain_encounter:0:17",
                current_index=0,
                selected_index=1,
                reason="field lead policy",
                party_slots=party,
            )

        rendered = output.getvalue()
        self.assertIn("Party slots", rendered)
        self.assertIn('0: Wurmple "Donna" Lv3', rendered)
        self.assertIn('1: Mudkip "Henry" Lv7', rendered)

    def test_campaign_frontier_shows_selected_restock_items(self):
        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink):
            from modules.console import print_campaign_frontier

            print_campaign_frontier(
                (),
                (),
                selected="restock_recovery_items",
                status="ready",
                reason="selected required recovery-item restock task",
                selected_detail="4x Antidote, 5x Potion, 10x Poké Ball",
            )

        self.assertIn("Items", output.getvalue())
        self.assertIn("4x Antidote, 5x Potion, 10x Poké Ball", output.getvalue())

    def test_campaign_frontier_shows_encounter_tradeoffs(self):
        from modules.console import print_campaign_frontier

        output = StringIO()
        sink = Console(file=output, color_system=None, width=200)
        with patch("modules.console.console", sink):
            print_campaign_frontier(
                (),
                ("complete_petalburg_wally", "obtain_encounter:0:17"),
                selected="complete_petalburg_wally",
                status="ready",
                reason="selected required campaign progression task",
                decision_details=(
                    {
                        "task_id": "obtain_encounter:0:17",
                        "selected": False,
                        "reason": "deferred because its detour is not worth interrupting required progression",
                        "strategic_score": 4.5,
                        "route_cost": 12,
                        "detour_cost": 8,
                        "trainer_preparation_gain": 0.25,
                        "trainer_checkpoint_id": "route102:first_poochyena",
                    },
                ),
            )

        rendered = output.getvalue()
        self.assertIn("Encounter", rendered)
        self.assertIn("tradeoffs", rendered)
        self.assertIn("Route 102", rendered)
        self.assertIn("detour=8", rendered)
        self.assertIn("route102:first_poochyena", rendered)

    def test_campaign_frontier_shows_ordered_encounter_portfolio(self):
        from modules.console import print_campaign_frontier

        output = StringIO()
        sink = Console(file=output, color_system=None, width=220)
        with patch("modules.console.console", sink):
            print_campaign_frontier(
                (),
                ("obtain_encounter:0:17",),
                selected="obtain_encounter:0:17",
                status="ready",
                encounter_portfolio=(
                    {
                        "rank": 1,
                        "task_id": "obtain_encounter:0:17",
                        "area": (0, 17),
                        "method": "land",
                        "selected": True,
                        "portfolio_selected": True,
                        "score": 12.5,
                        "policy": "prefer",
                        "classification": "on_route",
                        "route_cost": 10,
                        "detour_cost": 0,
                        "trainer_preparation_gain": 1.25,
                        "species": ("Zigzagoon", "Wurmple"),
                    },
                ),
            )

        rendered = output.getvalue()
        self.assertIn("Ordered portfolio", rendered)
        self.assertIn("SELECTED", rendered)
        self.assertIn("Route 102", rendered)
        self.assertIn("Zigzagoon, Wurmple", rendered)

    def test_campaign_snapshot_shows_trainer_plan_and_readiness_reason(self):
        from modules.console import print_campaign_snapshot

        output = StringIO()
        sink = Console(file=output, color_system=None, width=240)
        result = SimpleNamespace(
            action_name="Tackle",
            principal_variation=("Tackle", "Tackle"),
            win_probability=0.92,
            active_faint_probability=0.02,
        )
        readiness = SimpleNamespace(
            status=SimpleNamespace(value="ready"),
            lead_name="Mudkip",
            lead_source="party",
            recommended_action="Tackle",
            roster_action="use_party_lead",
            win_probability=0.92,
            active_faint_probability=0.02,
            party_wipe_probability=0.01,
            run_loss_probability=0.0,
            required_hp_ratio=0.5,
            reason="party lead has a robust line",
            lead_assessments=(
                SimpleNamespace(
                    name="Mudkip",
                    source="party",
                    safe=True,
                    lines=(
                        SimpleNamespace(
                            opponent_species="Poochyena",
                            opponent_level=5,
                            result=result,
                        ),
                    ),
                ),
            ),
        )
        readiness_policy = SimpleNamespace(
            readiness_decision=SimpleNamespace(value="CONTINUE"),
            readiness_reason=SimpleNamespace(value="PARTY_HEALTHY"),
        )
        snapshot = SimpleNamespace(
            frame=1,
            player=SimpleNamespace(name="May", map_name="Route 102"),
            party=(),
        )
        checkpoint = SimpleNamespace(checkpoint_id="route102:first_poochyena", display_name="Calvin")
        with patch("modules.console.console", sink):
            print_campaign_snapshot(
                snapshot=snapshot,
                status=SimpleNamespace(objective="Travel", intent="Continue"),
                trainer_checkpoint=checkpoint,
                trainer_readiness=readiness,
                readiness=readiness_policy,
            )

        rendered = output.getvalue()
        self.assertIn("Readiness gate", rendered)
        self.assertIn("PARTY_HEALTHY", rendered)
        self.assertIn("Trainer plan", rendered)
        self.assertIn("first_action=Tackle", rendered)
        self.assertIn("Poochyena Lv5", rendered)
        self.assertIn("line=Tackle → Tackle", rendered)

    def test_campaign_snapshot_explains_trainer_preparation_ceiling_handoff(self):
        from modules.console import print_campaign_snapshot

        output = StringIO()
        sink = Console(file=output, color_system=None, width=240)
        readiness = SimpleNamespace(
            readiness_decision=SimpleNamespace(value="CONTINUE"),
            readiness_reason=SimpleNamespace(value="TRAINER_PREPARATION_CEILING"),
            preparation_current_level=6,
            preparation_ceiling_level=6,
        )
        snapshot = SimpleNamespace(
            frame=1,
            player=SimpleNamespace(name="May", map_name="Route 102"),
            party=(),
        )
        with patch("modules.console.console", sink):
            print_campaign_snapshot(
                snapshot=snapshot,
                status=SimpleNamespace(objective="Reach Calvin", intent="Continue"),
                readiness=readiness,
            )

        rendered = output.getvalue()
        self.assertIn("Prep boundary", rendered)
        self.assertIn("Lv6/6 reached", rendered)
        self.assertIn("no further training", rendered)
        self.assertIn("battle plan owns engagement risk", rendered)

    def test_navigation_pathfinding_record_includes_frame_and_elapsed_time(self):
        from modules.console import print_navigation_pathfinding

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        debug_context = types.SimpleNamespace(debug=True, debug_trace=True, frame=321)
        with patch("modules.console.console", sink), patch("modules.context.context", debug_context):
            print_navigation_pathfinding(
                phase="complete",
                call_id=7,
                algorithm="dijkstra",
                start=((0, 10), (6, 17)),
                goal="SemanticTarget(MAP)",
                goal_type="SemanticTarget",
                duration_ms=12.5,
                path_found=True,
            )

        rendered = output.getvalue()
        self.assertIn("NAVIGATION_PATHFINDING", rendered)
        self.assertIn("frame=321", rendered)
        self.assertIn("duration_ms=12.5", rendered)
        self.assertIn("Oldale Town", rendered)
        self.assertIn("(0, 10)", rendered)

    def test_navigation_pathfinding_record_is_visible_without_debug_trace(self):
        from modules.console import print_navigation_pathfinding

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        normal_context = types.SimpleNamespace(debug=False, debug_trace=False, frame=9)
        with patch("modules.console.console", sink), patch("modules.context.context", normal_context):
            print_navigation_pathfinding(
                phase="start",
                call_id=8,
                algorithm="astar",
                start=((0, 10), (1, 1)),
                goal="ReachLocation((0, 10), (4, 4))",
                goal_type="ReachLocation",
            )

        rendered = output.getvalue()
        self.assertIn("NAVIGATION_PATHFINDING", rendered)
        self.assertIn("frame=9", rendered)
        self.assertIn("id=8", rendered)

    def test_reach_warp_failure_diagnostic_is_allowed_by_trace_filter(self):
        from modules.console import diagnostic_print

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        debug_context = types.SimpleNamespace(debug=True, debug_trace=True)
        with patch("modules.console.console", sink), patch("modules.context.context", debug_context):
            diagnostic_print(
                "NAVIGATION_REACH_WARP_DIAGNOSTIC: target transition was rejected",
                trace=True,
                prefix="NAVIGATION_REACH_WARP_DIAGNOSTIC:",
            )

        self.assertIn("NAVIGATION_REACH_WARP_DIAGNOSTIC", output.getvalue())

    def test_map_formatting_does_not_relabel_coordinate_pairs(self):
        from modules.map_formatting import format_map_ids_in_text

        rendered = format_map_ids_in_text("map=(0, 10) coordinates=(10, 2)")

        self.assertIn("Oldale Town (0, 10)", rendered)
        self.assertIn("coordinates=(10, 2)", rendered)
        self.assertNotIn("Mauville City, House 1", rendered)

    def test_battle_action_uses_aligned_rich_panel(self):
        from modules.console import print_battle_action

        output = StringIO()
        sink = Console(file=output, color_system=None, width=100)
        with patch("modules.console.console", sink):
            print_battle_action(
                objective="defeat",
                target="KO",
                intent="select move",
                active="Torchic",
                opponent="Mudkip",
                action="UseMove(Growl)",
                source="PLANNER SAFE",
                evaluator="expectiminimax",
                reason="the modeled line survives the opponent response",
                active_hp="12/25",
                active_status="none",
                opponent_hp="15/15",
                opponent_status="none",
            )

        rendered = output.getvalue()
        self.assertIn("Battle Action", rendered)
        self.assertIn("Matchup", rendered)
        self.assertIn("Torchic  vs  Mudkip", rendered)
        self.assertIn("UseMove(Growl)", rendered)
        self.assertIn("Evaluator", rendered)
        self.assertIn("expectiminimax", rendered)
        self.assertIn("Active HP", rendered)
        self.assertIn("12/25 (none)", rendered)
        self.assertIn("Opponent HP", rendered)

    def test_battle_action_reason_preserves_stat_stage_numbers(self):
        from modules.console import print_battle_action

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink):
            print_battle_action(
                objective="defeat",
                target="KO",
                intent="select move",
                active="Treecko",
                opponent="Wurmple",
                action="UseMove(Pound)",
                source="PLANNER SAFE",
                reason="Live stages: active (attack=+2); opponent (defense=-1).",
            )

        self.assertIn("active (attack=+2); opponent (defense=-1)", output.getvalue())

    def test_battle_action_formats_expectiminimax_projection_as_nested_sections(self):
        from modules.console import print_battle_action

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        with patch("modules.console.console", sink):
            print_battle_action(
                objective="defeat",
                target="KO",
                intent="select move",
                active="Mudkip",
                opponent="Treecko",
                action="UseMove(Tackle)",
                source="PLANNER BEST_AVAILABLE",
                evaluator="expectiminimax",
                reason=(
                    "expectiminimax selected Tackle; win=unknown; party-wipe risk=unknown; "
                    "category=incomplete; line=projected opponent:Pound -> player:Tackle; "
                    "immediate=player=[Tackle:hit p=50.0% damage=5,Tackle:miss p=50.0%] "
                    "opponent=[Pound:hit p=100.0% damage=4] order=[player,opponent]; "
                    "search time budget expired"
                ),
            )

        rendered = output.getvalue()
        self.assertIn("Immediate projection", rendered)
        self.assertIn("Principal variation", rendered)
        self.assertIn("Player", rendered)
        self.assertIn("Opponent", rendered)
        self.assertIn("search time budget expired", rendered)

    def test_battle_switch_resolution_shows_requested_final_and_observed_targets(self):
        from modules.console import print_battle_switch_resolution

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink):
            print_battle_switch_resolution(
                status="mismatch",
                requested="Lotad",
                requested_index=3,
                final="Lotad",
                final_index=3,
                observed="Mudkip",
                observed_index=1,
                active="Wurmple",
                opponent="Shroomish",
                reason="observed party slot 1 instead of dispatched slot 3",
            )

        rendered = output.getvalue()
        self.assertIn("Battle Switch", rendered)
        self.assertIn("Requested", rendered)
        self.assertIn("Lotad [slot 3]", rendered)
        self.assertIn("Mudkip [slot 1]", rendered)
        self.assertIn("Mismatch", rendered)

    def test_battle_safety_stop_shows_repeated_action_and_matchup(self):
        from modules.console import print_battle_safety_stop

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink):
            print_battle_safety_stop(
                action="RotateLead(Lotad)",
                target="party slot 3",
                repetitions=4,
                active="Wurmple",
                opponent="Shroomish",
                reason="the same action produced no observable battle-state change",
            )

        rendered = output.getvalue()
        self.assertIn("Battle Safety Stop", rendered)
        self.assertIn("RotateLead(Lotad)", rendered)
        self.assertIn("Repetitions", rendered)
        self.assertIn("4", rendered)
        self.assertIn("Wurmple  vs  Shroomish", rendered)

    def test_battle_lifecycle_panels_include_conditional_and_aggregate_fields(self):
        from modules.console import print_battle_end, print_battle_event, print_battle_start, print_forced_replacement

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink):
            print_battle_start(
                battle_type="Trainer|WallyTutorial",
                encounter_type="tutorial",
                trainer_id=37,
                opponent_roster=("Poochyena", "Zigzagoon"),
                player_roster=("Treecko", "Mudkip"),
                opponent_envelope=("Poochyena Lv5: Tackle, Howl",),
                envelope_source="active_rom",
            )
            print_battle_event(event="pokemon fainted", side="player", pokemon="Poochyena")
            print_forced_replacement(
                fainted="Poochyena",
                selected="Wurmple",
                party_index=1,
                reason="selected as the safest legal replacement",
            )
            print_battle_end(
                outcome="Won",
                battle_type="Trainer",
                opponent_roster=("Poochyena",),
                total_damage_taken=12,
                damage_by_pokemon=(("Treecko", 8), ("Mudkip", 4)),
                items_used=("Potion -> Treecko",),
                moves_used=("Pound", "Absorb"),
                faints=("Poochyena",),
                switches=("player 0→1",),
            )

        rendered = output.getvalue()
        self.assertIn("Battle Start", rendered)
        self.assertIn("Trainer ID", rendered)
        self.assertIn("Poochyena, Zigzagoon", rendered)
        self.assertIn("Poochyena Lv5: Tackle, Howl", rendered)
        self.assertIn("active_rom", rendered)
        self.assertIn("Battle Event", rendered)
        self.assertIn("Forced Replacement", rendered)
        self.assertIn("Battle Finished", rendered)
        self.assertIn("Damage taken", rendered)
        self.assertIn("Potion -> Treecko", rendered)
        self.assertIn("Pound, Absorb", rendered)

    def test_in_progress_battle_summary_is_labeled_stopped(self):
        from modules.console import print_battle_end

        output = StringIO()
        sink = Console(file=output, color_system=None, width=120)
        with patch("modules.console.console", sink):
            print_battle_end(outcome="InProgress", battle_type="Wild", opponent_roster=("Shroomish",))

        self.assertIn("Battle Stopped", output.getvalue())
        self.assertNotIn("Battle Finished", output.getvalue())

    def test_recovery_destination_panel_exposes_each_candidate_result(self):
        from modules.console import print_recovery_destinations
        from modules.nuzlocke.resource_policy import RecoveryCandidateDiagnostic

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        with patch("modules.console.console", sink):
            print_recovery_destinations(
                (
                    RecoveryCandidateDiagnostic(
                        0,
                        "pokemon_center:petalburg",
                        (0, 5),
                        "failed",
                        reason="trainer exposure",
                    ),
                    RecoveryCandidateDiagnostic(
                        1,
                        "pokemon_center:oldale",
                        (0, 3),
                        "success",
                        distance=12,
                        reason="safe route",
                    ),
                )
            )

        rendered = output.getvalue()
        self.assertIn("Recovery Destination Search", rendered)
        self.assertIn("pokemon_center:petalburg", rendered)
        self.assertIn("trainer exposure", rendered)
        self.assertIn("pokemon_center:oldale", rendered)
        self.assertIn("safe route", rendered)

    def test_restock_and_money_panels_show_plans_candidates_and_wallets(self):
        from modules.console import (
            print_money_trainer_attempt,
            print_money_trainer_destinations,
            print_money_trainer_request,
            print_money_trainer_success,
            print_restock_attempt,
            print_restock_failure,
            print_restock_request,
            print_restock_success,
        )

        output = StringIO()
        sink = Console(file=output, color_system=None, width=140)
        candidate = types.SimpleNamespace(
            map_id=(0, 19),
            trainer_id="trainer:(0, 19):7",
            estimated_route_cost=12,
            source="static reachable map catalog",
        )
        with patch("modules.console.console", sink):
            print_restock_request(
                objective="restock_pokeballs",
                location=((0, 2), (3, 7)),
                wallet=56,
                cash_floor=0,
                required=("Poké Balls",),
                items=("none affordable",),
                reason="reserve is below policy target",
            )
            print_restock_attempt(
                method="Poké Mart purchase",
                destination=((0, 2), (3, 7)),
                items=("10x Poké Ball",),
                total_cost=2_000,
            )
            print_restock_success(
                method="Poké Mart purchase",
                detail="purchased 10x Poké Ball; wallet after $56",
            )
            print_restock_failure(
                method="Poké Mart purchase",
                error="insufficient money",
                action="seek trainer funding",
            )
            print_money_trainer_request(
                location=((0, 2), (3, 7)),
                wallet=56,
                reason="restock plan is unaffordable",
            )
            print_money_trainer_destinations((candidate,), selected=candidate.trainer_id)
            print_money_trainer_attempt(
                trainer=candidate.trainer_id,
                destination=candidate.map_id,
                wallet=56,
            )
            print_money_trainer_success(trainer=candidate.trainer_id, wallet_before=56, wallet_after=456)

        rendered = output.getvalue()
        for title in (
            "Restock Request",
            "Restock Attempt",
            "Restock Success",
            "Restock Failure",
            "Money Funding Request",
            "Money Trainer Search",
            "Money Trainer Attempt",
            "Money Trainer Success",
        ):
            self.assertIn(title, rendered)
        self.assertIn("10x Poké Ball", rendered)
        self.assertIn("trainer:(0, 19):7", rendered)
        self.assertIn("$56", rendered)
        self.assertIn("$400", rendered)

    def test_profile_output_is_disabled_without_profile_flag(self):
        from modules.console import profile_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="unused")
        context = types.SimpleNamespace(debug=False, debug_trace=True, debug_profile=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            profile_print(message)

        message.assert_not_called()
        sink.print.assert_not_called()

    def test_profile_output_does_not_require_trace_flag(self):
        from modules.console import profile_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="profile")
        context = types.SimpleNamespace(debug=False, debug_trace=False, debug_profile=True)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            profile_print(message)

        message.assert_called_once_with()
        sink.print.assert_called_once_with("profile")

    def test_disabled_diagnostic_does_not_format_message(self):
        from modules.console import diagnostic_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="unused")
        context = types.SimpleNamespace(debug=False, debug_trace=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message)

        message.assert_not_called()
        sink.print.assert_not_called()

    def test_trace_requires_explicit_trace_level(self):
        from modules.console import diagnostic_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="CAMPAIGN_RECOVERY_SOURCE: trace")
        context = types.SimpleNamespace(debug=True, debug_trace=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True, prefix="CAMPAIGN_RECOVERY_SOURCE")
        message.assert_not_called()

        context.debug_trace = True
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True, prefix="CAMPAIGN_RECOVERY_SOURCE")
        message.assert_called_once_with()
        sink.print.assert_called_once_with("CAMPAIGN_RECOVERY_SOURCE: trace")

    def test_filtered_lazy_trace_does_not_evaluate_message(self):
        from modules.console import diagnostic_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="CAMPAIGN_PLAN_TRACE: expensive")
        context = types.SimpleNamespace(debug=True, debug_trace=True)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True, prefix="UNSELECTED_TRACE")

        message.assert_not_called()
        sink.print.assert_not_called()


MALE_SIGN = chr(0x2642)  # not representable in cp1252


class TestUnencodableOutput(unittest.TestCase):
    """A character the terminal code page cannot encode must not crash the bot."""

    @staticmethod
    def _cp1252_stream():
        import io

        raw = io.BytesIO()
        return raw, io.TextIOWrapper(raw, encoding="cp1252", write_through=True)

    def test_tee_survives_a_character_the_terminal_cannot_encode(self):
        from modules.console import _ConsoleLogTee

        raw, terminal = self._cp1252_stream()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            tee = _ConsoleLogTee(terminal, log_path)
            try:
                tee.write("Nidoran" + MALE_SIGN + "\n")
            finally:
                tee.close()
            # The log file is UTF-8 and keeps the exact text.
            self.assertIn("Nidoran" + MALE_SIGN, log_path.read_text(encoding="utf-8"))
        self.assertIn(b"Nidoran", raw.getvalue())

    def test_reconfigure_makes_a_cp1252_stream_replace_instead_of_raising(self):
        from modules.console import _tolerate_unencodable_output

        raw, terminal = self._cp1252_stream()
        with self.assertRaises(UnicodeEncodeError):
            terminal.write(MALE_SIGN)
        _tolerate_unencodable_output(terminal)
        terminal.write(MALE_SIGN)
        self.assertEqual(raw.getvalue()[-1:], b"?")

    def test_streams_without_reconfigure_are_left_alone(self):
        from modules.console import _tolerate_unencodable_output

        _tolerate_unencodable_output(object(), None, StringIO())
