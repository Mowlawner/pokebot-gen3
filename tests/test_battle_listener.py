import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from modules.memory import GameState
from modules.modes import FrameInfo
from modules.modes import BattleAction
from modules.modes._listeners import _battle_return_to_field_complete
from modules.modes._listeners import BattleListener
from modules.modes.util.lead_rotation import ensure_campaign_field_lead
from modules.battle_handler import _advance_after_party_wipe
from modules.battle_handler import handle_fainted_pokemon
from modules.battle_state import BattleOutcome, EncounterType


class BattleListenerLifecycleTests(unittest.TestCase):
    def test_campaign_field_lead_handoff_rotates_only_at_a_stable_boundary(self):
        emulator = SimpleNamespace()
        fake_context = SimpleNamespace(bot_mode="Campaign Progression", emulator=emulator)
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        strategy = SimpleNamespace(
            choose_field_lead=Mock(return_value=SimpleNamespace(selected_index=1, reason="test"))
        )
        field_context = SimpleNamespace(source="test", objective_id="reach_petalburg", route_maps=((0, 10), (0, 17)))
        menu_steps = iter(())

        with (
            patch("modules.modes.util.lead_rotation.context", fake_context),
            patch("modules.modes.util.lead_rotation.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes.util.lead_rotation.battle_is_active", return_value=False),
            patch(
                "modules.modes.util.lead_rotation.get_last_battle_outcome",
                return_value=BattleOutcome.Won,
            ),
            patch(
                "modules.modes.util.lead_rotation.get_global_script_context",
                return_value=SimpleNamespace(is_active=False),
            ),
            patch("modules.modes.util.lead_rotation.player_avatar_is_standing_still", return_value=True),
            patch("modules.modes.util.lead_rotation.get_party", return_value=party),
            patch("modules.menuing.get_party", return_value=(object(), object())),
            patch(
                "modules.modes.util.lead_rotation.MenuWrapper",
                return_value=SimpleNamespace(step=lambda: menu_steps),
            ) as menu,
            patch("modules.modes.util.lead_rotation.diagnostic_print"),
        ):
            list(ensure_campaign_field_lead(strategy, field_context, reason="recovery_completed"))

        strategy.choose_field_lead.assert_called_once_with(field_context)
        menu.assert_called_once()

    def test_post_battle_rotation_waits_for_standing_field(self):
        listener = BattleListener()
        fake_context = SimpleNamespace(bot_mode="Campaign Progression", controller_stack=[])
        strategy = SimpleNamespace(choose_new_lead_after_battle=Mock(return_value=1))

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_game_state", side_effect=(GameState.OVERWORLD,) * 3),
            patch(
                "modules.modes._listeners.get_global_script_context",
                side_effect=(
                    SimpleNamespace(is_active=True),
                    SimpleNamespace(is_active=False),
                    SimpleNamespace(is_active=False),
                ),
            ),
            patch("modules.modes._listeners.player_avatar_is_standing_still", side_effect=(False, True)),
            patch.object(listener, "rotate_lead_pokemon", return_value=iter(())) as rotate,
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            list(listener._post_battle_rotation(strategy, 0))

        strategy.choose_new_lead_after_battle.assert_called_once_with()
        rotate.assert_called_once_with(1, 0)

    def test_capture_completion_applies_level_balancing_rotation(self):
        listener = BattleListener()
        emulator = SimpleNamespace(reset_held_buttons=Mock(), press_button=Mock())
        capture_strategy = object()
        capture_factory = Mock(return_value=capture_strategy)
        fake_context = SimpleNamespace(
            bot_mode="Campaign Progression",
            controller_stack=[],
            emulator=emulator,
            bot_mode_instance=SimpleNamespace(capture_battle_strategy=capture_factory),
            config=SimpleNamespace(battle=SimpleNamespace(save_after_catching=False)),
        )
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes.util._util_helper.context", fake_context),
            patch("modules.modes._listeners.get_party", return_value=party),
            patch("modules.battle_strategies.default.get_party", return_value=party),
            patch("modules.modes._listeners.plugin_battle_started", return_value=iter(())),
            patch("modules.modes._listeners.CatchStrategy") as default_capture_strategy,
            patch("modules.modes._listeners.handle_battle", return_value=iter(())) as handle_battle,
            patch.object(listener, "_wait_until_battle_is_over", return_value=iter(())),
            patch(
                "modules.modes._listeners.get_game_state",
                side_effect=(GameState.BATTLE, GameState.OVERWORLD),
            ),
            patch(
                "modules.modes._listeners.get_global_script_context",
                return_value=SimpleNamespace(is_active=False),
            ),
            patch("modules.modes._listeners.player_avatar_is_standing_still", return_value=True),
            patch.object(listener, "rotate_lead_pokemon", return_value=iter(())) as rotate,
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            list(listener.catch())

        rotate.assert_not_called()
        capture_factory.assert_called_once_with()
        default_capture_strategy.assert_not_called()
        handle_battle.assert_called_once_with(capture_strategy)

    def test_run_away_completion_applies_level_balancing_rotation(self):
        listener = BattleListener()
        emulator = SimpleNamespace(reset_held_buttons=Mock(), press_button=Mock())
        fake_context = SimpleNamespace(bot_mode="Campaign Progression", controller_stack=[], emulator=emulator)
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes.util._util_helper.context", fake_context),
            patch("modules.modes._listeners.get_party", return_value=party),
            patch("modules.battle_strategies.default.get_party", return_value=party),
            patch("modules.modes._listeners.plugin_battle_started", return_value=iter(())),
            patch("modules.modes._listeners.handle_battle", return_value=iter(())),
            patch.object(listener, "_wait_until_battle_is_over", return_value=iter(())),
            patch(
                "modules.modes._listeners.get_game_state",
                side_effect=(GameState.BATTLE, GameState.OVERWORLD),
            ),
            patch(
                "modules.modes._listeners.get_global_script_context",
                return_value=SimpleNamespace(is_active=False),
            ),
            patch("modules.modes._listeners.player_avatar_is_standing_still", return_value=True),
            patch.object(listener, "rotate_lead_pokemon", return_value=iter(())) as rotate,
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            list(listener.run_away_from_battle())

        rotate.assert_not_called()

    def test_tutorial_battle_honors_explicit_fight_request(self):
        listener = BattleListener()
        bot_mode = SimpleNamespace(on_battle_started=lambda _encounter: BattleAction.Fight)
        fake_context = SimpleNamespace(bot_mode="Campaign Progression", controller_stack=[])
        frame = FrameInfo(1, GameState.BATTLE, [], [], [], None)

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_encounter_type", return_value=EncounterType.Tutorial),
            patch("modules.modes._listeners.get_battle_state", return_value=SimpleNamespace(type=0)),
            patch("modules.modes._listeners.get_opponent", return_value=object()),
            patch("modules.modes._listeners.get_game_state", return_value=GameState.BATTLE),
            patch("modules.modes._listeners.DefaultBattleStrategy", return_value=object()),
            patch.object(listener, "fight", return_value=iter(())) as fight,
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            listener.handle_frame(bot_mode, frame)
            listener.handle_frame(bot_mode, frame)

        fight.assert_called_once()
        self.assertEqual(len(fake_context.controller_stack), 1)

    def test_restored_party_selection_resumes_battle_without_replaying_start(self):
        listener = BattleListener()
        bot_mode = object()
        fake_context = SimpleNamespace(bot_mode="Campaign Progression", controller_stack=[])
        frame = FrameInfo(
            1,
            GameState.PARTY_MENU,
            ["EventScript_DoTrainerBattle"],
            ["task_handlechoosemoninput"],
            [],
            None,
        )

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_game_state", return_value=GameState.PARTY_MENU),
            patch("modules.modes._listeners.battle_is_active", return_value=True),
            patch("modules.modes._listeners.get_last_battle_outcome", return_value=BattleOutcome.InProgress),
            patch("modules.modes._listeners.DefaultBattleStrategy", return_value=object()),
            patch.object(listener, "fight", return_value=iter(())) as fight,
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            listener.handle_frame(bot_mode, frame)

        self.assertTrue(listener._in_battle)
        self.assertTrue(listener._reported_start_of_battle)
        self.assertEqual(listener._current_action.name, "Fight")
        fight.assert_called_once()
        self.assertEqual(len(fake_context.controller_stack), 1)

    def test_completed_battle_choose_task_is_not_restored_as_active_battle(self):
        listener = BattleListener()
        frame = FrameInfo(
            1,
            GameState.PARTY_MENU,
            ["EventScript_DoTrainerBattle"],
            ["task_handlechoosemoninput"],
            [],
            None,
        )

        with (
            patch("modules.modes._listeners.battle_is_active", return_value=False),
            patch("modules.modes._listeners.get_last_battle_outcome", return_value=BattleOutcome.Won),
        ):
            self.assertFalse(listener._is_restored_battle_party_selection(frame))

    def test_completed_battle_party_menu_is_dismissed_once_after_restart(self):
        listener = BattleListener()
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, frame=1)
        frame = FrameInfo(
            1,
            GameState.PARTY_MENU,
            [],
            ["task_handlechoosemoninput"],
            [],
            None,
        )

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.battle_is_active", return_value=False),
            patch("modules.modes._listeners.get_last_battle_outcome", return_value=BattleOutcome.Won),
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            self.assertTrue(listener._dismiss_stale_completed_battle_party_menu(frame))
            self.assertTrue(listener._dismiss_stale_completed_battle_party_menu(frame))

        emulator.press_button.assert_called_once_with("B")

    def test_stale_party_menu_cleanup_yields_to_post_battle_rotation(self):
        listener = BattleListener()
        listener._post_battle_rotation_active = True
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, frame=1)
        frame = FrameInfo(
            1,
            GameState.PARTY_MENU,
            [],
            ["task_handlechoosemoninput"],
            [],
            None,
        )

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            self.assertFalse(listener._dismiss_stale_completed_battle_party_menu(frame))

        emulator.press_button.assert_not_called()

    def test_stale_party_menu_cleanup_closes_owning_start_menu(self):
        listener = BattleListener()
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, frame=1)
        stale_party_frame = FrameInfo(
            1,
            GameState.PARTY_MENU,
            [],
            ["task_handlechoosemoninput"],
            [],
            None,
        )
        start_menu_frame = FrameInfo(
            2,
            GameState.PARTY_MENU,
            [],
            ["task_showstartmenu"],
            [],
            stale_party_frame,
        )
        start_menu_ready_frame = FrameInfo(
            3,
            GameState.PARTY_MENU,
            [],
            ["task_showstartmenu"],
            [],
            start_menu_frame,
        )
        transient_overworld_frame = FrameInfo(
            2,
            GameState.OVERWORLD,
            [],
            [],
            [],
            stale_party_frame,
        )
        overworld_frame = FrameInfo(
            3,
            GameState.OVERWORLD,
            [],
            [],
            [],
            start_menu_frame,
        )

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.battle_is_active", return_value=False),
            patch("modules.modes._listeners.get_last_battle_outcome", return_value=BattleOutcome.Won),
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            self.assertTrue(listener._dismiss_stale_completed_battle_party_menu(stale_party_frame))
            self.assertTrue(listener._dismiss_stale_completed_battle_party_menu(transient_overworld_frame))
            self.assertTrue(listener._dismiss_stale_completed_battle_party_menu(start_menu_frame))
            self.assertTrue(listener._dismiss_stale_completed_battle_party_menu(start_menu_ready_frame))
            self.assertFalse(listener._dismiss_stale_completed_battle_party_menu(overworld_frame))

        self.assertEqual(emulator.press_button.call_args_list, [call("B"), call("B")])

    def test_return_to_field_gate_does_not_require_map_objects_or_avatar(self):
        frame = SimpleNamespace(active_tasks=[])

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_OVERWORLD"),
            patch("modules.modes._listeners.get_global_script_context", return_value=None),
        ):
            self.assertTrue(_battle_return_to_field_complete(frame))

    def test_return_to_field_gate_waits_for_return_callback(self):
        frame = SimpleNamespace(active_tasks=[])

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_RETURNTOFIELD"),
            patch("modules.modes._listeners.get_global_script_context", return_value=None),
        ):
            self.assertFalse(_battle_return_to_field_complete(frame))

    def test_return_to_field_gate_waits_for_return_task(self):
        frame = SimpleNamespace(active_tasks=["Task_ReturnToFieldNoScript"])

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_OVERWORLD"),
            patch("modules.modes._listeners.get_global_script_context", return_value=None),
        ):
            self.assertFalse(_battle_return_to_field_complete(frame))

    def test_return_to_field_gate_releases_before_post_battle_script_finishes(self):
        frame = SimpleNamespace(active_tasks=[])
        script = SimpleNamespace(is_active=True, native_function_name="WaitForMovementFinish")

        with (
            patch("modules.modes._listeners.get_game_state_symbol", return_value="CB2_OVERWORLD"),
            patch("modules.modes._listeners.get_global_script_context", return_value=script),
        ):
            self.assertTrue(_battle_return_to_field_complete(frame))

    def test_post_battle_wait_does_not_press_b_in_overworld_field_script(self):
        listener = BattleListener()
        listener._in_battle = True
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, frame=1, bot_mode="Campaign Progression")

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            wait = listener._wait_until_battle_is_over()
            next(wait)

        emulator.press_button.assert_not_called()

    def test_post_battle_wait_advances_each_ready_field_message_once(self):
        listener = BattleListener()
        listener._in_battle = True
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, frame=1, bot_mode="Campaign Progression")
        script = SimpleNamespace(
            is_active=True,
            native_function_name="WaitForAorBPress",
            script_function_name="Std_MsgboxDefault",
        )

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.modes._listeners.get_global_script_context", return_value=script),
            patch("modules.modes._listeners.is_field_message_task_waiting_for_input", return_value=False),
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            wait = listener._wait_until_battle_is_over()
            next(wait)
            next(wait)

        emulator.press_button.assert_called_once_with("A")

    def test_post_battle_wait_advances_battle_transition_state(self):
        listener = BattleListener()
        listener._in_battle = True
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, frame=1, bot_mode="Campaign Progression")

        with (
            patch("modules.modes._listeners.context", fake_context),
            patch("modules.modes._listeners.get_game_state", return_value=GameState.BATTLE_ENDING),
            patch("modules.modes._listeners.diagnostic_print"),
        ):
            wait = listener._wait_until_battle_is_over()
            next(wait)

        emulator.press_button.assert_called_once_with("B")

    def test_party_wipe_advances_choose_mon_task_with_confirm(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator)

        with (
            patch("modules.battle_handler.context", fake_context),
            patch("modules.battle_handler.battle_is_active", side_effect=(True, False)),
            patch("modules.battle_handler.get_game_state", return_value=GameState.PARTY_MENU),
            patch(
                "modules.battle_handler.task_is_active", side_effect=lambda name: name == "Task_HandleChooseMonInput"
            ),
        ):
            transition = _advance_after_party_wipe()
            next(transition)
            with self.assertRaises(StopIteration):
                next(transition)

        emulator.press_button.assert_called_once_with("A")

    def test_strategy_can_declare_a_nuzlocke_party_wipe_with_rom_healthy_pokemon(self):
        active = SimpleNamespace(party_index=0, current_hp=0)
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(left_battler=active, right_battler=None, active_battlers=[active]),
            is_double_battle=False,
        )

        class FakeParty:
            non_fainted_pokemon = [SimpleNamespace(name="healed-dead", current_hp=10, is_egg=False)]

            def __iter__(self):
                return iter(self.non_fainted_pokemon)

        party = FakeParty()
        strategy = Mock()
        strategy.has_replacement_after_faint.return_value = False

        with (
            patch("modules.battle_handler.get_battle_state", return_value=battle_state),
            patch("modules.battle_handler.get_party", return_value=party),
            patch("modules.battle_handler.diagnostic_print"),
            patch("modules.battle_handler._advance_after_party_wipe", return_value=iter((None,))) as advance,
        ):
            self.assertEqual(list(handle_fainted_pokemon(strategy)), [None])

        strategy.has_replacement_after_faint.assert_called_once_with(battle_state)
        advance.assert_called_once_with()

    def test_faint_selection_reconciles_stale_battle_order_with_visible_party(self):
        fainted = SimpleNamespace(current_hp=0, party_index=0)
        selected = SimpleNamespace(name="Kathy", current_hp=7, is_egg=False)
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(left_battler=fainted, right_battler=None, active_battlers=[fainted]),
            is_double_battle=False,
            is_trainer_battle=True,
            map_battle_party_index=lambda _index: 0,
            _battler_party_order=bytearray(b"\x10#E"),
        )

        class FakeParty:
            non_fainted_pokemon = [selected]

            def __iter__(self):
                return iter((SimpleNamespace(name="Torchic", current_hp=0, is_egg=False), selected))

            def __getitem__(self, index):
                return tuple(self)[index]

            def get_index_for_pokemon(self, pokemon):
                return 1 if pokemon is selected else 0

        strategy = Mock()
        strategy.has_replacement_after_faint.return_value = True
        strategy.choose_new_lead_after_faint.return_value = 1

        with (
            patch("modules.battle_handler.get_battle_state", return_value=battle_state),
            patch("modules.battle_handler.get_party", return_value=FakeParty()),
            patch("modules.battle_handler.get_party_size", return_value=2),
            patch(
                "modules.battle_handler.get_game_state",
                side_effect=(GameState.BATTLE, GameState.PARTY_MENU, GameState.OVERWORLD),
            ),
            patch(
                "modules.battle_handler.task_is_active", side_effect=lambda name: name == "Task_HandleChooseMonInput"
            ),
            patch(
                "modules.battle_handler.get_current_battle_script_instruction",
                return_value="BattleScript_HandleFaintedMon",
            ),
            patch("modules.battle_handler.battle_is_active", return_value=True),
            patch("modules.battle_handler.get_current_party_menu_index", return_value=0),
            patch("modules.battle_handler.context", SimpleNamespace(bot_mode="Campaign Progression", emulator=Mock())),
            patch("modules.battle_handler.scroll_to_party_menu_index", return_value=iter(())) as scroll,
            patch("modules.battle_handler.diagnostic_print"),
        ):
            list(handle_fainted_pokemon(strategy))

        scroll.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
