from unittest.mock import Mock, call, patch
from types import SimpleNamespace

from modules.battle_action_selection import (
    _battle_move_is_executable,
    _BattleProgressGuard,
    _log_battle_action_chosen,
    _battle_action_details,
    _log_trainer_replacement_choice,
    _resolve_battle_party_menu_index,
    battle_action_use_item,
    battle_action_selection_is_ready,
    handle_battle_action_selection,
)
from modules.memory import GameState
from modules.battle_strategies import TurnAction
from modules.battle_state import BattleType
from modules.context import context
from modules.items import get_item_by_name


def test_move_boundary_rejects_exhausted_or_disabled_snapshot():
    move = SimpleNamespace(name="Scratch")
    active = SimpleNamespace(
        current_hp=10,
        moves=(SimpleNamespace(move=move, pp=0),),
        disabled_move=None,
    )
    state = SimpleNamespace(own_side=SimpleNamespace(active_battler=active))
    assert _battle_move_is_executable(state, 0, 0) == (False, "Scratch has no PP remaining")

    active.moves = (SimpleNamespace(move=move, pp=1),)
    active.disabled_move = move
    active.disabled_turns_remaining = 2
    valid, reason = _battle_move_is_executable(state, 0, 0)
    assert valid is False
    assert reason == "Scratch is disabled"


def test_progress_guard_stops_only_after_repeated_identical_action_and_state():
    guard = _BattleProgressGuard(limit=2)
    assert guard.observe(("hp", 10), "UseMove", 0) is False
    assert guard.observe(("hp", 10), "UseMove", 0) is False
    assert guard.observe(("hp", 10), "UseMove", 0) is True
    guard.reset()
    assert guard.observe(("hp", 9), "UseMove", 0) is False


def test_selected_battle_action_is_printed_to_the_rich_console():
    active = SimpleNamespace(
        species=SimpleNamespace(name="Treecko"),
        moves=(SimpleNamespace(move=SimpleNamespace(name="Leer")),),
    )
    opponent = SimpleNamespace(species=SimpleNamespace(name="Poochyena"))
    battle_state = SimpleNamespace(
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    previous_source = context.battle_decision_source
    previous_detail = context.battle_decision_detail
    context.battle_decision_source = "TEST POLICY"
    context.battle_decision_detail = "selected stat setup"

    try:
        with (
            patch("modules.battle_action_selection.print_battle_action") as rich_console,
            patch("modules.battle_action_selection.diagnostic_print"),
        ):
            _log_battle_action_chosen(battle_state, SimpleNamespace(), TurnAction.UseMove, 0, 0)

        printed = rich_console.call_args.kwargs
        assert printed["action"] == "UseMove(Leer)"
        assert printed["reason"] == "selected stat setup"
    finally:
        context.battle_decision_source = previous_source
        context.battle_decision_detail = previous_detail


def test_trainer_replacement_log_shows_incoming_opponent_and_policy_reason():
    current = SimpleNamespace(party_index=0, species=SimpleNamespace(name="Mudkip"))
    defeated = SimpleNamespace(party_index=0, species=SimpleNamespace(name="Zigzagoon"))
    incoming = SimpleNamespace(
        index=1,
        is_empty=False,
        is_valid=True,
        current_hp=10,
        species=SimpleNamespace(name="Taillow"),
    )
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=current),
        opponent=SimpleNamespace(active_battler=defeated),
    )
    previous_source = context.battle_decision_source
    previous_detail = context.battle_decision_detail
    context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
    context.battle_decision_detail = "current Pokémon is the safest eligible response"

    try:
        with (
            patch("modules.battle_action_selection.get_opponent_party", return_value=(incoming,)),
            patch("modules.battle_action_selection.print_battle_action") as rich_console,
            patch("modules.battle_action_selection.diagnostic_print"),
        ):
            _log_trainer_replacement_choice(battle_state, None, "TRAINER REPLACEMENT POLICY")

        printed = rich_console.call_args.kwargs
        assert printed["opponent"] == "Taillow"
        assert printed["source"] == "NUZLOCKE TRAINER SWITCH"
        assert "incoming opponent=Taillow" in printed["reason"]
        assert "previous opponent=Zigzagoon" in printed["reason"]
    finally:
        context.battle_decision_source = previous_source
        context.battle_decision_detail = previous_detail


def test_battle_party_menu_target_prefers_visible_identity_over_stale_order():
    """A stale packed order must not make a Lotad decision select Treecko."""

    target = SimpleNamespace(name="Lotad")
    visible_party = SimpleNamespace(get_index_for_pokemon=Mock(return_value=1))
    battle_state = SimpleNamespace(
        _battler_party_order=bytes((0x10, 0x23, 0x45)),
        map_battle_party_index=Mock(return_value=0),
    )

    with (
        patch("modules.battle_action_selection.get_party", return_value=visible_party),
        patch("modules.battle_action_selection.get_current_party_menu_index", return_value=0),
        patch("modules.battle_action_selection.diagnostic_print"),
    ):
        assert _resolve_battle_party_menu_index(battle_state, 1, target, mapped_index=0) == 1

    visible_party.get_index_for_pokemon.assert_called_once_with(target)


def test_rotate_lead_uses_visible_party_identity_at_menu_boundary():
    """The executor must scroll to Lotad even when the packed map says slot 0."""

    treecko = SimpleNamespace(name="Treecko", data=b"tree")
    lotad = SimpleNamespace(name="Lotad", data=b"lotad")

    class FakeParty:
        def __getitem__(self, index):
            return (treecko, lotad)[index]

        def get_index_for_pokemon(self, pokemon):
            return (treecko, lotad).index(pokemon)

    active = SimpleNamespace(party_index=0, current_hp=20, species=SimpleNamespace(name="Treecko"))
    opponent = SimpleNamespace(current_hp=10, species=SimpleNamespace(name="Shroomish"))
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_safari_zone_encounter=False,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
        _battler_party_order=bytes((0x10, 0x23, 0x45)),
        map_battle_party_index=Mock(return_value=0),
    )
    strategy = SimpleNamespace(decide_turn=Mock(return_value=(TurnAction.RotateLead, 1)))
    emulator = Mock()
    previous_bot_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"

    try:
        with (
            patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
            patch("modules.battle_action_selection.battle_is_active", side_effect=(True, False)),
            patch(
                "modules.battle_action_selection.get_main_battle_callback",
                return_value="HandleTurnActionSelectionState",
            ),
            patch("modules.battle_action_selection.get_battle_controller_callback", return_value="HandleInputChooseAction"),
            patch("modules.battle_action_selection.get_party", return_value=FakeParty()),
            patch("modules.battle_action_selection.get_party_size", return_value=2),
            patch(
                "modules.battle_action_selection.get_game_state",
                side_effect=(GameState.PARTY_MENU, GameState.OVERWORLD),
            ),
            patch("modules.battle_action_selection.get_current_party_menu_index", return_value=0),
            patch("modules.battle_action_selection.scroll_to_battle_action", return_value=iter(())),
            patch("modules.battle_action_selection.scroll_to_party_menu_index", return_value=iter(())) as scroll,
            patch("modules.battle_action_selection._trace_battle_replacement_boundary"),
            patch("modules.battle_action_selection.context.emulator", emulator),
        ):
            list(handle_battle_action_selection(strategy))
    finally:
        context.bot_mode = previous_bot_mode

    scroll.assert_called_once_with(1)


def test_battle_action_handler_refreshes_state_after_using_item():
    """A completed Potion must be evaluated from the post-use HP snapshot."""

    potion = get_item_by_name("Potion")
    low_active = SimpleNamespace(
        party_index=0,
        current_hp=4,
        total_hp=20,
        species=SimpleNamespace(name="Shroomish"),
    )
    healed_active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        total_hp=20,
        species=SimpleNamespace(name="Shroomish"),
    )
    opponent = SimpleNamespace(
        party_index=1,
        current_hp=10,
        total_hp=10,
        species=SimpleNamespace(name="Poochyena"),
    )
    low_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        is_safari_zone_encounter=False,
        own_side=SimpleNamespace(active_battler=low_active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    healed_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        is_safari_zone_encounter=False,
        own_side=SimpleNamespace(active_battler=healed_active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    strategy = SimpleNamespace(
        decide_turn=Mock(
            side_effect=((TurnAction.UseItem, (potion, 0)), (TurnAction.SwitchToManual, None))
        )
    )
    previous_bot_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"

    try:
        with (
            patch("modules.battle_action_selection.get_battle_state", side_effect=(low_state, healed_state)) as read_state,
            patch("modules.battle_action_selection.battle_is_active", side_effect=(True, True, False)),
            patch("modules.battle_action_selection.get_main_battle_callback", return_value="HandleTurnActionSelectionState"),
            patch("modules.battle_action_selection.get_battle_controller_callback", return_value="HandleInputChooseAction"),
            patch("modules.battle_action_selection.battle_action_use_item", return_value=iter(())) as use_item,
            patch("modules.battle_action_selection.context.set_manual_mode"),
            patch("modules.battle_action_selection.diagnostic_print"),
        ):
            list(handle_battle_action_selection(strategy))
    finally:
        context.bot_mode = previous_bot_mode

    assert read_state.call_count == 2
    use_item.assert_called_once_with(low_state, potion, 0)
    assert strategy.decide_turn.call_count == 2
    assert strategy.decide_turn.call_args_list[1].args == (healed_state,)


def test_campaign_planner_retry_does_not_enter_manual_mode():
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_safari_zone_encounter=False,
    )
    strategy = SimpleNamespace(
        decide_turn=Mock(return_value=(TurnAction.SwitchToManual, None))
    )
    previous_bot_mode = context.bot_mode
    previous_retry_requested = getattr(context, "battle_planner_retry_requested", False)
    previous_retry_count = getattr(context, "battle_planner_retry_count", 0)
    context.bot_mode = "Campaign Progression"
    context.battle_planner_retry_requested = True
    context.battle_planner_retry_count = 1

    try:
        with (
            patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
            patch("modules.battle_action_selection.battle_is_active", side_effect=(True, False)),
            patch(
                "modules.battle_action_selection.get_main_battle_callback",
                return_value="HandleTurnActionSelectionState",
            ),
            patch(
                "modules.battle_action_selection.get_battle_controller_callback",
                return_value="HandleInputChooseAction",
            ),
            patch("modules.battle_action_selection.context.set_manual_mode") as set_manual,
            patch("modules.battle_action_selection.diagnostic_print"),
        ):
            list(handle_battle_action_selection(strategy))
    finally:
        context.bot_mode = previous_bot_mode
        context.battle_planner_retry_requested = previous_retry_requested
        context.battle_planner_retry_count = previous_retry_count

    set_manual.assert_not_called()
    strategy.decide_turn.assert_called_once_with(battle_state)


def test_battle_item_executor_uses_fresh_confirmations_and_waits_for_menus():
    potion = get_item_by_name("Potion")

    class FakeEmulator:
        def __init__(self):
            self.fresh = []
            self.plain = []

        def press_button_fresh(self, button):
            self.fresh.append(button)

        def press_button(self, button):
            self.plain.append(button)

    emulator = FakeEmulator()
    battle_state = SimpleNamespace(map_battle_party_index=lambda index: index)

    previous_emulator = context.emulator
    previous_profile = context.profile
    context.emulator = emulator
    context.profile = SimpleNamespace(rom=SimpleNamespace(is_rse=True))
    try:
        with (
            patch("modules.battle_action_selection.scroll_to_battle_action", return_value=iter(())),
            patch("modules.battle_action_selection.scroll_to_item_in_bag", return_value=iter(())),
            patch("modules.battle_action_selection._capture_party_member", return_value=SimpleNamespace(name="Wurmple")),
            patch("modules.battle_action_selection._scroll_to_battle_party_member", return_value=iter(())),
            patch(
                "modules.battle_action_selection.get_game_state",
                side_effect=(GameState.BATTLE, GameState.BAG_MENU, GameState.PARTY_MENU),
            ),
            patch(
                "modules.battle_action_selection.get_game_state_symbol",
                return_value="BATTLEMAINCB2",
            ),
            patch("modules.battle_action_selection.battle_is_active", return_value=True),
        ):
            list(battle_action_use_item(battle_state, potion, 0))
    finally:
        context.emulator = previous_emulator
        context.profile = previous_profile

    assert emulator.fresh == ["A", "A", "A", "A"]


def test_campaign_rejected_switch_replans_without_substituting_a_legacy_move():
    """A stale campaign switch must return to the planner at the turn boundary."""

    move = SimpleNamespace(name="Scratch")
    active = SimpleNamespace(
        party_index=0,
        current_hp=10,
        moves=(SimpleNamespace(move=move, pp=1),),
        disabled_move=None,
        species=SimpleNamespace(name="Torchic"),
    )
    dead_target = SimpleNamespace(
        party_index=1,
        current_hp=0,
        is_egg=False,
        species=SimpleNamespace(name="Lotad"),
    )
    opponent = SimpleNamespace(
        party_index=0,
        current_hp=10,
        species=SimpleNamespace(name="Poochyena"),
    )
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        is_safari_zone_encounter=False,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active, active_battlers=(active,)),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    strategy = SimpleNamespace(
        requires_planner_owned_actions=Mock(return_value=True),
        decide_turn=Mock(side_effect=((TurnAction.RotateLead, 1), (TurnAction.UseMove, 0))),
    )
    previous_bot_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"

    try:
        with (
            patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
            patch("modules.battle_action_selection.battle_is_active", side_effect=(True, True, False)),
            patch("modules.battle_action_selection.battle_action_selection_is_ready", return_value=True),
            patch(
                "modules.battle_action_selection.get_battle_controller_callback",
                return_value="HandleInputChooseAction",
            ),
            patch("modules.battle_action_selection.get_main_battle_callback", return_value="BattleMainCB2"),
            patch("modules.battle_action_selection.get_party", return_value=(active, dead_target)),
            patch("modules.battle_action_selection.get_party_size", return_value=2),
            patch("modules.battle_action_selection.battle_action_use_move", return_value=iter(())) as use_move,
            patch("modules.battle_action_selection._log_battle_action_chosen"),
            patch("modules.battle_action_selection._log_battle_action_lifecycle"),
            patch("modules.battle_action_selection._trace_battle_replacement_boundary"),
            patch("modules.battle_action_selection.diagnostic_print"),
        ):
            list(handle_battle_action_selection(strategy))
    finally:
        context.bot_mode = previous_bot_mode

    assert strategy.decide_turn.call_count == 2
    use_move.assert_called_once_with(TurnAction.UseMove, 0, 0, battle_state)


def test_trainer_switch_prompt_can_select_strategy_replacement():
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
        map_battle_party_index=lambda index: index + 1,
    )
    strategy = SimpleNamespace(choose_trainer_replacement=Mock(return_value=1))
    emulator = Mock()

    with (
        patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
        patch("modules.battle_action_selection.battle_is_active", side_effect=(True, False)),
        patch(
            "modules.battle_action_selection.get_main_battle_callback",
            return_value="HandleTurnActionSelectionState",
        ),
        patch(
            "modules.battle_action_selection.get_battle_controller_callback",
            return_value="PlayerHandleYesNoBox",
        ),
        patch("modules.battle_action_selection.get_party_size", return_value=2),
        patch(
            "modules.battle_action_selection.get_game_state",
            side_effect=(GameState.PARTY_MENU, GameState.OVERWORLD),
        ),
        patch("modules.battle_action_selection.scroll_to_party_menu_index", return_value=iter(())),
        patch("modules.battle_action_selection.context.emulator", emulator),
    ):
        list(handle_battle_action_selection(strategy))

    strategy.choose_trainer_replacement.assert_called_once_with(battle_state)
    assert emulator.press_button.call_args_list == [call("Up"), call("A"), call("A")]


def test_trainer_switch_prompt_answers_no_when_strategy_has_no_replacement():
    battle_state = SimpleNamespace(is_double_battle=False, is_trainer_battle=True)
    strategy = SimpleNamespace(choose_trainer_replacement=Mock(return_value=None))
    emulator = Mock()

    with (
        patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
        patch("modules.battle_action_selection.battle_is_active", side_effect=(True, False)),
        patch(
            "modules.battle_action_selection.get_main_battle_callback",
            return_value="HandleTurnActionSelectionState",
        ),
        patch(
            "modules.battle_action_selection.get_battle_controller_callback",
            return_value="PlayerHandleYesNoBox",
        ),
        patch("modules.battle_action_selection.context.emulator", emulator),
    ):
        list(handle_battle_action_selection(strategy))

    strategy.choose_trainer_replacement.assert_called_once_with(battle_state)
    emulator.press_button.assert_called_once_with("B")


def test_trainer_switch_prompt_is_handled_during_non_action_main_callback():
    """The replacement prompt must not fall through to battle's B fallback."""

    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
        map_battle_party_index=lambda index: index + 1,
    )
    strategy = SimpleNamespace(choose_trainer_replacement=Mock(return_value=1))
    emulator = Mock()

    with (
        patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
        patch("modules.battle_action_selection.battle_is_active", side_effect=(True, False)),
        patch(
            "modules.battle_action_selection.get_main_battle_callback",
            return_value="BattleMainCB2",
        ),
        patch(
            "modules.battle_action_selection.get_battle_controller_callback",
            return_value="PlayerHandleYesNoBox",
        ),
        patch("modules.battle_action_selection.get_party_size", return_value=2),
        patch(
            "modules.battle_action_selection.get_game_state",
            side_effect=(GameState.PARTY_MENU, GameState.OVERWORLD),
        ),
        patch("modules.battle_action_selection.scroll_to_party_menu_index", return_value=iter(())),
        patch("modules.battle_action_selection.context.emulator", emulator),
    ):
        list(handle_battle_action_selection(strategy))

    strategy.choose_trainer_replacement.assert_called_once_with(battle_state)
    assert emulator.press_button.call_args_list == [call("Up"), call("A"), call("A")]


def test_trainer_switch_prompt_is_ready_while_rom_is_rendering_message():
    """The transient render state must not fall through to battle's B fallback."""

    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp=8)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=0)),
    )

    with (
        patch("modules.battle_action_selection.get_main_battle_callback", return_value="BattleMainCB2"),
        patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
        patch("modules.battle_action_selection.get_battle_controller_callback", return_value="PlayerBufferRunCommand"),
        patch(
            "modules.battle_action_selection.read_symbol",
            return_value=b"BUG CATCHER RICK is about to use WURMPLE. Will Bruce change POKEMON?\xff",
        ),
        patch(
            "modules.battle_action_selection.decode_string",
            return_value="BUG CATCHER RICK is about to use WURMPLE. Will Bruce change POKEMON?",
        ),
    ):
        assert battle_action_selection_is_ready(battle_state) is True


def test_trainer_switch_prompt_not_ready_for_fainted_player_or_live_opponent():
    base_state = dict(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp=8)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=0)),
    )

    with (
        patch("modules.battle_action_selection.get_main_battle_callback", return_value="BattleMainCB2"),
        patch("modules.battle_action_selection.get_battle_controller_callback", return_value="PlayerBufferRunCommand"),
        patch("modules.battle_action_selection.decode_string", return_value="Trainer is about to use WURMPLE. Will Bruce change POKEMON?"),
        patch("modules.battle_action_selection.read_symbol", return_value=b"prompt"),
    ):
        fainted_player = SimpleNamespace(**base_state)
        fainted_player.own_side = SimpleNamespace(active_battler=SimpleNamespace(current_hp=0))
        assert battle_action_selection_is_ready(fainted_player) is False

        live_opponent = SimpleNamespace(**base_state)
        live_opponent.opponent = SimpleNamespace(active_battler=SimpleNamespace(current_hp=1))
        assert battle_action_selection_is_ready(live_opponent) is False


def test_trainer_switch_prompt_advances_text_before_waiting_for_yes_no_controller():
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp=8)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=0)),
    )
    emulator = Mock()

    with (
        patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
        patch("modules.battle_action_selection.battle_is_active", side_effect=(True, False)),
        patch("modules.battle_action_selection.get_main_battle_callback", return_value="BattleMainCB2"),
        patch(
            "modules.battle_action_selection.get_battle_controller_callback",
            return_value="CompleteOnInactiveTextPrinter2",
        ),
        patch("modules.battle_action_selection.read_symbol", return_value=b"prompt"),
        patch(
            "modules.battle_action_selection.decode_string",
            return_value="Trainer is about to use WURMPLE. Will Bruce change POKEMON?",
        ),
        patch("modules.battle_action_selection.context.emulator", emulator),
    ):
        list(handle_battle_action_selection(SimpleNamespace()))

    emulator.press_button.assert_called_once_with("B")


def test_battle_action_log_explains_capture_weakening_move():
    move = SimpleNamespace(name="Absorb")
    active = SimpleNamespace(
        species=SimpleNamespace(name="Treecko"),
        moves=(SimpleNamespace(move=move),),
    )
    opponent = SimpleNamespace(species=SimpleNamespace(name="Wurmple"))
    battle_state = SimpleNamespace(
        nuzlocke_capture_target=True,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    strategy = SimpleNamespace()
    context.battle_decision_source = "NUZLOCKE CAPTURE"
    context.battle_decision_detail = "selected nonlethal move"

    _log_battle_action_chosen(
        battle_state,
        strategy,
        TurnAction.UseMove,
        0,
        0,
    )

    assert "objective=capture" in context.message
    assert "target=lower HP" in context.message
    assert "intent=select move" in context.message
    assert "Absorb" in context.message
    assert "selected nonlethal move" in context.message


def test_battle_action_display_uses_defeat_for_first_battle():
    active = SimpleNamespace(
        species=SimpleNamespace(name="Treecko"),
        moves=(SimpleNamespace(move=SimpleNamespace(name="Pound")),),
    )
    opponent = SimpleNamespace(species=SimpleNamespace(name="Zigzagoon"))
    battle_state = SimpleNamespace(
        type=BattleType.FirstBattle,
        is_trainer_battle=False,
        nuzlocke_capture_target=True,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    previous_intent = context.campaign_wild_battle_intent
    context.campaign_wild_battle_intent = "transit"
    try:
        with (
            patch(
                "modules.battle_action_selection.get_item_bag",
                return_value=SimpleNamespace(
                    poke_balls=(SimpleNamespace(quantity=3),),
                ),
            ),
            patch("modules.battle_action_selection.get_party", return_value=(active,)),
            patch("modules.battle_action_selection.get_opponent_party", return_value=(opponent,)),
        ):
            details = _battle_action_details(
                battle_state,
                SimpleNamespace(),
                TurnAction.UseMove,
                0,
                0,
            )
    finally:
        context.campaign_wild_battle_intent = previous_intent

    assert details[0] == "defeat"


def test_trainer_switch_prompt_can_navigate_emerald_buffer_to_party_menu():
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp=8)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=0)),
        map_battle_party_index=lambda index: index + 1,
    )
    strategy = SimpleNamespace(choose_trainer_replacement=Mock(return_value=1))
    emulator = Mock()

    with (
        patch("modules.battle_action_selection.get_battle_state", return_value=battle_state),
        patch("modules.battle_action_selection.battle_is_active", side_effect=(True, True, True, False)),
        patch("modules.battle_action_selection.get_main_battle_callback", return_value="BattleMainCB2"),
        patch(
            "modules.battle_action_selection.get_battle_controller_callback",
            side_effect=(
                "PlayerBufferRunCommand",
                "PlayerBufferRunCommand",
                "PlayerBufferRunCommand",
                "PlayerBufferRunCommand",
                "OpenPartyMenuToChooseMon",
                "OpenPartyMenuToChooseMon",
            ),
        ),
        patch("modules.battle_action_selection.read_symbol", return_value=b"prompt"),
        patch(
            "modules.battle_action_selection.decode_string",
            return_value="Trainer is about to use WURMPLE. Will Bruce change POKEMON?",
        ),
        patch("modules.battle_action_selection.get_party_size", return_value=2),
        patch(
            "modules.battle_action_selection.get_game_state",
            side_effect=(GameState.PARTY_MENU, GameState.OVERWORLD),
        ),
        patch("modules.battle_action_selection.scroll_to_party_menu_index", return_value=iter(())),
        patch("modules.battle_action_selection._trace_battle_replacement_boundary"),
        patch("modules.battle_action_selection.context.emulator", emulator),
    ):
        list(handle_battle_action_selection(strategy))

    strategy.choose_trainer_replacement.assert_called_once_with(battle_state)
    assert emulator.press_button.call_args_list == [call("Up"), call("A"), call("A")]
