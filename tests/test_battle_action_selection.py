from unittest.mock import Mock, call, patch
from types import SimpleNamespace

from modules.battle_action_selection import (
    _log_battle_action_chosen,
    _resolve_battle_party_menu_index,
    battle_action_selection_is_ready,
    handle_battle_action_selection,
)
from modules.memory import GameState
from modules.battle_strategies import TurnAction
from modules.context import context


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
