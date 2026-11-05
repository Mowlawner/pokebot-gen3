from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from modules.battle_handler import (
    _BATTLE_SETUP_TRANSITION_TIMEOUT,
    _WALLY_TUTORIAL_SETUP_TRANSITION_TIMEOUT,
    _battle_setup_transition_timeout,
    _print_battle_start,
    _print_battle_events,
    _print_battle_end,
    handle_battle,
)
from modules.battle_observation import DamageOccurred, PokemonFainted, PokemonSwitched
from modules.battle_state import BattleType
from modules.context import context
from modules.modes._interface import BotModeError


def _strict_strategy():
    return SimpleNamespace(allows_legacy_fallback=lambda: False)


@pytest.mark.parametrize(
    "setup_callback",
    [
        "BeginBattleIntro",
        "BattleTurnPassed",
        "BattleIntroGetMonsData",
        "SetActionsAndBattlersTurnOrder",
        "CheckFocusPunch_ClearVarsBeforeTurnStarts",
        "FreeResetData_ReturnToOvOrDoEvolutions",
        "RunTurnActionsFunctions",
        "HandleAction_UseMove",
        "HandleEndTurn_ContinueBattle",
        "ReturnFromBattleToOverworld",
        "TryEvolvePokemon",
        "WaitForEvoSceneToFinish",
        "TryDoEventsBeforeFirstTurn",
    ],
)
def test_campaign_waits_through_first_turn_setup_before_dispatching_action(setup_callback):
    """The opening Emerald callback is ROM-owned until the action menu exists."""

    emulator = Mock()
    player = SimpleNamespace(money=0)
    previous_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"

    try:
        with (
            patch("modules.battle_handler.battle_is_active", side_effect=(True, True, False)),
            patch("modules.battle_handler.observe_current_battle"),
            patch("modules.battle_handler.get_main_battle_callback", return_value=setup_callback),
            patch("modules.battle_handler.get_current_battle_script_instruction", return_value="BattleScript_None"),
            patch("modules.battle_handler.battle_action_selection_is_ready", side_effect=(False, True)),
            patch("modules.battle_handler.handle_battle_action_selection", return_value=iter(())) as select_action,
            patch("modules.battle_handler.task_is_active", return_value=False),
            patch("modules.battle_handler.get_player", return_value=player),
            patch("modules.battle_handler.get_party", return_value=[]),
            patch("modules.battle_handler.get_encounter_type", return_value=None),
            patch("modules.battle_handler.get_last_battle_outcome", return_value=None),
            patch("modules.battle_handler.diagnostic_print"),
            patch.object(context, "emulator", emulator),
        ):
            list(handle_battle(_strict_strategy()))

        select_action.assert_called_once()
        emulator.press_button.assert_called_once_with("A")
    finally:
        context.bot_mode = previous_mode


def test_campaign_still_stops_on_unknown_unhandled_battle_state():
    previous_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"
    player = SimpleNamespace(money=0)

    try:
        with (
            patch("modules.battle_handler.battle_is_active", return_value=True),
            patch("modules.battle_handler.observe_current_battle"),
            patch("modules.battle_handler.get_main_battle_callback", return_value="UnknownBattleCallback"),
            patch("modules.battle_handler.get_current_battle_script_instruction", return_value="UnknownScript"),
            patch("modules.battle_handler.battle_action_selection_is_ready", return_value=False),
            patch("modules.battle_handler.task_is_active", return_value=False),
            patch("modules.battle_handler.get_player", return_value=player),
            patch("modules.battle_handler.get_party", return_value=[]),
            patch("modules.battle_handler.get_encounter_type", return_value=None),
            patch("modules.battle_handler.diagnostic_print"),
            patch.object(context, "emulator", Mock()),
        ):
            with pytest.raises(BotModeError, match="unhandled battle state"):
                list(handle_battle(_strict_strategy()))
    finally:
        context.bot_mode = previous_mode


def test_campaign_times_out_if_first_turn_setup_never_completes():
    previous_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"
    player = SimpleNamespace(money=0)

    try:
        with (
            patch("modules.battle_handler.battle_is_active", return_value=True),
            patch("modules.battle_handler.observe_current_battle"),
            patch("modules.battle_handler.get_main_battle_callback", return_value="TryDoEventsBeforeFirstTurn"),
            patch("modules.battle_handler.get_current_battle_script_instruction", return_value="BattleScript_None"),
            patch("modules.battle_handler.battle_action_selection_is_ready", return_value=False),
            patch("modules.battle_handler.task_is_active", return_value=False),
            patch("modules.battle_handler.get_player", return_value=player),
            patch("modules.battle_handler.get_party", return_value=[]),
            patch("modules.battle_handler.get_encounter_type", return_value=None),
            patch("modules.battle_handler.diagnostic_print"),
            patch.object(context, "emulator", Mock()),
        ):
            with pytest.raises(BotModeError, match="ROM-owned battle setup callback"):
                list(handle_battle(_strict_strategy()))

        assert _BATTLE_SETUP_TRANSITION_TIMEOUT == 600
    finally:
        context.bot_mode = previous_mode


@pytest.mark.parametrize(
    ("battle_type", "expected_timeout"),
    [
        (BattleType.Trainer, _BATTLE_SETUP_TRANSITION_TIMEOUT),
        (BattleType.Trainer | BattleType.WallyTutorial, _WALLY_TUTORIAL_SETUP_TRANSITION_TIMEOUT),
    ],
)
def test_campaign_uses_extended_setup_watchdog_for_wally_tutorial(battle_type, expected_timeout):
    with patch(
        "modules.battle_handler.get_battle_state",
        return_value=SimpleNamespace(type=battle_type),
    ):
        assert _battle_setup_transition_timeout() == expected_timeout


@pytest.mark.parametrize(
    "learned_move_script",
    [
        "BattleScript_LearnedNewMove",
        "BattleScript_ForgotAndLearnedNewMove",
    ],
)
def test_campaign_advances_move_learning_message_before_generic_script_transition(learned_move_script):
    """A learned-move message is actionable text, not an opaque ROM wait."""

    previous_mode = context.bot_mode
    context.bot_mode = "Campaign Progression"
    player = SimpleNamespace(money=0)
    emulator = Mock()

    try:
        with (
            patch("modules.battle_handler.battle_is_active", side_effect=(True, False)),
            patch("modules.battle_handler.observe_current_battle"),
            patch("modules.battle_handler.get_main_battle_callback", return_value="RunBattleScriptCommands_PopCallbacksStack"),
            patch(
                "modules.battle_handler.get_current_battle_script_instruction",
                return_value=learned_move_script,
            ),
            patch("modules.battle_handler.battle_action_selection_is_ready", return_value=False),
            patch("modules.battle_handler.task_is_active", return_value=False),
            patch("modules.battle_handler.get_player", return_value=player),
            patch("modules.battle_handler.get_party", return_value=[]),
            patch("modules.battle_handler.get_encounter_type", return_value=None),
            patch("modules.battle_handler.get_last_battle_outcome", return_value=None),
            patch("modules.battle_handler.diagnostic_print"),
            patch.object(context, "emulator", emulator),
        ):
            list(handle_battle(_strict_strategy()))

        emulator.press_button.assert_called_once_with("B")
    finally:
        context.bot_mode = previous_mode


def test_battle_end_summary_aggregates_observed_damage_and_selected_actions():
    knowledge = SimpleNamespace(
        damage_history=(
            DamageOccurred(2, "player", 0, 8, 20, 12),
            DamageOccurred(3, "player", 1, 4, 15, 11),
            DamageOccurred(4, "opponent", 0, 5, 20, 15),
        ),
        faint_history=(PokemonFainted(5, "player", 1, "Wurmple"),),
        switch_history=(PokemonSwitched(6, "player", 0, 1),),
        observations=(
            SimpleNamespace(
                battle_type=("Trainer",),
                player_party=(
                    SimpleNamespace(party_index=0, species="Treecko"),
                    SimpleNamespace(party_index=1, species="Wurmple"),
                ),
            ),
        ),
        opponent_pokemon=(SimpleNamespace(species="Poochyena"),),
    )

    with (
        patch("modules.battle_handler.get_last_completed_battle_knowledge", return_value=knowledge),
        patch(
            "modules.battle_handler.get_last_completed_battle_actions",
            return_value=(
                ("move", "UseMove(Pound)"),
                ("item", "UseItem(Potion -> Treecko)"),
                ("free_switch", "switch to Wurmple"),
            ),
        ),
        patch("modules.battle_handler.print_battle_end") as print_end,
        patch("modules.battle_handler.diagnostic_print"),
    ):
        _print_battle_end(SimpleNamespace(name="Won"))

    fields = print_end.call_args.kwargs
    assert fields["battle_type"] == "Trainer"
    assert fields["opponent_roster"] == ("Poochyena",)
    assert fields["total_damage_taken"] == 12
    assert dict(fields["damage_by_pokemon"]) == {"Treecko": 8, "Wurmple": 4}
    assert fields["items_used"] == ("UseItem(Potion -> Treecko)",)
    assert fields["moves_used"] == ("UseMove(Pound)",)
    assert fields["faints"] == ["Wurmple"]
    assert fields["switches"] == ("player 0→1",)


def test_battle_end_summary_accepts_terminal_knowledge_directly():
    current = SimpleNamespace(
        damage_history=(DamageOccurred(2, "player", 0, 3, 20, 17),),
        faint_history=(),
        switch_history=(),
        observations=(SimpleNamespace(battle_type=("Wild",), player_party=()),),
        opponent_pokemon=(),
    )

    with (
        patch("modules.battle_handler.get_last_completed_battle_knowledge", side_effect=AssertionError("stale read")),
        patch("modules.battle_handler.get_last_completed_battle_actions", side_effect=AssertionError("stale read")),
        patch("modules.battle_handler.print_battle_end") as print_end,
        patch("modules.battle_handler.diagnostic_print"),
    ):
        _print_battle_end(SimpleNamespace(name="Won"), knowledge=current, actions=())

    assert print_end.call_args.kwargs["total_damage_taken"] == 3


def test_battle_events_render_observed_opponent_faints():
    knowledge = SimpleNamespace(
        events=(PokemonFainted(5, "opponent", 0, "Nincada"),),
    )
    with patch("modules.battle_handler.print_battle_event") as print_event:
        assert _print_battle_events(knowledge, 0) == 1

    assert print_event.call_args.kwargs == {
        "event": "pokemon fainted",
        "side": "opponent",
        "pokemon": "Nincada",
        "detail": "party slot 0 reached 0 HP",
    }


def test_battle_events_correlate_observed_switch_with_pending_dispatch():
    knowledge = SimpleNamespace(
        events=(PokemonSwitched(5, "player", 0, 1),),
    )
    previous_pending = context.battle_pending_switches
    previous_replan = context.battle_switch_replan_required
    previous_status = context.battle_switch_reconciliation_status
    context.battle_pending_switches = [
        {
            "requested_name": "Lotad",
            "requested_index": 3,
            "final_name": "Lotad",
            "final_index": 3,
            "before_index": 0,
            "active": "Wurmple",
            "opponent": "Shroomish",
        }
    ]
    try:
        with (
            patch("modules.battle_handler._battle_event_pokemon_name", return_value="Mudkip"),
            patch("modules.battle_handler.get_party", return_value=()),
            patch("modules.battle_handler.print_battle_event"),
            patch("modules.battle_handler.print_battle_switch_resolution") as print_resolution,
        ):
            assert _print_battle_events(knowledge, 0) == 1

        assert context.battle_pending_switches == []
        fields = print_resolution.call_args.kwargs
        assert fields["status"] == "mismatch"
        assert fields["requested"] == "Lotad"
        assert fields["final_index"] == 3
        assert fields["observed"] == "Mudkip"
        assert fields["observed_index"] == 1
        assert context.battle_switch_replan_required is True
        assert context.battle_switch_reconciliation_status == "mismatch"
    finally:
        context.battle_pending_switches = previous_pending
        context.battle_switch_replan_required = previous_replan
        context.battle_switch_reconciliation_status = previous_status


def test_battle_start_does_not_use_opponent_original_trainer_as_battle_name():
    state = SimpleNamespace(is_trainer_battle=True, type_names=("Trainer",))
    opponent = SimpleNamespace(
        species=SimpleNamespace(name="Zigzagoon"),
        original_trainer=SimpleNamespace(name="Tiana"),
    )
    with (
        patch("modules.battle_handler.get_battle_state", return_value=state),
        patch("modules.battle_handler.get_encounter_type", return_value=None),
        patch("modules.battle_handler.get_opponent_party", return_value=(opponent,)),
        patch("modules.battle_handler.get_party", return_value=()),
        patch("modules.battle_handler._battle_trainer_id", return_value=603),
        patch("modules.battle_handler.print_battle_start") as print_start,
        patch("modules.battle_handler.diagnostic_print"),
    ):
        _print_battle_start()

    assert print_start.call_args.kwargs["trainer_name"] == "Trainer #603"


def test_battle_start_prefers_rom_checkpoint_name_over_opponent_ot_name():
    state = SimpleNamespace(
        is_trainer_battle=True,
        type_names=("Trainer",),
        opponent=SimpleNamespace(active_battler=SimpleNamespace()),
    )
    opponent = SimpleNamespace(
        species=SimpleNamespace(name="Poochyena"),
        original_trainer=SimpleNamespace(name="Noah"),
    )
    checkpoint = SimpleNamespace(
        display_name="Youngster Calvin",
        party=(SimpleNamespace(species="Poochyena", level=5, moves=("Tackle", "Howl")),),
        source="active_rom",
    )
    previous_checkpoint = context.campaign_trainer_checkpoint
    context.campaign_trainer_checkpoint = None
    try:
        with (
            patch("modules.battle_handler.get_battle_state", return_value=state),
            patch("modules.battle_handler.get_encounter_type", return_value=None),
            patch("modules.battle_handler.get_opponent_party", return_value=(opponent,)),
            patch("modules.battle_handler.get_party", return_value=()),
            patch("modules.battle_handler._battle_trainer_id", return_value=318),
            patch("modules.battle_handler.trainer_checkpoint_for_battle", return_value=checkpoint),
            patch("modules.battle_handler.print_battle_start") as print_start,
            patch("modules.battle_handler.diagnostic_print"),
        ):
            _print_battle_start()

        assert context.campaign_trainer_checkpoint is checkpoint
    finally:
        context.campaign_trainer_checkpoint = previous_checkpoint

    assert print_start.call_args.kwargs["trainer_name"] == "Youngster Calvin"
