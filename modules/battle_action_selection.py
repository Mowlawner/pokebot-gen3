from typing import Generator

from modules.battle_menuing import scroll_to_battle_action, scroll_to_move
from modules.battle_state import (
    BattleState,
    get_battle_state,
    battle_is_active,
    get_main_battle_callback,
    get_battle_controller_callback,
    get_current_battle_script_instruction,
)
from modules.battle_strategies import BattleStrategy, TurnAction, SafariTurnAction
from modules.context import context
from modules.console import diagnostic_print
from modules.debug import debug
from modules.game import decode_string
from modules.items import Item, ItemBattleUse, get_pokeblocks
from modules.memory import (
    read_symbol,
    unpack_uint32,
    get_game_state,
    GameState,
    get_game_state_symbol,
    unpack_uint16,
)
from modules.menuing import get_current_party_menu_index, scroll_to_item_in_bag, scroll_to_party_menu_index
from modules.pokemon_party import get_party, get_party_size
from modules.state_cache import state_cache


_BATTLE_ACTION_MAIN_CALLBACKS = (
    "HandleTurnActionSelectionState",
    "sub_8012324",
)
_TRAINER_REPLACEMENT_CALLBACKS = (
    "PlayerHandleYesNoBox",
    "PlayerHandleYesNoInput",
)
_TRAINER_REPLACEMENT_TEXT_CALLBACKS = (
    "CompleteOnInactiveTextPrinter",
    "CompleteOnInactiveTextPrinter2",
)
_TRAINER_REPLACEMENT_BUFFER_CALLBACK = "PlayerBufferRunCommand"
_TRAINER_REPLACEMENT_UNSET = object()

_last_battle_replacement_trace_signature = None


def _trainer_replacement_prompt_is_visible(battle_state: BattleState) -> bool:
    """Return whether Emerald is displaying its optional trainer switch prompt.

    The ROM does not expose the prompt through the player controller
    immediately.  While the message is being rendered, the main battle
    callback is still running ``BattleScript_FaintedMonTryChoose`` and the
    controller is ``PlayerBufferRunCommand``.  Reading the rendered text is
    therefore the reliable boundary between that transient state and an
    unrelated battle callback.
    """

    if getattr(battle_state, "is_double_battle", False) or not getattr(battle_state, "is_trainer_battle", False):
        return False

    try:
        own_active = battle_state.own_side.active_battler
        opponent_active = battle_state.opponent.active_battler
        # A forced player replacement also occurs around fainted-mon scripts,
        # but it has a fainted player battler rather than an alive one and is
        # consequently excluded here.
        if own_active is None or opponent_active is None:
            return False
        if own_active.current_hp <= 0 or opponent_active.current_hp > 0:
            return False

        displayed_string = decode_string(read_symbol("gDisplayedStringBattle"))
        return "is about to use " in displayed_string and "change POK" in displayed_string
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        # Prompt detection is a guard against the generic fallback.  If the
        # ROM is between valid memory states, let the next frame reevaluate it.
        return False


def _safe_callback(battler_index: int) -> str:
    try:
        return get_battle_controller_callback(battler_index)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return "<unavailable>"


def _active_party_indices(battle_state: BattleState) -> frozenset[int]:
    """Return every party slot currently deployed on the player's side.

    ``active_battlers`` is authoritative in a stable battle snapshot, but it
    can be temporarily incomplete while Emerald is moving between battle
    callbacks. Include the single-battler and left/right projections as a
    defensive union so a stale switch decision can never open the party menu
    on an already-active slot.
    """

    own_side = getattr(battle_state, "own_side", None)
    candidates = list(getattr(own_side, "active_battlers", ()) or ())
    candidates.extend(
        battler
        for battler in (
            getattr(own_side, "active_battler", None),
            getattr(own_side, "left_battler", None),
            getattr(own_side, "right_battler", None),
        )
        if battler is not None
    )
    return frozenset(
        getattr(battler, "party_index", None)
        for battler in candidates
        if getattr(battler, "party_index", None) is not None
    )


def _capture_party_member(party_index: int):
    """Capture a party member before opening a battle-owned party menu.

    Numeric party positions are not stable across the battle-menu boundary:
    Emerald may expose the battle's current order while the overworld party
    reader still exposes the normal order.  The Pokémon's identity is stable,
    so keep it for the later menu-coordinate resolution.  This is deliberately
    best-effort because some legacy/test strategies do not provide party data.
    """

    try:
        return get_party()[party_index]
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return None


def _battle_party_order_indices(battle_state: BattleState) -> tuple[int, ...]:
    """Decode the ROM's packed battle-party order for diagnostics."""

    order = getattr(battle_state, "_battler_party_order", None)
    if order is None:
        return ()
    result = []
    try:
        for battle_party_index in range(6):
            byte = order[battle_party_index // 2]
            result.append(byte >> 4 if battle_party_index % 2 == 0 else byte & 0xF)
    except (IndexError, TypeError, ValueError):
        return tuple(result)
    return tuple(result)


def _resolve_battle_party_menu_index(
    battle_state: BattleState,
    regular_party_index: int,
    target_pokemon=None,
    mapped_index: int | None = None,
) -> int:
    """Resolve a regular party slot to the cursor coordinate used by the menu.

    ``gBattlePartyCurrentOrder`` is useful when the menu cannot be inspected,
    but the visible party snapshot is authoritative once the menu is open.  In
    particular, this prevents a stale battle-order buffer from causing a
    decision for Lotad to select the already-active Treecko.
    """

    if mapped_index is None:
        try:
            mapped_index = battle_state.map_battle_party_index(regular_party_index)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            mapped_index = None

    visible_index = None
    if target_pokemon is not None:
        try:
            visible_index = get_party().get_index_for_pokemon(target_pokemon)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            visible_index = None

    resolved_index = visible_index if visible_index is not None else mapped_index
    if resolved_index is None:
        raise RuntimeError(
            f"Could not resolve battle party menu slot for regular party index #{regular_party_index}."
        )

    diagnostic_print(
        lambda: (
            "BATTLE_PARTY_MENU_TARGET: "
            f"regular_index={regular_party_index!r} "
            f"mapped_index={mapped_index!r} visible_index={visible_index!r} "
            f"resolved_index={resolved_index!r} "
            f"battle_order={_battle_party_order_indices(battle_state)!r} "
            f"cursor_before={get_current_party_menu_index()!r} "
            f"target={getattr(target_pokemon, 'name', None)!r}"
        ),
        trace=True,
    )
    return resolved_index


def _scroll_to_battle_party_member(
    battle_state: BattleState,
    regular_party_index: int,
    target_pokemon=None,
    mapped_index: int | None = None,
    menu_already_open: bool = False,
) -> Generator:
    """Open-menu boundary plus identity-first cursor navigation."""

    if not menu_already_open:
        while get_game_state() != GameState.PARTY_MENU:
            yield

    # The menu can reorder the visible party in the same emulator frame in
    # which it opens. Do not let a pre-menu party cache hide that reorder.
    state_cache.party.invalidate()
    menu_index = _resolve_battle_party_menu_index(
        battle_state,
        regular_party_index,
        target_pokemon,
        mapped_index,
    )
    yield from scroll_to_party_menu_index(menu_index)


def _fallback_for_invalid_switch(battle_state: BattleState):
    """Find one executable action after rejecting an active-slot switch."""

    try:
        from modules.battle_strategies._util import BattleStrategyUtil

        util = BattleStrategyUtil(battle_state)
        active = battle_state.own_side.active_battler
        opponent = battle_state.opponent.active_battler
        if active is not None and opponent is not None:
            move_index = util.get_strongest_move_against(active, opponent)
            if move_index is not None:
                return TurnAction.UseMove, move_index
        if not getattr(battle_state, "is_trainer_battle", False):
            escape = util.get_best_escape_method()
            if escape is not None:
                return escape
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        pass
    return None


def _battle_species_name(battler) -> str:
    try:
        species = getattr(battler, "species", None)
        return getattr(species, "name", None) or getattr(battler, "name", None) or "unknown"
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return "unknown"


def _battle_action_details(
    battle_state: BattleState,
    strategy: BattleStrategy,
    action,
    index,
    battler_index: int,
) -> tuple[str, ...]:
    """Normalize a strategy result into GUI- and trace-friendly language."""

    try:
        capture = bool(getattr(battle_state, "nuzlocke_capture_target", False))
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        capture = False
    objective = "capture" if capture else "defeat"
    intent = "select action"
    target = "KO"
    action_label = getattr(action, "name", str(action))

    own_side = getattr(battle_state, "own_side", None)
    if battler_index == 2:
        active = getattr(own_side, "right_battler", None)
    else:
        active = getattr(own_side, "active_battler", None)
    opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
    active_name = _battle_species_name(active)
    opponent_name = _battle_species_name(opponent)

    if action in (
        TurnAction.UseMove,
        TurnAction.UseMoveAgainstRightSideOpponent,
        TurnAction.UseMoveAgainstPartner,
    ):
        intent = "select move"
        target = "lower HP" if capture else "KO"
        try:
            learned_move = active.moves[index]
            move_name = getattr(getattr(learned_move, "move", None), "name", None) or "move"
            action_label = f"{action_label}({move_name})"
        except (AttributeError, IndexError, TypeError, KeyError):
            action_label = f"{action_label}(index={index!r})"
    elif action is TurnAction.RotateLead:
        intent = "switch party member"
        target = "safe matchup / EXP rotation"
        try:
            replacement = get_party()[index]
            replacement_name = _battle_species_name(replacement)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            replacement_name = f"party slot {index!r}"
        action_label = f"{action_label}({replacement_name})"
    elif action is TurnAction.UseItem:
        intent = "select item"
        item = index[0] if isinstance(index, tuple) else index
        item_name = getattr(item, "name", str(item))
        action_label = f"{action_label}({item_name})"
        if "ball" in item_name.lower():
            objective = "capture"
            target = "capture"
        elif getattr(item, "battle_use", None) is not None:
            target = "restore HP/PP"
    elif action in (TurnAction.RunAway, SafariTurnAction.RunAway):
        objective = "flee"
        target = "flee"
        intent = "select Run"
    elif action is SafariTurnAction.ThrowBall:
        objective = "capture"
        target = "capture"
        intent = "select Safari Ball"
    elif action is SafariTurnAction.Pokeblock:
        objective = "capture"
        target = "improve catch odds"
        intent = "select Pokéblock"
    elif action in (SafariTurnAction.GoNear, SafariTurnAction.Bait, SafariTurnAction.Rock):
        objective = "capture"
        target = "improve catch odds"
        intent = "select Safari action"
    elif action in (TurnAction.SwitchToManual, SafariTurnAction.SwitchToManual):
        intent = "pause for manual input"
        target = "manual review"
        action_label = "manual"

    source = getattr(context, "battle_decision_source", None) or type(strategy).__name__
    reason = getattr(context, "battle_decision_detail", None) or "selected by battle policy"
    return objective, target, intent, action_label, source, reason, active_name, opponent_name


def _log_battle_action_chosen(
    battle_state: BattleState,
    strategy: BattleStrategy,
    action,
    index,
    battler_index: int,
) -> None:
    """Publish every selected battle action through the GUI and opt-in trace."""

    (
        objective,
        target,
        intent,
        action_label,
        source,
        reason,
        active_name,
        opponent_name,
    ) = _battle_action_details(battle_state, strategy, action, index, battler_index)
    message = (
        f"BATTLE ACTION: objective={objective}; target={target}; intent={intent}; "
        f"active={active_name}; opponent={opponent_name}; action={action_label}; "
        f"source={source}; reason={reason}"
    )
    context.message = message
    diagnostic_print(
        lambda: (
            "BATTLE_ACTION_CHOSEN: "
            f"objective={objective!r} target={target!r} intent={intent!r} "
            f"active={active_name!r} opponent={opponent_name!r} action={action_label!r} "
            f"source={source!r} reason={reason!r}"
        ),
        trace=True,
        prefix="BATTLE_ACTION_CHOSEN:",
    )


def _log_trainer_replacement_choice(battle_state: BattleState, replacement_index: int | None, source: str) -> None:
    """Expose the Yes/No replacement decision in the same GUI vocabulary."""

    current = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
    opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
    replacement_name = "stay active"
    target = "keep current matchup"
    intent = "select No"
    if replacement_index is not None:
        try:
            replacement_name = _battle_species_name(get_party()[replacement_index])
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            replacement_name = f"party slot {replacement_index}"
        target = "safe matchup / EXP rotation"
        intent = "select Yes and switch"
    reason = getattr(context, "battle_decision_detail", None) or "replacement policy found no better legal option"
    context.message = (
        f"BATTLE ACTION: objective=defeat; target={target}; intent={intent}; "
        f"active={_battle_species_name(current)}; opponent={_battle_species_name(opponent)}; "
        f"action={'switch to ' + replacement_name if replacement_index is not None else 'stay active'}; "
        f"source={source}; reason={reason}"
    )
    diagnostic_print(
        lambda: (
            "BATTLE_ACTION_CHOSEN: objective='defeat' "
            f"target={target!r} intent={intent!r} active={_battle_species_name(current)!r} "
            f"opponent={_battle_species_name(opponent)!r} replacement={replacement_name!r} "
            f"source={source!r} reason={reason!r}"
        ),
        trace=True,
        prefix="BATTLE_ACTION_CHOSEN:",
    )


def _battler_trace_summary(battler) -> tuple | None:
    if battler is None:
        return None
    try:
        species = getattr(getattr(battler, "species", None), "name", None)
        return (
            getattr(battler, "party_index", None),
            species,
            getattr(battler, "current_hp", None),
            getattr(battler, "level", None),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return ("<unreadable>",)


def _trace_battle_replacement_boundary(event: str, battle_state: BattleState | None = None) -> None:
    """Trace the ROM/controller boundary without changing battle behavior."""

    global _last_battle_replacement_trace_signature
    try:
        main_callback = get_main_battle_callback()
        script_instruction = get_current_battle_script_instruction()
        controller_callbacks = tuple(_safe_callback(index) for index in range(4))
        game_state = get_game_state()
        active_own = _battler_trace_summary(
            getattr(getattr(battle_state, "own_side", None), "active_battler", None)
        )
        active_opponent = _battler_trace_summary(
            getattr(getattr(battle_state, "opponent", None), "active_battler", None)
        )
        signature = (main_callback, script_instruction, controller_callbacks, game_state, active_own, active_opponent)
        if signature == _last_battle_replacement_trace_signature:
            return
        _last_battle_replacement_trace_signature = signature
        emulator = getattr(context, "emulator", None)
        diagnostic_print(
            lambda: (
                "BATTLE_REPLACEMENT_TRACE: "
                f"event={event!r} context_frame={getattr(context, 'frame', None)!r} "
                f"emulator_frame={emulator.get_frame_count() if emulator else None!r} "
                f"main_callback={main_callback!r} script={script_instruction!r} "
                f"controller_callbacks={controller_callbacks!r} game_state={game_state!r} "
                f"battle_state_id={id(battle_state) if battle_state is not None else None!r} "
                f"battle_cache_age={state_cache.battle_state.age_in_frames!r} "
                f"is_trainer={getattr(battle_state, 'is_trainer_battle', None)!r} "
                f"is_double={getattr(battle_state, 'is_double_battle', None)!r} "
                f"own_active={active_own!r} opponent_active={active_opponent!r}"
            ),
            trace=True,
            prefix="BATTLE_REPLACEMENT_TRACE:",
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return


def battle_action_selection_is_ready(battle_state: BattleState | None = None) -> bool:
    """Return whether the player's battle action controller needs input.

    Trainer replacement prompts are owned by the per-battler controller while
    Emerald briefly runs a different main battle callback.  Treat those
    prompts as action-selection work in their own right; otherwise the outer
    battle handler's generic ``B`` fallback answers every prompt with No
    before the strategy can inspect it.
    """

    if get_main_battle_callback() in _BATTLE_ACTION_MAIN_CALLBACKS:
        return True
    if battle_state is None:
        battle_state = get_battle_state()
    _trace_battle_replacement_boundary("readiness_check", battle_state)
    if (
        not getattr(battle_state, "is_double_battle", False)
        and getattr(battle_state, "is_trainer_battle", False)
        and get_battle_controller_callback(0) in _TRAINER_REPLACEMENT_CALLBACKS
    ):
        return True

    # During the first frames of the optional replacement prompt the player
    # controller is still PlayerBufferRunCommand.  Treat that interval as
    # readiness without issuing input; handle_battle_action_selection will
    # yield until the controller reaches one of the callbacks above.
    return _trainer_replacement_prompt_is_visible(battle_state)


@debug.track
def handle_battle_action_selection(strategy: BattleStrategy) -> Generator:
    battle_state = get_battle_state()
    previous_battler_index = None
    # Emerald advances from PlayerHandleYesNoBox to
    # PlayerHandleYesNoInput after it creates the replacement prompt. Keep
    # both callbacks in the same transaction so the handler does not miss the
    # prompt between frames and leave the ROM's default No selected.
    trainer_replacement_callbacks = _TRAINER_REPLACEMENT_CALLBACKS
    trainer_replacement_buffer_step = None
    trainer_replacement_decision = _TRAINER_REPLACEMENT_UNSET
    trainer_replacement_pokemon = None
    trainer_replacement_resolved = False

    while battle_is_active() and battle_action_selection_is_ready(battle_state):
        _trace_battle_replacement_boundary("handler_loop", battle_state)
        controller_callback = get_battle_controller_callback(0)
        replacement_prompt_visible = _trainer_replacement_prompt_is_visible(battle_state)
        if not replacement_prompt_visible:
            trainer_replacement_buffer_step = None
            trainer_replacement_decision = _TRAINER_REPLACEMENT_UNSET
            trainer_replacement_pokemon = None
            trainer_replacement_resolved = False
        diagnostic_print(
            lambda: (
                "BATTLE_REPLACEMENT_DISPATCH: "
                f"context_frame={getattr(context, 'frame', None)!r} "
                f"emulator_frame={context.emulator.get_frame_count()!r} "
                f"strategy={type(strategy).__name__!r} "
                f"main_callback={get_main_battle_callback()!r} "
                f"controller_0={controller_callback!r} "
                f"is_trainer={getattr(battle_state, 'is_trainer_battle', None)!r} "
                f"is_double={getattr(battle_state, 'is_double_battle', None)!r} "
                f"replacement_branch={bool(
                    not getattr(battle_state, 'is_double_battle', False)
                    and getattr(battle_state, 'is_trainer_battle', False)
                    and _safe_callback(0) in trainer_replacement_callbacks
                )!r}"
            ),
            trace=True,
            prefix="BATTLE_REPLACEMENT_DISPATCH:",
        )
        if (
            replacement_prompt_visible
            and controller_callback in _TRAINER_REPLACEMENT_TEXT_CALLBACKS
        ):
            # The optional-switch message is first owned by the battle text
            # printer.  Advance that message explicitly.  B is safe at this
            # boundary because the Yes/No prompt has not been handed to the
            # player controller yet; at PlayerBufferRunCommand it is instead
            # interpreted as the default No unless policy explicitly chooses
            # that answer.
            diagnostic_print(
                lambda: (
                    "BATTLE_REPLACEMENT_INPUT: "
                    f"phase='advance_prompt_text' button='B' controller={controller_callback!r}"
                ),
                trace=True,
                prefix="BATTLE_REPLACEMENT_INPUT:",
            )
            press_button_fresh = (
                getattr(context.emulator, "press_button_fresh", None)
                if hasattr(type(context.emulator), "press_button_fresh")
                else None
            )
            if callable(press_button_fresh):
                press_button_fresh("B")
            else:
                context.emulator.press_button("B")
            yield
            continue
        if replacement_prompt_visible and controller_callback == "OpenPartyMenuToChooseMon":
            if trainer_replacement_resolved and trainer_replacement_decision is not None:
                yield
                continue
            if trainer_replacement_decision not in (_TRAINER_REPLACEMENT_UNSET, None):
                for _ in range(120):
                    if get_game_state() == GameState.PARTY_MENU:
                        yield from _scroll_to_battle_party_member(
                            battle_state,
                            trainer_replacement_decision,
                            trainer_replacement_pokemon,
                            menu_already_open=True,
                        )
                        context.emulator.press_button("A")
                        yield
                        while get_game_state() == GameState.PARTY_MENU:
                            context.emulator.press_button("A")
                            yield
                        trainer_replacement_resolved = True
                        break
                    if not battle_is_active():
                        break
                    yield
            continue
        if (
            replacement_prompt_visible
            and controller_callback == _TRAINER_REPLACEMENT_BUFFER_CALLBACK
        ):
            # Emerald keeps PlayerBufferRunCommand installed while it owns
            # the replacement yes/no input. It does not transition through
            # PlayerHandleYesNoBox/Input on this path, so reproduce the
            # controller's menu sequence here. The policy is consulted at
            # the real choice below; this only moves the cursor to Yes.
            if trainer_replacement_resolved and trainer_replacement_decision is not None:
                yield
                continue
            if trainer_replacement_decision is _TRAINER_REPLACEMENT_UNSET:
                try:
                    trainer_replacement_decision = strategy.choose_trainer_replacement(battle_state)
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
                    diagnostic_print(
                        lambda: f"TRAINER_REPLACEMENT_PROMPT: strategy_error={error!r} answer='No'",
                        trace=True,
                    )
                    trainer_replacement_decision = None
                if (
                    trainer_replacement_decision is not None
                    and (
                        trainer_replacement_decision < 0
                        or trainer_replacement_decision >= get_party_size()
                    )
                ):
                    diagnostic_print(
                        lambda: (
                            "TRAINER_REPLACEMENT_PROMPT: "
                            f"invalid_party_index={trainer_replacement_decision!r} answer='No'"
                        ),
                        trace=True,
                    )
                    trainer_replacement_decision = None

                _log_trainer_replacement_choice(
                    battle_state,
                    None if trainer_replacement_decision is _TRAINER_REPLACEMENT_UNSET else trainer_replacement_decision,
                    "TRAINER REPLACEMENT POLICY",
                )
                if trainer_replacement_decision is not None:
                    trainer_replacement_pokemon = _capture_party_member(trainer_replacement_decision)
                if trainer_replacement_decision is not None:
                    diagnostic_print(
                        lambda: (
                            "TRAINER_REPLACEMENT_PROMPT: "
                            f"answer='Yes' replacement_party_index={trainer_replacement_decision!r}"
                        ),
                        trace=True,
                    )

            if trainer_replacement_decision is None:
                diagnostic_print(
                    lambda: "TRAINER_REPLACEMENT_PROMPT: answer='No'",
                    trace=True,
                )
                press_button_fresh = (
                    getattr(context.emulator, "press_button_fresh", None)
                    if hasattr(type(context.emulator), "press_button_fresh")
                    else None
                )
                if callable(press_button_fresh):
                    press_button_fresh("B")
                else:
                    context.emulator.press_button("B")
                trainer_replacement_resolved = True
                yield
                continue

            if trainer_replacement_buffer_step is None:
                button = "Up"
                trainer_replacement_buffer_step = "up_sent"
            else:
                button = "A"
                trainer_replacement_buffer_step = None
            diagnostic_print(
                lambda: (
                    "BATTLE_REPLACEMENT_INPUT: "
                    f"phase='buffer_prompt_navigation' button={button!r}"
                ),
                trace=True,
                prefix="BATTLE_REPLACEMENT_INPUT:",
            )
            press_button_fresh = (
                getattr(context.emulator, "press_button_fresh", None)
                if hasattr(type(context.emulator), "press_button_fresh")
                else None
            )
            if callable(press_button_fresh):
                press_button_fresh(button)
            else:
                context.emulator.press_button(button)
            yield
            continue
        if (
            not battle_state.is_double_battle
            and battle_state.is_trainer_battle
            and controller_callback in trainer_replacement_callbacks
        ):
            trainer_replacement_buffer_step = None
            # In Switch mode Emerald asks before each incoming trainer
            # Pokémon. The ordinary battle fallback pressed B here, which
            # permanently selected No. An opt-in strategy can instead return
            # a validated party slot; the prompt is answered Yes and the
            # battle-owned party menu is completed before the next action.
            _trace_battle_replacement_boundary("prompt_before_strategy", battle_state)
            try:
                replacement_index = strategy.choose_trainer_replacement(battle_state)
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
                diagnostic_print(
                    lambda: f"TRAINER_REPLACEMENT_PROMPT: strategy_error={error!r} answer='No'",
                    trace=True,
                )
                replacement_index = None
            _trace_battle_replacement_boundary("prompt_after_strategy", battle_state)
            if replacement_index is None:
                diagnostic_print(
                    lambda: "TRAINER_REPLACEMENT_PROMPT: answer='No'",
                    trace=True,
                )
                context.emulator.press_button("B")
                yield
                continue
            if replacement_index < 0 or replacement_index >= get_party_size():
                diagnostic_print(
                    lambda: (
                        "TRAINER_REPLACEMENT_PROMPT: "
                        f"invalid_party_index={replacement_index!r} answer='No'"
                    ),
                    trace=True,
                )
                context.emulator.press_button("B")
                yield
                continue

            _log_trainer_replacement_choice(battle_state, replacement_index, "TRAINER REPLACEMENT POLICY")
            trainer_replacement_pokemon = _capture_party_member(replacement_index)
            diagnostic_print(
                lambda: (
                    "TRAINER_REPLACEMENT_PROMPT: "
                    f"answer='Yes' replacement_party_index={replacement_index!r}"
                ),
                trace=True,
            )
            # The battle yes/no box opens with No selected in the Gen III
            # battle UI. Move to Yes, confirm it, then wait for the ROM-owned
            # replacement menu to become ready.
            press_button_fresh = (
                getattr(context.emulator, "press_button_fresh", None)
                if hasattr(type(context.emulator), "press_button_fresh")
                else None
            )
            if callable(press_button_fresh):
                diagnostic_print(
                    lambda: (
                        "BATTLE_REPLACEMENT_INPUT: "
                        f"phase='request_up' implementation='press_button_fresh' "
                        f"emulator_frame={context.emulator.get_frame_count()!r} "
                        f"previous_inputs={getattr(context.emulator, 'get_previous_frame_inputs', lambda: None)()!r} "
                        f"current_inputs={getattr(context.emulator, 'get_current_frame_inputs', lambda: None)()!r}"
                    ),
                    trace=True,
                    prefix="BATTLE_REPLACEMENT_INPUT:",
                )
                press_button_fresh("Up")
            else:
                diagnostic_print(
                    lambda: "BATTLE_REPLACEMENT_INPUT: phase='request_up' implementation='press_button'",
                    trace=True,
                    prefix="BATTLE_REPLACEMENT_INPUT:",
                )
                context.emulator.press_button("Up")
            yield
            if callable(press_button_fresh):
                diagnostic_print(
                    lambda: (
                        "BATTLE_REPLACEMENT_INPUT: "
                        f"phase='request_a' implementation='press_button_fresh' "
                        f"emulator_frame={context.emulator.get_frame_count()!r} "
                        f"previous_inputs={getattr(context.emulator, 'get_previous_frame_inputs', lambda: None)()!r} "
                        f"current_inputs={getattr(context.emulator, 'get_current_frame_inputs', lambda: None)()!r}"
                    ),
                    trace=True,
                    prefix="BATTLE_REPLACEMENT_INPUT:",
                )
                press_button_fresh("A")
            else:
                diagnostic_print(
                    lambda: "BATTLE_REPLACEMENT_INPUT: phase='request_a' implementation='press_button'",
                    trace=True,
                    prefix="BATTLE_REPLACEMENT_INPUT:",
                )
                context.emulator.press_button("A")
            yield
            for _ in range(120):
                if get_game_state() == GameState.PARTY_MENU:
                    yield from _scroll_to_battle_party_member(
                        battle_state,
                        replacement_index,
                        trainer_replacement_pokemon,
                        menu_already_open=True,
                    )
                    context.emulator.press_button("A")
                    yield
                    while get_game_state() == GameState.PARTY_MENU:
                        context.emulator.press_button("A")
                        yield
                    break
                if get_battle_controller_callback(0) not in trainer_replacement_callbacks:
                    break
                yield
            continue
        elif get_battle_controller_callback(0) in (
            "HandleInputChooseAction",
            "sub_802C098",
            "bx_battle_menu_t6_2",
        ):
            battler_index = 0
            # Decision metadata belongs to one action only. Clear it before
            # asking a strategy so a quiet fallback cannot inherit the reason
            # for the previous turn's switch or item.
            context.battle_decision_source = None
            context.battle_decision_detail = None
            if battle_state.is_safari_zone_encounter:
                action, index = strategy.decide_turn_in_safari_zone(battle_state)
            elif battle_state.is_double_battle:
                action, index = strategy.decide_turn_in_double_battle(battle_state, 0)
            else:
                action, index = strategy.decide_turn(battle_state)

        elif battle_state.is_double_battle and get_battle_controller_callback(2) in (
            "HandleInputChooseAction",
            "sub_802C098",
        ):
            battler_index = 2
            context.battle_decision_source = None
            context.battle_decision_detail = None
            action, index = strategy.decide_turn_in_double_battle(battle_state, 1)

        elif get_battle_controller_callback(0) in (
            "HandleInputChooseMove",
            "HandleAction_ChooseMove",
        ) or (
            battle_state.is_double_battle
            and get_battle_controller_callback(2) in ("HandleInputChooseMove", "HandleAction_ChooseMove")
        ):
            context.emulator.press_button("B")
            yield
            continue

        else:
            yield
            continue

        _log_battle_action_chosen(battle_state, strategy, action, index, battler_index)

        if context.bot_mode == "Manual":
            yield
            continue

        # In double battles, both player Pokémon will do their inputs in the same run. So the active
        # battler might change in the middle of this function. If we detect that, we update the battle
        # state as the previous Pokémon's actions might have changed it (e.g. by using potions.)
        if previous_battler_index != battler_index:
            battle_state = get_battle_state()
            previous_battler_index = battler_index

        diagnostic_print(
            lambda: (
                "BATTLE_ACTION: "
                f"source={getattr(context, 'battle_decision_source', None)} "
                f"menu_ready={get_battle_controller_callback(battler_index)} "
                f"selected={getattr(action, 'name', action)} move_index={index}"
            ),
            trace=True,
        )

        match action:
            case TurnAction.UseMove | TurnAction.UseMoveAgainstRightSideOpponent | TurnAction.UseMoveAgainstPartner:
                yield from battle_action_use_move(action, battler_index, index, battle_state)

            case TurnAction.UseItem:
                if isinstance(index, tuple):
                    index, target_index = index
                else:
                    target_index = None

                if not isinstance(index, Item):
                    raise RuntimeError(f"`TurnAction.UseItem` needs an item object as its first argument.")
                if index.battle_use == ItemBattleUse.NotUsable:
                    raise RuntimeError(f"Item `{index.name}` cannot be used in battle.")
                if index.battle_use in (ItemBattleUse.Healing, ItemBattleUse.PpRecovery) and target_index is None:
                    raise RuntimeError(f"Item `{index.name}` needs a target Pokémon.")

                yield from battle_action_use_item(battle_state, index, target_index)

            case TurnAction.RotateLead:
                if index >= get_party_size():
                    raise RuntimeError(
                        f"Cannot switch in party slot #{index} because the party only has {get_party_size()} Pokémon."
                    )

                active_indices = _active_party_indices(battle_state)
                if index in active_indices:
                    # A stale strategy decision must be replanned before the
                    # party menu is opened.  Sending SHIFT for an active slot
                    # is not recoverable in Emerald: the ROM leaves the menu
                    # open and repeats "already in battle" forever.
                    selector = getattr(strategy, "select_valid_switch_target", None)
                    fallback_index = selector(battle_state) if callable(selector) else None
                    if fallback_index is not None and fallback_index not in _active_party_indices(battle_state):
                        diagnostic_print(
                            lambda: (
                                "BATTLE_SWITCH_REPLANNED: "
                                f"requested={index!r} replacement={fallback_index!r} reason='target already active'"
                            ),
                            trace=True,
                        )
                        index = fallback_index
                    else:
                        # Do not re-run the same stale strategy decision. In
                        # particular, Emerald leaves the party menu open
                        # after "already in battle", so retrying SHIFT is an
                        # infinite loop. Use the current battler's legal move
                        # (or flee in a wild battle) when possible.
                        fallback_action = _fallback_for_invalid_switch(battle_state)
                        diagnostic_print(
                            lambda: (
                                "BATTLE_SWITCH_REJECTED: "
                                f"requested={index!r} active_indices={sorted(active_indices)!r} "
                                f"fallback={fallback_action!r} reason='target already active'"
                            ),
                            trace=True,
                        )
                        if fallback_action is not None:
                            fallback, fallback_index = fallback_action
                            context.battle_decision_source = "BATTLE SWITCH GUARD"
                            context.battle_decision_detail = (
                                f"rejected party slot {index}; it is already active; "
                                f"executing {getattr(fallback, 'name', fallback)} instead"
                            )
                            context.message = (
                                f"BATTLE SWITCH GUARD: slot {index} is already active; "
                                f"using {getattr(fallback, 'name', fallback)} instead"
                            )
                            if fallback is TurnAction.UseMove:
                                yield from battle_action_use_move(
                                    fallback, battler_index, fallback_index, battle_state
                                )
                            elif fallback in (TurnAction.RunAway, SafariTurnAction.RunAway):
                                yield from scroll_to_battle_action(3)
                                context.emulator.press_button("A")
                                yield
                            else:
                                context.set_manual_mode()
                                yield
                        else:
                            context.battle_decision_source = "BATTLE SWITCH GUARD"
                            context.battle_decision_detail = (
                                f"rejected party slot {index}; it is already active and no legal fallback was found"
                            )
                            context.message = (
                                f"BATTLE SWITCH GUARD: slot {index} is already active; no legal fallback"
                            )
                            context.set_manual_mode()
                            yield
                        continue

                # Re-read the active projection after a selector fallback.
                # A battle callback can change the side projection between
                # the initial strategy decision and this executor boundary.
                if index in _active_party_indices(battle_state):
                    diagnostic_print(
                        lambda: (
                            "BATTLE_SWITCH_REJECTED: "
                            f"requested={index!r} reason='selector returned active slot'"
                        ),
                        trace=True,
                    )
                    fallback_action = _fallback_for_invalid_switch(battle_state)
                    context.battle_decision_source = "BATTLE SWITCH GUARD"
                    context.battle_decision_detail = f"selector returned already-active party slot {index}"
                    context.message = f"BATTLE SWITCH GUARD: selector returned active slot {index}; replanning"
                    if fallback_action is not None and fallback_action[0] is TurnAction.UseMove:
                        yield from battle_action_use_move(
                            fallback_action[0], battler_index, fallback_action[1], battle_state
                        )
                    elif fallback_action is not None and fallback_action[0] in (
                        TurnAction.RunAway,
                        SafariTurnAction.RunAway,
                    ):
                        yield from scroll_to_battle_action(3)
                        context.emulator.press_button("A")
                        yield
                    else:
                        context.set_manual_mode()
                        yield
                    continue

                # A strategy can emit a switch based on an earlier snapshot.
                # Recheck the matchup at the menu boundary so an invalid
                # generic/planner switch cannot send out a zero-damage member
                # and fail later in the battle GUI. Specialized strategies may
                # authorize a verified multi-step continuation explicitly.
                is_valid_switch = False
                validator = getattr(strategy, "is_switch_target_valid", None)
                if not callable(validator):
                    # Third-party/legacy strategies may implement the
                    # original BattleStrategy contract without the optional
                    # matchup validator. Preserve their established switch
                    # behavior; built-in strategies all provide validation.
                    is_valid_switch = True
                else:
                    is_valid_switch = bool(validator(battle_state, index))
                if not is_valid_switch:
                    selector = getattr(strategy, "select_valid_switch_target", None)
                    fallback_index = selector(battle_state) if callable(selector) else None
                    if fallback_index is None:
                        raise RuntimeError(
                            f"Cannot switch in {get_party()[index].name}: it has no valid matchup switch target."
                        )
                    diagnostic_print(
                        lambda: ("BATTLE_SWITCH_REPLANNED: " f"requested={index!r} replacement={fallback_index!r}"),
                        trace=True,
                    )
                    index = fallback_index

                target_pokemon = _capture_party_member(index)
                try:
                    in_battle_index = battle_state.map_battle_party_index(index)
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
                    in_battle_index = None

                yield from scroll_to_battle_action(2)
                # JP roms need a little delay
                for _ in range(5):
                    yield
                context.emulator.press_button("A")
                yield from _scroll_to_battle_party_member(
                    battle_state,
                    index,
                    target_pokemon,
                    in_battle_index,
                )
                while get_game_state() == GameState.PARTY_MENU:
                    context.emulator.press_button("A")
                    yield

            case TurnAction.RunAway | SafariTurnAction.RunAway:
                if battle_state.is_trainer_battle:
                    raise RuntimeError("Tried to run away from a trainer battle.")
                yield from scroll_to_battle_action(3)
                context.emulator.press_button("A")
                yield

            case SafariTurnAction.ThrowBall:
                yield from scroll_to_battle_action(0)
                context.emulator.press_button("A")
                yield

            case SafariTurnAction.GoNear:
                if context.rom.is_frlg:
                    raise RuntimeError("The 'Go Near' option is not available in FR/LG Safari battles.")

                yield from scroll_to_battle_action(2)
                context.emulator.press_button("A")
                yield

            case SafariTurnAction.Pokeblock:
                if context.rom.is_frlg:
                    raise RuntimeError("The 'Pokéblock' option is not available in FR/LG Safari battles.")

                yield from battle_action_use_pokeblock(index)
                break

            case SafariTurnAction.Bait:
                if context.rom.is_rse:
                    raise RuntimeError("The 'Bait' option is not available in R/S/E Safari battles.")

                yield from scroll_to_battle_action(1)
                context.emulator.press_button("A")
                yield

            case SafariTurnAction.Rock:
                if context.rom.is_rse:
                    raise RuntimeError("The 'Rock' option is not available in R/S/E Safari battles.")

                yield from scroll_to_battle_action(2)
                context.emulator.press_button("A")
                yield

            case TurnAction.SwitchToManual | SafariTurnAction.SwitchToManual:
                context.set_manual_mode()
                yield

            case _:
                raise RuntimeError(f"Invalid turn action: {action}")


@debug.track
def battle_action_use_item(battle_state: BattleState, item: Item, target_index: int = 0):
    target_pokemon = _capture_party_member(target_index) if target_index is not None else None
    yield from scroll_to_battle_action(1)
    context.emulator.press_button("A")
    yield
    yield from scroll_to_item_in_bag(item)
    context.emulator.press_button("A")
    for _ in range(4 if context.rom.is_rse else 5):
        yield
    context.emulator.press_button("A")
    yield
    if target_index is not None:
        try:
            in_battle_index = battle_state.map_battle_party_index(target_index)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            in_battle_index = None
        yield from _scroll_to_battle_party_member(
            battle_state,
            target_index,
            target_pokemon,
            in_battle_index,
        )
        context.emulator.press_button("A")
        yield
        while get_game_state_symbol() != "BATTLEMAINCB2":
            context.emulator.press_button("B")
            yield


@debug.track
def battle_action_use_pokeblock(poke_block_index: int):
    yield from scroll_to_battle_action(1)
    while get_game_state_symbol() == "BATTLEMAINCB2":
        context.emulator.press_button("A")
        yield
    while get_game_state_symbol() not in ("CB2_POKEBLOCKMENU", "SUB_810B674"):
        yield
    for _ in range(22):
        yield

    if len(get_pokeblocks()) <= poke_block_index:
        raise RuntimeError(
            f"Cannot use Pokéblock with index '{poke_block_index}' because the Pokéblock Case only contains {len(get_pokeblocks())} blocks."
        )

    def _get_pokeblock_scroll_position() -> int:
        if context.rom.is_rs:
            data = read_symbol("gUnknown_02039248")
            position_offset = data[0]
            scroll_offset = data[1]
        else:
            data = read_symbol("sSavedPokeblockData")
            position_offset = unpack_uint16(data[4:6])
            scroll_offset = unpack_uint16(data[6:8])
        return position_offset + scroll_offset

    while _get_pokeblock_scroll_position() != poke_block_index:
        if _get_pokeblock_scroll_position() < poke_block_index:
            context.emulator.press_button("Down")
        else:
            context.emulator.press_button("Up")
        yield
        yield

    while get_game_state_symbol() in ("CB2_POKEBLOCKMENU", "SUB_810B674"):
        context.emulator.press_button("A")
        yield
    while get_game_state_symbol() != "BATTLEMAINCB2":
        context.emulator.press_button("B")
        yield
    while get_battle_controller_callback(0) == "CompleteWhenChosePokeblock":
        yield
    yield


@debug.track
def battle_action_use_move(
    action: TurnAction, battler_index: int, move_index: int, battle_state: BattleState
) -> Generator:
    diagnostic_print(
        lambda: (
            "BATTLE_EXECUTOR: "
            f"source={getattr(context, 'battle_decision_source', None)} "
            f"action={getattr(action, 'name', action)} move_index={move_index}"
        ),
        trace=True,
    )
    if context.rom.is_rs:
        yield

    # Choose the 'Fight' option.
    if get_battle_controller_callback(battler_index) not in (
        "HandleInputChooseMove",
        "HandleAction_ChooseMove",
    ):
        yield from scroll_to_battle_action(0)
        yield
        context.emulator.press_button("A")
    for _ in range(6 if context.rom.is_rse else 8):
        yield

    # It's possible the game does not actually offer the move list at this point, for example
    # if the Pokémon is completely out of PP. In that case we only get a message saying that
    # Struggle is being used. So we just want to confirm that message and then stop the further
    # execution of this function.
    if get_battle_controller_callback(battler_index) not in (
        "HandleInputChooseMove",
        "HandleAction_ChooseMove",
    ):
        while get_main_battle_callback() in (
            "HandleTurnActionSelectionState",
            "sub_8012324",
        ) and get_battle_controller_callback(battler_index) not in (
            "HandleInputChooseMove",
            "HandleAction_ChooseMove",
        ):
            context.emulator.press_button("A")
            yield
        return

    # Select move and target
    yield from scroll_to_move(move_index, battler_index > 1)
    while get_battle_controller_callback(battler_index) in (
        "HandleInputChooseMove",
        "HandleAction_ChooseMove",
    ):
        context.emulator.press_button("A")
        yield
    diagnostic_print(
        lambda: (
            "BATTLE_ACTION_ACCEPTED: "
            f"move_index={move_index} callback={get_battle_controller_callback(battler_index)}"
        ),
        trace=True,
    )
    if action is TurnAction.UseMove:
        target_index = 1
        if battle_state.opponent.left_battler is None:
            raise RuntimeError("Cannot attack the left-side opponent Pokémon because there isn't one.")
    elif action is TurnAction.UseMoveAgainstRightSideOpponent:
        target_index = 3
        if battle_state.opponent.right_battler is None:
            raise RuntimeError("Cannot attack the right-side opponent Pokémon because there isn't one.")
    elif action is TurnAction.UseMoveAgainstPartner:
        target_index = 0 if battler_index == 2 else 2
        if (battler_index == 0 and battle_state.own_side.right_battler is None) or (
            battler_index == 2 and battle_state.own_side.left_battler is None
        ):
            raise RuntimeError("Cannot attack the partnering Pokémon because there isn't one.")
    else:
        target_index = 1
    while get_battle_controller_callback(battler_index) == "HandleInputChooseTarget":
        current_target = unpack_uint32(read_symbol("gMultiUsePlayerCursor"))
        if current_target == target_index:
            context.emulator.press_button("A")
            yield
        else:
            context.emulator.press_button("Down")
            yield
