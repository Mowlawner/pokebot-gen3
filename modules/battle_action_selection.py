from typing import Generator

from modules.battle_menuing import scroll_to_battle_action, scroll_to_move
from modules.battle_observation import record_selected_battle_action
from modules.battle_identity import battle_pokemon_display_name, battle_pokemon_identity
from modules.battle_planner import normalized_battle_objective
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
from modules.console import (
    diagnostic_print,
    print_battle_action,
    print_battle_safety_stop,
    print_battle_switch_resolution,
)
from modules.debug import debug
from modules.game import decode_string
from modules.items import Item, ItemBattleUse, get_item_bag, get_pokeblocks
from modules.memory import (
    read_symbol,
    unpack_uint32,
    get_game_state,
    GameState,
    get_game_state_symbol,
    unpack_uint16,
)
from modules.menuing import get_current_party_menu_index, scroll_to_item_in_bag, scroll_to_party_menu_index
from modules.pokemon_party import get_opponent_party, get_party, get_party_size
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
_BATTLE_ITEM_TRANSITION_TIMEOUT = 180
_BATTLE_ITEM_RESOLUTION_TIMEOUT = 240
_BATTLE_MOVE_CALLBACK_TIMEOUT = 180
_BATTLE_TARGET_SELECTION_TIMEOUT = 120
_BATTLE_NO_PROGRESS_LIMIT = 3

_last_battle_replacement_trace_signature = None


def _battle_progress_signature(battle_state: BattleState) -> tuple:
    """Return the battle facts that should change after a completed turn."""

    try:
        own = getattr(battle_state, "own_side", None)
        opponent = getattr(battle_state, "opponent", None)
        battlers = tuple(
            battler
            for side in (own, opponent)
            for battler in (
                getattr(side, "left_battler", None),
                getattr(side, "right_battler", None),
            )
            if battler is not None
        )
        return tuple(
            (
                getattr(battler, "party_index", None),
                getattr(getattr(battler, "species", None), "name", None),
                getattr(battler, "current_hp", None),
                getattr(getattr(battler, "status_permanent", None), "value", None),
                tuple(getattr(move, "pp", None) for move in getattr(battler, "moves", ())),
            )
            for battler in battlers
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        # A transient ROM read is not evidence of a stalled turn. Returning a
        # unique value lets the normal battle loop reevaluate on the next
        # frame instead of stopping on incomplete memory.
        return ("unreadable", id(battle_state))


class _BattleProgressGuard:
    """Bound identical action/state repetitions at the executor boundary."""

    def __init__(self, limit: int = _BATTLE_NO_PROGRESS_LIMIT):
        self.limit = limit
        self._last = None
        self.repetitions = 0

    def observe(self, state_signature: tuple, action, target) -> bool:
        current = (state_signature, action, target)
        if current == self._last:
            self.repetitions += 1
        else:
            self._last = current
            self.repetitions = 0
        return self.repetitions >= self.limit

    def reset(self) -> None:
        self._last = None
        self.repetitions = 0


def _battle_item_target_has_effect(
    item: Item, target_index: int | None, battle_state: BattleState | None = None
) -> bool:
    """Revalidate a healing target immediately before opening the bag.

    Strategy decisions are made from a snapshot. A previous action, a battle
    callback transition, or a stale menu can make that snapshot obsolete. Do
    not spend a battle turn on an item that cannot change the target.
    """

    if target_index is None:
        return False
    try:
        target = None
        if battle_state is not None:
            active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
            if getattr(active, "party_index", None) == target_index:
                target = active
        if target is None:
            target = get_party()[target_index]
        current_hp = int(getattr(target, "current_hp", 0))
        total_hp = int(getattr(target, "total_hp", current_hp))
        name = str(getattr(item, "name", "")).casefold()
        full_hp = name in {"full restore", "max potion"}
        hp_effect = full_hp or int(getattr(item, "parameter", 0) or 0) > 0
        status = getattr(getattr(target, "status_condition", None), "value", None)
        status_cures = {
            "antidote": {"poison", "badly poisoned"},
            "burn heal": {"burned"},
            "ice heal": {"frozen"},
            "awakening": {"asleep"},
            "paralyze heal": {"paralysed"},
        }
        cures_status = name in {"full heal", "full restore"} or status in status_cures.get(name, set())
        return (hp_effect and current_hp < total_hp) or (cures_status and status not in (None, "none"))
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False


def _battle_move_is_executable(
    battle_state: BattleState, battler_index: int, move_index: int
) -> tuple[bool, str]:
    """Validate a move decision at the exact menu boundary.

    A planner decision is made from a snapshot, but PP, Disable, fainting, or
    a battler projection can change before the executor opens Fight. Returning
    a reason lets the caller discard the stale action and replan instead of
    selecting an invalid cursor position and leaving the battle menu open.
    """

    try:
        own_side = battle_state.own_side
        battler = own_side.right_battler if battler_index == 2 else own_side.active_battler
        if battler is None:
            return False, "player battler is unavailable"
        if not isinstance(move_index, int) or isinstance(move_index, bool):
            return False, f"move index {move_index!r} is not an integer"
        moves = getattr(battler, "moves", ())
        if move_index < 0 or move_index >= len(moves):
            return False, f"move index {move_index!r} is outside the visible move list"
        learned_move = moves[move_index]
        if learned_move is None or getattr(learned_move, "move", None) is None:
            return False, f"move slot {move_index} is empty"
        if getattr(learned_move, "pp", 0) <= 0:
            return False, f"{learned_move.move.name} has no PP remaining"
        disabled_move = getattr(battler, "disabled_move", None)
        if disabled_move is learned_move.move and getattr(battler, "disabled_turns_remaining", 1) > 0:
            return False, f"{learned_move.move.name} is disabled"
        if getattr(battler, "current_hp", 1) <= 0:
            return False, "player battler has fainted"
        return True, "move is executable"
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as error:
        return False, f"move validation failed: {type(error).__name__}"


def _battle_executor_timeout(
    *, phase: str, action, battler_index: int, move_index: int, battle_state: BattleState
) -> RuntimeError:
    """Build a visible error when ROM-owned action input stops progressing."""

    try:
        frame = context.emulator.get_frame_count()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        frame = None
    callback = _safe_callback(battler_index)
    main_callback = get_main_battle_callback()
    script = get_current_battle_script_instruction()
    game_state = get_game_state()
    diagnostic_print(
        lambda: (
            "BATTLE_EXECUTOR_TIMEOUT: "
            f"phase={phase!r} action={getattr(action, 'name', action)!r} "
            f"move_index={move_index!r} battler_index={battler_index!r} "
            f"callback={callback!r} main_callback={main_callback!r} script={script!r} "
            f"game_state={game_state!r} emulator_frame={frame!r}"
        ),
        trace=True,
        prefix="BATTLE_EXECUTOR_TIMEOUT:",
    )
    return RuntimeError(
        "Battle action executor stopped progressing: "
        f"phase={phase!r} action={getattr(action, 'name', action)!r} move_index={move_index!r} "
        f"battler_index={battler_index!r} callback={callback!r} main_callback={main_callback!r} "
        f"script={script!r} game_state={game_state!r} emulator_frame={frame!r}"
    )


def _close_battle_party_menu() -> Generator:
    """Advance the party menu back to battle with a bounded confirmation loop."""

    for _ in range(_BATTLE_ITEM_TRANSITION_TIMEOUT):
        if get_game_state() != GameState.PARTY_MENU:
            return
        context.emulator.press_button("A")
        yield
    raise RuntimeError(
        "Battle party menu did not close after confirmation: "
        f"callback={_safe_callback(0)!r} main_callback={get_main_battle_callback()!r} "
        f"script={get_current_battle_script_instruction()!r} "
        f"game_state={get_game_state()!r} emulator_frame={context.emulator.get_frame_count()!r}"
    )


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


def _strategy_requires_planner_owned_actions(strategy: BattleStrategy) -> bool:
    """Return whether a strategy owns the complete action decision.

    Campaign strategies expose this contract explicitly.  Legacy strategies
    retain their existing matchup validation and fallback behavior.
    """

    try:
        return bool(getattr(strategy, "requires_planner_owned_actions")())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _mechanical_switch_validation(battle_state: BattleState, party_index: int) -> tuple[bool, str]:
    """Validate only ROM/menu legality for a planner-owned switch."""

    try:
        active_indices = _active_party_indices(battle_state)
        if party_index in active_indices:
            return False, "target is already active"
        party = get_party()
        if not isinstance(party_index, int) or isinstance(party_index, bool):
            return False, "target party slot is not an integer"
        if party_index < 0 or party_index >= len(party):
            return False, "target party slot is outside the observed party"
        target = party[party_index]
        if getattr(target, "is_egg", False):
            return False, "target is an egg"
        if getattr(target, "current_hp", 0) <= 0:
            return False, "target has 0 HP"
        return True, "mechanically executable"
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as error:
        return False, f"mechanical switch validation failed: {type(error).__name__}"


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
    """Find one executable action after rejecting a stale switch decision."""

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


def _battle_species_name(battler, party=None) -> str:
    try:
        return battle_pokemon_display_name(battler, party=party)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return "unknown"


def _battle_party_member_name(index: int) -> str:
    try:
        party = get_party()
        return _battle_species_name(party[index], party)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return f"party slot {index}"


def _switch_matchup_names(battle_state: BattleState, battler_index: int) -> tuple[str, str]:
    own_side = getattr(battle_state, "own_side", None)
    active = (
        getattr(own_side, "right_battler", None)
        if battler_index == 2
        else getattr(own_side, "active_battler", None)
    )
    opponent = _battle_action_target(battle_state, TurnAction.RotateLead, active)
    return (
        _battle_species_name(active, _safe_battle_party(get_party)),
        _battle_species_name(opponent, _safe_battle_party(get_opponent_party)),
    )


def _log_switch_resolution(
    *,
    battle_state: BattleState,
    battler_index: int,
    status: str,
    requested_index: int,
    final_index: int | None = None,
    observed_index: int | None = None,
    observed_name: str | None = None,
    reason: str,
) -> None:
    active_name, opponent_name = _switch_matchup_names(battle_state, battler_index)
    print_battle_switch_resolution(
        status=status,
        requested=_battle_party_member_name(requested_index),
        requested_index=requested_index,
        final=_battle_party_member_name(final_index) if final_index is not None else None,
        final_index=final_index,
        observed=observed_name,
        observed_index=observed_index,
        active=active_name,
        opponent=opponent_name,
        reason=reason,
    )


def _register_pending_switch(
    *,
    battle_state: BattleState,
    battler_index: int,
    requested_index: int,
    final_index: int,
    reason: str,
) -> None:
    active_name, opponent_name = _switch_matchup_names(battle_state, battler_index)
    pending = getattr(context, "battle_pending_switches", None)
    if not isinstance(pending, list):
        pending = []
        context.battle_pending_switches = pending
    existing = next(
        (
            request
            for request in pending
            if request.get("battler_index") == battler_index
        ),
        None,
    )
    if existing is not None:
        diagnostic_print(
            lambda: (
                "BATTLE_SWITCH_RECONCILIATION: "
                f"status='blocked_pending' battler_index={battler_index!r} "
                f"transaction={existing.get('transaction_id')!r} target={existing.get('final_index')!r}"
            ),
            trace=True,
            prefix="BATTLE_SWITCH_RECONCILIATION",
        )
        return
    context.battle_switch_transaction_sequence = getattr(context, "battle_switch_transaction_sequence", 0) + 1
    transaction_id = context.battle_switch_transaction_sequence
    active_battler = (
        getattr(getattr(battle_state, "own_side", None), "right_battler", None)
        if battler_index == 2
        else getattr(getattr(battle_state, "own_side", None), "active_battler", None)
    )
    target_pokemon = _capture_party_member(final_index)
    target_identity = _battle_identity_trace(target_pokemon, _safe_battle_party(get_party))
    before_identity = _battle_identity_trace(active_battler, _safe_battle_party(get_party))
    pending.append(
        {
            "transaction_id": transaction_id,
            "battler_index": battler_index,
            "requested_index": requested_index,
            "requested_name": _battle_party_member_name(requested_index),
            "requested_identity": target_identity,
            "final_index": final_index,
            "final_name": _battle_party_member_name(final_index),
            "final_identity": target_identity,
            "before_index": getattr(
                getattr(
                    getattr(battle_state, "own_side", None),
                    "right_battler" if battler_index == 2 else "active_battler",
                    None,
                ),
                "party_index",
                None,
            ),
            "before_identity": before_identity,
            "active": active_name,
            "opponent": opponent_name,
            "reason": reason,
        }
    )
    context.battle_switch_reconciliation_status = "pending"


def _safe_battle_party(party_reader):
    """Read a party for diagnostics without making logging emulator-dependent."""

    try:
        return party_reader() or ()
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return ()


def _battle_identity_trace(battler, party=None) -> tuple[str, str | None, int | None]:
    try:
        return battle_pokemon_identity(battler, party=party)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return "unknown", None, None


def _battle_action_target(battle_state: BattleState, action, active):
    """Resolve the battler actually targeted by a selected action.

    ``active_battler`` is a convenience property for solo battles. Using it
    for every double-battle log makes a right-side target appear to be the
    left-side opponent, and can make duplicate species look like the action
    targeted the wrong Pokémon. Keep the display tied to the action's ROM
    target semantics.
    """

    own_side = getattr(battle_state, "own_side", None)
    opponent_side = getattr(battle_state, "opponent", None)
    if action is TurnAction.UseMoveAgainstRightSideOpponent:
        return getattr(opponent_side, "right_battler", None)
    if action is TurnAction.UseMoveAgainstPartner:
        partner_for = getattr(own_side, "partner_for", None)
        if callable(partner_for):
            return partner_for(active)
        return getattr(own_side, "right_battler", None)
    if getattr(battle_state, "is_double_battle", False):
        return getattr(opponent_side, "left_battler", None) or getattr(opponent_side, "active_battler", None)
    return getattr(opponent_side, "active_battler", None)


def _battle_hp_status(battler) -> tuple[str | None, str | None]:
    """Return the live HP and permanent status without speculative reads."""

    if battler is None:
        return None, None
    try:
        current = getattr(battler, "current_hp", None)
        maximum = getattr(battler, "total_hp", getattr(battler, "max_hp", None))
        hp = f"{current}/{maximum}" if current is not None and maximum is not None else None
        status = getattr(getattr(battler, "status_permanent", None), "value", None)
        if status is None:
            status = getattr(battler, "status", None)
        return hp, str(status) if status is not None else None
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None, None


def _next_trainer_opponent(battle_state: BattleState):
    """Return the incoming trainer Pokémon represented by the replacement prompt."""

    try:
        current = battle_state.opponent.active_battler
        current_index = current.party_index
        return next(
            (
                pokemon
                for pokemon in (get_opponent_party() or ())
                if getattr(pokemon, "index", -1) > current_index
                and not getattr(pokemon, "is_empty", False)
                and getattr(pokemon, "is_valid", True)
                and getattr(pokemon, "current_hp", 0) > 0
            ),
            None,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return None


def _battle_action_details(
    battle_state: BattleState,
    strategy: BattleStrategy,
    action,
    index,
    battler_index: int,
) -> tuple[str, ...]:
    """Normalize a strategy result into GUI- and trace-friendly language."""

    capture_target = bool(getattr(battle_state, "nuzlocke_capture_target", False))
    try:
        capture = capture_target
        if capture:
            try:
                capture = any(
                    getattr(ball, "quantity", 0) > 0
                    for ball in get_item_bag().poke_balls
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                # A synthetic or transient action snapshot may not expose the
                # bag. Preserve the encounter objective until an authoritative
                # planner boundary can resolve its resources.
                pass
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        capture = False
    objective = normalized_battle_objective(
        battle_state,
        capture_target=capture_target,
        capture_available=capture,
    )
    intent = "select action"
    target = "KO"
    action_label = getattr(action, "name", str(action))

    own_side = getattr(battle_state, "own_side", None)
    if battler_index == 2:
        active = getattr(own_side, "right_battler", None)
    else:
        active = getattr(own_side, "active_battler", None)
    opponent = _battle_action_target(battle_state, action, active)
    active_party = _safe_battle_party(get_party)
    opponent_party = _safe_battle_party(get_opponent_party)
    active_name = _battle_species_name(active, active_party)
    opponent_name = _battle_species_name(opponent, opponent_party)

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
            replacement_name = _battle_species_name(replacement, get_party())
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            replacement_name = f"party slot {index!r}"
        action_label = f"{action_label}({replacement_name})"
    elif action is TurnAction.UseItem:
        intent = "select item"
        item = index[0] if isinstance(index, tuple) else index
        target_index = index[1] if isinstance(index, tuple) and len(index) > 1 else None
        item_name = getattr(item, "name", str(item))
        target_suffix = ""
        if target_index is not None:
            try:
                target_suffix = f" -> {_battle_species_name(get_party()[target_index], get_party())}"
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                target_suffix = f" -> party slot {target_index}"
        action_label = f"{action_label}({item_name}{target_suffix})"
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
    active = battle_state.own_side.right_battler if battler_index == 2 else battle_state.own_side.active_battler
    opponent_target = _battle_action_target(battle_state, action, active)
    active_hp, active_status = _battle_hp_status(active)
    opponent_hp, opponent_status = _battle_hp_status(opponent_target)
    if action in (
        TurnAction.UseMove,
        TurnAction.UseMoveAgainstRightSideOpponent,
        TurnAction.UseMoveAgainstPartner,
    ):
        record_selected_battle_action("move", action_label)
    elif action is TurnAction.UseItem:
        record_selected_battle_action("item", action_label)
    elif action is TurnAction.RotateLead:
        record_selected_battle_action("switch", action_label)
    message = (
        f"BATTLE ACTION: objective={objective}; target={target}; intent={intent}; "
        f"active={active_name}; opponent={opponent_name}; action={action_label}; "
        f"source={source}; reason={reason}"
    )
    context.message = message
    # The GUI consumes context.message, while Rich gets an aligned panel for
    # human-readable output. Keep the structured trace below for diagnostics.
    print_battle_action(
        objective=objective,
        target=target,
        intent=intent,
        active=active_name,
        opponent=opponent_name,
        action=action_label,
        source=source,
        evaluator=getattr(context, "battle_evaluation_method", None),
        reason=reason,
        active_hp=active_hp,
        active_status=active_status,
        opponent_hp=opponent_hp,
        opponent_status=opponent_status,
    )
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


def _log_battle_action_lifecycle(
    phase: str,
    battle_state: BattleState,
    strategy: BattleStrategy,
    action,
    index,
    battler_index: int,
    *,
    reason: str | None = None,
) -> None:
    """Trace action selection separately from validation and input dispatch."""

    details = _battle_action_details(battle_state, strategy, action, index, battler_index)
    active_name, opponent_name = details[-2:]
    active_party = _safe_battle_party(get_party)
    opponent_party = _safe_battle_party(get_opponent_party)
    active = battle_state.own_side.right_battler if battler_index == 2 else battle_state.own_side.active_battler
    target = _battle_action_target(battle_state, action, active)
    active_species, active_nickname, active_index = _battle_identity_trace(active, active_party)
    opponent_species, opponent_nickname, opponent_index = _battle_identity_trace(target, opponent_party)
    diagnostic_print(
        lambda: (
            "BATTLE_ACTION_LIFECYCLE: "
            f"phase={phase!r} action={details[3]!r} active={active_name!r} opponent={opponent_name!r} "
            f"active_species={active_species!r} active_nickname={active_nickname!r} active_party_index={active_index!r} "
            f"opponent_species={opponent_species!r} opponent_nickname={opponent_nickname!r} opponent_party_index={opponent_index!r} "
            f"reason={reason or getattr(context, 'battle_decision_detail', None)!r}"
        ),
        trace=True,
        prefix="BATTLE_ACTION_LIFECYCLE:",
    )


def _log_trainer_replacement_choice(battle_state: BattleState, replacement_index: int | None, source: str) -> None:
    """Expose the Yes/No replacement decision in the same GUI vocabulary."""

    current = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
    player_party = _safe_battle_party(get_party)
    opponent_party = _safe_battle_party(get_opponent_party)
    defeated_opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
    incoming_opponent = _next_trainer_opponent(battle_state)
    active_opponent_after_switch = incoming_opponent or defeated_opponent
    replacement_name = "stay active"
    target = "keep current matchup"
    intent = "select No"
    if replacement_index is not None:
        try:
            replacement_name = _battle_species_name(get_party()[replacement_index], player_party)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            replacement_name = f"party slot {replacement_index}"
        target = "safe matchup / EXP rotation"
        intent = "select Yes and switch"
    reason = getattr(context, "battle_decision_detail", None) or "replacement policy found no better legal option"
    reason = (
        f"defeated_opponent={_battle_species_name(defeated_opponent, opponent_party)} "
        f"(previous opponent={_battle_species_name(defeated_opponent, opponent_party)}); "
        f"incoming_opponent={_battle_species_name(incoming_opponent, opponent_party)} "
        f"(incoming opponent={_battle_species_name(incoming_opponent, opponent_party)}); "
        f"active_opponent_after_switch={_battle_species_name(active_opponent_after_switch, opponent_party)}; {reason}"
    )
    decision_source = getattr(context, "battle_decision_source", None) or source
    context.message = (
        f"BATTLE ACTION: objective=defeat; target={target}; intent={intent}; "
        f"active={_battle_species_name(current, player_party)}; opponent={_battle_species_name(active_opponent_after_switch, opponent_party)}; "
        f"action={'switch to ' + replacement_name if replacement_index is not None else 'stay active'}; "
        f"source={decision_source}; reason={reason}"
    )
    print_battle_action(
        objective="defeat",
        target=target,
        intent=intent,
        active=_battle_species_name(current, player_party),
        opponent=_battle_species_name(active_opponent_after_switch, opponent_party),
        action=("switch to " + replacement_name if replacement_index is not None else "stay active"),
        source=decision_source,
        evaluator=getattr(context, "battle_evaluation_method", None),
        reason=reason,
        active_hp=_battle_hp_status(current)[0],
        active_status=_battle_hp_status(current)[1],
        opponent_hp=_battle_hp_status(active_opponent_after_switch)[0],
        opponent_status=_battle_hp_status(active_opponent_after_switch)[1],
    )
    diagnostic_print(
        lambda: (
            "BATTLE_ACTION_CHOSEN: objective='defeat' "
            f"target={target!r} intent={intent!r} active={_battle_species_name(current, player_party)!r} "
            f"opponent={_battle_species_name(active_opponent_after_switch, opponent_party)!r} replacement={replacement_name!r} "
            f"defeated_opponent={_battle_species_name(defeated_opponent, opponent_party)!r} "
            f"incoming_opponent={_battle_species_name(incoming_opponent, opponent_party)!r} "
            f"active_opponent_after_switch={_battle_species_name(active_opponent_after_switch, opponent_party)!r} "
            f"source={decision_source!r} reason={reason!r}"
        ),
        trace=True,
        prefix="BATTLE_ACTION_CHOSEN:",
    )
    if replacement_index is not None:
        record_selected_battle_action("free_switch", f"switch to {replacement_name}")


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
    previous_battler_index = None
    progress_guard = _BattleProgressGuard()
    # Emerald advances from PlayerHandleYesNoBox to
    # PlayerHandleYesNoInput after it creates the replacement prompt. Keep
    # both callbacks in the same transaction so the handler does not miss the
    # prompt between frames and leave the ROM's default No selected.
    trainer_replacement_callbacks = _TRAINER_REPLACEMENT_CALLBACKS
    trainer_replacement_buffer_step = None
    trainer_replacement_decision = _TRAINER_REPLACEMENT_UNSET
    trainer_replacement_pokemon = None
    trainer_replacement_resolved = False

    while battle_is_active():
        # Battle actions mutate the ROM-owned battler structures.  The
        # previous implementation kept the initial snapshot for the entire
        # handler generator, so a Potion could be selected repeatedly from
        # the same pre-heal HP value after the first use completed.
        battle_state = get_battle_state()
        if not battle_action_selection_is_ready(battle_state):
            break
        progress_signature = _battle_progress_signature(battle_state)
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
                        yield from _close_battle_party_menu()
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
                _log_trainer_replacement_choice(battle_state, None, "TRAINER REPLACEMENT POLICY")
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
                    yield from _close_battle_party_menu()
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

        # Campaign Progression may need to wait for a transiently incomplete
        # battle boundary. A retry request deliberately uses the existing
        # SwitchToManual transport without changing bot mode or pressing a
        # battle button; the next frame rebuilds the live planner facts.
        if (
            action is TurnAction.SwitchToManual
            and getattr(context, "battle_planner_retry_requested", False)
        ):
            context.battle_planner_retry_requested = False
            diagnostic_print(
                lambda: (
                    "BATTLE_PLANNER_RETRY: "
                    f"status='wait_for_next_frame' retry={getattr(context, 'battle_planner_retry_count', None)!r}"
                ),
                trace=True,
                prefix="BATTLE_PLANNER_RETRY:",
            )
            yield
            continue

        if action is TurnAction.RotateLead:
            pending_switches = getattr(context, "battle_pending_switches", None)
            pending_for_battler = next(
                (
                    request
                    for request in (pending_switches or ())
                    if request.get("battler_index") == battler_index
                ),
                None,
            )
            if pending_for_battler is not None:
                context.battle_decision_source = "BATTLE SWITCH RECONCILIATION"
                context.battle_decision_detail = (
                    f"deferred switch until transaction {pending_for_battler.get('transaction_id')!r} "
                    f"to party slot {pending_for_battler.get('final_index')!r} is observed"
                )
                context.message = f"BATTLE REPLAN: waiting for switch transaction {pending_for_battler.get('transaction_id')!r}"
                diagnostic_print(
                    lambda: (
                        "BATTLE_SWITCH_RECONCILIATION: "
                        f"status='blocked_pending' battler_index={battler_index!r} "
                        f"transaction={pending_for_battler.get('transaction_id')!r} "
                        f"requested_target={index!r}"
                    ),
                    trace=True,
                    prefix="BATTLE_SWITCH_RECONCILIATION",
                )
                yield
                continue

        if action in (
            TurnAction.UseMove,
            TurnAction.UseMoveAgainstRightSideOpponent,
            TurnAction.UseMoveAgainstPartner,
        ):
            move_is_valid, move_reason = _battle_move_is_executable(battle_state, battler_index, index)
            if not move_is_valid:
                # Do not let a stale planner result select an empty, exhausted,
                # or disabled move slot. The action callback is still at the
                # turn boundary, so the next iteration can rebuild the planner
                # facts and choose another legal one-action turn.
                context.battle_decision_source = "BATTLE ACTION REPLAN"
                context.battle_decision_detail = f"discarded stale move action: {move_reason}"
                context.message = f"BATTLE REPLAN: {move_reason}"
                diagnostic_print(
                    lambda: (
                        "BATTLE_MOVE_REJECTED: "
                        f"battler_index={battler_index!r} move_index={index!r} reason={move_reason!r}"
                    ),
                    trace=True,
                    prefix="BATTLE_MOVE_REJECTED:",
                )
                yield
                continue

        if progress_guard.observe(progress_signature, action, index):
            context.battle_decision_source = "BATTLE PROGRESS GUARD"
            context.battle_decision_detail = (
                f"the same action {getattr(action, 'name', action)}({index!r}) produced no observable "
                f"battle-state change for {progress_guard.repetitions + 1} consecutive decisions"
            )
            context.message = (
                "BATTLE SAFETY STOP: no observable progress after repeated identical actions; "
                "manual review required"
            )
            try:
                _, _, _, action_label, _, _, active_name, opponent_name = _battle_action_details(
                    battle_state, strategy, action, index, battler_index
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
                action_label = f"{getattr(action, 'name', action)}({index!r})"
                active_name = "unknown"
                opponent_name = "unknown"
            target_label = (
                f"party slot {index}"
                if action is TurnAction.RotateLead
                else str(index)
            )
            print_battle_safety_stop(
                action=action_label,
                target=target_label,
                repetitions=progress_guard.repetitions + 1,
                active=active_name,
                opponent=opponent_name,
                reason=context.battle_decision_detail,
            )
            diagnostic_print(
                lambda: (
                    "BATTLE_PROGRESS_GUARD: stopping autonomous input; "
                    f"action={getattr(action, 'name', action)!r} target={index!r} "
                    f"repetitions={progress_guard.repetitions + 1!r}"
                ),
                trace=True,
                prefix="BATTLE_PROGRESS_GUARD:",
            )
            context.set_manual_mode()
            return

        _log_battle_action_chosen(battle_state, strategy, action, index, battler_index)

        planner_owned_actions = _strategy_requires_planner_owned_actions(strategy)

        if context.bot_mode == "Manual":
            yield
            continue

        # In double battles, both player Pokémon will do their inputs in the same run. So the active
        # battler might change in the middle of this function. If we detect that, we update the battle
        # state as the previous Pokémon's actions might have changed it (e.g. by using potions.)
        if previous_battler_index is None:
            previous_battler_index = battler_index
        elif previous_battler_index != battler_index:
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
                _log_battle_action_lifecycle("validated", battle_state, strategy, action, index, battler_index)
                _log_battle_action_lifecycle("dispatched", battle_state, strategy, action, index, battler_index)
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

                if index.battle_use is ItemBattleUse.Healing and not _battle_item_target_has_effect(
                    index, target_index, battle_state
                ):
                    # The action became stale before the menu opened. Keep
                    # the battle action boundary alive and replan from the
                    # current party state instead of repeatedly opening and
                    # closing the bag on a full or already-cured target.
                    context.battle_decision_source = "BATTLE ACTION REPLAN"
                    context.battle_decision_detail = (
                        f"discarded stale {index.name} action; target party slot {target_index} "
                        "has no remaining HP or status effect for that item"
                    )
                    context.message = f"BATTLE REPLAN: {index.name} no longer improves the selected target"
                    diagnostic_print(
                        lambda: (
                            "BATTLE_ITEM_REJECTED: "
                            f"item={index.name!r} target_index={target_index!r} reason='no target effect'"
                        ),
                        trace=True,
                        prefix="BATTLE_ITEM_REJECTED:",
                    )
                    yield
                    continue

                _log_battle_action_lifecycle(
                    "validated", battle_state, strategy, action, (index, target_index), battler_index
                )
                _log_battle_action_lifecycle(
                    "dispatched", battle_state, strategy, action, (index, target_index), battler_index
                )
                yield from battle_action_use_item(battle_state, index, target_index)

            case TurnAction.RotateLead:
                requested_switch_index = index
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= get_party_size()
                ):
                    reason = f"requested party slot {index!r} is outside the observed party"
                    _log_switch_resolution(
                        battle_state=battle_state,
                        battler_index=battler_index,
                        status="rejected",
                        requested_index=requested_switch_index,
                        reason=reason,
                    )
                    if planner_owned_actions:
                        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX REPLAN"
                        context.battle_decision_detail = f"discarded stale switch: {reason}"
                        context.message = f"CAMPAIGN EXPECTIMINIMAX REPLAN: {reason}"
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_PLANNER_ACTION_REJECTED: "
                                f"action='switch' target={index!r} reason={reason!r} next='replan'"
                            ),
                            trace=False,
                        )
                        yield
                        continue
                    raise RuntimeError(
                        f"Cannot switch in party slot #{index}: {reason}."
                    )

                active_indices = _active_party_indices(battle_state)
                if index in active_indices:
                    if planner_owned_actions:
                        reason = "requested target was already active"
                        _log_switch_resolution(
                            battle_state=battle_state,
                            battler_index=battler_index,
                            status="rejected",
                            requested_index=requested_switch_index,
                            reason=reason,
                        )
                        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX REPLAN"
                        context.battle_decision_detail = f"discarded stale switch: {reason}"
                        context.message = f"CAMPAIGN EXPECTIMINIMAX REPLAN: {reason}"
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_PLANNER_ACTION_REJECTED: "
                                f"action='switch' target={index!r} reason={reason!r} next='replan'"
                            ),
                            trace=False,
                        )
                        yield
                        continue
                    # A stale strategy decision must be replanned before the
                    # party menu is opened.  Sending SHIFT for an active slot
                    # is not recoverable in Emerald: the ROM leaves the menu
                    # open and repeats "already in battle" forever.
                    selector = getattr(strategy, "select_valid_switch_target", None)
                    fallback_index = selector(battle_state) if callable(selector) else None
                    if fallback_index is not None and fallback_index not in _active_party_indices(battle_state):
                        _log_switch_resolution(
                            battle_state=battle_state,
                            battler_index=battler_index,
                            status="replanned",
                            requested_index=requested_switch_index,
                            final_index=fallback_index,
                            reason="requested target was already active; selector supplied a legal replacement",
                        )
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
                        _log_switch_resolution(
                            battle_state=battle_state,
                            battler_index=battler_index,
                            status="rejected",
                            requested_index=requested_switch_index,
                            reason="requested target was already active and no legal switch fallback was found",
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
                    _log_switch_resolution(
                        battle_state=battle_state,
                        battler_index=battler_index,
                        status="rejected",
                        requested_index=requested_switch_index,
                        reason="selector returned an already-active party slot",
                    )
                    diagnostic_print(
                        lambda: (
                            "BATTLE_SWITCH_REJECTED: "
                            f"requested={index!r} reason='selector returned active slot'"
                        ),
                        trace=True,
                    )
                    if planner_owned_actions:
                        reason = "selector returned an already-active party slot"
                        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX REPLAN"
                        context.battle_decision_detail = f"discarded stale switch: {reason}"
                        context.message = f"CAMPAIGN EXPECTIMINIMAX REPLAN: {reason}"
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_PLANNER_ACTION_REJECTED: "
                                f"action='switch' target={index!r} reason={reason!r} next='replan'"
                            ),
                            trace=False,
                        )
                        yield
                        continue
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
                switch_rejection_reason = "strategy validator rejected the target"
                validator = getattr(strategy, "is_switch_target_valid", None)
                if planner_owned_actions:
                    is_valid_switch, switch_rejection_reason = _mechanical_switch_validation(
                        battle_state, index
                    )
                elif not callable(validator):
                    # Third-party/legacy strategies may implement the
                    # original BattleStrategy contract without the optional
                    # matchup validator. Preserve their established switch
                    # behavior; built-in strategies all provide validation.
                    is_valid_switch = True
                else:
                    is_valid_switch = bool(validator(battle_state, index))
                if not is_valid_switch:
                    if planner_owned_actions:
                        _log_switch_resolution(
                            battle_state=battle_state,
                            battler_index=battler_index,
                            status="rejected",
                            requested_index=requested_switch_index,
                            reason=switch_rejection_reason,
                        )
                        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX REPLAN"
                        context.battle_decision_detail = (
                            f"discarded stale switch to {_battle_party_member_name(index)}; "
                            f"{switch_rejection_reason}"
                        )
                        context.message = (
                            f"CAMPAIGN EXPECTIMINIMAX REPLAN: switch target {_battle_party_member_name(index)} "
                            f"is not mechanically executable ({switch_rejection_reason})"
                        )
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_PLANNER_ACTION_REJECTED: "
                                f"action='switch' target={index!r} reason={switch_rejection_reason!r} next='replan'"
                            ),
                            trace=False,
                        )
                        yield
                        continue
                    switch_rejection_reason = "strategy validator rejected the target"
                    reason_getter = getattr(strategy, "switch_target_validation_reason", None)
                    if callable(reason_getter):
                        try:
                            switch_rejection_reason = str(reason_getter(battle_state, index))
                        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
                            switch_rejection_reason = "switch validator rejected the target without a readable reason"
                    selector = getattr(strategy, "select_valid_switch_target", None)
                    fallback_index = selector(battle_state) if callable(selector) else None
                    if fallback_index is not None:
                        fallback_is_valid = fallback_index not in _active_party_indices(battle_state)
                        if callable(validator):
                            fallback_is_valid = fallback_is_valid and bool(validator(battle_state, fallback_index))
                        if not fallback_is_valid:
                            fallback_index = None
                    if fallback_index is not None:
                        _log_switch_resolution(
                            battle_state=battle_state,
                            battler_index=battler_index,
                            status="replanned",
                            requested_index=requested_switch_index,
                            final_index=fallback_index,
                            reason=switch_rejection_reason,
                        )
                        diagnostic_print(
                            lambda: (
                                "BATTLE_SWITCH_REPLANNED: "
                                f"requested={index!r} replacement={fallback_index!r}"
                            ),
                            trace=True,
                        )
                        index = fallback_index
                    else:
                        # The strategy decision and the menu-boundary facts can
                        # diverge across a callback transition. A switch that
                        # no longer passes validation must never crash the
                        # battle controller or open a party menu with no legal
                        # target. Replan to the active battler's executable
                        # move, or flee from a wild battle; if neither exists,
                        # stop for manual review with the actual reason.
                        fallback_action = _fallback_for_invalid_switch(battle_state)
                        target_pokemon = get_party()[index]
                        target_name = _battle_species_name(target_pokemon, get_party())
                        diagnostic_print(
                            lambda: (
                                "BATTLE_SWITCH_REJECTED: "
                                f"requested={index!r} target={target_name!r} fallback={fallback_action!r} "
                                f"reason={switch_rejection_reason!r}"
                            ),
                            trace=True,
                        )
                        _log_switch_resolution(
                            battle_state=battle_state,
                            battler_index=battler_index,
                            status="rejected",
                            requested_index=requested_switch_index,
                            reason=switch_rejection_reason,
                        )
                        if fallback_action is None:
                            context.battle_decision_source = "BATTLE SWITCH GUARD"
                            context.battle_decision_detail = (
                                f"rejected switch to {target_name}; {switch_rejection_reason}; "
                                "no validated replacement or executable fallback"
                            )
                            context.message = (
                                f"BATTLE SWITCH GUARD: {target_name} was rejected: {switch_rejection_reason}; "
                                "manual review required"
                            )
                            context.set_manual_mode()
                            yield
                        else:
                            fallback, fallback_index = fallback_action
                            context.battle_decision_source = "BATTLE SWITCH GUARD"
                            context.battle_decision_detail = (
                                f"rejected switch to {target_name}; {switch_rejection_reason}; replanned to "
                                f"{getattr(fallback, 'name', fallback)}"
                            )
                            context.message = (
                                f"BATTLE SWITCH GUARD: {target_name} was rejected: {switch_rejection_reason}; "
                                f"using {getattr(fallback, 'name', fallback)}"
                            )
                            if fallback in (
                                TurnAction.UseMove,
                                TurnAction.UseMoveAgainstRightSideOpponent,
                                TurnAction.UseMoveAgainstPartner,
                            ):
                                yield from battle_action_use_move(
                                    fallback, battler_index, fallback_index, battle_state
                                )
                            elif fallback is TurnAction.UseItem:
                                yield from battle_action_use_item(battle_state, fallback_index, None)
                            elif fallback is TurnAction.RunAway:
                                yield from scroll_to_battle_action(3)
                                context.emulator.press_button("A")
                                yield
                            else:
                                context.set_manual_mode()
                                yield
                    continue

                _log_battle_action_lifecycle("validated", battle_state, strategy, action, index, battler_index)
                _log_battle_action_lifecycle("dispatched", battle_state, strategy, action, index, battler_index)

                switch_reason = getattr(context, "battle_decision_detail", None) or "validated by switch executor"
                _register_pending_switch(
                    battle_state=battle_state,
                    battler_index=battler_index,
                    requested_index=requested_switch_index,
                    final_index=index,
                    reason=switch_reason,
                )
                _log_switch_resolution(
                    battle_state=battle_state,
                    battler_index=battler_index,
                    status="dispatched",
                    requested_index=requested_switch_index,
                    final_index=index,
                    reason=switch_reason,
                )

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
                yield from _close_battle_party_menu()

            case TurnAction.RunAway | SafariTurnAction.RunAway:
                if battle_state.is_trainer_battle:
                    raise RuntimeError("Tried to run away from a trainer battle.")
                _log_battle_action_lifecycle("validated", battle_state, strategy, action, index, battler_index)
                _log_battle_action_lifecycle("dispatched", battle_state, strategy, action, index, battler_index)
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
    """Execute one battle-item turn with explicit ROM-state boundaries."""

    def press_battle_button(button: str) -> None:
        # A plain press_button can be suppressed when the same button was
        # present on the immediately preceding emulator frame. Battle item
        # selection is especially sensitive to this because the next step
        # waits for the bag callback; use a genuine fresh edge at each
        # confirmation boundary.
        press_fresh = getattr(type(context.emulator), "press_button_fresh", None)
        if callable(press_fresh):
            context.emulator.press_button_fresh(button)
        else:
            context.emulator.press_button(button)

    target_pokemon = _capture_party_member(target_index) if target_index is not None else None
    yield from scroll_to_battle_action(1)
    press_battle_button("A")
    yield
    for _ in range(_BATTLE_ITEM_TRANSITION_TIMEOUT):
        if get_game_state() is GameState.BAG_MENU:
            break
        if not battle_is_active():
            raise RuntimeError(f"Battle ended before item {item.name!r} opened the bag")
        yield
    else:
        diagnostic_print(
            lambda: (
                "BATTLE_ITEM_TRANSITION_TIMEOUT: "
                f"item={item.name!r} expected='BAG_MENU' "
                f"game_state={get_game_state()!r} callback={get_game_state_symbol()!r}"
            ),
            trace=True,
            prefix="BATTLE_ITEM_TRANSITION_TIMEOUT:",
        )
        raise RuntimeError(f"Battle item {item.name!r} did not open the bag menu")
    yield from scroll_to_item_in_bag(item)
    press_battle_button("A")
    for _ in range(4 if context.rom.is_rse else 5):
        yield
    press_battle_button("A")
    yield
    if target_index is not None:
        for _ in range(_BATTLE_ITEM_TRANSITION_TIMEOUT):
            if get_game_state() is GameState.PARTY_MENU:
                break
            if not battle_is_active():
                raise RuntimeError(f"Battle ended before item {item.name!r} opened the party menu")
            yield
        else:
            diagnostic_print(
                lambda: (
                    "BATTLE_ITEM_TARGET_TIMEOUT: "
                    f"item={item.name!r} expected='PARTY_MENU' "
                    f"game_state={get_game_state()!r} callback={get_game_state_symbol()!r}"
                ),
                trace=True,
                prefix="BATTLE_ITEM_TARGET_TIMEOUT:",
            )
            raise RuntimeError(f"Battle item {item.name!r} did not open the party menu")
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
        press_battle_button("A")
        yield
        for _ in range(_BATTLE_ITEM_RESOLUTION_TIMEOUT):
            if get_game_state_symbol() == "BATTLEMAINCB2":
                return
            if not battle_is_active():
                return
            press_battle_button("B")
            yield
        raise RuntimeError(f"Battle item {item.name!r} did not return to battle action selection")


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
        for _ in range(_BATTLE_MOVE_CALLBACK_TIMEOUT):
            if not (
                get_main_battle_callback() in ("HandleTurnActionSelectionState", "sub_8012324")
                and get_battle_controller_callback(battler_index)
                not in ("HandleInputChooseMove", "HandleAction_ChooseMove")
            ):
                break
            context.emulator.press_button("A")
            yield
        else:
            raise _battle_executor_timeout(
                phase="wait_for_move_list",
                action=action,
                battler_index=battler_index,
                move_index=move_index,
                battle_state=battle_state,
            )
        return

    # Select move and target. The fourth cursor position is absent when the
    # active Pokémon knows fewer than four moves; pass the occupied slots to
    # the menu navigator so it never probes that empty position.
    active_battler = (
        battle_state.own_side.right_battler
        if battler_index == 2
        else battle_state.own_side.active_battler
    )
    move_count = len(getattr(active_battler, "moves", ()))
    yield from scroll_to_move(
        move_index,
        battler_index > 1,
        tuple(range(move_count)) if 0 < move_count < 4 else None,
    )
    for _ in range(_BATTLE_MOVE_CALLBACK_TIMEOUT):
        if get_battle_controller_callback(battler_index) not in (
            "HandleInputChooseMove",
            "HandleAction_ChooseMove",
        ):
            break
        context.emulator.press_button("A")
        yield
    else:
        raise _battle_executor_timeout(
            phase="accept_move",
            action=action,
            battler_index=battler_index,
            move_index=move_index,
            battle_state=battle_state,
        )
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
    for _ in range(_BATTLE_TARGET_SELECTION_TIMEOUT):
        if get_battle_controller_callback(battler_index) != "HandleInputChooseTarget":
            break
        current_target = unpack_uint32(read_symbol("gMultiUsePlayerCursor"))
        if current_target == target_index:
            context.emulator.press_button("A")
            yield
        else:
            context.emulator.press_button("Down")
            yield
    else:
        raise _battle_executor_timeout(
            phase="select_target",
            action=action,
            battler_index=battler_index,
            move_index=move_index,
            battle_state=battle_state,
        )
