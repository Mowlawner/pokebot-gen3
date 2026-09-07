from typing import Generator, TYPE_CHECKING

from modules.battle_action_selection import battle_action_selection_is_ready, handle_battle_action_selection
from modules.battle_evolution_scene import handle_evolution_scene
from modules.battle_move_replacing import handle_move_replacement_dialogue
from modules.battle_state import (
    battle_is_active,
    get_battle_controller_callback,
    get_main_battle_callback,
    get_current_battle_script_instruction,
    BattlePokemon,
    get_battle_state,
    get_encounter_type,
    get_last_battle_outcome,
    HandledBattleResult,
)
from modules.battle_observation import observe_current_battle
from modules.battle_strategies import BattleStrategy
from modules.context import context
from modules.console import diagnostic_print
from modules.debug import debug
from modules.game import decode_string
from modules.items import Item, get_item_by_index
from modules.keyboard import handle_naming_screen
from modules.memory import get_game_state, GameState, read_symbol, unpack_uint16
from modules.menuing import get_current_party_menu_index, scroll_to_party_menu_index
from modules.menu_parsers import switch_requested
from modules.player import get_player
from modules.plugins import plugin_should_nickname_pokemon
from modules.pokemon import StatusCondition
from modules.pokemon_party import get_party, get_party_size, PartyPokemon
from modules.tasks import task_is_active
from modules.nuzlocke.pokemon_naming import generate_pokemon_nickname

if TYPE_CHECKING:
    from modules.encounter import EncounterInfo


_last_handled_battle_result: HandledBattleResult | None = None
_last_battle_prompt_trace_signature = None


def _trace_battle_prompt_state(event: str) -> None:
    """Record the ROM evidence around a possible trainer replacement prompt.

    This is intentionally diagnostic-only.  In particular, it records the
    displayed replacement text separately from the controller callback and
    party-menu task, because those signals do not necessarily change on the
    same frame.
    """

    global _last_battle_prompt_trace_signature
    if not getattr(context, "debug", False) or not getattr(context, "debug_trace", False):
        return
    try:
        battle_state = get_battle_state()
        controller_callbacks = tuple(get_battle_controller_callback(index) for index in range(4))
        displayed_string_raw = read_symbol("gDisplayedStringBattle", size=0x12C)
        displayed_string = decode_string(displayed_string_raw)
        forced_replacement_text = read_symbol("sText_UseNextPkmn")
        trainer_replacement_text = read_symbol("sText_EnemyAboutToSwitchPkmn")
        forced_replacement_text_decoded = decode_string(forced_replacement_text)
        trainer_replacement_text_decoded = decode_string(trainer_replacement_text)
        displayed_replacement_prompt = bool(switch_requested())
        displayed_forced_replacement_text = forced_replacement_text in displayed_string_raw
        displayed_trainer_replacement_text = trainer_replacement_text in displayed_string_raw
        displayed_trainer_replacement_prompt = (
            "is about to use " in displayed_string and "change POK" in displayed_string
        )
        choose_mon_task = task_is_active("Task_HandleChooseMonInput")
        selection_menu_task = task_is_active("Task_HandleSelectionMenuInput")
        active_battler = read_symbol("gActiveBattler", size=1)[0]
        main_callback_pointer = read_symbol("gBattleMainFunc", size=4).hex()
        script_pointer = read_symbol("gBattleScriptCurrInstr", size=4).hex()
        controller_pointers = tuple(
            read_symbol("gBattlerControllerFuncs", offset=4 * index, size=4).hex() for index in range(4)
        )
        battle_buffer_a = read_symbol("gBattleBufferA", size=8).hex()
        battle_communication = read_symbol("gBattleCommunication", size=8).hex()
        signature = (
            get_main_battle_callback(),
            get_current_battle_script_instruction(),
            controller_callbacks,
            get_game_state(),
            displayed_replacement_prompt,
            displayed_string,
            displayed_forced_replacement_text,
            displayed_trainer_replacement_text,
            displayed_trainer_replacement_prompt,
            forced_replacement_text_decoded,
            trainer_replacement_text_decoded,
            choose_mon_task,
            selection_menu_task,
            active_battler,
            main_callback_pointer,
            script_pointer,
            controller_pointers,
            battle_buffer_a,
            battle_communication,
        )
        if signature == _last_battle_prompt_trace_signature:
            return
        _last_battle_prompt_trace_signature = signature
        diagnostic_print(
            lambda: (
                "BATTLE_PROMPT_STATE: "
                f"event={event!r} "
                f"context_frame={getattr(context, 'frame', None)!r} "
                f"emulator_frame={context.emulator.get_frame_count()!r} "
                f"main_callback={signature[0]!r} script={signature[1]!r} "
                f"controller_callbacks={controller_callbacks!r} "
                f"game_state={signature[3]!r} "
                f"displayed_string={displayed_string!r} "
                f"displayed_replacement_prompt={displayed_replacement_prompt!r} "
                f"displayed_forced_replacement_text={displayed_forced_replacement_text!r} "
                f"displayed_trainer_replacement_text={displayed_trainer_replacement_text!r} "
                f"displayed_trainer_replacement_prompt={displayed_trainer_replacement_prompt!r} "
                f"forced_replacement_text_decoded={forced_replacement_text_decoded!r} "
                f"trainer_replacement_text_decoded={trainer_replacement_text_decoded!r} "
                f"choose_mon_task={choose_mon_task!r} "
                f"selection_menu_task={selection_menu_task!r} "
                f"active_battler={active_battler!r} "
                f"main_callback_pointer={main_callback_pointer!r} "
                f"script_pointer={script_pointer!r} "
                f"controller_pointers={controller_pointers!r} "
                f"gBattleBufferA_head={battle_buffer_a!r} "
                f"gBattleCommunication={battle_communication!r} "
                f"is_trainer={getattr(battle_state, 'is_trainer_battle', None)!r} "
                f"is_double={getattr(battle_state, 'is_double_battle', None)!r}"
            ),
            trace=True,
            prefix="BATTLE_PROMPT_STATE:",
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return


@debug.track
def handle_battle(
    strategy: BattleStrategy,
) -> Generator[None, None, HandledBattleResult]:
    """
    This is the main battle-handling function that will attempt to finish the
    battle, calling the battle strategy's callbacks whenever a decision is
    needed.
    :param strategy: The battle strategy that should be queried each time there
                     is a decision to make.
    :return: A data class structure listing some things that have changed
             during the battle.
    """
    before_cash = get_player().money
    before_party = get_party()
    encounter_type = get_encounter_type()

    stolen_items: list[tuple[int, Item]] = []
    items_before_pickup: list[Item | None] | None = None

    while battle_is_active() and context.bot_mode != "Manual":
        # Keep the planner's knowledge current immediately before any battle
        # strategy/action-selection callback.
        observe_current_battle()
        instruction = get_current_battle_script_instruction()
        _trace_battle_prompt_state("handler_loop")
        if battle_action_selection_is_ready():
            yield from handle_battle_action_selection(strategy)
        elif get_current_battle_script_instruction() == "BattleScript_ItemSteal":
            result = yield from handle_item_stealing()
            if result is not None:
                stolen_items.append(result)
        elif (
            get_current_battle_script_instruction() == "BattleScript_PayDayMoneyAndPickUpItems"
            and items_before_pickup is None
        ):
            items_before_pickup = [pokemon.held_item for pokemon in get_party()]
            yield
        elif instruction == "BattleScript_AskToLearnMove":
            yield from handle_move_replacement_dialogue(strategy)
        elif task_is_active("Task_EvolutionScene"):
            yield from handle_evolution_scene(strategy)
        elif (
            instruction == "BattleScript_HandleFaintedMon" or task_is_active("Task_HandleChooseMonInput")
        ) and get_battle_state().own_side.is_fainted:
            yield from handle_fainted_pokemon(strategy)
        elif instruction in (
            "BattleScript_TryNicknameCaughtMon",
            "BattleScript_CaughtPokemonSkipNewDex",
        ):
            yield from handle_nickname_caught_pokemon(context.stats.last_encounter)
        else:
            diagnostic_print(
                lambda: (
                    "BATTLE_HANDLER_FALLBACK_INPUT: "
                    f"context_frame={getattr(context, 'frame', None)!r} "
                    f"emulator_frame={context.emulator.get_frame_count()!r} "
                    f"strategy={type(strategy).__name__!r} "
                    f"main_callback={get_main_battle_callback()!r} "
                    f"script={get_current_battle_script_instruction()!r} "
                    f"controller_0={get_battle_controller_callback(0)!r} "
                    f"game_state={get_game_state()!r} "
                    f"is_trainer={getattr(get_battle_state(), 'is_trainer_battle', None)!r} "
                    f"is_double={getattr(get_battle_state(), 'is_double_battle', None)!r} "
                    "reason='unhandled_battle_handler_branch'"
                ),
                trace=True,
                prefix="BATTLE_HANDLER_FALLBACK_INPUT:",
            )
            context.emulator.press_button("B")
            yield

    # Capture the terminal/transition frame so BattleEnded can be derived
    # without making the handler's result or control flow depend on it.
    observe_current_battle()

    outcome = get_last_battle_outcome()
    party_indices_with_picked_up_items = []
    party_indices_that_took_damage_or_changed_status = []
    party_indices_that_gained_exp = []
    party_indices_that_evolved = []

    after_party = get_party()
    for index in range(len(after_party)):
        if len(before_party) <= index:
            break

        before: PartyPokemon = before_party[index]
        after: PartyPokemon = after_party[index]

        if before.current_hp > after.current_hp or (
            before.status_condition is StatusCondition.Healthy and after.status_condition is not StatusCondition.Healthy
        ):
            party_indices_that_took_damage_or_changed_status.append(index)

        if before.total_exp < after.total_exp:
            party_indices_that_gained_exp.append(index)

            if before.species.name != after.species.name:
                party_indices_that_evolved.append(index)

        if items_before_pickup is not None and index < len(items_before_pickup):
            if after.held_item is not items_before_pickup[index] and after.held_item is not None:
                party_indices_with_picked_up_items.append(index)

    party_indices_with_stolen_items = set()
    for party_index, item in stolen_items:
        after: PartyPokemon = after_party[party_index]
        if after.held_item is item:
            party_indices_with_stolen_items.add(party_index)

    result = HandledBattleResult(
        outcome,
        encounter_type,
        get_player().money - before_cash,
        list(party_indices_with_stolen_items),
        party_indices_with_picked_up_items,
        party_indices_that_took_damage_or_changed_status,
        party_indices_that_gained_exp,
        party_indices_that_evolved,
    )

    global _last_handled_battle_result
    _last_handled_battle_result = result

    return result


@debug.track
def handle_item_stealing() -> Generator[None, None, tuple[int, Item] | None]:
    item_index = unpack_uint16(read_symbol("gLastUsedItem"))
    if read_symbol("gBattlerAttacker")[0] == 0:
        stolen_by = get_battle_state().own_side.left_battler.party_index
    elif read_symbol("gBattlerAttacker")[0] == 2:
        stolen_by = get_battle_state().own_side.right_battler.party_index
    else:
        stolen_by = None

    while get_current_battle_script_instruction() == "BattleScript_ItemSteal":
        context.emulator.press_button("B")
        yield

    if stolen_by is not None:
        return stolen_by, get_item_by_index(item_index)


@debug.track
def handle_fainted_pokemon(strategy: BattleStrategy):
    fainted_pokemon: BattlePokemon | None = None
    battle_state = get_battle_state()
    if battle_state.own_side.left_battler is None or battle_state.own_side.left_battler.current_hp == 0:
        fainted_pokemon = battle_state.own_side.left_battler
    elif battle_state.is_double_battle and (
        battle_state.own_side.right_battler is None or battle_state.own_side.right_battler.current_hp == 0
    ):
        fainted_pokemon = battle_state.own_side.right_battler

    if fainted_pokemon is None:
        # I can't remember how this could ever happen, but I'll leave this check in here for now.
        context.message = "fainted_pokemon was None"
        context.debug_stepping_mode()
        # context.emulator.press_button("B")
        # for _ in range(180):
        #     yield
        return

    # Emerald still owns the battle until it has shown the whiteout flow when
    # the last party member faints.  There is no replacement to select in
    # this case, so sending the normal party-selection input would leave the
    # handler pressing B forever (the choose-mon task ignores it).  Advance
    # the ROM's native transition and let WhiteoutListener take ownership once
    # GameState.WHITEOUT is exposed.
    has_replacement = getattr(strategy, "has_replacement_after_faint", None)
    if callable(has_replacement):
        replacement_available = bool(has_replacement(battle_state))
    else:
        replacement_available = bool(get_party().non_fainted_pokemon)
    diagnostic_print(
        lambda: (
            "BATTLE_FAINT_REPLACEMENT: "
            f"strategy={type(strategy).__name__!r} available={replacement_available!r} "
            f"party={[(pokemon.name, pokemon.current_hp, pokemon.is_egg) for pokemon in get_party()]!r} "
            f"active={[battler.party_index for battler in battle_state.own_side.active_battlers]!r}"
        ),
        trace=True,
    )
    if not replacement_available:
        yield from _advance_after_party_wipe()
        return

    if not battle_state.is_trainer_battle:
        if strategy.should_flee_after_faint(battle_state):
            if context.bot_mode != "Manual":
                while (
                    battle_is_active()
                    and get_current_battle_script_instruction() != "BattleScript_FaintedMonEnd"
                    and get_game_state() != GameState.PARTY_MENU
                    and get_main_battle_callback() != "HandleEndTurn_FinishBattle"
                ):
                    context.emulator.press_button("B")
                    yield

                if battle_is_active() and get_main_battle_callback() == "HandleEndTurn_FinishBattle":
                    while get_main_battle_callback() == "HandleEndTurn_FinishBattle":
                        context.emulator.press_button("B")
                        yield
                    return

    new_lead_index = strategy.choose_new_lead_after_faint(battle_state)
    regular_new_lead_index = new_lead_index

    # If `choose_new_lead_after_faint()` has been called while NOT being in the party selection screen,
    # `get_party()` still contains the 'original' (overworld) party order. Thus, we have to map the new
    # index to the in-battle party index (because that's what we're going to select later) but only after
    # all the sanity checks have been done.
    # On the other hand, if this function was called IN the party menu, `get_party()` already returns the
    # in-battle order and no mapping is needed.
    #
    # In practice, this function will be called OUTSIDE the party menu if the battle strategy chose to
    # send out the next Pokémon (without trying to escape) because then the call happens during the
    # dialogue.
    # Whereas the function will be called IN the party menu if the strategy tried to escape and failed,
    # because then the game automatically opens the party menu.
    index_needs_mapping = get_game_state() != GameState.PARTY_MENU

    if context.bot_mode == "Manual":
        yield
        return

    if new_lead_index < 0 or new_lead_index >= get_party_size():
        raise RuntimeError(f"Cannot send out party index #{new_lead_index} because that does not exist.")

    new_lead = get_party()[new_lead_index]
    if new_lead.is_egg:
        raise RuntimeError(f"Cannot send out party index #{new_lead_index} because it is an egg.")
    if new_lead.current_hp <= 0:
        raise RuntimeError(f"Cannot send out {new_lead.name} (#{new_lead_index}) because it has 0 HP.")

    while (
        battle_is_active()
        and get_current_battle_script_instruction() != "BattleScript_FaintedMonEnd"
        and not task_is_active("Task_HandleChooseMonInput")
        and not task_is_active("HandleBattlePartyMenu")
    ):
        context.emulator.press_button("A")
        yield

    # This will trigger if a battle ends at the same time as the player's Pokémon faints.
    # That happens if the player uses moves like Explosion or Self Destruct, or receives
    # fatal recoil damage during the finishing blow.
    # In this case, the game obviously never asks to choose a new lead and ends the battle
    # instead.
    if get_current_battle_script_instruction() == "BattleScript_FaintedMonEnd":
        return

    if index_needs_mapping:
        new_lead_index = battle_state.map_battle_party_index(new_lead_index)

    # A save-state can preserve a battle's party-order buffer from an earlier
    # menu boundary.  Emerald has already reordered the visible party by the
    # time the menu task is active, so use the selected Pokémon's identity as
    # the final source of truth for the visible menu slot when available.
    try:
        visible_party_index = get_party().get_index_for_pokemon(new_lead)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        visible_party_index = None
    if visible_party_index is not None and visible_party_index != new_lead_index:
        diagnostic_print(
            lambda: (
                "BATTLE_FAINT_SELECTION_RECONCILED: "
                f"mapped_index={new_lead_index} visible_index={visible_party_index}"
            ),
            trace=True,
        )
        new_lead_index = visible_party_index

    diagnostic_print(
        lambda: (
            "BATTLE_FAINT_SELECTION: "
            f"regular_index={regular_new_lead_index} menu_index={new_lead_index} "
            f"battle_order={getattr(battle_state, '_battler_party_order', None)!r} "
            f"party={[(pokemon.name, pokemon.current_hp, pokemon.is_egg) for pokemon in get_party()]!r} "
            f"cursor={get_current_party_menu_index()!r}"
        ),
        trace=True,
    )

    yield from scroll_to_party_menu_index(new_lead_index)
    while get_game_state() == GameState.PARTY_MENU:
        context.emulator.press_button("A")
        yield


@debug.track
def _advance_after_party_wipe() -> Generator[None, None, None]:
    """Advance a trainer battle after the player has no replacement Pokémon.

    ``Task_HandleChooseMonInput`` is also used for the final-party branch, but
    that task does not accept the ordinary cancel input.  A positive input is
    what lets the Emerald battle engine continue into its whiteout script;
    outside that task, battle text/teardown is advanced with B as usual.
    """

    while battle_is_active():
        if get_game_state() == GameState.WHITEOUT:
            return
        if task_is_active("Task_HandleChooseMonInput") or task_is_active("Task_HandleSelectionMenuInput"):
            context.emulator.press_button("A")
        else:
            context.emulator.press_button("B")
        yield


@debug.track
def handle_nickname_caught_pokemon(encounter: "EncounterInfo"):
    nickname_choice = plugin_should_nickname_pokemon(encounter)
    if nickname_choice is None and context.nuzlocke_runtime is not None:
        try:
            nickname_choice = generate_pokemon_nickname(
                encounter.pokemon.species.name,
                encounter.pokemon.gender,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            nickname_choice = None
    if nickname_choice:
        yield from handle_naming_screen(nickname_choice)

    # Skip the nicknaming dialogue.
    while get_current_battle_script_instruction() in (
        "BattleScript_TryNicknameCaughtMon",
        "BattleScript_CaughtPokemonSkipNewDex",
    ):
        context.emulator.press_button("B")
        yield


def get_last_handled_battle_result() -> HandledBattleResult | None:
    return _last_handled_battle_result
