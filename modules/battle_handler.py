from typing import Generator, TYPE_CHECKING

from modules.battle_action_selection import battle_action_selection_is_ready, handle_battle_action_selection
from modules.battle_evolution_scene import handle_evolution_scene
from modules.battle_move_replacing import handle_move_replacement_dialogue
from modules.battle_state import (
    battle_is_active,
    get_battle_controller_callback,
    get_main_battle_callback,
    get_current_battle_script_instruction,
    BattleType,
    BattlePokemon,
    get_battle_state,
    get_encounter_type,
    get_last_battle_outcome,
    HandledBattleResult,
)
from modules.battle_observation import (
    BattleEnded,
    DamageOccurred,
    HealingOccurred,
    PokemonFainted,
    PokemonSwitched,
    StatusChanged,
    begin_battle_observation,
    get_last_completed_battle_actions,
    get_last_completed_battle_knowledge,
    observe_current_battle,
)
from modules.battle_identity import battle_pokemon_display_name, battle_pokemon_identity
from modules.battle_strategies import BattleStrategy
from modules.context import context
from modules.console import (
    diagnostic_print,
    print_battle_end,
    print_battle_event,
    print_battle_start,
    print_forced_replacement,
    print_battle_switch_resolution,
)
from modules.debug import debug
from modules.game import decode_string
from modules.items import Item, get_item_by_index
from modules.keyboard import handle_naming_screen
from modules.memory import get_game_state, GameState, read_symbol, unpack_uint16
from modules.menuing import get_current_party_menu_index, scroll_to_party_menu_index
from modules.menu_parsers import switch_requested
from modules.modes._interface import BotModeError
from modules.player import get_player
from modules.plugins import plugin_should_nickname_pokemon
from modules.pokemon import StatusCondition
from modules.pokemon_party import get_opponent_party, get_party, get_party_size, PartyPokemon
from modules.tasks import task_is_active
from modules.trainer_checkpoints import trainer_checkpoint_for_battle
from modules.nuzlocke.pokemon_naming import generate_pokemon_nickname
from modules.nuzlocke.identity import PokemonIdentity

if TYPE_CHECKING:
    from modules.encounter import EncounterInfo


_last_handled_battle_result: HandledBattleResult | None = None
_last_battle_prompt_trace_signature = None

# These callbacks belong to the ROM while it is setting up the first turn.
# They can be observed after the battle listener has handed the battle to the
# campaign strategy, but before the player's action controller is ready.
_ROM_OWNED_BATTLE_SETUP_CALLBACKS = frozenset(
    {
        "BeginBattleIntro",  # Battle intro animation/setup
        "BattleTurnPassed",  # Advance to the next turn
        "CheckFocusPunch_ClearVarsBeforeTurnStarts",  # Turn-resolution setup
        "FreeResetData_ReturnToOvOrDoEvolutions",  # Battle cleanup/evolution gate
        "RunTurnActionsFunctions",  # Dispatch the selected move/switch/item
        "SetActionsAndBattlersTurnOrder",  # Resolve the selected turn actions
        "TurnValuesCleanUp",  # Finish turn bookkeeping
        "ReturnFromBattleToOverworld",  # Final battle ownership transition
        "TryDoEventsBeforeFirstTurn",  # Emerald
        "TryEvolvePokemon",  # Optional evolution scene setup
        "WaitForEvoSceneToFinish",  # Evolution scene completion
        "BattleBeginFirstTurn",  # Ruby/Sapphire/FireRed/LeafGreen
    }
)
# BattleIntroDrawPartySummaryScreens can legitimately spend several seconds
# waiting for the intro animation/controller boundary to complete.
_BATTLE_SETUP_TRANSITION_TIMEOUT = 600
# Wally's tutorial includes a ROM-owned catch sequence, battle animations, and
# dialogue that can legitimately outlast ordinary first-turn setup. Keep a
# finite watchdog, but do not convert that known scripted battle to Manual at
# the ordinary 600-frame boundary.
_WALLY_TUTORIAL_SETUP_TRANSITION_TIMEOUT = 2400


def _is_wally_tutorial_battle() -> bool:
    """Return whether the active battle is Emerald's scripted Wally battle."""

    try:
        return BattleType.WallyTutorial in get_battle_state().type
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False


def _battle_setup_transition_timeout() -> int:
    """Return the watchdog budget for the current ROM-owned battle setup."""

    return (
        _WALLY_TUTORIAL_SETUP_TRANSITION_TIMEOUT
        if _is_wally_tutorial_battle()
        else _BATTLE_SETUP_TRANSITION_TIMEOUT
    )


def _battle_setup_transition_is_active() -> bool:
    """Return whether the ROM is still preparing the battle's first turn."""

    callback = get_main_battle_callback()
    return callback in _ROM_OWNED_BATTLE_SETUP_CALLBACKS or callback.startswith(
        (
            "BattleIntro",
            "BattleScript",
            "HandleAction_",
            "HandleEndTurn_",
            "RunBattleScript",
        )
    )


def _battle_roster_names(party) -> tuple[str, ...]:
    try:
        return tuple(battle_pokemon_display_name(pokemon, party=party) for pokemon in party or ())
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return ()


def _battle_event_pokemon_name(event, side: str) -> str:
    """Resolve an observed event to species plus nickname when possible."""

    party_index = getattr(event, "party_index", None)
    try:
        party = get_party() if side == "player" else get_opponent_party()
        if isinstance(party_index, int):
            return battle_pokemon_display_name(party[party_index], party=party)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        pass
    return str(getattr(event, "species", None) or f"{side} party slot {party_index}")


def _battle_trainer_name(trainer_id: int | None) -> str | None:
    """Return a safe fallback when the ROM trainer catalog is unavailable.

    An opponent Pokémon's original-trainer field is not a battle-trainer
    identity. During battle setup it can temporarily contain the player's
    name, which made the battle panel attribute the player's name to the
    opponent. The catalog/checkpoint path supplies the real name whenever it
    is known; an ID is the only honest fallback.
    """

    return f"Trainer #{trainer_id}" if trainer_id is not None else None


def _battle_type_label(state) -> str:
    """Return a useful battle label even when wild battles have no flags."""

    names = tuple(getattr(state, "type_names", ()) or ())
    if names:
        return "|".join(names)
    if getattr(state, "is_trainer_battle", False):
        return "Trainer"
    return "Wild"


def _battle_trainer_id() -> int | None:
    """Read the ROM trainer identifier when the current ROM exposes it."""

    try:
        return unpack_uint16(read_symbol("gTrainerBattleOpponent_A", size=2))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _print_battle_start() -> None:
    try:
        state = get_battle_state()
        battle_type = _battle_type_label(state)
        encounter = get_encounter_type()
        opponent_party = get_opponent_party()
        trainer_checkpoint = (
            trainer_checkpoint_for_battle(
                state,
                getattr(getattr(state, "opponent", None), "active_battler", None),
            )
            if getattr(state, "is_trainer_battle", False)
            else None
        )
        # The battle envelope and the campaign readiness snapshot must share
        # the same trainer facts.  Battle setup is the last authoritative
        # fallback for a trainer whose overworld object was not visible during
        # route planning; retain the resolved checkpoint so the next campaign
        # boundary cannot report the trainer as unknown.
        if trainer_checkpoint is not None:
            context.campaign_trainer_checkpoint = trainer_checkpoint
            context.campaign_active_trainer_checkpoint = trainer_checkpoint
            existing_facts = {
                getattr(fact, "checkpoint_id", None): fact
                for fact in tuple(getattr(context, "campaign_trainer_facts", ()) or ())
                if getattr(fact, "checkpoint_id", None) is not None
            }
            checkpoint_id = getattr(trainer_checkpoint, "checkpoint_id", None)
            if checkpoint_id is not None:
                existing_facts[checkpoint_id] = trainer_checkpoint
                context.campaign_trainer_facts = tuple(
                    sorted(
                        existing_facts.values(),
                        key=lambda fact: (
                            getattr(fact, "map_id", ()),
                            getattr(fact, "checkpoint_id", ""),
                        ),
                    )
                )
        elif getattr(state, "is_trainer_battle", False):
            # Do not let a checkpoint from a previous battle identify this
            # battle. An unresolved trainer remains unknown to readiness.
            context.campaign_active_trainer_checkpoint = None
        else:
            context.campaign_active_trainer_checkpoint = None
        trainer_id = _battle_trainer_id() if getattr(state, "is_trainer_battle", False) else None
        # Pokémon OT data is not a reliable trainer identity in the live
        # battle projection: the opponent party can inherit the player's
        # trainer object while the battle is loading. The ROM-backed
        # checkpoint is keyed by gTrainerBattleOpponent_A and is the same
        # catalog used for the move envelope, so prefer its display name.
        trainer_name = None
        if getattr(state, "is_trainer_battle", False):
            trainer_name = getattr(trainer_checkpoint, "display_name", None) or _battle_trainer_name(trainer_id)
        opponent_envelope = (
            tuple(
                f"{member.species} Lv{member.level}: {', '.join(member.moves) or 'unknown'}"
                for member in trainer_checkpoint.party
            )
            if trainer_checkpoint is not None
            else None
        )
        context.battle_authoritative_faints = []
        context.battle_readiness = None
        context.battle_evaluation_method = None
        print_battle_start(
            battle_type=battle_type,
            encounter_type=getattr(encounter, "value", encounter),
            trainer_name=trainer_name,
            trainer_id=trainer_id,
            opponent_roster=_battle_roster_names(opponent_party),
            player_roster=_battle_roster_names(get_party()),
            opponent_envelope=opponent_envelope,
            envelope_source=getattr(trainer_checkpoint, "source", None),
        )
        diagnostic_print(
            lambda: (
                "BATTLE_START: "
                f"battle_type={battle_type!r} encounter_type={getattr(encounter, 'value', encounter)!r} "
                f"trainer_name={trainer_name!r} trainer_id={trainer_id!r} "
                f"opponent_roster={_battle_roster_names(opponent_party)!r} "
                f"envelope_source={getattr(trainer_checkpoint, 'source', None)!r}"
            ),
            trace=True,
            prefix="BATTLE_START:",
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return


def _print_pending_switch_observation(event: PokemonSwitched, observed_name: str) -> None:
    pending = getattr(context, "battle_pending_switches", None)
    if not isinstance(pending, list):
        return
    pending_position = next(
        (
            position
            for position, request in enumerate(pending)
            if request.get("before_index") == event.before_party_index
            and request.get("final_index") == event.after_party_index
        ),
        None,
    )
    if pending_position is None:
        pending_position = next(
            (
                position
                for position, request in enumerate(pending)
                if request.get("before_index") == event.before_party_index
            ),
            None,
        )
    if pending_position is None:
        pending_position = next(
            (
                position
                for position, request in enumerate(pending)
                if request.get("final_index") == event.after_party_index
            ),
            None,
        )
    if pending_position is None and len(pending) == 1:
        pending_position = 0
    if pending_position is None:
        return

    request = pending.pop(pending_position)
    expected_index = request.get("final_index")
    matches = expected_index == event.after_party_index
    status = "observed" if matches else "mismatch"
    reason = (
        "observed dispatched switch target"
        if matches
        else f"observed party slot {event.after_party_index} instead of dispatched slot {expected_index}"
    )
    print_battle_switch_resolution(
        status=status,
        requested=request.get("requested_name"),
        requested_index=request.get("requested_index"),
        final=request.get("final_name"),
        final_index=expected_index,
        observed=observed_name,
        observed_index=event.after_party_index,
        active=request.get("active"),
        opponent=request.get("opponent"),
        reason=reason,
    )
    context.battle_switch_reconciliation_status = status
    if status == "mismatch":
        context.battle_switch_replan_required = True
        diagnostic_print(
            lambda: (
                "BATTLE_SWITCH_RECONCILIATION: "
                f"status='mismatch' transaction={request.get('transaction_id')!r} "
                f"requested={request.get('requested_index')!r} expected={expected_index!r} "
                f"observed={event.after_party_index!r} action='fresh_observation_and_replan'"
            ),
            trace=True,
            prefix="BATTLE_SWITCH_RECONCILIATION",
        )
    else:
        diagnostic_print(
            lambda: (
                "BATTLE_SWITCH_RECONCILIATION: "
                f"status='observed' transaction={request.get('transaction_id')!r} "
                f"target={event.after_party_index!r}"
            ),
            trace=True,
            prefix="BATTLE_SWITCH_RECONCILIATION",
        )


def _flush_pending_switch_resolutions() -> None:
    pending = getattr(context, "battle_pending_switches", None)
    if not isinstance(pending, list):
        return
    for request in tuple(pending):
        print_battle_switch_resolution(
            status="not observed",
            requested=request.get("requested_name"),
            requested_index=request.get("requested_index"),
            final=request.get("final_name"),
            final_index=request.get("final_index"),
            observed="none before battle ended",
            active=request.get("active"),
            opponent=request.get("opponent"),
            reason="battle ended before a matching PokemonSwitched event was observed",
        )
    if pending:
        context.battle_switch_reconciliation_status = "unresolved_at_battle_end"
    pending.clear()


def _print_battle_events(knowledge, already_logged: int) -> int:
    """Render only significant derived transitions from the observation stream."""

    if knowledge is None:
        return already_logged
    events = getattr(knowledge, "events", ())
    for event in events[already_logged:]:
        if isinstance(event, (DamageOccurred, HealingOccurred)):
            # Damage is summarized at battle end and fainting gets an explicit
            # record at the authoritative replacement handler below for the
            # player's side.
            continue
        if isinstance(event, PokemonFainted):
            if event.side == "opponent":
                observed_name = _battle_event_pokemon_name(event, event.side)
                print_battle_event(
                    event="pokemon fainted",
                    side=event.side,
                    pokemon=observed_name,
                    detail=f"party slot {event.party_index} reached 0 HP",
                )
                diagnostic_print(
                    lambda: (
                        "BATTLE_ACTION_LIFECYCLE: phase='observed' event='faint' "
                        f"side={event.side!r} pokemon={observed_name!r} party_index={event.party_index!r}"
                    ),
                    trace=True,
                    prefix="BATTLE_ACTION_LIFECYCLE:",
                )
            continue
        if isinstance(event, PokemonSwitched):
            party = get_party() if event.side == "player" else get_opponent_party()
            before_name = _battle_event_pokemon_name(
                type("ObservedPokemon", (), {"party_index": event.before_party_index, "species": None})(),
                event.side,
            )
            after_name = _battle_event_pokemon_name(
                type("ObservedPokemon", (), {"party_index": event.after_party_index, "species": None})(),
                event.side,
            )
            print_battle_event(
                event="pokemon switched",
                side=event.side,
                detail=(
                    f"{before_name} [party slot {event.before_party_index}] → "
                    f"{after_name} [party slot {event.after_party_index}]"
                ),
            )
            if event.side == "player":
                _print_pending_switch_observation(event, after_name)
            diagnostic_print(
                lambda: (
                    "BATTLE_ACTION_LIFECYCLE: phase='observed' event='switch' "
                    f"side={event.side!r} before={before_name!r} after={after_name!r}"
                ),
                trace=True,
                prefix="BATTLE_ACTION_LIFECYCLE:",
            )
        elif isinstance(event, StatusChanged):
            print_battle_event(
                event="status changed",
                side=event.side,
                detail=f"party slot {event.party_index}: {event.before!r} → {event.after!r}",
            )
        elif isinstance(event, BattleEnded):
            print_battle_event(event="battle ended", detail=f"outcome={event.outcome!r}")
    return len(events)


_BATTLE_END_UNSET = object()


def _print_battle_end(
    outcome,
    *,
    experience_gained=(),
    knowledge=_BATTLE_END_UNSET,
    actions=_BATTLE_END_UNSET,
) -> None:
    try:
        if knowledge is _BATTLE_END_UNSET:
            knowledge = get_last_completed_battle_knowledge()
        if actions is _BATTLE_END_UNSET:
            actions = get_last_completed_battle_actions()
        damage_by_party: dict[int | None, int] = {}
        faints = []
        switches = []
        total_damage = 0
        if knowledge is not None:
            for event in knowledge.damage_history:
                if event.side == "player":
                    total_damage += event.amount
                    damage_by_party[event.party_index] = damage_by_party.get(event.party_index, 0) + event.amount
            faints = [
                _battle_event_pokemon_name(event, event.side)
                for event in knowledge.faint_history
                if event.side == "player"
            ]
            for record in getattr(context, "battle_authoritative_faints", ()) or ():
                species = record[1] if isinstance(record, tuple) and len(record) > 1 else str(record)
                if species and species not in faints:
                    faints.append(species)
            switches = [
                f"{event.side} {event.before_party_index}→{event.after_party_index}"
                for event in knowledge.switch_history
            ]
            if knowledge.observations:
                first_observation = knowledge.observations[0]
                battle_type = "|".join(getattr(first_observation, "battle_type", ()) or ())
                if not battle_type:
                    battle_type = "Trainer" if getattr(first_observation, "is_trainer", False) else "Wild"
            else:
                battle_type = "unknown"
            opponent_roster = tuple(
                pokemon.species
                for pokemon in getattr(knowledge, "opponent_pokemon", ())
                if getattr(pokemon, "species", None)
            )
            try:
                live_opponent_roster = _battle_roster_names(get_opponent_party())
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                live_opponent_roster = ()
            if live_opponent_roster:
                opponent_roster = live_opponent_roster
        else:
            state = get_battle_state()
            battle_type = _battle_type_label(state)
            opponent_roster = _battle_roster_names(get_opponent_party())
        move_names = tuple(name for kind, name in actions if kind == "move")
        item_names = tuple(name for kind, name in actions if kind == "item")
        selected_switch_names = tuple(name for kind, name in actions if kind in {"switch", "free_switch"})
        switch_names = tuple(switches) if switches else selected_switch_names
        names_by_party = {}
        try:
            names_by_party.update(
                {
                    pokemon.index: battle_pokemon_display_name(pokemon, party=get_party())
                    for pokemon in get_party()
                }
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            pass
        if knowledge is not None:
            for observation in knowledge.observations:
                for pokemon in observation.player_party:
                    names_by_party.setdefault(pokemon.party_index, (
                    f'{pokemon.species} "{pokemon.nickname}"'
                    if getattr(pokemon, "nickname", None)
                    and str(pokemon.nickname).casefold() != str(pokemon.species).casefold()
                    else pokemon.species or f"slot {pokemon.party_index}"
                    ))
        print_battle_end(
            outcome=getattr(outcome, "name", outcome),
            battle_type=battle_type,
            opponent_roster=opponent_roster,
            total_damage_taken=total_damage,
            damage_by_pokemon=((names_by_party.get(index, f"slot {index}"), amount) for index, amount in damage_by_party.items()),
            items_used=item_names,
            moves_used=move_names,
            experience_gained=experience_gained,
            faints=faints,
            switches=switch_names,
        )
        diagnostic_print(
            lambda: (
                "BATTLE_FINISHED: "
                f"outcome={getattr(outcome, 'name', outcome)!r} battle_type={battle_type!r} "
                f"total_damage_taken={total_damage!r} items_used={item_names!r} moves_used={move_names!r} "
                f"faints={faints!r} switches={switch_names!r}"
            ),
            trace=True,
            prefix="BATTLE_FINISHED:",
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return


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
    begin_battle_observation()
    before_cash = get_player().money
    before_party = get_party()
    encounter_type = get_encounter_type()

    stolen_items: list[tuple[int, Item]] = []
    items_before_pickup: list[Item | None] | None = None
    setup_transition_state = None
    setup_transition_frames = 0
    battle_start_printed = False
    battle_event_count = 0
    context.battle_pending_switches = []
    context.battle_switch_transaction_sequence = 0
    context.battle_switch_reconciliation_status = "idle"
    context.battle_switch_replan_required = False
    context.battle_empty_root_replan_used = False
    context.battle_planner_retry_requested = False
    context.battle_planner_retry_count = 0

    while battle_is_active() and context.bot_mode != "Manual":
        # Keep the planner's knowledge current immediately before any battle
        # strategy/action-selection callback.
        knowledge = observe_current_battle()
        if not battle_start_printed:
            _print_battle_start()
            battle_start_printed = True
        battle_event_count = _print_battle_events(knowledge, battle_event_count)
        instruction = get_current_battle_script_instruction()
        _trace_battle_prompt_state("handler_loop")
        if battle_action_selection_is_ready():
            setup_transition_state = None
            setup_transition_frames = 0
            yield from handle_battle_action_selection(strategy)
        elif get_current_battle_script_instruction() == "BattleScript_ItemSteal":
            setup_transition_state = None
            setup_transition_frames = 0
            result = yield from handle_item_stealing()
            if result is not None:
                stolen_items.append(result)
        elif (
            get_current_battle_script_instruction() == "BattleScript_PayDayMoneyAndPickUpItems"
            and items_before_pickup is None
        ):
            setup_transition_state = None
            setup_transition_frames = 0
            items_before_pickup = [pokemon.held_item for pokemon in get_party()]
            yield
        elif instruction == "BattleScript_AskToLearnMove":
            setup_transition_state = None
            setup_transition_frames = 0
            yield from handle_move_replacement_dialogue(strategy)
        elif instruction in (
            "BattleScript_LearnedNewMove",
            "BattleScript_ForgotAndLearnedNewMove",
        ):
            # RunBattleScriptCommands_PopCallbacksStack is also the ROM
            # callback used for ordinary move resolution.  A move-learning
            # message is a separate, input-owning boundary: it is displayed
            # after the action/experience flow and needs the same B advance
            # used by the other battle message handlers.  Handle it before
            # the broad ROM-transition classifier below, otherwise the
            # classifier repeatedly sends the starter-intro A pulse and the
            # battle remains on "learned MOVE" indefinitely.
            setup_transition_state = None
            setup_transition_frames = 0
            context.emulator.press_button("B")
            yield
        elif task_is_active("Task_EvolutionScene"):
            setup_transition_state = None
            setup_transition_frames = 0
            yield from handle_evolution_scene(strategy)
        elif (
            instruction == "BattleScript_HandleFaintedMon" or task_is_active("Task_HandleChooseMonInput")
        ) and get_battle_state().own_side.is_fainted:
            setup_transition_state = None
            setup_transition_frames = 0
            yield from handle_fainted_pokemon(strategy)
        elif instruction in (
            "BattleScript_TryNicknameCaughtMon",
            "BattleScript_CaughtPokemonSkipNewDex",
        ):
            setup_transition_state = None
            setup_transition_frames = 0
            yield from handle_nickname_caught_pokemon(context.stats.last_encounter)
        elif _battle_setup_transition_is_active():
            current_callback = get_main_battle_callback()
            current_state = (current_callback, instruction)
            if current_state != setup_transition_state:
                setup_transition_state = current_state
                setup_transition_frames = 0
                # A single main callback owns several scripts during a turn.
                # Each script is its own ROM transition state, so a long
                # animation or message must not inherit the previous script's
                # watchdog budget.
            setup_transition_frames += 1
            if setup_transition_frames == 1 or setup_transition_frames % 30 == 0:
                diagnostic_print(
                    lambda: (
                        "BATTLE_HANDLER_TRANSITION_WAIT: "
                        f"strategy={type(strategy).__name__!r} "
                        f"main_callback={current_callback!r} "
                        f"script={instruction!r} "
                        f"frames={setup_transition_frames!r} "
                        "reason='rom_owned_battle_setup'"
                    ),
                    trace=True,
                    prefix="BATTLE_HANDLER_TRANSITION_WAIT:",
                )
            transition_timeout = _battle_setup_transition_timeout()
            if setup_transition_frames >= transition_timeout:
                diagnostic_print(
                    lambda: (
                        "BATTLE_HANDLER_TRANSITION_TIMEOUT: "
                        f"strategy={type(strategy).__name__!r} "
                        f"main_callback={current_callback!r} "
                        f"script={instruction!r} "
                        f"frames={setup_transition_frames!r} "
                        f"timeout={transition_timeout!r} "
                        "reason='rom_owned_battle_setup_timeout'"
                    ),
                    trace=True,
                    prefix="BATTLE_HANDLER_TRANSITION_TIMEOUT:",
                )
                raise BotModeError(
                    "Campaign Progression remained in a ROM-owned battle setup callback "
                    f"({current_callback}) for {setup_transition_frames} frames "
                    f"(timeout={transition_timeout})."
                )
            # The intro callbacks own the battle, but the ROM still expects
            # the normal A advancement pulse used by the starter controller.
            # Do not use B: that can answer a later battle prompt out of order.
            context.emulator.press_button("A")
            yield
        else:
            setup_transition_state = None
            setup_transition_frames = 0
            allows_legacy_fallback = getattr(strategy, "allows_legacy_fallback", None)
            if callable(allows_legacy_fallback) and not allows_legacy_fallback():
                diagnostic_print(
                    lambda: (
                        "BATTLE_HANDLER_STRICT_STOP: "
                        f"strategy={type(strategy).__name__!r} "
                        f"main_callback={get_main_battle_callback()!r} "
                        f"script={get_current_battle_script_instruction()!r} "
                        f"game_state={get_game_state()!r} "
                        "reason='unhandled_battle_handler_branch'"
                    ),
                    trace=True,
                    prefix="BATTLE_HANDLER_STRICT_STOP:",
                )
                raise BotModeError(
                    "Campaign Progression encountered an unhandled battle state; "
                    "the legacy fallback input was disabled."
                )
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
    terminal_knowledge = observe_current_battle()
    terminal_actions = get_last_completed_battle_actions()
    battle_event_count = _print_battle_events(terminal_knowledge, battle_event_count)
    _flush_pending_switch_resolutions()

    outcome = get_last_battle_outcome()
    party_indices_with_picked_up_items = []
    party_indices_that_took_damage_or_changed_status = []
    party_indices_that_gained_exp = []
    party_indices_that_evolved = []
    experience_gained: list[tuple[int, int]] = []

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
            experience_gained.append((index, after.total_exp - before.total_exp))

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
        experience_gained,
    )

    notify_battle_end = getattr(strategy, "notify_battle_ended", None)
    if callable(notify_battle_end):
        notify_battle_end(outcome)

    global _last_handled_battle_result
    _last_handled_battle_result = result

    names_by_party = {}
    try:
        names_by_party.update(
            {
                pokemon.index: battle_pokemon_display_name(pokemon, party=get_party())
                for pokemon in get_party()
            }
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        pass
    _print_battle_end(
        outcome,
        knowledge=terminal_knowledge,
        actions=terminal_actions,
        experience_gained=(
            (names_by_party.get(index, f"slot {index}"), amount)
            for index, amount in experience_gained
        ),
    )

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

    fainted_name = battle_pokemon_display_name(fainted_pokemon, party=get_party())
    # A voluntary switch decision belongs to the preceding action boundary.
    # Forced replacement diagnostics must be generated from this faint event
    # and the replacement strategy, never from stale decision text.
    context.battle_decision_source = None
    context.battle_decision_detail = None
    context.battle_evaluation_method = None
    print_battle_event(
        event="pokemon fainted",
        side="player",
        pokemon=fainted_name,
        detail=f"party slot {getattr(fainted_pokemon, 'party_index', None)} reached 0 HP",
    )
    diagnostic_print(
        lambda: (
            "BATTLE_EVENT: event='pokemon_fainted' side='player' "
            f"pokemon={fainted_name!r} party_index={getattr(fainted_pokemon, 'party_index', None)!r}"
        ),
        trace=True,
        prefix="BATTLE_EVENT:",
    )
    party_index = getattr(fainted_pokemon, "party_index", None)
    authoritative_name = fainted_name
    try:
        party_member = get_party()[party_index]
        identity = PokemonIdentity.from_pokemon(party_member)
        mode = getattr(context, "bot_mode_instance", None)
        pending = getattr(mode, "_pending_fainted_identities", None)
        if identity is not None and mode is not None:
            if pending is None:
                pending = set()
                mode._pending_fainted_identities = pending
            if isinstance(pending, set):
                pending.add(identity)
            else:
                mode._pending_fainted_identities = frozenset(pending) | {identity}
        records = getattr(context, "battle_authoritative_faints", None)
        if not isinstance(records, list):
            records = []
            context.battle_authoritative_faints = records
        records.append((party_index, authoritative_name, identity))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        pass
    notify_faint = getattr(strategy, "notify_pokemon_fainted", None)
    if callable(notify_faint):
        notify_faint()

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
        f"party={[(battle_pokemon_display_name(pokemon, party=get_party()), pokemon.current_hp, pokemon.is_egg) for pokemon in get_party()]!r} "
            f"active={[battler.party_index for battler in battle_state.own_side.active_battlers]!r}"
        ),
        trace=True,
    )
    if not replacement_available:
        print_forced_replacement(
            fainted=fainted_name,
            selected=None,
            party_index=None,
            reason="no legal living party member remains; advancing into whiteout",
            available=False,
        )
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

    forced_reason = (
        getattr(context, "battle_decision_detail", None)
        or "forced replacement selected by the battle strategy"
    )
    print_forced_replacement(
        fainted=fainted_name,
        selected=battle_pokemon_display_name(new_lead, party=get_party()),
        party_index=regular_new_lead_index,
        reason=forced_reason,
        available=True,
    )
    diagnostic_print(
        lambda: (
            "BATTLE_FORCED_REPLACEMENT: "
            f"fainted={fainted_name!r} selected={battle_pokemon_display_name(new_lead, party=get_party())!r} "
            f"party_index={regular_new_lead_index!r} reason={forced_reason!r}"
        ),
        trace=True,
        prefix="BATTLE_FORCED_REPLACEMENT:",
    )

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
