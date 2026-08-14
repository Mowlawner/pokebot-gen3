import random
from typing import Generator, Callable

from modules.console import diagnostic_print
from modules.context import context
from modules.encounter import handle_encounter, EncounterInfo
from modules.gui.multi_select_window import Selection, ask_for_choice
from modules.map_data import MapFRLG, MapRSE
from modules.memory import GameState, get_game_state, get_game_state_symbol
from modules.menuing import PokemonPartyMenuNavigator, StartMenuNavigator
from modules.modes.util.walking import navigate_to
from modules.player import get_player_avatar, player_avatar_is_controllable
from modules.pokemon_party import get_party_size, get_party
from modules.runtime import get_sprites_path
from modules.save_data import get_save_data
from modules.tasks import get_global_script_context, get_tasks
from ._asserts import SavedMapLocation, assert_save_game_exists, assert_saved_on_map
from ._interface import BattleAction, BotMode, BotModeError
from .util import (
    ensure_facing_direction,
    soft_reset,
    wait_for_n_frames,
    wait_for_task_to_start_and_finish,
    wait_for_unique_rng_value,
    wait_until_task_is_active,
    wait_until_task_is_not_active,
)
from ..battle_state import get_main_battle_callback, EncounterType


HOENN_STARTERS = ("Treecko", "Torchic", "Mudkip")


def _configured_hoenn_starter() -> str | None:
    start_game = getattr(getattr(context, "config", None), "start_game", None)
    configured = getattr(start_game, "starter", None)
    if configured in (*HOENN_STARTERS, "Random"):
        return configured
    return None


def _resolve_hoenn_starter(starter_choice: str) -> str:
    if starter_choice == "Random":
        return random.choice(HOENN_STARTERS)
    return starter_choice


def _report_starters_state(
    phase: str,
    decision: str,
    *,
    starter_selection_mode: str = "interactive_prompt",
    starter_selection_value: object = "not selected",
) -> None:
    """Report only the state needed to diagnose entry into starter selection."""
    try:
        avatar = get_player_avatar()
        location = (avatar.map_group_and_number, avatar.local_coordinates)
        facing = avatar.facing_direction
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        location = None
        facing = None

    try:
        game_state = get_game_state()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        game_state = None
    try:
        game_state_symbol = get_game_state_symbol()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        game_state_symbol = None
    try:
        controllable = player_avatar_is_controllable()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        controllable = None
    try:
        script = get_global_script_context()
        script_active = script.is_active
        script_stack = script.stack if script_active else []
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        script_active = None
        script_stack = None
    try:
        active_tasks = [task.symbol for task in (get_tasks() or [])]
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        active_tasks = None
    try:
        controller_depth = len(context.controller_stack)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        controller_depth = None

    diagnostic_print(
        "[bold yellow]Starters phase: "
        f"phase={phase} decision={decision!r} "
        f"mode={getattr(context, 'bot_mode', None)!r} "
        f"mode_instance={type(getattr(context, 'bot_mode_instance', None)).__name__} "
        f"controller_depth={controller_depth!r} "
        f"location={location!r} facing={facing!r} "
        f"game_state={game_state!r} game_state_symbol={game_state_symbol!r} "
        f"controllable={controllable!r} script_active={script_active!r} "
        f"script_stack={script_stack!r} active_tasks={active_tasks!r} "
        f"starter_selection_mode={starter_selection_mode!r} "
        f"starter_selection_value={starter_selection_value!r}[/]",
        trace=True,
    )


def run_frlg() -> Generator:
    starter_choice = ask_for_choice(
        [
            Selection("Bulbasaur", get_sprites_path() / "pokemon" / "normal" / "Bulbasaur.png"),
            Selection(
                "Charmander",
                get_sprites_path() / "pokemon" / "normal" / "Charmander.png",
            ),
            Selection("Squirtle", get_sprites_path() / "pokemon" / "normal" / "Squirtle.png"),
            Selection("Random", get_sprites_path() / "pokemon" / "normal" / "Unown (qm).png"),
        ],
        window_title="Select a starter...",
    )
    if starter_choice is None:
        return

    while context.bot_mode != "Manual":
        yield from soft_reset(mash_random_keys=True)
        starter = starter_choice
        if starter == "Random":
            starter = random.choice(["Bulbasaur", "Charmander", "Squirtle"])
        match starter:
            case "Bulbasaur":
                yield from navigate_to(map=MapFRLG.PALLET_TOWN_PROFESSOR_OAKS_LAB, coordinates=(8, 5))
            case "Charmander":
                yield from navigate_to(map=MapFRLG.PALLET_TOWN_PROFESSOR_OAKS_LAB, coordinates=(10, 5))
            case "Squirtle":
                yield from navigate_to(map=MapFRLG.PALLET_TOWN_PROFESSOR_OAKS_LAB, coordinates=(9, 5))
        yield from ensure_facing_direction("Up")
        yield from wait_for_unique_rng_value()

        # Wait for and confirm the first question (the 'Do you choose ...')
        while get_party_size() == 0:
            context.emulator.press_button("A")
            yield

        # Wait for and say no to the second question (the 'Do you want to give ... a nickname')
        yield from wait_for_task_to_start_and_finish("Task_YesNoMenu_HandleInput", button_to_press="B")

        # Wait for the rival to pick up their starter
        yield from wait_until_task_is_active("Task_Fanfare", button_to_press="B")

        # Wait for the main menu to pop up
        yield from wait_until_task_is_active("Task_StartMenuHandleInput", button_to_press="Start")

        # Spam 'A' until we see the summary screen
        yield from wait_until_task_is_active("Task_DuckBGMForPokemonCry", button_to_press="A")

        handle_encounter(
            EncounterInfo.create(get_party()[0], EncounterType.Gift),
            disable_auto_catch=True,
            do_not_log_battle_action=True,
        )


def run_rse_hoenn(
    get_active_encounter: Callable[[], EncounterInfo],
    reset_before_selection: bool = True,
) -> Generator:
    configured_starter = _configured_hoenn_starter()
    configured_selection = configured_starter is not None
    starter_selection_mode = "configured" if configured_selection else "interactive_prompt"
    _report_starters_state(
        "RSE_HOENN_CHOICE_BOUNDARY",
        "resolve configured starter" if configured_selection else "BEFORE ask_for_choice: call about to begin",
        starter_selection_mode=starter_selection_mode,
        starter_selection_value=configured_starter if configured_selection else "not selected",
    )
    if configured_selection:
        starter_choice = configured_starter
    else:
        # Set up: Ask for starter choice because we cannot deduce that from the player location.
        starter_choice = ask_for_choice(
            [
                Selection("Treecko", get_sprites_path() / "pokemon" / "normal" / "Treecko.png"),
                Selection("Torchic", get_sprites_path() / "pokemon" / "normal" / "Torchic.png"),
                Selection("Mudkip", get_sprites_path() / "pokemon" / "normal" / "Mudkip.png"),
                Selection("Random", get_sprites_path() / "pokemon" / "normal" / "Unown (qm).png"),
            ],
            window_title="Select a starter...",
        )
    _report_starters_state(
        "RSE_HOENN_CHOICE_BOUNDARY",
        "AFTER ask_for_choice: call returned",
        starter_selection_value=starter_choice,
    )
    if starter_choice is None:
        _report_starters_state(
            "RSE_HOENN_ENTRY",
            "starter choice cancelled; return",
            starter_selection_value=starter_choice,
        )
        return
    _report_starters_state("RSE_HOENN_ENTRY", f"starter choice={starter_choice!r}")

    while context.bot_mode != "Manual":
        _report_starters_state("RSE_HOENN_LOOP", "iteration entered")
        if reset_before_selection and not configured_selection:
            _report_starters_state("RSE_HOENN_LOOP", "yielding from soft_reset")
            yield from soft_reset(mash_random_keys=True)
            _report_starters_state("RSE_HOENN_LOOP", "resumed after soft_reset")

        # Starter bag can be accessed from the right or from the bottom, make sure we are looking
        # at it in either case.
        _report_starters_state("RSE_HOENN_BAG_SETUP", "face starter bag before waiting for interaction task")
        avatar = get_player_avatar()
        if avatar.local_coordinates == (8, 14):
            _report_starters_state("RSE_HOENN_BAG_SETUP", "yielding from ensure_facing_direction(Left)")
            yield from ensure_facing_direction("Left")
            _report_starters_state("RSE_HOENN_BAG_SETUP", "resumed after ensure_facing_direction(Left)")
        else:
            _report_starters_state("RSE_HOENN_BAG_SETUP", "yielding from ensure_facing_direction(Up)")
            yield from ensure_facing_direction("Up")
            _report_starters_state("RSE_HOENN_BAG_SETUP", "resumed after ensure_facing_direction(Up)")

        # Open bag
        bag_task = "Task_StarterChoose2" if context.rom.is_rs else "Task_HandleStarterChooseInput"
        _report_starters_state(
            "RSE_HOENN_BAG_SETUP",
            f"wait for {bag_task} and press A",
        )
        if context.rom.is_rs:
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "yielding from wait_until_task_is_active(Task_StarterChoose2)")
            yield from wait_until_task_is_active("Task_StarterChoose2", "A")
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "resumed after wait_until_task_is_active(Task_StarterChoose2)")
        else:
            _report_starters_state(
                "RSE_HOENN_BAG_INTERACTION",
                "yielding from wait_until_task_is_active(Task_HandleStarterChooseInput)",
            )
            yield from wait_until_task_is_active("Task_HandleStarterChooseInput", "A")
            _report_starters_state(
                "RSE_HOENN_BAG_INTERACTION",
                "resumed after wait_until_task_is_active(Task_HandleStarterChooseInput)",
            )
        _report_starters_state("RSE_HOENN_BAG_INTERACTION", "starter task became active; choose configured starter")

        starter = _resolve_hoenn_starter(starter_choice)

        # Select the correct starter
        if starter == "Treecko":
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "direct yield before selecting Treecko")
            yield
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "resumed after direct yield before selecting Treecko")
            context.emulator.press_button("Left")
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "direct yield after selecting Treecko")
            yield
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "resumed after direct yield after selecting Treecko")
        elif starter == "Mudkip":
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "direct yield before selecting Mudkip")
            yield
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "resumed after direct yield before selecting Mudkip")
            context.emulator.press_button("Right")
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "direct yield after selecting Mudkip")
            yield
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "resumed after direct yield after selecting Mudkip")

        if not configured_selection:
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "yielding from wait_for_unique_rng_value")
            yield from wait_for_unique_rng_value()
            _report_starters_state("RSE_HOENN_BAG_INTERACTION", "resumed after wait_for_unique_rng_value")

        # Wait until the starter Pokémon has been sent out into battle before resetting.
        # The Pokémon is already generated as soon as the battle starts, but to make it
        # more like a real person is playing, we wait until the Pokémon is actually
        # _visible_.
        first_turn_callback = "TryDoEventsBeforeFirstTurn" if context.rom.is_emerald else "BattleBeginFirstTurn"
        battle_has_begun = False
        while True:
            main_battle_callback = get_main_battle_callback()
            _report_starters_state(
                "RSE_HOENN_BATTLE_VISIBILITY_LOOP",
                f"iteration: battle_has_begun={battle_has_begun!r} "
                f"main_battle_callback={main_battle_callback!r} "
                f"expected_callback={first_turn_callback!r}",
            )
            if get_game_state() == GameState.OVERWORLD and main_battle_callback == "ReturnFromBattleToOverworld":
                _report_starters_state(
                    "RSE_HOENN_BATTLE_VISIBILITY_LOOP",
                    "battle completed while delegated controller was active; leaving loop",
                )
                break
            if not battle_has_begun and main_battle_callback == first_turn_callback:
                battle_has_begun = True
            elif battle_has_begun and main_battle_callback != first_turn_callback:
                _report_starters_state(
                    "RSE_HOENN_BATTLE_VISIBILITY_LOOP",
                    "visibility condition complete; leaving loop",
                )
                break

            context.emulator.press_button("A")
            _report_starters_state(
                "RSE_HOENN_BATTLE_VISIBILITY_LOOP",
                f"direct yield: battle_has_begun={battle_has_begun!r} "
                f"main_battle_callback={main_battle_callback!r}",
            )
            yield
            _report_starters_state(
                "RSE_HOENN_BATTLE_VISIBILITY_LOOP",
                "resumed after direct yield",
            )

        _report_starters_state(
            "RSE_HOENN_BATTLE_READY",
            "starter received and opening battle is visible",
            starter_selection_mode=starter_selection_mode,
            starter_selection_value=starter,
        )

        diagnostic_print("STARTER_FLOW: run_rse_hoenn entered battle delegation", trace=True)
        handle_encounter(
            get_active_encounter(),
            do_not_log_battle_action=True,
            disable_auto_catch=True,
        )
        diagnostic_print("STARTER_FLOW: run_rse_hoenn encounter handler returned", trace=True)

        # The opening handoff is a one-shot sequence.  The normal saved-game
        # mode keeps looping here so it can reset and hunt for a shiny starter,
        # but the opening controller must return as soon as the starter has
        # been received and the first battle has begun.
        if not reset_before_selection:
            _report_starters_state(
                "RSE_HOENN_COMPLETE",
                "opening battle begun; return control to the normal bot controller",
                starter_selection_mode=starter_selection_mode,
                starter_selection_value=starter,
            )
            diagnostic_print("STARTER_FLOW: run_rse_hoenn returning to StartersMode", trace=True)
            return


def run_rse_johto(get_active_encounter: Callable[[], EncounterInfo]):
    starter_choice = ask_for_choice(
        [
            Selection("Chikorita", get_sprites_path() / "pokemon" / "normal" / "Chikorita.png"),
            Selection("Cyndaquil", get_sprites_path() / "pokemon" / "normal" / "Cyndaquil.png"),
            Selection("Totodile", get_sprites_path() / "pokemon" / "normal" / "Totodile.png"),
            Selection("Random", get_sprites_path() / "pokemon" / "normal" / "Unown (qm).png"),
        ],
        window_title="Select a starter...",
    )
    if starter_choice is None:
        return

    while context.bot_mode != "Manual":
        yield from soft_reset(mash_random_keys=True)
        starter = starter_choice
        if starter == "Random":
            starter = random.choice(["Chikorita", "Cyndaquil", "Totodile"])
        match starter:
            case "Chikorita":
                yield from navigate_to(map=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB, coordinates=(10, 5))
            case "Cyndaquil":
                yield from navigate_to(map=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB, coordinates=(8, 5))
            case "Totodile":
                yield from navigate_to(map=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB, coordinates=(9, 5))

        if get_party_size() >= 6:
            raise BotModeError("This mode requires at least one empty party slot, but your party is full.")

        yield from ensure_facing_direction("Up")

        yield from wait_for_unique_rng_value()

        # Wait for and confirm the first question (the 'Do you choose ...')
        yield from wait_for_task_to_start_and_finish("Task_HandleYesNoInput", button_to_press="A")

        # Wait for and say no to the second question (the 'Do you want to give ... a nickname')
        yield from wait_for_task_to_start_and_finish("Task_HandleYesNoInput", button_to_press="B")

        yield from wait_until_task_is_not_active("Task_Fanfare", "B")
        yield from wait_for_task_to_start_and_finish("Task_DrawFieldMessage", "A")

        yield from wait_for_n_frames(2)
        context.emulator.press_button("A")

        # Navigate to the summary screen to check for shininess
        yield from StartMenuNavigator("POKEMON").step()
        yield from PokemonPartyMenuNavigator(get_party_size() - 1, "summary").step()

        handle_encounter(
            EncounterInfo.create(get_party()[-1], EncounterType.Gift),
            disable_auto_catch=True,
            do_not_log_battle_action=True,
        )


class StartersMode(BotMode):
    @staticmethod
    def name() -> str:
        return "Starters"

    @staticmethod
    def is_selectable() -> bool:
        if context.rom.is_emerald:
            from .opening import OpeningSequenceState, get_opening_sequence_state

            if get_opening_sequence_state() is OpeningSequenceState.STARTER_SELECTION:
                return True
        player_avatar = get_player_avatar()
        if context.rom.is_frlg:
            return player_avatar.map_group_and_number == (4, 3)
        if context.rom.is_rse:
            return player_avatar.map_group_and_number in [(0, 16), (1, 4)]

    def __init__(self):
        super().__init__()
        self._active_encounter: EncounterInfo | None = None
        self._opening_handoff_active = False

    def on_battle_started(self, encounter: EncounterInfo | None) -> BattleAction | None:
        self._active_encounter = encounter
        if self._opening_handoff_active:
            _report_starters_state(
                "RSE_HOENN_BATTLE_HANDOFF",
                "opening battle detected; delegate to the normal battle controller",
            )
            return BattleAction.Fight
        return BattleAction.CustomAction

    def run(self) -> Generator:
        _report_starters_state("MODE_ENTRY", "STARTERS MODE RUN ENTERED")
        # A fresh-game opening controller hands off at the starter bag. This
        # path intentionally does not require a save state.
        from .opening import consume_starter_handoff

        if consume_starter_handoff():
            self._opening_handoff_active = True
            _report_starters_state("MODE_ENTRY", "handoff consumed; enter RSE Hoenn starter generator")
            diagnostic_print("STARTER_FLOW: StartersMode entered battle delegation", trace=True)
            yield from run_rse_hoenn(
                lambda: self._active_encounter,
                reset_before_selection=False,
            )
            diagnostic_print("STARTER_FLOW: StartersMode battle delegation returned", trace=True)
            # The opening controller temporarily switches to this mode so the
            # normal starter generator can own the interaction. Once the
            # opening battle returns, hand ownership back to the opening
            # controller so the main loop can continue the scripted sequence.
            if context.bot_mode == "Starters":
                _report_starters_state(
                    "RSE_HOENN_CONTROLLER_RETURN",
                    "starter generator returned; handing control back to Start New Game",
                )
                diagnostic_print("STARTER_FLOW: StartersMode setting bot_mode=Start New Game", trace=True)
                context.bot_mode = "Start New Game"
            diagnostic_print("STARTER_FLOW: StartersMode returning", trace=True)
            return

        _report_starters_state("MODE_ENTRY", "no handoff pending; validate saved-game starter mode path")

        assert_save_game_exists("There is no saved game. Cannot soft reset.")

        if context.rom.is_frlg:
            assert_saved_on_map(
                [SavedMapLocation(MapFRLG.PALLET_TOWN_PROFESSOR_OAKS_LAB)],
                error_message="The game has not been saved while standing in front of one of the starter Poké balls.",
            )
            yield from run_frlg()
        elif context.rom.is_rse:
            assert_saved_on_map(
                [
                    # Hoenn Starter Bag
                    SavedMapLocation(MapRSE.ROUTE101, (7, 14), facing=True),
                    # Johto Starters (on table)
                    SavedMapLocation(MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB),
                ],
                error_message=(
                    "The game has not been saved in front of the starter Pokémon bag (for Hoenn starters) "
                    "or in front of one of the starter Poké balls (for Johto starters.)"
                ),
            )

            if get_save_data().get_map_group_and_number() == MapRSE.ROUTE101:
                yield from run_rse_hoenn(lambda: self._active_encounter)
            else:
                yield from run_rse_johto(lambda: self._active_encounter)
