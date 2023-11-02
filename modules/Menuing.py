from modules.Config import config, ForceManualMode
from modules.Gui import GetROM, GetEmulator, SetMessage
from modules.Memory import GetGameState, GetCursorOptions, ParseTasks, GetTaskFunc, GetTask
from modules.Console import console
from modules.Enums import GameState, TaskFunc
from modules.MenuParsers import (
    get_party_menu_cursor_pos,
    parse_start_menu,
    parse_party_menu,
    parse_menu,
)
from modules.Pokemon import GetParty


class BaseMenuNavigator:
    def __init__(self, step: str = "None"):
        self.navigator = None
        self.current_step = step

    def step(self):
        """
        Iterates through the steps of navigating the menu for the desired outcome.
        """
        while not self.current_step == "exit":
            if not self.navigator:
                self.get_next_func()
                self.update_navigator()
            else:
                for _ in self.navigator:
                    yield _
                self.navigator = None

    def get_next_func(self):
        """
        Advances through the steps of navigating the menu.
        """
        ...

    def update_navigator(self):
        """
        Sets the navigator for the object to follow the steps for the desired outcome.
        """
        ...


class PokemonPartySubMenuNavigator(BaseMenuNavigator):
    def __init__(self, desired_option: str | int):
        super().__init__()
        self.party_menu_internal = None
        self.update_party_menu()
        self.wait_counter = 0
        self.desired_option = desired_option

    def update_party_menu(self):
        party_menu_internal = parse_party_menu()
        if self.party_menu_internal != party_menu_internal:
            self.party_menu_internal = party_menu_internal

    def wait_for_init(self):
        while self.party_menu_internal["numActions"] > 8:
            if self.wait_counter > 30:
                SetMessage("Error navigating menu. Manual mode is now on.")
                ForceManualMode()
            self.update_party_menu()
            self.wait_counter += 1
            yield

    def get_index_from_option(self) -> int:
        for i in range(self.party_menu_internal["numActions"]):
            if GetCursorOptions(self.party_menu_internal["actions"][i]) == self.desired_option or (
                self.desired_option in ("SHIFT", "SWITCH", "SEND_OUT")
                and GetCursorOptions(self.party_menu_internal["actions"][i]) in ("SEND_OUT", "SWITCH", "SHIFT")
            ):
                return i
        SetMessage(f"Couldn't find option {self.desired_option}. Switching to manual mode.")
        ForceManualMode()

    def select_desired_option(self):
        if isinstance(self.desired_option, str):
            self.desired_option = self.get_index_from_option()
        if self.desired_option < 0 or self.desired_option > parse_menu()["maxCursorPos"]:
            SetMessage(f"Error selecting option {self.desired_option}. Switching to manual mode.")
            ForceManualMode()
        while parse_menu()["cursorPos"] != self.desired_option:
            if parse_menu()["cursorPos"] < self.desired_option:
                up_presses = parse_menu()["cursorPos"] + self.party_menu_internal["numActions"] - self.desired_option
                down_presses = self.desired_option - parse_menu()["cursorPos"]
            else:
                up_presses = parse_menu()["cursorPos"] - self.desired_option
                down_presses = self.desired_option - parse_menu()["cursorPos"] + self.party_menu_internal["numActions"]
            if down_presses > up_presses:
                GetEmulator().PressButton("Up")
            else:
                GetEmulator().PressButton("Down")
            yield

    @staticmethod
    def confirm_desired_option():
        GetEmulator().PressButton("A")
        yield

    def get_next_func(self):
        match self.current_step:
            case "None":
                self.current_step = "wait_for_init"
            case "wait_for_init":
                self.current_step = "navigate_to_option"
            case "navigate_to_option":
                self.current_step = "confirm_option"
            case "confirm_option":
                self.current_step = "exit"

    def update_navigator(self):
        match self.current_step:
            case "wait_for_init":
                self.navigator = self.wait_for_init()
            case "navigate_to_option":
                self.navigator = self.select_desired_option()
            case "confirm_option":
                self.navigator = self.confirm_desired_option()


class StartMenuNavigator(BaseMenuNavigator):
    """
    Opens the start menu and moves to the option with the desired index from the menu.

    :param desired_option: The option to select from the menu.
    """

    def __init__(self, desired_option: str):
        super().__init__()
        self.desired_option = desired_option
        self.start_menu = parse_start_menu()

    def update_start_menu(self):
        self.start_menu = parse_start_menu()

    def get_next_func(self):
        match self.current_step:
            case "None":
                self.current_step = "open_start_menu"
            case "open_start_menu":
                self.current_step = "navigate_to_option"
            case "navigate_to_option":
                self.current_step = "confirm_option"
            case "confirm_option":
                self.current_step = "exit"

    def update_navigator(self):
        match self.current_step:
            case "open_start_menu":
                self.navigator = self.open_start_menu()
            case "navigate_to_option":
                self.navigator = self.navigate_to_option()
            case "confirm_option":
                self.navigator = self.confirm_option()

    def open_start_menu(self):
        while not self.start_menu["open"]:
            self.update_start_menu()
            GetEmulator().PressButton("Start")
            yield

    def navigate_to_option(self):
        while self.start_menu["cursor_pos"] != self.start_menu["actions"].index(self.desired_option):
            self.update_start_menu()
            if self.start_menu["cursor_pos"] == self.start_menu["actions"].index(self.desired_option):
                up_presses = 0
                down_presses = 0
            elif self.start_menu["cursor_pos"] < self.start_menu["actions"].index(self.desired_option):
                up_presses = (
                    self.start_menu["cursor_pos"]
                    + len(self.start_menu["actions"])
                    - self.start_menu["actions"].index(self.desired_option)
                )
                down_presses = self.start_menu["actions"].index(self.desired_option) - self.start_menu["cursor_pos"]
            else:
                up_presses = self.start_menu["cursor_pos"] - self.start_menu["actions"].index(self.desired_option)
                down_presses = (
                    self.start_menu["actions"].index(self.desired_option)
                    - self.start_menu["cursor_pos"]
                    + len(self.start_menu["actions"])
                )
            if down_presses > up_presses:
                GetEmulator().PressButton("Up")
            elif up_presses > down_presses or (up_presses > 0 or down_presses > 0):
                GetEmulator().PressButton("Down")
            yield

    def confirm_option(self):
        while self.start_menu["open"]:
            self.update_start_menu()
            GetEmulator().PressButton("A")
            yield


class PokemonPartyMenuNavigator(BaseMenuNavigator):
    def __init__(self, idx: int, mode: str, battle_state: bool = False):
        super().__init__()
        self.idx = idx
        self.game = GetROM().game_title
        self.mode = mode
        self.primary_option = None
        self.get_primary_option()
        self.battle_state = battle_state

    def get_primary_option(self):
        if self.mode in ["take_item", "give_item"]:
            self.primary_option = "ITEM"
        if self.mode == "switch":
            self.primary_option = "SWITCH"

    def get_next_func(self):
        match self.current_step:
            case "None":
                self.current_step = "navigate_to_mon"
            case "navigate_to_mon":
                self.current_step = "select_mon"
            case "select_mon":
                self.current_step = "select_option"
            case "select_option":
                match self.mode:
                    case "take_item":
                        self.current_step = "select_take_item"
                    case "give_item":
                        self.current_step = "select_give_item"
                    case "switch":
                        self.current_step = "select_switch"
                    case _:
                        self.current_step = "exit"
            case "select_take_item" | "select_give_item" | "select_switch":
                self.current_step = "exit"

    def update_navigator(self):
        match self.current_step:
            case "navigate_to_mon":
                self.navigator = self.navigate_to_mon()
            case "select_mon":
                self.navigator = self.select_mon()
            case "select_option":
                self.navigator = self.select_option()
            case "select_take_item":
                self.navigator = self.select_take_item()
            case "select_give_item":
                self.navigator = self.select_give_item()

    def navigate_to_mon(self):
        while get_party_menu_cursor_pos()["slot_id"] != self.idx:
            if get_party_menu_cursor_pos()["slot_id"] > self.idx:
                GetEmulator().PressButton("Up")
            else:
                GetEmulator().PressButton("Down")
            yield

    def select_mon(self):
        if self.game in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
            while GetTask("TASK_HANDLESELECTIONMENUINPUT") == {}:
                GetEmulator().PressButton("A")
                yield
        else:
            while "SUB_8089D94" not in [task["func"] for task in ParseTasks()]:
                GetEmulator().PressButton("A")
                yield

    def select_option(self):
        if self.game in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
            if self.battle_state:
                while parse_party_menu()["numActions"] > 3:
                    yield from PokemonPartySubMenuNavigator(self.primary_option).step()
                else:
                    while GetTask("TASK_HANDLESELECTIONMENUINPUT") != {} and GetTask("TASK_HANDLESELECTIONMENUINPUT")['isActive']:
                        yield from PokemonPartySubMenuNavigator(self.primary_option).step()
        else:
            while "SUB_8089D94" in [task["func"] for task in ParseTasks()] and "SUB_808A060" not in [
                task["func"] for task in ParseTasks()
            ]:
                yield from PokemonPartySubMenuNavigator(self.primary_option).step()

    def select_shift(self):
        if self.game in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
            while "TASK_HANDLESELECTIONMENUINPUT" in [task["func"] for task in ParseTasks()]:
                yield from PokemonPartySubMenuNavigator("SHIFT").step()
        else:
            while "TASK_HANDLEPOPUPMENUINPUT" in [task["func"] for task in ParseTasks()]:
                yield from PokemonPartySubMenuNavigator(0).step()

    def select_take_item(self):
        if self.game in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
            while GetParty()[self.idx]["item"]["name"] != "None":
                yield from PokemonPartySubMenuNavigator("TAKE_ITEM").step()
        else:
            while "SUB_808A060" in [task["func"] for task in ParseTasks()]:
                yield from PokemonPartySubMenuNavigator(1).step()

    def select_give_item(self):
        if self.game in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
            while GetParty()[self.idx]["item"]["name"] != "None":
                yield from PokemonPartySubMenuNavigator("GIVE_ITEM").step()
        else:
            while "SUB_808A060" in [task["func"] for task in ParseTasks()]:
                yield from PokemonPartySubMenuNavigator(0).step()


class CheckForPickup(BaseMenuNavigator):
    """
    class that handles pickup farming.
    """

    def __init__(self, encounter_total: int):
        super().__init__()
        self.party = GetParty()
        self.pokemon_with_pickup = 0
        self.pokemon_with_pickup_and_item = []
        self.current_mon = -1
        self.pickup_threshold_met = None
        self.check_threshold_met = False
        self.check_pickup_threshold(encounter_total)
        self.checked = False
        self.game = GetROM().game_title
        self.party_menu_opener = None
        self.party_menu_navigator = None

    def get_pokemon_with_pickup_and_item(self):
        for i, mon in enumerate(self.party):
            if mon["ability"] == "Pickup":
                self.pokemon_with_pickup += 1
                if mon["item"]["name"] != "None":
                    self.pokemon_with_pickup_and_item.append(i)

    def check_pickup_threshold(self, encounter_total):
        if config["cheats"]["pickup"]:
            self.check_threshold_met = True
            self.checked = True
        else:
            self.check_threshold_met = encounter_total >= config["battle"]["pickup_check_frequency"]
        self.get_pokemon_with_pickup_and_item()
        self.pickup_threshold_met = (
            self.check_threshold_met and len(self.pokemon_with_pickup_and_item) >= config["battle"]["pickup_threshold"]
        )
        if self.pickup_threshold_met:
            console.print("Pickup threshold is met! Gathering items.")

    def open_party_menu(self):
        while not PartyMenuIsOpen():
            if self.party_menu_opener is None:
                self.party_menu_opener = StartMenuNavigator("POKEMON")
            if self.party_menu_opener.current_step != "exit":
                yield from self.party_menu_opener.step()
            else:
                GetEmulator().PressButton("A")
                yield

    def return_to_party_menu(self):
        if self.game in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
            while GetTask("TASK_PRINTANDWAITFORTEXT") != {} and GetTask("TASK_PRINTANDWAITFORTEXT")["isActive"]:
                GetEmulator().PressButton("B")
                yield
        else:
            while TaskFunc.PARTY_MENU not in [GetTaskFunc(task["func"]) for task in ParseTasks()]:
                GetEmulator().PressButton("B")
                yield

    def should_open_party_menu(self):
        if not config["cheats"]["pickup"] and self.check_threshold_met and not self.checked:
            return True
        elif self.pickup_threshold_met:
            return True
        else:
            return False

    def get_next_func(self):
        match self.current_step:
            case "None":
                if self.should_open_party_menu():
                    self.current_step = "open_party_menu"
                else:
                    self.current_step = "exit"
            case "open_party_menu":
                self.checked = True
                if self.pickup_threshold_met:
                    self.current_mon = self.pokemon_with_pickup_and_item[0]
                    self.current_step = "take_mon_item"
                else:
                    self.current_step = "exit_to_overworld"
            case "take_mon_item":
                self.current_step = "return_to_party_menu"
            case "return_to_party_menu":
                if self.current_mon == self.pokemon_with_pickup_and_item[-1]:
                    self.current_step = "exit_to_overworld"
                else:
                    self.get_next_mon()
                    self.current_step = "take_mon_item"
            case "exit_to_overworld":
                self.current_step = "exit"

    def get_next_mon(self):
        next_idx = self.pokemon_with_pickup_and_item.index(self.current_mon) + 1
        if next_idx > len(self.pokemon_with_pickup_and_item) - 1:
            SetMessage("I forgot how to count. Switching to manual mode.")
            ForceManualMode()
        else:
            self.current_mon = self.pokemon_with_pickup_and_item[next_idx]

    def update_navigator(self):
        match self.current_step:
            case "open_party_menu":
                self.navigator = self.open_party_menu()
            case "take_mon_item":
                self.navigator = PokemonPartyMenuNavigator(idx=self.current_mon, mode="take_item", battle_state=False).step()
            case "return_to_party_menu":
                self.navigator = self.return_to_party_menu()
            case "exit_to_overworld":
                self.navigator = PartyMenuExit().step()


class PartyMenuExit(BaseMenuNavigator):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def get_next_func(self):
        match self.current_step:
            case "None":
                self.current_step = "exit_party_menu"
            case "exit_party_menu":
                self.current_step = "wait_for_start_menu"
            case "wait_for_start_menu":
                self.current_step = "exit_start_menu"
            case "exit_start_menu":
                self.current_step = "exit"

    def update_navigator(self):
        match self.current_step:
            case "exit_party_menu" | "exit_start_menu":
                self.navigator = self.exit_menu()
            case "wait_for_start_menu":
                self.navigator = self.wait_for_start_menu()

    @staticmethod
    def exit_menu():
        while GetGameState() != GameState.OVERWORLD or parse_start_menu()["open"]:
            GetEmulator().PressButton("B")
            yield

    def wait_for_start_menu(self):
        while GetGameState() == GameState.OVERWORLD and not parse_start_menu()["open"]:
            if self.counter > 60:
                SetMessage("Error exiting to overworld. Switching to manual mode.")
                ForceManualMode()
            else:
                GetEmulator().PressButton("B")
                self.counter += 1
                yield


def PartyMenuIsOpen() -> bool:
    if GetROM().game_title in ["POKEMON EMER", "POKEMON FIRE", "POKEMON LEAF"]:
        return GetGameState() == GameState.PARTY_MENU
    else:
        return TaskFunc.PARTY_MENU in [GetTaskFunc(task["func"]) for task in ParseTasks()]


def SwitchPokemonActive() -> bool:
    """
    helper function to determine if the switch Pokémon menu is active
    """
    tasks = ParseTasks()
    for task in tasks:
        if task["func"] == "HANDLEPARTYMENUSWITCHPOKEMONINPUT" and task["isActive"]:
            return True
    return False
