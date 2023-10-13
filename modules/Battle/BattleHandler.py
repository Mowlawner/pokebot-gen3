from ..Config import config
from ..Enums import GameState
from ..Memory import GetGameState
from ..Menu.Menu import Menu


class BattleHandler:
    """
    Main class to handle battle functions
    """

    def __init__(self):
        self.battle_on = True
        self.battle_mode = config['battle']['battle_method']
        self.move_to_use = -1
        self.mon_to_switch = -1
        self.current_menu = Menu()

    @property
    def in_battle(self):
        return GetGameState() == GameState.BATTLE


