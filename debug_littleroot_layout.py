from modules.navigation import GoalAwareNavigator, NavigationWorld, NavigableTile, NavigationGoal, EncounterMode
from modules.goals import ReachInteractionPosition
from modules.map import get_map_data
from modules.map_data import MapRSE
from modules.overworld import Location
from modules.navigation import Location
from modules.map_path import Direction

def debug_path():
    # Littleroot Town is (0, 9)
    # Start: (5,9), Facing: North
    start_pos = (5, 9)
    # The rival house is at (6, 5)? Let's check map layout
    
    # We want to go to the rival's house trigger.
    # The script symbol is "LittlerootTown_MaysHouse_2F_EventScript_RivalsPokeBall" (or similar)
    # Wait, the interaction is usually inside the rival's house.
    
    # I'll create a simple world and try to pathfind from (5,9) to the house entrance (which leads to the interaction)
    # Wait, if I'm on the town map, the rival house is a warp or a door trigger.
    
    # Let's check the map layout for Littleroot town to understand the rival house location.
    map_id = MapRSE.LITTLEROOT_TOWN.value
    print(f"Map ID: {map_id}")
    
    # Find the rival house entrance.
    # It's usually near the top right.
    
    # I'll just print the tile types for the map to see where the house is
    for y in range(0, 15):
        row = ""
        for x in range(0, 15):
            tile = get_map_data(map_id, (x, y))
            if (x, y) == (5, 9):
                row += "S"
            elif tile.tile_type == "Warp": # Assuming tile_type is Warp or similar
                row += "W"
            else:
                row += "."
        print(row)

if __name__ == "__main__":
    debug_path()
