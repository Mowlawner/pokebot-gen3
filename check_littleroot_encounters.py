from modules.map import get_wild_encounters_for_map
from modules.map_data import MapRSE
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

def check_encounters():
    # Littleroot Town is (0, 9)
    map_group = 0
    map_number = 9
    
    mock_rom = MagicMock()
    mock_rom.is_frlg = False
    mock_context = SimpleNamespace(rom=mock_rom)
    
    with patch('modules.map.context', mock_context):
        encounters = get_wild_encounters_for_map(map_group, map_number)
        print(f"Encounters: {encounters}")
        if encounters:
            print(f"Land encounters count: {len(encounters.land_encounters)}")
        else:
            print("Encounters is None or empty")

if __name__ == "__main__":
    check_encounters()
