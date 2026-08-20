from modules.map import get_map_data
from modules.map_data import MapRSE


def debug_tiles():
    # Littleroot Town is (0, 9)
    map_id = MapRSE.LITTLEROOT_TOWN
    for y in range(9, 11):
        for x in range(4, 10):
            tile = get_map_data(map_id, (x, y))
            print(
                f"({x},{y}): type={tile.tile_type}, has_encounters={tile.has_encounters}, behaviour={tile._tile_behaviour}"
            )


if __name__ == "__main__":
    debug_tiles()
