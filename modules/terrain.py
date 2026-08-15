"""Static terrain traversal costs used by navigation."""

NORMAL_TERRAIN_COST = 1
ENCOUNTER_TERRAIN_COST = 2

# These names are the ROM tile types for walkable tall grass across the
# supported Gen 3 games.  Keep the mapping here so navigation does not need
# to interpret ROM/map data itself.
TALL_GRASS_TILE_TYPES = frozenset({
    "Tall Grass",
    "Long Grass",
    "Long Grass South Edge",
})


def terrain_cost(tile_type: str) -> int:
    """Return the static cost of entering a tile with the given map type."""
    if tile_type in TALL_GRASS_TILE_TYPES:
        return ENCOUNTER_TERRAIN_COST
    return NORMAL_TERRAIN_COST
