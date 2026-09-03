"""Emerald locations that can sell ordinary Poké Balls.

This catalog deliberately describes only purchase-capable shops.  The
Lilycove unused-Mart map and Battle Frontier exchange services are not
included: their names resemble a Mart, but they are not ordinary Poké Ball
shops.  The live Mart inventory remains the final authority at execution
time.
"""

from __future__ import annotations

from enum import Enum

from modules.map_data import MapRSE


class PokeballSourceRSE(Enum):
    """Known Emerald Poké Ball purchase sources."""

    OLDALE_POKEMART = (
        "pokemart:oldale",
        MapRSE.OLDALE_TOWN,
        MapRSE.OLDALE_TOWN_MART,
        ("OldaleTown_Mart_EventScript_Clerk",),
        "pokemart",
    )
    PETALBURG_POKEMART = (
        "pokemart:petalburg",
        MapRSE.PETALBURG_CITY,
        MapRSE.PETALBURG_CITY_MART,
        ("PetalburgCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    SLATEPORT_POKEMART = (
        "pokemart:slateport",
        MapRSE.SLATEPORT_CITY,
        MapRSE.SLATEPORT_CITY_MART,
        ("SlateportCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    MAUVILLE_POKEMART = (
        "pokemart:mauville",
        MapRSE.MAUVILLE_CITY,
        MapRSE.MAUVILLE_CITY_MART,
        ("MauvilleCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    RUSTBORO_POKEMART = (
        "pokemart:rustboro",
        MapRSE.RUSTBORO_CITY,
        MapRSE.RUSTBORO_CITY_MART,
        ("RustboroCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    LAVARIDGE_POKEMART = (
        "pokemart:lavaridge",
        MapRSE.LAVARIDGE_TOWN,
        MapRSE.LAVARIDGE_TOWN_MART,
        ("LavaridgeTown_Mart_EventScript_Clerk",),
        "pokemart",
    )
    FALLARBOR_POKEMART = (
        "pokemart:fallarbor",
        MapRSE.FALLARBOR_TOWN,
        MapRSE.FALLARBOR_TOWN_MART,
        ("FallarborTown_Mart_EventScript_Clerk",),
        "pokemart",
    )
    VERDANTURF_POKEMART = (
        "pokemart:verdanturf",
        MapRSE.VERDANTURF_TOWN,
        MapRSE.VERDANTURF_TOWN_MART,
        ("VerdanturfTown_Mart_EventScript_Clerk",),
        "pokemart",
    )
    FORTREE_POKEMART = (
        "pokemart:fortree",
        MapRSE.FORTREE_CITY,
        MapRSE.FORTREE_CITY_MART,
        ("FortreeCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    MOSSDEEP_POKEMART = (
        "pokemart:mossdeep",
        MapRSE.MOSSDEEP_CITY,
        MapRSE.MOSSDEEP_CITY_MART,
        ("MossdeepCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    SOOTOPOLIS_POKEMART = (
        "pokemart:sootopolis",
        MapRSE.SOOTOPOLIS_CITY,
        MapRSE.SOOTOPOLIS_CITY_MART,
        ("SootopolisCity_Mart_EventScript_Clerk",),
        "pokemart",
    )
    LILYCOVE_DEPARTMENT_STORE_2F = (
        "department_store:lilycove:2f",
        MapRSE.LILYCOVE_CITY,
        MapRSE.LILYCOVE_CITY_DEPARTMENT_STORE_2F,
        (
            "LilycoveCity_DepartmentStore_2F_EventScript_ClerkLeft",
            "LilycoveCity_DepartmentStore_2F_EventScript_ClerkRight",
        ),
        "department_store",
    )

    @property
    def source_id(self) -> str:
        """Return the stable catalog identifier."""

        return self.value[0]

    @property
    def outdoor_map(self) -> MapRSE:
        """Return the exterior map containing the source entrance."""

        return self.value[1]

    @property
    def interior_map(self) -> MapRSE:
        """Return the map containing the shop clerk."""

        return self.value[2]

    @property
    def clerk_symbols(self) -> tuple[str, ...]:
        """Return ROM symbols for the shop's clerk affordance."""

        return self.value[3]

    @property
    def source_type(self) -> str:
        """Return the source family for diagnostics and future routing."""

        return self.value[4]

    @property
    def interaction_id(self) -> str:
        """Return the semantic interaction alias used by navigation."""

        return "pokemart_clerk"


def emerald_pokeball_sources() -> tuple[PokeballSourceRSE, ...]:
    """Return all cataloged ordinary Poké Ball purchase sources."""

    return tuple(PokeballSourceRSE)


def emerald_pokeball_sources_for_map(map_id) -> tuple[PokeballSourceRSE, ...]:
    """Return sources whose exterior or clerk map matches ``map_id``."""

    map_id = map_id.value if isinstance(map_id, MapRSE) else map_id
    return tuple(
        source
        for source in PokeballSourceRSE
        if source.outdoor_map.value == map_id or source.interior_map.value == map_id
    )


def emerald_pokeball_source_entrances(source: PokeballSourceRSE, graph=None):
    """Return ROM-derived exterior entrance locations for ``source``.

    Entrances are obtained from the same static world graph used by campaign
    routing.  This keeps coordinates ROM-owned instead of duplicating every
    city door in policy code, and returns an empty tuple when static metadata
    is not available yet (for example during a title-screen boundary).
    """

    if graph is None:
        try:
            from modules.world_navigation import get_world_map_graph

            graph = get_world_map_graph()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return ()
    exterior = source.outdoor_map.value
    interior = source.interior_map.value
    entrances = []
    for edge in getattr(graph, "edges", ()):
        if edge.source_map != exterior or edge.destination_map != interior:
            continue
        entrances.extend((exterior, coordinates) for coordinates in edge.source_coordinates)
    if not entrances:
        # Department Store 2F is reached through the exterior -> 1F -> 2F
        # chain rather than by a direct exterior warp.  Report the first
        # route edge as its exterior entrance while keeping the clerk target
        # on the actual purchasing floor.
        try:
            route = graph.route(exterior, interior)
            if route.edges:
                entrances.extend((exterior, coordinates) for coordinates in route.edges[0].source_coordinates)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return tuple(dict.fromkeys(entrances))
