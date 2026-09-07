from modules.map import MapLocation
from modules.map_data import MapFRLG, PokemonCenter, MapRSE, get_map_enum
from modules.map_path import calculate_path, PathFindingError
from modules.modes import BotModeError
from modules.player import get_player_location
from modules.console import diagnostic_print
import json
from time import perf_counter_ns

_closest_pokemon_centers: dict[MapFRLG | MapRSE, list[PokemonCenter]] = {
    # Hoenn
    MapRSE.ROUTE101: [PokemonCenter.OldaleTown],
    MapRSE.ROUTE102: [PokemonCenter.OldaleTown, PokemonCenter.PetalburgCity],
    MapRSE.ROUTE103: [PokemonCenter.OldaleTown],
    MapRSE.ROUTE104: [PokemonCenter.PetalburgCity, PokemonCenter.RustboroCity],
    # The Woods are a separate dungeon map rather than a child of Route 104,
    # so name-prefix parent discovery cannot infer their nearby source.
    MapRSE.PETALBURG_WOODS: [PokemonCenter.PetalburgCity],
    MapRSE.ROUTE105: [PokemonCenter.PetalburgCity, PokemonCenter.DewfordTown],
    MapRSE.ROUTE106: [PokemonCenter.DewfordTown],
    MapRSE.ROUTE107: [PokemonCenter.DewfordTown],
    MapRSE.ROUTE108: [PokemonCenter.DewfordTown],
    MapRSE.ROUTE109: [PokemonCenter.SlateportCity],
    MapRSE.ROUTE110: [PokemonCenter.SlateportCity, PokemonCenter.MauvilleCity],
    MapRSE.ROUTE111: [
        PokemonCenter.MauvilleCity,
        PokemonCenter.MauvilleCity,
        PokemonCenter.FallarborTown,
    ],
    MapRSE.ROUTE112: [
        PokemonCenter.LavaridgeTown,
        PokemonCenter.MauvilleCity,
        PokemonCenter.FallarborTown,
    ],
    MapRSE.ROUTE113: [PokemonCenter.FallarborTown],
    MapRSE.ROUTE114: [PokemonCenter.FallarborTown],
    MapRSE.ROUTE115: [PokemonCenter.RustboroCity],
    MapRSE.ROUTE116: [PokemonCenter.RustboroCity],
    MapRSE.ROUTE117: [PokemonCenter.MauvilleCity, PokemonCenter.VerdanturfTown],
    MapRSE.ROUTE118: [PokemonCenter.MauvilleCity],
    MapRSE.ROUTE119: [PokemonCenter.FortreeCity, PokemonCenter.MauvilleCity],
    MapRSE.ROUTE120: [PokemonCenter.FortreeCity],
    MapRSE.ROUTE121: [PokemonCenter.LilycoveCity],
    MapRSE.ROUTE122: [PokemonCenter.LilycoveCity],
    MapRSE.ROUTE123: [PokemonCenter.LilycoveCity, PokemonCenter.MauvilleCity],
    MapRSE.ROUTE124: [PokemonCenter.LilycoveCity, PokemonCenter.MossdeepCity],
    MapRSE.ROUTE125: [PokemonCenter.MossdeepCity],
    MapRSE.ROUTE126: [PokemonCenter.MossdeepCity],
    MapRSE.ROUTE127: [PokemonCenter.MossdeepCity],
    MapRSE.ROUTE128: [PokemonCenter.EvergrandeCity],
    MapRSE.ROUTE129: [PokemonCenter.EvergrandeCity],
    MapRSE.ROUTE130: [PokemonCenter.PacifidlogTown],
    MapRSE.ROUTE131: [PokemonCenter.PacifidlogTown],
    MapRSE.ROUTE132: [PokemonCenter.PacifidlogTown],
    MapRSE.ROUTE133: [PokemonCenter.PacifidlogTown, PokemonCenter.SlateportCity],
    MapRSE.ROUTE134: [PokemonCenter.SlateportCity],
    MapRSE.PETALBURG_CITY: [PokemonCenter.PetalburgCity],
    MapRSE.SLATEPORT_CITY: [PokemonCenter.SlateportCity],
    MapRSE.MAUVILLE_CITY: [PokemonCenter.MauvilleCity],
    MapRSE.RUSTBORO_CITY: [PokemonCenter.RustboroCity],
    MapRSE.FORTREE_CITY: [PokemonCenter.FortreeCity],
    MapRSE.LILYCOVE_CITY: [PokemonCenter.LilycoveCity],
    MapRSE.MOSSDEEP_CITY: [PokemonCenter.MossdeepCity],
    MapRSE.EVER_GRANDE_CITY: [PokemonCenter.EvergrandeCity],
    # Littleroot has no Pokémon Center, but its indoor maps can still be
    # exited and routed to Oldale.  Keeping the outdoor parent in the same
    # registry lets every Littleroot interior use the normal cross-map
    # recovery planner.
    MapRSE.LITTLEROOT_TOWN: [PokemonCenter.OldaleTown],
    MapRSE.OLDALE_TOWN: [PokemonCenter.OldaleTown],
    # Birch's Lab has no Center of its own, but the player can leave
    # Littleroot and reach Oldale's Center through Route 101.  Keep the
    # interior in the recovery registry so post-battle readiness can still
    # route an injured party to a healing source after the lab dialogue.
    MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB: [PokemonCenter.OldaleTown],
    MapRSE.DEWFORD_TOWN: [PokemonCenter.DewfordTown],
    MapRSE.LAVARIDGE_TOWN: [PokemonCenter.LavaridgeTown],
    MapRSE.FALLARBOR_TOWN: [PokemonCenter.FallarborTown],
    MapRSE.VERDANTURF_TOWN: [PokemonCenter.VerdanturfTown],
    MapRSE.PACIFIDLOG_TOWN: [PokemonCenter.PacifidlogTown],
    # Kanto
    MapFRLG.ROUTE1: [PokemonCenter.PalletTown, PokemonCenter.ViridianCity],
    MapFRLG.ROUTE2: [PokemonCenter.ViridianCity, PokemonCenter.PewterCity],
    MapFRLG.ROUTE3: [PokemonCenter.PewterCity, PokemonCenter.Route4],
    MapFRLG.ROUTE4: [PokemonCenter.Route4, PokemonCenter.CeruleanCity],
    MapFRLG.ROUTE6: [PokemonCenter.VermilionCity],
    MapFRLG.ROUTE7: [PokemonCenter.CeladonCity],
    MapFRLG.ROUTE9: [PokemonCenter.Route10],
    MapFRLG.ROUTE10: [PokemonCenter.Route10],
    MapFRLG.ROUTE11: [PokemonCenter.VermilionCity],
    MapFRLG.ROUTE18: [PokemonCenter.FuchsiaCity],
    MapFRLG.ROUTE19: [PokemonCenter.FuchsiaCity],
    MapFRLG.ROUTE20: [PokemonCenter.CinnabarIsland, PokemonCenter.FuchsiaCity],
    MapFRLG.ROUTE21_NORTH: [PokemonCenter.PalletTown, PokemonCenter.CinnabarIsland],
    MapFRLG.ROUTE21_SOUTH: [PokemonCenter.PalletTown, PokemonCenter.CinnabarIsland],
    MapFRLG.ROUTE22: [PokemonCenter.ViridianCity],
    MapFRLG.ROUTE24: [PokemonCenter.CeruleanCity],
    MapFRLG.VIRIDIAN_CITY: [PokemonCenter.ViridianCity],
    MapFRLG.PEWTER_CITY: [PokemonCenter.PewterCity],
    MapFRLG.CERULEAN_CITY: [PokemonCenter.CeruleanCity],
    MapFRLG.LAVENDER_TOWN: [PokemonCenter.LavenderTown],
    MapFRLG.VERMILION_CITY: [PokemonCenter.VermilionCity],
    MapFRLG.CELADON_CITY: [PokemonCenter.CeladonCity],
    MapFRLG.FUCHSIA_CITY: [PokemonCenter.FuchsiaCity],
    MapFRLG.CINNABAR_ISLAND: [PokemonCenter.CinnabarIsland],
    MapFRLG.SAFFRON_CITY: [PokemonCenter.SaffronCity],
}


def _recovery_parent_map(map_id: MapFRLG | MapRSE) -> MapFRLG | MapRSE | None:
    """Return the registered outdoor map owning an indoor map.

    The map groups in RSE/FRLG use stable names such as
    ``PETALBURG_CITY_GYM`` and ``ROUTE104_PRETTY_PETAL_FLOWER_SHOP``.
    Recovery previously required an exact outdoor-map key, so a battle that
    ended inside any ordinary building appeared to have no healing route.
    Use the longest registered name prefix as a ROM-derived boundary.  This
    does not claim that the indoor map is itself a Center; it only supplies
    the outdoor recovery candidates whose route is then validated by the
    world planner.
    """

    try:
        enum_map = map_id if isinstance(map_id, (MapFRLG, MapRSE)) else get_map_enum(map_id)
    except (TypeError, ValueError):
        return None
    parents = tuple(
        candidate
        for candidate in _closest_pokemon_centers
        if isinstance(candidate, type(enum_map)) and enum_map.name.startswith(f"{candidate.name}_")
    )
    return max(parents, key=lambda candidate: len(candidate.name), default=None)


def _candidates_for_map(map_id: MapFRLG | MapRSE) -> tuple[PokemonCenter, ...]:
    """Return direct or ROM-name-derived recovery candidates."""

    candidates = _closest_pokemon_centers.get(map_id, ())
    if candidates:
        return tuple(dict.fromkeys(candidates))
    parent = _recovery_parent_map(map_id)
    return tuple(dict.fromkeys(_closest_pokemon_centers.get(parent, ()))) if parent is not None else ()


def pokemon_center_candidates(
    location: MapLocation | tuple[MapFRLG | MapRSE, tuple[int, int]] | None = None,
) -> tuple[PokemonCenter, ...]:
    """Return all registered recovery sources for the current map."""
    if isinstance(location, MapLocation):
        map_id = location.map_group_and_number
    elif isinstance(location, (MapFRLG, MapRSE)):
        map_id = location
    elif location is not None:
        map_id = location[0]
    else:
        map_id = get_player_location()[0]
    return _candidates_for_map(map_id)


def find_closest_pokemon_center(
    location: MapLocation | tuple[MapFRLG | MapRSE, tuple[int, int]] | None = None,
) -> PokemonCenter:
    if isinstance(location, MapLocation):
        training_spot_map = location.map_group_and_number
    elif isinstance(location, (MapFRLG, MapRSE)):
        training_spot_map = location
    elif location is not None:
        training_spot_map = location[0]
    else:
        training_spot_map = get_player_location()[0]
    pokemon_center = None
    path_length_to_pokemon_center = None

    candidates = pokemon_center_candidates(training_spot_map)
    parent_map = _recovery_parent_map(training_spot_map)
    diagnostic_print(
        lambda: "recovery_center_candidate_discovery: "
        + json.dumps(
            {
                "source": location,
                "source_map": training_spot_map,
                "parent_map": parent_map,
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
            default=str,
            sort_keys=True,
        ),
        trace=True,
    )
    if candidates:
        for pokemon_center_candidate in candidates:
            operation_started = perf_counter_ns()
            path_to = path_from = None
            try:
                diagnostic_print(
                    lambda: "recovery_path_start: "
                    + json.dumps(
                        {
                            "operation": "path_to",
                            "source": location,
                            "destination_center": pokemon_center_candidate,
                            "candidate": pokemon_center_candidate,
                        },
                        default=str,
                        sort_keys=True,
                    ),
                    trace=True,
                )
                path_to = calculate_path(location, pokemon_center_candidate.value)
                diagnostic_print(
                    lambda: "recovery_path_complete: "
                    + json.dumps(
                        {
                            "operation": "path_to",
                            "candidate": pokemon_center_candidate,
                            "path_length": len(path_to),
                            "duration_ms": round((perf_counter_ns() - operation_started) / 1_000_000, 3),
                            "success": True,
                        },
                        default=str,
                        sort_keys=True,
                    ),
                    trace=True,
                )
                operation_started = perf_counter_ns()
                diagnostic_print(
                    lambda: "recovery_path_start: "
                    + json.dumps(
                        {
                            "operation": "path_from",
                            "source": pokemon_center_candidate.value,
                            "destination": location,
                            "candidate": pokemon_center_candidate,
                        },
                        default=str,
                        sort_keys=True,
                    ),
                    trace=True,
                )
                path_from = calculate_path(pokemon_center_candidate.value, location)
                diagnostic_print(
                    lambda: "recovery_path_complete: "
                    + json.dumps(
                        {
                            "operation": "path_from",
                            "candidate": pokemon_center_candidate,
                            "path_length": len(path_from),
                            "duration_ms": round((perf_counter_ns() - operation_started) / 1_000_000, 3),
                            "success": True,
                        },
                        default=str,
                        sort_keys=True,
                    ),
                    trace=True,
                )
                path_length = len(path_to) + len(path_from)
                diagnostic_print(
                    lambda: "recovery_path_complete: "
                    + json.dumps(
                        {
                            "candidate": pokemon_center_candidate,
                            "path_length": path_length,
                            "duration_ms": round((perf_counter_ns() - operation_started) / 1_000_000, 3),
                            "success": True,
                        },
                        default=str,
                        sort_keys=True,
                    ),
                    trace=True,
                )
                if path_length_to_pokemon_center is None or path_length < path_length_to_pokemon_center:
                    pokemon_center = pokemon_center_candidate
                    path_length_to_pokemon_center = path_length
                    diagnostic_print(
                        lambda: "recovery_center_selected: "
                        + json.dumps(
                            {"candidate": pokemon_center_candidate, "path_length": path_length},
                            default=str,
                            sort_keys=True,
                        ),
                        trace=True,
                    )
            except PathFindingError as error:
                diagnostic_print(
                    lambda: "recovery_path_failure: "
                    + json.dumps(
                        {
                            "candidate": pokemon_center_candidate,
                            "duration_ms": round((perf_counter_ns() - operation_started) / 1_000_000, 3),
                            "success": False,
                            "exception_type": type(error).__name__,
                            "error": str(error),
                        },
                        default=str,
                        sort_keys=True,
                    ),
                    trace=True,
                )
                pass

    if pokemon_center is None:
        # ``calculate_path`` intentionally does not cross map warps.  An
        # indoor-to-outdoor recovery route is nevertheless executable by the
        # world navigation planner, so retain the first registered Center as
        # the recovery destination and let that planner validate the route.
        if parent_map is not None and training_spot_map != parent_map and candidates:
            diagnostic_print(
                lambda: "recovery_center_selected: "
                + json.dumps(
                    {
                        "candidate": candidates[0],
                        "path_length": None,
                        "selection": "indoor_parent_fallback",
                        "source_map": training_spot_map,
                        "parent_map": parent_map,
                    },
                    default=str,
                    sort_keys=True,
                ),
                trace=True,
            )
            return candidates[0]
        raise BotModeError("Could not find a suitable path from here to a Pokemon Center nearby.")

    return pokemon_center


def map_has_pokemon_center_nearby(map_enum: MapFRLG | MapRSE | tuple[int, int]) -> bool:
    if isinstance(map_enum, tuple):
        map_enum = get_map_enum(map_enum)
    return bool(_candidates_for_map(map_enum))
