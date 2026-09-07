"""Static full-party healing affordances from the Pokémon Emerald ROM.

The campaign planner deals in destinations, while the interaction layer deals
in observed triggers.  This catalog is the bridge between those two layers:
it identifies where a healing script can be reached, but does not pretend to
know the final activation tile until the destination map is observed.
"""

from dataclasses import dataclass
from enum import Enum

from modules.map_data import MapRSE, PokemonCenter


def _map_id_value(map_id):
    """Return the tuple form used by live map observations and world graphs."""

    return getattr(map_id, "value", map_id)


def _normalized_location(location):
    """Normalize catalog locations without changing the public enum API."""

    if not isinstance(location, tuple) or len(location) != 2:
        return location
    return _map_id_value(location[0]), location[1]


class HealingSourceRSE(Enum):
    """Emerald full-party healing sources verified against pret/pokeemerald."""

    OLDALE_POKEMON_CENTER = (
        "pokemon_center:oldale",
        PokemonCenter.OldaleTown.value,
        MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F,
        "OldaleTown_PokemonCenter_1F_EventScript_Nurse",
    )
    PETALBURG_POKEMON_CENTER = (
        "pokemon_center:petalburg",
        PokemonCenter.PetalburgCity.value,
        MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        "PetalburgCity_PokemonCenter_1F_EventScript_Nurse",
    )
    RUSTBORO_POKEMON_CENTER = (
        "pokemon_center:rustboro",
        PokemonCenter.RustboroCity.value,
        MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F,
        "RustboroCity_PokemonCenter_1F_EventScript_Nurse",
    )
    DEWFORD_POKEMON_CENTER = (
        "pokemon_center:dewford",
        PokemonCenter.DewfordTown.value,
        MapRSE.DEWFORD_TOWN_POKEMON_CENTER_1F,
        "DewfordTown_PokemonCenter_1F_EventScript_Nurse",
    )
    SLATEPORT_POKEMON_CENTER = (
        "pokemon_center:slateport",
        PokemonCenter.SlateportCity.value,
        MapRSE.SLATEPORT_CITY_POKEMON_CENTER_1F,
        "SlateportCity_PokemonCenter_1F_EventScript_Nurse",
    )
    MAUVILLE_POKEMON_CENTER = (
        "pokemon_center:mauville",
        PokemonCenter.MauvilleCity.value,
        MapRSE.MAUVILLE_CITY_POKEMON_CENTER_1F,
        "MauvilleCity_PokemonCenter_1F_EventScript_Nurse",
    )
    VERDANTURF_POKEMON_CENTER = (
        "pokemon_center:verdanturf",
        PokemonCenter.VerdanturfTown.value,
        MapRSE.VERDANTURF_TOWN_POKEMON_CENTER_1F,
        "VerdanturfTown_PokemonCenter_1F_EventScript_Nurse",
    )
    LAVARIDGE_POKEMON_CENTER = (
        "pokemon_center:lavaridge",
        PokemonCenter.LavaridgeTown.value,
        MapRSE.LAVARIDGE_TOWN_POKEMON_CENTER_1F,
        "LavaridgeTown_PokemonCenter_1F_EventScript_Nurse",
    )
    FALLARBOR_POKEMON_CENTER = (
        "pokemon_center:fallarbor",
        PokemonCenter.FallarborTown.value,
        MapRSE.FALLARBOR_TOWN_POKEMON_CENTER_1F,
        "FallarborTown_PokemonCenter_1F_EventScript_Nurse",
    )
    FORTREE_POKEMON_CENTER = (
        "pokemon_center:fortree",
        PokemonCenter.FortreeCity.value,
        MapRSE.FORTREE_CITY_POKEMON_CENTER_1F,
        "FortreeCity_PokemonCenter_1F_EventScript_Nurse",
    )
    LILYCOVE_POKEMON_CENTER = (
        "pokemon_center:lilycove",
        PokemonCenter.LilycoveCity.value,
        MapRSE.LILYCOVE_CITY_POKEMON_CENTER_1F,
        "LilycoveCity_PokemonCenter_1F_EventScript_Nurse",
    )
    MOSSDEEP_POKEMON_CENTER = (
        "pokemon_center:mossdeep",
        PokemonCenter.MossdeepCity.value,
        MapRSE.MOSSDEEP_CITY_POKEMON_CENTER_1F,
        "MossdeepCity_PokemonCenter_1F_EventScript_Nurse",
    )
    EVERGRANDE_POKEMON_CENTER = (
        "pokemon_center:evergrande",
        PokemonCenter.EvergrandeCity.value,
        MapRSE.EVER_GRANDE_CITY_POKEMON_CENTER_1F,
        "EverGrandeCity_PokemonCenter_1F_EventScript_Nurse",
    )
    PACIFIDLOG_POKEMON_CENTER = (
        "pokemon_center:pacifidlog",
        PokemonCenter.PacifidlogTown.value,
        MapRSE.PACIFIDLOG_TOWN_POKEMON_CENTER_1F,
        "PacifidlogTown_PokemonCenter_1F_EventScript_Nurse",
    )
    PLAYERS_MOTHER = (
        "players_house:mother",
        (MapRSE.LITTLEROOT_TOWN, (14, 8)),
        MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
        "PlayersHouse_1F_EventScript_Mom",
    )
    ROUTE111_REST_STOP = (
        "route111:old_lady_rest_stop",
        (MapRSE.ROUTE111, (26, 18)),
        MapRSE.ROUTE111_OLD_LADYS_REST_STOP,
        "Route111_OldLadysRestStop_EventScript_OldLady",
    )
    ROUTE119_WEATHER_INSTITUTE_BED = (
        "route119:weather_institute_bed",
        (MapRSE.ROUTE119, (6, 32)),
        MapRSE.ROUTE119_WEATHER_INSTITUTE_1F,
        "Route119_WeatherInstitute_1F_EventScript_Bed",
    )
    SS_TIDAL_BED = (
        "ss_tidal:bed",
        (MapRSE.SS_TIDAL_ROOMS, (15, 11)),
        MapRSE.SS_TIDAL_ROOMS,
        "SSTidalRooms_EventScript_Bed",
    )

    @property
    def source_id(self) -> str:
        """Return the stable identifier used by healing execution."""

        return self.value[0]

    @property
    def outdoor_location(self):
        """Return the outdoor map and coordinates leading to this source."""

        return self.value[1]

    @property
    def interior_map(self) -> MapRSE:
        """Return the interior map containing the healing interaction."""

        return self.value[2]

    @property
    def script_symbol(self) -> str:
        """Return the ROM script symbol for this healing interaction."""

        return self.value[3]


def emerald_healing_sources() -> tuple[HealingSourceRSE, ...]:
    """Return every cataloged Emerald full-party healing source."""

    return tuple(HealingSourceRSE)


def emerald_healing_sources_for_map(map_id) -> tuple[HealingSourceRSE, ...]:
    """Return catalog sources whose outdoor entrance is on ``map_id``."""
    normalized_map = _map_id_value(map_id)
    return tuple(
        source for source in HealingSourceRSE if _normalized_location(source.outdoor_location)[0] == normalized_map
    )


def emerald_healing_source_for_destination(destination) -> HealingSourceRSE | None:
    """Return the source whose outdoor destination exactly matches ``destination``."""

    normalized_destination = _normalized_location(destination)
    return next(
        (source for source in HealingSourceRSE if _normalized_location(source.outdoor_location) == normalized_destination),
        None,
    )
