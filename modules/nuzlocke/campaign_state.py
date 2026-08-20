"""Read-only campaign facts assembled from normalized state and projections.

This module is intentionally an adapter, not another emulator reader or
projection.  It combines already-materialized values for future campaign
predicates and keeps unavailable values distinct from known empty/false ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Generic, TypeVar

from .events import BattleEnded
from .projection import CampaignProjection, ObservedCampaignState
from .rules import LocationEncounter, NuzlockeCampaignState, NuzlockeRulesProjection
from .identity import PokemonIdentity
from .snapshots import (
    InventorySnapshot,
    NamedFlag,
    NamedVariable,
    NuzlockeSnapshot,
    PartyPokemonSnapshot,
    StoragePokemonSnapshot,
)

T = TypeVar("T")
MapId = tuple[int, int]


class FactStatus(Enum):
    """Whether a fact is available to a planner."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Fact(Generic[T]):
    """A value with explicit availability semantics.

    ``value`` is ``None`` for ``UNKNOWN`` and ``UNAVAILABLE``.  A known empty
    tuple, ``False``, or zero remains a known value.
    """

    value: T | None
    status: FactStatus

    @classmethod
    def known(cls, value: T) -> "Fact[T]":
        return cls(value, FactStatus.KNOWN)

    @classmethod
    def unknown(cls) -> "Fact[T]":
        return cls(None, FactStatus.UNKNOWN)

    @classmethod
    def unavailable(cls) -> "Fact[T]":
        return cls(None, FactStatus.UNAVAILABLE)

    @property
    def is_known(self) -> bool:
        return self.status is FactStatus.KNOWN


class RunStatus(Enum):
    ACTIVE = "active"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CampaignFacts:
    """Semantic, read-only facts derived from one normalized observation."""

    text_speed_fast: Fact[bool]
    new_game_setup_complete: Fact[bool]
    wall_clock_set: Fact[bool]
    rival_met: Fact[bool]
    birch_rescued: Fact[bool]
    starter_obtained: Fact[bool]
    intro_rival_battle_complete: Fact[bool]
    pokedex_received: Fact[bool]
    pokeballs_available: Fact[bool]
    pokeballs_ready: Fact[bool]
    nuzlocke_started: Fact[bool]

    def __getitem__(self, name: str) -> Fact[bool]:
        return getattr(self, name)


def _flag(flags: tuple[NamedFlag, ...], name: str, available: bool) -> Fact[bool]:
    match = next((flag for flag in flags if flag.name == name), None)
    return Fact.known(match.value) if match is not None else (Fact.unknown() if available else Fact.unavailable())


def _var(variables: tuple[NamedVariable, ...], name: str, available: bool) -> Fact[int]:
    match = next((variable for variable in variables if variable.name == name), None)
    return Fact.known(match.value) if match is not None else (Fact.unknown() if available else Fact.unavailable())


def derive_campaign_facts(
    snapshot: NuzlockeSnapshot,
    inventory: Fact[InventorySnapshot],
    nuzlocke_started: Fact[bool],
) -> CampaignFacts:
    observation = snapshot.campaign_observation
    available = observation.available
    text_speed = (
        Fact.known(observation.text_speed == 2)
        if observation.text_speed is not None
        else (Fact.unknown() if available else Fact.unavailable())
    )
    intro = _var(observation.variables, "LITTLEROOT_INTRO_STATE", available)
    rival = _var(observation.variables, "LITTLEROOT_RIVAL_STATE", available)
    lab = _var(observation.variables, "BIRCH_LAB_STATE", available)
    pokemon_get = _flag(observation.flags, "SYS_POKEMON_GET", available)
    setup = Fact.known(intro.value >= 3) if intro.is_known else Fact(None, intro.status)
    rival_met = Fact.known(rival.value >= 3) if rival.is_known else Fact(None, rival.status)
    starter = (
        # State 2 is the ROM-owned lab nickname/rival prompt sequence. Keep
        # obtain_starter active until that script sets state 3 and releases.
        Fact.known(lab.value >= 3 and pokemon_get.value)
        if lab.is_known and pokemon_get.is_known
        else Fact(None, lab.status if lab.status is not FactStatus.KNOWN else pokemon_get.status)
    )
    pokedex = _flag(observation.flags, "RECEIVED_POKEDEX_FROM_BIRCH", available)
    system_dex = _flag(observation.flags, "SYS_POKEDEX_GET", available)
    if pokedex.is_known and system_dex.is_known:
        pokedex = Fact.known(pokedex.value or system_dex.value)
    balls = (
        Fact.known(sum(item.quantity for item in inventory.value.poke_balls) > 0)
        if inventory.is_known
        else Fact(None, inventory.status)
    )
    ready = (
        Fact.known(balls.value and lab.value >= 5)
        if balls.is_known and lab.is_known
        else Fact(None, balls.status if balls.status is not FactStatus.KNOWN else lab.status)
    )
    return CampaignFacts(
        text_speed,
        setup,
        _flag(observation.flags, "SET_WALL_CLOCK", available),
        rival_met,
        _flag(observation.flags, "RESCUED_BIRCH", available),
        starter,
        _flag(observation.flags, "DEFEATED_RIVAL_ROUTE103", available),
        pokedex,
        balls,
        ready,
        nuzlocke_started,
    )


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Planner-facing view of current and replayed campaign facts.

    The object performs no emulator or filesystem access.  Nested snapshot,
    event, and projection values are immutable inputs and are never mutated.
    ``canonical_area`` is supplied by a future map-area adapter; this class
    deliberately does not guess it from a raw map ID.
    """

    raw_map: Fact[MapId]
    canonical_area: Fact[str]
    coordinates: Fact[tuple[int, int]]
    badges: Fact[tuple[NamedFlag, ...]]
    inventory: Fact[InventorySnapshot]
    party: Fact[tuple[PartyPokemonSnapshot, ...]]
    storage: Fact[tuple[StoragePokemonSnapshot, ...]]
    story_flags: Fact[tuple[NamedFlag, ...]]
    encounters: Fact[tuple[LocationEncounter, ...]]
    run_status: Fact[RunStatus]
    dead_pokemon: Fact[frozenset[PokemonIdentity]]
    alive_pokemon: Fact[frozenset[PokemonIdentity]]
    rules_legal: Fact[bool]
    last_completed_battle: Fact[BattleEnded]
    session_id: str | None
    known_session_ids: tuple[str, ...]
    campaign_facts: CampaignFacts = dataclass_field(
        default_factory=lambda: CampaignFacts(*(Fact.unavailable() for _ in range(10)))
    )

    @classmethod
    def from_runtime_state(
        cls,
        *,
        snapshot: NuzlockeSnapshot,
        observed_projection: CampaignProjection | ObservedCampaignState | None = None,
        rules_projection: NuzlockeRulesProjection | NuzlockeCampaignState | None = None,
        canonical_area: str | None = None,
    ) -> "CampaignState":
        """Construct a facade from already-normalized/projected values."""

        observed = (
            observed_projection.state if isinstance(observed_projection, CampaignProjection) else observed_projection
        )
        rules = rules_projection.state if isinstance(rules_projection, NuzlockeRulesProjection) else rules_projection

        raw_map = (
            Fact.known((snapshot.player.map_group, snapshot.player.map_number))
            if snapshot.player_available
            and snapshot.player.map_group is not None
            and snapshot.player.map_number is not None
            else Fact.unknown()
        )
        coordinates = (
            Fact.known(snapshot.player.coordinates)
            if snapshot.player_available and snapshot.player.coordinates is not None
            else Fact.unknown()
        )
        area = Fact.known(canonical_area) if canonical_area is not None else Fact.unknown()

        badges = Fact.known(snapshot.progression.badges) if snapshot.game_state_available else Fact.unknown()
        inventory = Fact.known(snapshot.inventory) if snapshot.inventory_available else Fact.unknown()
        party = Fact.known(snapshot.party) if snapshot.party_available else Fact.unknown()
        storage = Fact.known(snapshot.pc.pokemon) if snapshot.pc_available else Fact.unknown()

        if rules is None:
            encounters = Fact.unknown()
            run_status = Fact.unknown()
            dead_pokemon = Fact.unknown()
            alive_pokemon = Fact.unknown()
            rules_legal = Fact.unknown()
        else:
            encounters = Fact.known(rules.encounters)
            run_status = Fact.known(RunStatus.LOST if rules.run_lost else RunStatus.ACTIVE)
            dead_pokemon = Fact.known(frozenset(rules.dead_pokemon))
            alive_pokemon = Fact.known(frozenset(rules.alive_pokemon))
            rules_legal = Fact.known(rules.legal)

        if observed is None:
            story_flags = Fact.unavailable()
            last_battle = Fact.unavailable()
            session_id = None
            session_ids: tuple[str, ...] = ()
        else:
            # Story/event flags are not part of the current normalized
            # snapshot or event model.  Preserve that limitation explicitly.
            story_flags = Fact.unavailable()
            last_battle = Fact.known(observed.last_battle_end)
            session_id = observed.current_session_id
            session_ids = observed.known_session_ids

        return cls(
            raw_map,
            area,
            coordinates,
            badges,
            inventory,
            party,
            storage,
            story_flags,
            encounters,
            run_status,
            dead_pokemon,
            alive_pokemon,
            rules_legal,
            last_battle,
            session_id,
            session_ids,
            derive_campaign_facts(
                snapshot,
                inventory,
                Fact.known(observed.nuzlocke_started) if observed is not None else Fact.unavailable(),
            ),
        )

    def has_badge(self, name: str) -> Fact[bool]:
        if not self.badges.is_known:
            return Fact(None, self.badges.status)
        return Fact.known(any(flag.name == name and flag.value for flag in self.badges.value or ()))

    def item_quantity(self, name: str) -> Fact[int]:
        if not self.inventory.is_known:
            return Fact(None, self.inventory.status)
        pockets = self.inventory.value.items + self.inventory.value.poke_balls + self.inventory.value.key_items
        quantity = next((item.quantity for item in pockets if item.name == name), 0)
        return Fact.known(quantity)

    def key_item_quantity(self, name: str) -> Fact[int]:
        if not self.inventory.is_known:
            return Fact(None, self.inventory.status)
        quantity = next((item.quantity for item in self.inventory.value.key_items if item.name == name), 0)
        return Fact.known(quantity)

    def encounter_for(self, location: MapId) -> Fact[LocationEncounter]:
        if not self.encounters.is_known:
            return Fact(None, self.encounters.status)
        return Fact.known(
            next(
                (encounter for encounter in self.encounters.value or () if encounter.location == location),
                LocationEncounter(location),
            )
        )

    @property
    def current_encounter(self) -> Fact[LocationEncounter]:
        if not self.raw_map.is_known:
            return Fact(None, self.raw_map.status)
        return self.encounter_for(self.raw_map.value)

    def pokemon_is_dead(self, identity) -> Fact[bool]:
        if not self.dead_pokemon.is_known:
            return Fact(None, self.dead_pokemon.status)
        if identity is None:
            return Fact.unknown()
        return Fact.known(identity in self.dead_pokemon.value)

    def run_is_lost(self) -> Fact[bool]:
        if not self.run_status.is_known:
            return Fact(None, self.run_status.status)
        return Fact.known(self.run_status.value is RunStatus.LOST)

    def run_is_legal(self) -> Fact[bool]:
        if not self.rules_legal.is_known:
            return Fact(None, self.rules_legal.status)
        if self.run_status.value is RunStatus.LOST:
            return Fact.known(False)
        return self.rules_legal
