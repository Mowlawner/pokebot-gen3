"""Read-only campaign facts assembled from normalized state and projections.

This module is intentionally an adapter, not another emulator reader or
projection.  It combines already-materialized values for future campaign
predicates and keeps unavailable values distinct from known empty/false ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from enum import Enum
from typing import Generic, TypeVar

from .events import BattleEnded
from .projection import CampaignProjection, ObservedCampaignState
from .rules import LocationEncounter, NuzlockeCampaignState, NuzlockeRulesProjection
from .identity import PokemonIdentity
from .snapshots import (
    CampaignObservationLifecycle,
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
        """Wrap a value that was observed and is authoritative."""

        return cls(value, FactStatus.KNOWN)

    @classmethod
    def unknown(cls) -> "Fact[T]":
        """Create a fact whose source exists but could not be resolved."""

        return cls(None, FactStatus.UNKNOWN)

    @classmethod
    def unavailable(cls) -> "Fact[T]":
        """Create a fact whose source is unavailable at this boundary."""

        return cls(None, FactStatus.UNAVAILABLE)

    @property
    def is_known(self) -> bool:
        """Return whether ``value`` is an authoritative observation."""

        return self.status is FactStatus.KNOWN


class RunStatus(Enum):
    """Reduced lifecycle status of the current ruleset run."""

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
    visited_petalburg: Fact[bool]
    devon_goods_recovered: Fact[bool]
    visited_rustboro: Fact[bool]
    first_badge_obtained: Fact[bool]
    # Emerald's Petalburg Wally sequence is completed by the ROM after the
    # tutorial battle returns the player to the gym.  This is a save-backed
    # variable, not an event-store milestone.
    petalburg_wally_scene_complete: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    # The Woods scene, Goods recovery, and Goods return are separate ROM
    # boundaries.  Keep them distinct so a planner cannot skip the Rusturf
    # battle merely because the Petalburg Woods scene has completed.
    petalburg_woods_scene_complete: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    devon_goods_returned: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    devon_goods_delivered: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    devon_goods_reported: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    devon_goods_stolen: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    rustboro_city_state: Fact[int] = dataclass_field(default_factory=Fact.unavailable)
    rusturf_tunnel_state: Fact[int] = dataclass_field(default_factory=Fact.unavailable)
    devon_corp_3f_state: Fact[int] = dataclass_field(default_factory=Fact.unavailable)
    devon_corp_3f_scene_complete: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)
    roxanne_available: Fact[bool] = dataclass_field(default_factory=Fact.unavailable)

    def __getitem__(self, name: str) -> Fact[bool]:
        """Access a named campaign fact using attribute-style semantics."""

        return getattr(self, name)


def _flag(flags: tuple[NamedFlag, ...], name: str, available: bool) -> Fact[bool]:
    """Resolve a named observed flag while preserving availability semantics."""

    if not available:
        return Fact.unavailable()
    match = next((flag for flag in flags if flag.name == name), None)
    return Fact.known(match.value) if match is not None else Fact.unknown()


def _var(variables: tuple[NamedVariable, ...], name: str, available: bool) -> Fact[int]:
    """Resolve a named observed variable while preserving availability semantics."""

    if not available:
        return Fact.unavailable()
    match = next((variable for variable in variables if variable.name == name), None)
    return Fact.known(match.value) if match is not None else Fact.unknown()


def derive_campaign_facts(
    snapshot: NuzlockeSnapshot,
    inventory: Fact[InventorySnapshot],
    nuzlocke_started: Fact[bool],
) -> CampaignFacts:
    """Derive semantic campaign facts from one snapshot and inventory fact.

    ``nuzlocke_started`` is retained in the call signature for compatibility
    with older adapters, but the default campaign boundary is now the current
    ROM's Pokédex receipt.  Durable ``NuzlockeStarted`` history is not allowed
    to manufacture a boundary for a different save.
    """

    observation = snapshot.campaign_observation
    available = observation.available
    text_speed = (
        Fact.unavailable()
        if not available
        else Fact.known(observation.text_speed == 2) if observation.text_speed is not None else Fact.unknown()
    )
    intro = _var(observation.variables, "LITTLEROOT_INTRO_STATE", available)
    rival = _var(observation.variables, "LITTLEROOT_RIVAL_STATE", available)
    lab = _var(observation.variables, "BIRCH_LAB_STATE", available)
    pokemon_get = _flag(observation.flags, "SYS_POKEMON_GET", available)
    setup = Fact.known(intro.value >= 3) if intro.is_known else Fact(None, intro.status)
    rival_met = Fact.known(rival.value >= 3) if rival.is_known else Fact(None, rival.status)
    # Party insertion happens *before* GiveStarterEvent has handled the
    # starter nickname Yes/No prompt.  It is therefore not completion of the
    # campaign objective: using it here lets CampaignController unmount the
    # only capability that can observe and answer that prompt.  The ROM's
    # post-prompt lab state and SYS_POKEMON_GET are the completion boundary.
    if lab.is_known and pokemon_get.is_known:
        starter = Fact.known(lab.value >= 3 and pokemon_get.value)
    else:
        starter = Fact(None, lab.status if lab.status is not FactStatus.KNOWN else pokemon_get.status)
    pokedex = _flag(observation.flags, "RECEIVED_POKEDEX_FROM_BIRCH", available)
    system_dex = _flag(observation.flags, "SYS_POKEDEX_GET", available)
    if pokedex.is_known and system_dex.is_known:
        pokedex = Fact.known(pokedex.value or system_dex.value)
    defeated_rival = _flag(observation.flags, "DEFEATED_RIVAL_ROUTE103", available)
    hidden_rival = _flag(observation.flags, "HIDE_ROUTE_103_RIVAL", available)
    # Emerald hides the Route 103 rival as the final part of the post-battle
    # scene.  Depending on when the observation is taken, the battle flag and
    # hide flag do not become visible at the same boundary.  Either is a
    # ROM-backed completion signal; retain explicit uncertainty if neither
    # signal is available instead of making a missing flag look like false.
    if defeated_rival.is_known and hidden_rival.is_known:
        intro_rival_complete = Fact.known(defeated_rival.value or hidden_rival.value)
    elif defeated_rival.is_known and defeated_rival.value:
        intro_rival_complete = Fact.known(True)
    elif hidden_rival.is_known and hidden_rival.value:
        intro_rival_complete = Fact.known(True)
    else:
        # Both flags being readable and false is an authoritative negative
        # observation.  Do not manufacture a ``KNOWN`` fact whose value is
        # ``None``: selectors treat that as an incomplete/unknown value and
        # can leave the active capability and GUI target stranded after a
        # transient object disappearance.
        if defeated_rival.is_known and hidden_rival.is_known:
            intro_rival_complete = Fact.known(False)
        else:
            intro_rival_complete = Fact(
                None,
                defeated_rival.status if not defeated_rival.is_known else hidden_rival.status,
            )
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
    petalburg_city_state = _var(observation.variables, "PETALBURG_CITY_STATE", available)
    petalburg_gym_state = _var(observation.variables, "PETALBURG_GYM_STATE", available)
    # The city state reaches 3 immediately before the ROM warps back into the
    # gym for the post-battle return script.  Treating it as completion by
    # itself lets the campaign selector replace the Wally capability while
    # that script still owns the field.  The gym state reaches 2 only after
    # the return dialogue and movement have completed, so both save-backed
    # variables are required for the milestone.
    if petalburg_city_state.is_known and petalburg_city_state.value < 3:
        petalburg_wally_scene = Fact.known(False)
    elif petalburg_city_state.is_known and petalburg_gym_state.is_known:
        petalburg_wally_scene = Fact.known(petalburg_city_state.value >= 3 and petalburg_gym_state.value >= 2)
    else:
        petalburg_wally_scene = Fact(
            None,
            petalburg_gym_state.status if petalburg_city_state.is_known else petalburg_city_state.status,
        )
    petalburg_woods_state = _var(observation.variables, "PETALBURG_WOODS_STATE", available)
    # The Woods researcher/Aqua scene completes in the ROM by setting this
    # map-local state variable.  RECOVERED_DEVON_GOODS is a later flag set in
    # Rusturf Tunnel after the goods are returned, so it cannot be used as the
    # completion boundary for the Woods objective.
    petalburg_woods_scene = (
        Fact.known(petalburg_woods_state.value >= 1)
        if petalburg_woods_state.is_known
        else Fact(None, petalburg_woods_state.status)
    )
    devon_goods_recovered = _flag(observation.flags, "RECOVERED_DEVON_GOODS", available)
    devon_goods_stolen = _flag(observation.flags, "DEVON_GOODS_STOLEN", available)
    devon_goods_returned = _flag(observation.flags, "RETURNED_DEVON_GOODS", available)
    devon_goods_delivered = _flag(observation.flags, "DELIVERED_DEVON_GOODS", available)
    devon_goods_reported = _flag(
        observation.flags,
        "INTERACTED_WITH_DEVON_EMPLOYEE_GOODS_STOLEN",
        available,
    )
    rustboro_city_state = _var(observation.variables, "RUSTBORO_CITY_STATE", available)
    rusturf_tunnel_state = _var(observation.variables, "RUSTURF_TUNNEL_STATE", available)
    devon_corp_3f_state = _var(observation.variables, "DEVON_CORP_3F_STATE", available)
    devon_corp_3f_scene = (
        Fact.known(devon_corp_3f_state.value >= 1)
        if devon_corp_3f_state.is_known
        else Fact(None, devon_corp_3f_state.status)
    )
    visited_rustboro = _flag(observation.flags, "VISITED_RUSTBORO_CITY", available)
    first_badge = _flag(observation.flags, "DEFEATED_RUSTBORO_GYM", available)
    if visited_rustboro.is_known and first_badge.is_known:
        roxanne_available = Fact.known(visited_rustboro.value and not first_badge.value)
    else:
        roxanne_available = Fact(
            None,
            visited_rustboro.status if not visited_rustboro.is_known else first_badge.status,
        )
    return CampaignFacts(
        text_speed,
        setup,
        _flag(observation.flags, "SET_WALL_CLOCK", available),
        rival_met,
        _flag(observation.flags, "RESCUED_BIRCH", available),
        starter,
        intro_rival_complete,
        pokedex,
        balls,
        ready,
        pokedex,
        _flag(observation.flags, "VISITED_PETALBURG_CITY", available),
        devon_goods_recovered,
        visited_rustboro,
        first_badge,
        petalburg_wally_scene,
        petalburg_woods_scene,
        devon_goods_returned,
        devon_goods_delivered,
        devon_goods_reported,
        devon_goods_stolen,
        rustboro_city_state,
        rusturf_tunnel_state,
        devon_corp_3f_state,
        devon_corp_3f_scene,
        roxanne_available,
    )


def reconcile_nuzlocke_started(
    facts: CampaignFacts,
    persisted: Fact[bool],
) -> Fact[bool]:
    """Return the current ROM-owned Nuzlocke boundary.

    ``persisted`` is retained for callers written against the pre-Pokédex
    boundary API.  Durable ``NuzlockeStarted`` history remains replayable for
    rules/audit purposes, but cannot activate, delay, or override the current
    save's Pokédex receipt fact.
    """
    del persisted
    return facts.pokedex_received


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
        default_factory=lambda: CampaignFacts(*(Fact.unavailable() for _ in CampaignFacts.__dataclass_fields__))
    )
    # This is an observation boundary, not remembered campaign progress.  It
    # lets the planner mount title/menu opening work while keeping an
    # uninitialized save block out of completion decisions.
    campaign_lifecycle: CampaignObservationLifecycle = CampaignObservationLifecycle.UNAVAILABLE

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

        observation_lifecycle = snapshot.campaign_observation.lifecycle
        # Preserve compatibility with older fixture constructors that set
        # ``available=True`` before the lifecycle field existed.
        if (
            observation_lifecycle is CampaignObservationLifecycle.UNAVAILABLE
            and snapshot.campaign_observation.available
        ):
            observation_lifecycle = CampaignObservationLifecycle.ACTIVE

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
            _campaign_facts_for_runtime(snapshot, inventory, observed),
            observation_lifecycle,
        )

    def has_badge(self, name: str) -> Fact[bool]:
        """Return whether the named badge flag is known to be set."""

        if not self.badges.is_known:
            return Fact(None, self.badges.status)
        return Fact.known(any(flag.name == name and flag.value for flag in self.badges.value or ()))

    def item_quantity(self, name: str) -> Fact[int]:
        """Return the observed quantity of an item across all pockets."""

        if not self.inventory.is_known:
            return Fact(None, self.inventory.status)
        pockets = self.inventory.value.items + self.inventory.value.poke_balls + self.inventory.value.key_items
        quantity = next((item.quantity for item in pockets if item.name == name), 0)
        return Fact.known(quantity)

    def key_item_quantity(self, name: str) -> Fact[int]:
        """Return the observed quantity of a named key item."""

        if not self.inventory.is_known:
            return Fact(None, self.inventory.status)
        quantity = next((item.quantity for item in self.inventory.value.key_items if item.name == name), 0)
        return Fact.known(quantity)

    def encounter_for(self, location: MapId) -> Fact[LocationEncounter]:
        """Return the reduced encounter record for a map location."""

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
        """Return the encounter record for the currently observed map."""

        if not self.raw_map.is_known:
            return Fact(None, self.raw_map.status)
        return self.encounter_for(self.raw_map.value)

    def pokemon_is_dead(self, identity) -> Fact[bool]:
        """Return whether an identity is known to be permanently dead."""

        if not self.dead_pokemon.is_known:
            return Fact(None, self.dead_pokemon.status)
        if identity is None:
            return Fact.unknown()
        return Fact.known(identity in self.dead_pokemon.value)

    def run_is_lost(self) -> Fact[bool]:
        """Return whether the reduced run status is a known loss."""

        if not self.run_status.is_known:
            return Fact(None, self.run_status.status)
        return Fact.known(self.run_status.value is RunStatus.LOST)

    def run_is_legal(self) -> Fact[bool]:
        """Return whether all currently reduced Nuzlocke rules are legal."""

        if not self.rules_legal.is_known:
            return Fact(None, self.rules_legal.status)
        if self.run_status.value is RunStatus.LOST:
            return Fact.known(False)
        return self.rules_legal


def _campaign_facts_for_runtime(
    snapshot: NuzlockeSnapshot,
    inventory: Fact[InventorySnapshot],
    observed: ObservedCampaignState | None,
) -> CampaignFacts:
    """Combine current-save facts with durable rules/history projections."""
    current = derive_campaign_facts(snapshot, inventory, Fact.unavailable())
    # Pokédex receipt is the ROM-owned default Nuzlocke boundary.  The legacy
    # NuzlockeStarted event remains replayable for old stores, but it cannot
    # create, delay, or override this current-save fact.
    return dataclass_replace(current, nuzlocke_started=current.pokedex_received)
