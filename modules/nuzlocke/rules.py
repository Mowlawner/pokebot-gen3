"""Deterministic, read-only Nuzlocke rules reduction.

This module consumes immutable observation events.  It does not inspect the
emulator, alter events, or persist campaign state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Any

from .events import (
    BattleEnded,
    BattleStarted,
    Event,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    StorageChanged,
    PokemonCaptured,
    PokemonFainted,
    WhiteoutOccurred,
    NuzlockeStarted,
)
from .identity import PokemonIdentity
from .rule_config import CampaignRuleId, CampaignRulesConfig
from .persistence import JsonEventStore, deserialize_event
from modules.pokemon import get_species_by_index, get_species_by_name

NO_ENCOUNTER = "none"
PENDING = "pending"
CAPTURED = "captured"
FAINTED = "fainted"
LOST = "lost_without_capture"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LocationEncounter:
    """Reduced first-encounter state for one map location."""

    location: tuple[int, int]
    status: str = NO_ENCOUNTER
    pokemon_identity: PokemonIdentity | None = None
    frame: int | None = None
    eligible: bool = True
    species: str | None = None


@dataclass(frozen=True, slots=True)
class RuleViolation:
    """A rule breach observed at a location and frame."""

    reason: str
    location: tuple[int, int] | None
    frame: int


@dataclass(frozen=True, slots=True)
class RuleAssessment:
    """A pure, rule-local view of reduced campaign legality."""

    rule_id: CampaignRuleId
    legal: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NuzlockeCampaignState:
    """Immutable legality state produced by the enabled Nuzlocke rules."""

    encounters: tuple[LocationEncounter, ...] = ()
    dead_pokemon: tuple[PokemonIdentity, ...] = ()
    unknown_faints: tuple[PokemonFainted, ...] = ()
    violations: tuple[RuleViolation, ...] = ()
    run_lost: bool = False
    last_event_sequence: int = 0
    # Species caught under an eligible encounter.  Keep the original spelling
    # for diagnostics, while comparisons use case-folded values.
    captured_species: tuple[str, ...] = ()
    # Branch metadata is needed for branched evolutionary families. The
    # species tuple remains the public compatibility view; this parallel tuple
    # records the branch known at capture time, e.g. ("wurmple", "cascoon").
    captured_evolution_branches: tuple[tuple[str, str], ...] = ()

    @property
    def legal(self) -> bool:
        """Return whether no rule violation has been reduced so far."""

        return not self.violations

    @property
    def first_encounters(self) -> tuple[LocationEncounter, ...]:
        """Return encounters eligible for the run's first-encounter rule."""

        return tuple(e for e in self.encounters if e.eligible)

    @property
    def unresolved_encounters(self) -> tuple[LocationEncounter, ...]:
        """Return eligible encounters whose outcomes are not settled."""

        return tuple(e for e in self.first_encounters if e.status in (PENDING, UNKNOWN))

    @property
    def alive_pokemon(self) -> tuple[PokemonIdentity, ...]:
        """Return captured first-encounter Pokémon not marked dead."""

        return tuple(
            e.pokemon_identity
            for e in self.first_encounters
            if e.status == CAPTURED and e.pokemon_identity is not None and e.pokemon_identity not in self.dead_pokemon
        )

    @property
    def caught_first_encounters(self) -> tuple[PokemonIdentity, ...]:
        """Return identities captured from eligible first encounters."""

        return tuple(
            e.pokemon_identity for e in self.first_encounters if e.status == CAPTURED and e.pokemon_identity is not None
        )

    def encounter_for(self, location: tuple[int, int]) -> LocationEncounter:
        """Return the first encounter at a location, or an explicit empty view."""
        return next((e for e in self.first_encounters if e.location == location), LocationEncounter(location))


class FaintingRule:
    """Reduce permanent fainting and whiteout events when enabled."""

    rule_id = CampaignRuleId.FAINTING

    def apply(
        self,
        state: NuzlockeCampaignState,
        event: Event,
        *,
        encounter_eligible: bool,
        active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]],
    ) -> NuzlockeCampaignState:
        """Reduce faint and whiteout events into permanent run state."""

        if isinstance(event, PokemonFainted):
            if event.identity is None:
                return replace(state, unknown_faints=state.unknown_faints + (event,))
            if event.identity not in state.dead_pokemon:
                return replace(state, dead_pokemon=state.dead_pokemon + (event.identity,))
        elif isinstance(event, WhiteoutOccurred):
            return replace(state, run_lost=True)
        return state

    def evaluate(self, state: NuzlockeCampaignState) -> RuleAssessment:
        """Assess whether the reduced run has suffered a whiteout."""

        return RuleAssessment(
            self.rule_id,
            not state.run_lost,
            ("whiteout occurred",) if state.run_lost else (),
        )

    def constrain(self, state: NuzlockeCampaignState, candidate):
        """Leave candidate selection unchanged for this observation rule."""

        return candidate


class OneEncounterPerAreaRule:
    """Reduce first-wild-encounter ownership without changing event history."""

    rule_id = CampaignRuleId.ONE_ENCOUNTER_PER_AREA

    def apply(
        self,
        state: NuzlockeCampaignState,
        event: Event,
        *,
        encounter_eligible: bool,
        active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]],
    ) -> NuzlockeCampaignState:
        """Reduce eligible wild battles into first-encounter outcomes."""

        if isinstance(event, BattleStarted):
            if not (encounter_eligible and event.is_wild and not event.is_trainer and event.location is not None):
                return state
            location = event.location
            existing = next((e for e in state.encounters if e.location == location), None)
            if existing is None:
                identity = event.opponent_pokemon_identities[0] if len(event.opponent_pokemon_identities) == 1 else None
                species = event.opponent_species[0] if len(event.opponent_species) == 1 else None
                state = replace(
                    state,
                    encounters=state.encounters
                    + (LocationEncounter(location, PENDING, identity, event.frame, True, species),),
                )
            elif existing.status not in (PENDING, UNKNOWN):
                identity = event.opponent_pokemon_identities[0] if len(event.opponent_pokemon_identities) == 1 else None
                # A repeat wild battle is still legal to fight.  The
                # one-encounter rule constrains capture eligibility, not the
                # player's ability to battle or escape incidental encounters.
                # Keep an explicit ineligible history record so later
                # capture handling cannot mistake it for a first encounter.
                state = replace(
                    state,
                    encounters=state.encounters + (LocationEncounter(location, UNKNOWN, identity, event.frame, False),),
                )
            active_wild[location] = event.opponent_pokemon_identities
        elif isinstance(event, PokemonCaptured):
            locations = ((event.location, ()),) if event.location is not None else tuple(active_wild.items())
            for location, opponent_ids in locations:
                if not opponent_ids or event.identity in opponent_ids:
                    current = next((item for item in state.encounters if item.location == location), None)
                    # Capture events from an incidental repeat must not
                    # mutate the already-resolved first encounter.  Only a
                    # pending eligible battle owns a capture outcome.
                    if current is not None and current.status == PENDING:
                        return self._resolve(state, location, CAPTURED, event.identity)
        elif isinstance(event, BattleEnded) and event.is_wild and not event.is_trainer and event.location is not None:
            current = next((e for e in state.encounters if e.location == event.location), None)
            if current is not None and current.status == PENDING:
                outcome = event.outcome.lower()
                status = (
                    UNKNOWN
                    if outcome in {"unknown", "incomplete", "in progress", "inprogress", ""}
                    else (FAINTED if outcome in {"fainted", "lost", "whiteout", "opponent fainted"} else LOST)
                )
                state = self._resolve(state, event.location, status, current.pokemon_identity)
            active_wild.pop(event.location, None)
        return state

    def evaluate(self, state: NuzlockeCampaignState) -> RuleAssessment:
        """Assess whether the encounter history contains a repeat encounter."""

        reasons = tuple(violation.reason for violation in state.violations)
        return RuleAssessment(self.rule_id, not reasons, reasons)

    def constrain(self, state: NuzlockeCampaignState, candidate):
        """Leave candidate selection unchanged for this observation rule."""

        return candidate

    @staticmethod
    def _resolve(
        state: NuzlockeCampaignState, location: tuple[int, int], status: str, identity: PokemonIdentity | None
    ) -> NuzlockeCampaignState:
        """Return state with the eligible encounter at ``location`` resolved."""

        return replace(
            state,
            encounters=tuple(
                (
                    replace(item, status=status, pokemon_identity=identity or item.pokemon_identity)
                    if item.location == location and item.eligible
                    else item
                )
                for item in state.encounters
            ),
        )


def _normalize_species(species: str | None) -> str | None:
    """Return a stable, case-insensitive species key."""

    if not isinstance(species, str):
        return None
    value = species.strip().casefold()
    return value or None


def _species_family(species: str | None) -> frozenset[str]:
    """Return normalized evolutionary-line members for a species name."""

    normalized = _normalize_species(species)
    if normalized is None:
        return frozenset()
    try:
        value = get_species_by_name(species.strip())
    except KeyError:
        # Live event producers normally preserve the ROM's title casing, but
        # older logs and lightweight integrations may provide lowercase
        # species names.  The species module's public name lookup is exact,
        # so resolve that compatibility form against the indexed catalogue.
        value = None
        for index in range(1000):
            try:
                candidate = get_species_by_index(index)
            except IndexError:
                break
            if _normalize_species(getattr(candidate, "name", None)) == normalized:
                value = candidate
                break
        if value is None:
            return frozenset({normalized})
    try:
        return frozenset(
            name
            for name in (_normalize_species(get_species_by_index(index).name) for index in value.family)
            if name is not None
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return frozenset({normalized})


def _wurmple_branch(species: str | None, identity: PokemonIdentity | None = None) -> str | None:
    """Return a Wurmple branch, preserving the personality-based split."""

    normalized = _normalize_species(species)
    if normalized == "wurmple" and identity is not None:
        # Pokemon.wurmple_evolution uses the upper 16 bits of the personality
        # value, modulo ten. Capture events retain that stable identity even
        # though they intentionally do not retain emulator-facing Pokémon data.
        return "silcoon" if ((identity.personality_value >> 16) & 0xFFFF) % 10 <= 4 else "cascoon"
    if normalized in {"silcoon", "beautifly"}:
        return "silcoon"
    if normalized in {"cascoon", "dustox"}:
        return "cascoon"
    return None


def _captured_branch(state: NuzlockeCampaignState, species: str) -> str | None:
    """Find persisted branch metadata for one captured species."""

    normalized = _normalize_species(species)
    if normalized is None:
        return None
    for captured_species, branch in state.captured_evolution_branches:
        if _normalize_species(captured_species) == normalized:
            return branch
    return _wurmple_branch(species)


def _species_clause_conflict(state: NuzlockeCampaignState, species: tuple[str, ...]) -> str | None:
    """Return the prior capture that makes a candidate species ineligible.

    Normal evolutionary families share one clause slot. Wurmple is the
    deliberate exception: its personality-selected branch blocks Wurmple and
    that branch's descendants, while leaving the alternate branch available.
    The common ancestor itself is always blocked once any Wurmple-family
    capture exists.
    """

    for candidate in species:
        candidate_name = _normalize_species(candidate)
        if candidate_name is None:
            continue
        candidate_family = _species_family(candidate)
        candidate_branch = _wurmple_branch(candidate)
        for captured in state.captured_species:
            captured_name = _normalize_species(captured)
            if captured_name is None:
                continue
            captured_family = _species_family(captured)
            if not candidate_family.intersection(captured_family):
                continue
            if candidate_name == captured_name:
                return captured
            if "wurmple" not in candidate_family or "wurmple" not in captured_family:
                return captured
            # Wurmple itself is the shared ancestor, regardless of the branch
            # selected by the individual that was caught.
            if candidate_name == "wurmple" or captured_name == "wurmple":
                captured_branch = _captured_branch(state, captured)
                if candidate_name == "wurmple" or captured_branch is None:
                    return captured
                if candidate_branch == captured_branch:
                    return captured
                continue
            captured_branch = _captured_branch(state, captured)
            if candidate_branch is None or captured_branch is None or candidate_branch == captured_branch:
                return captured
    return None


def species_clause_conflict_reason(state: NuzlockeCampaignState, species: tuple[str, ...]) -> str | None:
    """Return a user-facing Species Clause explanation, if one applies."""

    captured = _species_clause_conflict(state, species)
    return f"previous encounter was already obtained species/evolution line: {captured}" if captured else None


def _species_is_captured(state: NuzlockeCampaignState, species: tuple[str, ...]) -> bool:
    """Return whether any observed opponent species is already captured."""

    return _species_clause_conflict(state, species) is not None


class SpeciesClauseRule:
    """Skip eligible encounters for species already captured in the run."""

    rule_id = CampaignRuleId.SPECIES_CLAUSE

    def apply(
        self,
        state: NuzlockeCampaignState,
        event: Event,
        *,
        encounter_eligible: bool,
        active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]],
    ) -> NuzlockeCampaignState:
        """Record only species captured from a pending eligible encounter."""

        if not isinstance(event, PokemonCaptured):
            return state

        locations = (event.location,) if event.location is not None else tuple(active_wild)
        for location in locations:
            encounter = next(
                (item for item in state.encounters if item.location == location and item.eligible),
                None,
            )
            if encounter is None or encounter.status != PENDING:
                continue
            species = _normalize_species(event.species) or _normalize_species(encounter.species)
            if species is None:
                # Old capture records have no species and cannot safely
                # establish Species Clause history.
                return state
            if species in {_normalize_species(item) for item in state.captured_species}:
                return state
            stored = (
                event.species.strip() if isinstance(event.species, str) and event.species.strip() else encounter.species
            )
            branch = (
                event.evolution_branch
                if event.evolution_branch in {"silcoon", "cascoon"}
                else _wurmple_branch(species, event.identity)
            )
            branches = state.captured_evolution_branches
            if branch is not None and stored is not None:
                branches = branches + ((_normalize_species(stored) or stored, branch),)
            return replace(
                state,
                captured_species=state.captured_species + (stored or species,),
                captured_evolution_branches=branches,
            )
        return state

    def evaluate(self, state: NuzlockeCampaignState) -> RuleAssessment:
        """Species Clause is a legality constraint, not a violation detector."""

        return RuleAssessment(self.rule_id, True)

    def constrain(self, state: NuzlockeCampaignState, candidate):
        """Leave tactical candidate selection to the battle policy."""

        return candidate


class NuzlockeRulesProjection:
    """Reduce an ordered event stream into immutable Nuzlocke state."""

    def __init__(
        self,
        *,
        encounters_active: bool = True,
        rule_config: CampaignRulesConfig | None = None,
    ) -> None:
        """Create a reducer with the configured rule set and encounter gate."""

        self._state = NuzlockeCampaignState()
        self._seen: set[tuple[Any, ...]] = set()
        self._active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]] = {}
        self._encounters_active = encounters_active
        self._rule_config = rule_config or CampaignRulesConfig()
        self._rules = tuple(
            rule
            for rule in (SpeciesClauseRule(), OneEncounterPerAreaRule(), FaintingRule())
            if self._rule_config.is_enabled(rule.rule_id)
        )

    @property
    def rule_config(self) -> CampaignRulesConfig:
        """Return the immutable run configuration used by this reduction."""
        return self._rule_config

    @property
    def rule_assessments(self) -> tuple[RuleAssessment, ...]:
        """Return each enabled rule's current pure assessment."""

        return tuple(rule.evaluate(self._state) for rule in self._rules)

    def set_encounters_active(self, active: bool) -> None:
        """Enable the baseline rules boundary once Poké Balls exist."""
        self._encounters_active = self._encounters_active or active

    @property
    def state(self) -> NuzlockeCampaignState:
        """Return the current immutable reduced state."""

        return self._state

    def apply(
        self,
        event: Event,
        *,
        sequence: int | None = None,
        event_id: str | None = None,
        encounter_eligible: bool | None = None,
    ) -> None:
        """Apply one ordered event, ignoring only an already-seen event key."""

        if not isinstance(
            event,
            (
                BattleStarted,
                BattleEnded,
                PokemonCaptured,
                PokemonFainted,
                WhiteoutOccurred,
                GameStateChanged,
                MapChanged,
                PartyChanged,
                StorageChanged,
                NuzlockeStarted,
            ),
        ):
            raise TypeError(f"Unsupported Nuzlocke rules event: {type(event).__name__}")
        key = (event_id,) if event_id is not None else (type(event).__name__, event)
        if key in self._seen:
            return
        expected = self._state.last_event_sequence + 1
        actual = expected if sequence is None else sequence
        if actual != expected:
            raise ValueError(f"Event sequence out of order: expected {expected}, got {actual}")
        self._seen.add(key)
        state = self._state
        eligible_now = self._encounters_active if encounter_eligible is None else encounter_eligible
        if isinstance(event, BattleStarted) and event.encounter_eligible is not None:
            # A persisted BattleStarted owns the eligibility decision made at
            # its observation boundary.  This takes precedence over the
            # caller's current runtime state during replay.
            eligible_now = event.encounter_eligible
        if (
            isinstance(event, BattleStarted)
            and eligible_now
            and event.is_wild
            and not event.is_trainer
            and self._rule_config.is_enabled(CampaignRuleId.SPECIES_CLAUSE)
            and _species_is_captured(state, event.opponent_species)
        ):
            # Species Clause changes encounter ownership, not battle legality:
            # the duplicate may still be fought or fled from, but it cannot
            # claim the area's first encounter.
            eligible_now = False
        for rule in self._rules:
            state = rule.apply(
                state,
                event,
                encounter_eligible=eligible_now,
                active_wild=self._active_wild,
            )
        self._state = replace(state, last_event_sequence=actual)
        if isinstance(event, NuzlockeStarted):
            # Older persisted BattleStarted records predate the explicit
            # encounter_eligible field.  The durable campaign boundary is
            # enough to restore the historical rule activation point for
            # those records; newer records still use their own frame-local
            # eligibility decision above.
            self._encounters_active = True

    def apply_record(self, record: dict[str, Any]) -> None:
        """Validate and apply one persisted event-store record."""

        required = {"event_id", "session_id", "sequence", "frame", "type", "payload"}
        if set(record) != required:
            raise ValueError("Malformed event record")
        event = deserialize_event({"type": record["type"], "payload": record["payload"]})
        if record["frame"] != event.frame:
            raise ValueError("Event record frame does not match payload")
        self.apply(event, sequence=record["sequence"], event_id=record["event_id"])


def reduce_rules(events: Iterable[Event]) -> NuzlockeCampaignState:
    """Reduce an in-memory event iterable into Nuzlocke campaign state."""

    projection = NuzlockeRulesProjection()
    for event in events:
        projection.apply(event)
    return projection.state


def load_rules(event_store: JsonEventStore) -> NuzlockeCampaignState:
    """Replay all records from an event store through the rules projection."""

    projection = NuzlockeRulesProjection()
    for record in event_store.iter_records():
        projection.apply_record(record)
    return projection.state
