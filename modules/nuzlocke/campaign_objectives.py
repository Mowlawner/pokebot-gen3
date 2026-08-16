"""Pure, ordered campaign-objective selection.

Objectives describe what the campaign should accomplish.  They do not read
the emulator or execute tactical goals; a future integration layer can use
``execution_id`` and ``tactical_target`` after selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from modules.goals import Goal, ReachWarp, early_pokeball_goal, introductory_rival_goal
from modules.map_data import MapRSE

from .campaign_state import CampaignState, Fact, FactStatus, RunStatus


class ObjectiveStatus(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CampaignPredicate:
    """A named pure predicate over :class:`CampaignState`."""

    predicate_id: str
    description: str
    evaluator: Callable[[CampaignState], Fact[bool]]

    def evaluate(self, state: CampaignState) -> Fact[bool]:
        return self.evaluator(state)


@dataclass(frozen=True, slots=True)
class CampaignObjective:
    objective_id: str
    description: str
    prerequisites: tuple[CampaignPredicate, ...]
    completion: CampaignPredicate
    failure: CampaignPredicate | None = None
    execution_id: str | None = None
    tactical_target: Goal | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveSelection:
    objective: CampaignObjective | None
    status: ObjectiveStatus
    reason: str


def current_area_is(area: str) -> CampaignPredicate:
    return CampaignPredicate(
        f"current_area_is:{area}",
        f"current canonical area is {area}",
        lambda state: (
            Fact.known(state.canonical_area.value == area)
            if state.canonical_area.is_known
            else Fact(None, state.canonical_area.status)
        ),
    )


def has_item(item: str, minimum: int = 1) -> CampaignPredicate:
    def evaluate(state: CampaignState) -> Fact[bool]:
        quantity = state.item_quantity(item)
        if not quantity.is_known:
            return Fact(None, quantity.status)
        return Fact.known(quantity.value >= minimum)

    return CampaignPredicate(
        f"has_item:{item}:{minimum}",
        f"has at least {minimum} {item}",
        evaluate,
    )


def battle_completed_at(location: tuple[int, int], *, trainer: bool = True) -> CampaignPredicate:
    def evaluate(state: CampaignState) -> Fact[bool]:
        battle = state.last_completed_battle
        if not battle.is_known:
            return Fact(None, battle.status)
        if battle.value is None:
            return Fact.known(False)
        successful = battle.value.outcome.lower() in {"won", "caught"}
        return Fact.known(successful and battle.value.location == location and battle.value.is_trainer == trainer)

    return CampaignPredicate(
        f"battle_completed_at:{location}:{trainer}",
        f"completed a {'trainer' if trainer else 'wild'} battle at {location}",
        evaluate,
    )


def party_has_usable_pokemon() -> CampaignPredicate:
    def evaluate(state: CampaignState) -> Fact[bool]:
        if not state.party.is_known:
            return Fact(None, state.party.status)
        dead = state.dead_pokemon
        if not dead.is_known:
            return Fact(None, dead.status)
        usable = any(
            pokemon.current_hp > 0 and (pokemon.identity is None or pokemon.identity not in dead.value)
            for pokemon in state.party.value or ()
        )
        return Fact.known(usable)

    return CampaignPredicate("party_has_usable_pokemon", "party has a usable Pokémon", evaluate)


def encounter_available() -> CampaignPredicate:
    def evaluate(state: CampaignState) -> Fact[bool]:
        encounter = state.current_encounter
        if not encounter.is_known:
            return Fact(None, encounter.status)
        if encounter.value.status == "unknown":
            return Fact.unknown()
        return Fact.known(encounter.value.status == "none" and encounter.value.eligible)

    return CampaignPredicate("encounter_available", "current area has an unused encounter", evaluate)


def initial_emerald_campaign() -> tuple[CampaignObjective, ...]:
    """Return the deliberately small first campaign slice."""

    return (
        CampaignObjective(
            objective_id="reach_oldale",
            description="Reach Oldale Town",
            prerequisites=(),
            completion=current_area_is("OLDALE_TOWN"),
            execution_id="reach_oldale",
            tactical_target=ReachWarp(destination_map=MapRSE.OLDALE_TOWN.value),
        ),
        CampaignObjective(
            objective_id="confirm_early_pokeballs",
            description="Obtain or confirm the early-game Poké Balls",
            prerequisites=(),
            completion=has_item("Poke Ball"),
            execution_id="obtain_early_pokeballs",
            tactical_target=early_pokeball_goal(),
        ),
        CampaignObjective(
            objective_id="complete_intro_rival",
            description="Complete the introductory rival battle",
            prerequisites=(party_has_usable_pokemon(),),
            completion=battle_completed_at(MapRSE.ROUTE103.value, trainer=True),
            execution_id="intro_rival",
            tactical_target=introductory_rival_goal(),
        ),
    )


def select_campaign_objective(
    state: CampaignState,
    objectives: tuple[CampaignObjective, ...] | None = None,
) -> ObjectiveSelection:
    """Select the first non-complete objective in deterministic order."""

    if state.run_status.status is not FactStatus.KNOWN:
        return ObjectiveSelection(None, ObjectiveStatus.UNKNOWN, "run status is unavailable")
    if state.run_status.value is RunStatus.LOST:
        return ObjectiveSelection(None, ObjectiveStatus.FAILED, "run is lost")

    legal = state.run_is_legal()
    if legal.status is not FactStatus.KNOWN:
        return ObjectiveSelection(None, ObjectiveStatus.UNKNOWN, "run legality is unavailable")
    if not legal.value:
        return ObjectiveSelection(None, ObjectiveStatus.FAILED, "run is not legal")

    ordered = initial_emerald_campaign() if objectives is None else objectives
    for objective in ordered:
        if objective.failure is not None:
            failure = objective.failure.evaluate(state)
            if failure.status is not FactStatus.KNOWN:
                return ObjectiveSelection(
                    objective, ObjectiveStatus.UNKNOWN, f"failure state is {failure.status.value}"
                )
            if failure.value:
                return ObjectiveSelection(objective, ObjectiveStatus.FAILED, "objective failure predicate is true")

        completion = objective.completion.evaluate(state)
        if completion.status is not FactStatus.KNOWN:
            return ObjectiveSelection(objective, ObjectiveStatus.UNKNOWN, f"completion is {completion.status.value}")
        if completion.value:
            continue

        for prerequisite in objective.prerequisites:
            result = prerequisite.evaluate(state)
            if result.status is not FactStatus.KNOWN:
                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.UNKNOWN,
                    f"prerequisite {prerequisite.predicate_id} is {result.status.value}",
                )
            if not result.value:
                return ObjectiveSelection(
                    objective,
                    ObjectiveStatus.BLOCKED,
                    f"prerequisite {prerequisite.predicate_id} is false",
                )
        return ObjectiveSelection(objective, ObjectiveStatus.READY, "all prerequisites are satisfied")

    return ObjectiveSelection(None, ObjectiveStatus.COMPLETE, "all campaign objectives are complete")
