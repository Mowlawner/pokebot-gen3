"""Emerald-owned campaign milestone and boss metadata.

Generic campaign code must consume these definitions through this capability
module instead of interpreting Emerald event flags or leader ordering itself.
Only the currently implemented campaign slice is represented here.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.goals import (
    EncounterMode,
    Goal,
    GoalConstraints,
    NavigationGoal,
    SemanticTarget,
    TrainerMode,
)
from modules.map_data import MapRSE

from .campaign_state import CampaignFacts, CampaignState, Fact, FactStatus


@dataclass(frozen=True, slots=True)
class EmeraldBossDefinition:
    """ROM-specific metadata for one implemented Emerald boss objective."""

    boss_id: str
    objective_id: str
    display_name: str
    level_cap: int
    completion_fact: str
    preparation_level: int = 0


@dataclass(frozen=True, slots=True)
class EmeraldBossProgression:
    """Observed status of the next boss in the registered Emerald sequence."""

    active_boss: EmeraldBossDefinition | None
    status: FactStatus


@dataclass(frozen=True, slots=True)
class EmeraldObjectiveDefinition:
    """ROM-owned completion and availability facts for one objective."""

    objective_id: str
    completion_fact: str
    semantic_target: str | None = None
    availability_facts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmeraldCapabilityDefinition:
    """Registry-owned execution contract for an Emerald objective.

    The campaign selector remains generic: it chooses an objective ID.  This
    table is the ROM capability registry that tells the execution adapter
    which observation-driven executor owns that ID and, where useful, which
    semantic target should be used for readiness/navigation.
    """

    objective_id: str
    capability_id: str = "emerald_observation"
    semantic_target: SemanticTarget | None = None
    readiness_goal: Goal | None = None
    preparation_level: int | None = None


EMERALD_BOSSES: tuple[EmeraldBossDefinition, ...] = (
    EmeraldBossDefinition(
        "roxanne",
        "defeat_roxanne",
        "Roxanne",
        15,
        "first_badge_obtained",
        14,
    ),
)

EMERALD_OBJECTIVES: tuple[EmeraldObjectiveDefinition, ...] = tuple(
    EmeraldObjectiveDefinition(name, name, target)
    for name, target in (
        ("rival_met", "rival"),
        ("birch_rescued", "birch"),
        ("starter_obtained", "starter"),
        ("intro_rival_battle_complete", "route103_rival"),
        ("pokedex_received", "pokedex"),
        ("pokeballs_ready", "birch_bag"),
        ("visited_petalburg", "petalburg"),
        ("petalburg_wally_scene_complete", "petalburg_wally"),
        ("petalburg_woods_scene_complete", "petalburg_woods"),
        ("devon_goods_stolen", "rustboro_goods_stolen"),
        ("devon_goods_reported", "devon_goods_reported"),
        ("devon_goods_recovered", "rusturf_tunnel_goods"),
        ("devon_goods_returned", "rustboro_goods_returned"),
        ("devon_corp_3f_scene_complete", "devon_corp_3f"),
        ("visited_rustboro", "rustboro"),
        ("first_badge_obtained", "rustboro_gym"),
    )
)


EMERALD_CAPABILITY_DEFINITIONS: tuple[EmeraldCapabilityDefinition, ...] = (
    EmeraldCapabilityDefinition("set_text_speed"),
    EmeraldCapabilityDefinition("complete_new_game_setup"),
    EmeraldCapabilityDefinition("set_wall_clock"),
    EmeraldCapabilityDefinition("meet_rival"),
    EmeraldCapabilityDefinition(
        "rescue_birch",
        semantic_target=SemanticTarget.interaction(
            MapRSE.ROUTE101.value,
            interaction_id="Route101_EventScript_BirchsBag",
        ),
    ),
    EmeraldCapabilityDefinition("obtain_starter"),
    EmeraldCapabilityDefinition(
        "complete_intro_rival",
        semantic_target=SemanticTarget.interaction(
            MapRSE.ROUTE103.value,
            interaction_id="Route103_EventScript_Rival",
        ),
    ),
    EmeraldCapabilityDefinition(
        "receive_pokedex",
        semantic_target=SemanticTarget.map(MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value),
    ),
    EmeraldCapabilityDefinition(
        "restock_pokeballs",
        # The executor selects the nearest cataloged source from the current
        # map, so this capability intentionally has no fixed target.
    ),
    EmeraldCapabilityDefinition(
        "restock_recovery_items",
        # The executor selects the nearest observed Poké Mart and validates
        # which medicines that shop actually sells at execution time.
    ),
    EmeraldCapabilityDefinition(
        "reach_petalburg",
        semantic_target=SemanticTarget.map(MapRSE.PETALBURG_CITY.value),
    ),
    EmeraldCapabilityDefinition(
        "complete_petalburg_wally",
        semantic_target=SemanticTarget.interaction(
            MapRSE.PETALBURG_CITY_GYM.value,
            interaction_id="PetalburgCity_Gym_EventScript_Norman",
        ),
    ),
    EmeraldCapabilityDefinition(
        "complete_petalburg_woods",
        semantic_target=SemanticTarget.interaction(
            MapRSE.PETALBURG_WOODS.value,
            interaction_id="devon_goods_researcher",
        ),
    ),
    EmeraldCapabilityDefinition(
        "recover_devon_goods",
        semantic_target=SemanticTarget.interaction(
            MapRSE.RUSTURF_TUNNEL.value,
            interaction_id="rusturf_tunnel_goods",
        ),
    ),
    EmeraldCapabilityDefinition(
        "complete_rustboro_goods_stolen",
        semantic_target=SemanticTarget.interaction(
            MapRSE.RUSTBORO_CITY.value,
            interaction_id="rustboro_goods_stolen",
        ),
    ),
    EmeraldCapabilityDefinition(
        "report_devon_goods",
        semantic_target=SemanticTarget.interaction(
            MapRSE.RUSTBORO_CITY.value,
            interaction_id="rustboro_goods_report",
        ),
    ),
    EmeraldCapabilityDefinition(
        "return_devon_goods",
        semantic_target=SemanticTarget.interaction(
            MapRSE.RUSTBORO_CITY.value,
            interaction_id="rustboro_return_devon_goods",
        ),
    ),
    EmeraldCapabilityDefinition(
        "meet_mr_stone",
        semantic_target=SemanticTarget.map(MapRSE.RUSTBORO_CITY_DEVON_CORP_3F.value),
    ),
    EmeraldCapabilityDefinition(
        "reach_rustboro",
        semantic_target=SemanticTarget.map(MapRSE.RUSTBORO_CITY.value),
    ),
    EmeraldCapabilityDefinition(
        "prepare_roxanne",
        capability_id="emerald_preparation",
        semantic_target=SemanticTarget.map(MapRSE.ROUTE116.value),
        preparation_level=14,
    ),
    EmeraldCapabilityDefinition(
        "defeat_roxanne",
        semantic_target=SemanticTarget.interaction(
            MapRSE.RUSTBORO_CITY_GYM.value,
            interaction_id="roxanne",
        ),
        # Keep the gym destination available to the readiness analyzer. The
        # interaction executor still resolves Roxanne from live perception.
        readiness_goal=NavigationGoal(
            SemanticTarget.interaction(MapRSE.RUSTBORO_CITY_GYM.value, interaction_id="roxanne"),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            encounter_mode=EncounterMode.AVOID,
        ),
    ),
)


def emerald_bosses() -> tuple[EmeraldBossDefinition, ...]:
    """Return the ordered boss data covered by the current Emerald slice."""
    return EMERALD_BOSSES


def emerald_objectives() -> tuple[EmeraldObjectiveDefinition, ...]:
    """Return the registered Emerald campaign objective definitions."""

    return EMERALD_OBJECTIVES


def emerald_capability_definition(objective_id: str) -> EmeraldCapabilityDefinition | None:
    """Return the immutable execution definition for an objective ID."""
    return next((item for item in EMERALD_CAPABILITY_DEFINITIONS if item.objective_id == objective_id), None)


def evaluate_emerald_fact(state: CampaignState, name: str) -> Fact[bool]:
    """Resolve a registered Emerald fact without generic flag-name parsing."""
    registered = {item.completion_fact for item in EMERALD_OBJECTIVES}
    if name not in registered and name not in {
        "text_speed_fast",
        "new_game_setup_complete",
        "wall_clock_set",
        "nuzlocke_started",
    }:
        return state.campaign_facts[name]
    return state.campaign_facts[name]


def active_emerald_boss(facts: CampaignFacts) -> EmeraldBossProgression:
    """Return the next undefeated implemented boss without guessing facts."""
    for boss in EMERALD_BOSSES:
        completed: Fact[bool] = facts[boss.completion_fact]
        if completed.status is not FactStatus.KNOWN:
            return EmeraldBossProgression(None, completed.status)
        if not completed.value:
            return EmeraldBossProgression(boss, FactStatus.KNOWN)
    return EmeraldBossProgression(None, FactStatus.KNOWN)
