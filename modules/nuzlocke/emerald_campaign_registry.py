"""Emerald-owned campaign milestone and boss metadata.

Generic campaign code must consume these definitions through this capability
module instead of interpreting Emerald event flags or leader ordering itself.
Only the currently implemented campaign slice is represented here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .campaign_state import CampaignFacts, CampaignState, Fact, FactStatus


@dataclass(frozen=True, slots=True)
class EmeraldBossDefinition:
    boss_id: str
    objective_id: str
    display_name: str
    level_cap: int
    completion_fact: str


@dataclass(frozen=True, slots=True)
class EmeraldBossProgression:
    active_boss: EmeraldBossDefinition | None
    status: FactStatus


@dataclass(frozen=True, slots=True)
class EmeraldObjectiveDefinition:
    objective_id: str
    completion_fact: str
    semantic_target: str | None = None
    availability_facts: tuple[str, ...] = ()


EMERALD_BOSSES: tuple[EmeraldBossDefinition, ...] = (
    EmeraldBossDefinition(
        "roxanne",
        "defeat_roxanne",
        "Roxanne",
        15,
        "first_badge_obtained",
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
        ("devon_goods_recovered", "petalburg_woods"),
        ("visited_rustboro", "rustboro"),
        ("first_badge_obtained", "rustboro_gym"),
    )
)


def emerald_bosses() -> tuple[EmeraldBossDefinition, ...]:
    """Return the ordered boss data covered by the current Emerald slice."""
    return EMERALD_BOSSES


def emerald_objectives() -> tuple[EmeraldObjectiveDefinition, ...]:
    return EMERALD_OBJECTIVES


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
