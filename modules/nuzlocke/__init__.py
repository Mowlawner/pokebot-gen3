"""Read-only normalized observations for future Nuzlocke domain code."""

from .snapshots import (
    BattleSnapshot,
    BattlePokemonSnapshot,
    ItemQuantity,
    InventorySnapshot,
    MoveSnapshot,
    NamedFlag,
    NamedVariable,
    CampaignObservationSnapshot,
    CampaignObservationLifecycle,
    NuzlockeSnapshot,
    PartyPokemonSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    PokemonSnapshot,
    StorageSnapshot,
    StoragePokemonSnapshot,
    get_nuzlocke_snapshot,
)
from .identity import PokemonIdentity
from .events import NuzlockeStarted, PokemonCaptured, PokemonStorageLocation, StorageChanged
from .runtime import NuzlockeRuntime
from .policy import EventStatistics, PersistenceClass, classify_event
from .diagnostics import BoundedEventTrace
from .persistence import (
    EventStoreCorruptionError,
    EventStoreError,
    JsonEventStore,
    deserialize_event,
    serialize_event,
)
from .projection import (
    CampaignProjection,
    ObservedBattle,
    ObservedCampaignState,
    ObservedEvent,
    ObservedPokemon,
    load_campaign_projection,
    reduce_events,
)
from .rules import (
    NuzlockeCampaignState,
    NuzlockeRulesProjection,
    FaintingRule,
    LocationEncounter,
    OneEncounterPerAreaRule,
    RuleAssessment,
    RuleViolation,
    load_rules,
    reduce_rules,
)
from .rule_config import CampaignRule, CampaignRuleId, CampaignRulesConfig
from .emerald_campaign_registry import (
    EmeraldBossDefinition,
    EmeraldBossProgression,
    EmeraldObjectiveDefinition,
    EmeraldCapabilityDefinition,
    active_emerald_boss,
    emerald_bosses,
    emerald_capability_definition,
    emerald_objectives,
)
from .level_cap import BattleEntryDecision, LevelCapAssessment, LevelCapRule, assess_level_cap, evaluate_battle_entry
from .preparation import (
    PreparationDecision,
    PreparationInput,
    PreparationStrategy,
    RiskTolerance,
    decide_preparation,
    discover_trainer_ids,
)
from .campaign_state import CampaignFacts, CampaignState, Fact, FactStatus, RunStatus, derive_campaign_facts
from .encounter_catalog import (
    EncounterOpportunity,
    available_encounter_methods,
    campaign_encounter_methods,
    encounter_opportunities,
    encounter_candidates_for_location,
)
from .capture_policy import (
    CapturePolicyContext,
    CapturePolicyWeights,
    CaptureRecommendation,
    EncounterCandidate,
    EncounterMethod,
    choose_encounter,
    score_encounter,
)
from .trainer_policy import TrainerPolicyInput, choose_trainer_mode
from .emerald_healing_catalog import (
    HealingSourceRSE,
    emerald_healing_sources,
    emerald_healing_sources_for_map,
    emerald_healing_source_for_destination,
)
from .campaign_simulation import CampaignCheckpoint, CampaignSimulation, simulate_checkpoint, simulate_checkpoints
from .fixture_state import (
    FixtureStateUnavailable,
    audit_fixture,
    load_campaign_state_from_fixture,
    load_offline_normalized_snapshot,
)
from .campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    party_fully_restored,
    heal_party_objective,
    ObjectiveSelection,
    ObjectiveStatus,
    initial_emerald_campaign,
    select_campaign_objective,
    available_campaign_tasks,
    select_available_campaign_task,
    encounter_task,
    RouteContext,
    RouteRelation,
    measure_route_context,
    campaign_task_diagnostics,
    EncounterClassification,
    EncounterRecommendation,
    EncounterEvaluationPolicy,
    EncounterEvaluation,
    evaluate_encounter_opportunity,
    progression_destination,
)
from .campaign_execution import (
    CampaignExecutionAdapter,
    CampaignExecutionResult,
    CampaignExecutionStatus,
    adapt_campaign_execution,
)
from .resource_policy import (
    EncounterPolicy,
    HealingResource,
    PartyResource,
    ReadinessImportance,
    ResourceDecision,
    ResourceObjective,
    ResourceSnapshot,
    RouteRecovery,
    assess_campaign_resources,
    assess_wild_encounter,
)


def __getattr__(name):
    """Load runtime modules lazily to avoid agent-control import cycles.

    ``agent_control`` imports a lightweight nuzlocke submodule during its own
    initialization. Eagerly importing the controller/resource runtime from
    this package initializer re-enters that partially initialized module.
    These exports remain available from ``modules.nuzlocke`` while avoiding
    import-order-dependent collection failures under pytest.
    """
    if name in {"CampaignController", "CampaignControllerState", "CampaignControllerStatus"}:
        from .campaign_controller import CampaignController, CampaignControllerState, CampaignControllerStatus

        return {
            "CampaignController": CampaignController,
            "CampaignControllerState": CampaignControllerState,
            "CampaignControllerStatus": CampaignControllerStatus,
        }[name]
    if name in {"HealingSource", "HealingSourceType", "discover_healing_source", "execute_heal_party"}:
        from .resource_runtime import HealingSource, HealingSourceType, discover_healing_source, execute_heal_party

        return {
            "HealingSource": HealingSource,
            "HealingSourceType": HealingSourceType,
            "discover_healing_source": discover_healing_source,
            "execute_heal_party": execute_heal_party,
        }[name]
    raise AttributeError(name)


__all__ = [
    "BattleSnapshot",
    "BattlePokemonSnapshot",
    "ItemQuantity",
    "InventorySnapshot",
    "MoveSnapshot",
    "NamedFlag",
    "NamedVariable",
    "CampaignObservationSnapshot",
    "CampaignObservationLifecycle",
    "NuzlockeSnapshot",
    "PartyPokemonSnapshot",
    "PlayerSnapshot",
    "ProgressionSnapshot",
    "PokemonSnapshot",
    "StorageSnapshot",
    "StoragePokemonSnapshot",
    "PokemonIdentity",
    "PokemonStorageLocation",
    "StorageChanged",
    "PokemonCaptured",
    "NuzlockeStarted",
    "get_nuzlocke_snapshot",
    "NuzlockeRuntime",
    "PersistenceClass",
    "EventStatistics",
    "classify_event",
    "BoundedEventTrace",
    "EventStoreError",
    "EventStoreCorruptionError",
    "JsonEventStore",
    "serialize_event",
    "deserialize_event",
    "CampaignProjection",
    "ObservedBattle",
    "ObservedCampaignState",
    "ObservedEvent",
    "ObservedPokemon",
    "load_campaign_projection",
    "reduce_events",
    "NuzlockeCampaignState",
    "NuzlockeRulesProjection",
    "CampaignRule",
    "CampaignRuleId",
    "CampaignRulesConfig",
    "FaintingRule",
    "OneEncounterPerAreaRule",
    "RuleAssessment",
    "LocationEncounter",
    "RuleViolation",
    "load_rules",
    "reduce_rules",
    "EmeraldBossDefinition",
    "EmeraldBossProgression",
    "active_emerald_boss",
    "emerald_bosses",
    "EmeraldObjectiveDefinition",
    "EmeraldCapabilityDefinition",
    "emerald_capability_definition",
    "emerald_objectives",
    "BattleEntryDecision",
    "LevelCapAssessment",
    "LevelCapRule",
    "assess_level_cap",
    "evaluate_battle_entry",
    "PreparationDecision",
    "PreparationInput",
    "PreparationStrategy",
    "RiskTolerance",
    "decide_preparation",
    "discover_trainer_ids",
    "CampaignState",
    "Fact",
    "FactStatus",
    "RunStatus",
    "CampaignFacts",
    "derive_campaign_facts",
    "EncounterOpportunity",
    "encounter_opportunities",
    "HealingSourceRSE",
    "emerald_healing_sources",
    "emerald_healing_sources_for_map",
    "emerald_healing_source_for_destination",
    "CampaignCheckpoint",
    "CampaignSimulation",
    "simulate_checkpoint",
    "simulate_checkpoints",
    "FixtureStateUnavailable",
    "load_campaign_state_from_fixture",
    "load_offline_normalized_snapshot",
    "audit_fixture",
    "EncounterClassification",
    "EncounterRecommendation",
    "EncounterEvaluationPolicy",
    "EncounterEvaluation",
    "evaluate_encounter_opportunity",
    "progression_destination",
    "CampaignObjective",
    "CampaignPredicate",
    "ObjectiveSelection",
    "ObjectiveStatus",
    "initial_emerald_campaign",
    "select_campaign_objective",
    "CampaignExecutionAdapter",
    "CampaignExecutionResult",
    "CampaignExecutionStatus",
    "adapt_campaign_execution",
    "CampaignController",
    "CampaignControllerState",
    "CampaignControllerStatus",
]
