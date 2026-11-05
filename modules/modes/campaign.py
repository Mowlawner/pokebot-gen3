"""Autonomous execution of the small ordered Emerald campaign."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import time
from typing import Generator
import traceback

from modules.context import context
from modules.agent_control import navigation_background_search_active, notify_battle_ended
from modules.map_formatting import format_location
from modules.console import (
    diagnostic_print,
    print_navigation_cancellation,
    print_recovery_destinations,
    print_recovery_handoff,
    print_recovery_request,
)
from modules.nuzlocke.campaign_controller import CampaignController, runtime_campaign_state
from modules.nuzlocke.campaign_objectives import campaign_planning_signature, plan_campaign
from modules.nuzlocke.campaign_state import FactStatus
from modules.nuzlocke.campaign_status import CampaignStatus

from ._interface import BattleAction, BotMode
from modules.nuzlocke.resource_policy import (
    DuplicateEncounterPolicy,
    PartyResource,
    ResourceObservationStatus,
    ResourceDecision,
    ResourceSnapshot,
    RouteRecovery,
    WildBattleIntent,
    WildBattlePolicyInput,
    assess_wild_encounter,
    choose_wild_battle_intent,
)
from modules.nuzlocke.resource_runtime import (
    execute_campaign_recovery,
    execute_planned_recovery,
    execute_whiteout_continuation,
    RunLostError,
    observe_resource_snapshot,
    observe_route_recovery,
    plan_route_recovery,
    prepare_route_recovery,
)
from modules.nuzlocke.emerald_healing_catalog import emerald_healing_sources_for_map
from modules.nuzlocke.emerald_healing_catalog import emerald_healing_source_for_destination
from modules.nuzlocke.level_cap import evaluate_battle_entry
from modules.modes._interface import BotModeError
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    CampaignReadinessPolicy,
    ReadinessObservationScheduler,
    build_progression_readiness_diagnostic,
)
from modules.battle_readiness import (
    BattleReadiness,
    BattleReadinessStatus,
    established_search_probability,
)
from modules.nuzlocke.snapshots import get_nuzlocke_snapshot
from modules.nuzlocke.identity import PokemonIdentity
from modules.overworld import (
    perceive_overworld,
    OverworldObservationResult,
    publish_shared_overworld_observation,
    prewarm_static_map_observation,
    static_trainer_observations,
    trainer_hazard_locations,
)
from modules.interaction_state import InteractionPhase, observe_interaction
from modules.player import get_player_avatar, player_avatar_is_controllable
from modules.memory import GameState, get_game_state
from modules.pokemon_party import get_party
from modules.pokemon import get_species_by_index
from modules.goals import (
    ActivateTrigger,
    EncounterMode,
    EngageTrainer,
    GoalConstraints,
    NavigationGoal,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    TrainerMode,
)
from modules.navigation import (
    NavigationWorld,
    RouteCostAnalyzer,
    RouteAnalysis,
    TrainerAvoidability,
    TrainerRouteRisk,
)
from modules.stutter_trace import suppress_background_instrumentation
from modules.world_navigation import get_world_map_graph
from modules.modes.util.map import pokemon_center_candidates
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeCaptureStrategy,
    NuzlockeLevelBalancingBattleStrategy,
    NuzlockeRunAwayStrategy,
    RoxanneBattleStrategy,
)
from modules.battle_strategies._interface import BattleStrategy
from modules.battle_state import BattleOutcome, BattleType, get_battle_state, get_battle_type
from modules.modes.util.lead_rotation import ensure_campaign_field_lead
from modules.nuzlocke.field_lead import build_campaign_field_lead_context
from modules.trainer_checkpoints import (
    KNOWN_TRAINER_CHECKPOINTS,
    merge_trainer_checkpoints,
    trainer_checkpoint_identity,
    trainer_checkpoint_for_battle,
    trainer_checkpoint_for_source,
    trainer_checkpoint_for_id,
    trainer_checkpoints_for_map,
    trainer_checkpoints_for_overworld,
)
from modules.nuzlocke.trainer_preparation import build_exp_budget, projected_trainer_experience
from modules.trainer_readiness import assess_live_trainer_checkpoint, live_storage_readiness_members


_RECOVERY_ROUTE_ANALYSIS_MAX_EXPANSIONS = 4000
# Recovery is a safety action.  It may spend longer composing a known Center
# route than speculative route analysis, but it still remains cancellable and
# bounded by the background planner.
_RECOVERY_ROUTE_SEARCH_TIMEOUT_SECONDS = 15.0
# Advisory route comparisons are preference work. After a timeout or worker
# cancellation, wait briefly before retrying the same envelope so the frame
# loop cannot submit the same expensive search repeatedly.
_ADVISORY_ROUTE_RETRY_COOLDOWN_SECONDS = 1.0


def _campaign_goal_owns_trainer_exposure(goal) -> bool:
    """Return whether the tactical loop stages a concrete trainer battle."""

    return isinstance(goal, EngageTrainer) or (
        isinstance(goal, NavigationGoal)
        and isinstance(getattr(goal, "target", None), EngageTrainer)
    )


def _campaign_trainer_id(goal) -> str | None:
    target = goal.target if isinstance(goal, NavigationGoal) else goal
    return target.trainer_id if isinstance(target, EngageTrainer) else None


def _encounter_stage_readiness_key():
    """Return the semantic encounter-stage facts that invalidate readiness."""

    evidence = getattr(context, "campaign_encounter_stage", None)
    if evidence is None:
        return None
    return (
        getattr(evidence, "current_map", None),
        getattr(evidence, "next_map", None),
        getattr(evidence, "world_signature", None),
        getattr(evidence, "readiness", None),
        bool(getattr(evidence, "blocked", False)),
        tuple(getattr(evidence, "known_hazard_ids", ()) or ()),
        tuple(getattr(evidence, "unresolved_hazard_ids", ()) or ()),
        tuple(
            getattr(checkpoint, "checkpoint_id", None)
            for checkpoint in tuple(getattr(evidence, "known_hazard_checkpoints", ()) or ())
        ),
    )


def _runtime_campaign_facts_or_none():
    """Return live campaign facts when the runtime is observable.

    Route hazard analysis also runs during unit and transition boundaries where
    the emulator may not be attached yet.  Missing live facts mean that no
    checkpoint has been authoritatively cleared; they must not make route
    analysis fail before it can publish the registered hazard envelope.
    """

    try:
        return getattr(runtime_campaign_state(), "campaign_facts", None)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _checkpoint_is_cleared(campaign_facts, checkpoint) -> bool:
    """Return whether a checkpoint has an authoritative clearance fact."""

    clearance_fact = getattr(checkpoint, "clearance_fact", None)
    if campaign_facts is None or not isinstance(clearance_fact, str):
        return False
    clearance = getattr(campaign_facts, clearance_fact, None)
    return clearance is not None and clearance.status is FactStatus.KNOWN and clearance.value is True


def _pc_trainer_readiness_candidates():
    """Return normalized PC candidates without making storage mandatory."""

    try:
        from modules.pokemon_storage import get_pokemon_storage

        storage = get_pokemon_storage()
        return live_storage_readiness_members(storage) if storage is not None else ()
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError, IndexError):
        return ()


def _known_trainer_checkpoint_on_route(overworld, readiness_goal, route_analysis):
    """Return the first known undefeated trainer exposed by the selected route.

    A map traversal is often represented by a ``SemanticTarget`` rather than
    an ``EngageTrainer`` goal.  In that case the navigation plan is the
    authoritative indication that the selected route exposes a trainer. Keep
    the lookup limited to route locations and registered checkpoints so an
    unrelated trainer on the same map cannot trigger preparation.
    """

    if overworld is None or route_analysis is None:
        return None
    campaign_facts = _runtime_campaign_facts_or_none()
    route = getattr(route_analysis, "normal_route", None)
    if route is None:
        return None
    map_id = getattr(getattr(overworld, "map_id", None), "value", getattr(overworld, "map_id", None))
    route_locations = set()
    for action in getattr(route, "actions", ()):
        for location in (getattr(action, "source", None), getattr(action, "destination", None)):
            if not isinstance(location, tuple) or len(location) != 2:
                continue
            location_map = getattr(location[0], "value", location[0])
            if location_map == map_id:
                route_locations.add((location_map, location[1]))
    if not route_locations and not getattr(route, "forced_trainer_exposure", False):
        return None
    objects_by_id = {
        getattr(obj, "trainer_id", None): obj
        for obj in getattr(overworld, "objects", ())
        if getattr(obj, "trainer_id", None) is not None
    }
    constraints = getattr(readiness_goal, "constraints", None)
    trainer_mode = getattr(getattr(constraints, "trainer_mode", None), "name", None)
    triggers = tuple(getattr(overworld, "triggers", ()))
    for trigger in triggers:
        trainer_id = getattr(trigger, "affordance_id", None)
        if not trainer_id:
            continue
        hazard_locations = tuple(
            location
            for location in (
                *getattr(trigger, "hazard_locations", ()),
                *getattr(trigger, "activation_locations", ()),
                *getattr(trigger, "engagement_locations", ()),
            )
            if isinstance(location, tuple) and len(location) == 2
        )
        route_hazards = {
            (getattr(location[0], "value", location[0]), location[1])
            for location in hazard_locations
        }
        hazard_intersection = route_hazards.intersection(route_locations)
        if not hazard_intersection:
            # IGNORE explicitly permits the route to approach the trainer.
            # If sight-line geometry is incomplete, the trigger's concrete
            # locations still have to intersect the selected path before it
            # can be considered a route trainer.
            trigger_locations = {
                (getattr(location[0], "value", location[0]), location[1])
                for location in getattr(trigger, "locations", ())
                if isinstance(location, tuple) and len(location) == 2
            }
            if trainer_mode != "IGNORE" or not trigger_locations.intersection(route_locations):
                continue

        # Static trainer triggers are deliberately published before their
        # runtime object event exists.  A trigger with non-empty hazard
        # geometry is evidence that the ROM template is currently active;
        # requiring a matching runtime object drops precisely the trainer
        # that a camera-scoped object table has not spawned yet.  A runtime
        # object still wins when its defeated state is authoritative.
        trainer = objects_by_id.get(trainer_id)
        if trainer is not None:
            active = getattr(trainer, "trainer_defeated", None) is False
        else:
            active = bool(route_hazards)
        if not active:
            continue

        # A route that intersects a live trainer hazard is an exposure even
        # when the navigator did not set its fallback marker. The marker is
        # still useful telemetry, but action/hazard intersection is the
        # stronger fact for campaign preparation.
        checkpoint = trainer_checkpoint_for_source(
            trainer_id,
            script_symbol=_trainer_script_for_id(overworld, trainer_id),
        )
        if checkpoint is not None and not _checkpoint_is_cleared(campaign_facts, checkpoint):
            return trainer_id

    # ``GoalAwareNavigator`` marks a route as forced only after it has
    # exhausted every trainer-avoiding path and rerun the search with trainer
    # hazards ignored.  In that case the selected route can legitimately end
    # beside the trainer rather than on one of the exact sight-line tiles
    # captured above (for example when a map connection or an object load
    # boundary compresses the route actions). Resolve the registered
    # mandatory checkpoint by map and stable affordance identity as a final
    # sight-line-aware fallback.
    if getattr(route, "forced_trainer_exposure", False):
        for checkpoint in KNOWN_TRAINER_CHECKPOINTS:
            if (
                not getattr(checkpoint, "mandatory", False)
                or checkpoint.map_id != map_id
            ):
                continue
            if _checkpoint_is_cleared(campaign_facts, checkpoint):
                continue
            trainer_id = checkpoint.trainer_affordance_id
            trainer = objects_by_id.get(trainer_id)
            if trainer is not None and getattr(trainer, "trainer_defeated", None) is not False:
                continue
            matching_triggers = tuple(
                trigger
                for trigger in triggers
                if getattr(trigger, "affordance_id", None) == trainer_id
            )
            if trainer is None and not matching_triggers:
                continue
            if trainer_checkpoint_for_source(
                trainer_id,
                script_symbol=_trainer_script_for_id(overworld, trainer_id),
            ) is not None:
                return trainer_id
    return None


def _static_route_trainer_facts(overworld, route_analysis) -> tuple:
    """Resolve future-map trainer facts whose static sight lines meet a route."""

    route = getattr(route_analysis, "normal_route", None)
    if route is None:
        return ()
    current_map = getattr(getattr(overworld, "map_id", None), "value", getattr(overworld, "map_id", None))
    route_locations = set()
    route_maps = set()
    for action in getattr(route, "actions", ()):
        for location in (getattr(action, "source", None), getattr(action, "destination", None)):
            if not isinstance(location, tuple) or len(location) != 2:
                continue
            map_id = getattr(location[0], "value", location[0])
            route_maps.add(map_id)
            route_locations.add((map_id, location[1]))
    for map_id in getattr(route, "maps", ()) or ():
        route_maps.add(getattr(map_id, "value", map_id))
    route_maps.discard(current_map)
    if not route_maps or not route_locations:
        return ()

    facts = []
    for map_id in sorted(route_maps, key=repr):
        try:
            # Resolve the checkpoint facts from the same static object table
            # used by perception. This also keeps the ROM trainer catalog and
            # the route hazard identity on one stable map/local-object ID.
            map_facts = {
                fact.trainer_affordance_id: fact
                for fact in trainer_checkpoints_for_map(map_id)
            }
            from modules.map import get_map_metadata

            metadata = get_map_metadata(map_id)
            tiles = prewarm_static_map_observation(map_id)
            trainers = static_trainer_observations(map_id, metadata, ())
            for trainer in trainers:
                trainer_id = getattr(trainer, "trainer_id", None)
                checkpoint = map_facts.get(trainer_id)
                if checkpoint is None:
                    continue
                hazards = trainer_hazard_locations(trainer, tiles, trainers)
                if hazards.intersection(route_locations):
                    facts.append(checkpoint)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    return tuple(
        sorted(
            {trainer_checkpoint_identity(fact): fact for fact in facts}.values(),
            key=lambda fact: (getattr(fact, "map_id", ()), getattr(fact, "checkpoint_id", "")),
        )
    )


def _trainer_checkpoint_keys(checkpoint) -> set[tuple]:
    """Return aliases that identify one logical trainer across data sources."""

    if checkpoint is None:
        return set()
    map_id = getattr(getattr(checkpoint, "map_id", None), "value", getattr(checkpoint, "map_id", None))
    keys = set()
    for field in ("checkpoint_id", "trainer_affordance_id"):
        value = getattr(checkpoint, field, None)
        if value:
            keys.add((field, value))
    rom_id = getattr(checkpoint, "rom_trainer_id", None)
    if rom_id is not None:
        keys.add(("rom_trainer_id", int(rom_id)))
    display_name = str(getattr(checkpoint, "display_name", "") or "").strip().casefold()
    if map_id is not None and display_name:
        keys.add(("display_name", map_id, display_name))
    return keys


def _trainer_checkpoint_is_defeated(checkpoint) -> bool:
    defeated = set(getattr(context, "campaign_defeated_trainer_ids", ()) or ())
    return bool(defeated and _trainer_checkpoint_keys(checkpoint).intersection(defeated))


def _trainer_checkpoints_match(left, right) -> bool:
    """Return whether two checkpoint records identify the same trainer.

    The campaign route and the active battle use different identifiers.  The
    route fact is keyed by map/local object, while the battle record is keyed
    by the ROM trainer ID.  Preserve the records separately so defeating one
    logical trainer prunes every frontier alias, but only merge records when
    their map, ROM ID, or display identity proves that relationship.
    """

    if left is None or right is None:
        return False
    if left is right:
        return True
    left_checkpoint_id = getattr(left, "checkpoint_id", None)
    right_checkpoint_id = getattr(right, "checkpoint_id", None)
    if left_checkpoint_id is not None and left_checkpoint_id == right_checkpoint_id:
        return True
    left_rom_id = getattr(left, "rom_trainer_id", None)
    right_rom_id = getattr(right, "rom_trainer_id", None)
    if left_rom_id is not None and right_rom_id is not None:
        try:
            if int(left_rom_id) == int(right_rom_id):
                return True
        except (TypeError, ValueError):
            pass
    left_map = getattr(getattr(left, "map_id", None), "value", getattr(left, "map_id", None))
    right_map = getattr(getattr(right, "map_id", None), "value", getattr(right, "map_id", None))
    left_name = str(getattr(left, "display_name", "") or "").strip().casefold()
    right_name = str(getattr(right, "display_name", "") or "").strip().casefold()
    return left_map is not None and left_map == right_map and bool(left_name) and left_name == right_name


def _deduplicate_trainer_checkpoints(checkpoints) -> tuple:
    """Retain distinct identity records in stable order."""

    result = []
    seen = set()
    for checkpoint in checkpoints:
        if checkpoint is None:
            continue
        identity = (
            getattr(checkpoint, "checkpoint_id", None),
            getattr(checkpoint, "trainer_affordance_id", None),
            getattr(checkpoint, "rom_trainer_id", None),
            getattr(getattr(checkpoint, "map_id", None), "value", getattr(checkpoint, "map_id", None)),
            str(getattr(checkpoint, "display_name", "") or "").casefold(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(checkpoint)
    return tuple(result)


def _trainer_checkpoints_for_battle_start() -> tuple:
    """Resolve the route and live-ROM identities for the active trainer."""

    planned = getattr(context, "campaign_trainer_checkpoint", None)
    live = None
    try:
        battle_state = get_battle_state()
        if getattr(battle_state, "is_trainer_battle", False):
            live = trainer_checkpoint_for_battle(
                battle_state,
                getattr(getattr(battle_state, "opponent", None), "active_battler", None),
            )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        live = None

    active = [checkpoint for checkpoint in (planned, live) if checkpoint is not None]
    if not active:
        return ()

    # The readiness frontier can contain a richer static/ROM record than the
    # singular planned checkpoint. Retain those records when they match either
    # active identity, so a later battle teardown can clear every alias.
    for fact in tuple(getattr(context, "campaign_trainer_facts", ()) or ()):
        if any(_trainer_checkpoints_match(fact, checkpoint) for checkpoint in active):
            active.append(fact)
    return _deduplicate_trainer_checkpoints(active)


def _remember_defeated_trainer(checkpoint) -> None:
    """Record a won trainer battle and immediately prune its frontier facts."""

    if checkpoint is None:
        return
    defeated = getattr(context, "campaign_defeated_trainer_ids", None)
    if not isinstance(defeated, set):
        defeated = set(defeated or ())
        context.campaign_defeated_trainer_ids = defeated
    defeated.update(_trainer_checkpoint_keys(checkpoint))
    context.campaign_trainer_facts = tuple(
        fact
        for fact in tuple(getattr(context, "campaign_trainer_facts", ()) or ())
        if not _trainer_checkpoint_is_defeated(fact)
    )
    if _trainer_checkpoint_is_defeated(getattr(context, "campaign_trainer_checkpoint", None)):
        context.campaign_trainer_checkpoint = None
    diagnostic_print(
        lambda: (
            "CAMPAIGN_TRAINER_DEFEATED: "
            f"checkpoint={getattr(checkpoint, 'checkpoint_id', None)!r} "
            f"trainer={getattr(checkpoint, 'trainer_affordance_id', None)!r} "
            "action='pruned_frontier_fact'"
        ),
        trace=True,
        prefix="CAMPAIGN_TRAINER_DEFEATED",
    )


def _mandatory_trainer_on_outbound_map(overworld, readiness_goal, route_analysis):
    """Return a live mandatory checkpoint when route evidence is incomplete.

    Route analysis is the preferred source because it knows whether the
    selected path actually crosses a trainer's sight line.  A background
    analysis can still be pending, however, and static trainer geometry can
    be absent for one observation while the ROM map and registered checkpoint
    already prove that an outbound map contains a mandatory trainer.  Keep
    this fallback narrow: it applies only to a map traversal, only to
    registered mandatory checkpoints on the current or destination map, and
    only while the checkpoint is still undefeated.
    """

    if overworld is None:
        return None
    if not _is_map_traversal_readiness_goal(readiness_goal):
        return None
    target = getattr(readiness_goal, "target", readiness_goal)
    target_map = getattr(target, "target_map", None)
    if target_map is None:
        target_map = getattr(target, "map_id", None)
    if target_map is None:
        target_location = getattr(target, "location", None)
        if isinstance(target_location, tuple) and len(target_location) == 2:
            target_map = target_location[0]
    if target_map is None:
        target_map = getattr(target, "destination_map", None)
    if target_map is None:
        destination = getattr(target, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            target_map = destination[0]
    current_map = getattr(overworld, "map_id", None)
    current_map = getattr(current_map, "value", current_map)
    target_map = getattr(target_map, "value", target_map)
    if target_map is None:
        return None

    objects_by_id = {
        getattr(obj, "trainer_id", None): obj
        for obj in getattr(overworld, "objects", ())
        if getattr(obj, "trainer_id", None) is not None
    }
    campaign_facts = _runtime_campaign_facts_or_none()
    # A trainer can live on an intermediate map between the current map and
    # an interaction destination.  The old fallback only compared the
    # checkpoint with the two endpoints, so Route 102/Calvin was invisible
    # while the selected destination was Petalburg Gym.  Use the ROM-backed
    # world corridor when the detailed route analysis is unavailable or has
    # no executable AVOID route.  This is a map reachability fact, not an
    # inference from the observed numeric state of the save.
    corridor_maps = {current_map, target_map}
    normal_route = getattr(route_analysis, "normal_route", None)
    for map_id in getattr(normal_route, "maps", ()) or ():
        corridor_maps.add(getattr(map_id, "value", map_id))
    if len(corridor_maps) <= 2 and current_map != target_map:
        try:
            corridor_maps.update(
                getattr(map_id, "value", map_id)
                for map_id in get_world_map_graph().route(current_map, target_map).maps
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            # An unavailable static graph is unknown evidence. Do not turn it
            # into an open route or invent a trainer owner from map names.
            pass
    diagnostic_print(
        lambda: (
            "CAMPAIGN_TRAINER_CORRIDOR: "
            f"current_map={current_map!r} target_map={target_map!r} "
            f"maps={tuple(sorted(corridor_maps, key=repr))!r} "
            f"route_analysis_present={route_analysis is not None!r} "
            f"normal_route_present={normal_route is not None!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_TRAINER_CORRIDOR",
    )
    for checkpoint in KNOWN_TRAINER_CHECKPOINTS:
        if (
            not getattr(checkpoint, "mandatory", False)
            or getattr(getattr(checkpoint, "map_id", None), "value", getattr(checkpoint, "map_id", None))
            not in corridor_maps
        ):
            continue
        if _checkpoint_is_cleared(campaign_facts, checkpoint):
            continue
        trainer_id = checkpoint.trainer_affordance_id
        trainer = objects_by_id.get(trainer_id)
        if trainer is not None and getattr(trainer, "trainer_defeated", None) is True:
            continue
        matching_triggers = tuple(
            trigger
            for trigger in getattr(overworld, "triggers", ())
            if getattr(trigger, "affordance_id", None) == trainer_id
        )
        if any(getattr(trigger, "trainer_defeated", None) is True for trigger in matching_triggers):
            # Static trainer observations remain available after the runtime
            # object leaves the camera. Preserve their authoritative battle
            # flag so a completed checkpoint is not promoted again merely
            # because its clearance milestone (for example Petalburg) is
            # still ahead of the current encounter objective.
            continue
        # The registered checkpoint is the authoritative static fact for a
        # mandatory trainer. Object and trigger observations are
        # camera/runtime data and can both be absent while the player is
        # crossing the map. Treating that absence as proof that Calvin is not
        # present is exactly how the campaign entered Route 102 with
        # ``Trainer facts none (unknown)``.
        diagnostic_print(
            lambda: (
                "CAMPAIGN_MANDATORY_TRAINER_CORRIDOR: "
                f"trainer={trainer_id!r} checkpoint={getattr(checkpoint, 'checkpoint_id', None)!r} "
                f"map={getattr(checkpoint, 'map_id', None)!r} "
                "reason='verified mandatory checkpoint lies on ROM corridor'"
            ),
            trace=True,
            prefix="CAMPAIGN_MANDATORY_TRAINER_CORRIDOR",
        )
        return trainer_id
    return None


def _trainer_risk_selected_for_engagement(route_analysis, readiness_policy):
    """Choose the strongest route risk promoted by campaign policy."""

    if route_analysis is None:
        return None
    risks = tuple(getattr(route_analysis, "trainer_risks", ()) or ())
    candidates = tuple(
        risk
        for risk in risks
        if getattr(risk, "confidence", "known") == "known"
        and getattr(readiness_policy, "trainer_risk_requires_engagement", lambda _: False)(risk)
    )
    if not candidates:
        return None
    order = {
        "avoidable": 0,
        "costly": 1,
        "practically_unavoidable": 2,
        "unavoidable": 3,
        "imminent": 4,
    }
    selected = max(
        candidates,
        key=lambda risk: (
            order.get(getattr(getattr(risk, "avoidability", None), "value", None), -1),
            -(getattr(risk, "detour", None) or 0),
            str(getattr(risk, "trainer_id", "")),
        ),
    )
    return getattr(selected, "trainer_id", None)


def _encounter_stage_route_risks(overworld, objective=None) -> tuple[TrainerRouteRisk, ...]:
    """Translate a blocked encounter stage into route-risk evidence.

    The encounter capability is often the first owner to test a map
    connection. Its AVOID/IGNORE comparison can therefore be newer and more
    precise than the readiness worker's route envelope. Known hazards are
    promotable; unresolved hazards remain an explicit unknown safety fact.
    """

    if not getattr(objective, "objective_id", "").startswith("obtain_encounter:"):
        return ()
    evidence = getattr(context, "campaign_encounter_stage", None)
    if evidence is None:
        return ()
    evidence_reason = str(getattr(evidence, "reason", ""))
    if not getattr(evidence, "blocked", False) and not evidence_reason.startswith(
        "forced_trainer_exposure:"
    ):
        return ()
    current_map = getattr(getattr(overworld, "map_id", None), "value", getattr(overworld, "map_id", None))
    evidence_map = getattr(getattr(evidence, "current_map", None), "value", getattr(evidence, "current_map", None))
    if current_map != evidence_map:
        return ()
    # A blocked safe probe is not, by itself, proof that the known trainers
    # form a permanent route cut. Moving trainers can temporarily close the
    # grass corridor, and the next stable observation may expose a safe exit.
    # Only the affirmative forced-exposure result is strong enough to claim a
    # known unavoidable checkpoint. Keep transient failures diagnostic so the
    # encounter capability can retry under a changed trainer position.
    forced_exposure = evidence_reason.startswith("forced_trainer_exposure:")
    risks = []
    for trainer_id in tuple(getattr(evidence, "known_hazard_ids", ()) or ()):
        risks.append(
            TrainerRouteRisk(
                trainer_id=trainer_id,
                avoidability=(
                    TrainerAvoidability.UNAVOIDABLE
                    if forced_exposure
                    else TrainerAvoidability.UNKNOWN
                ),
                exposed_on_normal_route=True,
                confidence="known" if forced_exposure else "unknown",
                reason=(
                    "encounter-stage proved forced exposure across the selected exits"
                    if forced_exposure
                    else "encounter-stage safe search is temporarily unavailable; trainer movement may change the corridor"
                ),
            )
        )
    for trainer_id in tuple(getattr(evidence, "unresolved_hazard_ids", ()) or ()):
        risks.append(
            TrainerRouteRisk(
                trainer_id=trainer_id,
                avoidability=TrainerAvoidability.UNKNOWN,
                exposed_on_normal_route=True,
                confidence="unknown",
                reason="encounter-stage route is trainer constrained but the ROM roster is unresolved",
            )
        )
    return tuple(risks)


def _encounter_stage_checkpoint_for_id(overworld, trainer_id, evidence=None):
    """Reuse the checkpoint resolved by the encounter-stage probe.

    Encounter staging can resolve a runtime/static script just before the
    capability returns to the campaign controller. The next readiness pass
    may observe a camera-scoped world without that script, so resolving the
    same stable affordance again can incorrectly turn a known trainer into an
    unknown one. Stage evidence is the authoritative handoff for this
    boundary; source resolution remains the fallback for older evidence and
    ordinary route analysis.
    """

    evidence = evidence if evidence is not None else getattr(context, "campaign_encounter_stage", None)
    for checkpoint in tuple(getattr(evidence, "known_hazard_checkpoints", ()) or ()):
        if getattr(checkpoint, "trainer_affordance_id", None) == trainer_id:
            return checkpoint
    return trainer_checkpoint_for_source(
        trainer_id,
        script_symbol=_trainer_script_for_id(overworld, trainer_id),
    )


def _trainer_script_for_id(overworld, trainer_id: str | None) -> str | None:
    """Return the ROM script symbol paired with one stable trainer affordance."""

    if not trainer_id or overworld is None:
        return None
    for obj in getattr(overworld, "objects", ()):
        if getattr(obj, "trainer_id", None) == trainer_id:
            return getattr(obj, "script", None)
    for trigger in getattr(overworld, "triggers", ()):
        if getattr(trigger, "affordance_id", None) != trainer_id:
            continue
        script = getattr(trigger, "script_symbol", None)
        if script:
            return script
        trigger_id = getattr(trigger, "trigger_id", "")
        if isinstance(trigger_id, str) and trigger_id.startswith("static_object:"):
            parts = trigger_id.split(":", 2)
            return parts[2] if len(parts) == 3 else None
    return None


def _merge_route_trainer_facts(overworld, visible_facts, route_analysis, additional_facts=()):
    """Publish ROM-backed facts for trainer hazards found by route analysis.

    Perception only exposes runtime objects and static triggers in the current
    camera scope. The route analyzer can still identify a trainer hazard at
    the same readiness boundary, so its stable affordance IDs are an
    additional source of facts. Unresolved IDs remain absent; the snapshot
    must not invent a roster merely to look complete.
    """

    route = getattr(route_analysis, "normal_route", None)
    current_map = getattr(getattr(overworld, "map_id", None), "value", getattr(overworld, "map_id", None))

    def normalized_location(location):
        if not isinstance(location, tuple) or len(location) != 2:
            return None
        return getattr(location[0], "value", location[0]), location[1]

    route_locations = set()
    route_maps = set()
    if route is not None:
        for action in getattr(route, "actions", ()):
            for location in (getattr(action, "source", None), getattr(action, "destination", None)):
                normalized = normalized_location(location)
                if normalized is not None:
                    route_locations.add(normalized)
                    route_maps.add(normalized[0])
        route_maps.update(getattr(map_id, "value", map_id) for map_id in getattr(route, "maps", ()) or ())

    # A route analysis is the authority for future trainer facts. A visible
    # trainer is useful to the frontier only when its geometry intersects the
    # selected route; otherwise a whole gym or branch can look executable.
    route_trainer_ids = {
        getattr(risk, "trainer_id", None)
        for risk in tuple(getattr(route_analysis, "trainer_risks", ()) or ())
        if getattr(risk, "trainer_id", None)
    }
    route_trainer_ids.update(
        getattr(fact, "trainer_affordance_id", None)
        for fact in tuple(additional_facts or ())
        if getattr(fact, "trainer_affordance_id", None)
    )
    if route is not None and route_locations:
        for trigger in getattr(overworld, "triggers", ()):
            trainer_id = getattr(trigger, "affordance_id", None)
            if not trainer_id:
                continue
            geometry = {
                normalized_location(location)
                for location in (
                    *getattr(trigger, "hazard_locations", ()),
                    *getattr(trigger, "engagement_locations", ()),
                    *getattr(trigger, "activation_locations", ()),
                    *getattr(trigger, "locations", ()),
                )
            }
            if geometry.intersection(route_locations):
                route_trainer_ids.add(trainer_id)

    def route_relevant(checkpoint) -> bool:
        if _trainer_checkpoint_is_defeated(checkpoint):
            return False
        if route is None:
            # No complete route exists during a map/script boundary. Keep the
            # display empty until the next route envelope is available; an
            # explicit trainer owner is republished by
            # _remember_campaign_trainer_fact().
            return current_map is None
        return getattr(checkpoint, "trainer_affordance_id", None) in route_trainer_ids

    facts_by_identity = {}
    for fact in tuple(visible_facts or ()):
        if getattr(fact, "checkpoint_id", None) is None or not route_relevant(fact):
            continue
        facts_by_identity[trainer_checkpoint_identity(fact)] = fact
    for risk in tuple(getattr(route_analysis, "trainer_risks", ()) or ()):
        trainer_id = getattr(risk, "trainer_id", None)
        if not trainer_id:
            continue
        checkpoint = trainer_checkpoint_for_source(
            trainer_id,
            script_symbol=_trainer_script_for_id(overworld, trainer_id),
        )
        if checkpoint is not None:
            if _trainer_checkpoint_is_defeated(checkpoint):
                continue
            identity = trainer_checkpoint_identity(checkpoint)
            facts_by_identity[identity] = merge_trainer_checkpoints(
                facts_by_identity.get(identity), checkpoint
            )
    for checkpoint in tuple(additional_facts or ()):
        if _trainer_checkpoint_is_defeated(checkpoint):
            continue
        identity = trainer_checkpoint_identity(checkpoint)
        facts_by_identity[identity] = merge_trainer_checkpoints(
            facts_by_identity.get(identity), checkpoint
        )
    return tuple(
        sorted(
            facts_by_identity.values(),
            key=lambda fact: (
                getattr(fact, "map_id", ()),
                getattr(fact, "checkpoint_id", ""),
            ),
        )
    )


def _remember_campaign_trainer_fact(checkpoint) -> tuple:
    """Publish one resolved trainer fact across camera/readiness boundaries."""

    if checkpoint is None:
        return tuple(getattr(context, "campaign_trainer_facts", ()) or ())
    existing = tuple(getattr(context, "campaign_trainer_facts", ()) or ())
    merged = {
        trainer_checkpoint_identity(fact): fact
        for fact in existing
        if getattr(fact, "checkpoint_id", None) is not None
        and not _trainer_checkpoint_is_defeated(fact)
    }
    if getattr(checkpoint, "checkpoint_id", None) is not None and not _trainer_checkpoint_is_defeated(checkpoint):
        identity = trainer_checkpoint_identity(checkpoint)
        merged[identity] = merge_trainer_checkpoints(merged.get(identity), checkpoint)
    result = tuple(
        sorted(
            merged.values(),
            key=lambda fact: (
                getattr(fact, "map_id", ()),
                getattr(fact, "checkpoint_id", ""),
            ),
        )
    )
    context.campaign_trainer_facts = result
    return result


def _is_map_traversal_readiness_goal(goal) -> bool:
    """Return whether readiness should inspect the route for trainer hazards."""

    target = getattr(goal, "target", goal)
    return (
        getattr(getattr(target, "kind", None), "name", None) in {"MAP", "LOCATION", "INTERACTION"}
        or isinstance(target, (ReachLocation, ReachWarp))
    )


def _objective_owns_scripted_trainer_battle(objective) -> bool:
    """Return whether the objective's capability owns a story battle scene."""

    return bool(
        getattr(getattr(objective, "resource_policy", None), "mandatory_battle", False)
    )


class _RoutePlanningBudget:
    """Cooperative wall-clock budget shared by a background route job."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.deadline = time.monotonic() + timeout_seconds
        self.cancelled = False
        self.cancel_reason: str | None = None

    def cancel(self, reason: str = "background route search was superseded") -> None:
        self.cancelled = True
        self.cancel_reason = reason

    def check(self) -> None:
        if self.cancelled:
            raise TimeoutError(self.cancel_reason or "background route search was superseded")
        if time.monotonic() >= self.deadline:
            reason = f"background route search exceeded {self.timeout_seconds:g} second budget"
            self.cancel(reason)
            raise TimeoutError(reason)


class _SingleFlightBackgroundPlanner:
    """Run one prioritized, cancellable route job without blocking frames.

    Route searches are Python object-graph walks.  Running two of them at once
    does not provide useful parallelism under the GIL, but it does make both
    searches compete with the emulator and frame loop.  A critical request
    still supersedes speculative work; the existing worker observes the
    cancellation token and the executor then runs the critical request.
    """

    # Route calculation runs off the frame thread. Give semantic map-by-map
    # composition enough time to finish while retaining a hard cooperative
    # deadline so a stale request cannot occupy the single worker forever.
    DEFAULT_TIMEOUT_SECONDS = 8.0

    def __init__(self):
        # Keep route analysis single-flight at the process level.  This is a
        # scheduling decision, not a correctness limit: ``poll`` cancels a
        # lower-priority job before submitting a higher-priority one.  One
        # worker prevents speculative recovery/trainer analysis from
        # monopolizing the GIL alongside the emulator callback thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nuzbot-route")
        self._jobs = {}
        self._completed = {}
        # A replacement request is retained only until the canceled worker
        # releases the executor slot.  Submitting it immediately would put it
        # behind that worker in ThreadPoolExecutor's queue, which is how a
        # frame-bound analysis could accumulate stale searches behind one
        # recovery calculation.
        self._deferred = {}

    @classmethod
    def _job_label(cls, key) -> str:
        domain = cls._key_domain(key)
        digest = cls._key_digest(key)
        if domain == "recovery" and isinstance(key, tuple) and len(key) > 1:
            request_key = key[1]
            if isinstance(request_key, tuple) and request_key:
                return f"{domain} {format_location(request_key[0])} key={digest}"
        return f"{domain} key={digest}"

    def _emit_cancellation(
        self,
        *,
        key,
        phase: str,
        reason: str,
        replacement_key=None,
        changed=(),
        duration_ms: float | None = None,
    ) -> None:
        print_navigation_cancellation(
            phase=phase,
            reason=reason,
            changed=changed,
            duration_ms=duration_ms,
            request_label=self._job_label(key),
            replacement_label=(self._job_label(replacement_key) if replacement_key is not None else None),
            worker="campaign_route",
        )

    def _cancel_job(self, key, job, *, reason: str, replacement_key=None, phase: str = "requested") -> None:
        job["budget"].cancel(reason)
        if not job.get("cancellation_reported", False):
            self._emit_cancellation(
                key=key,
                phase=phase,
                reason=reason,
                replacement_key=replacement_key,
                changed=("priority",) if replacement_key is not None else (),
                duration_ms=(time.monotonic() - job["started_at"]) * 1000.0,
            )
            job["cancellation_reported"] = True

    def _run(self, operation, budget, key):
        # Navigation's trace/profiler hooks refer to the mutable frame trace
        # and process-wide timing state. Suppress those hooks only on this
        # worker; the route calculation itself remains unchanged.
        with suppress_background_instrumentation():
            budget.check()
            try:
                accepts_budget = bool(inspect.signature(operation).parameters)
            except (TypeError, ValueError):
                accepts_budget = False
            try:
                return operation(budget) if accepts_budget else operation()
            except TimeoutError as error:
                if budget.cancelled and not self._job_was_reported(key):
                    self._emit_cancellation(
                        key=key,
                        phase="terminal",
                        reason=str(error),
                        duration_ms=None,
                    )
                raise

    def _job_was_reported(self, key) -> bool:
        job = self._jobs.get(key)
        return bool(job and job.get("cancellation_reported", False))

    @staticmethod
    def _key_digest(key) -> str:
        """Return a compact stable identifier for route-job diagnostics."""

        return hashlib.sha1(repr(key).encode("utf-8", errors="replace")).hexdigest()[:12]

    @classmethod
    def _key_domain(cls, key) -> str:
        domain = cls._job_domain(key)
        return domain or "unknown"

    def poll(self, key, operation, *, priority: int = 1, timeout_seconds: float | None = None):
        """Return ``pending``/``complete``/``error`` for the requested job."""

        cached = self._completed.get(key)
        if cached is not None:
            value, error = cached
            return "error" if error is not None else "complete", value, error

        job = self._jobs.get(key)
        if job is not None:
            if job["future"].done():
                return self._harvest(key, job)
            if time.monotonic() >= job["budget"].deadline:
                reason = f"background route search exceeded {job['budget'].timeout_seconds:g} second budget"
                if self._key_domain(key) == "recovery":
                    # Recovery planning can already have a valid incumbent
                    # when a later Center candidate exceeds the wall-clock
                    # budget. Let the worker observe cancellation and return
                    # that partial RouteRecovery instead of discarding the
                    # whole candidate batch here.
                    if not job.get("timeout_requested", False):
                        self._cancel_job(key, job, reason=reason, phase="requested")
                        job["timeout_requested"] = True
                        diagnostic_print(
                            lambda: (
                                "BACKGROUND_ROUTE_JOB: "
                                "event='timeout' "
                                f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                                f"elapsed_ms={(time.monotonic() - job['started_at']) * 1000.0:.1f} "
                                "action='harvest_partial_recovery'"
                            ),
                            trace=True,
                            prefix="BACKGROUND_ROUTE_JOB",
                        )
                    if job["future"].cancel():
                        # A queued job has no incumbent to preserve. Retain
                        # the historical terminal error behavior for this
                        # case rather than resubmitting the canceled work.
                        error = TimeoutError(reason)
                        self._jobs.pop(key, None)
                        self._completed[key] = (None, error)
                        self._emit_cancellation(
                            key=key,
                            phase="terminal",
                            reason=str(error),
                            duration_ms=(time.monotonic() - job["started_at"]) * 1000.0,
                        )
                        return "error", None, error
                    return "pending", None, None

                self._cancel_job(key, job, reason=reason, phase="terminal")
                error = TimeoutError(reason)
                # A running thread cannot be force-killed safely. Retain it
                # until it observes cancellation, but make the timeout a
                # terminal result for this request immediately.
                self._completed[key] = (None, error)
                if job["future"].cancel():
                    self._jobs.pop(key, None)
                diagnostic_print(
                    lambda: (
                        "BACKGROUND_ROUTE_JOB: "
                        "event='timeout' "
                        f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                        f"elapsed_ms={(time.monotonic() - job['started_at']) * 1000.0:.1f} "
                        f"active_jobs={len(self._jobs)!r} completed_jobs={len(self._completed)!r}"
                    ),
                    trace=True,
                    prefix="BACKGROUND_ROUTE_JOB",
                )
                return "error", None, error
            return "pending", None, None

        # Keep one logical worker owner at a time. A completed future that has
        # not been harvested still owns the result boundary: letting another
        # domain consume the slot would make the owner lose its completion
        # event. A higher-priority request may harvest that completed result,
        # or cancel a running lower-priority request, but it never queues work
        # behind the canceled future.
        for other_key, other in tuple(self._jobs.items()):
            if other["future"].done():
                current_domain = self._key_domain(key)
                other_domain = self._key_domain(other_key)
                domain_owner_must_harvest = (
                    other_domain is not None
                    and current_domain != other_domain
                    and priority >= other["priority"]
                )
                if not domain_owner_must_harvest:
                    self._harvest(other_key, other)
                else:
                    return "pending", None, None

        active_jobs = tuple(self._jobs.items())
        if active_jobs:
            higher_priority_job = False
            for other_key, other in active_jobs:
                if other["priority"] <= priority:
                    # Equal or higher-priority work owns the single route
                    # worker. Do not enqueue a second request behind it.
                    return "pending", None, None
                reason = "higher-priority campaign route request superseded the active request"
                self._cancel_job(
                    other_key,
                    other,
                    reason=reason,
                    replacement_key=key,
                )
                higher_priority_job = True
                if other["future"].cancel():
                    self._jobs.pop(other_key, None)
            if higher_priority_job and self._jobs:
                self._deferred[key] = (operation, priority, timeout_seconds)
                diagnostic_print(
                    lambda: (
                        "BACKGROUND_ROUTE_JOB: "
                        "event='deferred' reason='waiting_for_superseded_worker' "
                        f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                        f"priority={priority!r} active_jobs={len(self._jobs)!r}"
                    ),
                    trace=True,
                    prefix="BACKGROUND_ROUTE_JOB",
                )
                return "pending", None, None

        deferred = self._deferred.pop(key, None)
        if deferred is not None:
            operation, priority, timeout_seconds = deferred
        effective_timeout = self.DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        budget = _RoutePlanningBudget(effective_timeout)
        future = self._executor.submit(self._run, operation, budget, key)
        self._jobs[key] = {
            "future": future,
            "budget": budget,
            "priority": priority,
            "started_at": time.monotonic(),
            "cancellation_reported": False,
        }
        diagnostic_print(
            lambda: (
                "BACKGROUND_ROUTE_JOB: "
                "event='submitted' "
                f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                f"priority={priority!r} timeout_seconds={effective_timeout!r} "
                f"active_jobs={len(self._jobs)!r} completed_jobs={len(self._completed)!r}"
            ),
            trace=True,
            prefix="BACKGROUND_ROUTE_JOB",
        )
        return "pending", None, None

    def _harvest(self, key, job):
        elapsed_ms = (time.monotonic() - job["started_at"]) * 1000.0
        try:
            value = job["future"].result()
            error = None
        except Exception as caught:
            value = None
            error = caught
        self._jobs.pop(key, None)
        if error is not None and job.get("timeout_requested", False) and not job.get(
            "terminal_timeout_reported", False
        ):
            self._emit_cancellation(
                key=key,
                phase="terminal",
                reason=str(error),
                duration_ms=elapsed_ms,
            )
            job["terminal_timeout_reported"] = True
        # A discarded running job may finish after its request has been
        # superseded. Do not make that stale result look like a valid cache
        # hit if a caller happens to poll the old key during cleanup.
        if not job.get("discarded", False):
            self._completed[key] = (value, error)
        diagnostic_print(
            lambda: (
                "BACKGROUND_ROUTE_JOB: "
                f"event={'error' if error is not None else 'complete'!r} "
                f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                f"elapsed_ms={elapsed_ms:.1f} exception_type={type(error).__name__ if error else None!r} "
                f"exception={error!r} traceback={traceback.format_exc() if error is not None else None!r} "
                f"partial_result={bool(getattr(value, 'calculation_timed_out', False))!r} "
                f"active_jobs={len(self._jobs)!r} completed_jobs={len(self._completed)!r}"
            ),
            trace=True,
            prefix="BACKGROUND_ROUTE_JOB",
        )
        return "error" if error is not None else "complete", value, error

    def discard(self, key) -> None:
        """Forget a completed/retired request before an explicit retry."""

        self._completed.pop(key, None)
        self._deferred.pop(key, None)
        job = self._jobs.get(key)
        if job is not None:
            self._cancel_job(
                key,
                job,
                reason="campaign route request discarded by caller",
            )
            if job["future"].done():
                self._jobs.pop(key, None)
            elif job["future"].cancel():
                # A queued job has not started and can be removed from the
                # executor immediately. Leaving it queued lets stale route
                # analyses run one after another behind the current request.
                self._jobs.pop(key, None)
            else:
                # A running thread cannot be force-killed safely. Keep its
                # logical slot until it observes cancellation so a timeout
                # retry cannot submit a duplicate search while the stale
                # worker is still consuming CPU. Remove it as soon as the
                # future finishes; otherwise repeated frame-boundary
                # invalidations accumulate completed stale jobs forever.
                job["discarded"] = True

                def cleanup(future, *, key=key):
                    current = self._jobs.get(key)
                    if current is not None and current.get("future") is future and current.get("discarded"):
                        self._jobs.pop(key, None)

                job["future"].add_done_callback(cleanup)

    def is_pending(self, key) -> bool:
        """Return whether a request still has a running worker."""

        job = self._jobs.get(key)
        return job is not None and not job["future"].done()

    def elapsed_seconds(self, key) -> float | None:
        """Return elapsed worker time for a pending diagnostic, if present."""

        job = self._jobs.get(key)
        if job is None:
            return None
        return max(0.0, time.monotonic() - job["started_at"])

    @staticmethod
    def _job_domain(key):
        if isinstance(key, tuple) and key and key[0] in {"recovery", "analysis"}:
            return key[0]
        return None

    def close(self) -> None:
        """Stop accepting route jobs when an embedding discards the mode."""

        self._executor.shutdown(wait=False, cancel_futures=True)


def _overworld_position_is_coherent(overworld) -> bool:
    """Return whether the observed avatar position belongs to the map model.

    During a map connection Emerald can publish the destination map header
    before it publishes a valid destination coordinate (for example ``y ==
    -1``).  That observation is useful to the transition executor, but it is
    not safe input for campaign readiness or recovery route selection.
    """
    tile_at = getattr(overworld, "tile_at", None)
    if not callable(tile_at):
        # Lightweight test/dry-run observations may not expose topology. Keep
        # their historical behavior and let the normal readiness checks apply.
        return True
    coordinates = getattr(overworld, "player_coordinates", None)
    if not isinstance(coordinates, tuple) or len(coordinates) != 2:
        return False
    try:
        return tile_at(coordinates) is not None
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False


class CampaignProgressionMode(BotMode):
    """Bot mode that delegates each frame to observation-driven campaign planning."""

    @staticmethod
    def name() -> str:
        """Return the user-facing mode name."""

        return "Campaign Progression"

    @staticmethod
    def is_selectable() -> bool:
        """Return whether the active ROM supports Emerald campaign execution."""

        return context.rom is not None and context.rom.is_emerald

    def owns_evolution_policy(self) -> bool:
        """Let campaign strategy decide evolution independently of the profile."""

        return True

    def on_pokemon_evolving_after_battle(self, pokemon, party_index: int) -> bool:
        """Allow useful evolutions while preserving pre-evolution-only moves.

        The ROM presents this callback after the level-up move window. If the
        pre-evolution has a later level-up move that the target species never
        learns, the campaign delays the evolution and gets another chance at
        the next level. Otherwise the stronger species is preferred.
        """

        species = getattr(pokemon, "species", None)
        level = getattr(pokemon, "level", None)
        evolutions = tuple(
            evolution
            for evolution in getattr(species, "evolutions", ())
            if str(getattr(evolution, "method", "")).startswith("level")
        )
        if not evolutions or not isinstance(level, int):
            return True

        try:
            target_species = tuple(get_species_by_index(evolution.target_species_index) for evolution in evolutions)
        except (KeyError, IndexError, TypeError, ValueError):
            return True

        target_moves = {
            entry.move.name
            for target in target_species
            for entry in getattr(getattr(target, "learnset", None), "level_up", ())
        }
        banned_moves = set(getattr(getattr(getattr(context, "config", None), "battle", None), "banned_moves", ()))
        protected_move = next(
            (
                entry.move
                for entry in getattr(getattr(species, "learnset", None), "level_up", ())
                if entry.level > level
                and entry.move.name not in target_moves
                and entry.move.name not in banned_moves
            ),
            None,
        )
        if protected_move is not None:
            context.message = (
                f"Delaying {pokemon.name}'s evolution to preserve {protected_move.name}, "
                "which its evolved form does not learn."
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_EVOLUTION_DECISION: "
                    f"allowed=False party_index={party_index} pokemon={pokemon.name!r} "
                    f"species={species.name!r} move={protected_move.name!r} "
                    f"targets={[target.name for target in target_species]!r}"
                ),
                trace=True,
            )
            return False

        diagnostic_print(
            lambda: (
                "CAMPAIGN_EVOLUTION_DECISION: "
                f"allowed=True party_index={party_index} pokemon={pokemon.name!r} "
                f"species={species.name!r} targets={[target.name for target in target_species]!r}"
            ),
            trace=True,
        )
        return True

    def on_whiteout(self) -> bool:
        """Resume from a recoverable whiteout using a living PC Pokémon."""

        self._whiteout_continuation_pending = True
        context.message = "Whiteout recovered; replacing the fainted party from the PC."
        return True

    def __init__(self):
        """Create the campaign controller and readiness scheduler."""

        self._readiness_policy = CampaignReadinessPolicy()
        self._readiness_scheduler = ReadinessObservationScheduler(
            self._readiness_input,
            self._cheap_readiness_context,
            max_age_ticks=15,
            transient_context=self._readiness_transient_context,
        )
        self.controller = CampaignController(
            runtime_campaign_state,
            selector=self._select_campaign,
            readiness_provider=self._readiness_scheduler.observe,
            # CampaignPlan owns recovery selection.  There is deliberately no
            # fallback here that can reopen the healing catalog or select a
            # different Center after the plan has been composed.
            recovery_factory=execute_campaign_recovery,
            readiness_policy=self._readiness_policy,
        )
        self._readiness_evaluated = False
        self._campaign_selection_key = None
        self._campaign_selection = None
        self._campaign_boundary_context_seen = False
        self._last_campaign_boundary_context = None
        self._active_battle_wild: bool | None = None
        # Battle teardown clears the transient opponent projection before this
        # mode receives the outcome. Keep every route/ROM alias latched at the
        # battle-start boundary so a confirmed win can prune the frontier.
        self._active_trainer_checkpoints: tuple = ()
        # Whiteout is not necessarily a failed run. When the runtime observes
        # a living PC roster, the listener leaves the player at the Center and
        # this mode replaces the fainted party before campaign planning resumes.
        self._whiteout_continuation_pending = False
        # The battle listener can hand control back before the snapshot
        # observer publishes its faint event. Retain stable identities across
        # that narrow boundary so Center recovery can place them in the death
        # record box after healing.
        self._pending_fainted_identities: set[PokemonIdentity] = set()
        self._lead_handoff_pending = False
        # Recovery and opportunistic route analysis share one worker so a
        # stale search cannot queue behind a newer one indefinitely.
        self._background_route_planner = _SingleFlightBackgroundPlanner()
        # A player coordinate changes on every walking frame. Keep one
        # in-flight analysis for the same map/objective/hazard envelope. The
        # tactical controller validates the exact start before executing the
        # resulting movement, so walking does not require rebuilding this
        # advisory comparison every frame.
        self._pending_route_analysis = None
        # Route safety analysis is an envelope observation.  The tactical
        # navigator still validates every movement plan against the live
        # position, but the expensive campaign analysis should not be
        # discarded and rebuilt just because the avatar walked one tile.
        self._route_analysis_anchor = None
        self._prepared_recovery_request = None
        self._prepared_recovery_anchor = None
        self._last_recovery_request_panel_key = None
        self._last_recovery_destination_panel_key = None
        self._last_recovery_calculation_panel_key = None
        self._advisory_route_retry_deadlines = {}
        self._advisory_route_retry_notices = set()

    def _clear_background_route_state(self) -> None:
        """Discard frame-bound recovery inputs after a semantic boundary."""

        prepared = getattr(self, "_prepared_recovery_request", None)
        planner = getattr(self, "_background_route_planner", None)
        if planner is not None and hasattr(prepared, "key"):
            planner.discard(("recovery", prepared.key))
        pending_analysis = getattr(self, "_pending_route_analysis", None)
        if planner is not None and pending_analysis is not None:
            planner.discard(pending_analysis["key"])
        self._pending_route_analysis = None
        self._route_analysis_anchor = None
        self._prepared_recovery_request = None
        self._prepared_recovery_anchor = None
        self._last_recovery_calculation_panel_key = None
        getattr(self, "_advisory_route_retry_deadlines", {}).clear()
        getattr(self, "_advisory_route_retry_notices", set()).clear()

    @staticmethod
    def _recovery_anchor():
        """Return the live location boundary for one recovery calculation.

        The prepared request contains the exact starting coordinate.  A
        running worker is allowed to finish when a transient snapshot moves
        within the same map, but a completed request is retired before a new
        coordinate is planned.  This preserves route correctness without
        turning each pending frame into a new search.
        """

        runtime = getattr(context, "nuzlocke_runtime", None)
        snapshot = getattr(runtime, "latest_snapshot", None)
        player = getattr(snapshot, "player", None)
        if player is None:
            return None
        return (
            getattr(player, "map_group", None),
            getattr(player, "map_number", None),
            getattr(player, "coordinates", None),
        )

    def _select_campaign(self, state):
        """Reuse map-level planning until a relevant campaign fact changes."""

        planning_key = campaign_planning_signature(state)
        restock_deferral = getattr(context, "campaign_restock_deferral", None)
        # Restock deferral is a transient execution fact. It intentionally
        # lives outside CampaignState, but it must still invalidate a cached
        # restock selection or the controller will remount the same objective
        # on the next frame.
        key = (
            planning_key
            if restock_deferral is None
            else (planning_key, ("campaign_restock_deferral", restock_deferral))
        )
        if key == self._campaign_selection_key and self._campaign_selection is not None:
            cached_objective = getattr(self._campaign_selection, "objective", None)
            if cached_objective is None:
                return self._campaign_selection
            try:
                completion = cached_objective.completion.evaluate(state)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # A partially observed state or a test seam may not expose a
                # usable completion predicate. Preserve the existing cache in
                # that case; the normal planning call remains the authority
                # when the signature changes.
                return self._campaign_selection
            if completion.status is not FactStatus.KNOWN or completion.value is not True:
                return self._campaign_selection

            # A capability can publish an authoritative completion fact and
            # request a frontier refresh before the next planning signature is
            # rebuilt. Never remount the just-completed objective from that
            # stale selection. Replan immediately even when the signature is
            # unchanged because the completion predicate is the execution
            # boundary's source of truth.
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_SELECTION_CACHE_INVALIDATE: "
                    f"objective={cached_objective.objective_id!r} "
                    f"completion_status={completion.status.value!r} "
                    f"completion_value={completion.value!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_SELECTION_CACHE_INVALIDATE",
            )
            self._campaign_selection_key = None
            self._campaign_selection = None
        selection = plan_campaign(state)
        self._campaign_selection_key = key
        self._campaign_selection = selection
        return selection

    def _observe_route_recovery_async(
        self, *, candidate_limit: int | None = None, critical: bool = False
    ) -> RouteRecovery:
        """Prepare recovery on the frame thread and search it in one worker."""

        # Most unit-level callers construct the mode with ``__new__`` to
        # exercise readiness in isolation. Preserve their synchronous seam;
        # the live mode always initializes the background planner.
        planner = getattr(self, "_background_route_planner", None)
        if planner is None:
            return observe_route_recovery(candidate_limit=candidate_limit)
        anchor = self._recovery_anchor()
        prepared = self._prepared_recovery_request
        prepared_key = ("recovery", prepared.key) if hasattr(prepared, "key") else None
        same_map = (
            prepared is not None
            and self._prepared_recovery_anchor is not None
            and anchor is not None
            and self._prepared_recovery_anchor[:2] == anchor[:2]
        )
        pending_same_map_request = same_map and planner.is_pending(prepared_key)
        if (
            prepared is None
            or self._prepared_recovery_anchor != anchor
            or getattr(prepared, "critical", False) is not critical
        ) and not pending_same_map_request:
            if prepared_key is not None:
                planner.discard(prepared_key)
            try:
                prepared = prepare_route_recovery(candidate_limit=candidate_limit, critical=critical)
            except Exception as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_PREPARATION_FAILURE: "
                        f"exception_type={type(error).__name__!r} exception={error!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_PREPARATION_FAILURE",
                )
                return RouteRecovery(observation_available=False, observation_error=str(error))
            self._prepared_recovery_request = prepared if hasattr(prepared, "key") else None
            self._prepared_recovery_anchor = anchor
            if hasattr(prepared, "ordered_sources"):
                request_panel_key = (
                    anchor,
                    critical,
                    tuple(
                        (
                            getattr(source, "source_id", None),
                            getattr(source, "outdoor_location", None),
                        )
                        for _, source in prepared.ordered_sources
                    ),
                )
                if request_panel_key != self._last_recovery_request_panel_key:
                    self._last_recovery_request_panel_key = request_panel_key
                    print_recovery_request(
                        location=getattr(prepared, "location", anchor),
                        critical=critical,
                        reason=(
                            "party HP/status reached the critical recovery threshold"
                            if critical
                            else "party reached the opportunistic recovery threshold"
                        ),
                        candidate_count=len(prepared.ordered_sources),
                        destinations=(
                            getattr(source, "source_id", None) or repr(getattr(source, "outdoor_location", None))
                            for _, source in prepared.ordered_sources
                        ),
                    )
        if isinstance(prepared, RouteRecovery):
            return prepared

        status, result, error = planner.poll(
            ("recovery", prepared.key),
            lambda budget, prepared=prepared: plan_route_recovery(prepared, cancel_check=budget.check),
            priority=0 if critical else 1,
            timeout_seconds=_RECOVERY_ROUTE_SEARCH_TIMEOUT_SECONDS,
        )
        if status == "pending":
            elapsed_getter = getattr(planner, "elapsed_seconds", None)
            elapsed_seconds = (
                elapsed_getter(("recovery", prepared.key))
                if callable(elapsed_getter)
                else None
            )
            calculation_panel_key = ("pending", prepared.key)
            if calculation_panel_key != getattr(self, "_last_recovery_calculation_panel_key", None):
                self._last_recovery_calculation_panel_key = calculation_panel_key
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_CALCULATION: "
                        f"state='pending' request_key={prepared.key!r} "
                        f"elapsed_ms={elapsed_seconds * 1000.0 if elapsed_seconds is not None else None!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_CALCULATION",
                )
            return RouteRecovery(
                observation_available=False,
                observation_error="route_calculation_pending",
                calculation_pending=True,
            )
        if error is not None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_CALCULATION: "
                    f"state='error' exception_type={type(error).__name__!r} exception={error!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_CALCULATION",
            )
            destination_diagnostics = tuple(getattr(error, "candidate_diagnostics", ()) or ())
            destination_panel_key = (prepared.key, repr(destination_diagnostics), "error")
            if destination_diagnostics and destination_panel_key != self._last_recovery_destination_panel_key:
                self._last_recovery_destination_panel_key = destination_panel_key
                print_recovery_destinations(destination_diagnostics)
            timed_out = isinstance(error, TimeoutError)
            candidate_search_incomplete = any(
                getattr(item, "outcome", None) in {"pruned", "pending", "timed_out"}
                for item in destination_diagnostics
            )
            if timed_out or candidate_search_incomplete:
                planner.discard(("recovery", prepared.key))
                self._clear_background_route_state()
            return RouteRecovery(
                # A non-empty catalog is not evidence that a route was
                # found. Reporting it as available makes readiness return
                # RECOVER even though CampaignPlan has no executable
                # RecoveryStop, producing a false campaign block.
                # Treat worker failures as unavailable observations so the
                # same request can be retried after a transient boundary.
                observation_available=timed_out,
                observation_error=str(error),
                calculation_timed_out=timed_out or candidate_search_incomplete,
                candidate_diagnostics=destination_diagnostics,
            )
        calculation_panel_key = (
            "complete",
            prepared.key,
            getattr(result, "center_location", None),
            getattr(result, "distance_to_center", None),
        )
        if calculation_panel_key != getattr(self, "_last_recovery_calculation_panel_key", None):
            self._last_recovery_calculation_panel_key = calculation_panel_key
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_CALCULATION: "
                    f"state='complete' center={getattr(result, 'center_location', None)!r} "
                    f"distance={getattr(result, 'distance_to_center', None)!r} "
                    f"calculation_timed_out={getattr(result, 'calculation_timed_out', False)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_CALCULATION",
            )
            selected_route = getattr(result, "route", None)
            selected_metrics = getattr(selected_route, "metrics", None)
            print_recovery_handoff(
                phase="calculation_complete",
                start=getattr(prepared, "location", None),
                destination=getattr(result, "center_location", None),
                source_id=getattr(getattr(result, "healing_source", None), "source_id", None),
                route_actions=(
                    len(getattr(selected_route, "actions", ()))
                    if selected_route is not None
                    else None
                ),
                route_cost=getattr(selected_metrics, "total_route_cost", None),
                detail=(
                    "partial recovery route calculated after candidate timeout; "
                    "awaiting campaign controller mount"
                    if getattr(result, "calculation_timed_out", False)
                    else "recovery route calculated; awaiting campaign controller mount"
                ),
            )
        destination_diagnostics = tuple(getattr(result, "candidate_diagnostics", ()) or ())
        destination_panel_key = (prepared.key, repr(destination_diagnostics))
        if destination_panel_key != self._last_recovery_destination_panel_key:
            self._last_recovery_destination_panel_key = destination_panel_key
            print_recovery_destinations(destination_diagnostics)
        return result

    @staticmethod
    def _route_analysis_envelope_key(world, goal, candidates) -> tuple:
        """Return inputs that remain stable while walking one route envelope."""
        _blocked, transitions = world.revision_key()
        # ``NavigationWorld`` overlays live NPC occupancy on top of the ROM
        # tile catalog.  That occupancy changes as a wandering NPC moves, but
        # it is already revalidated by the tactical planner before each input.
        # Including it here turns a stable campaign analysis into a new search
        # every frame. Keep static collision topology in the envelope and let
        # the live tactical plan own transient occupancy.
        tiles = getattr(world, "tiles", {})
        static_tiles = getattr(tiles, "_static_tiles", tiles)
        static_blocked = tuple(
            sorted(
                (location[1] for location, tile in static_tiles.items() if getattr(tile, "blocked", False)),
                key=repr,
            )
        )
        # Trainer state is live perception.  It can change while the player
        # remains on the same map and without changing tile occupancy or warp
        # revisions: an object can spawn, turn, become defeated, or expose a
        # newly decoded sight line.  Reusing an analysis across that boundary
        # can route through a trainer using stale safety evidence.
        trigger_revision = tuple(
            sorted(
                [
                    (
                        getattr(trigger, "trigger_id", None),
                        getattr(trigger, "affordance_id", None),
                        getattr(trigger, "hazard_kind", None),
                        getattr(trigger, "condition_active", None),
                        getattr(trigger, "trainer_defeated", None),
                        tuple(sorted(getattr(trigger, "hazard_locations", ()) or (), key=repr)),
                    )
                    for trigger in getattr(world, "triggers", ())
                ],
                key=repr,
            )
        )
        return (
            repr(goal),
            tuple(repr(candidate) for candidate in candidates),
            getattr(world, "running_shoes", None),
            getattr(world, "player_elevation", None),
            getattr(world, "surfing", None),
            transitions,
            static_blocked,
            trigger_revision,
        )

    @classmethod
    def _route_analysis_key(cls, world, start, goal, candidates) -> str:
        """Build an envelope key for one immutable route/hazard observation."""

        # The route result is used to publish trainer hazards and compare
        # campaign-level alternatives.  Movement execution remains exact
        # start-position work owned by AgentControlLoop.  Keeping the
        # background key independent of the current tile prevents a walking
        # campaign from superseding its own analysis every frame.
        return repr(cls._route_analysis_envelope_key(world, goal, candidates))

    def _analyze_route_async(
        self,
        world,
        graph,
        start,
        goal,
        candidates,
        *,
        tactical_goal=None,
        allow_without_tactical_owner=False,
    ):
        """Run opportunistic route composition without blocking a frame.

        A completed recovery route is itself the executable owner while the
        controller is waiting for the parent-route comparison.  In that
        boundary there is intentionally no mounted tactical loop to guard,
        so continuation analysis must be allowed to start from the campaign
        readiness worker.
        """

        active_loop = getattr(context, "active_agent_control_loop", None)
        active_loop_goal = getattr(active_loop, "_goal", None)
        tactical_owner_goal = (
            active_loop_goal
            if _campaign_goal_owns_trainer_exposure(active_loop_goal)
            else tactical_goal
        )
        trainer_owner_id = _campaign_trainer_id(tactical_owner_goal)
        trainer_checkpoint = trainer_checkpoint_for_id(trainer_owner_id)
        if trainer_owner_id is not None and trainer_checkpoint is not None:
            campaign_facts = _runtime_campaign_facts_or_none()
            if not _checkpoint_is_cleared(campaign_facts, trainer_checkpoint):
                pending = getattr(self, "_pending_route_analysis", None)
                planner = getattr(self, "_background_route_planner", None)
                if pending is not None and planner is not None:
                    planner.discard(pending["key"])
                self._pending_route_analysis = None
                self._route_analysis_anchor = None
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_ROUTE_ANALYSIS: "
                        f"state='deferred' owner='trainer' trainer={trainer_owner_id!r} "
                        "reason='unresolved trainer choke point owns tactical route'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_ROUTE_ANALYSIS",
                )
                return None, False, False

        planner = getattr(self, "_background_route_planner", None)
        if planner is None:
            return RouteCostAnalyzer(
                world,
                graph=graph,
                max_expansions=_RECOVERY_ROUTE_ANALYSIS_MAX_EXPANSIONS,
            ).analyze(start, goal, candidates), False, False

        # Tactical navigation has priority over campaign diagnostics. On the
        # first readiness pass the tactical generator has not been mounted
        # yet, so there is no active worker for the guard below to observe.
        # Let the controller mount that executable owner first; a later
        # readiness boundary can start this advisory analysis while the
        # tactical route is cached or after its search has completed.
        tactical_owner_mounted = active_loop is not None and (
            tactical_goal is None or active_loop_goal == tactical_goal
        )
        if (
            tactical_goal is not None
            and not tactical_owner_mounted
            and not allow_without_tactical_owner
        ):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                    "state='deferred' reason='tactical_loop_mount_pending'"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
            )
            return None, False, False

        # Tactical navigation has priority over campaign diagnostics. Check
        # this before constructing any route key so a transition boundary or
        # lightweight test double cannot be dereferenced while an active
        # tactical search already owns the frame.
        if navigation_background_search_active():
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                    "state='deferred' reason='tactical_navigation_search_active'"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
            )
            return None, True, False

        # Tactical navigation has priority over campaign diagnostics. Both
        # analyzers can ask the same global navigator for a multi-map route;
        # allowing them to run concurrently competes for CPU and can leave
        # the tactical controller waiting on a second search. Route analysis
        # is advisory and is safe to poll again after the executable route is
        # no longer being composed.
        key = ("analysis", self._route_analysis_key(world, start, goal, candidates))
        envelope_key = self._route_analysis_envelope_key(world, goal, candidates)
        retry_deadlines = getattr(self, "_advisory_route_retry_deadlines", None)
        if retry_deadlines is None:
            retry_deadlines = self._advisory_route_retry_deadlines = {}
        retry_notices = getattr(self, "_advisory_route_retry_notices", None)
        if retry_notices is None:
            retry_notices = self._advisory_route_retry_notices = set()
        retry_deadline = retry_deadlines.get(key)
        if retry_deadline is not None:
            if time.monotonic() < retry_deadline:
                if key not in retry_notices:
                    retry_notices.add(key)
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                            f"state='cooldown' request_key={key[1]!r} "
                            f"remaining_seconds={max(0.0, retry_deadline - time.monotonic()):.3f}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
                    )
                return None, True, False
            retry_deadlines.pop(key, None)
            retry_notices.discard(key)
        owner_checkpoint = getattr(trainer_checkpoint, "checkpoint_id", None)
        diagnostic_edge = (start, getattr(getattr(goal, "target", goal), "target_map", None), len(candidates))
        operation = lambda budget: RouteCostAnalyzer(
                world,
                graph=graph,
                max_expansions=_RECOVERY_ROUTE_ANALYSIS_MAX_EXPANSIONS,
            ).analyze(start, goal, candidates, cancel_check=budget.check)
        pending = getattr(self, "_pending_route_analysis", None)
        if pending is not None and pending["key"] != key:
            # A changed map, objective, or trainer hazard envelope invalidates
            # the old search.  A changed player tile does not, because the
            # key intentionally represents the stable envelope rather than a
            # frame-by-frame tactical start position.
            planner.discard(pending["key"])
            self._pending_route_analysis = None

        status, result, error = planner.poll(key, operation, priority=2)
        if status == "pending":
            self._pending_route_analysis = {
                "key": key,
                "envelope_key": envelope_key,
                "anchor": start,
                "operation": operation,
            }
        elif getattr(self, "_pending_route_analysis", None) is not None and self._pending_route_analysis["key"] == key:
            self._pending_route_analysis = None
        if status == "pending":
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                    f"state='pending' request_key={key[1]!r} owner={trainer_owner_id!r} "
                    f"checkpoint={owner_checkpoint!r} candidate_edge={diagnostic_edge!r} priority=2"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
            )
            return None, True, False
        if error is not None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                    f"state='error' exception_type={type(error).__name__!r} exception={error!r} "
                    f"owner={trainer_owner_id!r} checkpoint={owner_checkpoint!r} "
                    f"candidate_edge={diagnostic_edge!r} priority=2"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
            )
            timed_out = isinstance(error, TimeoutError)
            if timed_out:
                # The request key is stable, so discard the terminal worker
                # result and allow the readiness scheduler's cooldown to
                # submit a fresh bounded request later.
                planner.discard(key)
                retry_deadlines[key] = time.monotonic() + _ADVISORY_ROUTE_RETRY_COOLDOWN_SECONDS
                retry_notices.discard(key)
            return None, False, timed_out
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                f"state='complete' request_key={key[1]!r} owner={trainer_owner_id!r} "
                f"checkpoint={owner_checkpoint!r} candidate_edge={diagnostic_edge!r} priority=2"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
        )
        retry_deadlines.pop(key, None)
        retry_notices.discard(key)
        return result, False, False

    def _campaign_controller(self):
        """Return the controller owned by this mode instance."""

        return self.controller

    @staticmethod
    def _cheap_readiness_context():
        """Read inexpensive boundary inputs used to age readiness diagnostics.

        Exact coordinates are intentionally excluded.  Walking normally changes
        them every frame, but does not change the party's readiness facts or the
        set of map-level recovery affordances.  Battle completion explicitly
        invalidates the scheduler, while the map identity below naturally
        invalidates it across a map transition. Encounter-stage evidence is
        also included because a tactical encounter probe can discover a route
        trainer after the ordinary readiness observation was cached.
        """

        runtime = getattr(context, "nuzlocke_runtime", None)
        snapshot = getattr(runtime, "latest_snapshot", None)
        encounter_stage_key = _encounter_stage_readiness_key()
        if snapshot is not None:
            player = getattr(snapshot, "player", None)
            map_id = (
                getattr(player, "map_group", None),
                getattr(player, "map_number", None),
            )
            state = getattr(getattr(snapshot, "game_state", None), "name", None)
            controllable = getattr(player, "controllable", None)
            # The runtime snapshot is intentionally immutable and can lag one
            # frame behind gMapHeader while a door/map transition settles.
            # Read the live avatar boundary for map ownership so readiness is
            # guaranteed to run once on the destination map.
            try:
                avatar = get_player_avatar()
                live_map_id = getattr(avatar, "map_group_and_number", None)
                if live_map_id is not None:
                    map_id = getattr(live_map_id, "value", live_map_id)
                controllable = player_avatar_is_controllable()
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                pass
            return (
                state,
                map_id,
                controllable,
                getattr(snapshot, "readiness_revision", None),
                encounter_stage_key,
            )
        try:
            avatar = get_player_avatar()
            state = getattr(get_game_state(), "name", None)
            map_id = avatar.map_group_and_number
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        try:
            # A script can leave the game state and map unchanged while
            # releasing control on a later frame. Include this cheap
            # boundary so an unavailable readiness result is refreshed
            # immediately when tactical ownership returns. Keep a failed
            # control read distinct from a failed avatar/map read so test
            # doubles and transient emulator boundaries still have a useful
            # map-level cache key.
            controllable = player_avatar_is_controllable()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            controllable = None
        # Runtime already materializes the inventory and party snapshot once
        # per emulator frame. Reuse those immutable observations so HP/status,
        # item, healing, and party-composition changes invalidate readiness
        # without rereading save blocks on the hot path.
        ball_count = None
        if snapshot is not None and getattr(snapshot, "inventory_available", False):
            try:
                ball_count = sum(
                    item.quantity for item in snapshot.inventory.poke_balls if item.name.casefold() != "master ball"
                )
            except (AttributeError, TypeError, ValueError):
                ball_count = None
        if snapshot is None:
            # Lightweight embedders may not mount the Nuzlocke runtime. Keep
            # their historical coarse key shape and behavior.
            return (state, map_id, controllable, ball_count, encounter_stage_key)

        try:
            party_signature = tuple(
                (
                    pokemon.party_index,
                    pokemon.species,
                    pokemon.level,
                    pokemon.current_hp,
                    pokemon.max_hp,
                    pokemon.status,
                    pokemon.fainted,
                    pokemon.egg,
                    pokemon.held_item,
                    tuple((move.name, move.pp, move.total_pp, move.pp_ups) for move in pokemon.moves),
                )
                for pokemon in snapshot.party
            )
        except (AttributeError, TypeError, ValueError):
            party_signature = None
        try:
            inventory_signature = tuple(
                (item.name, item.quantity)
                for item in (
                    tuple(snapshot.inventory.items)
                    + tuple(snapshot.inventory.poke_balls)
                    + tuple(snapshot.inventory.key_items)
                )
            )
        except (AttributeError, TypeError, ValueError):
            inventory_signature = None
        try:
            campaign_observation = snapshot.campaign_observation
            campaign_signature = (
                tuple((flag.name, flag.value) for flag in campaign_observation.flags),
                tuple((variable.name, variable.value) for variable in campaign_observation.variables),
                campaign_observation.text_speed,
                getattr(campaign_observation.lifecycle, "value", campaign_observation.lifecycle),
            )
        except (AttributeError, TypeError, ValueError):
            campaign_signature = None
        return (
            state,
            map_id,
            controllable,
            ball_count,
            party_signature,
            inventory_signature,
            campaign_signature,
            encounter_stage_key,
        )

    @staticmethod
    def _readiness_transient_context():
        """Return the live avatar boundary used for transient readiness.

        The stable readiness cache intentionally ignores walking coordinates.
        A transient ``OVERWORLD_UNAVAILABLE`` result is different: it can be
        produced while the avatar still has a warp-time coordinate such as
        ``(10, -1)`` and must be refreshed when the live avatar reaches its
        actual destination tile.
        """

        try:
            avatar = get_player_avatar()
            return avatar.map_group_and_number, avatar.local_coordinates
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _recovery_candidate_goals(current_location, recovery, *, is_rse):
        """Return a bounded set of recovery destinations for readiness analysis.

        ``observe_route_recovery`` has already selected the nearest safe
        source.  Readiness only needs to compare that route with the active
        objective, so enumerating every healing source in the region is both
        redundant and capable of blocking the tactical loop for a long time.
        Keep same-map sources when present (there can be more than one in a
        future ROM), otherwise compare the selected destination only.
        """
        selected_goal = getattr(recovery, "navigation_goal", None)
        if selected_goal is not None:
            # Recovery runtime already selected and validated one catalog
            # source. Reuse its executable interior-warp goal so the
            # readiness composition measures the same route that recovery
            # will execute.
            return (selected_goal,)
        if is_rse:
            local_sources = emerald_healing_sources_for_map(current_location[0])
            if local_sources:
                return tuple(ReachLocation(source.outdoor_location) for source in local_sources)
            selected = getattr(recovery, "center_location", None)
            if selected is not None:
                return (ReachLocation(selected),)

        centers = pokemon_center_candidates(current_location)
        return tuple(ReachLocation(center.value) for center in centers[:1])

    def _readiness_input(self, objective, goal):
        """Build the current readiness diagnostic from shared ROM observations."""

        readiness_policy = getattr(self, "_readiness_policy", None) or CampaignReadinessPolicy()
        self._readiness_evaluated = True
        runtime = getattr(context, "nuzlocke_runtime", None)
        session_id = getattr(runtime, "session_id", None)
        if session_id is not None and getattr(context, "campaign_defeated_trainer_session_id", None) != session_id:
            context.campaign_defeated_trainer_ids = set()
            context.campaign_defeated_trainer_session_id = session_id
            context.campaign_trainer_facts = ()
            context.campaign_encounter_stage = None
            context.campaign_capability_completion_latch = None
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_FRONTIER_RESET: "
                    f"session_id={session_id!r} reason='observed campaign session changed'"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_FRONTIER_RESET",
            )
        # Perceive the overworld first.  The callback-based game-state reader
        # can briefly report UNKNOWN at a battle/script boundary even when
        # the authoritative overworld observation has already become stable.
        # Several Emerald capabilities intentionally have no tactical goal:
        # their observation-driven delegate owns the script interaction. They
        # still have a declared map destination, which is sufficient for
        # readiness to compare continuing with returning to a healing source.
        # Without this projection a damaged party is classified as
        # opportunistically unknown and the campaign waits in place.
        readiness_goal = goal
        # Interaction capabilities can own a destination without exposing a
        # navigation goal of their own.  Readiness still needs to inspect the
        # outbound route: this is how a mandatory story battle is claimed by
        # the trainer checkpoint envelope before the interaction capability
        # starts walking.  Preserve the original interaction goal for
        # execution; this projection is only for safety analysis.
        if isinstance(readiness_goal, ActivateTrigger) or readiness_goal is None:
            destination = getattr(objective, "destination", None)
            if isinstance(destination, tuple) and len(destination) == 2:
                target_map = (
                    destination[0]
                    if isinstance(destination[0], tuple)
                    and len(destination[0]) == 2
                    and isinstance(destination[1], tuple)
                    else destination
                )
                readiness_goal = NavigationGoal(
                    SemanticTarget.map(target_map),
                    constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
                    encounter_mode=EncounterMode.AVOID,
                )

        trainer_checkpoint = None
        trainer_readiness = None
        trainer_engagement_required = False
        overworld = perceive_overworld()
        publish_shared_overworld_observation(overworld)
        try:
            visible_trainer_facts = trainer_checkpoints_for_overworld(overworld)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            visible_trainer_facts = ()
        previous_trainer_facts = tuple(getattr(context, "campaign_trainer_facts", ()) or ())
        if visible_trainer_facts:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_FACTS: "
                    f"facts={tuple((fact.trainer_affordance_id, fact.display_name, fact.checkpoint_id, fact.source) for fact in visible_trainer_facts)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_FACTS",
            )
        trainer_target = (
            getattr(readiness_goal, "target", None)
            if not isinstance(readiness_goal, EngageTrainer)
            else readiness_goal
        )
        if isinstance(trainer_target, EngageTrainer):
            trainer_engagement_required = True
            trainer_checkpoint = trainer_checkpoint_for_source(
                trainer_target.trainer_id,
                script_symbol=_trainer_script_for_id(overworld, trainer_target.trainer_id),
            )
            if trainer_checkpoint is not None:
                try:
                    trainer_readiness = assess_live_trainer_checkpoint(
                        trainer_checkpoint,
                        get_party(),
                        expected_exp=projected_trainer_experience(trainer_checkpoint),
                        pc_candidates=_pc_trainer_readiness_candidates(),
                    )
                    context.campaign_trainer_checkpoint = trainer_checkpoint
                    _remember_campaign_trainer_fact(trainer_checkpoint)
                    context.campaign_trainer_readiness = trainer_readiness
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_READINESS: "
                            f"checkpoint={trainer_checkpoint.checkpoint_id!r} "
                            f"trainer={trainer_target.trainer_id!r} "
                            f"status={trainer_readiness.status.value!r} "
                            f"win_probability={established_search_probability(trainer_readiness.search, 'win_probability')!r} "
                            f"catastrophic_probability={established_search_probability(trainer_readiness.search, 'catastrophic_probability')!r} "
                            f"recommended_action={trainer_readiness.recommended_action!r} "
                            f"lead={trainer_readiness.lead_name!r} "
                            f"lead_source={trainer_readiness.lead_source!r} "
                            f"roster_action={trainer_readiness.roster_action!r} "
                            f"reason={trainer_readiness.reason!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_READINESS",
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    trainer_readiness = BattleReadiness(
                        trainer_checkpoint.checkpoint_id,
                        BattleReadinessStatus.UNKNOWN,
                        reason="known trainer checkpoint could not be evaluated from current party facts",
                    )
                    context.campaign_trainer_checkpoint = trainer_checkpoint
                    context.campaign_trainer_readiness = trainer_readiness
            else:
                context.campaign_trainer_checkpoint = None
                trainer_readiness = BattleReadiness(
                    None,
                    BattleReadinessStatus.UNKNOWN,
                    reason="trainer checkpoint is not registered for conservative pre-battle planning",
                )
                context.campaign_trainer_readiness = trainer_readiness
        else:
            context.campaign_trainer_checkpoint = None
            context.campaign_trainer_readiness = None

        snapshot = getattr(runtime, "latest_snapshot", None) if runtime is not None else None
        if snapshot is None:
            snapshot = get_nuzlocke_snapshot()
        diagnostic_print(
            lambda: (
                "READINESS_INPUT_LIFECYCLE: "
                f"controller_id={id(self.controller)!r} evaluation_id={id(snapshot)!r} "
                f"perceive_type={type(overworld).__name__!r} "
                f"perceive_status={getattr(getattr(overworld, 'status', None), 'value', None)!r}"
            ),
            trace=True,
        )
        overworld_availability = Availability.KNOWN
        overworld_reason = None
        stable_overworld = False
        if isinstance(overworld, OverworldObservationResult):
            diagnostic_print(
                lambda: (
                    "READINESS_OVERWORLD_OBSERVATION_FAILURE: "
                    f"status={getattr(overworld.status, 'value', overworld.status)!r} "
                    f"reason={getattr(overworld, 'reason', None)!r} "
                    f"frame={getattr(snapshot, 'frame', None)!r} "
                    f"game_state={getattr(get_game_state(), 'name', get_game_state())!r}"
                ),
                trace=True,
            )
            overworld_availability = Availability.UNKNOWN
            overworld_reason = overworld.reason
            overworld = None
        elif not getattr(overworld, "controllable", False):
            # Coordinates and party state can remain readable while a battle
            # return or scripted post-battle sequence still owns the field.
            # Do not let readiness mount recovery during that interval; doing
            # so replaces the campaign loop before its dialogue can advance.
            overworld_availability = Availability.UNKNOWN
            overworld_reason = "player avatar is not controllable"
        elif getattr(overworld, "transition_in_progress", False):
            # A controllable bit can lead the rest of a warp/script transition.
            # Do not let recovery take ownership until the transition signals
            # have cleared.
            overworld_availability = Availability.UNKNOWN
            overworld_reason = "overworld transition is in progress"
        elif not _overworld_position_is_coherent(overworld):
            # The destination header and avatar coordinates are read from
            # different ROM structures.  A map connection can therefore
            # briefly expose an out-of-bounds coordinate while the map is
            # settling.  Do not mount recovery from that half-observation;
            # doing so would replace the campaign loop before the transition
            # executor can observe its postcondition.
            overworld_availability = Availability.UNKNOWN
            overworld_reason = "avatar coordinate is not present in current overworld topology"
        else:
            # The avatar can become controllable before a post-battle native
            # script has released the field. Use the shared interaction
            # classifier so recovery cannot mount during dialogue or a native
            # script wait, while allowing it as soon as the interaction phase
            # is genuinely clear.
            try:
                interaction = observe_interaction()
                interaction_phase = getattr(interaction, "interaction_phase", InteractionPhase.NONE)
                dialogue_owned = interaction_phase in {
                    InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
                    InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
                    InteractionPhase.CHOICE_MENU_INPUT_WAIT,
                } or (
                    interaction_phase is InteractionPhase.SCRIPT_NATIVE_WAIT
                    and (
                        # A native script can own the field before it
                        # exposes a dialogue box (for example, the
                        # post-battle rival exit movement). Readiness
                        # must not mount a recovery plan across that
                        # ownership boundary.
                        getattr(interaction, "script_active", False)
                        or getattr(interaction, "dialogue_waiting", False)
                        or getattr(interaction, "field_message_lifecycle_active", False)
                        or getattr(interaction, "native_function", None)
                        in {"WaitForAorBPress", "IsFieldMessageBoxHidden"}
                    )
                )
                if dialogue_owned:
                    overworld_availability = Availability.UNKNOWN
                    overworld_reason = "interaction phase is active: " f"{getattr(interaction_phase, 'name', None)}"
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                pass
            stable_overworld = overworld_availability is Availability.KNOWN

        overworld_availability_key = (
            overworld_availability.value,
            overworld_reason,
            getattr(overworld, "map_id", None),
            getattr(overworld, "player_coordinates", None),
        )
        if getattr(self, "_last_overworld_availability_key", None) != overworld_availability_key:
            self._last_overworld_availability_key = overworld_availability_key
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_OVERWORLD_AVAILABILITY: "
                    f"availability={overworld_availability.value!r} "
                    f"reason={overworld_reason!r} "
                    f"map={getattr(overworld, 'map_id', None)!r} "
                    f"coordinates={getattr(overworld, 'player_coordinates', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_OVERWORLD_AVAILABILITY",
            )

        # ``get_nuzlocke_snapshot`` and ``perceive_overworld`` read different
        # emulator signals.  When only the callback classifier is unknown,
        # use the validated stable-overworld boundary instead of deferring
        # indefinitely.  Never normalize a known non-overworld state, and
        # never do this while dialogue or a transition owns the field.
        if stable_overworld and getattr(snapshot, "game_state", None) is GameState.UNKNOWN:
            snapshot = replace(snapshot, game_state=GameState.OVERWORLD)
            diagnostic_print(
                lambda: (
                    "READINESS_GAME_STATE_NORMALIZED: "
                    "snapshot_state='UNKNOWN' observation_state='STABLE_OVERWORLD' "
                    "normalized_state='OVERWORLD'"
                ),
                trace=True,
            )
        resources = observe_resource_snapshot()
        # The party reader can briefly return a valid empty tuple while a
        # battle-return script is handing control back. That is not evidence
        # of a wiped party: treat it as unavailable unless the campaign
        # snapshot independently confirms that an empty party is real.
        resource_party_available = bool(resources.party) or bool(getattr(snapshot, "party_available", False))
        resource_observation_valid = (
            resources.observation_status is ResourceObservationStatus.VALID and resource_party_available
        )
        resource_availability = Availability.KNOWN if resource_observation_valid else Availability.UNKNOWN
        trace = getattr(context, "stutter_trace", None)
        if trace is not None and callable(getattr(trace, "mark", None)):
            trace.mark("campaign_snapshot_party_available", getattr(snapshot, "party_available", None))
            trace.mark("campaign_snapshot_party_count", len(getattr(snapshot, "party", ()) or ()))
            trace.mark("campaign_resource_party_count", len(resources.party))
            trace.mark("campaign_resource_usable_party_count", len(resources.usable_party))
            trace.mark("campaign_resource_worst_hp_ratio", resources.worst_hp_ratio)
            trace.mark("campaign_resource_party_available", resource_party_available)
        # Route recovery is only a policy input when the party may actually
        # need recovery.  Computing it here on every readiness refresh walks
        # the map synchronously, which stalls the active tactical controller
        # even when the party is healthy and already navigating normally.
        objective_id = getattr(objective, "objective_id", None)
        critical_hp_threshold = readiness_policy.critical_hp_ratio_for(objective_id)
        status_hp_threshold = readiness_policy.status_hp_ratio_for(objective_id)
        critical_recovery_needed = resource_observation_valid and (
            not resources.usable_party
            or any(member.fainted for member in resources.party)
            or resources.has_poisoned_party
            or resources.needs_status_recovery_at(status_hp_threshold)
            or resources.worst_hp_ratio < critical_hp_threshold
        )
        # The main readiness policy owns opportunistic recovery for every
        # stable-overworld objective. Objective resource metadata still
        # controls battle/encounter posture, but never whether this safety
        # check is eligible.
        opportunistic_recovery_needed = (
            resource_observation_valid
            and resources.worst_hp_ratio <= readiness_policy.opportunistic_hp_ratio
        )
        recovery_needed = critical_recovery_needed or opportunistic_recovery_needed
        # Critical recovery is a safety decision, not a detour optimization.
        # ``observe_route_recovery`` already selected and retained the
        # executable route to the nearest safe healing source. Running the
        # full normal-vs-recovery RouteCostAnalyzer here launches additional
        # world searches immediately after every damaging battle and can make
        # the emulator appear frozen for many seconds.
        if trace is not None and callable(getattr(trace, "mark", None)):
            trace.mark("campaign_recovery_needed", recovery_needed)
            trace.mark("campaign_critical_recovery", critical_recovery_needed)
        if overworld_availability is not Availability.KNOWN:
            recovery = RouteRecovery(
                observation_available=False,
                observation_error=overworld_reason or "overworld_observation_unavailable",
            )
        elif not resource_observation_valid:
            recovery = RouteRecovery(
                observation_available=False,
                observation_error=resources.observation_error or "resource_observation_unavailable",
            )
        else:
            recovery = (
                # Critical recovery already selected the nearest safe source
                # in the recovery observer. A second candidate only repeats
                # the expensive synchronous route search at the worst
                # possible time: immediately after a battle.
                self._observe_route_recovery_async(
                    candidate_limit=None,
                    critical=critical_recovery_needed,
                )
                if recovery_needed
                else RouteRecovery()
            )
        recovery_available = (
            recovery.center_available
            or getattr(recovery, "healing_source_available", False)
            or bool(resources.bag_healing_items)
        )
        recovery_availability = (
            Availability.UNKNOWN
            if not recovery.observation_available
            else (
                Availability.UNKNOWN
                if resources.observation_status is not ResourceObservationStatus.VALID
                else Availability.KNOWN if recovery_available else Availability.UNAVAILABLE
            )
        )
        # Diagnostics must remain strictly non-failing when a partial snapshot
        # is observed during startup or a map/battle transition.
        try:
            party_rows = []
            for pokemon in getattr(snapshot, "party", None) or ():
                current_hp = getattr(pokemon, "current_hp", None)
                max_hp = getattr(pokemon, "max_hp", None)
                fainted = getattr(pokemon, "fainted", None)
                if fainted is True or current_hp is None or current_hp <= 0:
                    continue
                ratio = current_hp / max_hp if isinstance(max_hp, (int, float)) and max_hp > 0 else None
                party_rows.append({"current_hp": current_hp, "max_hp": max_hp, "hp_ratio": ratio})
        except Exception:
            party_rows = None
        resource_status = getattr(getattr(resources, "observation_status", None), "value", None)
        recovery_center_available = getattr(recovery, "center_available", None)
        recovery_safe = getattr(recovery, "safe_to_reach_center", None)
        recovery_distance = getattr(recovery, "distance_to_center", None)
        diagnostic_print(
            lambda: (
                "READINESS_INPUT: "
                f"frame={getattr(snapshot, 'frame', None)!r} map={getattr(overworld, 'map_id', None)!r} "
                f"location={getattr(overworld, 'player_coordinates', None)!r} "
                f"party={party_rows!r} party_status={getattr(snapshot, 'party_available', None)!r} resource_status={resource_status!r} "
                f"resource_party_count={len(resources.party)!r} resource_usable_party_count={len(resources.usable_party)!r} "
                f"resource_party_available={resource_party_available!r} recovery_needed={recovery_needed!r} "
                f"recovery_availability={getattr(recovery_availability, 'value', None)!r} center_available={recovery_center_available!r} "
                f"center_safe={recovery_safe!r} distance_to_center={recovery_distance!r} "
                "distance_metric='world navigation route cost' "
                f"nearest_center={getattr(recovery, 'center_location', None)!r}"
            ),
            trace=True,
        )
        route_analysis = None
        route_analysis_pending = False
        route_analysis_timed_out = False
        # A targetless capability has no tactical route to compare against.
        # Its readiness policy uses the already-selected recovery route and
        # its bounded distance; composing a normal route plus recovery
        # outbound/continuation routes here launches redundant synchronous
        # searches in one application frame.
        targetless_route_owner = (
            objective is not None
            and goal is None
            and (
                getattr(objective, "objective_id", "").startswith("reach_")
                or getattr(objective, "objective_id", "").startswith("obtain_encounter:")
            )
        )
        direct_targetless_recovery = objective is not None and goal is None and not targetless_route_owner
        if direct_targetless_recovery and recovery_needed and not critical_recovery_needed:
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='targetless capability uses selected recovery route directly' "
                    f"distance_to_center={getattr(recovery, 'distance_to_center', None)!r}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        recovery_distance_for_policy = getattr(recovery, "distance_to_center", None)
        nearby_recovery_route_ready = (
            recovery_needed
            and not critical_recovery_needed
            and getattr(recovery, "center_available", False) is True
            and getattr(recovery, "safe_to_reach_center", False) is True
            and getattr(recovery, "route", None) is not None
            and isinstance(recovery_distance_for_policy, (int, float))
            and not isinstance(recovery_distance_for_policy, bool)
            and recovery_distance_for_policy <= readiness_policy.opportunistic_detour_threshold
        )
        # A short route to the Center is only sufficient when this capability
        # has no parent destination. A targetful objective must compare the
        # complete ``start -> Center -> parent goal`` route with the direct
        # route; the Center leg alone says nothing about the outbound
        # continuation.
        if nearby_recovery_route_ready and direct_targetless_recovery:
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='targetless capability uses selected recovery route directly' "
                    f"distance_to_center={recovery_distance_for_policy!r}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        completed_recovery_route = (
            recovery_needed
            and getattr(recovery, "center_available", False) is True
            and getattr(recovery, "safe_to_reach_center", False) is True
            and getattr(recovery, "route", None) is not None
        )
        if completed_recovery_route and not critical_recovery_needed and direct_targetless_recovery:
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='completed_recovery_route_is_authoritative' "
                    f"destination={getattr(recovery, 'center_location', None)!r} "
                    f"distance_to_center={getattr(recovery, 'distance_to_center', None)!r}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        if overworld is None or readiness_goal is None:
            diagnostic_print(
                lambda: (
                    "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE\n"
                    "exception_type=UnavailableContext\n"
                    "exception_message=route-analysis precondition unavailable\n"
                    "failing_operation=precondition\n"
                    f"active_tactical_goal={readiness_goal!r}\n"
                    f"goal_type={type(readiness_goal).__name__ if readiness_goal is not None else None}\n"
                    f"navigation_world_available={overworld is not None}\n"
                    f"world_graph_available=False\n"
                    "candidate_count=0\n"
                    "candidate_destinations=[]"
                ),
                trace=True,
            )
        if (
            recovery_needed
            and not critical_recovery_needed
            and not direct_targetless_recovery
            and not getattr(recovery, "calculation_pending", False)
            and not getattr(recovery, "calculation_timed_out", False)
            and overworld is not None
            and overworld_availability is Availability.KNOWN
            and readiness_goal is not None
        ):
            operation = "initialization"
            candidate_goals = ()
            current_location = None
            world = None
            graph = None
            try:
                operation = "current location"
                current_location = (overworld.map_id, overworld.player_coordinates)
                operation = "NavigationWorld.from_overworld"
                world = NavigationWorld.from_overworld(overworld)
                operation = "candidate enumeration"
                if context.rom.is_rse:
                    candidates = self._recovery_candidate_goals(
                        current_location,
                        recovery,
                        is_rse=True,
                    )
                else:
                    candidates = self._recovery_candidate_goals(
                        current_location,
                        recovery,
                        is_rse=False,
                    )
                candidate_goals = candidates
                operation = "WorldMapGraph acquisition"
                graph = get_world_map_graph()
                operation = "RouteCostAnalyzer.analyze"
                route_analysis, route_analysis_pending, route_analysis_timed_out = self._analyze_route_async(
                    world,
                    graph,
                    current_location,
                    readiness_goal,
                    candidates,
                    tactical_goal=goal,
                    allow_without_tactical_owner=completed_recovery_route,
                )
                if route_analysis_pending:
                    diagnostic_print(
                        lambda: (
                            "READINESS_ROUTE_ANALYSIS: "
                            f"state='pending' start={current_location!r} goal={readiness_goal!r}"
                        ),
                        trace=True,
                        prefix="READINESS_ROUTE_ANALYSIS",
                    )
                elif route_analysis is None:
                    diagnostic_print(
                        lambda: (
                            "READINESS_ROUTE_ANALYSIS: "
                            f"state='deferred' start={current_location!r} goal={readiness_goal!r}"
                        ),
                        trace=True,
                        prefix="READINESS_ROUTE_ANALYSIS",
                    )
                else:
                    diagnostic_print(
                        lambda: (
                            "READINESS_ROUTE_ANALYSIS: "
                            f"state='complete' start={current_location!r} goal={readiness_goal!r} "
                            f"normal_cost={route_analysis.normal_cost!r} "
                            f"normal_destination={getattr(route_analysis.normal_route, 'destination', None)!r} "
                            "candidates="
                            f"{[
                                {
                                    'destination': repr(candidate.destination),
                                    'reachable': candidate.reachable,
                                    'total_cost': candidate.total_cost,
                                    'detour': candidate.detour,
                                    'reason': candidate.reason,
                                    'first_destination': getattr(candidate.first_route, 'destination', None),
                                    'continuation_destination': getattr(candidate.continuation_route, 'destination', None),
                                }
                                for candidate in route_analysis.candidates
                            ]!r}"
                        ),
                        trace=True,
                        prefix="READINESS_ROUTE_ANALYSIS",
                    )
            except (RuntimeError, TypeError, ValueError) as error:
                diagnostic_print(
                    lambda: (
                        "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE\n"
                        f"exception_type={type(error).__name__}\n"
                        f"exception_message={error}\n"
                        f"failing_operation={operation}\n"
                        f"active_tactical_goal={readiness_goal!r}\n"
                        f"goal_type={type(readiness_goal).__name__ if readiness_goal is not None else None}\n"
                        f"current_map_location={current_location!r}\n"
                        f"navigation_world_available={world is not None}\n"
                        f"world_graph_available={graph is not None}\n"
                        f"candidate_count={len(candidate_goals)}\n"
                        f"candidate_destinations={[repr(candidate) for candidate in candidate_goals]!r}\n"
                        f"traceback={traceback.format_exc()}"
                    ),
                    trace=True,
                )
                route_analysis = None
        elif recovery_needed and critical_recovery_needed:
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='critical recovery uses selected recovery route' "
                    f"recovery_route_available={getattr(recovery, 'route', None) is not None}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        # Trainer exposure is a campaign safety fact even when no recovery
        # detour is being considered.  The older implementation only had a
        # RouteAnalysis on the opportunistic-recovery path, so a healthy party
        # could traverse directly to Petalburg without ever inspecting its
        # normal route for Calvin's sight line. Reuse the bounded background
        # analyzer with no intermediate destinations to obtain that route.
        trainer_route_goal = readiness_goal or getattr(recovery, "navigation_goal", None)
        if (
            recovery_needed
            and getattr(recovery, "calculation_pending", False)
            and route_analysis is None
        ):
            # Recovery owns the shared route worker until its selected healing
            # route is known. Trainer-risk analysis is advisory at this
            # boundary; launching it here only queues stale work behind the
            # recovery calculation and makes the readiness loop appear stuck.
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='recovery_calculation_pending' "
                    f"recovery_error={getattr(recovery, 'observation_error', None)!r}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        if (
            route_analysis is None
            and not route_analysis_pending
            and not route_analysis_timed_out
            and not getattr(recovery, "calculation_pending", False)
            and overworld_availability is Availability.KNOWN
            and _is_map_traversal_readiness_goal(trainer_route_goal)
            and (
                goal is not None
                or getattr(objective, "objective_id", "").startswith("reach_")
                    or getattr(objective, "objective_id", "").startswith("obtain_encounter:")
                    or (
                        getattr(objective, "destination", None) is not None
                        and getattr(objective, "objective_id", None) != "receive_pokedex"
                    )
                    or (
                    direct_targetless_recovery
                    and getattr(objective, "objective_id", "").startswith("restock_")
                )
            )
        ):
            try:
                current_location = (overworld.map_id, overworld.player_coordinates)
                world = NavigationWorld.from_overworld(overworld)
                graph = get_world_map_graph()
                route_analysis, route_analysis_pending, route_analysis_timed_out = self._analyze_route_async(
                    world,
                    graph,
                    current_location,
                    trainer_route_goal,
                    (),
                    tactical_goal=goal,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_ROUTE_ANALYSIS: "
                        f"state={'pending' if route_analysis_pending else 'timed_out' if route_analysis_timed_out else 'complete'} "
                        f"normal_destination={getattr(getattr(route_analysis, 'normal_route', None), 'destination', None)!r} "
                        f"forced_trainer_exposure={getattr(getattr(route_analysis, 'normal_route', None), 'forced_trainer_exposure', None)!r} "
                        f"trainer_risks={[(getattr(risk, 'trainer_id', None), getattr(getattr(risk, 'avoidability', None), 'value', None), getattr(risk, 'detour', None)) for risk in getattr(route_analysis, 'trainer_risks', ())]!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_ROUTE_ANALYSIS",
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                route_analysis = None

        encounter_stage_risks = _encounter_stage_route_risks(overworld, objective)
        if encounter_stage_risks:
            if route_analysis is None:
                route_analysis = RouteAnalysis(
                    readiness_goal,
                    None,
                    None,
                    (),
                    encounter_stage_risks,
                )
            else:
                route_analysis = replace(
                    route_analysis,
                    trainer_risks=tuple(getattr(route_analysis, "trainer_risks", ()) or ())
                    + encounter_stage_risks,
                )
            for risk in encounter_stage_risks:
                if getattr(risk, "confidence", "known") != "known":
                    continue
                checkpoint = _encounter_stage_checkpoint_for_id(
                    overworld,
                    risk.trainer_id,
                    getattr(context, "campaign_encounter_stage", None),
                )
                if checkpoint is not None:
                    _remember_campaign_trainer_fact(checkpoint)
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_STAGE_ROUTE_RISK: "
                    f"risks={[(risk.trainer_id, risk.avoidability.value, risk.confidence) for risk in encounter_stage_risks]!r} "
                    f"reason={getattr(getattr(context, 'campaign_encounter_stage', None), 'reason', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_STAGE_ROUTE_RISK",
            )

        static_route_trainer_facts = _static_route_trainer_facts(overworld, route_analysis)
        route_trainer_facts = _merge_route_trainer_facts(
            overworld,
            (*previous_trainer_facts, *visible_trainer_facts),
            route_analysis,
            static_route_trainer_facts,
        )
        if route_trainer_facts != previous_trainer_facts:
            context.campaign_trainer_facts = route_trainer_facts
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_FACTS: "
                    f"source='route_analysis' facts={tuple((fact.trainer_affordance_id, fact.display_name, fact.checkpoint_id, fact.source) for fact in route_trainer_facts)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_FACTS",
            )

        # Route analysis supplies two separate facts: a trainer actually
        # intersects the selected path, and a trainer-free alternative exists
        # but costs enough to be strategically unattractive.  Promote either
        # case to an explicit EngageTrainer goal only when the checkpoint is
        # known and the policy says the exposure deserves preparation. Cheap
        # avoidance therefore remains the default, while Calvin-like
        # forced/near-forced routes become deliberate, prepared battles.
        route_trainer_id = _known_trainer_checkpoint_on_route(
            overworld,
            readiness_goal,
            route_analysis,
        )
        if route_trainer_id is None and static_route_trainer_facts:
            route_trainer_id = static_route_trainer_facts[0].trainer_affordance_id
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_STATIC_ROUTE_TRAINER: "
                    f"trainer={route_trainer_id!r} "
                    f"checkpoint={static_route_trainer_facts[0].checkpoint_id!r} "
                    "reason='future-map static sight line intersects normal route'"
                ),
                trace=True,
                prefix="CAMPAIGN_STATIC_ROUTE_TRAINER",
            )
        if route_trainer_id is None:
            route_trainer_id = _mandatory_trainer_on_outbound_map(
                overworld,
                readiness_goal,
                route_analysis,
            )
            if route_trainer_id is not None:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_MANDATORY_TRAINER_FALLBACK: "
                        f"trainer={route_trainer_id!r} "
                        f"current_map={getattr(overworld, 'map_id', None)!r} "
                        f"target={getattr(getattr(readiness_goal, 'target', readiness_goal), 'target_map', None)!r} "
                        "reason='registered mandatory trainer on outbound map'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_MANDATORY_TRAINER_FALLBACK",
                )
        encounter_objective = getattr(objective, "objective_id", "").startswith("obtain_encounter:")
        promoted_risk_trainer_id = _trainer_risk_selected_for_engagement(
            route_analysis,
            readiness_policy,
        )
        # An encounter route owns movement only while it can avoid the
        # trainer. A route risk that reaches the preparation threshold changes
        # ownership: the campaign must prepare for that ROM-backed trainer
        # before mounting the encounter capability. Cheaply avoidable Cindy
        # style sight lines remain encounter-owned and are routed around.
        selected_route_trainer_id = (
            promoted_risk_trainer_id
            if encounter_objective
            else route_trainer_id or promoted_risk_trainer_id
        )
        if (
            selected_route_trainer_id is not None
            and not isinstance(readiness_goal, EngageTrainer)
            and not _objective_owns_scripted_trainer_battle(objective)
        ):
            readiness_goal = NavigationGoal(
                EngageTrainer(selected_route_trainer_id),
                constraints=GoalConstraints(trainer_mode=TrainerMode.ENGAGE),
                encounter_mode=EncounterMode.AVOID,
            )
            trainer_engagement_required = True
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_RISK_PROMOTED: "
                    f"trainer={selected_route_trainer_id!r} "
                    f"route_exposure={route_trainer_id!r} "
                    f"policy_threshold={getattr(readiness_policy, 'trainer_risk_prepare_at', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_RISK_PROMOTED",
            )

        if trainer_readiness is None:
            route_trainer_id = selected_route_trainer_id
            if route_trainer_id is not None:
                # Route promotion is already a campaign safety decision. Keep
                # the preparation owner set even if live party/PC facts are
                # transiently unavailable while resolving the checkpoint;
                # the readiness policy must wait for that evidence instead
                # of allowing a scripted capability to walk into the battle.
                trainer_engagement_required = True
                trainer_checkpoint = next(
                    (
                        fact
                        for fact in tuple(getattr(context, "campaign_trainer_facts", ()) or ())
                        if getattr(fact, "trainer_affordance_id", None) == route_trainer_id
                    ),
                    None,
                )
                if trainer_checkpoint is None:
                    trainer_checkpoint = _encounter_stage_checkpoint_for_id(
                        overworld,
                        route_trainer_id,
                        getattr(context, "campaign_encounter_stage", None),
                    )
                if trainer_checkpoint is not None:
                    try:
                        trainer_readiness = assess_live_trainer_checkpoint(
                            trainer_checkpoint,
                            get_party(),
                            expected_exp=projected_trainer_experience(trainer_checkpoint),
                            pc_candidates=_pc_trainer_readiness_candidates(),
                        )
                        context.campaign_trainer_checkpoint = trainer_checkpoint
                        _remember_campaign_trainer_fact(trainer_checkpoint)
                        context.campaign_trainer_readiness = trainer_readiness
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_TRAINER_ROUTE_READINESS: "
                                f"checkpoint={trainer_checkpoint.checkpoint_id!r} "
                                f"trainer={route_trainer_id!r} "
                                f"status={trainer_readiness.status.value!r} "
                                f"win_probability={established_search_probability(trainer_readiness.search, 'win_probability')!r} "
                                f"catastrophic_probability={established_search_probability(trainer_readiness.search, 'catastrophic_probability')!r} "
                                f"recommended_action={trainer_readiness.recommended_action!r} "
                                f"lead={trainer_readiness.lead_name!r} "
                                f"lead_source={trainer_readiness.lead_source!r} "
                                f"roster_action={trainer_readiness.roster_action!r} "
                                f"reason={trainer_readiness.reason!r}"
                            ),
                            trace=True,
                            prefix="CAMPAIGN_TRAINER_ROUTE_READINESS",
                        )
                    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                        trainer_readiness = BattleReadiness(
                            trainer_checkpoint.checkpoint_id,
                            BattleReadinessStatus.UNKNOWN,
                            reason="known route trainer checkpoint could not be evaluated from current party facts",
                        )
                        context.campaign_trainer_checkpoint = trainer_checkpoint
                        context.campaign_trainer_readiness = trainer_readiness

        return build_progression_readiness_diagnostic(
            snapshot,
            objective_id=objective.objective_id if objective is not None else None,
            objective_status="ready",
            destination=getattr(objective, "destination", None),
            navigation_goal=readiness_goal,
            campaign_mode="Campaign Progression",
            overworld=overworld,
            resource_snapshot=resources,
            recovery=recovery,
            recovery_availability=recovery_availability,
            overworld_availability=overworld_availability,
            overworld_reason=overworld_reason,
            resource_availability=resource_availability,
            resource_reason=resources.observation_error,
            route_analysis=route_analysis,
            targetless=goal is None,
            route_analysis_pending=route_analysis_pending,
            route_analysis_timed_out=route_analysis_timed_out,
            trainer_checkpoint=trainer_checkpoint,
            trainer_readiness=trainer_readiness,
            trainer_engagement_required=trainer_engagement_required,
            trainer_risks=tuple(getattr(route_analysis, "trainer_risks", ()) or ()),
            encounter_stage=getattr(context, "campaign_encounter_stage", None),
        )

    def requires_explicit_battle_policy(self) -> bool:
        """Keep Campaign Progression away from the global battle fallback."""

        return True

    def _campaign_battle_strategy_for_objective(self, objective):
        """Return the strict strategy used when a battle is already active."""

        objective_id = getattr(objective, "objective_id", None)
        if objective_id == "defeat_roxanne":
            return RoxanneBattleStrategy()
        if objective_id == "complete_intro_rival":
            return EmeraldIntroRivalBattleStrategy()
        return NuzlockeLevelBalancingBattleStrategy()

    def battle_strategy_for_restored_battle(self) -> BattleStrategy:
        """Resume a save-state battle without constructing the legacy strategy."""

        controller = getattr(self, "controller", None)
        selection = getattr(controller, "last_selection", None)
        objective = getattr(selection, "objective", None)
        return self._campaign_battle_strategy_for_objective(objective)

    def battle_strategy_for_run_away(self) -> BattleStrategy:
        """Use the strict escape policy for Campaign wild-battle exits."""

        return NuzlockeRunAwayStrategy()

    def on_battle_started(self, encounter) -> BattleAction | BattleStrategy:
        """Choose campaign battle policy, leaving execution to the listener."""
        self._active_battle_wild = encounter is not None
        self._active_trainer_checkpoints = ()
        context.campaign_wild_battle_intent = None
        try:
            first_battle = BattleType.FirstBattle in get_battle_type()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            first_battle = False
        if first_battle:
            # Emerald's introductory wild battle is mandatory: it is neither
            # a legal encounter nor an escapable transit battle.
            context.campaign_wild_battle_intent = WildBattleIntent.DEFEAT.value
            diagnostic_print(
                lambda: "CAMPAIGN_WILD_BATTLE_INTENT: intent='defeat' reason='mandatory FirstBattle'",
                trace=True,
                prefix="CAMPAIGN_WILD_BATTLE_INTENT",
            )
        if encounter is None:
            self._active_trainer_checkpoints = _trainer_checkpoints_for_battle_start()
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_BATTLE_HANDOFF: "
                    f"latched={tuple(getattr(checkpoint, 'checkpoint_id', None) for checkpoint in self._active_trainer_checkpoints)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_BATTLE_HANDOFF",
            )
        scheduler = getattr(self, "_readiness_scheduler", None)
        if scheduler is not None:
            scheduler.invalidate("battle_started")

        controller = getattr(self, "controller", None)
        objective = (
            controller.last_selection.objective
            if controller is not None and controller.last_selection is not None
            else None
        )
        if objective is not None:
            legality = evaluate_battle_entry(
                runtime_campaign_state().campaign_facts,
                get_party(),
                objective_id=objective.objective_id,
                rule_config=getattr(getattr(context, "nuzlocke_runtime", None), "rule_config", None),
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_LEVEL_CAP_ENTRY: "
                    f"objective={objective.objective_id!r} allowed={legality.allowed!r} "
                    f"reason={legality.reason!r} boss={getattr(legality.assessment.active_boss, 'boss_id', None)!r} "
                    f"cap={legality.assessment.level_cap!r} illegal_party_indices={legality.assessment.illegal_party_indices!r}"
                ),
                trace=True,
            )
            if not legality.allowed:
                raise BotModeError(f"Campaign battle entry rejected: {legality.reason}")

            # A known unavoidable trainer must have a legal EXP recipient plan
            # before the battle can begin.  This is separate from the current
            # level check: a level-cap-legal party can still be close enough
            # to the next level that the trainer reward would make the run
            # illegal after the battle.
            checkpoint = getattr(context, "campaign_trainer_checkpoint", None)
            if encounter is None and checkpoint is not None and legality.assessment.level_cap is not None:
                budget = build_exp_budget(
                    get_party(),
                    target_level=None,
                    level_cap=legality.assessment.level_cap,
                    unavoidable_checkpoints=(checkpoint,),
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_EXP_ENTRY: "
                        f"trainer={getattr(checkpoint, 'checkpoint_id', None)!r} "
                        f"projected_exp={budget.unavoidable_exp!r} cap_headroom={budget.cap_headroom!r} "
                        f"legal={budget.mandatory_exp_is_legal!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_EXP_ENTRY",
                )
                if budget.mandatory_exp_is_legal is not True:
                    raise BotModeError(
                        "Campaign trainer battle entry rejected: projected trainer EXP cannot be absorbed below the active cap"
                    )

            # A preparation area still owns the first legal wild encounter.
            # Resolve that ownership before selecting the training strategy;
            # otherwise the strategy would defeat a catchable encounter.
            if encounter is not None and controller is not None and controller.last_selection is not None:
                runtime = getattr(context, "nuzlocke_runtime", None)
                try:
                    location = get_player_avatar().map_group_and_number
                    legal_capture_target = bool(
                        runtime is not None and runtime.capture_target_for(location, is_wild=True, is_trainer=False)
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                    legal_capture_target = False
                if legal_capture_target and not first_battle:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_WILD_CAPTURE_HANDOFF: "
                            f"location={location!r} action='Catch' objective="
                            f"{controller.last_selection.objective.objective_id!r}"
                        ),
                        trace=False,
                    )
                    return BattleAction.Catch
            if encounter is not None:
                # Objective-specific preparation enters this branch before
                # the ordinary wild-boundary handler below. Assign the
                # training intent here so the battle planner receives
                # ``training`` even for registered preparation objectives.
                resource_policy = getattr(objective, "resource_policy", None)
                if not first_battle:
                    context.campaign_wild_battle_intent = choose_wild_battle_intent(
                        WildBattlePolicyInput(
                            preparation_requested=bool(getattr(context, "campaign_preparation_target", None)),
                            policy_intent=getattr(
                                resource_policy,
                                "wild_battle_intent",
                                WildBattleIntent.AUTO,
                            ),
                            duplicate_policy=getattr(
                                resource_policy,
                                "duplicate_encounter_policy",
                                DuplicateEncounterPolicy.RUN,
                            ),
                        )
                    ).value
            if objective.objective_id == "prepare_roxanne":
                return NuzlockeLevelBalancingBattleStrategy()
            if objective.objective_id == "defeat_roxanne":
                return RoxanneBattleStrategy()
            if objective.objective_id == "complete_intro_rival":
                return EmeraldIntroRivalBattleStrategy()
        if encounter is not None and controller is not None and controller.last_selection is not None:
            # The Nuzlocke rules projection is authoritative about whether
            # this wild battle owns the area's first legal encounter.  Make
            # that decision at the listener boundary so Campaign Progression
            # can use the campaign capture strategy; relying only on the later
            # battle-state property lets the opening turn fall through to a
            # defeating move when the projection has just crossed the battle
            # start boundary.
            runtime = getattr(context, "nuzlocke_runtime", None)
            try:
                location = get_player_avatar().map_group_and_number
                legal_capture_target = bool(
                    runtime is not None and runtime.capture_target_for(location, is_wild=True, is_trainer=False)
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                legal_capture_target = False
            if legal_capture_target and not first_battle:
                context.campaign_wild_battle_intent = WildBattleIntent.CAPTURE.value
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_WILD_CAPTURE_HANDOFF: "
                        f"location={location!r} action='Catch' objective="
                        f"{controller.last_selection.objective.objective_id!r}"
                    ),
                    trace=False,
                )
                return BattleAction.Catch
            objective = controller.last_selection.objective
            resource_policy = getattr(objective, "resource_policy", None)
            duplicate_encounter = False
            try:
                encounter_species = (
                    getattr(getattr(encounter, "pokemon", None), "species_name_for_stats", None)
                    or getattr(getattr(getattr(encounter, "pokemon", None), "species", None), "name", None)
                )
                if runtime is not None and encounter_species:
                    eligibility = runtime.capture_eligibility_for(
                        location,
                        is_wild=True,
                        is_trainer=False,
                        species=(encounter_species,),
                    )
                    duplicate_encounter = bool(getattr(eligibility, "duplicate", False))
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                duplicate_encounter = False
            policy_intent = getattr(
                resource_policy,
                "wild_battle_intent",
                WildBattleIntent.AUTO,
            )
            duplicate_policy = getattr(
                resource_policy,
                "duplicate_encounter_policy",
                DuplicateEncounterPolicy.RUN,
            )
            intent = (
                WildBattleIntent.DEFEAT
                if first_battle
                else choose_wild_battle_intent(
                    WildBattlePolicyInput(
                        preparation_requested=bool(getattr(context, "campaign_preparation_target", None)),
                        transit_requested=resource_policy is None,
                        duplicate_encounter=duplicate_encounter,
                        policy_intent=policy_intent,
                        duplicate_policy=duplicate_policy,
                    )
                )
            )
            context.campaign_wild_battle_intent = intent.value
            preparation_requested = bool(getattr(context, "campaign_preparation_target", None))
            party = None
            snapshot = None
            decision = None
            low_health_for_preparation = False
            if preparation_requested:
                try:
                    party = tuple(get_party())
                    ratios = tuple(
                        pokemon.current_hp / pokemon.total_hp
                        for pokemon in party
                        if getattr(pokemon, "total_hp", 0) > 0 and getattr(pokemon, "current_hp", 0) > 0
                    )
                    low_health_for_preparation = bool(
                        ratios
                        and min(ratios)
                        <= (getattr(getattr(self, "_readiness_policy", None), "opportunistic_hp_ratio", 0.75))
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, ZeroDivisionError):
                    low_health_for_preparation = False
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_WILD_BATTLE_INTENT: "
                    f"intent={intent.value!r} objective={getattr(objective, 'objective_id', None)!r} "
                    f"preparation={preparation_requested!r} "
                    f"duplicate={duplicate_encounter!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_WILD_BATTLE_INTENT",
            )
            if resource_policy is not None:
                if party is None:
                    party = tuple(get_party())
                snapshot = ResourceSnapshot(
                    tuple(
                        PartyResource(
                            pokemon.current_hp,
                            pokemon.total_hp,
                            getattr(pokemon.status_condition, "value", None),
                            pokemon.current_hp <= 0,
                        )
                        for pokemon in party
                    )
                )
                decision = assess_wild_encounter(resource_policy, snapshot, RouteRecovery())
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RESOURCE: "
                        f"objective={resource_policy.objective_id} encounter=wild "
                        f"hp_ratio={snapshot.worst_hp_ratio:.3f} "
                        f"missing_hp={snapshot.total_missing_hp} "
                        f"encounter_policy={resource_policy.encounters.value} "
                        f"decision={decision.value}"
                    ),
                    trace=True,
                )
                if decision is ResourceDecision.PREFER_RUN:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RESOURCE: wild intent retains planner ownership "
                            f"intent={intent.value!r} preparation={preparation_requested!r}"
                        ),
                        trace=True,
                    )
            request_recovery = getattr(controller, "request_recovery_after_wild_battle", None)
            resource_boundary_requires_recovery = (
                low_health_for_preparation
                or (
                    decision is ResourceDecision.PREFER_RUN
                    and snapshot is not None
                    and snapshot.worst_hp_ratio <= getattr(
                        getattr(self, "_readiness_policy", None), "opportunistic_hp_ratio", 0.75
                    )
                )
            )
            if encounter is not None and resource_boundary_requires_recovery and callable(request_recovery):
                request_recovery("wild_battle_resource_boundary")
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RESOURCE: wild battle completed its battle-boundary handoff "
                        f"intent={intent.value!r} preparation={preparation_requested!r} "
                        f"low_health_for_preparation={low_health_for_preparation!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_WILD_BATTLE_RECOVERY_HANDOFF",
                )
        # Ordinary campaign-owned battles should rotate the lowest-level
        # living party member into the lead.  The explicit handlers above
        # retain their specialized policies, and legal wild captures were
        # handed to the campaign capture strategy before reaching this point. This applies
        # to trainer battles too, where ``encounter`` is None.
        return self._campaign_battle_strategy_for_objective(objective)

    def capture_battle_strategy(self):
        """Return the campaign capture policy for the battle listener."""

        return NuzlockeCaptureStrategy()

    def on_spotted_by_trainer(self) -> None:
        """Recheck campaign readiness when a trainer battle becomes imminent."""

        controller = getattr(self, "controller", None)
        request_recheck = getattr(controller, "request_readiness_recheck", None)
        if callable(request_recheck):
            request_recheck("trainer_spotted")

    def on_battle_ended(self, outcome) -> None:
        """Invalidate readiness after a battle changes party or route state."""

        if outcome is BattleOutcome.Won and getattr(self, "_active_battle_wild", None) is not True:
            checkpoints = tuple(getattr(self, "_active_trainer_checkpoints", ()) or ())
            if not checkpoints:
                # This is only a compatibility fallback for a battle that was
                # entered before the latch existed. Do not use the readiness
                # checkpoint here: it may already describe the next trainer.
                checkpoint = getattr(context, "campaign_active_trainer_checkpoint", None)
                checkpoints = (checkpoint,) if checkpoint is not None else ()
            for checkpoint in checkpoints:
                _remember_defeated_trainer(checkpoint)
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_BATTLE_HANDOFF: "
                    f"outcome='Won' cleared={tuple(getattr(checkpoint, 'checkpoint_id', None) for checkpoint in checkpoints)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_BATTLE_HANDOFF",
            )
        self._active_trainer_checkpoints = ()
        context.campaign_active_trainer_checkpoint = None

        pending_fainted = getattr(self, "_pending_fainted_identities", None)
        if pending_fainted is None:
            pending_fainted = self._pending_fainted_identities = set()
        try:
            party = get_party()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            party = ()
        pending_fainted.update(
            identity
            for pokemon in party
            if getattr(pokemon, "current_hp", 0) <= 0
            and (identity := PokemonIdentity.from_pokemon(pokemon)) is not None
        )

        # Campaign tactical loops are suspended while BattleListener owns the
        # battle. Trainer battles can move/defeat an overworld object, so the
        # cached route must be discarded. Wild battles do not alter route
        # geometry; retaining their route avoids a synchronous world search
        # after every encounter. Readiness is still refreshed below in both
        # cases so HP, party, and Poké Ball changes are observed.
        wild_battle = getattr(self, "_active_battle_wild", None)
        self._active_battle_wild = None
        context.campaign_wild_battle_intent = None
        if wild_battle is not True:
            self._clear_background_route_state()
        controller = getattr(self, "controller", None)
        selection = getattr(controller, "last_selection", None)
        objective = getattr(selection, "objective", None)
        notify_battle_ended(invalidate_navigation=wild_battle is not True)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_BATTLE_ENDED: "
                f"frame={getattr(context, 'frame', None)!r} outcome={outcome!r} "
                f"objective={getattr(getattr(self.controller, 'last_selection', None), 'objective', None)!r} "
                f"facts={runtime_campaign_state().campaign_facts!r}"
            ),
            trace=True,
        )
        scheduler = getattr(self, "_readiness_scheduler", None)
        request_recheck = getattr(controller, "request_readiness_recheck", None)
        if callable(request_recheck):
            request_recheck("battle_ended")
        elif scheduler is not None:
            # Keep lightweight/test embeddings safe when they provide only a
            # scheduler and not the full campaign controller. The normal mode
            # path invalidates through the controller so the pending
            # ownership boundary and scheduler cache stay synchronized.
            scheduler.invalidate("battle_ended")
        # The battle listener owns the immediate return-to-field boundary.
        # Keep a second, mode-level event so special battle strategies and a
        # capture/recovery handoff cannot skip the overworld lead policy.
        self._lead_handoff_pending = True

    def _observe_campaign_boundary(self, controller) -> None:
        """Invalidate campaign planning when a coarse ROM boundary changes."""

        current = self._cheap_readiness_context()
        if not getattr(self, "_campaign_boundary_context_seen", False):
            self._campaign_boundary_context_seen = True
            self._last_campaign_boundary_context = current
            return

        previous = self._last_campaign_boundary_context
        self._last_campaign_boundary_context = current
        if current == previous:
            return

        reason = "campaign_boundary_changed"
        if isinstance(previous, tuple) and isinstance(current, tuple):
            if len(previous) >= 5 and len(current) >= 5 and previous[-1] != current[-1]:
                reason = "encounter_stage_changed"
            elif len(previous) > 1 and len(current) > 1 and previous[1] != current[1]:
                reason = "map_changed"
            elif len(previous) > 0 and len(current) > 0 and previous[0] != current[0]:
                reason = "game_state_changed"
            elif len(previous) > 2 and len(current) > 2 and previous[2] != current[2]:
                reason = "control_boundary_changed"
            elif len(previous) > 3 and len(current) > 3 and previous[3] != current[3]:
                reason = "pokeball_inventory_changed"
            elif len(previous) > 4 and len(current) > 4 and previous[4] != current[4]:
                reason = "party_state_changed"
            elif len(previous) > 5 and len(current) > 5 and previous[5] != current[5]:
                reason = "inventory_changed"
            elif len(previous) > 6 and len(current) > 6 and previous[6] != current[6]:
                reason = "campaign_fact_changed"

        request_recheck = getattr(controller, "request_readiness_recheck", None)
        request_refresh = getattr(controller, "request_refresh", None)
        if callable(request_recheck):
            request_recheck(reason)
        elif callable(request_refresh):
            request_refresh(reason)

    def run(self) -> Generator:
        """Yield frame boundaries while the campaign controller remains active."""

        # CampaignProgression owns campaign intent from the first observation.
        # The selected objective's executor owns frame-local emulator details.
        previous = None
        previous_phase = None
        while True:
            if getattr(self, "_whiteout_continuation_pending", False):
                self._whiteout_continuation_pending = False
                diagnostic_print(
                    "CAMPAIGN_WHITEOUT_CONTINUATION: starting PC roster replacement",
                    trace=True,
                    prefix="CAMPAIGN_WHITEOUT_CONTINUATION",
                )
                try:
                    yield from execute_whiteout_continuation()
                except Exception as error:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_WHITEOUT_CONTINUATION_FAILED: "
                            f"exception_type={type(error).__name__!r} error={error!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_WHITEOUT_CONTINUATION_FAILED",
                    )
                    if isinstance(error, RunLostError):
                        # A run with no living PC continuation is a normal
                        # autonomous terminal outcome. Reset the emulator and
                        # let the opening/campaign capability start the next
                        # fresh run; manual mode must never be the loss path.
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_RUN_LOST_RESTART: "
                                f"reason={error!r} action='emulator_reset_and_restart'"
                            ),
                            trace=True,
                            prefix="CAMPAIGN_RUN_LOST_RESTART",
                        )
                        context.message = "Run lost; restarting the campaign autonomously."
                        context.emulator.reset_held_buttons()
                        context.emulator.reset()
                        return
                    context.message = f"Whiteout continuation failed: {error}. Switched back to manual mode."
                    context.set_manual_mode()
                    return
                self.controller.request_refresh("whiteout_continuation_completed")
                self._lead_handoff_pending = True
            controller = self._campaign_controller()
            self._observe_campaign_boundary(controller)
            state = controller.step()
            marker = (state.status, state.objective_id, state.reason)
            lead_handoff_pending = getattr(self, "_lead_handoff_pending", False)
            if (
                getattr(state, "execution_phase", "CAMPAIGN") == "RECOVERY"
                and previous_phase != "RECOVERY"
            ):
                # A readiness pass may have started speculative trainer-route
                # analysis while the recovery route was still being resolved.
                # Once the controller owns the recovery stop, that analysis
                # cannot affect the mounted plan and only competes with the
                # emulator while the recovery generator is executing.
                self._clear_background_route_state()
            if (
                getattr(state, "execution_phase", "CAMPAIGN") == "CAMPAIGN"
                and state.status.value == "ready"
                and (
                    lead_handoff_pending
                    or previous_phase == "RECOVERY"
                    or (
                        previous is not None
                        and getattr(previous[0], "value", previous[0]) == "ready"
                        and previous[1] is not None
                        and state.objective_id is not None
                        and previous[1] != state.objective_id
                    )
                )
            ):
                if previous_phase == "RECOVERY":
                    self._clear_background_route_state()
                reason = (
                    "battle_or_capture_completed"
                    if lead_handoff_pending
                    else "recovery_completed"
                    if previous_phase == "RECOVERY"
                    else "campaign_objective_handoff"
                )
                self._lead_handoff_pending = False
                # Trainer exposure has its own final staging boundary. The
                # general field-lead policy must not rotate the party here,
                # between trainer readiness and the engagement route.
                trainer_exposure_owner = _campaign_goal_owns_trainer_exposure(
                    getattr(state, "tactical_goal", None)
                )
                if trainer_exposure_owner:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                            f"rotated=False reason={reason!r} "
                            "deferred_to='trainer_exposure_staging'"
                        ),
                        trace=True,
                    )
                    previous = marker
                    previous_phase = getattr(state, "execution_phase", "CAMPAIGN")
                    continue
                try:
                    leveling_strategy = NuzlockeLevelBalancingBattleStrategy()
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
                    # Opening campaign objectives can hand off before the
                    # starter has been inserted into the party. The lead
                    # policy has nothing to enforce at that boundary; defer
                    # it until the next battle/recovery event.
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                            f"rotated=False reason={reason!r} "
                            f"strategy_unavailable={type(error).__name__!r}"
                        ),
                        trace=True,
                    )
                else:
                    objective = getattr(getattr(state, "selection", None), "objective", None)
                    field_context = build_campaign_field_lead_context(
                        objective,
                        campaign_state=runtime_campaign_state(),
                    )
                    yield from ensure_campaign_field_lead(
                        leveling_strategy,
                        field_context,
                        reason=reason,
                    )
            if marker != previous:
                diagnostic_print(
                    f"CAMPAIGN: status={state.status.value} objective={state.objective_id!r} "
                    f"reason={state.reason!r}",
                    trace=True,
                )
                previous = marker
            previous_phase = getattr(state, "execution_phase", "CAMPAIGN")
            if state.status.value == "complete":
                context.campaign_status = CampaignStatus.complete_status()
            elif state.status.value != "ready" and state.objective_id is None:
                context.campaign_status = CampaignStatus(None, None, f"Campaign {state.status.value}")
            # Startup observations can be transiently unavailable while the
            # ROM is booting. Keep the campaign controller mounted so the
            # next frame can re-observe and plan; UNKNOWN/BLOCKED are not
            # emulator phases and are not ownership handoffs.
            if state.status.value in {"unknown", "blocked"}:
                yield
                continue
            if state.status.value != "ready":
                return
            yield
