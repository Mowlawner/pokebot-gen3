from types import SimpleNamespace

from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    CampaignReadinessPolicy,
    ReadinessDecision,
    ReadinessReason,
    build_progression_readiness_diagnostic,
    evaluate_progression_readiness,
)
from modules.nuzlocke.resource_policy import PartyResource, ResourceObservationStatus, ResourceSnapshot, RouteRecovery
from modules.navigation import IntermediateRouteAnalysis, RouteAnalysis
from modules.goals import ReachLocation


def snapshot(party, available=True):
    return SimpleNamespace(
        party=tuple(party),
        party_available=available,
        player_available=True,
        player=SimpleNamespace(map_group=1, map_number=2, coordinates=(3, 4)),
        game_state=SimpleNamespace(name="OVERWORLD"),
    )


def mon(index, hp, maximum, *, fainted=False, status="none"):
    return SimpleNamespace(
        party_index=index,
        species=f"MON{index}",
        current_hp=hp,
        max_hp=maximum,
        fainted=fainted,
        status=status,
    )


def test_party_health_and_usable_state_are_reported_without_policy():
    diagnostic = build_progression_readiness_diagnostic(
        snapshot((mon(0, 20, 20), mon(1, 2, 25, status="poison"))),
        objective_id="reach_petalburg",
        objective_status="ready",
        navigation_goal="goal",
    )
    assert [(p.current_hp, p.max_hp, p.hp_ratio, p.usable) for p in diagnostic.party] == [
        (20, 20, 1.0, True),
        (2, 25, 0.08, True),
    ]
    assert diagnostic.has_usable_pokemon is True
    assert diagnostic.readiness == "NOT_EVALUATED"
    assert diagnostic.navigation_goal == "goal"


def test_fainted_and_no_usable_party_are_distinguished():
    diagnostic = build_progression_readiness_diagnostic(
        snapshot((mon(0, 0, 20, fainted=True), mon(1, 0, 25, fainted=True)))
    )
    assert diagnostic.has_usable_pokemon is False
    assert all(p.fainted and not p.usable for p in diagnostic.party)


def test_unavailable_party_is_reported_as_unavailable():
    diagnostic = build_progression_readiness_diagnostic(snapshot((), available=False))
    assert diagnostic.party == ()
    assert diagnostic.has_usable_pokemon is None


def test_valid_resource_party_fills_transient_snapshot_gap():
    diagnostic = build_progression_readiness_diagnostic(
        snapshot((), available=False),
        resource_snapshot=ResourceSnapshot(
            (PartyResource(20, 23),),
            observation_status=ResourceObservationStatus.VALID,
        ),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        overworld_availability=Availability.KNOWN,
        recovery_availability=Availability.KNOWN,
    )

    assert diagnostic.party_availability is Availability.KNOWN
    assert diagnostic.party_count == 1
    assert diagnostic.lowest_hp_ratio == 20 / 23
    assert CampaignReadinessPolicy().evaluate(diagnostic).decision is ReadinessDecision.CONTINUE


def test_trainer_hazard_and_distance_relationships_are_observational():
    obj = SimpleNamespace(
        local_id=7,
        trainer_type="Normal",
        trainer_range=3,
        trainer_defeated=False,
        location=((1, 2), (5, 4)),
    )
    overworld = SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=(obj,))
    diagnostic = build_progression_readiness_diagnostic(snapshot((mon(0, 10, 20),)), overworld=overworld)
    hazard = diagnostic.trainers[0]
    assert (hazard.object_id, hazard.trainer_type, hazard.trainer_range) == (7, "Normal", 3)
    assert hazard.distance == 2
    assert hazard.relationship == "nearby"
    assert hazard.path_relevant is None


def test_defeated_and_unavailable_distance_context_are_preserved():
    obj = SimpleNamespace(
        local_id=8,
        trainer_type="Moving",
        trainer_range=2,
        trainer_defeated=True,
        location=((9, 9), (1, 1)),
    )
    overworld = SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=(obj,))
    hazard = build_progression_readiness_diagnostic(snapshot((mon(0, 10, 20),)), overworld=overworld).trainers[0]
    assert hazard.defeated is True
    assert hazard.distance is None
    assert hazard.relationship == "map_present"


def test_non_trainers_and_missing_hazards_are_not_fabricated():
    obj = SimpleNamespace(
        local_id=9,
        trainer_type=None,
        trainer_range=None,
        trainer_defeated=None,
        location=((1, 2), (5, 4)),
    )
    overworld = SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=(obj,))
    diagnostic = build_progression_readiness_diagnostic(snapshot((mon(0, 10, 20),)), overworld=overworld)
    assert diagnostic.trainers == ()


def test_availability_and_aggregates_are_policy_neutral_and_serializable():
    diagnostic = build_progression_readiness_diagnostic(
        snapshot((mon(0, 2, 25), mon(1, 0, 20, fainted=True))),
        navigation_goal="reach",
        recovery_availability=Availability.UNKNOWN,
    )
    assert diagnostic.party_count == 2
    assert diagnostic.usable_count == 1
    assert diagnostic.fainted_count == 1
    assert diagnostic.lowest_hp_ratio == 2 / 25
    assert diagnostic.recovery_availability is Availability.UNKNOWN
    encoded = diagnostic.as_dict()
    assert encoded["availability"]["party"] == "known"
    assert encoded["availability"]["recovery"] == "unknown"
    assert encoded["readiness"] == "NOT_EVALUATED"


def readiness(hp, *, trainer=None, recovery=Availability.KNOWN, party_available=True):
    objects = (
        ()
        if trainer is None
        else (
            SimpleNamespace(
                local_id=1,
                trainer_type="Normal",
                trainer_range=trainer[1],
                trainer_defeated=trainer[2],
                location=((1, 2), trainer[0]),
            ),
        )
    )
    world = SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=objects)
    return build_progression_readiness_diagnostic(
        snapshot((mon(0, hp, 20),), available=party_available),
        overworld=world,
        overworld_availability=Availability.KNOWN,
        recovery_availability=recovery,
    )


def test_policy_healthy_and_low_hp_without_imminent_trainer_continue():
    policy = CampaignReadinessPolicy()
    assert policy.evaluate(readiness(20)).decision is ReadinessDecision.CONTINUE
    result = policy.evaluate(readiness(2))
    assert result == type(result)(ReadinessDecision.RECOVER, ReadinessReason.CRITICAL_PARTY_HP)


def test_policy_recovers_a_poisoned_party_before_overworld_steps_drain_hp():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 20, 20, status="poisoned"),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery=RouteRecovery(center_available=True, distance_to_center=8, safe_to_reach_center=True),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
    )
    result = CampaignReadinessPolicy().evaluate(value)
    assert result == type(result)(ReadinessDecision.RECOVER, ReadinessReason.POISONED_PARTY)


def test_policy_critical_hp_at_or_below_threshold_before_trainer_recovers():
    policy = CampaignReadinessPolicy()
    trainer = ((4, 4), 4, False)  # distance 1, within range
    assert policy.evaluate(readiness(4, trainer=trainer)).decision is ReadinessDecision.RECOVER
    assert policy.evaluate(readiness(4, trainer=trainer)).reason is ReadinessReason.CRITICAL_PARTY_HP
    # Opportunistic recovery must not be bypassed merely because a trainer is
    # imminent.  With no route analysis, the safe result is to re-observe.
    assert policy.evaluate(readiness(5, trainer=trainer)).decision is ReadinessDecision.UNKNOWN


def test_policy_trainer_provenance_and_recovery_provenance_are_explicit():
    policy = CampaignReadinessPolicy()
    assert policy.evaluate(readiness(2, trainer=None)).decision is ReadinessDecision.RECOVER
    no_world = build_progression_readiness_diagnostic(
        snapshot((mon(0, 2, 20),)), recovery_availability=Availability.KNOWN
    )
    assert policy.evaluate(no_world).reason is ReadinessReason.OVERWORLD_UNAVAILABLE
    trainer = ((4, 4), 4, False)
    unavailable = policy.evaluate(readiness(2, trainer=trainer, recovery=Availability.UNAVAILABLE))
    assert unavailable == type(unavailable)(ReadinessDecision.UNKNOWN, ReadinessReason.RECOVERY_UNAVAILABLE)
    unknown = policy.evaluate(readiness(2, trainer=trainer, recovery=Availability.UNKNOWN))
    assert unknown.reason is ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN


def test_policy_handles_fainted_or_unavailable_party_and_defeated_trainer():
    policy = CampaignReadinessPolicy()
    fainted = build_progression_readiness_diagnostic(
        snapshot((mon(0, 0, 20, fainted=True),)),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
    )
    assert policy.evaluate(fainted).decision is ReadinessDecision.RECOVER
    unavailable = policy.evaluate(readiness(2, party_available=False))
    assert unavailable.reason is ReadinessReason.PARTY_INFORMATION_UNKNOWN
    defeated = policy.evaluate(readiness(2, trainer=((4, 4), 4, True)))
    assert defeated.decision is ReadinessDecision.RECOVER


def test_policy_evaluation_is_deterministic_and_does_not_mutate_input():
    value = readiness(2, trainer=((4, 4), 4, False))
    first = evaluate_progression_readiness(value)
    second = evaluate_progression_readiness(value)
    assert first == second
    assert value.readiness_decision is None
    assert first.readiness_decision is ReadinessDecision.RECOVER


def test_policy_opportunistically_recovers_meaningfully_damaged_party_at_nearby_center():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 9, 20),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery=RouteRecovery(center_available=True, distance_to_center=8, safe_to_reach_center=True),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis=route_analysis(8),
    )
    result = CampaignReadinessPolicy().evaluate(value)
    assert result.decision is ReadinessDecision.RECOVER
    assert result.reason is ReadinessReason.OPPORTUNISTIC_RECOVERY


def test_policy_rechecks_before_imminent_trainer_for_opportunistic_recovery():
    value = readiness(
        9,
        trainer=((4, 4), 4, False),
        recovery=Availability.KNOWN,
    )
    result = CampaignReadinessPolicy().evaluate(value)
    assert result.decision is ReadinessDecision.UNKNOWN
    assert result.reason is ReadinessReason.OPPORTUNISTIC_ROUTE_UNAVAILABLE


def test_policy_does_not_opportunistically_recover_for_minor_damage_or_long_detour():
    world = SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=())
    near = RouteRecovery(center_available=True, distance_to_center=8, safe_to_reach_center=True)
    far = RouteRecovery(center_available=True, distance_to_center=21, safe_to_reach_center=True)
    minor = build_progression_readiness_diagnostic(
        snapshot((mon(0, 18, 20),)),
        overworld=world,
        recovery=near,
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
    )
    distant = build_progression_readiness_diagnostic(
        snapshot((mon(0, 9, 20),)),
        overworld=world,
        recovery=far,
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
    )
    assert CampaignReadinessPolicy().evaluate(minor).decision is ReadinessDecision.CONTINUE
    assert CampaignReadinessPolicy().evaluate(distant).decision is ReadinessDecision.UNKNOWN


def test_policy_recovers_capability_boundary_without_normal_route_goal():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 10, 25),)),
        objective_id="receive_pokedex",
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery=RouteRecovery(center_available=True, distance_to_center=38, safe_to_reach_center=True),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        targetless=True,
    )
    result = CampaignReadinessPolicy().evaluate(value)
    assert result.decision is ReadinessDecision.RECOVER
    assert result.reason is ReadinessReason.OPPORTUNISTIC_RECOVERY


def route_analysis(*detours):
    goal = ReachLocation(((1, 2), (8, 8)))
    return RouteAnalysis(
        goal,
        None,
        100,
        tuple(
            IntermediateRouteAnalysis(
                ReachLocation(((1, 2), (index, 0))),
                100 + detour,
                detour,
                True,
            )
            for index, detour in enumerate(detours)
        ),
    )


def test_policy_selects_minimum_route_detour_not_absolute_distance():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis=route_analysis(40, 15),
    )
    result = CampaignReadinessPolicy().evaluate(value)
    assert result.decision is ReadinessDecision.RECOVER


def test_policy_does_not_heal_high_hp_for_cheap_detour():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 20, 23),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis=route_analysis(49),
    )
    result = CampaignReadinessPolicy().evaluate(value)
    assert result.decision is ReadinessDecision.CONTINUE
    assert result.reason is ReadinessReason.PARTY_HEALTHY


def test_policy_continues_when_route_detour_exceeds_threshold():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis=route_analysis(51),
    )
    assert CampaignReadinessPolicy().evaluate(value).decision is ReadinessDecision.CONTINUE


def test_policy_does_not_fallback_to_absolute_distance_without_route_analysis():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery=RouteRecovery(center_available=True, distance_to_center=1, safe_to_reach_center=True),
        recovery_availability=Availability.KNOWN,
    )
    assert CampaignReadinessPolicy().evaluate(value).decision is ReadinessDecision.UNKNOWN


def test_policy_uses_completed_nearby_recovery_route_when_detour_analysis_is_unavailable():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery=RouteRecovery(
            center_available=True,
            distance_to_center=12,
            safe_to_reach_center=True,
            route=object(),
        ),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
    )

    result = CampaignReadinessPolicy().evaluate(value)

    assert result.decision is ReadinessDecision.RECOVER
    assert result.reason is ReadinessReason.OPPORTUNISTIC_RECOVERY


def test_policy_uses_recovery_route_when_route_composition_has_no_reachable_candidate():
    value = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=()),
        recovery=RouteRecovery(
            center_available=True,
            distance_to_center=12,
            safe_to_reach_center=True,
            route=object(),
        ),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis=RouteAnalysis(ReachLocation(((1, 2), (8, 8))), object(), 100, ()),
    )

    result = CampaignReadinessPolicy().evaluate(value)

    assert result.decision is ReadinessDecision.RECOVER
    assert result.reason is ReadinessReason.OPPORTUNISTIC_RECOVERY


def test_policy_keeps_moving_or_recovers_when_route_analysis_times_out():
    world = SimpleNamespace(map_id=(1, 2), player_coordinates=(3, 4), objects=())
    near = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=world,
        recovery=RouteRecovery(center_available=True, distance_to_center=8, safe_to_reach_center=True),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis_timed_out=True,
    )
    result = CampaignReadinessPolicy().evaluate(near)
    assert result.decision is ReadinessDecision.RECOVER
    assert result.reason is ReadinessReason.OPPORTUNISTIC_RECOVERY

    far = build_progression_readiness_diagnostic(
        snapshot((mon(0, 12, 20),)),
        overworld=world,
        recovery=RouteRecovery(center_available=True, distance_to_center=80, safe_to_reach_center=True),
        recovery_availability=Availability.KNOWN,
        overworld_availability=Availability.KNOWN,
        route_analysis_timed_out=True,
    )
    result = CampaignReadinessPolicy().evaluate(far)
    assert result.decision is ReadinessDecision.CONTINUE
    assert result.reason is ReadinessReason.OPPORTUNISTIC_ROUTE_UNAVAILABLE
