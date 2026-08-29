# Campaign Recovery Planning Audit

Date: 2026-08-29

## Executive summary

Planned recovery is feasible and is the better architecture for this campaign
system. The repository already contains most of the difficult primitives:

- `RouteCostAnalyzer` can compare a normal route with a route through an
  intermediate destination.
- `NavigationWorld` and `plan_with_world_navigation` can compose routes across
  map warps.
- campaign objectives already carry destinations, tactical goals, priorities,
  and resource policy metadata.
- the controller already has a safety-objective ownership concept and can
  resume an interrupted parent objective.
- the recovery runtime can navigate to a Center and complete the nurse flow.

The first integration step is now present. `CampaignPlan` and `RecoveryStop`
capture a selected reachable healing destination, its parent objective, route,
and reason, and the controller advances the plan after the recovery generator
completes. Recovery selection is still initiated by a readiness-triggered
side-channel, however: the dynamic planner does not yet choose recovery before
the readiness path, and execution still mounts a separate recovery generator.

That partial integration improves ownership bookkeeping but does not yet move
campaign intent fully into planning. Readiness still understands the emulator
boundary, and dialogue/transition guards remain necessary because recovery can
take control at arbitrary frame boundaries. A complete planned waypoint would
still move that decision into campaign planning and leave dialogue/action
ownership with the normal execution layers.

## Current architecture and observed failure

The current flow is approximately:

```text
select static campaign objective
    -> adapt objective to tactical goal
        -> run readiness provider every controller refresh
            -> evaluate HP and route availability
                -> build a transient CampaignPlan on RECOVER
                    -> mount the separate recovery generator
```

The relevant implementation points are:

- `modules/nuzlocke/campaign_objectives.py`: static objective model and
  deterministic selector. `select_campaign_objective()` selects the first
  incomplete objective; it does not add dynamic recovery work.
- `modules/nuzlocke/campaign_planner.py`: now provides `CampaignPlan` and
  `RecoveryStop`; `build_campaign_plan()` converts a readiness recovery result
  and its reachable route analysis into one transient stop.
- `modules/modes/campaign.py`: builds readiness input and route analysis. It
  enumerates Center candidates and calls `RouteCostAnalyzer`, but returns the
  result as diagnostic/policy input rather than changing the selected plan.
- `modules/nuzlocke/readiness_diagnostics.py`: pure HP/recovery policy. It can
  return `RECOVER`, but that result is not a route waypoint.
- `modules/nuzlocke/campaign_controller.py`: stores and advances the transient
  plan, but still turns `RECOVER` into a separate recovery generator and
  temporarily replaces the campaign tactical loop.
- `modules/nuzlocke/resource_runtime.py`: `execute_planned_recovery()` now
  receives the selected destination and healing source, navigates to that
  destination, discovers the interior interaction, and verifies restoration.
  The older nearest-Center recovery path remains for compatibility.
- `modules/navigation.py`: already models intermediate route analysis through
  `RouteCostAnalyzer` and returns first/continuation plans.

The live trace demonstrated the failure precisely. Immediately after the
introductory rival battle, the party was at `10/25 HP`; Oldale Center was
reachable in 38 steps. However, the game was still executing the battle-return
script, so readiness returned `UNKNOWN / OVERWORLD_UNAVAILABLE`. The tactical
objective continued across that boundary, and no recovery waypoint was queued.

The later Lab case exposed a second independent problem: legacy
`calculate_path()` cannot cross warps. The world planner can now find the Lab
to Oldale route, and the recovery observation fallback uses that world planner
when the legacy pathfinder raises. This route-discovery fix is separate from
the still-incomplete planning/ownership integration.

## Feasibility assessment

### Route composition: high feasibility

`RouteCostAnalyzer.analyze()` already accepts intermediate destinations and
calculates:

1. the normal route to the campaign goal;
2. a route to an intermediate destination;
3. a continuation route from that destination to the goal;
4. total cost and detour.

This is directly applicable to a Pokémon Center waypoint. The existing world
planner is also the appropriate execution planner because it handles map
connections and warps.

### Objective integration: partial implementation

The repository now has the planned execution envelope proposed below. The
`CampaignObjective` model remains the permanent parent objective, while
`CampaignPlan` holds a transient `RecoveryStop`; the normal campaign selector
still does not inject recovery work before readiness evaluates it.

The existing `HEAL_PARTY` safety objective and `interrupted_objective_id`
fields continue to support parent/safety ownership. The current implementation
is therefore a compatibility bridge toward the cleaner envelope rather than
the completed dynamic planner:

```text
CampaignPlan(parent=reach_littleroot,
             stops=[RecoveryStop(OldaleCenter),
                    ParentObjective(reach_littleroot)])
```

This preserves the parent objective and makes recovery an explicit first stop.

### Emulator interaction safety: high feasibility

Dialogue detection remains necessary for the executor, but it no longer needs
to decide campaign intent. The route executor can wait for a stable,
controllable overworld before issuing movement, while the interaction
executor remains responsible for advancing dialogue and Center menus.

The existing `observe_interaction()` classifier is suitable as an execution
boundary. The readiness policy should not need to mount a generator merely to
avoid pressing A during `WaitForAorBPress`.

### Recovery execution: medium feasibility, partially implemented

The planned recovery runtime now receives the selected destination and healing
source, so it does not reselect a different Center during its route phase. It
still discovers the interior interaction from a fresh observation and does not
consume the planned `RecoveryStop.route` directly; the route is recomputed by
the normal navigation executor. The older nearest-Center runtime remains in
use for compatibility paths.

The Lab-to-Oldale route also shows why all recovery route validation and
execution should use one world-navigation abstraction. Validation now falls
back to the world planner when the legacy pathfinder cannot cross a warp, but
the two paths still coexist and can produce mismatched assumptions.

### Performance: medium risk, manageable

World route analysis can be expensive, especially with debug tracing and
cross-map searches. It should run when the parent objective or relevant route
context changes, not once per frame. The existing readiness scheduler and
navigation caches can be reused, but the planned route must be invalidated on
map changes, dynamic blockers, battle completion, party HP changes, and
recovery completion.

## Proposed implementation and current status

### 1. Introduce a pure recovery recommendation — partial

Keep HP policy independent of emulator input. Change its output from an
implicit instruction to a structured recommendation containing:

- recommendation: none, opportunistic recovery, or mandatory recovery;
- selected recovery source;
- route cost to source;
- continuation cost to the parent objective;
- detour cost;
- reason and confidence/availability state.

The current `RouteRecovery`, `RouteAnalysis`, and `ReadinessResult` types are
the foundations. `build_campaign_plan()` now filters unreachable candidates
and selects the lowest-detour reachable candidate, but the recommendation is
still produced only after readiness has already made the recovery decision.
Avoid duplicating route costs in unrelated structures.

Critical HP should select a recovery stop whenever a safe executable source is
known. Opportunistic HP should select it when the composed route detour is
within the configured threshold. An imminent trainer can remain a policy
input, but its meaning should be explicit: it may suppress optional recovery,
not override critical recovery.

### 2. Add a dynamic campaign-plan layer — partial

After selecting the required parent objective and before adapting it to a
tactical goal:

1. determine the parent objective's progression destination;
2. enumerate currently known full-party recovery sources;
3. use the world planner to evaluate source-to-parent route composition;
4. apply the pure recovery recommendation;
5. if selected, return a plan whose next stop is the Center and whose parent
   remains the campaign objective.

The current `CampaignPlan` is a planner result and does not add a permanent
campaign fact, but it is still constructed inside the readiness-triggered
controller path. It has not yet replaced that side-channel with a campaign
planner that owns recovery selection before tactical adaptation.

Conceptually:

```text
parent objective: reach Littleroot
normal route:     current -> Littleroot
candidate route:  current -> Oldale Center -> Littleroot
policy:           recover opportunistically
planned stops:    Oldale Center, Littleroot
```

### 3. Make recovery stops first-class execution targets — partial

`RecoveryStop` now represents most of the required data:

- source identity (`PokemonCenter.OldaleTown`);
- destination location/goal;
- parent objective ID;
- completion predicate: party restored;
- execution capability: navigate, enter Center, heal, verify (currently
  supplied by the recovery factory rather than stored on the stop).

The controller stores the plan and advances its active stop after the recovery
generator completes. It still changes from campaign control to a separate
recovery generator, so the final ownership boundary described here is not yet
implemented.

On completion, the planner resumes the parent with a fresh route from the
actual post-healing location. If the stop becomes unreachable, the plan is
invalidated and readiness is recomputed; it must not silently continue under
the assumption that the party healed.

### 4. Unify recovery route validation and execution — partial

Recovery observation now falls back to
`NavigationWorld`/`plan_with_world_navigation` when legacy pathfinding cannot
cross a warp, and planned execution uses the same world-navigation family to
reach the selected destination. A single typed route-planning entry point for
both observation and execution does not yet exist.

The current special-case Center registry can remain as a candidate source
registry, but candidate registration must not itself imply reachability. In
particular, Birch's Lab should be considered available only when the world
planner can route out through the appropriate warp sequence.

### 5. Reduce readiness guards to execution boundaries — outstanding

Once recovery is a planned stop, readiness should not inspect or control
dialogue. The executor still needs a small safety rule:

- do not emit movement during dialogue, choice input, battle, or an active
  transition;
- resume the already-selected plan when the overworld becomes controllable.

That is materially narrower than using dialogue state to decide whether the
campaign should recover.

## Migration sequence

1. **Implemented:** add pure plan/recovery-stop data types and unit tests.
2. **Outstanding:** extract the existing `RouteCostAnalyzer` recovery
   candidate calculation from `campaign.py` into the planner layer.
3. **Partial:** add selector/planner tests for normal, opportunistic, critical,
   unavailable, and stale-observation cases. Current tests cover explicit plan
   construction and unreachable-candidate filtering, but not the full live
   selector path.
4. **Partial:** add a dynamic recovery stop to the controller while retaining
   the current side-channel as a compatibility path. The controller now stores
   and completes a `CampaignPlan`, but readiness still initiates recovery.
5. **Partial:** route the recovery stop through the world planner and pass the
   selected Center into execution. The destination/source are passed; the
   planned route itself is not consumed directly.
6. **Outstanding:** remove the readiness-triggered recovery generator and its
   deferred-recovery latch after live validation confirms equivalent behavior.
7. **Outstanding:** validate with the introductory rival profile, a route with
   an on-route Center, a critical-HP case, a no-Center route, and a
   dialogue-heavy case.

## Risks and mitigations

### Risk: recovery oscillates with the parent objective

Mitigation: keep the selected recovery stop active until its completion
predicate is verified. Do not re-run parent selection while the stop is
incomplete.

### Risk: stale route assumptions after a warp or dynamic blocker

Mitigation: invalidate the plan on map/transition signature changes and replan
from the observed location. A route cost is advisory; execution must verify
each transition.

### Risk: critical party health but no reachable Center

Mitigation: distinguish `known unavailable` from `observation unavailable`.
Critical policy may then choose a bag item, preserve/run behavior, or a
terminal safety response according to explicit rules.

### Risk: route-analysis cost causes visible lag

Mitigation: cache by map, coordinates, parent goal, route graph version, and
party/navigation capabilities; compute on meaningful state changes rather than
every frame. Use stutter traces to verify analysis is not on the hot path.

### Risk: Center interaction completes visually but party is not restored

Mitigation: retain the existing post-interaction party verification and make
the stop incomplete/failed if restoration is not observed.

## Acceptance criteria

The implementation should satisfy all of the following. Current status is
included so this section does not imply that the proposal is already complete:

1. **Partial:** with a party below the opportunistic threshold and an
   acceptable Center detour, the selected plan explicitly visits the Center
   before the parent objective. The pure plan builder does this, but selection
   still originates in readiness.
2. **Partial:** with a critical party member and a known safe Center route,
   recovery is selected regardless of optional detour policy. The policy and
   reachable-candidate filtering exist, but this is not live-validated.
3. **Partial:** dialogue, choices, battle teardown, and map transitions
   continue to be advanced by their existing owners; no readiness path presses
   their input. Unit ownership guards exist, but recovery still takes over a
   separate controller loop.
4. **Partial:** after healing is verified, the parent objective resumes from
   the actual current location. Controller tests cover release/resumption;
   emulator validation remains outstanding.
5. **Unit-tested:** a Center candidate that cannot be reached by the
   executable world planner is not treated as available.
6. **Outstanding:** the introductory-rival profile reaches the Center before
   returning to the Lab when the injured-party condition is present.
7. **Outstanding:** stutter tracing shows route analysis is bounded/cached and
   does not create the post-battle speed regression.

## Conclusion

The proposed system remains feasible without replacing the campaign
architecture. A first waypoint/envelope implementation and parent-plan
resumption now exist, while recovery selection still begins in controller-side
readiness and execution still uses a separate recovery generator. The current
guards are therefore still evidence of an incomplete ownership boundary, not
evidence that dynamic recovery is inherently unsafe.

The next highest-value work is to move recovery selection into campaign-plan
construction, pass and execute the planned route directly, unify route
validation and execution, and live-validate the critical/dialogue-heavy
scenarios. This should eliminate the remaining readiness/dialogue coupling
while preserving the desired opportunistic and critical-health behavior.
