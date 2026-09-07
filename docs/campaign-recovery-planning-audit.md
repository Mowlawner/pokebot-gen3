# Campaign Recovery Planning Audit

Date: 2026-09-03

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

## Recovery cleanup implementation status

The requested recovery cleanup is implemented. It is deliberately narrower than
the long-term ownership migration described later in this document:

1. `recover_before_completion` has been removed from the resource schema,
   campaign objectives, policy evaluation, and the post-battle recovery latch.
   Recovery is now governed by the ordinary campaign readiness policy, so the
   completed objective cannot carry a special recovery instruction across the
   rival battle handoff.
2. Recovery candidate validation is bounded. The observer evaluates at most
   four catalog candidates by default, and each speculative world search is
   limited to 4,000 expansions and 512 route-cost units. After a candidate is
   found, later searches receive its lexicographic
   `(encounter_opportunities, total_route_cost)` ceiling. A budget exhaustion is
   recorded as a candidate failure and does not stall the controller.
3. Readiness is mounted from execution shape and campaign readiness, not from
   the `receive_pokedex` objective ID or objective-specific resource metadata.
   Tactical goals, destination-bearing capabilities, and targetless capabilities
   after the Pokédex boundary use the same readiness policy. Targetless
   capabilities reuse the selected recovery route directly, avoiding a second
   global route-composition search.

The remaining limitation is architectural rather than a freeze-safety issue:
the controller still executes a selected recovery stop through its transient
recovery generator. Moving recovery selection and execution entirely into the
normal campaign-plan executor remains a separate follow-up.

## P0 closeout status

The implementation pass closed the deterministic seams needed for P0, but not
the final live exit criteria:

- Runtime startup and emulator-reset hydration now replay durable campaign and
  rules history. Battle-start encounter eligibility is persisted at the
  observation boundary, and unit tests cover hydration, legacy activation,
  deferred save-boundary commits, ephemeral sequence rebasing, and reset
  preservation. Live runtime writes are intentionally deferred until an
  in-game save, manual save state, or enabled automatic shutdown save state.
  `--no-save-state` suppresses only the automatic shutdown boundary.
- Emerald capability ownership now comes from the ROM registry for Devon
  Goods, Rustboro, and Roxanne. Devon's left/right decompilation script
  symbols and Roxanne's gym script symbol are bound explicitly. Objective
  planning tests progress through `DEFEATED_RUSTBORO_GYM`. The optional Devon
  Goods chain is not a literal Roxanne prerequisite; Roxanne availability is
  the Rustboro-visited/not-yet-defeated ROM fact, while gym execution also
  requires the registered party preparation level.
- The fresh no-save live contract remains verified through Petalburg City. A
  raw-state live trace now proves the Petalburg gate owns the blocked tile,
  `ShowGymToPlayer` runs, Wally's tutorial battle runs, and
  `ReturnFromWallyTutorial` completes before the campaign mounts
  `recover_devon_goods`. The Devon Goods completion and first badge have not
  yet been verified live.
- The full default test suite passes 1,354 tests, with 38 emulator-tier tests
  skipped by default. The focused readiness/campaign-recovery tier passes 162
  tests, and the route-recovery observation tier passes 13 tests.
- Process restart/resume has deterministic unit coverage but still needs a
  ROM-backed demonstration. The repaired ignored `test_begin_nuzlocke` profile
  now has 18 contiguous events, matching provenance, with `NuzlockeStarted` at
  sequence 18. The earlier corruption note is stale; event-store corruption
  detection remains strict.

P0 should therefore remain open. The Petalburg script boundary is no longer
the blocker, and preparation now has a unit-tested trainer-only handoff after
an area's first encounter resolves. A disposable replay from
`test_return_to_lab_after_rival_fight` completed the Pokémon Center nurse flow:
the render-boundary rescue issued a fresh B pulse, released the held B before
the actionable A, and returned to overworld movement. The reported final-box
stall is not reproduced from the current checkpoint.

The Route 103 investigation also found no avoidable route detour: the selected
rival activation positions require 11 encounterable grass moves, versus 12
for the other observed candidates. The two opening Growl actions are an
intentional `EmeraldIntroRivalBattleStrategy` policy and now have focused
coverage. A controlled 1x replay in this environment reported sound
initialization failure and then settled at roughly 20.6 ms per frame (about
48.5 FPS) from the beginning; the emulator core and controller work were each
below 1 ms. This identifies the sleep-based no-audio limiter as the likely
explanation for the apparent 80% speed, rather than a Route 103 speed setting.
A working audio device or a more precise fallback limiter is the appropriate
performance follow-up. Recovery route analysis now has finite expansion and
route-cost budgets, and candidate searches are pruned by the best incumbent, so
an unreachable or future healing source cannot monopolize the frame loop.

The immediate closeout phase is therefore a focused ROM-boundary pass: finish
the Devon-to-Rustboro-to-Roxanne path, produce the authoritative first-badge
fact, capture that path as a fixture, and then prove restart/resume against the
fixture. Retained Rustboro checkpoints currently stall inside mGBA's
`run_frame` before the first runtime update, so they cannot serve as live
evidence. The remaining planned recovery-envelope work described below is the
next architectural phase after that evidence is green; the bounded-search and
universal-readiness cleanup is already complete.

## ROM-truth and state-ownership audit

The `test_ng` contract exposed a separate correctness issue from recovery:
the profile is intentionally metadata-only so its first run starts at a fresh
ROM title screen. mGBA still needs a save-data backend and creates a
placeholder `current_save.sav`; in the observed case that file was 128 KiB of
`0xff`, not a valid in-game save. Reading event flags from that buffer made
`first_badge_obtained` appear true and produced the GUI's `Campaign complete`
status.

The fix establishes an explicit lifecycle boundary in
`CampaignObservationSnapshot`:

- `FRESH_START` is emitted only for an observed title/main-menu lifecycle
  before a save is loaded or a new game has started. Save-backed flags and
  variables remain unavailable, even if their raw placeholder bytes decode as
  true or `0xffff`.
- `ACTIVE` permits save-backed campaign facts and provenance reconciliation.
- `UNAVAILABLE` means that the lifecycle or read itself could not be
  established.

The planner has one corresponding exception: a `FRESH_START` observation may
mount the opening setup capability (`set_text_speed`) so startup can proceed.
It cannot satisfy the terminal goal. This is an observation boundary, not a
remembered phase or synthetic campaign milestone.

The intended campaign model is a recursive list of goals/tasks, not a fixed
linear state machine. A goal describes an observable game condition and its
producer capability; the resolver recursively expands unmet prerequisites and
returns the currently available required and optional work. Most conditions
should be read from the ROM's flags, variables, map, party, inventory, and
other live structures. Durable projections are appropriate only for facts the
ROM does not preserve, such as encounter ownership, captures, faint history,
and whiteouts. Execution state may remember a mounted generator or an
in-progress interaction, but it must never become campaign completion state.

Petalburg is the concrete example that the placeholder graph currently hides:

- Pokédex receipt changes the ROM gate that lets the player leave Oldale
  westward toward Petalburg; simply reaching Petalburg is not itself the
  meaningful objective.
- Norman's Wally tutorial must occur in Petalburg. Its ROM flag/variable
  update is the prerequisite that opens the Petalburg Woods progression toward
  Rustboro.
- The goal resolver should therefore select the currently available gate or
  scene task, then delegate movement, dialogue, interaction, and battle work
  to their respective executors until the ROM reports the resulting fact.

| State or machinery | Authority and purpose | Audit result |
| --- | --- | --- |
| `get_event_flag()` / `get_event_var()` and `CampaignFacts` | Current ROM/save-backed story state; `first_badge_obtained` comes directly from `DEFEATED_RUSTBORO_GYM` | Keep as campaign authority; reject uninitialized reads |
| Recursive goal/task resolver and capability registry | Resolve available work from current facts and recursively expand unmet prerequisites | Make this the primary campaign interface; do not replace it with a selected-goal state machine |
| `CampaignController` objective, tactical loop, recovery phase, and debounce fields | Process-local ownership needed to resume a capability across frames | Legitimate execution state; never use as completion evidence |
| `CampaignProjection` / `NuzlockeRulesProjection` | Replayable encounter, capture, faint, whiteout, and battle history | Legitimate durable Nuzlocke history; not a substitute for current ROM story facts |
| `nuzlocke_started`, `NuzlockeStarted`, and the removed `start_nuzlocke` objective | Current campaign fact: ROM-derived Pokédex receipt. Legacy event: bot-written but replay-compatible rules/audit metadata | Keep the event readable for migration; keep it outside campaign gating and derive current activation from the ROM |
| Provenance sidecar high-water facts and party identity | Save identity/history compatibility check across profile replacement | Validation only; fresh-start observations cannot create or advance it, and it must not complete objectives |
| `EmeraldOpeningMode` / `OpeningSequenceState` | Legacy scripted opening flow and compatibility diagnostics | Retire; do not extend or use it as the campaign architecture |

`modules/modes/opening.py` is therefore not currently the source of the
`test_ng` false completion, but it is still legacy scripted machinery that
must not be expanded. Its `phase` is rebased from ROM observations and is not
the intended source of campaign progress. The machinery that caused the
reported completion was the combination of an all-`0xff` placeholder save and
treating those reads as available campaign facts. The durable
event/provenance files are a separate profile-side effect: they may survive a
run even when the game itself was never saved. A strict fresh-run check should
use `--no-save-state` and a clean event/provenance sidecar, or the repository
should add an explicit read-only/non-persistent profile mode rather than
inferring that policy from a profile name.

The current runtime already uses the live `pokedex_received` fact to activate
the one-encounter-per-area rule. The synthetic `start_nuzlocke` objective and
controller boundary handoff have been removed. `reach_petalburg` remains as a
ROM-observed traversal objective, while the actual Wally, Woods, Devon Goods,
Rustboro, and Roxanne boundaries are represented as recursively resolved
objectives. Legacy `NuzlockeStarted` records remain readable for migration and
rules projection, but cannot activate the current save's campaign.

### Completed cleanup boundary

The following items are the concrete removal/migration list from this audit:

1. Remove `start_nuzlocke` from the placeholder campaign list. Retain
   `reach_petalburg` as an observed traversal objective and add the actual
   Norman/Wally, Woods, Devon Goods, Rustboro, and Roxanne ROM-state goals to
   the recursive registry.
2. Remove the `CampaignController.refresh()` special case that invokes
   `runtime_campaign_boundary()`, and remove that campaign-boundary write from
   the progression path.
3. Derive `nuzlocke_started` from the current ROM's Pokédex receipt. If
   `NuzlockeStarted` is retained for migration or explicit ruleset metadata,
   it must not be a campaign objective, a completion fact, a substitute for
   Pokédex receipt, or the source of encounter eligibility when the ROM says
   the Pokédex has not been received.
4. Make encounter activation a Nuzlocke-rules policy: the implemented default
   is Pokédex receipt. An explicit alternative new-game boundary, including
   the policy for encounters before Poké Balls are obtainable, remains later
   Nuzlocke-semantics work.
5. Migrate old logs by preserving the legacy event for replay/audit, while
   deriving current campaign progression and default encounter activation
   from the current ROM observation. A stale legacy event must neither open a
   ROM-gated route nor make a fresh `test_ng` run active.

This cleanup is intentionally kept separate from the fresh-save fix: the
objective graph now consumes observation-only campaign facts, while the event
schema remains backward-compatible for old stores. The fresh-save fix and the
current graph cleanup are complete; first-badge live validation and the full
recovery ownership migration remain open.

## Current architecture and observed failure

The current flow is approximately:

```text
observe ROM-backed facts + durable rule/history projections
    -> recursively resolve the goal/task frontier and available work
        -> select a task by campaign/rules policy
            -> delegate movement, interaction, dialogue, or battle execution
                -> re-observe ROM/projections for completion
                    -> add recovery as a transient execution waypoint when needed
```

The relevant implementation points are:

- `modules/nuzlocke/campaign_objectives.py`: declarative goal/objective model,
  producer index, available-task discovery, and the recursive
  `plan_campaign()` resolver. The older ordered `select_campaign_objective()`
  path and fixed tuple ordering remain compatibility scaffolding; the
  executable campaign path uses recursive producer resolution.
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
  The former nearest-Center compatibility entry point now delegates to this
  same planned executor; the hard-coded nurse helper is outside Campaign
  Progression and remains only for non-campaign modes.
- `modules/navigation.py`: already models intermediate route analysis through
  `RouteCostAnalyzer` and returns first/continuation plans.

The live trace demonstrated the failure precisely. Immediately after the
introductory rival battle, the party was at `10/25 HP`; Oldale Center was
reachable in 38 steps. However, the game was still executing the battle-return
script, so readiness returned `UNKNOWN / OVERWORLD_UNAVAILABLE`. The tactical
objective continued across that boundary, and no recovery waypoint was queued.

The later Lab case exposed a second independent problem: legacy
`calculate_path()` cannot cross warps. Campaign recovery now uses the world
planner for the Lab-to-Oldale route, including the final Center door warp.
This route-discovery fix is separate from the still-incomplete
planning/ownership integration.

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

## Recovery route performance implementation status

The synchronous route-search portion of recovery and opportunistic readiness
analysis has now been moved off the emulator frame loop. Preparation reads the
live location, overworld, graph, and bounded healing-source frontier into an
immutable request; a prioritized background worker pool performs the candidate
search. The readiness scheduler treats that work as pending rather than
caching an incomplete `UNKNOWN` result, and the controller displays
`Calculating recovery route` while it waits. A completed result is accepted
only for the matching location/world request, so a map transition or dynamic
world change cannot execute a stale route.

Recovery and normal route analysis use a small prioritized worker pool. A
critical recovery search can therefore supersede and cancel speculative route
analysis instead of waiting behind it; diagnostic/speculative analysis has the
lowest priority. Every worker job has a ten-second wall-clock budget and
cooperative cancellation checks. Worker-side stutter/profiler instrumentation
is suppressed, and the bounded A* search cooperatively yields so the emulator
thread continues to receive scheduling time on low-core machines.

The synchronous `observe_route_recovery()` entry point remains for
compatibility callers and tests; Campaign Progression uses the prepared and
asynchronous path. The recovery request is immutable while pending and its
key uses compact live-world revisions rather than scanning all static tiles.
Static map routes and location-cost estimates are cached, and exact recovery
results remain reusable for the same request. A timeout is reported
explicitly and retried after a cooldown, so a pathological route cannot leave
the campaign in an indefinite per-frame wait loop.

### Emulator interaction safety: high feasibility

Dialogue detection remains necessary for the executor, but it no longer needs
to decide campaign intent. The route executor can wait for a stable,
controllable overworld before issuing movement, while the interaction
executor remains responsible for advancing dialogue and Center menus.

The existing `observe_interaction()` classifier is suitable as an execution
boundary. The readiness policy should not need to mount a generator merely to
avoid pressing A during `WaitForAorBPress`.

### Recovery execution: medium feasibility, partially implemented

The planned recovery runtime receives the selected destination, healing source,
and (when available) the executable `RecoveryStop.route`, so it does not
reselect a different Center during its route phase. It discovers the interior
interaction from a fresh observation and executes it through the shared
observation/action boundary. The former nearest-Center runtime entry point
delegates to this same path; non-campaign modes may still use the older generic
Center helper.

The Lab-to-Oldale route also shows why all recovery route validation and
execution should use one world-navigation abstraction. Campaign recovery now
uses the world planner for cross-map warps and the final Center door, so route
selection and execution share the same navigation model.

### Performance: implemented with bounded follow-ups

World route analysis can be expensive, especially with debug tracing and
cross-map searches. Recovery candidate searches and opportunistic route
composition now run only when their request context changes, in prioritized
background workers rather than on the frame loop. The readiness scheduler
reuses completed observations across ordinary movement and invalidates them on
map, battle, party/resource, and other meaningful context changes. Party and
inventory revisions are carried by the shared snapshot, so readiness does not
rebuild a full health/inventory signature just because another frame elapsed.
Requests include compact dynamic-world revisions, and stale results are
discarded before they can affect execution.

Preparation and first-use static graph construction still deserve profiling on
new ROMs, but the multi-second synchronous candidate-search stall and the
full-tile request-key scan have been removed from the live campaign path.

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
7. **Partial:** validate with the introductory rival profile, a route with an
   on-route Center, a critical-HP case, a no-Center route, and a dialogue-heavy
   case. The route-search handoff and recovery selection have been live-tested
   from the recovery-stall profile; longer healing/resumption coverage remains.

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
7. **Partial:** stutter tracing shows bounded recovery searches running off the
   frame loop without the original multi-second stall. Persistent route-result
   caching and longer live healing validation remain.

## Conclusion

The proposed system remains feasible without replacing the campaign
architecture. A first waypoint/envelope implementation and parent-plan
resumption now exist, and the expensive route searches are now asynchronous
and bounded. Recovery selection still begins in controller-side readiness and
execution still uses a separate recovery generator. The current guards are
therefore still evidence of an incomplete ownership boundary, not evidence
that dynamic recovery is inherently unsafe.

The next highest-value work is to move recovery selection into campaign-plan
construction, pass and execute the planned route directly, unify route
validation and execution, and complete the longer critical/dialogue-heavy live
scenarios. Persistent caching and preparation profiling can then be evaluated
against traces rather than speculation. This should eliminate the remaining
readiness/dialogue coupling while preserving the desired opportunistic and
critical-health behavior.
