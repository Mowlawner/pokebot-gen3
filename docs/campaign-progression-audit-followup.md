# Campaign progression audit follow-up

Date: 2026-09-12

This document records the campaign progression audit decisions, the current
implementation status, and the remaining work. It replaces the earlier
planning-only version now that implementation and live validation have begun.

## Decisions and approvals

The following directions are approved:

- one risk model with separate active-faint, party-wipe, encounter-loss, and
  run-loss probabilities;
- one battle planner contract for victory, training, capture, switching, item
  use, single battles, double battles, and boss battles;
- correct Gen III action ordering, including action-specific priority and
  random speed ties;
- an explicit preparation phase before unavoidable or costly-to-avoid trainer
  checkpoints;
- a route-level trainer hazard envelope covering intermediate maps;
- explicit Mart transaction phases that retain ownership through purchase and
  exit dialogue;
- extension of the campaign registry; and
- performance work aimed at reducing frame-loop stutter.

The user clarified that projections and incomplete search results must remain
autonomous. Search uncertainty may lower confidence and make the action
risk-accepted, but ordinary node, horizon, or time exhaustion must not request
player intervention. The bot should execute the best legal action under the
current risk model, preserve the run when possible, and record the actual
outcome for the next decision. An unrecognized battle state with no legal
action contract remains a separate coverage failure.

Route 103 is not a mandatory encounter destination. Encounter acquisition is a
general utility decision. Open party capacity, conditioned species value,
team coverage, travel and battle cost, resource cost, future opportunities,
and the probability of revisiting the area should determine whether an
encounter is worth taking now.

Species Clause makes strategic routing feasible. Duplicate outcomes can be
rejected and rerolled according to the configured rules, so an area can be a
guaranteed target, an increased-probability target, or an ordinary candidate.
The planner must condition probabilities on durable species history and must
reserve the word “guaranteed” for a rule-defined guarantee.

## Current status

### Implemented or substantially implemented

- Expectiminimax is the primary stochastic evaluator when the battle response
  facts are complete. Its result keeps active faint, party wipe, encounter
  loss, and run loss separate.
- Search models action-specific priority and random speed ties, and retains
  status, accuracy, critical-hit, and damage-roll branches.
- Incomplete search results already choose an autonomous best-available legal
  action. A new cooperative elapsed-time budget labels time exhaustion
  separately from node exhaustion so it cannot be presented as a safety proof.
- Preparation-owned wild battles retain expectiminimax ownership through the
  battle-start handoff, so EXP actions, switching, limited healing, and escape
  compete under one risk model. An incomplete result remains autonomous; a
  proven guaranteed catastrophe still follows the safety boundary.
- Victory, training, and capture callers use the normalized battle planner
  contract where their facts are complete. Capture and EXP utility are part of
  the search state.
- Trainer facts can come from the ROM-backed catalog. Known trainer
  checkpoints have preparation and EXP-cap readiness handling.
- Route analysis publishes trainer risks and can promote unavoidable or
  costly-to-avoid checkpoints into an explicit preparation/engagement phase.
- Missing trainer sight-line geometry is conservative: it is reported as
  unknown and exposed on the normal route instead of being treated as safe.
- Route-analysis cache keys include trainer identity, state, and hazard
  geometry. Movement-only requests within an unchanged hazard envelope are
  coalesced while pending, and completed results remain tied to their exact
  start coordinate.
- Species-Clause conflict detection, durable species history, duplicate
  encounter conditioning, open-party-slot utility, and removal of the Route
  103 hardcoded encounter requirement are present in the campaign policy.
- Mart purchase and recovery ownership is represented as a transaction seam:
  the restock phase finishes purchase and exit dialogue before opportunistic
  recovery can take ownership.
- The Emerald registry and campaign capabilities cover the early campaign
  through Roxanne, including the introductory Route 103 progression.
- Campaign reachability now overlays the cached map graph with the registered
  early story corridor, field-move, badge, trainer, and late-game gates.
  Future encounter locations remain internal policy opportunities, while the
  human frontier shows only destinations assessed as reachable now. Route 102
  remains open on its checkpoint map, and Route 103 is selected in the live
  early profile without exposing Route 110, Route 111, or later areas.
- The campaign controller now yields ownership to battle transitions before
  coarse game-state changes are visible. Emerald battle-transition tasks and
  the corresponding battle game states suppress readiness refreshes,
  frontier rebuilds, and tactical-generator advancement until the battle
  listener takes over.
- Route-signature invalidation now handles the current seven-field schema,
  older six-field injected signatures, and future fields without diagnostic
  exceptions. Unknown or topology-affecting fields conservatively invalidate
  the cached route; only known runtime fields are compared against the
  remaining route.
- A ROM-owned scripted interruption that outlives one local input budget now
  yields back to the still-mounted capability instead of raising a tactical
  exception. This preserves restock ownership while automatic scenes such as
  the Running Shoes dialogue finish and avoids controller remount churn.
- Capability completion now has an explicit projection boundary. When an
  encounter or other capability returns after observing authoritative
  completion, the controller fences the completed objective, waits if the
  campaign facade is one frame behind, releases stale selection/readiness
  ownership, and only then selects the next frontier. The mode cache also
  validates a cached objective's completion predicate before reuse. This
  prevents a captured Route 103 or Route 102 encounter from being remounted
  indefinitely after its capability has returned.
- Route hazard inspection now tolerates a missing live emulator/runtime during
  unit and ROM transition boundaries. It publishes the registered checkpoint
  envelope with clearance treated as unconfirmed instead of raising while
  reading campaign facts. The interaction handoff probe also ignores stale
  emulator scripts when Manual mode owns the process.

### Partial or requiring live confirmation

- The battle engine's normalized mechanics still need broader Gen III coverage
  for every move effect, double-battle interaction, boss-specific behavior,
  and all switch/forced-switch transitions.
- Trainer catalog resolution is dynamic for known ROM facts, but the static
  hand-verified checkpoint registry and intermediate-map hazard coverage need
  systematic expansion and live confirmation.
- The Emerald trainer-script decoder now accounts for the command's battle-type
  byte before the two-byte trainer ID. The Route 103 starter-branch registry
  also maps the branch name to the player's starter, so a Treecko player
  resolves the rival's level-5 Torchic. Focused trainer tests pass, and the
  fresh live `test_ng` trace resolves `rom:BPEE0:532` without entering the
  mandatory-roster wait; the run then continued into trainer preparation and
  wild-battle training before its watchdog expired.
- Pre-battle trainer readiness now uses a short bounded projection without the
  live battle time cutoff. This prevents search-time exhaustion from being
  mistaken for evidence that more levels are required. The follow-up live
  `test_ng` run reached a healthy level-9 Treecko, recovered, and returned to
  Route 103 without the roster wait; it still ended at a temporary readiness
  availability boundary before the rival battle, so that routing seam needs
  separate live follow-up.
- The latest live ceiling run reached level 6 and remained on Route 103 until
  its watchdog expired. The near-rival profile also stopped at level 6 but
  entered a transient readiness-unavailable boundary before the rival
  interaction; this prevented a clean end-to-end rival dialogue validation.
  Focused tests and fresh-run logs do confirm that the generic hidden-message
  boundary emits the shared `ADVANCE_DIALOGUE` action.
- Preparation is implemented for registered checkpoints; the campaign needs
  the same contract for every later trainer gate.
- Mart phases have unit coverage for purchase ownership and affordability. A
  live near-Pokédex run now confirms combined medicine and Poké Ball purchase,
  purchase confirmation, exit dialogue, deferred ownership, and one-time
  restock completion. Partial affordability and damaged-party recovery after
  transaction closure still need live confirmation.
- The encounter policy can score conditioned candidates, but sequence planning
  across all future areas, reroll limits, encounter methods, and travel risk
  needs broader validation.
- The early registry/frontier gate has live validation from
  `test_begin_nuzlocke`: after the Pokédex boundary the visible encounter
  frontier contains Route 101, Route 102, and Route 103, and Route 103 is
  selected. Route 110, Route 111, and Altering Cave remain internal blocked
  candidates with explicit registry blockers.
- The battle-transition ownership guard has focused unit coverage and a fresh
  `test_ng` live trace. The trace reached the introductory Route 103 battle
  without campaign readiness refreshes during the transition-task window.
  The battle later ended in a loss; that is a separate battle-policy issue,
  not evidence that the transition guard failed.
- Escape actions are now filtered at the normalized battle boundary. The
  evaluator requires both a legal wild-battle escape fact and a nonzero
  current escape probability, so an unavailable Run Away action cannot enter
  the search tree. Rare opponent escape moves are represented explicitly when
  the move adapter identifies them, currently covering Teleport, and are
  reported as opponent-fled encounter outcomes rather than invented opponent
  Run Away nodes.
- Incomplete defeat and training searches now have a root-action fallback:
  when the cutoff winner is a zero-damage setup move, the evaluator selects
  the strongest legal immediate damaging move. Capture objectives retain the
  capture utility comparison because weakening can be the correct action
  there.
- The Emerald field-dialogue predicate now recognizes the ordinary
  `Std_MsgboxDefault` input boundary while `IsFieldMessageBoxHidden` remains
  active after the draw task ends. The state is still tied to an established
  field-message lifecycle, so it delegates to the shared dialogue input
  primitive without broadening unrelated script waits.
- Pre-battle readiness now uses a five-turn bounded projection and can accept a
  complete horizon line when it has a strong projected win envelope and a
  separately bounded run-loss probability. Preparation also stops at one level
  above the strongest known trainer member, so temporary uncertainty or a
  recovery boundary cannot restart open-ended wild-grass grinding. The live
  fresh profile reached only level-5 and level-6 preparation targets; it did
  not mount the former level-7-through-10 escalation sequence.
- The route-signature diagnostic fix passes the focused unit suite and a fresh
  `test_ng` run with a 120-second watchdog. The run produced no route-signature
  `KeyError`; it did still lose the introductory rival battle, which remains
  a battle-search or policy-quality issue. The Discord rich-presence thread
  also reports the expected sandbox permission failure and is unrelated to
  campaign control.
- The new battle deadline and route-job coalescing reduce worst-case work in
  the code path, but live profiling must establish useful limits for emulator
  frames and search quality.
- A fresh Route 102 live run observed Calvin as an undefeated trainer at
  distance 3, marked `imminent_trainer=True`, and exposed the registered
  `route102:first_poochyena` hazard. The party was critically injured, so
  recovery correctly retained ownership and took priority. That run ended
  before the post-recovery preparation mount; `CAMPAIGN_TRAINER_PREPARATION_MOUNTED`
  still needs a live confirmation after healing.
- A fresh Rustboro Mart live run reached
  `RustboroCity_Mart_EventScript_PokemartBasic`. When Poké Ball inventory
  changed during the transaction, the controller retained
  `restock_recovery_items` even though the planner selected another task. The
  run completed a purchase of `4x Antidote, 2x Potion` and issued the exit
  input afterward. The party was healthy, so deferred recovery after the Mart
  transaction remains unverified live.
- The near-Pokédex live profile also completed a combined purchase of `4x
  Antidote, 5x Potion, 6x Poké Ball` at Oldale Mart. Restock ownership was
  retained while the ROM advanced through `Task_ExitBuyMenu` and
  `Task_ReturnToShopMenu`. After the interruption-boundary change, the live
  profile completed the restock objective once, then completed Route 103 and
  Route 102 encounters and advanced through the Petalburg arrival boundary; no tactical
  interruption exception or restock remount loop appeared. The watchdog ended
  the run while navigating toward Petalburg.

### Open

- Extend the authoritative Emerald registry through later gyms, rivals,
  villain events, Elite Four, Champion, and Hall of Fame completion.
- Complete coverage for later trainer hazards and verify each checkpoint's ROM
  script, map, party, level cap, route exposure, and defeated-state fact.
- Finish the complete battle contract for double battles, boss battles,
  unsupported moves, and unknown battle observations.
- Establish measured risk budgets and search deadlines by battle objective.
- Add end-to-end validation for species-clause rerolls and strategic encounter
  ordering rather than only candidate-level scoring.
- Profile remaining frame-loop reads, readiness refreshes, route composition,
  and battle planning after the current deadline/coalescing change.
- Reduce repetitive tactical handoff diagnostics and profile the ROM-owned
  Petalburg arrival script separately. The latest live run reaches Petalburg
  and waits while Emerald's `ShowGymToPlayer`/`LeadPlayerToGym3` script owns
  movement; this is distinct from the earlier Route 103 encounter deadlock.

## Implementation order

1. **Implemented in the current reachability slice:** overlay registered
   mandatory trainer checkpoints, their campaign-clearance facts, the early
   story corridor, field-move requirements, badge gates, and post-league map
   requirements on the cached ROM map graph. Encounter tasks retain future
   utility and diagnostic visibility, but only destinations assessed as
   `reachable_now` may take execution ownership or appear in the human
   frontier. Route 102 remains encounter-eligible while routes beyond Calvin
   defer until the Petalburg traversal fact is authoritative. Route 111,
   New Mauville, and later regions defer until their registered requirements
   are observed. See [campaign-reachability.md](campaign-reachability.md).
   Extend this contract with verified field-move, story, puzzle, and later
   trainer gates before treating it as whole-game coverage.

2. **Keep the shared contracts stable.** Treat the battle risk dimensions,
   planner result, trainer hazard, encounter candidate, species history, Mart
   phase, and campaign capability types as the boundaries between policy and
   execution. Add fields only when they represent an observed or derived fact
   that the executor needs.

3. **Finish bounded battle search.** Use the elapsed-time deadline together
   with the node cap. Return the highest-value legal root action when the
   result is incomplete, mark the result as projected or risk-accepted, and
   test that the battle listener never turns ordinary budget exhaustion into a
   manual stop. Then expand the action/mechanics adapter for double battles,
   forced switches, and unsupported move effects.

4. **Complete trainer hazard envelopes.** Build a registry-backed hazard
   record for each intermediate map and expose identity, sight-line geometry,
   avoidability, confidence, engagement location, and defeated-state facts.
   Unknown geometry or state must remain an autonomous defer/reobserve result
   until the envelope is known or the registered checkpoint proves that the
   trainer is mandatory.

5. **Run preparation as an explicit phase.** For each selected checkpoint,
   reach safe training terrain, resolve the required encounter/resource
   decisions, train within the level cap, verify the party and lead, then
   engage. Re-evaluate after battles, captures, deaths, level changes, map
   transitions, and newly observed trainer facts.

6. **Finish encounter sequence policy.** Persist species history and pending
   duplicate state, apply the configured Species Clause variant, calculate
   conditioned distributions, and compare areas by expected utility. Include
   open slots, target probability, team value, battle/capture risk, travel,
   supplies, and the value of preserving future areas. Route 103 should win
   only when those facts make it the best candidate.

7. **Validate Mart ownership end to end.** Exercise medicine plus Poké Ball
   purchases, partial affordability, no-affordable-item cases, confirmation
   dialogue, menu exit, and recovery after transaction closure. Ensure a
   replan cannot transfer ownership to the Center while a Mart transaction is
   still active.

8. **Extend the Emerald registry.** Add later story gates, trainer
   checkpoints, gyms, rivals, villain events, Elite Four, Champion, and Hall
   of Fame transitions from verified ROM facts. Keep completion dependent on
   authoritative state transitions rather than map arrival alone.

9. **Profile and optimize the frame loop.** Measure emulator reads, readiness
   refreshes, route-worker submissions, search duration, and action latency.
   Reuse immutable observations, coalesce equivalent pending jobs, and
   invalidate cached work on coordinates, hazards, battles, inventory, party,
   and world revisions. Every optimization must preserve exact-start and
   current-hazard validation.

10. **Validate in layers.** Run focused unit tests, the complete configured
   suite, then live profiles for early encounters, Route 103 progression,
   trainer preparation, species-clause behavior, Mart transactions,
   Roxanne, restart/resume, and later campaign completion. Live runs must use
   `--no-save-state` and `-hl`; use `-s 0`, `-na`, `-nv`, and debug tracing or
   logging where useful.

## Validation recorded

Before the current performance change, the focused campaign/battle work had
`434 passed`. The broader focused campaign, battle, trainer, recovery, and
resource set after the current change is:

```text
415 passed in 1.22s
```

That set covers expectiminimax, campaign orchestration, background route
planning, time-budget classification, unknown trainer handling, hazard-cache
invalidation, route-job coalescing, trainer preparation, and resource
execution.

The transition-ownership controller tests were also run independently after
the guard was added:

```text
180 passed in 0.88s
```

The route-signature and campaign regression suites after the diagnostic fix
are:

```text
97 passed in 0.60s  # test_agent_control.py
200 passed in 0.78s # campaign/orchestration/navigation set
```

The live diagnostic trace is saved at:

```text
/tmp/nuzbot-live-keyerror-fix.log
```

The live near-Pokédex restock and post-encounter trace after the scripted
interruption fix is saved at:

```text
/tmp/nuzbot-live-restock-defer-fix.log
```

The fresh `test_ng` live validation used `--no-save-state`, `-hl`, `-nv`,
`-na`, and `-s 0`, with trace output written to
`/tmp/codex-test-ng-transition-guard-20260912.log`. It reached the Route 103
introductory battle and was stopped by the external watchdog. The run did
not conclusively validate post-Mart recovery because the party was healthy in
the available Mart profile; a damaged-party shop profile is still required.

The complete configured suite before the latest boundary fixes reported:

```text
1737 passed, 38 skipped, 1 failed in 4.10s
```

The current focused suites report:

```text
98 passed in 0.52s   # test_agent_control.py
115 passed in 0.72s  # test_nuzlocke_emerald_capabilities.py
200 passed in 0.78s  # campaign/orchestration/navigation set
```

The completion-boundary regression covers a capability returning with an
authoritative completion fact and the controller selecting the following
objective on the next refresh. A live `test_begin_nuzlocke` run used
`--no-save-state`, `-hl`, `-nv`, `-na`, and `-s 0`, with the full trace saved to
`/tmp/nuzbot-live-completion-release.log`. It completed the Route 103 and
Route 102 encounters once each, released both completed objectives, and
advanced through the Petalburg arrival boundary.

The latest complete configured suite, after making route-fact lookup and mode
ownership tolerant of those boundaries, reports:

```text
1751 passed, 38 skipped in 4.23s
```

This closes the former order-dependent
`CampaignControllerTests.test_tactical_capability_remounts_after_unknown_readiness_without_loop`
failure without changing the live Campaign Progression handoff. The stale
test emulator is ignored while Manual mode owns the process; an active
Campaign Progression mode still observes ROM-owned dialogue and script
transitions.

A live profile was run from the Route 103 handoff with:

```text
gtimeout --signal=TERM --kill-after=5 90s \
  /Users/eliasmills/.pyenv/versions/pokebot_gen_3/bin/python pokebot.py \
  _verify_calvin_handoff_20260912 -m 'Campaign Progression' \
  -hl -nv -na -s 0 --debug-trace \
  --debug-log /tmp/codex-calvin-live-20260912.log --no-save-state
```

The run reached Route 103, executed six expectiminimax decisions, completed
wins, and selected an autonomous `Run Away` action for an incomplete wild
battle projection. It timed out at the external 90-second limit without
saving state. Search timings in that run ranged from roughly 118 ms to 12.5
seconds, which motivated the elapsed-time deadline.

The profile did not directly exercise Calvin's trainer hazard path because the
observed Route 103 rival was a story interaction with no trainer affordance.
The trace also showed existing Discord Rich Presence socket and audio
initialization errors, plus recoverable early-campaign navigation errors. The
repeated route-analysis pending states were the performance issue addressed by
the current code change; a fresh live profile is still required to measure
the improvement.

The escape legality and opponent-escape changes were validated with a fresh
`test_begin_nuzlocke` profile using `--no-save-state`, `-hl`, `-nv`, `-na`, and
`-s 0`. The full trace is saved at
`/tmp/codex-battle-escape-live-20260912.log` and the captured console output is
at `/tmp/codex-battle-escape-live-20260912.stdout`. The run reached wild
capture battles and emitted no opponent-flee or unavailable-escape search
nodes. It was stopped by the 60-second watchdog. The run also retained the
known Discord IPC permission traceback and high-volume diagnostic output; both
are environmental or diagnostic overhead rather than evidence of this change
failing.

A second fresh `test_ng` profile validated the incomplete-search fallback. Its
full trace is saved at
`/tmp/codex-test-ng-incomplete-fallback-20260912.log` and its console capture
is at `/tmp/codex-test-ng-incomplete-fallback-20260912.stdout`. The previous
introductory-rival sequence selected `Pound, Leer, Pound, Pound`; the new run
did not select `Leer` from an incomplete root and advanced through the earlier
battles. It still lost the rival battle after Mudkip entered at 9/22 HP. The
search reported the resulting line as unsafe and eventually 0% winning, so
the remaining defect is campaign recovery or preparation before this mandatory
checkpoint, not an unavailable-action or incomplete-search safety claim.

A later Route 102 profile was run with the required headless, unthrottled,
no-save flags and debug trace. It observed Calvin's hazard and the critical
party recovery path, but recovery owned the controller for the remainder of
the run, so preparation was not mounted. The relevant trace is
`/tmp/codex-route102-live-20260912.log`.

The current early Route 102 profile was also run with the required flags:

```text
gtimeout --signal=TERM --kill-after=5 60s \
  /Users/eliasmills/.pyenv/versions/pokebot_gen_3/bin/python pokebot.py \
  _audit_route102_live -m 'Campaign Progression' \
  -hl -nv -na -s 0 --debug-trace \
  --debug-log /tmp/codex-route102-reachability-20260912.log \
  --no-save-state
```

Its frontier marked Route 102 `reachable_now`, deferred routes beyond the
Calvin checkpoint, and reported `route111:two_badges` with missing
`BADGE01_GET` and `BADGE02_GET`. The selector retained campaign/resource
ownership and never mounted Route 111. This run ended at the external timeout
while the party was at 2/25 HP, so recovery remained the correct owner.

The same validation found an existing preparation-loop defect in a later Route
116 profile: grass preparation repeatedly mounted on a map with no usable
encounter table and emitted thousands of identical `BotModeError` records.
That is outside this reachability change and remains open for the preparation
executor audit.

A Rustboro Mart profile was also run with the required flags plus performance
and stutter tracing. It reached the Mart, retained restock ownership through
an inventory change and competing objective selection, completed the purchase,
and issued the exit input only afterward. It did not need Center recovery
because the party was healthy. The trace is
`/tmp/codex-mart-live2-20260912.stdout`.

That Mart trace also recorded route-analysis precondition gaps while the world
graph was unavailable. A preserved stutter trace remains noisy under
high-volume tracing: its accumulated sample has a median total frame time of
about 48 ms, 95th percentile about 53 ms, and 46 samples above 100 ms. These
figures include prior runs and debug overhead, so they identify a profiling
target rather than a production baseline.

A deeper `test_begin_nuzlocke` live profile used the required headless,
unthrottled, no-save flags and wrote both the console stream and the full
debug log:

```text
/tmp/codex-test-begin-nuzlocke-deep-20260912.log
/tmp/codex-test-begin-nuzlocke-deep-20260912.stdout
```

The 120-second run ended at the external watchdog with exit 124. It did not
emit a traceback, `KeyError`, `BotModeError`, or campaign/battle exception. It
completed Route 103 and Route 102 encounter work, promoted Calvin's registered
Route 102 checkpoint, mounted preparation, and continued into Route 102 grass
training. The final watchdog boundary occurred during a wild Lotad battle, so
the run did not reach Calvin's engagement or a terminal campaign state.

The log also confirms two performance concerns. Background route and recovery
analysis can leave the campaign without a tactical loop for many consecutive
frames while the worker result is pending; the worker eventually completed in
this run, so that interval is a responsiveness problem rather than proof of a
deadlock. High-volume trace mode emits repeated tactical handoff and plan
records on nearly every frame, and every expectiminimax decision in this run
took roughly 0.76 seconds. These observations remain inputs to the measured
frame-loop and search-budget work.

The final near-Pokédex profile used the required headless, unthrottled,
no-save flags and was stopped by the 30-second external watchdog. Its human
frontier showed only Petalburg, Route 101, Route 102, and Route 103, and
selected Route 103. It also recorded the combined Oldale Mart purchase and
retained restock ownership through the shop task transitions. The trace
contains roughly 3,379 repeated tactical handoff records in 30 seconds under
high-volume debug tracing, so performance profiling remains open; the
repetition is diagnostic overhead and a useful target for reducing frame-loop
stutter.

## Latest Route 102 and field-lead follow-up

The campaign field-lead panel now includes every observed party slot with
species, nickname, and level. The trace record also names the old and new
Pokémon when a rotation occurs, so a slot change can be read without joining
the numeric party index to a separate snapshot. Lightweight policy test
doubles remain supported when they expose only the current lead.

The Oldale-to-Route-102 freeze had two related causes. An exhausted cached
trainer route could treat unresolved engagement geometry as a completed wait
at the map-entry boundary; the control loop now invalidates that route and
reobserves the destination map. After that fix exposed Calvin correctly, the
live object trigger dropped the defeated state when the trainer left the camera,
and the mandatory-checkpoint fallback promoted Calvin again. Trainer triggers
now retain defeated state, route analysis ignores defeated triggers without
hazard geometry, and the mandatory fallback honors a defeated static trigger.

Focused coverage for the field-lead renderer, cached trainer-route
invalidation, defeated-trigger route analysis, and defeated mandatory
checkpoints passes. The final live profile used `--no-save-state`, `-hl`,
`-nv`, `-na`, `-s 0`, `--debug-trace`, `--debug-log`, and `--stutter-trace`.
It crossed into Route 102, reobserved trainer geometry, engaged Calvin once,
completed the early encounter/battle sequence, and advanced to
the Petalburg arrival boundary. The watchdog later coincided with an independent five-second
background recovery route-search timeout; no `TRAINER_HAZARD_UNKNOWN`, repeated
`No legal route`, `KeyError`, or Python traceback occurred in that run.

The source-gate live validation seam is available through
`--gate-live-config` and the opt-in `Gate Edge Validation` mode. Two
source-derived disposable fixtures now cover the physical Petalburg
westward traversal boundary. `/tmp/nuzbot-wally-fixture.json` records
`PETALBURG_CITY_STATE = 0`, Gym Boy local ID 8 at `(5, 11)`, and a terminal
blocked traversal. `/tmp/nuzbot-wally-post-fixture2.json` records
`PETALBURG_CITY_STATE = 3`, Gym Boy at `(12, 15)`, and a terminal traversal
into Route 104. Both artifacts include the source predicate, ROM object
positions, issued action, terminal result, and empty pending/diagnostic sets.
The fixtures establish the route evidence but are not untouched organic save
captures; an organic pre-scene run remains an optional confirmation before
runtime promotion. Calvin still requires geometry-backed route cut-set
evidence and a stable post-battle clearance observation.

## Remaining live runs

## Route 116 and live validation follow-up

The Route 116 investigation found two related planning seams. A background
navigation result could remain pending after the destination map was already
observed, so the control loop waited instead of completing the map goal. The
coarse world graph could also rank a map edge that had no executable ROM
transition. Map-level completion now wins over a pending worker result, and
encounter routing filters coarse edges against ROM-backed transitions before
mounting each stage. Encounter scoring uses the same executable route evidence,
so a deferred detour cannot be mislabeled as an on-route encounter. A preferred
encounter on the current required progression map receives priority over
deferred encounter utility. Land encounters are also filtered when their ROM
map has no land encounter tile, which prevents the land-only executor from
mounting an impossible table.

Focused tests and the final full suite passed after these changes: `1782
passed, 38 skipped`; the skipped tests are the existing emulator-tier tests.
The final Route 116 live validation was run with
`--no-save-state`, `-hl`, `-nv`, `-na`, `-s 0`, `--debug-trace`,
`--debug-log`, and `--stutter-trace`. It selected Route 116, chose a grass
tile, issued encounter-roll inputs, entered a wild Nincada battle, and captured
it. The run later timed out while recovery/readiness analysis reported an
unresolved opportunistic route; the original Route 116 grass freeze did not
recur. Its logs are `/tmp/route116-live-progression.console.log` and
`/tmp/route116-live-progression.debug.log`.

The `test_begin_nuzlocke` live run reached Route 102 and published Calvin's
trainer fact. It also showed repeated watchdog reobservations and many
high-volume tactical handoff records under trace mode; these remain a measured
performance target. The `test_go_fight_rival` run resolved the rival's ROM
trainer roster, mounted preparation before engagement, completed the rival
battle, and later mounted recovery after damage. Both runs ended at the
external validation timeout rather than a Python exception.

- Repeat the Route 102/Calvin profile after recovery is available and confirm
  preparation mounting, the prepared checkpoint handoff, and expectiminimax
  battle ownership in the trace.
- Start with a damaged party at a shop with a restock objective and confirm
  the transaction remains mounted through all purchases and exit dialogue
  before Center recovery takes ownership.
- Run a Species-Clause profile with duplicate outcomes and verify the
  duplicate is rejected, the pending encounter remains owned, and the next
  area receives the updated conditioned utility.
- Repeat the Route 103 and trainer profiles with performance tracing disabled
  and record maximum search duration, pending-job reuse, frame latency, and
  route safety decisions as a production-like baseline.

The implementation is not complete until the later Emerald registry and the
remaining battle, trainer, encounter, Mart, and live validation coverage are
closed.

## 2026-09-22 `test_ng` Route 104 follow-up

The latest requested log is `profiles/test_ng/run.log` (19,125 lines,
4,345,024 bytes, modified 2026-09-22 09:29:37 -0400). The Petalburg Woods
warp was reached and the user confirmed that the transition completed. The
last navigation record shows route progress at `index=88; remaining=0` and
invalidates the completed route because the emulator entered a
`non_overworld_interaction` boundary at Route 104 `(11, 38)`
([run.log:19011](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:19011)). The log then emits the Woods encounter frontier and shuts down without a
post-warp campaign action. This is a post-transition action-resumption defect,
not evidence that the Woods warp failed.

The Billy exposure was also specific and unintended. The campaign objective
was the first Petalburg Woods encounter, while the current intent was
calculating a recovery route. The readiness snapshots repeatedly identify
`BILLY` from `rom:BPEE0:319`, report
`PREPARE: TRAINER_BATTLE_UNREADY`, and show `coverage=1/2`
([run.log:17148](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:17148)). The encounter-stage diagnostic had no safe Route 104 Woods approach, while permissive approaches were found through the Cindy/Billy hazard objects
([run.log:15938](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:15938)). The recovery route therefore exposed Billy even though readiness had already said that Billy was not ready. The route did not intentionally select Billy as a campaign objective.

Billy's battle then entered with Zigzagoon and Seedot. The Seedot line was
modeled as repeated `Bide` turns, but the live battle continued attacking
during that charge and ended with Torchic Derrick and Wingull Gerald fainting
([run.log:17370](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:17370), [run.log:18021](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:18021)). The tactical fix must either keep Billy avoided through the trainer-free grass route or make preparation and battle planning reliably produce a fast, survivable Seedot knockout when exposure is unavoidable.

Status after the follow-up implementation:

1. Implemented Route 104 safe-route discovery for the trainer-free grass
   corridor. Cindy and Billy remain `AVOID` hazards for encounter and recovery
   navigation, and safe-search failure does not authorize a permissive hazard
   route.
2. Implemented the recovery ownership guard so an explicitly unready trainer
   cannot be exposed by a recovery route. Focused fixture coverage is present;
   a matching-boundary live confirmation remains outstanding.
3. Implemented Bide-aware tactical evaluation, including the observed charge
   state in candidate comparison. Focused planner coverage is present;
   encounter-level live confirmation remains outstanding.
4. Implemented post-warp campaign resumption. `AgentControlLoop` preserves
   route ownership through source and destination non-overworld boundaries,
   waits for a controllable destination observation, and then releases the
   completed warp route. The regression is covered by
   `test_warp_route_survives_non_overworld_boundary_before_destination_action`.
5. Focused fixture validation is complete. A fresh diagnostic run was
   attempted with `--no-save-state` at
   `/tmp/pokebot-woods-postwarp-20260922-low.log`, but timed out at the Route
   103 introductory-rival boundary before Route 104. It is incomplete live
   validation and does not establish the Woods behavior.

## 2026-09-22 `test_ng` recovery transition failure

The subsequent run reached Route 103 and selected the correct trainer-safe
recovery route to the Oldale Pokémon Center. At the Center door transition,
the observation was `GameState.CHANGE_MAP` with no overworld model. Cached
navigation validation dereferenced `observation.overworld.map_id` and aborted
recovery before Route 104 ([run.log:11202](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:11202)).

`AgentControlLoop._cached_decision_instrumented` now yields the interaction
state selector when no overworld model is available, preserving an active
transition route for the next coherent destination observation. The focused
agent and transition tier passes 138 tests, including the same
`CHANGE_MAP`/`overworld=None` boundary. The broader recovery integration tier
still has four separate readiness ownership expectation failures and requires
its own reconciliation.

## 2026-09-22 Route 104 hazard-envelope investigation

The latest `profiles/test_ng/run.log` is the 13:20 run. It reaches Route 104
at `(39,63)`, then reports safe probes `not_found` for the southern Woods
exits `(10,38)` and `(11,38)` while permissive probes cross Cindy and Billy
([run.log:14397](/Users/eliasmills/PycharmProjects/pokebot-gen3/profiles/test_ng/run.log:14397)).
The fail-closed readiness result is therefore behaving correctly; the route
model is the remaining defect.

The ROM audit resolves Route 104 as a `40x80` map. Cindy is object `26` at
`(11,44)`, trainer range `3`, movement type `FACE_UP_AND_RIGHT`; Billy is
object `28` at `(18,67)`, movement type `WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT`,
with movement range `(6,5)`. A source-backed navigation probe from the live
boundary reproduces the failure when Cindy's unavailable runtime facing is
expanded to all four cardinal directions. Using the ROM-listed Up/Right
envelope produces safe paths to both southern Woods exits, including the real
grass corridor, with no Cindy or Billy hazard intersection.

The cause was the normal-trainer fallback in `trainer_hazard_locations`: an
unknown runtime facing became an all-directions hazard even when the ROM
`FACE_*` template identified the only possible facing directions. The fix in
`modules/overworld.py` uses the template envelope first and retains the
all-directions fallback only when both ROM movement and runtime facing are
unavailable. `tests/test_overworld.py` now covers the `FACE_UP_AND_RIGHT` /
unknown-facing case. The focused overworld and world-navigation tier passes
104 tests; a fresh live rerun is still required to confirm the emulator's
runtime object state takes this path.

## 2026-11-05 audit of the five outstanding Route 104 issues

This section records an independent re-audit of the 2026-09-22 claims. Code
audits were read-only; the fixes below are committed separately on
`improved-leveling`. Live-run results are recorded at the end and are kept
separate from code-level findings.

### Evidence base

- Latest requested log: `profiles/test_ng/run.log` (18,875 lines, 4,341,913
  bytes, modified 2026-09-22 20:23). It differs from the 09:29 log cited above.
  It reached Route 104, took a Wingull encounter, routed to the Woods warp at
  `(10,38)` with Cindy's hazards logged as directional ROM-envelope tiles, and
  ended at `cached goal position reached` (`index=9; remaining=0`) with no
  destination observation. It therefore neither confirms nor refutes post-warp
  resumption. A backup is at `/tmp/nuzbot-diag/test_ng_run_20_23_backup.log`.
- The full suite at the start of the audit was 2 failed / 2125 passed; the two
  Woods tests were order dependent (see issue 1).

### Issue 1: Route 104 trainer-free corridor

- `trainer_hazard_locations` ROM-envelope fix is real and on the live
  encounter-staging path.
- Fixed: `test_woods_coordinate_scene_geometry_releases_encounter_gate` and
  `test_woods_encounter_route_avoids_coordinate_scene_triggers` did not patch
  `get_map_metadata`, so their result depended on whether real map metadata was
  loaded by an earlier test. They are now hermetic, and a new test asserts both
  researcher scene events are avoided when only one is observed.
- Not done: a real-map Route 104 corridor test. Tile layouts come from ROM
  memory and offline tests have no ROM, so this needs a live run.
- Open decision: `_encounter_stage_route_cut_hazards` promotes a trainer common
  to every usable permissive exit to a mandatory hazard whenever the safe probe
  found nothing, if the trainer resolves to a ROM checkpoint. Live readiness
  resolved Billy as `rom:BPEE0:319`. A hazard-model false negative therefore
  still converts into permitted exposure. This is the mechanism of the original
  Billy incident. It was not changed because AGENTS.md forbids trainer-specific
  registry entries and requires route analysis to enumerate local alternatives.

### Issue 2: recovery and preparation exposure

- `_preparation_navigation_goal` defaulted to `TrainerMode.IGNORE`, and the map
  route into the training map had no trainer constraint. Both now default to
  `AVOID`.
- `_recovery_trainer_safety_error` only checked the one trainer named by a
  `prepare` readiness result. It now rejects a route that crosses any
  undefeated trainer hazard in the diagnostic, still naming an explicitly
  unready trainer when applicable.
- Regression found only by the live run: wrapping the grass goal in a
  `NavigationGoal` made `describe_preparation_target` return `None`, so no
  preparation target was published, wild battles got intent `avoid`, and the bot
  fled hundreds of encounters on Route 103 without earning XP. Fixed by
  unwrapping the goal, and an undescribable goal now logs
  `CAMPAIGN_PREPARATION_TARGET_UNDESCRIBED` instead of failing silently.

### Issue 3: Bide-aware tactical evaluation

- State plumbing and release modelling were already correct. Two gaps fixed:
  a Bide-locked opponent whose Bide was never revealed produced no opponent
  response (the release looked harmless), and an unreadable `gBideDmg` became 0.
  The search now injects Bide for an observed lock; an unreadable value on the
  release turn is assumed lethal and logged as `BATTLE_BIDE_DAMAGE_UNKNOWN`.
- New `BATTLE_BIDE_OBSERVED` log line records turns, stored and release damage.
- Not covered: `CampaignStrategy` acceptance with real `BattlePokemon` bytes, a
  switch-in absorbing the release, and non-campaign strategies that bypass the
  search.

### Issue 4: post-warp resumption

- The `AgentControlLoop` fix is real. A mutation check (making the invalidation
  unconditional) fails `test_warp_route_survives_non_overworld_boundary_before_destination_action`.
- An implicit step-on warp ending a route with `MOVE` already resumes correctly.
- A candidate mechanism for a post-warp stall, not attributable to the original
  run: `execute_campaign_encounter` waited without bound, and with trace-only
  diagnostics, for a settled destination observation and for mandatory-trainer
  geometry. Both waits are now logged in the normal log and bounded (unsettled
  observation raises `BotModeError`; an unresolved trainer gate returns to the
  frontier).
