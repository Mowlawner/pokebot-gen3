# ROM-derived story gate discovery plan

Date: 2026-09-14

Status: implementation in progress; source discovery, report-only compilation,
and fixture-backed route evidence are implemented. Runtime promotion and a
clean organic capture remain open.

## Purpose

Nuzbot currently knows about campaign gates through game-specific registry
entries. That works for a verified early Emerald slice, but it does not scale
to the full game and it can make the physical map graph look more open than
the story actually is. The next campaign layer should discover candidate
gates from the ROM, connect them to route feasibility, and validate them
against the live save.

The goal is not to turn every event flag into an objective. The goal is to
derive the meaningful actions that change the campaign frontier. A map arrival
is usually evidence or a prerequisite for an event; the event, trainer battle,
badge, item, or script completion that opens the next frontier is the progress
trigger.

## Executive decision

Story gate discovery should be a hybrid pipeline:

1. ROM analysis extracts candidate conditions and the scripts or map objects
   that use them.
2. World-graph analysis determines whether a condition affects traversal,
   object availability, or a mandatory interaction.
3. Live observation resolves the current flag, variable, map, object, and
   trainer state.
4. A verified registry supplies semantic names, overrides ambiguous analysis,
   and records game-specific facts that cannot be proven statically.
5. The campaign planner promotes only sufficiently supported candidates into
   blockers, preparation checkpoints, or objectives.

Unresolved candidates remain visible as diagnostics and conservative route
constraints. They must not become autonomous objectives merely because a
script contains a flag check.

## Current boundary

The current implementation already provides most of the consumer-side
contracts:

- `WorldMapGraph` supplies ROM-derived physical connections and map routes.
- `assess_campaign_reachability()` overlays campaign facts, registered map
  access requirements, and registered mandatory trainer checkpoints.
- `CampaignAccessRequirement` describes facts, capabilities, or badges needed
  before entering a map.
- `KNOWN_TRAINER_CHECKPOINTS` describes trainer hazards whose clearance and
  preparation affect route ownership.
- Emerald encounter candidates are discovered from ROM encounter tables and
  then filtered by Nuzlocke ownership and campaign reachability.

The missing producer is a ROM analysis layer that can create candidate access
requirements and trainer checkpoints with evidence. The existing registry
should remain the verified semantic layer while this producer is developed;
the first implementation should not replace it with unverified guesses.

## Definitions

### Candidate access gate

An observed or ROM-derived condition that may change whether a map, warp,
object, connection, or route segment can be used. Examples include a hidden
object becoming present after a flag, a blocking NPC disappearing, or a script
opening a route after a battle.

### Progress trigger

The meaningful action or event that satisfies an authoritative campaign fact.
Examples include completing Wally's tutorial, defeating a mandatory grunt,
receiving a badge, or recovering the Devon Goods. Reaching Petalburg is not a
progress trigger when the useful consequence is the Wally event inside the
city.

### Trainer checkpoint

A trainer or scripted battle that constrains a route. The checkpoint may be
mandatory, costly to avoid, or avoidable. Only the first category should
automatically block route execution. Costly or uncertain checkpoints should
cause preparation and diagnostics without being misclassified as mandatory.

### Candidate versus verified gate

A candidate has static or runtime evidence but unresolved semantics. A
verified gate has enough evidence to enter the campaign graph as a blocker or
objective, with a named completion fact and an executable capability.

## Proposed data model

The scanner should emit immutable candidate records rather than directly
mutating the campaign registry. The exact Python types can evolve, but each
record needs the following information:

```text
GateCandidate
  candidate_id
  game / ROM identity / revision
  kind: map_access | object_visibility | script_event | trainer_checkpoint
  source: map, object, script symbol, ROM address, or command offset
  guard: normalized flag/variable expression, if present
  affected maps, warps, objects, or route edges
  producer actions: scripts, trainer battles, item rewards, or flag writes
  current state: satisfied | unsatisfied | unknown | unavailable
  route evidence: routes blocked, alternate routes, cut-set confidence
  semantic confidence: high | medium | low
  verification status: candidate | observed | verified | rejected
  provenance and diagnostic explanation
```

Once promoted, a candidate should compile into one of the existing campaign
contracts:

- a `CampaignAccessRequirement` when it guards entry to a destination;
- a `TrainerCheckpoint` when it represents an intermediate battle hazard;
- a campaign fact and capability when it represents a meaningful progress
  trigger;
- a diagnostic-only record when it is useful evidence but not safe to plan.

The compiled record must retain the candidate ID and provenance. A log should
be able to answer “why is this route blocked?” with the source script, guard,
clearing action, and current observed state.

## Discovery pipeline

### 1. Extract the ROM facts

Build a game/revision-specific scanner that reads the sources already exposed
by the repository and adds script analysis where necessary:

- map headers, connections, warps, object events, collision, and encounter
  areas;
- object visibility, movement, and interaction predicates;
- map scripts and field scripts;
- event-flag and event-variable reads and writes;
- script commands for warps, object removal or movement, rewards, trainer
  battles, and scene completion;
- trainer identities and parties from the ROM trainer catalog;
- symbols and event names when the ROM revision provides them.

The first scanner does not need to interpret every command. It should produce
an explicit `unknown` result for unsupported or computed expressions, retain
the source address, and continue collecting independent evidence.

### 2. Build a guard and producer graph

Normalize each flag or variable use into a dependency graph:

```text
guarded object / warp / script
          ^
          |
     flag or variable
          ^
          |
script, trainer battle, reward, or other writer
```

This links “Route 104 is currently closed” to the event that can open it. It
also prevents a flag from becoming an objective simply because it is read by
an optional NPC script. The dependency graph is evidence; route impact still
has to be established separately.

### 3. Measure route impact

For each candidate, compare the observed world graph under the guarded and
unguarded states. Record:

- whether the guard changes a traversable warp, connection, collision tile,
  or required interaction;
- which destinations become reachable after the guard is satisfied;
- whether an alternate route exists;
- whether the guarded interaction lies on every route to the destination;
- whether the candidate affects only an optional object or dialogue branch.

A candidate that is the only route to a new campaign region is strong access
evidence. A candidate that changes only an NPC's dialogue is not an access
gate, even if its flag is persistent. A candidate with an unknown alternate
route remains unresolved rather than being promoted optimistically.

This analysis must operate on the physical graph and the dynamic event layer.
It must not encode area lists such as “Route 101, Route 102, Route 103” in
encounter policy.

### 4. Discover trainer checkpoints

Trainer discovery should combine three sources:

1. ROM trainer catalog entries and `trainerbattle` script commands.
2. Map object affordances, sight-line or interaction geometry, and defeated
   state flags.
3. Route analysis showing whether the trainer is mandatory, costly to avoid,
   or avoidable.

The resulting checkpoint should include the expected opponent roster when it
can be resolved, the preparation level/cap, engagement location, route
exposure, defeated-state fact, and a confidence explanation.

This is how Calvin can become an implicit preparation requirement without
creating a separate user-facing “prepare for Calvin” objective. The planner
can mount preparation because the selected route contains an uncleared,
mandatory checkpoint. Lady Cindy should instead remain an avoidable hazard if
the route analysis proves that the Woods entrance can be reached below her.

### 5. Resolve live state

At planning time, read the authoritative current state where available:

- event flags and variables from the active save;
- current map, coordinates, facing, and visible objects;
- badges, field-move capabilities, party, inventory, and Nuzlocke encounter
  ownership;
- defeated trainer state and current trainer identity;
- script/task state when a scene is in progress.

Unknown or unavailable observations must remain distinct from false. An
unknown clearance fact should produce a preparation or re-observation blocker,
not an open route. A known satisfied fact should remove the checkpoint and its
dependent routes from the active frontier.

### 6. Promote only verified semantics

Promotion requires evidence appropriate to the candidate type:

- **Access requirement:** a guarded route effect plus route impact, with a
  known completion fact or a verified script producer.
- **Mandatory trainer:** trainer identity plus route/collision evidence, and a
  stable defeated-state fact or observed post-battle state.
- **Progress trigger:** an observable completion fact and an executable
  capability that can cause it.
- **Optional or ambiguous event:** diagnostic-only until runtime evidence or a
  registry override resolves its meaning.

Game-specific registry entries remain the place for semantic names and
intentional overrides. The scanner should be able to say “this candidate is
likely the Wally gate” while the registry says exactly which fact and
capability represent it.

## Planner integration

The campaign planner should consume a merged gate catalog with this order of
authority:

1. current live facts and task state;
2. verified game/revision registry definitions;
3. validated ROM-derived candidates;
4. unresolved candidates as conservative diagnostics.

Reachability then evaluates encounter destinations, campaign targets, and
recovery routes against the same gate layer. An encounter remains a ROM-
derived opportunity with land viability for the current first-badge slice;
its area is not hardcoded into story planning. It can be visible as a future
opportunity while a gate prevents it from taking execution ownership.

Objective selection should emit a rationale record containing:

- the selected progress trigger or optional encounter;
- unmet prerequisites and route blockers;
- trainer checkpoints on the route and the preparation plan mounted for them;
- deferred encounters and why they were deferred;
- the route and risk cost of the selected work;
- the evidence and confidence behind the decision.

This gives the desired explanation when the bot chooses a Wally event over a
nearby encounter: it can state that the event opens the next corridor, which
encounters remain reachable later, and what travel or risk cost the choice
incurs. It also makes a mistaken choice auditable without treating a city
visit as progress.

When a gate or trainer is cleared, the controller should invalidate the
campaign reachability and frontier caches, remove the cleared checkpoint and
dependent unreachable trainers from human-facing frontier logs, and rebuild
from the new observed state. Completion must be established from ROM facts or
an explicit observed boundary before the old work is released.

## Implementation phases

### Phase 0 — Establish the scanner boundary

- Define `GateCandidate`, provenance, confidence, and verification states.
- Inventory the existing map, event flag/variable, symbol, script, and trainer
  readers.
- Add a report-only command that emits candidates for one ROM/revision.
- Keep scanner output deterministic and versioned.

Status: implemented. `gate_discovery` emits deterministic, versioned reports
without changing bot behavior.

Exit criterion: a report can identify the source and guard for known early
Emerald events without changing bot behavior.

### Phase 1 — Extract guard dependencies

- Parse supported flag/variable reads and writes.
- Normalize common script branches, object visibility, warps, trainer battles,
  rewards, and scene completion commands.
- Preserve unsupported expressions as explicit unknown nodes.
- Emit producer/consumer diagnostics and candidate records.

Status: implemented for the supported Emerald source command set. Unsupported
expressions remain explicit unknown candidates, and trainer rosters are
resolved from the pinned source catalog where available.

Exit criterion: the report links each verified early story event to its guard
and clearing action, with no invented flag names.

### Phase 2 — Connect candidates to route feasibility

- Overlay candidate effects on `WorldMapGraph` and observed dynamic objects.
- Detect route cut sets and alternate paths.
- Compile high-confidence map guards into
  `CampaignAccessRequirement` records.
- Keep optional object/dialogue checks diagnostic-only.
- Add cache keys and invalidation for guard state, topology, and ROM revision.

Status: implemented for the reviewed Wally candidate. The pure route-impact
analyzer, runtime edge-evidence normalizer, report-only compiler, pinned
Emerald review manifest, and opt-in live edge recorder are implemented. Two
disposable source-derived fixtures now exercise the actual Petalburg
westward traversal boundary: with `PETALBURG_CITY_STATE = 0` and Gym Boy
(local ID 8) at `(5, 11)`, traversal terminates blocked; with
`PETALBURG_CITY_STATE = 3` and Gym Boy at his post-scene position `(12, 15)`,
the same traversal reaches Route 104. Both observations have issued input,
terminal outcomes, ROM object positions, and empty pending/diagnostic sets.

These fixtures establish route evidence for the physical blocker. They do not
constitute an untouched organic save capture; that remains a separate live
confirmation. Calvin still lacks geometry-backed route cut-set evidence and a
stable post-battle clearance observation.

Exit criterion: a source report and live evidence identify the actual
Petalburg westward traversal blocker while Route 103 remains reachable before
that event. The fixture-backed criterion is met; an organic capture remains
useful confirmation before runtime promotion.

### Phase 3 — Auto-discover trainer checkpoints

- Correlate trainerbattle commands with trainer catalog identities.
- Discover defeated-state flags and object disappearance where available.
- Feed trainer geometry into route hazard analysis.
- Classify checkpoints as mandatory, costly-to-avoid, avoidable, or unknown.
- Compile mandatory checkpoints into the existing preparation contract.

Status: source trainer catalog extraction, route-risk classification, and
report-only trainer compilation are implemented. Geometry-backed route
classification and runtime promotion remain open.

Exit criterion: Calvin causes implicit preparation on the route that requires
him; an avoidable Woods trainer does not block a route that can safely pass
below her; scripted grunt battles appear as progress gates when their event
opens the next route.

### Phase 4 — Runtime verification and promotion

- Compare predicted guard state with live save observations.
- Record before/after facts at dialogue, trainer, reward, warp, and battle
  boundaries.
- Promote repeatedly confirmed candidates to verified registry entries or
  generated game-data artifacts.
- Add a safe fixture/replay path for ambiguous candidates instead of probing
  them destructively in a live Nuzlocke run.

Status: the opt-in `Gate Edge Validation` mode and
`--gate-live-config` recorder are implemented. The safe fixture path produces
deterministic JSON artifacts containing the raw save value, predicate result,
fixture parameters, ROM object positions, and terminal route outcome. The
Wally fixtures now agree with the source-derived physical blocker and its
`PETALBURG_CITY_STATE >= 3` traversal boundary. The campaign fact also waits
for the ROM's later gym return state, so the fixture artifacts do not claim to
prove that later boundary. The manifest remains report-only and the runtime
registry remains authoritative until the organic confirmation or an explicit
promotion review is complete.

Exit criterion: every promoted early-game gate has source provenance, a
completion fact, a capability, and a regression fixture.

### Phase 5 — Expand across the campaign

- Add puzzle and scripted-transition patterns.
- Add field-move, badge, villain, rival, legendary, Elite Four, and Hall of
  Fame gates.
- Support Ruby/Sapphire and FireRed/LeafGreen through game-specific scanner
  adapters and shared candidate contracts.
- Remove manual registry entries only when generated output and live
  validation provide equivalent evidence.

## Validation strategy

### Deterministic unit fixtures

Use small script and graph fixtures to verify:

- flag read/write dependency extraction;
- variable comparisons and unknown-expression handling;
- object visibility and warp guards;
- route cut-set detection and alternate-route preservation;
- trainer classification and defeated-state handling;
- candidate-to-registry compilation;
- cache invalidation after a gate clears.

### Emerald early-slice fixtures

The first ROM-backed acceptance set should cover:

- Route 103 being available before Calvin and before the Wally tutorial;
- Route 102's Calvin checkpoint and implicit preparation;
- the Wally tutorial opening the Petalburg Woods corridor;
- the Petalburg Woods grunt as a scripted trainer/progress gate;
- Rusturf Tunnel's Devon Goods event;
- encounter areas being derived from ROM tables and filtered by live
  reachability/ownership;
- no `reach_{city}` objective being generated when a meaningful scene exists;
- defeated or unreachable trainers disappearing from the active frontier after
  the relevant state change.

### Live evidence

Live runs should record candidate IDs, guard state, route assessment,
promotion decision, selected objective rationale, and post-event facts. A live
run may validate a candidate through ordinary progression, but the bot must
not intentionally trigger an uncertain event solely to classify it during a
Nuzlocke run.

## Risks and safeguards

### Script semantics are incomplete

Gen 3 scripts can branch on variables, call shared routines, use computed
values, or rely on engine behavior outside the local script. The scanner must
preserve unknown expressions and source locations. It should never infer an
open route from a parse failure.

### Persistent flags are not necessarily campaign gates

Many flags control optional NPCs, item collection, or visual state. Route
impact and executable completion evidence are required before promotion.

### Physical reachability is not semantic reachability

The map graph can contain connections that are unavailable at the current
story boundary. Dynamic object and script effects must overlay the graph, and
unresolved gates must remain conservative blockers.

### ROM revisions differ

Candidate IDs must include game and ROM revision. Symbols and generated data
should be treated as inputs to a versioned scanner adapter, not as universal
identifiers.

### False mandatory classification can damage a run

The planner must distinguish mandatory, costly-to-avoid, avoidable, and
unknown trainer hazards. Only mandatory checkpoints automatically seize route
ownership. Unknown geometry should trigger re-observation or a conservative
diagnostic path rather than an unsupported battle plan.

## Completion criteria

This work is complete for a game/revision when:

- all reachable campaign regions in the supported scope have candidate gate
  coverage;
- promoted gates have provenance, completion facts, and executable
  capabilities;
- mandatory trainer checkpoints are discovered or explicitly justified as
  exceptions;
- optional events do not appear as campaign objectives;
- encounter discovery remains ROM-derived and Nuzlocke ownership-aware;
- objective selection and deferral decisions are explainable in the logs;
- gate completion invalidates route/frontier state correctly; and
- ROM-backed replay and live runs agree on the discovered gate behavior.
