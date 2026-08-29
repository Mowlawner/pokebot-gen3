# Nuzbot Project Plan

## Mission

Build an autonomous, observation-driven Pokémon Generation 3 Nuzlocke bot.
The bot must be able to start or resume a run, observe the emulator, maintain
authoritative campaign and Nuzlocke state, choose legal actions, navigate the
world, manage resources, complete battles, recover from transitions and
failures, and verify completion without human intervention.

The current repository is an observation-driven Emerald opening and early
progression pilot. Its live-validated fresh-run path starts without save data
and reaches Petalburg City. The declarative campaign graph also describes the
next Petalburg Woods, Rustboro, and first-badge objectives, but those remain
outside the currently executable live slice. The campaign rules, immutable
observations, event projection, level-cap gateway, resource policy, and
adaptive navigation work are valuable foundations, but they do not yet
constitute a complete whole-game autonomous bot.

## Definition of done

Nuzbot is complete when, for each supported Gen 3 game:

- a fresh or persisted run can be started and resumed;
- campaign progress is derived from authoritative observations rather than
  assumed from the selected objective;
- every major story, route, gym, villain, rival, Elite Four, Champion, and
  completion transition has a registered objective and executable capability;
- Nuzlocke rules are configurable, enforced at the relevant action boundary,
  and represented durably in replayable history;
- encounters, captures, deaths, party/PC changes, battles, resets, and
  ambiguous observations are handled conservatively;
- navigation accounts for dynamic world state, gates, puzzles, scripted
  movement, trainer hazards, encounters, and recovery;
- preparation, battle, healing, shopping, PC use, PP, status, held items, and
  party selection are planned as one risk-aware campaign loop;
- the bot can recover from process restart, emulator reset, save-state load,
  malformed observations, stalled scripts, failed interactions, and wrong
  routes;
- completion means verified Hall of Fame/game-complete state, not completion
  of an early campaign slice; and
- fixture, emulator, replay, and live validation cover the complete lifecycle.

## Current-state findings

### Campaign and objective planning

The supported fresh-run Emerald path currently reaches Petalburg City, which
the user confirmed from a fresh startup with no save data. The declarative
objective graph continues through Devon Goods recovery, Rustboro, and Roxanne,
but the current execution adapter still reports Devon Goods, Rustboro, and
Roxanne as unsupported. The live controller therefore reaches an unsupported
objective boundary after the Petalburg milestone rather than a verified first
badge.

The objective planner is declarative and supports dependency resolution. The
nominal ultimate goal is still named `beat_elite_four`, but its temporary
completion predicate is `first_badge_obtained` and its implementation marker
points to Roxanne. Only Roxanne is present in the Emerald boss registry; the
remaining gyms, rivals, villains, Elite Four, Champion, and Hall of Fame are
not registered as executable campaign milestones.

Preparation decisions and trainer policy exist as pure components, but
automatic preparation insertion is still backlog work. There is no complete
campaign-level decision cycle connecting preparation, party choice, battle
planning, recovery, and progression.

### Observation and state

The repository has normalized snapshots, immutable facts, event detection,
event persistence, campaign projection, and explicit unknown/unavailable
semantics. Campaign facts remain narrow and Emerald-specific. Story flags are
explicitly unavailable in `CampaignState`, and most later-world facts are not
yet represented.

The event store is durable and replay helpers exist, but live startup creates
a fresh `NuzlockeRuntime` after opening the store without loading the existing
records into its in-memory projections. Restart/recovery correctness must be
addressed before persistence can be considered complete.

### Rules

Implemented baseline rules are one encounter per area, fainting/whiteout, and
an Emerald Roxanne level-cap gateway. `SPECIES_CLAUSE` is now present as a
configuration identifier, but its enforcement is not implemented. Missing
configurable semantics also include shiny clause, healing restrictions, item
restrictions, and the detailed encounter-source rules required by Gen 3.

### Navigation

World routing, local observed navigation, transition handling, encounter
policy, trainer policy, and recovery diagnostics exist. Semantic cross-map
goals now use a global observed-world planner, and the navigation module has a
shared-frontier path for multiple activation targets. These are navigation
foundations, not complete whole-game campaign coverage. The world planner
still needs dynamic story/HM/badge gates, puzzles, scripted and forced
movement, bikes, currents, waterfalls, diving, caves, one-way exits, trainer
sight lines, and interaction sequences across the whole game.

### Battles and resources

The existing battle planner and battle strategies provide useful damage,
capture, switching, and move-management primitives. Campaign mode validates
level caps and can apply the configured wild-encounter resource policy, but it
otherwise defaults to fighting. It does not yet choose a legal party, prepare
a boss-specific strategy, model uncertainty, conserve PP, or integrate
opponent/team knowledge with campaign risk.

Resource policy currently focuses primarily on HP, healing items, Pokémon
Centers, and limited PC withdrawal. Autonomous progression also needs
Poké Ball and item purchasing, money, PP, status cures, repels, key items,
held items, PC organization, party capacity, and replacement decisions after
death.

## Delivery roadmap

### P0 — Make the existing Emerald pilot genuinely complete

1. Replay the durable event store into the runtime projections on startup.
2. Implement observed execution for Devon Goods, Rustboro, and Roxanne.
3. Replace the execution adapter's disconnected objective-ID whitelist with
   registry-owned executable capability definitions.
4. Record the currently validated pilot contract explicitly: fresh startup
   through Petalburg City. Extend that contract to a verified first badge only
   after the remaining P0 execution and validation work is complete.
5. Add an end-to-end Emerald fixture from new game through the first badge.
6. Ensure unsupported objectives cannot be reported as live-ready work.

Current validation: a fresh Emerald run with no save data reaches Petalburg
City (user-confirmed). Remaining P0 exit criteria: a fresh run can reach and
verify the first badge, and a process restart can resume the same run without
losing encounter/death/history state.

### P1 — Mount autonomous preparation and tactical battle control

1. Make preparation a first-class interruptible objective.
2. Re-evaluate preparation after captures, level-ups, deaths, item changes,
   map transitions, and boss discovery.
3. Add legal party selection before battle entry instead of only rejecting an
   over-cap party.
4. Integrate campaign risk budgets and boss policies with the existing battle
   planner and strategies.
5. Add type coverage, move quality, matchup, uncertainty, switching, PP, and
   consumable decisions.
6. Add shopping, PC, status, PP, and party-management objectives.
7. Persist preparation decisions, seeds, rationale, and battle policies.

Exit criteria: the bot can prepare for and safely execute the first gym using
only current observations and registered campaign facts.

### P2 — Complete the Emerald campaign registry

1. Register all Emerald gyms, caps, leaders, rival battles, and villain
   milestones.
2. Register HM acquisition and badge-gated access.
3. Register required key items, puzzles, scripted interactions, and map gates.
4. Register Elite Four, Champion, and Hall of Fame objectives.
5. Add authoritative completion facts and executable capabilities for every
   registered objective.
6. Replace the placeholder ultimate goal with verified completion state.

Exit criteria: an Emerald run can progress from a fresh game through the Hall
of Fame with objective transitions verified from observations.

### P3 — Complete Nuzlocke semantics

1. Implement Species Clause and future rule extensions behind configuration.
2. Define canonical encounter-area identity separately from raw map IDs.
3. Model land, surfing, fishing, Rock Smash, static, gift, roamer, Safari,
   forced, egg, and duplicate-species encounters.
4. Define capture failure, fleeing, whiteout, death, and ambiguous-outcome
   behavior.
5. Add shiny and healing/item restriction policies where configured.
6. Validate every rule through ordered event replay and save-state boundaries.

Exit criteria: every configured rule is observable, replayable, testable, and
enforced at the correct action boundary.

### P4 — Complete adaptive world navigation

1. Make the world graph depend on observed story, badges, HMs, items, and
   movement capabilities.
2. Represent affordances and prerequisites explicitly.
3. Plan interactions and scripts, not only map destinations.
4. Add puzzle, bike, current, waterfall, diving, cave, and forced-movement
   capabilities.
5. Add recovery for wrong map, wrong tile, failed trigger, stalled script,
   unexpected battle, and unavailable observations.
6. Validate trainer avoidance/exposure and encounter policy across the world.

Exit criteria: navigation can reach every registered objective from arbitrary
valid observed locations and recover from expected execution failures.

### P5 — Expand beyond Emerald

1. Add Ruby/Sapphire capability and objective registries.
2. Add FireRed/LeafGreen capability and objective registries.
3. Keep campaign contracts and generic planning shared while keeping ROM
   facts, scripts, maps, bosses, and completion semantics ROM-owned.
4. Add per-ROM fixture and full-run validation.

Exit criteria: all supported Gen 3 games expose equivalent autonomous campaign
contracts with game-specific implementations.

## Testing strategy for the bot

Testing should be layered:

- pure rule, fact, objective, policy, and planner unit tests;
- normalized observation and event-reducer tests;
- persistence/replay/corruption tests;
- mocked controller and capability tests;
- save-state fixture tests for map, menu, battle, capture, death, and script
  boundaries;
- emulator-driven progression tests for each campaign milestone;
- full-run smoke tests per ROM;
- deterministic seeded-run and reset/restart tests; and
- negative tests proving conservative behavior when observations are unknown,
  unavailable, malformed, or contradictory.

## Pytest migration assessment

### Current testing structure

The test suite is primarily `unittest`:

- most files import `unittest` and use `unittest.TestCase`;
- many tests include direct `unittest.main()` entry points;
- `tests/utility.py` owns a custom `BotTestCase`, emulator setup, frame
  stepping, save-state decorators, RNG control, and global patches;
- emulator tests require three ROMs and `.ss1` save-state fixtures;
- tests use global `unittest.mock.patch` objects started at import time;
- a smaller and newer subset already uses plain `def test_*` functions;
- pytest configuration now exists in `pytest.ini`, and pytest is declared in
  `requirements-dev.txt` rather than the runtime `requirements.py`;
- CI now has a lightweight GitHub Actions collection job, but does not yet run
  behavioral tests or ROM/native-mGBA tests.

### What would migrate automatically

Pytest can collect the existing `unittest.TestCase` classes and ordinary test
functions without rewriting them. The existing `test_*.py` naming convention
already matches pytest discovery. `unittest.mock`, assertions, `setUp`,
`tearDown`, and `subTest` are generally supported.

An initial compatibility phase would likely require only:

1. add pytest as a development/test dependency;
2. add a minimal `pytest.ini` or `pyproject.toml` test configuration;
3. document `pytest`, focused selection, and ROM prerequisites;
4. run the existing suite under pytest and classify failures; and
5. add a CI test job separate from the Black job.

### Work that is not automatic

The custom emulator harness needs deliberate treatment:

- `tests.utility` starts global patches during import, so collection order
  and test isolation need review;
- `BotTestCase` and decorators encode emulator lifecycle and generator/frame
  timeout behavior that should eventually become fixtures or explicit helper
  APIs;
- timeout failures write `tests/last_frame_timeout.png`, creating shared
  output and possible parallel-test races;
- ROM and save-state availability must be expressed as skip markers or
  session fixtures rather than opaque setup failures;
- stateful global `context`, emulator memory, RNG, listeners, controller
  stacks, and caches need reset guarantees between tests;
- tests that call `unittest.main()` can remain temporarily, but should be
  removed or isolated once pytest is the supported runner;
- tests depending on exact generator behavior and frame counts need
  explicit pytest-friendly diagnostics; and
- long emulator tests need markers and a serial execution policy before any
  parallelization is considered.

### Estimated effort

Approximate effort for one developer familiar with the emulator harness:

| Scope | Estimate | Result |
|---|---:|---|
| Compatibility runner only | 0.5–1 day | Pytest invokes most existing tests; limited isolation and reporting improvements |
| Production migration | 3–6 days | Dependency/config/CI, harness fixtures, skip/marker policy, isolation fixes, docs, and failure triage |
| Broad idiomatic rewrite | 1–3 weeks | Convert classes/decorators to fixtures, parameterize state matrices, improve assertions, split unit/E2E suites, and remove legacy runner assumptions |

The estimate excludes fixing unrelated product failures exposed by the first
pytest run. Pytest is now installed in the configured project interpreter;
the first baseline is recorded below.

### Recommendation

Recommend a phased migration, but do not block Nuzbot campaign work on a full
rewrite.

The compatibility runner, test-tier markers, collection guard, and development
dependency boundary are now in place. Keep existing `unittest` tests working
while adding the behavioral unit CI gate and ROM availability checks. After
the campaign architecture stabilizes, convert the emulator harness and the
highest-value campaign tests to pytest fixtures and parameterized tests
incrementally.

This gives better collection, filtering, failure output, skip/marker support,
fixture composition, and future property/state-matrix testing at relatively
low risk. A wholesale rewrite now would consume several days to weeks while
the more important current risk is missing autonomous campaign capability,
not the assertion framework.

### Migration baseline (2026-08-27)

The first compatibility run collected 954 tests successfully under pytest
8.3.5 after fixing an import-order cycle in `modules.nuzlocke`: its runtime
controller/resource exports are now lazy, preventing agent-control collection
from re-entering a partially initialized module.

A full run reached 133 passing tests and 25 failures before being stopped
because a long emulator test continued running. The representative failures
were reproduced with the original unittest runner, confirming that they are
existing behavioral/test-baseline failures rather than pytest conversion
failures. The failures cluster around agent navigation/dialogue expectations,
campaign capability defaults, recovery expectations, and resource-policy
behavior.

The two pytest collection warnings caused by fixture-local enums named
`TestGameState` have been removed by renaming those enums to
`FixtureGameState`; collection is now warning-free for this issue.

The test-tier boundary is now implemented in `tests/conftest.py`: the shared
`BotTestCase` harness and native mGBA tests are classified as `emulator`,
while remaining tests are classified as `unit`. Emulator tests are skipped by
default and can be enabled with `--run-emulator`; `--run-e2e` also enables
future explicitly marked end-to-end tests. The 2026-08-27 collection snapshot
reported 916 unit tests and 38 emulator tests; the current unit run is
reported below.

Behavioral CI execution remains intentionally deferred. The repository now
has a collection job with the canonical runtime and development dependencies,
but CI still does not provision the native mGBA module or ROM fixtures needed
for emulator execution. The remaining testing boundary is to separate the
green fast unit tier from the ROM-backed emulator tier and decide when the
known behavioral baseline is suitable for gating.

The first dependency-boundary step is complete. Pytest is now declared in
`requirements-dev.txt` instead of the application's runtime
`required_modules`; the latter remains the canonical source for PokéBot's
platform-specific runtime dependency checks. This prevents the packaged bot
from acquiring a test runner while giving contributors and CI an explicit
development install target. A focused 37-test unit slice passes under the
configured Python 3.12 environment.

The first behavioral triage pass is also complete. Readiness/observation,
resource policy/runtime, interaction-state/task lifecycle, and the two
Littleroot encounter tests are now green. After the subsequent navigation,
agent-control, campaign, transition, capability, opening-flow, and benchmark
triage, the 2026-08-27 unit tier reported **916 passed, 38 deselected** under
the configured Python 3.12 environment. The 38 deselected tests are the
ROM/native-emulator tier, which remains outside the fast unit boundary.

The opening tests now explicitly select protagonist gender and phase when
testing gender-dependent house routing or observation reconciliation. Menu
and wall-clock assertions likewise track the current normalized observation
and ROM script-identity contracts. Semantic cross-map navigation is validated
against its current single global-frontier profile shape.

A lightweight GitHub Actions compatibility job now installs the canonical
Python runtime requirements plus `requirements-dev.txt` and verifies unit-tier
collection. It intentionally does not run behavioral tests yet: the current
job is a collection guard, and it does not provision native mGBA or
copyrighted ROM fixtures.

### Behavioral failure triage (2026-08-27)

The initial complete unit-tier run produced **833 passed, 83 failed, 38
deselected** under pytest 8.3.5. The failures reproduced the then-current
behavioral baseline;
they are not caused by unittest compatibility or pytest collection. The
distribution is:

This is a historical triage snapshot. Subsequent fixes resolved the listed
unit-tier failures; the current unit result is recorded in the validation
section below. The remaining deselected tests are ROM/native-emulator tests.

| Cluster | Tests | Initial classification | Triage action |
|---|---:|---|---|
| Agent action/control and interaction lifecycle | 25 | Mixed; likely real regressions at the observation/action boundary | Reconcile action selection with render-wait, dialogue ownership, movement confirmation, and post-warp settling contracts; add focused state-transition tests first |
| World navigation and transition geometry | 12 | Mixed; likely real navigation regressions plus changed transition expectations | Compare expected transition source/destination geometry against ROM metadata and verify one-step warp semantics before updating tests |
| Emerald opening and capability lifecycle | 20 | Mixed; high priority product behavior | Trace the opening sequence from truck through clock/rival events; fix repeated input and stale-state handling before changing assertions |
| Campaign controller/execution/recovery | 8 | Mixed; likely orchestration contract drift | Define objective ownership, completion clearing, recovery suspension, and tactical-loop release as explicit invariants |
| Readiness diagnostics/scheduler | 10 | Resolved | Fixtures now provide explicit observation availability; policy preserves UNKNOWN for absent authoritative observations |
| Resource policy/runtime | 7 | Resolved | Pure policy and recovery interaction lifecycle now pass with the current lazy observation/ownership model |
| Console diagnostics | 1 | Likely stale mock shape | Make the test fixture represent the current diagnostic sink contract |

The first five clusters should be treated as product work, not quarantined
test failures. The readiness cluster is the clearest low-risk cleanup: the
current implementation intentionally refuses to continue when authoritative
overworld/resource observations are unknown, while several older tests omit
those availability fields and expect the former optimistic behavior. Those
tests should be updated only after confirming the UNKNOWN contract is desired.

Recommended triage order:

1. Fix the readiness fixtures/contracts and keep the unit collection guard
   warning-free.
2. Isolate the agent interaction lifecycle and world-transition failures into
   small deterministic tests; establish whether each expectation is stale or
   exposes a regression.
3. Repair Emerald opening/capability sequencing, using trace output to verify
   that every input is followed by an observation before another input.
4. Reconcile campaign objective/recovery ownership and resource runtime
   generators.
5. Enable behavioral unit CI only when the remaining failures are either fixed
   or explicitly marked as known product gaps with issue-level tracking.

### Current validation (2026-08-29)

Under the configured Python 3.12.13 environment, the current unit tier passes
with **958 passed, 38 deselected** (`pytest -q -m unit`). The deselected tests
are the ROM/native-emulator tier. The user has also confirmed a fresh no-save
live run from startup through Petalburg City. Neither result validates the
first-badge P0 exit criteria, process-startup event replay, or full-game
completion.

## Immediate next actions

1. Fix runtime event-store replay.
2. Close the unsupported early-objective execution boundary.
3. Define and validate first-badge completion.
4. Add a behavioral unit-test CI job once the current baseline is triaged, so
   CI measures regressions rather than recording a known-red baseline.
5. Add a native mGBA/ROM-backed emulator job with private fixture
   provisioning.
6. Add first-badge fixture/live validation and a restart/resume scenario.
7. Begin preparation and campaign battle integration.
