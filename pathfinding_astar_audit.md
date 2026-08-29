# RIVai overworld pathfinding audit

Last reviewed: 2026-08-29

Status: investigation with follow-up implementation. Production navigation
still uses the default `algorithm="dijkstra"`; A* is available only when
explicitly requested by a benchmark or diagnostic caller. Since the original
audit, ordinary semantic cross-map goals use a global observed-world search,
multi-target activation searches share one frontier, and semantic cross-map
plans have an in-process cache. No emulator was run for this audit or its
follow-up validation.

## Architecture

There are two pathfinders:

* `modules/navigation.py::GoalAwareNavigator.plan` is the current goal-aware
  overworld search used by `agent_control.evaluate_goal`. It is a priority
  queue search whose `g` key is a lexicographic tuple, so it is Dijkstra rather
  than FIFO BFS. Its state is `(Location, facing)`, where `Location` is
  `(map_id, (x, y))`. Facing is retained whenever it is observed; explicit
  turn transitions are emitted for interaction and warp goals, while ordinary
  MOVE plans charge an implicit turn in their metrics.
* `modules/map_path.py::calculate_path` is a separate legacy ROM pathfinder
  used by older bot modes. It uses a scalar traversal cost plus Manhattan
  priority (and includes elevation, surf, waterfall, bike, forced movement,
  dynamic collision, scripted-event, and warp costs). It is not the search
  responsible for the goal-evaluation timing described in the milestone.

For the goal-aware planner, NORMAL minimizes, in order:

`(encounter opportunities, total route cost, movement actions,
encounter-terrain moves, encounter-terrain turns, non-encounter turns)`.

An encounter opportunity is a move onto encounter terrain or a direction
change while standing on encounter terrain. SEEK sets encounter contribution
to zero and uses unit movement cost; it does not inherit NORMAL's avoidance
penalty. Interaction sources are required to be present, unblocked, and
correctly faced. Warp entries, directional arrow warps, escalator activation
source tiles, and their required facing are represented by
`WarpObservation` and are preserved by the planner.

For ordinary semantic cross-map goals, planning still consults
`WorldMapGraph`, but then builds a lazy observed-world overlay and runs one
global state search across the reachable maps. It no longer evaluates only the
first map edge with one fresh search per activation side. The planner preserves
`(Location, facing)`, directed transitions, and the lexicographic encounter
objective while reconstructing the executable crossing actions. Explicit
observed map-connection goals retain a local approach path where the runtime
transition is already known.

The lower-level `plan_many_locations()` API also supports shared-frontier
search for multiple activation targets. Its current compatibility path is
exhaustive rather than candidate-pruned, but it avoids repeating the same
state search for every target.

## Profiling

`modules/profiler.py` now has disabled-by-default detailed records. Set
`context.debug_profile = True` and inspect `pathfinding_searches()` (or use
the existing profile snapshot) to obtain algorithm, map/start/goal, expanded
and generated nodes, frontier maximum, duration, result, encounter count,
route cost, candidate count, and cross-map aggregate fields. The cross-map
fields include candidates considered, searches performed, aggregate duration,
fastest/slowest search, and aggregate expanded nodes. When disabled, no
per-search record or timer is allocated.

For the legacy explicit-candidate path, candidate and activation-side counts
still describe attempted terminal states and can differ from search count. For
ordinary semantic cross-map goals, the profiler records the regular global
search instead; the cross-map candidate-evaluation profiler remains empty in
the current semantic fixture. This distinction is important when comparing
the earlier approximately 471 ms pathfinding report with current traces.

## Heuristic correctness

For the goal-aware lexicographic objective, an admissible A* lower bound is
`(0, Manhattan distance, 0, 0, 0, 0)` on a warp-free local graph. The primary
component must be zero: no positive encounter lower bound is generally safe,
because a detour may avoid all encounter opportunities. Every ordinary move
has secondary cost at least one in the current model, and turns are
non-negative, so Manhattan is a lower bound for the secondary component.
Facing does not invalidate it because turns do not change position.

Zero-cost warp edges invalidate Manhattan consistency (a warp can reduce
distance by more than its cost), so the experimental implementation returns a
zero heuristic for worlds containing warps. Interaction goals use the minimum
Manhattan distance to their known activation positions; incomplete geometry
also returns zero. This is conservative and cannot trade an encounter
opportunity for route length. It is intentionally not the heuristic used by
production.

## Deterministic comparison

The fixtures in `tests/test_navigation_astar_benchmark.py` cover named
development geometries for Littleroot → Route 101, first/final grass patches,
approaching May, escalator traversal, facing-constrained interaction,
cross-map warp candidate evaluation, and SEEK. They are abstract deterministic
world fixtures because this repository does not contain emulator-backed replay
states for those routes.

Thirty repeated local runs on four representative fixtures produced:

| algorithm | average nodes | worst nodes | average search ms | worst search ms |
|---|---:|---:|---:|---:|
| Dijkstra | 173.0 | 486 | 1.160 | 3.778 |
| A* | 40.0 | 86 | 0.453 | 1.162 |

The four cases were a straight route, first grass detour, final grass
geometry, and a larger detour map. A* selected the same action sequences and
the same encounter and route metrics in all regression fixtures. These are
local interpreter timings, not emulator timing claims.

## Findings and recommendation

1. A* can preserve the exact NORMAL and SEEK objective with the conservative
   heuristic above, but its useful heuristic is suppressed around warps.
2. The measured local node reduction is real in deterministic fixtures. The
   shared frontier now removes repeated local searches for multiple activation
   targets, while global semantic planning handles ordinary cross-map goals in
   one state search.
3. Candidate pruning and reverse/multi-source search remain unimplemented.
   Semantic cross-map result caching is now present, but its validity depends
   on the observed-world cache key and session lifetime.

Recommendation: **D) optimize and validate the cross-map architecture first**
has been partially executed through global planning and shared-frontier
search. Next, validate cache invalidation and live transition semantics, then
reassess **C) use A* only for warp-free local searches**. Keep Dijkstra as the
production default until emulator-backed route replays verify any A* use and
all interaction/encounter semantics.

## Cross-map audit milestone

The live call chain is:

`select_action(observation)` → `evaluate_goal(observation)` →
`NavigationWorld.from_overworld(...)` → `plan_with_world_navigation(...)`.

`evaluate_goal` constructs a `NavigationWorld` from the current observation.
Static tile objects and the map-level `WorldMapGraph` are reused through their
existing caches; dynamic blocked coordinates, warps, triggers, bindings, and
facing come from the current observation.

For ordinary semantic cross-map goals, `plan_with_world_navigation()` builds a
lazy global world overlay and `GoalAwareNavigator.plan()` searches the exact
reachable state graph. It does not stop after evaluating only the first map
edge. The search preserves facing, directed warps, interaction predicates,
and the lexicographic encounter objective. A completed semantic route may be
reused when the observed-world cache key is unchanged.

The explicit candidate compatibility path still derives activation positions
from the graph's source coordinates and runtime warp observations. When it is
used, `plan_many_locations()` searches all requested activation targets from
one shared `(Location, facing)` frontier, then the caller adds candidate-
specific facing/transition actions and ranks complete plans with
`navigation_candidate_key()`. It no longer creates one fresh search per
activation side or chooses a winner by action count alone.

Candidate pruning is still absent: shared search is exhaustive for the
requested targets, and the candidate diagnostics continue to report zero
rejected, pruned, or dominated candidates. Ordinary semantic cross-map goals
produce one regular pathfinding record rather than a per-candidate audit
record; the current semantic fixture has no `cross_map_goal_evaluations()`
records.

## Bounds and pruning analysis

Safe local lower bounds are available for a warp-free graph: encounter
opportunities have lower bound zero, and ordinary movement cost has lower bound
the Manhattan distance when all traversable moves cost at least one. Turn cost
also has lower bound zero. These are the same conservative bounds used by the
experimental A* reference. Facing cannot be discarded from the state because
stationary turns can create encounter opportunities and directional activation
requires a final facing.

Those bounds do not currently prove a candidate dominated before a complete
candidate search. A candidate may take a longer geometric route and still have
fewer encounter opportunities; NORMAL cannot trade that primary value for
distance. A zero encounter lower bound also cannot reject a candidate after a
best zero-exposure route unless its secondary lower bound is proven worse than
the best zero-exposure cost. For warp candidates, zero-cost transitions and
directional source requirements make a simple destination-map Manhattan bound
unsafe. Interaction source requirements, blocked sources, escalators, and
stationary encounter-terrain turns further prevent using coordinate distance
alone as a proof.

Consequently the measured current safe-pruning percentage is 0%. The new
diagnostics deliberately report zero rejected, pruned, and dominated
candidates; they are measurements of the current architecture, not claims that
future proofs are impossible. Any future pruning must compare a proven
lexicographic lower bound against the incumbent, including encounter count
first and route cost second.

## Ordering, reuse, and multi-source search

Candidate ordering is still promising only if it finds a strong incumbent
early. The signals listed in the milestone—geometry, destination distance,
exposure estimates, and prior success—are not sufficient proofs by themselves.
They can be benchmarked as orderings, but cannot change semantics unless used
only to order complete evaluation. The current shared-frontier path reduces
the representative fixture from six independent searches to one exhaustive
search; this is state reuse, not candidate pruning.

The existing static tile prewarm is safe only for immutable map tile metadata.
Dynamic blocking remains an observation-time overlay. The world graph cache is
safe under its ROM-id key and static map metadata. Semantic cross-map plans now
have an in-process cache keyed by player location/facing, movement capability,
semantic target, encounter settings, dynamic blocked coordinates, hazards, and
runtime transition identity/destination. The cache is cleared at an
emulator/session boundary, but it is not a general cache for every local or
explicit-candidate route and does not yet include a planner-version field.

Reverse search is not yet safe as a drop-in replacement. The state must
include facing, and reversing an interaction requires preserving source-
location and required-facing predicates. Reversing encounter costs is
possible in principle only with the exact action semantics for movement and
stationary turns. Directional warps and escalators are directed transitions,
not ordinary undirected edges. Forward multi-target search is now implemented
for activation positions, but a reverse multi-source search would still need a
new transition model preserving all of those predicates.

## Repeated benchmark result

The existing deterministic fixtures cover Littleroot/Route 101-style local
routes, grass detours, multi-side interaction, facing-constrained interaction,
escalator activation, cross-map candidates, and SEEK. The local A* comparison
still reports the historical 30-run interpreter measurements below:

| algorithm | average nodes | worst nodes | average search ms | worst search ms |
|---|---:|---:|---:|---:|
| Dijkstra | 173.0 | 486 | 1.160 | 3.778 |
| A* | 40.0 | 86 | 0.453 | 1.162 |

The current shared-frontier benchmark reports six independent local searches
and 59 expanded nodes for the old per-target model, versus one search and six
expanded nodes for `plan_many_locations()` (current local run: 0.087 ms for
the shared search). The semantic cross-map fixture likewise records one
global search for both algorithms and no per-candidate evaluation records.
These are local interpreter measurements, not emulator timing claims. No
emulator was run.

## Conclusions and implementation sequence

1. Ordinary semantic cross-map goals now use one global state search, with an
   in-process semantic-route cache for unchanged effective inputs. The
   explicit-candidate compatibility path uses one shared frontier rather than
   one fresh search per activation side.
2. Safe-pruning rate remains 0%. Shared-frontier search removes duplicate
   state exploration without proving any candidate dominated; candidate
   pruning and reverse search remain open work.
3. Safe bounds: zero encounter exposure, Manhattan movement lower bound only
   on warp-free graphs, and zero turn lower bound. They are not yet sufficient
   for general cross-map candidate pruning.
4. Candidate ordering has no current material pruning benefit because no safe
   incumbent bound exists. It should be benchmarked only as an experimental
   ordering.
5. Static tile and map-graph reuse already exists. Semantic cross-map route
   caching is now present, with explicit session-boundary invalidation; a
   general local-route cache is not.
6. Forward multi-target search is implemented for activation positions, while
   reverse search is not presently viable without extending the
   state/transition model for facing, turns, encounter exposure, interaction
   predicates, and directed warps.
7. Cross-map A* still does not reduce calls when the conservative warp
   heuristic is zero. Warp-free A* gains do not by themselves justify enabling
   A* in production.
8. Recommended next sequence: (a) collect profiler data on realistic frame
   traces; (b) validate semantic-cache invalidation and transition behavior;
   (c) benchmark safe candidate orderings; (d) design proof-backed candidate
   pruning or reverse search; (e) reassess production A* after emulator
   validation.

The follow-up changed cross-map semantic planning, added shared-frontier
search, corrected candidate ranking to use the canonical navigation objective,
and added semantic route caching. No production A* selection or speculative
candidate-pruning rule was enabled; Dijkstra remains the production default.

## Focused architecture audit (2026-08-16)

This section preserves the 2026-08-16 baseline. The cross-map call-path and
candidate-selection findings below were superseded by the follow-up shared
frontier/global-search work described in the sections above; remaining
open-work statements are retained where they still apply.

### Expensive call graph

```text
AgentControlLoop / caller
  -> select_action(observation)
       -> evaluate_goal(observation)
            -> NavigationWorld.from_overworld(observation.overworld)
                 -> static tile cache lookup / optional prewarm
                 -> optional _DynamicTileMapping wrapper
            -> plan_with_world_navigation(world, start, goal)
                 -> WorldMapGraph.route(...)
                 -> lazy global-world overlay
                 -> GoalAwareNavigator(world).plan(...)
                      -> lexicographic Dijkstra over (Location, facing)
                 -> append directed transition / score final route

The explicit-candidate compatibility path instead calls
`plan_many_locations()` once to share the local frontier across activation
targets.
```

Separate paths also matter when measuring a frame: `_overworld_actions` can
construct a world while enumerating fallback actions, and
`AgentActionExecutor._cached_decision` constructs one when validating a
completed cached interaction source. These are distinct from the normal
`select_action` planning call. `select_action` itself invokes one
`evaluate_goal`; tests or higher-level callers that call `evaluate_goal` and
then `select_action` evaluate twice.

### NavigationWorld ownership and reuse

`NavigationWorld` is a frozen dataclass containing:

| Field | Origin | Changes between goals on one observation? |
|---|---|---|
| `tiles` | static tile cache, optionally dynamic-block overlay | No, if the observation is identical |
| `warps` | current observation | No |
| `triggers` | current observation | No |
| `bindings` | current observation | No |
| `facing` | current observation | No |

The static tile dictionary is already reused by map identity and tile-tuple
identity. A fresh construction therefore usually allocates only the world
wrapper; with dynamic blocking it also allocates a lightweight mapping
wrapper. The world is safe to share across multiple goal evaluations made
against the same immutable `OverworldObservation`, provided no consumer
mutates the mapping or substitutes a different facing/obstacle snapshot.
Goals and encounter mode live in `NavigationGoal`, not in the world.

This is a correctness-preserving architectural seam, but its direct savings
are bounded: static tile materialization is already cached. A three-goal
model benchmark made 90 plans over 30 repetitions and produced 90 world
constructions in the separate model versus 30 in the shared model. Both
models made 90 searches and expanded 3,840 nodes; wall time was 23.68 ms
versus 24.00 ms in one local run, so the timing difference is noise at this
fixture size. The measurable win is allocation/construction count, not
pathfinding work. In the normal one-goal `select_action` path there is no
three-goal loop to collapse.

### Repeated candidate searches

The old per-edge implementation attempted activation sides one at a time.
The current explicit-candidate path groups those targets and calls
`plan_many_locations()` once from the shared `(start location, world.facing)`
state. The search is exhaustive for the requested targets, so unreachable
targets are still accounted for without repeating the state expansion. In the
current benchmark this reduces six independent searches and 59 expanded nodes
to one search and six expanded nodes.

Ordinary semantic cross-map goals take the global-world path instead and
record one global search. Candidate-specific transition actions are still
added after local target reconstruction where needed.

### Cross-map winner objective mismatch

The former cross-map winner rule selected by action count alone:

```python
min(candidate_plans, key=lambda item: len(item[1].actions))
```

That could violate NORMAL when a longer route avoided an encounter. The
current selector uses `navigation_candidate_key()` and ranks complete plans by
the canonical objective:

| candidate | local actions | encounter opportunities | route cost |
|---|---:|---:|---:|
| A | 1 | 1 | 1 |
| B | 5 | 0 | 8 |

The focused ranking test verifies that candidate B sorts ahead of A under
NORMAL. This correctness issue is resolved in the current worktree; future
candidate ordering or pruning must preserve the same ranking tuple.

### Multi-target search feasibility

A forward multi-target search is now implemented as
`GoalAwareNavigator.plan_many_locations()`. It retains `(Location, facing)` as
the state, preserves the encounter objective, reconstructs a plan for each
reachable target, and leaves candidate-specific facing/transition actions to
the caller. The implementation is intentionally exhaustive: it does not prove
arbitrary candidates dominated and therefore does not perform candidate
pruning.

A reverse multi-source search would still require a new reverse transition
model for directed warps, dynamic blocked tiles, facing-dependent turns, and
interaction predicates. That remains a design opportunity, not an implemented
optimization.

### Route-cache feasibility

Caching a completed plan is possible only for an immutable effective
observation. The ideal complete key must include:

```text
map/static tile identity and map-graph identity
player location and facing
dynamic blocked coordinates
warps: entry, destination, activation, required facing, source locations
triggers: locations, activation/navigation locations, requirements, target map
bindings and runtime/static availability
goal type, destination/trigger, constraints, encounter mode
search algorithm and relevant planner version
```

The current semantic cross-map cache includes player location/facing, movement
capability, semantic target, encounter settings, dynamic blocked coordinates,
hazards, and runtime transition identity/destination. Static tile metadata is
treated as process/session-stable and the cache is cleared at an emulator or
session boundary. This is sufficient for the current semantic route use, but
it is not a complete general local-route key: it does not include an explicit
planner-version field and does not cache every explicit-candidate route.

### Benchmark matrix and status

| Alternative | Calls / nodes | Route semantics | Result |
|---|---:|---|---|
| Current local Dijkstra | fixture-dependent; production baseline | baseline | production default |
| Shared world model | same 90 calls / 3,840 nodes as separate-world model | identical | safe, small allocation win |
| Shared-frontier multi-target search | old model: 6 calls / 59 nodes; current model: 1 call / 6 nodes in the representative fixture | identical target plans and candidate ranking | implemented |
| Semantic cross-map cache | repeated unchanged semantic requests reuse one plan | identical while the effective cache key is unchanged | implemented with session-boundary invalidation |
| Reused exact local search result | would reduce identical repeated `(world,start,goal,mode)` calls to one | identical only for a complete snapshot key | not implemented generally |
| Reverse/multi-source search | not implemented | requires a reverse transition model | defer |
| Experimental A* | warp-free: 173→40 average nodes in existing 30-run comparison; warp case: 32→32 | identical in fixtures | do not enable by default |

The shared-world row is an architectural benchmark. The shared-frontier and
semantic-cache rows describe current implementation. The benchmark tests
verify identical actions, encounter metrics, and route costs for Dijkstra/A*
fixtures, and the current shared-frontier test verifies the reduction in
searches and expanded nodes. No emulator-backed performance result exists.

### Recommended sequence

1. **Complete:** fix and regression-test the cross-map NORMAL winner objective;
   candidate ranking now uses the canonical navigation objective.
2. **Complete for the current candidate path:** thread a shared frontier
   through multi-target activation evaluation; benchmarked search count and
   node reductions are recorded above.
3. **Partial:** collect profiler data on realistic frame traces. Detailed
   search and cross-map diagnostics exist, but no emulator-backed performance
   characterization has been completed.
4. **Partial:** validate semantic-cache invalidation and determine whether a
   broader complete snapshot key is worthwhile; the current cache is limited
   to semantic cross-map plans.
5. **Complete for forward activation targets:** multi-target search preserves
   the required state and candidate ranking. Reverse search remains open.
6. **Open:** reassess warp-free A* after live transition validation. Its
   deterministic local gains do not reduce cross-map calls when the warp
   heuristic is zero.

Do not enable speculative candidate pruning, candidate reordering as if it
were pruning, reverse search, incomplete general route caching, production A*,
per-frame logging, or perception changes based only on deterministic fixtures.
Perception is a separate substantial cost and emulator traces are still needed
to establish production performance.

### Validation

The current unit tier passes **958 tests**, with **38 emulator tests
deselected**, under the configured Python 3.12.13 environment
(`pytest -q -m unit`). The focused A* benchmark module passes all six tests and
reports the current shared-frontier comparison of six searches/59 expanded
nodes versus one search/six expanded nodes. No emulator was run, so no
emulator-performance claim is made. The navigation follow-up changed
semantic cross-map planning, shared-frontier search, candidate ranking, and
semantic route caching; Dijkstra remains the production default.
