# RIVai overworld pathfinding audit

Status: investigation only. Production navigation still uses the default
`algorithm="dijkstra"`; A* is available only when explicitly requested by a
benchmark or diagnostic caller. No emulator was run for this audit.

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

Cross-map planning first gets a map-level route from `WorldMapGraph`. For its
first edge it enumerates every source boundary coordinate, then tries up to
four adjacent activation positions for an ordinary warp (or the explicit
activation source for an escalator-like observation). Each reachable candidate
gets a fresh local search; the shortest local action plan is selected. Thus a
single goal evaluation may perform one search per candidate, plus failed
activation-side probes. It does not reuse local search state.

## Profiling

`modules/profiler.py` now has disabled-by-default detailed records. Set
`context.debug_profile = True` and inspect `pathfinding_searches()` (or use
the existing profile snapshot) to obtain algorithm, map/start/goal, expanded
and generated nodes, frontier maximum, duration, result, encounter count,
route cost, candidate count, and cross-map aggregate fields. The cross-map
fields include candidates considered, searches performed, aggregate duration,
fastest/slowest search, and aggregate expanded nodes. When disabled, no
per-search record or timer is allocated.

The candidate count is not necessarily the search count: activation probing
can produce multiple searches for one warp candidate. This is the principal
architecture finding relevant to the reported approximately 471 ms of
pathfinding across three goal evaluations.

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
2. The measured local node reduction is real in deterministic fixtures, but
   cross-map planning can still dominate because it repeats searches for
   candidate activation positions.
3. The current architecture performs no candidate pruning, result caching,
   equivalent-state reuse, or shared multi-candidate search. Given the stated
   471 ms / 490 ms / three-evaluation profile, reducing the number of searches
   is likely to provide a larger and lower-risk first gain than replacing each
   search algorithm.

Recommendation: **D) optimize the cross-map/candidate-search architecture
first**, then reassess **C) use A* only for warp-free local searches**. Keep
Dijkstra as the production fallback until emulator-backed route replays verify
that candidate pruning and any A* use preserve all interaction and encounter
semantics.

## Cross-map audit milestone

The live call chain is:

`select_action(observation)` → `evaluate_goal(observation)` →
`NavigationWorld.from_overworld(...)` → `plan_with_world_navigation(...)`.

`evaluate_goal` constructs a fresh `NavigationWorld` for each overworld goal
evaluation. Static tile objects and the map-level `WorldMapGraph` are reused
through their existing caches; dynamic blocked coordinates, warps, triggers,
bindings, and facing come from the current observation. A cross-map goal asks
the graph for one map route and evaluates only its first edge. It does not
search all world-map routes or all outgoing edges.

For that edge, `source_coordinates` is the static candidate list. A runtime
warp with explicit activation locations contributes those locations. A normal
runtime warp contributes four adjacent activation-side states, in fixed
North/East/South/West geometry order. A source without a matching runtime warp
is searched as the source coordinate itself. Each attempted activation side
creates a new `GoalAwareNavigator(world).plan(start, ReachLocation(...))`.
The first reachable side ends that source's side loop; all source coordinates
are still evaluated. Therefore the current number of searches is:

`sum(attempted activation sides until first reachable side for each source)`.

It is not necessarily equal to either the number of warp candidates or the
number of activation sides. Searches always start at the current player state;
there are no searches from warp destinations. Candidate ordering is the graph's
source-coordinate order and has no pruning effect. The winning source is then
chosen by `len(local_plan.actions)`, after each local search has independently
optimized its own lexicographic search key. SEEK reaches the same candidate
machinery, but its local search sets encounter contributions to zero and uses
unit movement cost. No cross-map candidate is currently pruned or dominated.

The audit profiler now records one disabled-by-default cross-map evaluation and
one record per attempted activation side. It reports source-candidate count,
activation-side count, rejected/pruned/dominated counts, pathfound/no-route
counts, calls, expanded nodes, pathfinding time, total cross-map planning time,
running best encounter count, running best secondary cost, and the selected
candidate. Per-search records carry the candidate identifier. Normal operation
does not allocate these records or take profiling timers.

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

Candidate ordering is promising only if it finds a strong incumbent early. The
signals listed in the milestone—geometry, destination distance, exposure
estimates, and prior success—are not sufficient proofs by themselves. They can
be benchmarked as orderings, but cannot change semantics unless used only to
order complete evaluation. The current deterministic two-warp fixture performs
four searches in both Dijkstra and A* orderings; with no admissible candidate
bound, reordering alone cannot reduce that count.

The existing static tile prewarm is safe only for immutable map tile metadata.
Dynamic blocking remains an observation-time overlay. The world graph cache is
safe under its ROM-id key and static map metadata, while a completed candidate
route is stale when player position/facing, dynamic blocks, warps, triggers,
bindings, encounter mode, constraints, or relevant map data changes. Searches
from the same `(world snapshot, start state, target state, mode, constraints,
algorithm)` could be cached, but the current implementation does not expose a
validated snapshot key, so no route cache was added in this milestone.

Reverse or multi-source search is not yet safe as a drop-in replacement. The
state must include facing, and reversing an interaction requires preserving
source-location and required-facing predicates. Reversing encounter costs is
possible in principle only with the exact action semantics for movement and
stationary turns. Directional warps and escalators are directed transitions,
not ordinary undirected edges. A future multi-source search would need a
super-source over valid activation states and a reverse transition model that
preserves all of those predicates; the current representation has no such
reverse model. This is a design opportunity, not an implemented optimization.

## Repeated benchmark result

The existing deterministic fixtures cover Littleroot/Route 101-style local
routes, grass detours, multi-side interaction, facing-constrained interaction,
escalator activation, cross-map candidates, and SEEK. The cross-map fixture was
run for 30 deterministic repetitions per algorithm. It produced four fresh
searches per decision (two source coordinates, with failed side probes), and
the following medians on the local Python interpreter:

| algorithm | calls | expanded nodes total | pathfinding time | cross-map planning time |
|---|---:|---:|---:|---:|
| Dijkstra | 4 | 32 | 0.129 ms | 0.162 ms |
| A* | 4 | 32 | 0.160 ms | 0.193 ms |

This small warp-containing fixture correctly gives A* a zero heuristic, so it
does not reduce nodes or calls. The earlier warp-free fixtures show A* reducing
individual nodes substantially, but that is category A/C (faster or smaller
individual searches), not category B (fewer searches). The primary milestone
metric D—total goal-planning time—remains dominated by the number and shape of
candidate searches in realistic cross-map cases. No emulator was run.

## Conclusions and implementation sequence

1. Current searches per cross-map goal are the attempted activation sides per
   source, with one fresh search per attempt; the measured fixture is four.
2. Demonstrably unnecessary searches today: none proven; safe-pruning rate is
   0%. Failed side probes are repeated work, but eliminating them requires a
   validated multi-target search or a proof-equivalent cache.
3. Safe bounds: zero encounter exposure, Manhattan movement lower bound only
   on warp-free graphs, and zero turn lower bound. They are not yet sufficient
   for general cross-map candidate pruning.
4. Candidate ordering has no current material pruning benefit because no safe
   incumbent bound exists. It should be benchmarked only as an experimental
   ordering.
5. Static tile and map-graph reuse already exists. Dynamic route caching is
   possible only with a complete world-state invalidation key; it was not
   introduced.
6. Reverse/multi-source search is not presently viable without extending the
   state/transition model for facing, turns, encounter exposure, interaction
   predicates, and directed warps.
7. Cross-map A* saved no calls or nodes in the warp fixture and was slightly
   slower in this small timing sample. Warp-free A* gains do not address the
   architectural call count.
8. Recommended sequence, pending approval: (a) collect profiler data on
   realistic frame traces; (b) benchmark orderings without changing selection;
   (c) specify and test a full snapshot cache key; (d) design a proof-backed
   multi-source search; (e) only then consider production pruning or local A*.

No production algorithm, candidate order, route cache, or pruning rule was
changed by this milestone. Dijkstra remains the production default.

## Focused architecture audit (2026-08-16)

This section records the audit against the current worktree, including the
reuse and candidate-selection questions that are not answered by the local
A* comparison alone.

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
                 -> for each source coordinate
                      -> for each activation-side choice
                           -> GoalAwareNavigator(world).plan(...)
                                -> lexicographic Dijkstra over
                                   (Location, facing)
                 -> append directed transition / score final route
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

For one graph edge, the exact current search count is:

```text
sum over source candidates of
    activation-side attempts until the first reachable side
```

An ordinary runtime warp can attempt four adjacent source states. An
explicit activation-location warp (including escalators) uses its explicit
source states. A source with no matching runtime warp is searched as the
source coordinate. Every attempt starts from the same `(start location,
world.facing)` snapshot and creates a new navigator. The local goal differs
by activation coordinate, so these searches are not generally equivalent.

In the existing two-warp deterministic fixture, four calls occur: the first
side of each warp is unreachable and the second side is reachable. The
profiler reports four candidate records and four searches, with zero safe
prunes or dominance rejections. This is repeated work only in the sense that
the searches share a start and transition graph; no current invariant proves
that any one candidate has the same optimal route as another.

### Cross-map winner objective mismatch

The local search for each candidate correctly optimizes NORMAL's
lexicographic objective, but the cross-map winner is selected by:

```python
min(candidate_plans, key=lambda item: len(item[1].actions))
```

This can violate NORMAL. A deterministic sparse fixture demonstrated:

| candidate | local actions | encounter opportunities | route cost |
|---|---:|---:|---:|
| A | 1 | 1 | 1 |
| B | 5 | 0 | 8 |

The current selector chooses A, while NORMAL requires B. This is an
existing correctness bug, not a performance optimization. It was not fixed
in this milestone and no candidate ordering or pruning should be built on
the current winner rule until the objective is reviewed. A focused regression
fixture should be added when the correctness fix is scheduled; changing the
assertion now would change production semantics and is outside this audit.

### Multi-target search feasibility

A forward multi-target search is feasible in principle, but not as a simple
replacement using the current API. A safe version could put all valid
activation states into one search and retain, for each reached target, the
complete predecessor chain and metrics. It must keep `(Location, facing)` as
the state, retain stationary turns, charge encounter-terrain moves and turns,
and apply each target's source/facing predicate after the same forward
transition rules.

The existing search returns on the first satisfied goal and stores one
predecessor map. It has no result set for multiple goals and no representation
for candidate-specific completion actions (directional warp input,
escalator source/facing, or interaction completion). A reverse multi-source
search would require a new reverse transition model for directed warps,
dynamic blocked tiles, facing-dependent turns, and interaction predicates;
the current forward model cannot safely supply that. Therefore this is a
design opportunity, not a safe optimization to implement now.

### Route-cache feasibility

Caching a completed local plan is possible only for an immutable snapshot.
The minimum key must include:

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

The existing executor signature covers many runtime invalidation fields but
does not include the goal, player facing, static tile identity, or all
planner semantics as a standalone stable hash. It is therefore not a
complete route-cache key. Building and hashing the full key may also cost
meaningful time for large observations. No cache was added.

### Benchmark matrix and status

| Alternative | Calls / nodes | Route semantics | Result |
|---|---:|---|---|
| Current local Dijkstra | fixture-dependent; 4 calls / 32 nodes in the two-warp case | baseline | production default |
| Shared world model | same 90 calls / 3,840 nodes as separate-world model | identical | safe, small allocation win |
| Reused exact search result | would reduce identical repeated `(world,start,goal,mode)` calls to one | identical only for a complete snapshot key | feasible later, no cache added |
| Multi-target search | not implemented | requires new result/transition representation | defer |
| Experimental A* | warp-free: 173→40 average nodes in existing 30-run comparison; warp case: 32→32 | identical in fixtures | do not enable by default |

The shared-world and exact-reuse rows are architectural models, not
production modifications. The existing benchmark tests verify identical
actions, encounter metrics, and route costs for Dijkstra/A* fixtures. No
experimental multi-target implementation exists to benchmark honestly.

### Recommended sequence

1. Fix and regression-test the cross-map NORMAL winner objective separately;
   correctness risk is high, performance benefit is neutral-to-positive, and
   emulator validation is straightforward.
2. Thread one immutable `NavigationWorld` through any real multi-goal
   evaluation context; low complexity/risk and easy to validate, but expect a
   modest benefit because static tiles are already cached.
3. Add opt-in counters for world construction and goal identity to the
   existing profiler, then collect real frame traces. This separates
   perception, planning, candidate enumeration, search, scoring, and cached
   execution before choosing a larger change.
4. Specify a complete snapshot key and benchmark exact-result reuse on
   repeated unchanged decisions; medium complexity and high invalidation risk.
5. Design a forward multi-target search with explicit per-target completion
   records and prove equivalence against the current planner before any
   production use; potentially high benefit, highest implementation risk.
6. Reassess warp-free A* after architectural reuse. Its deterministic local
   gains do not reduce cross-map call count.

Do not optimize yet: speculative candidate pruning, candidate reordering as
if it were pruning, reverse search, incomplete route caching, production A*,
per-frame logging, or perception. Perception is a separate substantial cost
and deterministic Python fixtures cannot establish emulator performance.

### Validation

The focused unittest suite passed: 47 tests covering foundation, world
navigation, interaction/facing, encounters, diagnostics, and A* benchmark
fixtures. `pytest` is unavailable in the configured environment (`No module
named pytest`). No emulator was run, so no emulator-performance claim is
made. Production navigation files were not changed by this audit; the
worktree already contained user changes before the audit began.
