# Campaign gate compilation seam

`modules.nuzlocke.gate_discovery` produces source evidence. It does not decide
that a flag or trainer is a campaign requirement. `gate_impact` can annotate a
candidate after a collision, warp, or dynamic object observer supplies explicit
edge identities. `gate_compilation` is the report-only boundary that turns a
reviewed candidate into an existing campaign contract.

Promotion requires an explicit override:

- access gates name the `(map_group, map_number)` and at least one required
  badge, campaign fact, or capability;
- trainer gates name the checkpoint and affordance identities, supply a resolved
  roster, classify `mandatory` explicitly, and supply a clearance fact for a
  mandatory trainer;
- both require high-confidence route cut-set evidence and verified semantics.

The resulting `CampaignAccessRequirement` or `TrainerCheckpoint` retains the
source candidate ID, source location, source revision, guard expression,
producer actions, route confidence, and route observation references. A
candidate selected for promotion that lacks these facts produces a
`GateCompilationResult.diagnostics` entry and cannot affect the planner.

For the reviewed Wally candidate, the source meaning is explicit. In
`data/maps/PetalburgCity/scripts.inc`, the state-0 transition handler moves
Gym Boy (local ID 8) to `(5, 11)`; the state-2 frame script starts
`PetalburgCity_EventScript_WallyTutorial`; and that tutorial writes
`VAR_PETALBURG_CITY_STATE = 3` before warping the player to the gym. The
At the pinned revision these are the source statements at lines 15, 28, and
47 respectively. The candidate's `map_script_2 ... 2` guard identifies the
tutorial trigger. The access requirement uses the later
`PETALBURG_CITY_STATE >= 3` traversal boundary, while the campaign completion
fact additionally waits for the gym return state observed by the
campaign-state reader.

The reviewed Emerald manifest lives in
`modules.nuzlocke.emerald_gate_manifest`. It pins game `emerald`, source
revision `5eff78649e7170a877b961ef0b3da13b81a16038`, scanner version
`emerald-source-1`, and exact source fingerprints for the current reviewed
Wally access candidate and Route 102 Calvin checkpoint. A stale report,
changed source location, changed guard, or changed reviewed metadata is
rejected before compilation.

`compile_emerald_gate_manifest()` is still report-only. It accepts assessed
route impacts directly, or computes them from a `GateEdgeEffect` when the
caller supplies a graph, origin, and destinations. An edge effect without
that route context remains diagnostic-only. The existing hand-verified
Emerald registry remains authoritative until these route effects and live
completion observations are validated for runtime use.

## Live edge validation

The live recorder is an evidence collector, not a gate classifier. A caller
must arm a candidate, an exact edge, and a source-derived event-variable
predicate before the bot starts. The recorder reads that variable from the
active ROM save before accepting an attempt, classifies the predicate as
satisfied or unsatisfied, and accepts an observation only when it has both an
issued input and a terminal boundary: `blocked` or `open`. Each observation
includes the candidate ID, edge identity, source kind, predicate, raw variable
value, action kind, emulator frames when available, and session ID. An
unavailable or unreadable variable, an unissued action, and an unclassified
transition remain diagnostics.

The command-line entry point is:

```text
python pokebot.py <profile> --gate-live-config <config.json>
```

The configuration is deliberately small and reviewable:

```json
{
  "session_id": "wally-live-2026-09-14",
  "output": "/tmp/nuzbot-wally-live.json",
  "bindings": [
    {
      "candidate_id": "emerald:script_event:...",
      "edge": {
        "source_map": [0, 0],
        "destination_map": [0, 19],
        "kind": "map_connection",
        "warp_index": null
      },
      "guard": {
        "symbol": "PETALBURG_CITY_STATE",
        "operator": "greater_or_equal",
        "value": 3
      },
      "source_kind": "transition"
    }
  ]
}
```

With a binding loaded, select the opt-in `Gate Edge Validation` mode. Run it
from a disposable copy of a profile and pass `--no-save-state`; the mode makes
one ordinary navigation attempt and returns to Manual mode after the terminal
observation. A destination-map observation is an open boundary for a map
connection even when the ROM's entry coordinate differs from the source graph
coordinate. That preserves the map transition evidence without pretending the
coordinate mapping is exact.

The live evidence describes a physical Petalburg westward traversal boundary,
not a closed static map connection. Two disposable source-derived fixtures
were run from the saved checkpoint
`profiles/_p0_wally_resume_probe/states/2026-08-29_17-53-19.ss1`:

- `/tmp/nuzbot-wally-fixture.json` set
  `PETALBURG_CITY_STATE = 0`, placed Gym Boy (local ID 8) at `(5, 11)`, and
  started the player at `(29, 12)`. The issued westward traversal terminated
  `blocked: true` with an unsatisfied `PETALBURG_CITY_STATE >= 3` predicate.
- `/tmp/nuzbot-wally-post-fixture2.json` set
  `PETALBURG_CITY_STATE = 3`, started the player at `(13, 13)`, and retained
  Gym Boy at `(12, 15)`. The issued traversal reached Route 104 with
  `blocked: false` and a satisfied predicate.

Both artifacts contain schema version 2, the raw variable value, the source
predicate, dynamic ROM object positions, an issued action, a terminal outcome,
and empty `pending` and `diagnostics` arrays. They are deterministic replay
evidence, not untouched organic save captures. The organic pre-scene capture
remains an open confirmation item. The Route 102 Calvin candidate still needs
geometry-backed route cut-set evidence and a stable post-battle clearance
observation.
