# Early Emerald state-variable audit

Date: 2026-09-15

Source: `pret/pokeemerald`, revision
`5eff78649e7170a877b961ef0b3da13b81a16038`.

This audit records event-variable meanings that can strengthen local scene
guards. The variables are observations first. They become campaign blockers or
objectives only after route impact and completion evidence are verified.

## Reviewed variables

| Variable | Source-backed values and writers | Readers and affected behavior | Classification |
|---|---|---|---|
| `VAR_BIRCH_LAB_STATE` | `2` after the Route 101 bag/starter scene; `3` after Birch tells the player to see the rival; `4` in `Route103_EventScript_RivalEnd`; `5` after the Pokédex and Poké Ball handoff | Birch's lab dispatches the starter scene at `2` and Pokédex scene at `4`; the Route 103 rival object is actionable only in the `3` interval | Strong local scene guard. `== 3` is the active Route 103 rival boundary; `>= 4` corroborates rival completion |
| `VAR_LITTLEROOT_TOWN_STATE` | `1` after meeting the rival; `2` after the twin gives the Birch rescue instruction; `3` after the Pokédex handoff; `4` after Running Shoes | Route 101 edge triggers distinguish state `0` from the rescue prompt; town layout and Running Shoes triggers use later values | Local Littleroot/Route 101 transition state. Useful for scene routing, not yet a global map-access fact |
| `VAR_ROUTE101_STATE` | `1` after the map-name setup; `2` after the Birch rescue scene starts; `3` after taking the bag and choosing the starter | Multiple Route 101 coordinate triggers block exits while state `2`; the bag script writes state `3` and warps to Birch's lab | Local scripted-scene guard. State `2` must remain conservative because the player is inside the rescue boundary |
| `VAR_LITTLEROOT_INTRO_STATE` | `1/2` truck; `3` entered the house; `4` told to set the clock; `5` on the second floor; `6` clock set; `7` told to meet the rival | Opening map scripts, stairs, furniture, Mom positioning, and post-clock arrival behavior | Opening lifecycle state. It should control opening capabilities and diagnostics, not later route reachability |
| `VAR_LITTLEROOT_RIVAL_STATE` | `2` entered rival room; `3` met rival; `4` received the Pokédex | Rival-house interactions, object positioning, and the transition that marks the rival as leaving for Route 103 | Local interaction/layout state. The existing `rival_met` fact uses `>= 3`; the raw value is retained for diagnostics |
| `VAR_OLDALE_RIVAL_STATE` | `1` at the Route 103 rival end; `2` when Littleroot consumes the Oldale handoff | Oldale rival visibility and the return-to-Littleroot layout | Local scene handoff. It does not establish general route access |
| `VAR_OLDALE_TOWN_STATE` | The pinned source comments identify this variable as unused | No verified progression consumer | Do not promote |

## Reviewed flags

| Flag | Source-backed assignment and consumer | Planning meaning | Classification |
|---|---|---|---|
| `FLAG_ADVENTURE_STARTED` | `data/maps/OldaleTown/scripts.inc:8` calls `OldaleTown_EventScript_BlockWestEntrance` while the flag is unset; `data/maps/LittlerootTown_ProfessorBirchsLab/scripts.inc:510` sets it during `GivePokedex` | The unset state places `LOCALID_FOOTPRINTS_MAN` at `(1, 11)`, which blocks Oldale's west edge; the west map connection is `MAP_ROUTE102` | Strong source-backed map-edge guard, registered as `route102:oldale_adventure_started` with `pokedex_received` as completion evidence |

## Route 103 finding

The log's repeated `ActivateTrigger('introductory_rival')` searches occur while
the player is on Route 103 and the pathfinder cannot produce an interaction
route. The ROM object is the template at `(10, 3)`, local ID 2, using
`Route103_EventScript_Rival`, with `FLAG_HIDE_ROUTE_103_RIVAL` as its object
visibility flag.

The visibility flag is an object-template condition. It does not prove that
the battle has been completed. The source completion script writes both
`FLAG_DEFEATED_RIVAL_ROUTE103` and `VAR_BIRCH_LAB_STATE = 4`, so the bot keeps
the defeated flag as authoritative completion evidence and the lab state as a
source-backed corroborating boundary. The hide flag remains visibility-only.

The active object boundary is now guarded by `VAR_BIRCH_LAB_STATE == 3` in the
semantic trigger binding. Live overworld perception and static trainer catalog
resolution withhold the rival affordance when the variable is false or
unavailable. This prevents a stale map template or stale object event from
claiming the interaction before the starter handoff or after the rival scene.

## Early route-edge findings

The pinned source also makes the first local boundary behavior explicit:

| Map and state | Source coordinates and scripts | Planning meaning |
|---|---|---|
| Littleroot Town `0` | `(10, 1)` and `(11, 1)`, `NeedPokemonTriggerLeft/Right` | The player has no Pokémon. These entry tiles invoke the twin's rescue warning and push the player back from Route 101. |
| Littleroot Town `1` | `(11, 1)`, `GoSaveBirchTrigger` | The rival scene is complete and the town dispatches the Birch rescue instruction. |
| Littleroot Town `2` | no matching Route 101 warning trigger; `GoSaveBirch` writes `2` | The rescue handoff has been accepted. This is a local scene boundary, not proof that every later map is open. |
| Littleroot Town `3` | `(8..11, 9)` and `(10..11, 2)`, Running Shoes triggers | These are optional local movement or reward triggers after the Pokédex handoff. They do not establish route ownership. |
| Route 101 `1` | `(10, 19)` and `(11, 19)`, `StartBirchRescue` | Entering the rescue scene is the intended next action. |
| Route 101 `2` | `(10..11, 18)`, `(6, 15..18)`, and `(7, 13)`, `PreventExitSouth/West/North` | The rescue boundary remains active and the listed exits must be treated as scripted blockers. |
| Route 101 `3` | `BirchsBag` writes `VAR_ROUTE101_STATE = 3` and warps to Birch's lab | The bag scene has completed; the Route 101 exit-prevention triggers are inactive. |

These coordinates are covered by a source-shaped fixture in
`tests/test_emerald_state_guards.py`. The fixture verifies exact variable
predicates, producer writes, and the `ROUTE101_STATE == 2` versus `3`
blocked/open distinction. It does not promote either variable into a global
map-access requirement; that still requires route-edge evidence from the ROM
geometry and matching live-save validation.

## Work still required

1. Add route-edge observations from the ROM geometry for Littleroot state
   `0/1/2/3` and Route 101 state `2/3`. The source fixture proves the local
   predicates; it does not by itself prove which map edges are cut.
2. Add the same source guard treatment to later script-owned scenes only when
   the source assignment and consumer are explicit. Keep unknown reads closed.
3. Use route geometry to determine whether each local state affects a map edge,
   a destination, or only an NPC layout. Do not turn the raw state facts into
   global access requirements without that evidence.
4. Extend the guarded semantic binding and source provenance pattern through
   the later Emerald story corridor, then validate each promoted boundary with
   a save and sidecar describing the same campaign point.
5. Repeat the Route 103 live run after this guard is in place. A timeout or a
   run that stops before the rival interaction is incomplete validation; it
   cannot prove the guard or the battle policy.
6. Validate the Oldale west edge with a save and sidecar that describe the
   same pre- and post-Pokédex campaign boundary. The source and map geometry
   establish the candidate, while live edge evidence remains the final
   promotion check for generated gate reports.
