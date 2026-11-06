# Nuzlocke rules coverage

Comparison of the Bulbapedia "Nuzlocke Challenge" rule set against this
project's plan, configuration, and code. Reviewed 2026-11-05 from the page text
supplied by the project owner (the site blocks automated fetching).

Status words:

- **Specified**: stated in `PROJECT_PLAN.md` or `docs/`.
- **Implemented**: rule or behavior code was found. This review read the rule
  module (`modules/nuzlocke/rules.py`, `rule_config.py`) and the config
  template directly; other "implemented" entries come from code searches and
  have not each been exercised end to end.
- **Gap**: neither specified nor implemented.

Scope decisions from the project owner:

- ROM hacks and randomizers are **not supported**. Encounter tables, maps, and
  scripts are read from an unmodified ROM.
- Every rule beyond the two basic ones is **optional** and must be
  configurable. Optional rules default to off unless listed in
  `DEFAULT_NUZLOCKE_RULES` (`modules/nuzlocke/rule_config.py`).

## Basic and near-universal rules

| Page rule | Specified | Implemented |
| --- | --- | --- |
| Limited encounters (first wild Pokémon per area) | Yes | `one_encounter_per_area` |
| Double battle or horde: choose one | No | Not needed; Gen 3 has no wild double battles |
| Dying: fainted Pokémon are dead | Yes | `fainting` |
| Revival items and abilities forbidden | **Gap** | No prohibition found |
| Mandatory nicknames | **Gap** | Behavior exists (captures are nicknamed); not a configurable rule |
| "Met in" confirmation | No | N/A; area identity comes from ROM map data (see P3 item 2) |
| Gift Pokémon as separate encounters | Listed as an encounter method; semantics undecided | No |
| No resets | **Gap** | Not as a rule; legacy soft-reset hunting modes still exist in the repo |
| No cheating | No | No cheat support exists |
| Full wipe on white-out | Yes | `whiteout_failure` |
| No outside trading, trade-evolution clause | **Gap** | No trade support exists |
| Boxing dead Pokémon | Not stated | A "deposit fainted party at PC" routine exists |

## Optional rules

| Page rule | Specified | Implemented |
| --- | --- | --- |
| Set Mode Clause | **Gap** (config template documents it) | `set_battle_style`, optional |
| Level Cap | Yes (Roxanne now; all gyms in P2) | `level_cap` |
| Species/Dupes Clause | Was stale; corrected in the plan | `species_clause`, including evolutionary branches; no per-area duplicate cap |
| Slow Start | Yes (activation boundary and pre-ball encounters) | Default boundary is Pokédex receipt |
| Shiny Clause, Shiny Replacement | Listed as missing | No |
| HM Helper | **Gap** | No |
| Safari sectors; areas split by story progress | Safari listed as a method; area rule not specified | No |
| Random Starter | **Gap** | Random starter selection exists |
| No Items, No Heal Items, No Held Items | Generic "healing/item restrictions where configured" | No |
| No Evolving, No Child Support, No Exp Share, No Candy | **Gap** | No |
| Limited Pokémon Centers, No Buying, Limited Balls | Ball purchasing only | Ball thresholds are configurable |
| Fair Fight, Caught Only, One Per Gym, Ban List, Monotype, Notepad, Giftlocke, No Repels | **Gap** | No |
| Limited Training, Challenge Mode, Memory Only | N/A | Not applicable to Gen 3 or to an autonomous bot |

## Variants

Out of scope: Soul Link (two-player), Wonderlocke, Egglocke, Wedlocke,
Alphabetlocke, Lorelocke, Ballocke, and randomizer or ROM-hack runs. Hardcore
Nuzlocke is a bundle of item and over-leveling restrictions and is covered by
implementing those optional rules individually.

## Optional-rule backlog

Each item becomes a `CampaignRuleId` entry, off by default, enforced at the
relevant action boundary, and replayable from the event history (P3 exit
criteria). Proposed identifiers:

| Proposed id | Behavior when enabled |
| --- | --- |
| `shiny_clause` | A shiny may be caught outside the first-encounter limit and is not released on faint |
| `no_resets` | Forbid voluntary soft resets and save-state loads that undo progress; process-restart resume from a save state stays allowed |
| `hm_helper` | Allow a catch-and-release helper for a required field move, outside the one-per-area limit |
| `species_clause_cap` | **Implemented** as the config option `species_clause_max_duplicates_per_area` (default `null` = unlimited). After that many skipped duplicates in one area, the next encounter in the area counts even if it is a duplicate. The per-area count is recorded on the battle-start event so it survives replay and restart. |
| `no_revives` | Forbid Revive items and equivalent effects |
| `gift_static_separate` | Count gift and static encounters separately from wild ones |
| `mandatory_nicknames` | Make the existing nickname behavior an explicit, configurable rule |
| `no_heal_items`, `no_items`, `no_held_items` | Item restrictions |
| `no_evolving`, `no_exp_share`, `no_rare_candy`, `no_daycare` | Leveling restrictions |

`no_resets` is deferred: the bot resumes from save states, which must be
distinguished from undoing progress. When built, it should start as a warning,
not a block: the bot never rewinds itself in campaign mode, and on resume it
can compare the game against its own event history (a game that is *behind*
the history was rewound).

## Not tied to any rule

The page's strategy tips are already partly covered by the plan: avoiding
non-required trainers, preparing for mandatory battles, not under-leveling, and
critical-hit modeling. Baiting, PP stalling, sacking, and Repel manipulation
are not planned.
