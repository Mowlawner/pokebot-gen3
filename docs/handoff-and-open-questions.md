# Handoff and open questions

Second checkpoint, written 2026-11-05 so work can move to another computer. It replaces
the first checkpoint (commit `9aca98a8`). Read this first, then `AGENTS.md`, then the
documents it links. Everything here is on branch `improved-leveling`.

## 0. Do this before you move

1. **Push the branch.** When this was written, local `improved-leveling` was **14 commits ahead of
   `origin/improved-leveling`** (through `a64051ff`), plus the documentation commit that adds this file.
   Nothing from this checkpoint exists anywhere else until you run `git push origin improved-leveling`.
   Check with `git status -sb`. (The agent did not push; pushing is outward-facing.)
2. **Never commit** anything under `profiles/`, `roms/`, or save states. They are gitignored and
   must stay machine-local.
3. **Machine-local leftovers on the Windows machine** (safe to delete, not deleted): five
   throwaway profiles `profiles/diag_live_20261105`, `diag_live2_...` to `diag_live5_...`, and the
   run logs, stack dumps, and edit scripts in `%LOCALAPPDATA%\Temp\nuzbot-diag\`. Delete them
   yourself, or ask the agent to; the findings that matter are written down in this document and
   in `campaign-progression-audit-followup.md`.

### Commits since the first checkpoint (oldest first)

| Commit | What it does |
| --- | --- |
| `638c19ef` | `NUZLOCKE_CAMPAIGN_FIELD_LEAD` is always on, once per distinct decision, and lists the whole party |
| `b70e4272` | Capture planner flees when no lead plan or safe switch fits the faint bound and escape is legal |
| `d5acc465` | Route-cut guard: a trainer is promoted to must-fight only if it has a ROM checkpoint **and** exact hazard geometry (`trainer_hazard_geometry_is_exact`, `modules/overworld.py`) |
| `5aba9a15` | Records the owner decisions (first checkpoint's open questions) |
| `25a19f87`, `df890d5b` | Windows console crash fix: redirected stdout/stderr are written as UTF-8 (`modules/console.py`) |
| `ba9b3a9f` | Stall dump is a Python watchdog thread, not `faulthandler` (`modules/stall_dump.py`) |
| `bcc65f35` | Capture rationale lists each bench member with HP and why it was eligible or rejected (`Bench:`) |
| `5897d074` | Capture switch check and faint table use each bench member's **own** incoming damage |
| `9f699c50` | Records the live-run findings |
| `14ca5f4b`, `a64051ff` | `CAMPAIGN_PREPARATION_NO_LINE`: explains, stage by stage, why preparation found nothing to train on |
| `6990521f`, `f9ea3c87` | `CAMPAIGN_CONTROLLER_STALLED`: reports a mounted capability with no visible progress (real time, 3 min) |

### Test status

On the Windows machine with the project `.venv`: **18 failed, 2289 passed, 39 skipped**. The first
checkpoint's baseline on the original machine was 2258 passed, 34 skipped, no failures. The 18
failures are identical with and without this checkpoint's changes (checked by stashing):

- **4 in `tests/test_nuzlocke_fixture_state.py`** need the gitignored fixture profile
  `profiles/test_begin_nuzlocke/`, which is not on the Windows machine. Environment, not a bug.
- **14** (7 in `CampaignObjectiveTests` in each of `tests/test_nuzlocke_campaign_objectives.py` and
  `tests/test_nuzlocke_campaign_controller.py`) fail only in the full suite and pass when their file
  runs alone (the objectives file alone: 90 passed). The cause is **not established**. First check:
  run the full suite on a machine that has the fixture profile. If they pass there, they are
  environmental; if not, find the earlier test that pollutes shared state.

## 1. Where things stand

The bot starts a fresh Emerald run and reliably gets through: the opening, the starter, the intro
rival on Route 103, captures on Routes 101 to 104, Mart and Center trips, Wally's tutorial, Route 104
past Cindy and Billy, the Petalburg Woods warp, and into Aqua Grunt preparation. It also won
Calvin (Route 102) and Lyle (an optional Woods trainer). **No live run has reached the grunt fight
or the first badge.** Preparation for the grunt fails in the Woods; section 2 is the evidence.

### Validated in live runs

macOS runs 1 to 9 are summarized in `campaign-progression-audit-followup.md`. Windows runs are named
W1 to W5 here to avoid clashing with those numbers.

| Area | Evidence |
| --- | --- |
| Route 103 training, intro rival, Route 104 corridor with no trainer exposure, Woods warp | macOS runs 2 to 9, Windows W4, W5 |
| Staged trainer-safe travel fallback for preparation | macOS runs 8, 9 |
| Capture planner, plan comparison | macOS run 9; W4, W5 |
| **Capture switch using the bench member's own damage** | W5: Poochyena 8 HP led, planner switched to Mudkip "Albert" 18/24 (worst hit 6), about 98% capture, 1.8% faint |
| **Capture flee** | W4: Wingull, lead at 5/13 HP, best plan 28% capture / 72% faint, ran away safely. W5: Taillow |
| `Bench:` rationale lines | W5 |
| Console UTF-8 fix | W4, W5 ran without `PYTHONUTF8`; no encoding errors |
| Python stall dump | W4 survived four dumps; W5 |
| `CAMPAIGN_PREPARATION_NO_LINE` | W5 (first form; see the caveat in section 2) |

### Implemented and unit-tested, **not yet validated live**

- Route-cut guard (`d5acc465`): it only matters when the safe search fails; no run exercised it.
- Readiness that accepts a horizon projection with small unresolved mass, and preparation that ends
  as soon as the party has a ready line (`CAMPAIGN_PREPARATION_READY`): no run reached it.
- Capture planner's risk-accepted fallback (when escape is not legal).
- Species Clause duplicate cap (`species_clause_max_duplicates_per_area`).
- The improved trainer-readiness explanation (`a64051ff`) and the time-based
  `CAMPAIGN_CONTROLLER_STALLED` (`f9ea3c87`): both written after the last run.
- Recovery from the Woods: it timed out on a 10 s search in macOS runs 2 and 3 and did not recur;
  the cause is unexplained.

## 2. The blocking problem: preparing for the Aqua Grunt in the Petalburg Woods

Evidence is from W4 and W5. Code is in `modules/nuzlocke/resource_runtime.py` unless noted.

1. **Grass training is skipped once the area's first encounter counts as resolved.**
   `execute_campaign_preparation` passes `allow_wild=not _preparation_encounter_resolved(...)` to
   `_preparation_navigation_goal`, and that function only considers grass inside `if allow_wild:`.
   The docstring of `_preparation_encounter_resolved` says the opposite ("later wild battles remain
   legal training opportunities"). W5 logged: `grass: not used, the first encounter of this area is
   already resolved`. Note the Woods encounter was "resolved" without a Woods capture in the party.
2. **Every observed Woods trainer was rejected as "battle line is not proven safe".** That is
   `readiness.status != "ready"` in `trainer_preparation.select_trainer_source`. The W5 party was
   Poochyena Lv2, Mudkip Lv8 26/26, Wurmple Lv2, Lotad Lv4, target level 9, cap 15, with plenty of EXP
   headroom. The first `NO_LINE` record printed `win=0% catastrophic=0%` for each trainer, but that
   was **a display error**: `BattleReadiness.win_probability` is `None` when no search established the
   outcome, and `None` was shown as 0%. The record now gives each trainer's readiness status, reason,
   and the top party members' roster coverage and probabilities. **That fuller record has not been seen
   live.** To see it: run to the Woods and read `CAMPAIGN_PREPARATION_NO_LINE`.
   `assess_trainer_checkpoint` also skips, silently, any roster-versus-member pairing that raises; if every
   pairing is skipped the result looks like "no evidence". Whether that happens here is unknown.
3. **Preparation raises `BotModeError` and the controller retries forever.** The exception record is
   rate limited, so after the first two it is quiet. The party never changes and the readiness gate stays
   `PREPARE: TRAINER_BATTLE_UNREADY`. There is no backoff or escalation (first checkpoint's item 4).
4. **Silent wait on the scene trigger (W4).** After recovery the bot returned to the Woods with goal
   `ActivateTrigger devon_goods_researcher` under AVOID. That search always returns not found: the trigger
   is registered as a trainer hazard (`hazard_kind="trainer"` in `modules/trigger_bindings.py`), so
   `devon_goods_researcher:8` is in the AVOID hazard list while being the goal. Nothing is re-issued, the
   mounted capability waits, and the log stops. The idle watchdog did not cover it because it only counts
   steps with nothing mounted. `CAMPAIGN_CONTROLLER_STALLED` now reports this. The comment above
   `_objective_requires_trainer_battle` in `campaign_controller.py` says a generic `ActivateTrigger` is
   promoted to `EngageTrainer` by route-aware readiness; **that promotion did not happen** and was not
   investigated further. Under AVOID the bot never walked into an unprepared grunt fight.
5. **Lyle (W4).** The bot fought the optional Woods trainer Lyle (trainer 616, Wurmple) through an
   `EngageTrainer` goal, a "clearing trainer" training fight, and won. Whether this is intended under the
   AVOID rules is question 3 below.

## 3. Open questions (need the owner's decision)

1. **Wild grass training after the first encounter is resolved.** The docstring says it is legal; the code
   disables it. Recommendation: allow grass training for preparation after resolution (wild battles stay
   training only, never a new capture objective), because trainers are a limited XP source and W5 shows the
   Woods preparation has no other line. This is the likeliest single unblocker.
2. **Switch eligibility.** The capture planner only switches to a bench member who can survive two worst-case
   hits (a critical counts as one), applied before the exact calculation. In W5 a Lv8 Mudkip with 23/24 HP
   was rejected against a Taillow because 2 x 12 = 24 > 23, so a 13/13 Lv2 lead fled a capture. The exact
   faint probability for that switch is probably within the 3% bound. Options: keep the hard filter, or let
   the exact faint probability decide within the bound (the filter becomes a ranking hint).
3. **Optional trainer fights for training.** Run W4 engaged Lyle on purpose to gain XP. Confirm this is
   intended given `TrainerMode.AVOID` is a safety boundary, and under what conditions (proven-safe line only?).
4. **Does a fled first encounter use up the area?** After a flee the campaign re-selected
   `obtain_encounter:Route 102`, so the bot treats the area's encounter as still open. Confirm what your rules
   say. (The rules projection decides; the agent did not read `one_encounter_per_area`'s handling of flees.)
5. **Funding policy seeks trainers.** In W2 the bot, short of money for Poké Balls ($180 wallet), logged "restock plan
   is unaffordable; searching reachable undefeated trainers" to fund the restock. The agent did not read that code.
   Check it is consistent with the AVOID rules.
6. **Capture opportunity cost (future).** Weigh the wild Pokémon's value against the party member at risk, instead
   of fleeing whenever no plan is within the bound.
7. **`no_resets` (deferred).** Distinguish resuming from a save state from undoing progress. Sketch: start as a
   warning; on resume compare the game against the bot's own event history (a game behind the history was rewound).
8. **Defaults.** Capture faint bound 3% (`max_active_faint_probability`); readiness unresolved mass 2%
   (`_PRE_BATTLE_MAX_UNRESOLVED_PROBABILITY`). Confirm or tune.
9. **Experience spreading.** A weak preference for spreading XP so every member becomes ready sooner. Not
   implemented. `NUZLOCKE_CAMPAIGN_FIELD_LEAD` is now visible and lists the party, which is the evidence needed;
   W4 and W5 show Poochyena Lv2 staying the lead while a Lv8 to Lv9 member sits on the bench.

## 4. Decisions already made

- Every Nuzlocke rule beyond the two basic ones is optional and configurable, off by default. ROM hacks and
  randomizers are unsupported. See `nuzlocke-rules-coverage.md`.
- Species Clause duplicate cap: configurable, default unlimited.
- Preparation readiness is "a robust line exists", target level as a hint, level cap respected.
- **Route-cut rule (2026-11-05).** If a route runs any risk of meeting the trainer, prepare for that trainer; if a
  risk-free path exists, take it; a failed search is never proof. The AVOID search already uses the ROM's facing
  envelope and walk loops, so no separate "narrow" search was needed. The gap was promoting a trainer whose hazard
  area was partly a guess; now only an exact-geometry trainer with a ROM checkpoint is promoted.
- **Capture flee (2026-11-05).** Flee only if no safe switch can continue the capture and escape is legal with high
  or medium confidence; otherwise take the least risky plan, marked risk accepted.
- **Lead logging (2026-11-05).** Always on, once per distinct decision.
- **Stall dump** uses ordinary Python, not `faulthandler`, after a silent death on Windows.
- One shared battle model with a solver per battle kind; see `battle-decision-architecture.md`.

## 5. Next steps (suggested order)

1. **Resolve question 1 and, with it, unblock Woods preparation.** Then run live to the Woods and read the full
   `CAMPAIGN_PREPARATION_NO_LINE` record (needs permission).
2. **Preparation retry backoff and escalation.** A failed preparation should not retry identically forever;
   escalate (grass, a different training map, or a clear stop with the reason).
3. **Promote the Woods scene trigger to `EngageTrainer` when the party is ready**, or stop planning it under AVOID
   when it is the goal itself. Understand why route-aware readiness did not promote it.
4. **Switch eligibility (question 2).**
5. **Generic search pruning or iterative deepening.** Largest battle item. A Lv8 lead against a Lv9 trainer finishes
   in 33 ms with no bench but is incomplete after 30 s with four bench members, so nearly every campaign turn burns
   its 3 s budget and returns `BEST_AVAILABLE`.
6. Resolve the 14 order-dependent test failures (section 0).
7. Investigate the Woods recovery search that exceeded its 10 s budget (macOS runs 2, 3). Use
   `POKEBOT_DUMP_SLOW_SEARCH` if it recurs.
8. Capture v3: status moves as capture aids; move the damage model into `battle_model.py`.
9. The remaining optional-rule backlog in `nuzlocke-rules-coverage.md`.

## 6. Resuming on a new computer

The bot must run on Windows, macOS (Apple silicon), and Linux; this checkpoint was exercised on Windows (Git Bash)
after the first one on macOS.

1. **ROM.** `roms/` is gitignored. You need Pokemon Emerald (English), SHA1
   `f3ae088181bf583e55daf962a92bb46f4f1d07b7` (see `tests/README.md`).
2. **Interpreter.** Use the project's configured environment. The macOS machine used Python 3.12; the Windows
   `.venv` is Python 3.13.2 managed with `uv`. The Windows `.venv` had no pytest; the agent installed only the
   pinned one: `uv pip install --python .venv/Scripts/python.exe "pytest~=8.3.0"` (see `requirements-dev.txt`).
   Run tests with `python -m pytest -q`; emulator tests are skipped unless `--run-emulator` is given.
3. **A throwaway profile per run.** `profiles/` is gitignored. Create `profiles/<name>/` with `states/` (empty),
   `nuzlocke_events.json` containing `{"schema_version": 2}`, and `metadata.yml`:
   ```yaml
   version: 1
   rom:
     file_name: Pokemon - Emerald Version (USA, Europe).gba
     game_code: BPE
     revision: 0
     language: E
   ```
   A metadata-only profile gets an mGBA placeholder save and starts a fresh run. Use a new profile per run so the
   Nuzlocke sidecar starts clean.
4. **A diagnostic run** (do not add `--debug-trace` unless you want the very large, slow trace log):
   ```
   timeout 2700 python pokebot.py <name> -m "Campaign Progression" -hl -nv -na -s 0 \
       --no-save-state --debug-log <path-outside-the-repo>.log
   ```
   A fresh profile reaches the Woods in roughly 6 minutes on macOS and about 12 on the Windows machine.
5. **Optional stall tools** (environment variables set before the command): `POKEBOT_STALL_DUMP=<file>` with
   `POKEBOT_STALL_DUMP_SECONDS` (default 60) writes every Python thread's stack on an interval;
   `POKEBOT_DUMP_SLOW_SEARCH=<dir>` pickles a navigation search that failed after more than 3 s.
6. **Running in the background and watching it.** When launching with `nohup ... &` from a task shell, the shell's
   own "completed" notification is not the run ending. Watch the log with a monitor that reports process exit,
   log silence for 2 minutes, and the records in section 7. `PYTHONUTF8=1` is no longer needed.

Do not start live runs without the owner's permission; the owner has asked for that. The owner permitted Windows runs
W1 to W5 during this checkpoint and asked for no further run before this handoff.

## 7. Diagnostics worth knowing

- **`diagnostic_print` is silent unless `--debug` or `--debug-trace` is passed, even with `trace=False`.** Records that
  must be visible in an ordinary run use `print_always` (`modules/console.py`). Always on:
  `BATTLE_KIND`, `NAVIGATION_SEARCH_FAILURE`, `NAVIGATION_SEARCH_DUMP`, `CAMPAIGN_CONTROLLER_EXCEPTION` (rate limited),
  `CAMPAIGN_CONTROLLER_IDLE`, **`CAMPAIGN_CONTROLLER_STALLED`**, `CAMPAIGN_PREPARATION_MAP_ROUTE`,
  `CAMPAIGN_PREPARATION_READY`, **`CAMPAIGN_PREPARATION_NO_LINE`**, `CAMPAIGN_PREPARATION_TARGET_UNDESCRIBED`,
  `NUZLOCKE_CAMPAIGN_FIELD_LEAD`, `BATTLE_BIDE_OBSERVED`, `BATTLE_BIDE_DAMAGE_UNKNOWN`, and the bounded post-warp
  waits. Bold items are new in this checkpoint.
- **Capture rationale.** Every capture decision's rationale ends with `Bench:` lines: each bench member, its HP, its own
  worst-case hit, and why it was eligible or rejected. A flee says `so the encounter is abandoned`.
- **Preparation failure.** `CAMPAIGN_PREPARATION_NO_LINE` names the grass stage (disabled, no tiles, or counts and
  rejection reasons), each trainer's EXP, headroom, rejection reason, readiness status and reason and top members'
  coverage, trainers with no ROM checkpoint, and the target level and cap. It is also appended to the
  `BotModeError`. It is logged once per distinct text.
- **Silent-stall limits.** The stall dump runs in a Python thread, so it shows any Python-level stall but cannot show
  a thread stuck in native code that never releases the interpreter lock. Suspended generators (the controller's
  capabilities) do not appear in a stack dump; `CAMPAIGN_CONTROLLER_STALLED` reports the controller's view instead.
- **Reading logs.** The console wraps long records across lines and the logs can contain non-UTF-8 bytes; use
  `LC_ALL=C grep -a` (or strip non-ASCII first), and read the lines after a match.
- **Benign record:** `CAMPAIGN_CONTROLLER_EXCEPTION kind='refresh'` with `Could not find the template for local object
  #255` appears once per run at the Petalburg Wally stage (confirmed again in W5). It is probably the player's own
  map object; not confirmed.
- **macOS stack tip:** `lldb --batch -p <pid> -o 'expr (void)_Py_DumpTracebackThreads(2, (void*)0, (void*)0)' -o detach -o quit`
  writes all Python stacks to the process's stderr.
- **Live-run logs are not in the repository.** The findings that matter are in this document and in
  `campaign-progression-audit-followup.md`.

## 8. Suggested additions to `AGENTS.md` (not applied; it is the owner's file)

- `diagnostic_print` is silent without `--debug`; use `print_always` for records that must appear in an ordinary run.
- Do not start live emulator runs without permission, and use a throwaway profile with `--no-save-state`.
- Tools for silent stalls: `POKEBOT_STALL_DUMP`, `POKEBOT_DUMP_SLOW_SEARCH`, `CAMPAIGN_CONTROLLER_STALLED`.
- When a capture planner decision looks wrong, read the `Bench:` lines first; a bench member's eligibility depends on
  its own incoming damage, not the lead's.
- Never commit anything under `profiles/` or `roms/`.
