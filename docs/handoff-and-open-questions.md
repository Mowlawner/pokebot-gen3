# Handoff and open questions

Checkpoint written 2026-11-05 so work can resume on another computer. Read this
first, then `AGENTS.md`, then the documents it links. Everything here is on branch
`improved-leveling` (pushed to `origin`); the commit that adds this file is the
checkpoint. Full default test suite at the checkpoint: **2258 passed, 34 skipped**
(the skipped tests are the optional ROM/native-emulator tier).

## Where things stand

The bot starts a fresh Emerald run and, in live runs on this branch, reliably gets
through: the opening, the starter, the intro rival on Route 103, the first captures
on Routes 101 to 104, Mart and Center trips, Wally's tutorial in Petalburg, Route 104
past Cindy and Billy, the Petalburg Woods warp, and a Woods capture. It then enters
trainer preparation for the Aqua Grunt (a Lv9 Poochyena). **No live run has reached
the grunt fight or the first badge.**

### Validated in live runs

| Area | Evidence |
| --- | --- |
| Route 103 training and intro rival (preparation target is published) | runs 2 to 9 |
| Route 104 corridor past Cindy and Billy, no trainer exposure | runs 3 to 9 |
| Post-warp resumption into the Woods and a Woods capture | runs 2, 3, 5 to 9 |
| Staged trainer-safe travel fallback for preparation | runs 8, 9 (used 2 to 3 times each, no failures) |
| Capture planner (bounded plan comparison) | run 9: 5 capture battles, 8 planner decisions, first capture took 1 ball |
| Battle-kind classification logged once per battle | run 9 |
| Party panel shows where each member was met | runs 7 to 9 |

### Implemented and unit-tested, **not yet validated live**

These were written after the last live run, or exercise paths no run reached:

- Readiness accepts a horizon projection with small unresolved mass, and
  preparation ends as soon as the party has a ready line (`CAMPAIGN_PREPARATION_READY`).
- Capture planner v2: switching to a member who survives two worst-case hits, and
  the least-risky "risk accepted" fallback. A live deferral (to the generic search)
  was seen once; the new switch and risk-accepted paths were not.
- Species Clause duplicate cap (`species_clause_max_duplicates_per_area`).
- Recovery from the Woods: it timed out on a 10 s search in runs 2 and 3 and did not
  recur in runs 4 to 9; the cause is unexplained (see Open items).

## Live runs 1 to 4 on Windows (2026-11-05, after the decisions above)

Four fresh-profile runs on a Windows machine, all `--no-save-state`, logs outside the repo.

- **Run 1** crashed on Route 101: redirected stdout is cp1252 and the encounter panel
  printed a gender symbol. Fixed (`modules/console.py` writes redirected output as UTF-8).
- **Run 2** stalled silently at the Petalburg Mart; **run 3** died within a second of its
  first `faulthandler` stack dump. The stall dump is now a Python watchdog thread
  (`modules/stall_dump.py`); run 4 survived four dumps.
- **Run 4** reached the Woods in about 12 minutes: intro rival, Calvin, and Lyle (an optional
  Woods trainer, fought through an `EngageTrainer` goal) were won. It never reached the grunt.
- **Capture-flee bug found and fixed.** The new flee rule fired three times. Two were wrong:
  a Lv2 lead at 13/13 HP fled a Marill and a Wurmple while a Lv9 Treecko at 28/28 sat on the
  bench. The capture planner used the lead's incoming damage for every bench member, ignoring
  `PlannerSwitch.incoming_damage_max`. Each member now uses its own figures. Not yet re-run live.
- **Silent permanent wait (open).** After preparation raised `BotModeError: no available trainer
  has a proven safe, cap-compatible preparation line`, recovery healed the party and the bot
  returned to the Woods with goal `ActivateTrigger devon_goods_researcher` under AVOID. That
  search always returns not found (the scene trigger is itself a trainer hazard), nothing is
  re-issued, and the log stops. `CAMPAIGN_CONTROLLER_IDLE` never fires because its watchdog only
  counts steps when no capability is mounted (`idle=self._tactical_loop is None` in
  `campaign_controller.step`). Two things to resolve: the idle watchdog must cover a mounted loop
  whose goal has a cached search failure, and the goal should be promoted to `EngageTrainer`
  once the party is ready (or preparation should find a line).
- **Unconfirmed:** `_preparation_navigation_goal` skips every grass candidate when
  `allow_wild=False` (area encounter resolved), which contradicts the docstring of
  `_preparation_encounter_resolved`. Whether that applied in run 4 is not established; its
  diagnostic (`CAMPAIGN_PREPARATION_WILD_DISABLED`) only prints under `--debug`.

## Resuming on a new computer

1. **ROM.** `roms/` is gitignored. You need Pokemon Emerald (English), SHA1
   `f3ae088181bf583e55daf962a92bb46f4f1d07b7` (see `tests/README.md`).
2. **Interpreter.** Use the project's configured Python 3.12 environment. Run tests
   with `pytest -q`; emulator tests are skipped unless `--run-emulator` is passed.
3. **A throwaway profile.** `profiles/` is gitignored. Create `profiles/<name>/`
   with:
   - `metadata.yml`:
     ```yaml
     version: 1
     rom:
       file_name: Pokemon - Emerald Version (USA, Europe).gba
       game_code: BPE
       revision: 0
       language: E
     ```
   - `nuzlocke_events.json` containing `{"schema_version": 2}`, and an empty `states/`.
   A metadata-only profile gets an mGBA placeholder save and starts a fresh run.
4. **A diagnostic run** (no debug flags needed; do not add `--debug-trace` unless you
   want the slow, very large trace log):
   ```
   timeout 2700 python pokebot.py <name> -m "Campaign Progression" -hl -nv -na -s 0 \
       --no-save-state --debug-log /tmp/<name>.log
   ```
   On the development machine a fresh profile reached the Woods in about 6 minutes.
   Use a new profile per run so the Nuzlocke sidecar starts clean. Back up an
   existing profile's `run.log` first if you want to keep it; a run writes to the
   profile (this was not checked).

Do not start live runs without the owner's permission; the owner has asked for that.

## Diagnostics worth knowing

- **`diagnostic_print` is silent unless `--debug` or `--debug-trace` is passed, even
  with `trace=False`.** Records that must be visible in an ordinary run use
  `print_always` (`modules/console.py`). These are always on: `BATTLE_KIND`,
  `NAVIGATION_SEARCH_FAILURE`, `NAVIGATION_SEARCH_DUMP`,
  `CAMPAIGN_CONTROLLER_EXCEPTION` (rate limited), `CAMPAIGN_CONTROLLER_IDLE`,
  `CAMPAIGN_PREPARATION_MAP_ROUTE`, `CAMPAIGN_PREPARATION_READY`,
  `CAMPAIGN_PREPARATION_TARGET_UNDESCRIBED`, `BATTLE_BIDE_OBSERVED`,
  `BATTLE_BIDE_DAMAGE_UNKNOWN`, and the bounded post-warp waits.
- **Opt-in tools** (environment variables):
  - `POKEBOT_DUMP_SLOW_SEARCH=<dir>` pickles the exact world, start, and goal of a
    navigation search that fails after more than 3 s, for offline profiling.
  - `POKEBOT_STALL_DUMP=<file>` (with `POKEBOT_STALL_DUMP_SECONDS`, default 60)
    writes every thread's Python stack on an interval, independent of the log.
- **Reading logs.** The console wraps long records across lines and the logs contain
  non-UTF-8 bytes; use `LC_ALL=C grep -a`, and read the lines after a match.
- **A stack from a process with no tooling (macOS).** This worked:
  `lldb --batch -p <pid> -o 'expr (void)_Py_DumpTracebackThreads(2, (void*)0, (void*)0)' -o detach -o quit`
  writes all Python stacks to the process's stderr. `sample <pid> 3` shows native
  stacks only.
- **Benign record:** `CAMPAIGN_CONTROLLER_EXCEPTION kind='refresh'` with
  `Could not find the template for local object #255` appears once per run at the
  Petalburg Wally stage. It is probably the player's own map object; not confirmed.
- **Live-run logs are not in the repository.** The findings that matter are written
  in `campaign-progression-audit-followup.md`.

## Decisions already made

- Every Nuzlocke rule beyond the two basic ones is optional and configurable, off by
  default. ROM hacks and randomizers are unsupported. See `nuzlocke-rules-coverage.md`.
- Species Clause duplicate cap: configurable, default unlimited, user-set hard limit.
- Preparation readiness is "a robust line exists", with the target level as a hint
  and the level cap respected.
- Capture: choose the least risky plan; switch to another member only if it can
  survive two worst-case attacks (a critical counts as one).
- One shared battle model with a solver per battle kind; see
  `battle-decision-architecture.md`.

## Open questions (need the owner's decision)

1. **Route-cut rule. Decided and implemented 2026-11-05, not yet validated live.** The
   AVOID search already uses the ROM's facing envelope and walk loops, so a separate
   "narrow search" would add nothing. The gap was that a failed search promoted a trainer
   even when its hazard area was partly a guess. `_encounter_stage_route_cut_hazards` now
   promotes a trainer only if it resolves to a ROM checkpoint **and** its geometry is exact
   (`trainer_hazard_geometry_is_exact` in `modules/overworld.py`); an unknown facing, an
   over-covered fixed-axis walker, or unobserved geometry stays unresolved. Exact and
   unavoidable trainers still lead to preparation. Original question: when the safe search finds no route, a trainer common to every
   permissive route can be promoted to a mandatory hazard, which allows exposure.
   That is how Billy was exposed originally. Recommendation: promote a trainer to
   must-fight only if no path exists even under the *narrow* model (the ROM's real
   facing and sight lines); if a path exists there, the cautious map was too
   pessimistic, so take that path (or, for a walking trainer, wait until the loop
   moves away). A failed cautious search alone is never proof. Needs approval, then
   implementation in `_encounter_stage_route_cut_hazards`.
2. **Hopeless captures. Decided 2026-11-05 and implemented, not yet live-validated.**
   Flee only if no safe switch can continue the capture (and escape is legal); otherwise
   keep the least risky "risk accepted" plan. A future state may weigh the target's value
   against the member at risk.
3. **Cleanup. Closed.** The throwaway profiles and `/tmp/nuzbot-diag` were on the other
   machine; nothing exists here. Profiles and save states are never committed.
4. **Lead-selection logging. Done 2026-11-05.** `NUZLOCKE_CAMPAIGN_FIELD_LEAD` is now
   always on, logged once per distinct decision, with the whole party (slot, species,
   level, HP).
5. **`no_resets` (deferred).** The bot resumes from save states, which must be
   distinguished from undoing progress. Sketch: the bot never rewinds itself in
   campaign mode, and on resume compares the game against its own event history (a
   game *behind* the history was rewound). Start as a warning, not a block.
6. **Capture and readiness defaults.** The capture planner's faint bound is 3%
   (`max_active_faint_probability`) and readiness accepts a 2% unresolved mass
   (`_PRE_BATTLE_MAX_UNRESOLVED_PROBABILITY`). Confirm or tune these.
7. **Experience spreading.** The owner wants a weak preference for spreading
   experience so every member becomes ready sooner. Not implemented: the training-lead
   function already prefers the largest level deficit, and the lead-selection comments
   disagree about experience tagging ("a weak lead can safely tag" versus "without
   routine EXP tagging"). In live runs weaker members led many turns but barely
   leveled; the cause is not established. Needs the evidence from question 4.

## Next steps (suggested order)

1. **Validate live** what has not been run since the readiness fix: that the Woods
   preparation now ends through `CAMPAIGN_PREPARATION_READY`, that the grunt fight
   happens, and that the capture planner's new paths behave. Requires permission.
2. **Generic search pruning or iterative deepening.** Largest item. A Lv8 lead against a
   Lv9 trainer finishes in 33 ms with no bench but is incomplete after 30 s with four
   bench members, so nearly every campaign turn burns its 3 s budget and returns
   `BEST_AVAILABLE`. The capture rollout inside it also ignores the time budget.
3. Route-cut narrow-model check (question 1).
4. **Controller retry backoff and escalation.** A swallowed failure retries every
   frame with no backoff (a run logged 100+ identical failures in minutes).
5. Investigate the recovery search that exceeded its 10 s budget in runs 2 and 3.
   Live it ran at about 2.9 ms per node; the same-sized search on a simplified
   Route-104-sized world ran at about 0.02 ms per node offline, and a competing
   CPU-bound thread slowed it only 1.5 to 3 times, so thread contention is not the whole
   explanation. The real failing world was never captured; use the slow-search dump if
   it recurs.
6. Look into the `select No` intent seen in run 9 (unchecked).
7. Capture v3: status moves as capture aids; move the damage model out of the
   expectiminimax module into `battle_model.py`.
8. The remaining optional-rule backlog in `nuzlocke-rules-coverage.md`.

## Suggested additions to `AGENTS.md` (not applied; it is the owner's file)

- `diagnostic_print` is silent without `--debug`; use `print_always` for records that
  must appear in an ordinary run.
- Do not start live emulator runs without permission, and use a throwaway profile.
- Tools for silent stalls: `POKEBOT_STALL_DUMP`, `POKEBOT_DUMP_SLOW_SEARCH`.
