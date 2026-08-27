# Campaign Rules and Adaptive Navigation Plan

## Objective

Make Nuzlocke rules optional and let campaign strategy choose encounter and
trainer navigation policies based on the current run, party, progression
target, and risk tolerance.

## Delivery tracking

### Phase 1 — rule configuration and extension contracts

**Status: complete (2026-08-27).** The implementation introduced one immutable
`CampaignRulesConfig` for each runtime, with stable string IDs suitable for
global or profile YAML configuration. Its default preserves the historical
Nuzlocke behavior: `one_encounter_per_area` and `fainting` are enabled.
An explicit empty list creates an unrestricted campaign run.

The phase also establishes a pure `CampaignRule` extension protocol. Future
rules must evaluate already-materialized `CampaignState` values and return
constraints rather than inspect the emulator or mutate objectives. This keeps
legality, strategy, and execution separated.

Completed work:

- Added `CampaignRuleId`, `CampaignRulesConfig`, and the pure `CampaignRule`
  protocol in `modules/nuzlocke/rule_config.py`.
- Added `nuzlocke_rules.yml` configuration, with schema validation rejecting
  unknown rule IDs.
- Wired configuration through main runtime construction and reset handling.
- Made the existing rules reducer independently honor the two implemented
  baseline rules without changing the default reducer behavior.
- Added focused tests for defaults, unrestricted runs, reset preservation,
  schema validation, and the extension contract.

Acceptance criteria:

- Existing profiles retain first-encounter and fainting/whiteout enforcement.
- New Nuzlocke profiles also enable the level-cap rule by default; existing
  profiles can opt out by removing `level_cap` from their rule list.
- An empty enabled-rule list produces no encounter consumption, death records,
  or run-loss result from those events.
- The runtime exposes the immutable configuration used by its rule projection.
- Unknown configured rule IDs are rejected by configuration validation.
- Unit tests cover defaults, disabled baseline rules, reset preservation, and
  the pure rule-contract surface.

### Completed implementation roadmap

**Phase 2 — rule implementations and Emerald boss capability data.** Define
the two existing baseline rules as concrete pure assessments, add boss/cap data
behind an Emerald-only capability provider, and extend campaign facts only with
authoritative badge/leader completion signals. Acceptance: replayed events and
current fixtures yield unchanged baseline results; boss lookup is deterministic
and unavailable ROM data is represented explicitly rather than guessed.

**Status: complete (2026-08-27).** The reducer now dispatches to concrete
`OneEncounterPerAreaRule` and `FaintingRule` implementations selected by the
enabled-rule configuration. Both return pure `RuleAssessment` values, while
the projection retains ownership of ordered-event replay and the active-wild
correlation needed for capture association. Existing state fields, event
sequence behavior, and default rule set are unchanged.

`modules/nuzlocke/emerald_campaign_registry.py` now owns the implemented
Emerald boss metadata. Its first entry is Roxanne (level cap 15), and it finds
the active boss exclusively from the existing authoritative
`first_badge_obtained` fact. Unknown/unavailable completion data returns an
explicit unavailable result rather than guessing progression. This phase
exposes cap metadata; enforcement was completed in Phase 3.

Verification: the Nuzlocke reducer, runtime, campaign-state, rule-config, and
Emerald registry test suites pass (36 tests).

**Phase 3 — level-cap planning and battle-entry guard.** Implement
`LevelCapRule`, add a cap-aware party/training assessment to `CampaignState`,
and route every campaign battle entry through one rule-evaluation gateway.
Acceptance: illegal party/training choices are rejected at battle entry and a
leader victory advances the active cap only after its verified completion fact.

**Status: complete (2026-08-27).** Added the pure
`LevelCapAssessment`/`LevelCapRule` API and `evaluate_battle_entry` gateway.
The Emerald registry’s Roxanne cap (15) is now evaluated against the observed
party immediately when Campaign Progression receives a battle-start callback;
over-cap or unavailable legality is rejected conservatively with a diagnostic
and `BotModeError`. The gateway also exposes illegal party indices and a
human-readable reason for future training-objective constraints. The shared
`BattleListener → BotMode.on_battle_started` boundary is the campaign
battle-entry path, so both wild and trainer battles receive the same guard
without changing unrelated modes. After a leader victory, the registry
observes the authoritative badge fact and returns no active cap; no cap is
advanced merely because a battle was planned or entered.

The level-cap rule is enabled in the default Nuzlocke configuration and is
skipped when explicitly disabled (including unrestricted runs). Verification
currently covers over-cap, unknown, defeated-boss, and disabled-rule cases;
the focused Phase 3 suite passes 41 tests. The gateway remains deliberately
campaign-scoped; future ROMs add their own registry entries rather than adding
leader names to the generic battle strategy.

**Phase 4 — preparation decisions.** Add immutable preparation inputs and a
seeded `PreparationDecision`; initially support only wild-area preparation.
Record boss, seed, bounded target levels, policy, risk budget, and rationale.
Acceptance: equal run ID and facts produce equal decisions, while distinct
seeds can select permitted alternatives.

**Status: complete (2026-08-27).** Added the pure preparation layer in
`modules/nuzlocke/preparation.py`. `PreparationInput` captures the active boss,
party levels/health, active cap, available wild areas, risk tolerance, and
seed. `decide_preparation` produces a bounded 7–8 target before Roxanne (never
above the active cap), deterministically selects a wild area, reports missing
party data as unknown, and treats an already-ready party or absent boss as no
training requirement. The initial slice chose `WILD_FIRST`; trainer-target
selection and explicit trainer navigation policy were completed in Phase 5.
Balanced strategy remains an expansion item below.

Verification: preparation, level-cap, baseline-rule, runtime, campaign-state,
configuration, and Emerald registry tests pass (46 tests).

**Phase 5 — trainer preparation and route policy.** Discover stable trainer
affordances, execute `EngageTrainer` objectives, and normalize each
campaign-generated goal to explicit encounter/trainer modes. Acceptance:
selected trainers are intentionally engaged, avoid-mode fallback is structured,
and route ranking follows legality → trainer → encounter → movement ordering.

**Status: complete (2026-08-27).** Preparation decisions now accept stable
trainer IDs and can select a reproducible `TRAINERS_FIRST` target. Decisions
convert directly to `NavigationGoal(EngageTrainer(...))` with
`TrainerMode.ENGAGE` and `EncounterMode.IGNORE`; wild preparation emits
`SEEK` plus explicit trainer-ignore policy. Live trainer affordance discovery
now filters observed trainer hazard affordances into stable IDs, and the
execution adapter mounts any targeted `EngageTrainer` navigation objective.
Navigation diagnostics expose the explicit policy ordering
`legality > trainer > encounter > movement`; existing avoid-mode fallback
remains structured through `forced_trainer_exposure`. Verification covers
deterministic trainer selection, affordance filtering, targeted execution
adaptation, and policy mapping (9 focused tests).

**Phase 6 — progression registry and lifecycle validation.** Move Emerald
objective facts and targets behind the ROM capability registry, then add
fixture-based lifecycle tests for captures, faints, trainer exposure, boss
victory, reset/replay, and replan boundaries. Acceptance: generic campaign
code no longer interprets ROM flag names directly and all lifecycle transitions
produce one coherent decision diagnostic.

**Status: complete (2026-08-27).** Added an Emerald objective registry for the
documented introductory milestones and the Roxanne endpoint, including stable
semantic targets. Generic `campaign_fact` predicates now delegate through the
ROM-owned registry adapter, preserving existing fact values and unknown-state
semantics while keeping ROM vocabulary out of future generic extensions.
Added lifecycle-oriented registry tests covering milestone discovery and
predicate delegation. Existing event/runtime/controller suites continue to
cover capture, faint, reset/replay, and replan boundaries.

## Follow-up backlog

The numbered plan is complete for the current Emerald pilot. These are
deliberate extensions, not blocked phase work:

- Capture live save-state fixtures for trainer sight-line exposure and validate
  forced-exposure diagnostics end to end.
- Extend the ROM capability registry through later Emerald gyms and the Elite
  Four, then add separate Ruby/Sapphire and FireRed/LeafGreen interpretations.
- Implement `SpeciesClauseRule` and future shiny, healing, and item-restriction
  rules behind the enabled-rule registry.
- Mount preparation decisions into the campaign selector/controller as
  automatic preparation objectives.
- Implement the `BALANCED` strategy and use type coverage, move quality, and
  richer party-strength analysis in preparation scoring.
- Decide whether the cap gateway should be shared with non-campaign battle
  modes; it is intentionally campaign-scoped today.

## Separation of responsibilities

- **Rules** determine what is legal. Examples include one encounter per area,
  fainting, species clause, and gym-leader level caps.
- **Campaign strategy** decides what preparation or progression is desirable.
- **Navigation** executes the selected route using `EncounterMode` and
  `TrainerMode`.
- **Battle policy** enforces legal party and battle decisions at battle entry.

Rules must remain authoritative at battle entry. Planning alone is insufficient
because unexpected experience, shared EXP, or transitions can change legality.

## Modular rule architecture

Introduce an enabled rule registry in profile/run configuration. Each rule may
evaluate campaign state and modify or constrain available objectives.

Candidate rules:

- `LevelCapRule`
- `OneEncounterPerAreaRule`
- `FaintingRule`
- `SpeciesClauseRule`
- future shiny, healing, or item restrictions

`LevelCapRule` should expose the active gym-leader cap, party legality, and
whether a proposed training target is legal. Gym caps should advance after the
corresponding leader is defeated.

## Adaptive preparation

Before major battles, the campaign planner may create a preparation objective.
It should consider party size, levels, type coverage, move quality, health,
available trainers, safe wild-grinding areas, and the active level cap.

Preparation should not always train every Pokémon to the same level. Select a
bounded target and strategy, for example level 7–8 before Roxanne, using a
seeded random generator so runs vary while remaining reproducible.

Possible training preferences:

- trainers first;
- wild encounters first;
- balanced.

Trainer experience is generally preferred because it does not consume an area
encounter and provides money, but wild battles may be safer because they can be
escaped. The strategy should weigh both reward and risk.

## Encounter and trainer policy

`EncounterMode` and `TrainerMode` are campaign decisions, not permanent global
defaults.

Recommended initial semantics:

- Ordinary progression: `EncounterMode.IGNORE`, `TrainerMode.IGNORE`.
- Optional encounter acquisition: `EncounterMode.SEEK`.
- Planned wild grinding: `EncounterMode.SEEK` or `MOSTLY_AVOID`, depending on
  the desired risk level.
- Preserve experience opportunities or avoid an over-cap battle:
  `TrainerMode.AVOID`.
- Deliberate training battle: `TrainerMode.ENGAGE` with a specific trainer
  target.
- Uncommitted or strategically acceptable trainers:
  `TrainerMode.IGNORE`.

`TrainerMode.AVOID` should remain opt-in. It should first seek a route without
trainer sight-line hazards and fall back to an unavoidable exposure, recording
that fact for campaign strategy and diagnostics.

The eventual policy ordering should be explicit rather than hidden in route
costs:

1. obey hard Nuzlocke legality constraints;
2. satisfy the selected trainer policy;
3. satisfy the selected wild-encounter policy;
4. minimize movement and execution cost.

## Campaign decision cycle

1. Read campaign facts, party state, encounter ownership, and active rules.
2. Determine the next major progression cap and party legality.
3. Assess whether progression or preparation is preferable.
4. Select a preparation target and risk budget, with seeded variation.
5. Choose encounter and trainer policies for the resulting objective.
6. Generate navigation and battle objectives.
7. Re-evaluate after battles, captures, level-ups, map transitions, and rule
   state changes.

## Diagnostics

Record the selected boss, active cap, training target, encounter policy, trainer
policy, selected trainer if any, and the reason for the decision. Also record
when `TrainerMode.AVOID` falls back to forced trainer exposure.

## Implementation phases

1. Add the enabled-rule registry and rule interfaces.
2. Add gym-leader cap data and enforce `LevelCapRule` at battle entry.
3. Add adaptive, seeded preparation decisions.
4. Add explicit trainer engagement targets.
5. Propagate policy through campaign objectives and semantic navigation.
6. Validate route, battle, defeat-flag, and post-battle replan lifecycles.

## ROM-specific objective-state registry

The current Emerald pilot now uses a ROM-specific registry for its implemented
objective facts and targets. Extending that registry with additional milestones,
later gyms, and save-state fixtures is part of the follow-up backlog.

The generic campaign layer should query a ROM-specific capability registry rather
than interpreting flag names directly. Each registry can provide objective
availability and completion predicates, relevant flags and variables, semantic
targets, required live affordances, and save-state fixtures for validation.

Flag usage should distinguish completion flags, unlock/availability flags, and
visibility/state flags. `HIDE_*` flags should generally be combined with live NPC
state, script state, or another progression signal rather than assumed to be the
complete player-facing milestone.

The first proposed Emerald entries are the Petalburg/Wally tutorial,
running-shoes acquisition, Pokédex and Poké Ball acquisition, introductory rival
battle completion, and gym unlock/defeat states. Ruby/Sapphire and
FireRed/LeafGreen can later provide separate interpretations behind the same
interface.
