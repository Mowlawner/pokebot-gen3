# Campaign Rules and Adaptive Navigation Plan

## Objective

Make Nuzlocke rules optional and let campaign strategy choose encounter and
trainer navigation policies based on the current run, party, progression
target, and risk tolerance.

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

## Deferred: ROM-specific objective-state registry

The campaign should eventually use ROM-specific event flags and variables to
determine which objectives are available and complete. This work is intentionally
deferred until trainer and navigation policy work is stable.

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
