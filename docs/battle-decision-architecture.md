# Battle decision architecture

How the bot decides in battle, and why. This records the design agreed on
2026-11-05 and what is implemented so far. Evidence for the motivation is in
`docs/campaign-progression-audit-followup.md`.

## Principles

- **One shared battle model, several solvers.** The model (damage ranges,
  accuracy, criticals, status, speed order, special cases such as Bide) is
  common. The decision procedure is chosen by battle kind, because a trainer
  fight, a wild capture, and a forced flee are different problems.
- **Policy decides, solvers do not.** Whether a capture is legal comes from the
  Nuzlocke rules and the campaign, never from a solver.
- **Objectives are an ordering, not summed penalties.** First avoid losing a
  party Pokémon, then achieve the objective, then spend as few balls and items
  as possible, then gain EXP.
- **Bounded and explicit.** A solver finishes in bounded time, or it says it
  could not and the caller falls back. An incomplete result is never presented
  as a verified comparison.

## Layers

1. **Observation** builds a normalized `PlannerContext` (`plan_battle_state`).
2. **Classification** (`modules/battle_kind.py`) names the battle kind from
   facts the campaign already decided. It is logged once per battle as
   `BATTLE_KIND` with its reason, so a wrong classification is visible.
3. **Solver** is chosen by kind:
   - `WILD_CAPTURE`: `modules/battle_capture_planner.py` (below). Defers to the
     generic search when it cannot model the battle or find a safe plan.
   - `TRAINER`, `SCRIPTED_DEFEAT`, `WILD_DEFEAT`, `WILD_TRAINING`,
     `WILD_FLEE`: the existing expectiminimax search
     (`modules/battle_expectiminimax.py`), unchanged.
4. **Acceptance and execution** are unchanged (campaign acceptance gate, then
   the battle executor).

## The shared model

`modules/battle_model.py` holds the pure primitives a solver needs that do not
depend on a search object (currently `move_outcomes`). The larger model still
lives inside `battle_expectiminimax.py`; it can move here over time without
changing solver contracts.

## The capture planner

A wild capture is a single-agent problem with chance: the wild Pokémon is not
trying to stop the capture. Each turn the planner compares a small family of
plans exactly:

- **throw now**, or
- **use one damaging move `k` times (`k = 1..4`), then throw balls until none
  remain.**

Re-planning every turn from the live state recovers closed-loop behavior from
these open-loop plans. Each plan is scored with its probability of capture, of
losing the Pokémon in play, and of knocking out the target, plus expected balls.

Decision rule:

1. Consider only plans whose probability of losing the Pokémon in play is at
   most `max_active_faint_probability` (default 3%).
2. Among those, maximize the probability of capture. This weighs the risk of
   knocking the target out against the risk of running out of balls.
3. Among plans within 0.5% of the best capture probability, prefer lower faint
   probability, then fewer expected balls, then fewer turns.

It returns nothing, and the generic search decides, when: the battle is not a
legal capture, there is no usable ball or no catch-probability model, the
opponent's moves are unknown or unsupported, or no plan keeps the lead within
the faint bound.

### Approximations

- Damage rolls are three evenly spaced values between the planner's minimum and
  maximum.
- The opponent always uses its highest-damage known move (conservative for the
  faint bound).
- Status-inflicting and stat-changing moves are not used as capture aids.
- At most 12 throws are modeled.
- Only plain damaging moves are used to weaken; two-turn, self-destructing,
  Bide, counter, and unsupported moves are excluded. False Swipe is modeled as
  never knocking the target out.
- Switching to a sturdier Pokémon before capturing is not modeled; a fragile
  lead is deferred to the generic search.

### Worked example

A 13 HP lead against a Wurmple whose Tackle does 5 to 7 and can crit for 14:
even throwing immediately carries about a 36% chance of losing the lead over a
full ball sequence, so the planner defers. A 40 HP lead against the same target
weakens twice and captures at about 99% with no knockout risk.

## Not yet done

- A live capture run exercising the planner (it was validated by unit tests and
  offline scenarios; live runs so far predate it).
- Status moves and switching as capture aids.
- Moving the damage and outcome model out of the expectiminimax module.
- Bounding the capture rollout inside the generic search so it respects its
  time budget.
