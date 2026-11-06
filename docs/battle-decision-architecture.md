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

If the lead has no plan within the faint bound, the planner considers switching
to another party member, but only one who can survive two worst-case opponent
attacks (a critical hit counts as an attack), only when switching is legal, and
only once per stretch of battle so the lead cannot ping-pong. If still no plan is
within the bound, the planner flees when running is legal and its confidence is
high or medium (owner decision 2026-11-05: flee only when no safe switch can continue
the capture). A future refinement may weigh the wild Pokémon's value against the party
member at risk. If fleeing is not legal, it chooses the least risky plan overall
(lowest faint probability, then highest capture probability) and marks the decision as
risk accepted: it is classified best-available, never safe, and carries a result that
lets the campaign proceed for one turn without claiming safety.

A bench member is judged against the damage the opponent does to **it**, not to the
Pokémon in play. Each `PlannerSwitch` carries its own `incoming_damage_max` and
critical ceiling; the switch check and the faint table use them, scaling the lead's
damage spread to match, and fall back to the lead's numbers only when a member has no
figure. (Before 2026-11-05 every member used the lead's numbers, which rejected a
Lv9 Treecko at 28/28 HP against a Wurmple because a critical on a Lv2 Poochyena did 14
and 2 x 14 = 28, and the planner fled three live captures.)

Every decision's rationale ends with `Bench:` lines (`describe_switch_candidates`): each
bench member, its HP, its own worst-case hit, and whether it was eligible or why it was
rejected (needs HP above twice the worst hit, just switched out, switching illegal, a
switch already happened, or no bench). Read these first when a capture decision looks
wrong.

It returns nothing, and the generic search decides, only when the battle cannot
be modeled: it is not a legal capture, there is no usable ball or no
catch-probability model, or the opponent's moves are unknown or unsupported.

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
- A switch costs one opponent action (the incoming Pokémon takes a hit), and the
  incoming Pokémon's own moves are used for weakening.

### Worked example

A 13 HP lead against a Wurmple whose Tackle does 5 to 7 and can crit for 14:
even throwing immediately carries about a 36% chance of losing the lead over a
full ball sequence, so the planner defers. A 40 HP lead against the same target
weakens twice and captures at about 99% with no knockout risk.

## Trainer readiness (pre-battle)

Before a mandatory trainer, preparation trains until the live party has a ready
line against the trainer's known roster (`modules/trainer_readiness.py`). Each
party member is assessed against each roster Pokémon with a bounded, switch-free
search (5 turns, 16,000 nodes).

A line is sufficient when either:

- it is `terminal_complete` and safe, or
- it is a `horizon_projection` that was not cut off by a budget, whose unresolved
  probability (`1 - win - loss`) is at most 2%, and whose finished branches meet the
  win (90%), loss (5%), catastrophic (5%), and faint (5%) thresholds.

The second clause exists because a move that can miss lets a battle run past any
finite horizon, so the engine can never label such a line `terminal_complete`, even
for a lead that wins every time. The category is not relabeled; the decision rests
on explicit numbers, and a projection with no probability evidence never counts.
Measured against a 28 HP Poochyena: a Lv10 lead wins about 98.8% (ready), a Lv9 lead
about 96.7% with 1.6% faint (ready), a Lv8 lead about 70% with 30% faint (not ready).

The target level is a hint. Preparation (`execute_campaign_preparation`) also ends as
soon as the live party has a ready line (`CAMPAIGN_PREPARATION_READY`), re-evaluated
only when a member's species, level, max HP, or half-HP eligibility changes. The level
cap is unchanged.

## Known performance problem: the generic search and bench members

The expectiminimax search cannot finish when the party has bench members. A Lv8 lead
against a Lv9 trainer completes in 33 ms (118 nodes) with no switch options, but stays
incomplete after 3 s and after 30 s (159,000 nodes) with four bench members. As a
result nearly every campaign battle turn spends its full 3 s budget and returns
`BEST_AVAILABLE`. Candidate fixes: prune dominated switches and cap the number of
switch candidates, iterative deepening with an early exit when one root clearly
dominates, and reuse between turns. The capture rollout inside the generic search also
ignores the time budget; the dedicated capture planner avoids that path for wild
captures.

## Not yet done

- Live validation of the risk-accepted path (escape not legal), the readiness change,
  and the early stop. Not run live. The switch path and the flee path **were** seen
  live on 2026-11-05 (W4, W5 in `handoff-and-open-questions.md`).
- **Open question: the switch eligibility filter.** Requiring a bench member to survive
  two worst-case hits is stricter than the 3% faint bound it protects. In W5 a Lv8 Mudkip
  at 23/24 HP was rejected against a Taillow (2 x 12 = 24 > 23) and a 13/13 Lv2 lead fled.
  Options: keep it as a hard filter, or let the exact faint probability decide within the
  bound. Owner decision needed.
- Pruning or iterative deepening for the generic search (above).
- Status moves as capture aids.
- Moving the damage and outcome model out of the expectiminimax module.
- Bounding the capture rollout inside the generic search so it respects its
  time budget.
- Opportunity-cost weighing: whether the target is worth risking the Pokémon in play
  (today the bot flees whenever no plan is within the faint bound and escape is legal).
- Live validation of the flee path.
