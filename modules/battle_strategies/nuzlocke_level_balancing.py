"""Level-balancing battle strategies that respect Nuzlocke deaths."""

from __future__ import annotations

from modules.battle_state import BattleState
from modules.battle_observation import default_battle_recorder
from modules.battle_strategies import DefaultBattleStrategy, TurnAction
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_strategies.catch import CatchStrategy
from modules.battle_strategies.level_balancing import LevelBalancingBattleStrategy
from modules.context import context
from modules.console import diagnostic_print
from modules.nuzlocke.identity import PokemonIdentity
from modules.pokemon import get_move_by_name
from modules.pokemon_party import get_party


def _nuzlocke_dead_identities() -> frozenset[PokemonIdentity]:
    runtime = getattr(context, "nuzlocke_runtime", None)
    projection = getattr(runtime, "rules_projection", None)
    state = getattr(projection, "state", None)
    return frozenset(getattr(state, "dead_pokemon", ()) or ())


def _is_eligible_pokemon(pokemon) -> bool:
    if getattr(pokemon, "is_egg", False):
        return False
    # Ordinary Level Grind has no Nuzlocke runtime and retains its historical
    # behavior. During campaign preparation, an unreadable identity is still
    # a valid live party observation; only an identity explicitly present in
    # the death projection should exclude a Pokémon from the policy. The
    # party reader already filters malformed/empty records before they reach
    # this strategy.
    if getattr(context, "nuzlocke_runtime", None) is None:
        return True
    identity = PokemonIdentity.from_pokemon(pokemon)
    return identity is None or identity not in _nuzlocke_dead_identities()


def _eligible_indices(*, only_non_fainted: bool = False) -> tuple[int, ...]:
    return tuple(
        index
        for index, pokemon in enumerate(get_party())
        if _is_eligible_pokemon(pokemon) and (not only_non_fainted or pokemon.current_hp > 0)
    )


def _lowest_level_index(*, only_non_fainted: bool = False) -> int | None:
    indices = _eligible_indices(only_non_fainted=only_non_fainted)
    return min(indices, key=lambda index: (get_party()[index].level, index)) if indices else None


def _lowest_battle_capable_index(strategy: "NuzlockeLevelBalancingBattleStrategy") -> int | None:
    """Return the lowest-level living member that can actually take a turn."""

    candidates = tuple(
        index for index in _eligible_indices(only_non_fainted=True) if strategy.pokemon_can_battle(get_party()[index])
    )
    return min(candidates, key=lambda index: (get_party()[index].level, index)) if candidates else None


class NuzlockeLevelBalancingBattleStrategy(LevelBalancingBattleStrategy):
    """Balance living campaign Pokémon without ever sending out a dead one."""

    def __init__(self):
        super().__init__()
        # If the weak lead cannot safely continue, the inherited strategy may
        # switch to a stronger party member.  Keep that support Pokémon in
        # until the battle ends; otherwise the next turn's lowest-level
        # policy immediately switches back and the battle can ping-pong
        # between the two members without resolving.
        self._battle_support_index: int | None = None

    def pokemon_can_battle(self, pokemon) -> bool:
        return _is_eligible_pokemon(pokemon) and super().pokemon_can_battle(pokemon)

    def party_can_battle(self) -> bool:
        # A fainted/dead low-level member must not make a higher-level living
        # member appear unusable.  The lowest living eligible member is the
        # preferred lead, but any legal battle-capable member keeps the party
        # in the battle loop.
        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        index = _lowest_level_index(only_non_fainted=True)
        if index is None:
            raise RuntimeError("no living Pokémon is available for campaign preparation")
        return index

    def has_replacement_after_faint(self, battle_state: BattleState) -> bool:
        active_indices = {
            battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
        }
        return any(
            index not in active_indices and self.pokemon_can_battle(get_party()[index])
            for index in _eligible_indices(only_non_fainted=True)
        )

    def choose_new_lead_after_battle(self) -> int | None:
        index = _lowest_level_index(only_non_fainted=True)
        try:
            current_lead = get_party().first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_lead = 0
        selected = index if index is not None and index != current_lead else None
        diagnostic_print(
            lambda: (
                "NUZLOCKE_LEVEL_BALANCE_AFTER_BATTLE: "
                f"candidate={index!r} current_lead={current_lead!r} selected={selected!r} "
                f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
            ),
            trace=True,
        )
        return selected

    def decide_turn(self, battle_state: BattleState):
        # The battle listener normally hands a legal encounter directly to
        # NuzlockeCaptureStrategy.  There is a narrow ready-turn race where
        # the listener has already selected this ordinary campaign strategy,
        # but the battle-state projection now says that the encounter is
        # capture-owned.  Keep that fallback on the same capture policy so a
        # weak lead is not asked to fight or throw a ball without the capture
        # strategy's lead switch and no-ball behavior.
        if getattr(battle_state, "nuzlocke_capture_target", False):
            capture_strategy = getattr(self, "_nuzlocke_capture_strategy", None)
            if capture_strategy is None:
                capture_strategy = NuzlockeCaptureStrategy()
                self._nuzlocke_capture_strategy = capture_strategy
            capture_action = capture_strategy.decide_turn(battle_state)
            if capture_action[0] is not TurnAction.SwitchToManual:
                return capture_action

        # Lead rotation after the previous battle is a useful fallback, but
        # it cannot cover a newly captured member or a battle that begins
        # before the post-battle menu handoff is available.  Make the policy
        # explicit at every single-battle turn boundary: use the lowest-level
        # living, battle-capable member whenever switching is legal.
        if not getattr(battle_state, "is_double_battle", False):
            active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
            candidate = _lowest_battle_capable_index(self)
            active_index = getattr(active, "party_index", None)
            if getattr(self, "_battle_support_index", None) == active_index:
                action = super().decide_turn(battle_state)
                if action[0] is TurnAction.RotateLead:
                    self._battle_support_index = action[1]
                return action
            try:
                can_switch = BattleStrategyUtil(battle_state).can_switch()
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                can_switch = False
            if candidate is not None and candidate != active_index and can_switch:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_LEVEL_BALANCE_SWITCH: "
                        f"active={active_index!r} candidate={candidate!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                return TurnAction.rotate_lead(candidate)
        action = super().decide_turn(battle_state)
        if action[0] is TurnAction.RotateLead:
            self._battle_support_index = action[1]

        # The generic strategy accepts a ``BEST_AVAILABLE`` planner action
        # when it cannot prove a safe line.  That is useful as a last resort,
        # but it defeats the Nuzlocke training policy when the active Pokémon
        # is the weakest member: an uncertain attack spends the weak member's
        # turn without giving it a safe path to useful experience.  Give a
        # stronger, healthy member the battle instead when one can actually
        # damage the opponent and survive the visible response.
        if (
            not getattr(battle_state, "is_double_battle", False)
            and candidate is not None
            and candidate == active_index
            and action[0] not in (TurnAction.RotateLead, TurnAction.SwitchToManual)
            and getattr(context, "battle_decision_source", None) != "PLANNER SAFE"
        ):
            support_index = self._stronger_battle_support_target(battle_state, active_index)
            if support_index is not None:
                self._battle_support_index = support_index
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_LEVEL_BALANCE_SUPPORT_SWITCH: "
                        f"active={active_index!r} candidate={support_index!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                return TurnAction.rotate_lead(support_index)

        # The generic battle planner does not know the Nuzlocke death
        # projection. Never honor a planner switch to a healed dead Pokémon.
        if action[0] is TurnAction.RotateLead:
            replacement = get_party()[action[1]]
            active_indices = {
                battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
            }
            if not _is_eligible_pokemon(replacement) or replacement.current_hp <= 0 or action[1] in active_indices:
                util = BattleStrategyUtil(battle_state)
                move = util.get_strongest_move_against(
                    battle_state.own_side.active_battler,
                    battle_state.opponent.active_battler,
                )
                if move is not None:
                    return TurnAction.use_move(move)
        return action

    def _stronger_battle_support_target(self, battle_state: BattleState, active_index: int | None) -> int | None:
        """Choose a stronger healthy member for an unsafe weak-lead battle.

        This is deliberately a battle-support policy, not a replacement for
        the planner.  The planner has already failed to establish a safe line
        for the weak lead.  A support candidate therefore needs a usable
        damaging move, and—when an opponent response has been observed—must
        survive that response (including its critical-hit ceiling).
        """

        if active_index is None or active_index < 0 or active_index >= len(get_party()):
            return None
        try:
            opponent = battle_state.opponent.active_battler
            util = BattleStrategyUtil(battle_state)
            if not util.can_switch():
                return None
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        active_level = getattr(get_party()[active_index], "level", 0)
        known_response_moves = _known_opponent_move_names()
        candidates = []
        for index in _eligible_indices(only_non_fainted=True):
            if index == active_index:
                continue
            pokemon = get_party()[index]
            if getattr(pokemon, "level", 0) <= active_level or not self.pokemon_can_battle(pokemon):
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, opponent)
                if move_index is None:
                    continue
                damage = util.calculate_move_damage_range(pokemon.moves[move_index].move, pokemon, opponent)
                if damage.max <= 0:
                    continue
                incoming = _capture_response_damage_ceiling(
                    util,
                    opponent,
                    pokemon,
                    known_response_moves,
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                continue

            response_is_known = bool(known_response_moves) and incoming is not None
            response_is_survivable = response_is_known and incoming < pokemon.current_hp
            if known_response_moves and not response_is_survivable:
                continue
            candidates.append(
                (
                    damage.min >= opponent.current_hp,
                    response_is_survivable,
                    damage.min,
                    damage.max,
                    getattr(pokemon, "level", 0),
                    pokemon.current_hp,
                    -index,
                    index,
                )
            )
        return max(candidates)[-1] if candidates else None


class NuzlockeCaptureStrategy(CatchStrategy):
    """Capture wild encounters while keeping the weakest member in front.

    Campaign capture battles historically used :class:`CatchStrategy`
    directly, which meant the party's first member stayed active for every
    encounter.  The campaign lead policy leaves the weakest living member at
    the front between battles; capture turns then switch to a stronger member
    with a usable move, preferring one that can weaken without a KO. Keep the
    capture policy (including its no-ball fallback) otherwise unchanged.
    """

    def pokemon_can_battle(self, pokemon) -> bool:
        """Return whether ``pokemon`` is a legal living capture battler."""

        return _is_eligible_pokemon(pokemon) and not pokemon.is_egg and pokemon.current_hp > 0

    def party_can_battle(self) -> bool:
        """Return whether any living eligible party member remains usable."""

        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        """Choose the lowest-level eligible replacement, excluding actives."""

        active_indices = {
            battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
        }
        candidates = tuple(index for index in _eligible_indices(only_non_fainted=True) if index not in active_indices)
        if not candidates:
            raise RuntimeError("no living Pokémon is available for capture battle replacement")
        return min(candidates, key=lambda index: (get_party()[index].level, index))

    def has_replacement_after_faint(self, battle_state: BattleState) -> bool:
        """Report whether a legal non-active replacement exists."""

        active_indices = {
            battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
        }
        return any(
            index not in active_indices and self.pokemon_can_battle(get_party()[index])
            for index in _eligible_indices(only_non_fainted=True)
        )

    def choose_new_lead_after_battle(self) -> int | None:
        """Choose the weakest living member for the next encounter."""

        index = _lowest_level_index(only_non_fainted=True)
        try:
            current_lead = get_party().first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_lead = 0
        return index if index is not None and index != current_lead else None

    def decide_turn(self, battle_state: BattleState):
        """Switch from the weak lead to a stronger capture-capable battler."""

        if not getattr(battle_state, "is_double_battle", False):
            active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
            active_index = getattr(active, "party_index", None)
            candidate = self._stronger_capture_switch_target(battle_state, active_index)
            if candidate is not None:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_CAPTURE_LEVEL_BALANCE_SWITCH: "
                        f"active={active_index!r} candidate={candidate!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                return TurnAction.rotate_lead(candidate)

        # CatchStrategy delegates to the ordinary battle policy when the bag
        # has no usable ball, so an empty reserve still resolves as a normal
        # defeat/escape outcome instead of freezing the controller.
        return super().decide_turn(battle_state)

    def _stronger_capture_switch_target(self, battle_state: BattleState, active_index: int | None) -> int | None:
        """Find a stronger living member that can damage the capture target.

        A capture lead is intentionally the weakest living member so the
        party order remains useful for ordinary training.  Once a battle is
        active, prefer a stronger member with a nonlethal move; if none can
        weaken safely, accept a stronger member that can at least damage the
        target and therefore resolve the encounter. Members without a usable
        move are not selected because they cannot satisfy that battle role.
        """

        if active_index is None or active_index < 0 or active_index >= len(get_party()):
            return None
        active_level = getattr(get_party()[active_index], "level", 0)
        try:
            opponent = battle_state.opponent.active_battler
            util = BattleStrategyUtil(battle_state)
            can_switch = util.can_switch()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        if not can_switch:
            return None

        known_response_moves = _known_opponent_move_names()
        candidates = []
        for index in _eligible_indices(only_non_fainted=True):
            if index == active_index:
                continue
            pokemon = get_party()[index]
            if getattr(pokemon, "level", 0) <= active_level or not self.pokemon_can_battle(pokemon):
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, opponent)
                if move_index is None:
                    continue
                move = pokemon.moves[move_index].move
                damage = util.calculate_move_damage_range(move, pokemon, opponent)
                if damage.max <= 0:
                    continue
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                continue
            safe_to_weaken = damage.max < opponent.current_hp
            incoming_damage = _capture_response_damage_ceiling(
                util,
                opponent,
                pokemon,
                known_response_moves,
            )
            response_is_known = bool(known_response_moves) and incoming_damage is not None
            response_is_survivable = response_is_known and incoming_damage < pokemon.current_hp
            candidates.append(
                (
                    safe_to_weaken,
                    response_is_survivable,
                    response_is_known,
                    pokemon.level,
                    pokemon.current_hp,
                    -damage.max,
                    -index,
                    index,
                )
            )
        # If an opponent move has already been observed, do not knowingly
        # switch into a replacement that the observed response can KO.  When
        # no response is visible yet, retain the conservative fallback of
        # choosing the strongest usable member with the best weakening line;
        # the battle planner cannot manufacture move knowledge either.
        if known_response_moves:
            candidates = [candidate for candidate in candidates if candidate[1]]
        return max(candidates)[-1] if candidates else None


def _known_opponent_move_names() -> tuple[str, ...]:
    """Return only moves already visible to the human-facing battle model."""

    knowledge = getattr(default_battle_recorder, "knowledge", None)
    if knowledge is None:
        return ()
    names = {
        name
        for _, name in (
            tuple(getattr(knowledge, "observed_opponent_moves", ()))
            + tuple(getattr(knowledge, "known_opponent_moves", ()))
            + tuple(getattr(knowledge, "revealed_opponent_moves", ()))
        )
        if name
    }
    return tuple(sorted(names))


def _capture_response_damage_ceiling(
    util: BattleStrategyUtil,
    opponent,
    candidate,
    move_names: tuple[str, ...],
) -> int | None:
    """Calculate the worst visible response, including a critical hit.

    ``None`` means the response could not be established from visible facts;
    it is intentionally different from zero damage for a known status move.
    """

    if not move_names:
        return None
    damage_values = []
    for move_name in move_names:
        try:
            move = get_move_by_name(move_name)
            regular = util.calculate_move_damage_range(move, opponent, candidate).max
            critical = util.calculate_move_damage_range(
                move,
                opponent,
                candidate,
                is_critical_hit=True,
            ).max
            damage_values.append(max(regular, critical))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            return None
    return max(damage_values, default=0)


class EmeraldIntroRivalBattleStrategy(DefaultBattleStrategy):
    """Use Emerald's safe opening line for the scripted Route 103 battle.

    The introductory rival has a single low-level Pokémon and the player's
    starter has Growl. The generic planner cannot prove a multi-turn attack
    line when the rival's move is not human-visible, so its best-available
    fallback attacks immediately. Two opening Growls are a ROM-backed,
    trainer-battle-local safety policy: they reduce the only damaging move
    available in this fixed opening without affecting optional wild encounters.
    """

    _OPENING_GROWLS = 2

    def __init__(self):
        super().__init__()
        self._opening_growls_used = 0

    def decide_turn(self, battle_state: BattleState):
        if battle_state.is_trainer_battle and self._opening_growls_used < self._OPENING_GROWLS:
            active = battle_state.own_side.active_battler
            for index, learned_move in enumerate(active.moves):
                if (
                    learned_move is not None
                    and learned_move.move.name == "Growl"
                    and learned_move.pp > 0
                    and active.disabled_move is not learned_move.move
                ):
                    self._opening_growls_used += 1
                    context.battle_decision_source = "EMERALD INTRO RIVAL SETUP"
                    context.battle_decision_detail = f"opening Growl {self._opening_growls_used}/{self._OPENING_GROWLS}"
                    context.message = (
                        f"EMERALD INTRO RIVAL SETUP: Growl " f"{self._opening_growls_used}/{self._OPENING_GROWLS}"
                    )
                    return TurnAction.use_move(index)
        return super().decide_turn(battle_state)


class RoxanneBattleStrategy(NuzlockeLevelBalancingBattleStrategy):
    """Open a Roxanne battle with the strongest observed party matchup."""

    def __init__(self):
        super().__init__()
        self._opening_matchup_considered = False

    def decide_turn(self, battle_state: BattleState):
        if not self._opening_matchup_considered:
            self._opening_matchup_considered = True
            active = battle_state.own_side.active_battler
            opponent = battle_state.opponent.active_battler
            util = BattleStrategyUtil(battle_state)
            active_damage = util.get_strongest_move_against(active, opponent)
            active_max = (
                util.calculate_move_damage_range(active.moves[active_damage].move, active, opponent).max
                if active_damage is not None
                else 0
            )
            candidates = []
            for index in _eligible_indices(only_non_fainted=True):
                if index == active.party_index:
                    continue
                pokemon = get_party()[index]
                move = util.get_strongest_move_against(pokemon, opponent)
                if move is None:
                    continue
                damage = util.calculate_move_damage_range(pokemon.moves[move].move, pokemon, opponent).max
                candidates.append((damage, pokemon.current_hp, -index, index))
            if candidates and max(candidates)[0] > active_max and util.can_switch():
                return TurnAction.rotate_lead(max(candidates)[3])
        return super().decide_turn(battle_state)
