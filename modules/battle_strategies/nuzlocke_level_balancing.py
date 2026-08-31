"""Level-balancing battle strategies that respect Nuzlocke deaths."""

from __future__ import annotations

from modules.battle_state import BattleState
from modules.battle_strategies import DefaultBattleStrategy, TurnAction
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_strategies.level_balancing import LevelBalancingBattleStrategy
from modules.context import context
from modules.nuzlocke.identity import PokemonIdentity
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
    # behavior. During campaign preparation, an unreadable identity is not a
    # safe basis for selecting a Pokémon whose death status matters.
    if getattr(context, "nuzlocke_runtime", None) is None:
        return True
    identity = PokemonIdentity.from_pokemon(pokemon)
    return identity is not None and identity not in _nuzlocke_dead_identities()


def _eligible_indices(*, only_non_fainted: bool = False) -> tuple[int, ...]:
    return tuple(
        index
        for index, pokemon in enumerate(get_party())
        if _is_eligible_pokemon(pokemon)
        and (not only_non_fainted or pokemon.current_hp > 0)
    )


def _lowest_level_index(*, only_non_fainted: bool = False) -> int | None:
    indices = _eligible_indices(only_non_fainted=only_non_fainted)
    return min(indices, key=lambda index: (get_party()[index].level, index)) if indices else None


class NuzlockeLevelBalancingBattleStrategy(LevelBalancingBattleStrategy):
    """Balance living campaign Pokémon without ever sending out a dead one."""

    def pokemon_can_battle(self, pokemon) -> bool:
        return _is_eligible_pokemon(pokemon) and super().pokemon_can_battle(pokemon)

    def party_can_battle(self) -> bool:
        lowest = _lowest_level_index()
        return lowest is not None and self.pokemon_can_battle(get_party()[lowest])

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        index = _lowest_level_index(only_non_fainted=True)
        if index is None:
            raise RuntimeError("no living Pokémon is available for campaign preparation")
        return index

    def has_replacement_after_faint(self, battle_state: BattleState) -> bool:
        active_indices = {
            battler.party_index
            for battler in battle_state.own_side.active_battlers
            if battler is not None
        }
        return any(index not in active_indices for index in _eligible_indices(only_non_fainted=True))

    def choose_new_lead_after_battle(self) -> int | None:
        index = _lowest_level_index(only_non_fainted=True)
        return index if index is not None and index > 0 else None

    def decide_turn(self, battle_state: BattleState):
        action = super().decide_turn(battle_state)
        # The generic battle planner does not know the Nuzlocke death
        # projection. Never honor a planner switch to a healed dead Pokémon.
        if action[0] is TurnAction.RotateLead:
            replacement = get_party()[action[1]]
            active_indices = {
                battler.party_index
                for battler in battle_state.own_side.active_battlers
                if battler is not None
            }
            if (
                not _is_eligible_pokemon(replacement)
                or replacement.current_hp <= 0
                or action[1] in active_indices
            ):
                util = BattleStrategyUtil(battle_state)
                move = util.get_strongest_move_against(
                    battle_state.own_side.active_battler,
                    battle_state.opponent.active_battler,
                )
                if move is not None:
                    return TurnAction.use_move(move)
        return action


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
                    context.battle_decision_detail = (
                        f"opening Growl {self._opening_growls_used}/{self._OPENING_GROWLS}"
                    )
                    context.message = (
                        f"EMERALD INTRO RIVAL SETUP: Growl "
                        f"{self._opening_growls_used}/{self._OPENING_GROWLS}"
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
