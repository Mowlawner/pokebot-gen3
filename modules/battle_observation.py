"""Immutable battle observations, battle-local history, and event detection.

This module is deliberately an adapter around the existing memory readers.  It
does not make decisions and it does not expose the opponent party as planner
knowledge.  In particular, fields read from emulator memory are retained as
observations, while :class:`BattleKnowledge` decides which of them are visible
to a future planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Generic, TypeVar


class KnowledgeStatus(Enum):
    OBSERVED = "observed"
    KNOWN = "known"
    DERIVED = "derived"
    # Backwards-compatible spelling used by the first knowledge API.
    INFERRED = "derived"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class KnowledgeFact(Generic[T]):
    value: T | None
    status: KnowledgeStatus
    source: str | None = None


class KnowledgePolicy(Enum):
    STRICT_OBSERVED = "strict_observed"
    HUMAN_KNOWLEDGE = "human_knowledge"
    FULL_EMULATOR = "full_emulator"


class BattlePhase(Enum):
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ACTION_SELECTION = "action_selection"
    RESOLUTION = "resolution"
    SWITCHING = "switching"
    FAINT_HANDLING = "faint_handling"
    ENDING = "ending"
    COMPLETE = "complete"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MoveObservation:
    name: str
    pp: int | None = None
    total_pp: int | None = None
    disabled: bool = False
    encore: bool = False


@dataclass(frozen=True, slots=True)
class PokemonObservation:
    side: str
    party_index: int | None
    species: str | None
    level: int | None
    current_hp: int | None
    max_hp: int | None
    status: str | None
    temporary_status: tuple[str, ...] = ()
    moves: tuple[MoveObservation, ...] = ()
    stat_stages: tuple[tuple[str, int], ...] = ()
    effective_stats: tuple[tuple[str, int], ...] = ()
    ability: str | None = None
    held_item: str | None = None
    personality_value: int | None = None
    available_fields: frozenset[str] = frozenset()

    @property
    def identity(self) -> tuple[Any, ...]:
        if self.personality_value is not None:
            return (self.side, self.personality_value)
        return (self.side, self.party_index, self.species)


@dataclass(frozen=True, slots=True)
class BattleObservation:
    frame: int | None
    battle_id: str | None
    phase: BattlePhase
    active: bool
    ready_for_action: bool
    battle_type: tuple[str, ...] = ()
    is_trainer: bool | None = None
    is_wild: bool | None = None
    encounter_type: str | None = None
    outcome: str | None = None
    main_callback: str | None = None
    controller_callback: str | None = None
    script_instruction: str | None = None
    player_active: tuple[PokemonObservation, ...] = ()
    opponent_active: tuple[PokemonObservation, ...] = ()
    player_party: tuple[PokemonObservation, ...] = ()
    weather: str | None = None
    side_effects: tuple[tuple[str, Any], ...] = ()
    battler_count: int | None = None
    battle_flags: tuple[str, ...] = ()
    # current_turn is intentionally optional: Emerald's value is a battle
    # script/result byte in this reader, not a validated turn counter.
    turn_number: int | None = None
    # Populated only by a reliable action/script adapter.  Memory PP changes
    # alone are intentionally insufficient to populate this field.
    executed_move: tuple[str, str] | None = None
    executed_move_token: object | None = None
    move_failed: tuple[str, str] | None = None
    available_fields: frozenset[str] = frozenset()


class BattleEvent:
    """Marker base class for battle-local events."""


@dataclass(frozen=True, slots=True)
class TurnStarted(BattleEvent):
    frame: int | None


@dataclass(frozen=True, slots=True)
class MoveExecuted(BattleEvent):
    frame: int | None
    battler: str
    move: str


@dataclass(frozen=True, slots=True)
class MoveFailed(BattleEvent):
    frame: int | None
    battler: str
    move: str


@dataclass(frozen=True, slots=True)
class BattleActionRecord:
    """An action accepted only at an authoritative execution boundary."""

    turn_index: int
    frame: int | None
    battler: str
    move: str
    failed: bool = False
    source: KnowledgeStatus = KnowledgeStatus.OBSERVED


@dataclass(frozen=True, slots=True)
class DamageOccurred(BattleEvent):
    frame: int | None
    side: str
    party_index: int | None
    amount: int
    hp_before: int
    hp_after: int


@dataclass(frozen=True, slots=True)
class HealingOccurred(BattleEvent):
    frame: int | None
    side: str
    party_index: int | None
    amount: int
    hp_before: int
    hp_after: int


@dataclass(frozen=True, slots=True)
class StatusChanged(BattleEvent):
    frame: int | None
    side: str
    party_index: int | None
    before: str | None
    after: str | None


@dataclass(frozen=True, slots=True)
class StatStageChanged(BattleEvent):
    frame: int | None
    side: str
    party_index: int | None
    stat: str
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class PokemonFainted(BattleEvent):
    frame: int | None
    side: str
    party_index: int | None
    species: str | None


@dataclass(frozen=True, slots=True)
class SwitchRequired(BattleEvent):
    frame: int | None
    side: str


@dataclass(frozen=True, slots=True)
class PokemonSwitched(BattleEvent):
    frame: int | None
    side: str
    before_party_index: int | None
    after_party_index: int | None


@dataclass(frozen=True, slots=True)
class BattleEnded(BattleEvent):
    frame: int | None
    outcome: str | None


BattleEventType = (
    TurnStarted
    | MoveExecuted
    | MoveFailed
    | DamageOccurred
    | HealingOccurred
    | StatusChanged
    | StatStageChanged
    | PokemonFainted
    | SwitchRequired
    | PokemonSwitched
    | BattleEnded
)


def _battlers(observation: BattleObservation):
    return (*observation.player_active, *observation.opponent_active)


def events_between(previous: BattleObservation | None, current: BattleObservation) -> tuple[BattleEventType, ...]:
    """Derive only transitions supported by normalized observations.

    Move execution and failure require a future explicit action/script
    observation and are therefore not guessed from PP changes here.
    """
    if previous is None:
        initial: list[BattleEventType] = []
        if current.ready_for_action:
            initial.append(TurnStarted(current.frame))
        if current.executed_move is not None:
            initial.append(MoveExecuted(current.frame, current.executed_move[0], current.executed_move[1]))
        if current.move_failed is not None:
            initial.append(MoveFailed(current.frame, current.move_failed[0], current.move_failed[1]))
        return tuple(initial)
    events: list[BattleEventType] = []
    if current.ready_for_action and not previous.ready_for_action:
        events.append(TurnStarted(current.frame))
    if current.phase is BattlePhase.FAINT_HANDLING and previous.phase is not BattlePhase.FAINT_HANDLING:
        for side, battlers in (("player", current.player_active), ("opponent", current.opponent_active)):
            if any(p.current_hp == 0 for p in battlers):
                events.append(SwitchRequired(current.frame, side))
    if current.executed_move is not None and (
        current.executed_move != previous.executed_move or current.executed_move_token != previous.executed_move_token
    ):
        events.append(MoveExecuted(current.frame, current.executed_move[0], current.executed_move[1]))
    if current.move_failed is not None and current.move_failed != previous.move_failed:
        events.append(MoveFailed(current.frame, current.move_failed[0], current.move_failed[1]))
    for old, new in zip(_battlers(previous), _battlers(current)):
        if old.current_hp is not None and new.current_hp is not None and new.current_hp < old.current_hp:
            events.append(
                DamageOccurred(
                    current.frame,
                    new.side,
                    new.party_index,
                    old.current_hp - new.current_hp,
                    old.current_hp,
                    new.current_hp,
                )
            )
        elif old.current_hp is not None and new.current_hp is not None and new.current_hp > old.current_hp:
            events.append(
                HealingOccurred(
                    current.frame,
                    new.side,
                    new.party_index,
                    new.current_hp - old.current_hp,
                    old.current_hp,
                    new.current_hp,
                )
            )
        if old.status != new.status:
            events.append(StatusChanged(current.frame, new.side, new.party_index, old.status, new.status))
        old_stages, new_stages = dict(old.stat_stages), dict(new.stat_stages)
        for stat in old_stages.keys() | new_stages.keys():
            if old_stages.get(stat, 0) != new_stages.get(stat, 0):
                events.append(
                    StatStageChanged(
                        current.frame, new.side, new.party_index, stat, old_stages.get(stat, 0), new_stages.get(stat, 0)
                    )
                )
        if old.current_hp is not None and old.current_hp > 0 and new.current_hp == 0:
            events.append(PokemonFainted(current.frame, new.side, new.party_index, new.species))
    for old_active, new_active, side in (
        (previous.player_active, current.player_active, "player"),
        (previous.opponent_active, current.opponent_active, "opponent"),
    ):
        old_indexes = tuple(p.party_index for p in old_active)
        new_indexes = tuple(p.party_index for p in new_active)
        if old_indexes != new_indexes and old_indexes and new_indexes:
            events.append(PokemonSwitched(current.frame, side, old_indexes[0], new_indexes[0]))
    if previous.active and not current.active and current.outcome is not None:
        events.append(BattleEnded(current.frame, current.outcome))
    return tuple(events)


@dataclass(frozen=True, slots=True)
class BattleKnowledge:
    battle_id: str
    policy: KnowledgePolicy = KnowledgePolicy.STRICT_OBSERVED
    observations: tuple[BattleObservation, ...] = ()
    events: tuple[BattleEventType, ...] = ()
    opponent_pokemon: tuple[PokemonObservation, ...] = ()
    observed_opponent_moves: tuple[tuple[tuple[Any, ...], str], ...] = ()
    known_opponent_moves: tuple[tuple[tuple[Any, ...], str], ...] = ()
    inferred_opponent_moves: tuple[tuple[tuple[Any, ...], str], ...] = ()
    # Compatibility-friendly aggregate for consumers that only need moves
    # already revealed during this battle.
    revealed_opponent_moves: tuple[tuple[tuple[Any, ...], str], ...] = ()
    hp_history: tuple[tuple[tuple[Any, ...], int], ...] = ()
    damage_history: tuple[DamageOccurred, ...] = ()
    healing_history: tuple[HealingOccurred, ...] = ()
    status_history: tuple[StatusChanged, ...] = ()
    stat_stage_history: tuple[StatStageChanged, ...] = ()
    switch_history: tuple[PokemonSwitched, ...] = ()
    faint_history: tuple[PokemonFainted, ...] = ()
    outcome: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()
    action_history: tuple[BattleActionRecord, ...] = ()
    observed_player_moves: tuple[tuple[tuple[Any, ...], str], ...] = ()
    emulator_only_opponent_moves: tuple[tuple[tuple[Any, ...], str], ...] = ()

    def record(self, observation: BattleObservation) -> "BattleKnowledge":
        new_events = events_between(self.observations[-1] if self.observations else None, observation)
        opponents = list(self.opponent_pokemon)
        for pokemon in observation.opponent_active:
            if pokemon.identity not in [p.identity for p in opponents]:
                opponents.append(pokemon)
        observed_moves = list(self.observed_opponent_moves)
        known_moves = list(self.known_opponent_moves)
        player_moves = list(self.observed_player_moves)
        emulator_moves = list(self.emulator_only_opponent_moves)
        actions = list(self.action_history)
        current_turn_index = sum(isinstance(event, TurnStarted) for event in self.events)
        for event in new_events:
            if isinstance(event, (MoveExecuted, MoveFailed)):
                key = ((event.battler, None, None), event.move)
                actions.append(
                    BattleActionRecord(
                        current_turn_index,
                        event.frame,
                        event.battler,
                        event.move,
                        isinstance(event, MoveFailed),
                        KnowledgeStatus.OBSERVED,
                    )
                )
            if isinstance(event, MoveExecuted) and event.battler == "opponent":
                key = (("opponent", None, None), event.move)
                # The exact battler identity is not present in the current
                # low-level callback API; the event remains observed without
                # inventing an identity.
                if key not in observed_moves:
                    observed_moves.append(key)
            if isinstance(event, MoveExecuted) and event.battler == "player":
                key = (("player", None, None), event.move)
                if key not in player_moves:
                    player_moves.append(key)
        for pokemon in observation.opponent_active:
            for move in pokemon.moves:
                key = (pokemon.identity, move.name)
                if self.policy is KnowledgePolicy.FULL_EMULATOR and key not in known_moves:
                    known_moves.append((pokemon.identity, move.name))
                    emulator_moves.append(key)
        hp = list(self.hp_history)
        for pokemon in _battlers(observation):
            if pokemon.current_hp is not None:
                hp.append((pokemon.identity, pokemon.current_hp))
        damage = self.damage_history + tuple(e for e in new_events if isinstance(e, DamageOccurred))
        healing = self.healing_history + tuple(e for e in new_events if isinstance(e, HealingOccurred))
        status = self.status_history + tuple(e for e in new_events if isinstance(e, StatusChanged))
        stages = self.stat_stage_history + tuple(e for e in new_events if isinstance(e, StatStageChanged))
        switches = self.switch_history + tuple(e for e in new_events if isinstance(e, PokemonSwitched))
        faints = self.faint_history + tuple(e for e in new_events if isinstance(e, PokemonFainted))
        outcome = observation.outcome if observation.outcome is not None else self.outcome
        return BattleKnowledge(
            battle_id=self.battle_id,
            policy=self.policy,
            observations=self.observations + (observation,),
            events=self.events + new_events,
            opponent_pokemon=tuple(opponents),
            observed_opponent_moves=tuple(observed_moves),
            known_opponent_moves=tuple(known_moves),
            inferred_opponent_moves=self.inferred_opponent_moves,
            revealed_opponent_moves=tuple(observed_moves),
            hp_history=tuple(hp),
            damage_history=damage,
            healing_history=healing,
            status_history=status,
            stat_stage_history=stages,
            switch_history=switches,
            faint_history=faints,
            outcome=outcome,
            metadata=self.metadata,
            action_history=tuple(actions),
            observed_player_moves=tuple(player_moves),
            emulator_only_opponent_moves=tuple(emulator_moves),
        )

    def opponent_move_status(self, move: str) -> KnowledgeStatus:
        if any(name == move for _, name in self.observed_opponent_moves):
            return KnowledgeStatus.OBSERVED
        if any(name == move for _, name in self.known_opponent_moves):
            return KnowledgeStatus.KNOWN
        return KnowledgeStatus.UNKNOWN

    @property
    def opponent_move_facts(self) -> tuple[KnowledgeFact[str], ...]:
        """Visible move possibilities with their strongest source."""
        names = {name for _, name in self.known_opponent_moves}
        names.update(name for _, name in self.observed_opponent_moves)
        return tuple(
            KnowledgeFact(
                name,
                self.opponent_move_status(name),
                (
                    "battle_action"
                    if self.opponent_move_status(name) is KnowledgeStatus.OBSERVED
                    else "human_game_knowledge"
                ),
            )
            for name in sorted(names)
        )

    def fact(self, value: T | None, status: KnowledgeStatus, source: str | None = None) -> KnowledgeFact[T]:
        return KnowledgeFact(value, status, source)


class BattleKnowledgeRecorder:
    """Small stateful boundary used by runtime code; one recorder per bot."""

    def __init__(self, policy: KnowledgePolicy = KnowledgePolicy.STRICT_OBSERVED):
        self.policy = policy
        self.knowledge: BattleKnowledge | None = None
        self.last_completed_knowledge: BattleKnowledge | None = None
        self._last_active = False

    def record(self, observation: BattleObservation) -> BattleKnowledge | None:
        if observation.active and not self._last_active:
            self.knowledge = BattleKnowledge(observation.battle_id or "battle", self.policy)
        if self.knowledge is not None:
            self.knowledge = self.knowledge.record(observation)
            if not observation.active and self._last_active:
                self.last_completed_knowledge = self.knowledge
        self._last_active = observation.active
        return self.knowledge

    def reset(self) -> None:
        self.knowledge = None
        self.last_completed_knowledge = None
        self._last_active = False


def _safe(call: Callable[[], Any], default: Any = None) -> Any:
    try:
        return call()
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return default


def _pokemon_snapshot(pokemon: Any, side: str) -> PokemonObservation:
    def name(value):
        return getattr(value, "name", value) if value is not None else None

    raw_moves = _safe(lambda: pokemon.moves, ()) or ()
    moves = tuple(MoveObservation(name(m.move), m.pp, m.total_pp) for m in raw_moves if m is not None)
    stages = _safe(lambda: pokemon.stats_modifiers)
    stage_values = (
        tuple(
            (key, getattr(stages, key))
            for key in ("attack", "defence", "speed", "special_attack", "special_defence", "accuracy", "evasion")
        )
        if stages
        else ()
    )
    stats = _safe(lambda: pokemon.modified_stats())
    stat_values = (
        tuple(
            (key, getattr(stats, key))
            for key in ("hp", "attack", "defence", "speed", "special_attack", "special_defence")
        )
        if stats
        else ()
    )
    permanent_status = _safe(lambda: pokemon.status_permanent)
    if permanent_status is None:
        permanent_status = _safe(lambda: pokemon.status_condition)
    temporary_status = _safe(lambda: pokemon.status_temporary, ()) or ()
    return PokemonObservation(
        side,
        getattr(pokemon, "party_index", getattr(pokemon, "index", None)),
        name(_safe(lambda: pokemon.species)),
        _safe(lambda: pokemon.level),
        _safe(lambda: pokemon.current_hp),
        _safe(lambda: pokemon.total_hp),
        name(permanent_status),
        tuple(name(x) for x in temporary_status),
        moves,
        stage_values,
        stat_values,
        name(_safe(lambda: pokemon.ability)),
        name(_safe(lambda: pokemon.held_item)),
        _safe(lambda: pokemon.personality_value),
        frozenset(
            {
                "species",
                "level",
                "current_hp",
                "max_hp",
                "status",
                "moves",
                "stat_stages",
                "effective_stats",
                "ability",
                "held_item",
            }
        ),
    )


def _phase(
    active: bool, game_state: Any, main_callback: str | None, script: str | None, controller: str | None
) -> BattlePhase:
    if not active:
        return BattlePhase.ENDING if game_state == "BATTLE_ENDING" else BattlePhase.INACTIVE
    if game_state == "BATTLE_STARTING":
        return BattlePhase.STARTING
    if controller in {"HandleInputChooseAction", "sub_802C098", "bx_battle_menu_t6_2"}:
        return BattlePhase.ACTION_SELECTION
    if script in {"BattleScript_HandleFaintedMon", "BattleScript_FaintedMonEnd"}:
        return BattlePhase.FAINT_HANDLING
    if script and ("Switch" in script or "switch" in script):
        return BattlePhase.SWITCHING
    if main_callback in {"HandleTurnActionSelectionState", "sub_8012324"}:
        return BattlePhase.ACTION_SELECTION
    return BattlePhase.RESOLUTION


def capture_battle_observation(action: Any | None = None) -> BattleObservation:
    """Read one normalized point-in-time snapshot.

    Every reader is guarded because battle start/end frames legitimately contain
    partially initialized structures.  Such fields remain unavailable rather
    than being represented by fabricated defaults.
    """
    from modules.battle_state import (
        battle_is_active,
        get_battle_state,
        get_current_battle_script_instruction,
        get_encounter_type,
        get_last_battle_outcome,
        get_main_battle_callback,
    )
    from modules.context import context
    from modules.memory import get_game_state
    from modules.pokemon_party import get_party

    active = bool(_safe(battle_is_active, False))
    frame = _safe(lambda: context.emulator.get_frame_count())
    game_state = _safe(get_game_state)
    game_state_name = getattr(game_state, "name", game_state)
    main = _safe(get_main_battle_callback)
    script = _safe(get_current_battle_script_instruction)
    controller = _safe(
        lambda: __import__(
            "modules.battle_state", fromlist=["get_battle_controller_callback"]
        ).get_battle_controller_callback(0)
    )
    state = _safe(get_battle_state) if active else None
    own = tuple(
        _pokemon_snapshot(p, "player") for p in (_safe(lambda: state.own_side.active_battlers, ()) if state else ())
    )
    opponent = tuple(
        _pokemon_snapshot(p, "opponent") for p in (_safe(lambda: state.opponent.active_battlers, ()) if state else ())
    )
    party = tuple(_pokemon_snapshot(p, "player_party") for p in (_safe(get_party, ()) or ()))
    battle_type = tuple(_safe(lambda: state.type_names, ()) or ()) if state else ()
    side_effects = ()
    if state:
        for side_name, side in (("player", _safe(lambda: state.own_side)), ("opponent", _safe(lambda: state.opponent))):
            if side is not None:
                side_effects += tuple(
                    (name, _safe(lambda prop=prop: getattr(side, prop)))
                    for name, prop in (
                        ("reflect", "reflect_timer"),
                        ("lightscreen", "lightscreen_timer"),
                        ("mist", "mist_timer"),
                        ("safeguard", "safeguard_timer"),
                        ("follow_me", "follow_me_timer"),
                        ("spikes", "spikes_amount"),
                    )
                )
    outcome = _safe(get_last_battle_outcome)
    encounter = _safe(get_encounter_type)
    ready = (
        active
        and main in {"HandleTurnActionSelectionState", "sub_8012324"}
        and controller in {"HandleInputChooseAction", "sub_802C098", "bx_battle_menu_t6_2"}
    )
    observation = BattleObservation(
        frame,
        "active-battle" if active else None,
        _phase(active, game_state_name, main, script, controller),
        active,
        ready,
        battle_type,
        ("Trainer" in battle_type) if battle_type else None,
        ("Trainer" not in battle_type) if battle_type else None,
        getattr(encounter, "value", encounter),
        getattr(outcome, "name", outcome),
        main,
        controller,
        script,
        own,
        opponent,
        party,
        getattr(_safe(lambda: state.weather) if state else None, "name", None),
        side_effects,
        _safe(lambda: state._battler_count) if state else None,
        battle_type,
        None,
        None,
        None,
        None,
        frozenset(
            {"phase", "active", "ready_for_action", "battle_type", "callbacks", "active_pokemon", "party", "outcome"}
        ),
    )
    # ``action`` is supplied only by a ROM-specific adapter that has already
    # established a unique execution edge.  The current Emerald adapter has
    # no such reader and therefore passes None.
    if action is not None:
        return replace(
            observation,
            executed_move=(action.battler, action.move) if not action.failed else None,
            executed_move_token=action.token,
            move_failed=(action.battler, action.move) if action.failed else None,
        )
    return observation


# Planner-facing runtime knowledge is bounded by what a player can know.  The
# strict policy remains available to callers that need observation-only data,
# while FULL_EMULATOR remains an explicit debugging option.
default_battle_recorder = BattleKnowledgeRecorder(KnowledgePolicy.HUMAN_KNOWLEDGE)


def observe_current_battle(action: Any | None = None) -> BattleKnowledge | None:
    """Record the current frame for the observational runtime integration."""
    return default_battle_recorder.record(capture_battle_observation(action))
