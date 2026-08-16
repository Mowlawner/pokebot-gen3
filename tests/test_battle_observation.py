import unittest
from types import SimpleNamespace

from modules.battle_observation import (
    BattleEnded,
    BattleKnowledge,
    BattleObservation,
    BattlePhase,
    DamageOccurred,
    MoveExecuted,
    MoveFailed,
    KnowledgeFact,
    KnowledgePolicy,
    KnowledgeStatus,
    MoveObservation,
    PokemonObservation,
    PokemonSwitched,
    StatusChanged,
    events_between,
)
from modules.battle_synchronization import (
    AuthoritativeAction,
    BattleActionSynchronizer,
    automatic_action_marker_available,
)


def pokemon(side, index, hp, *, status=None, stages=(), moves=()):
    return PokemonObservation(
        side,
        index,
        "Treecko" if side == "player" else "Poochyena",
        10,
        hp,
        20,
        status,
        moves=moves,
        stat_stages=stages,
    )


def observation(frame, *, active=True, ready=True, player=(), opponent=(), outcome=None):
    return BattleObservation(
        frame,
        "battle-1" if active else None,
        BattlePhase.ACTION_SELECTION if ready else BattlePhase.RESOLUTION,
        active,
        ready,
        ("Trainer",),
        True,
        False,
        "trainer",
        outcome,
        player_active=player,
        opponent_active=opponent,
    )


class BattleObservationTests(unittest.TestCase):
    def test_normalized_snapshot_is_immutable_and_preserves_fields(self):
        move = MoveObservation("Tackle", pp=34, total_pp=35)
        mon = pokemon("player", 0, 18, status="Healthy", moves=(move,))
        current = observation(42, player=(mon,))

        self.assertEqual(current.frame, 42)
        self.assertTrue(current.ready_for_action)
        self.assertEqual(current.player_active[0].moves[0].pp, 34)
        with self.assertRaises(AttributeError):
            current.frame = 43

    def test_pokemon_snapshot_invokes_modified_stats_method(self):
        from modules.battle_observation import _pokemon_snapshot

        modified_stats = SimpleNamespace(
            hp=20,
            attack=31,
            defence=22,
            speed=30,
            special_attack=25,
            special_defence=24,
        )
        pokemon = SimpleNamespace(
            party_index=0,
            species=SimpleNamespace(name="Treecko"),
            level=10,
            current_hp=18,
            total_hp=20,
            status_permanent=SimpleNamespace(name="Healthy"),
            status_temporary=(),
            moves=(),
            stats_modifiers=SimpleNamespace(
                attack=0,
                defence=0,
                speed=0,
                special_attack=0,
                special_defence=0,
                accuracy=0,
                evasion=0,
            ),
            modified_stats=lambda: modified_stats,
            ability=SimpleNamespace(name="Overgrow"),
            held_item=None,
            personality_value=123,
        )

        snapshot = _pokemon_snapshot(pokemon, "player")

        self.assertEqual(
            dict(snapshot.effective_stats),
            {
                "hp": 20,
                "attack": 31,
                "defence": 22,
                "speed": 30,
                "special_attack": 25,
                "special_defence": 24,
            },
        )
        self.assertFalse(any(callable(value) for _, value in snapshot.effective_stats))

    def test_transition_events_cover_damage_status_stage_switch_faint_and_end(self):
        old = observation(
            1,
            ready=True,
            player=(pokemon("player", 0, 20, stages=(("attack", 0),)),),
            opponent=(pokemon("opponent", 0, 20),),
        )
        new = observation(
            2,
            ready=False,
            player=(pokemon("player", 2, 0, stages=(("attack", 1),)),),
            opponent=(pokemon("opponent", 0, 12, status="Poison"),),
        )
        events = events_between(old, new)

        self.assertTrue(any(isinstance(event, DamageOccurred) and event.amount == 8 for event in events))
        self.assertTrue(any(isinstance(event, StatusChanged) and event.after == "Poison" for event in events))
        self.assertTrue(any(isinstance(event, PokemonSwitched) for event in events))
        self.assertTrue(any(event.__class__.__name__ == "StatStageChanged" for event in events))
        self.assertTrue(any(event.__class__.__name__ == "PokemonFainted" for event in events))

        ended = observation(3, active=False, ready=False, outcome="Won")
        self.assertTrue(any(isinstance(event, BattleEnded) for event in events_between(new, ended)))

    def test_ambiguous_initial_and_resolution_frames_do_not_invent_move_events(self):
        current = observation(
            1, ready=False, opponent=(pokemon("opponent", 0, 20, moves=(MoveObservation("Tackle", 34, 35),)),)
        )
        events = events_between(None, current)
        self.assertEqual(events, ())
        self.assertFalse(
            any(
                event.__class__.__name__ == "MoveExecuted"
                for event in events_between(current, observation(2, ready=False))
            )
        )

    def test_knowledge_accumulates_history_and_policy_visibility(self):
        first = observation(1, opponent=(pokemon("opponent", 0, 20, moves=(MoveObservation("Tackle", 34, 35),)),))
        second = observation(
            2,
            ready=False,
            opponent=(pokemon("opponent", 0, 12, status="Poison", moves=first.opponent_active[0].moves),),
        )
        knowledge = BattleKnowledge("battle-1").record(first).record(second)

        self.assertEqual(len(knowledge.observations), 2)
        self.assertEqual(knowledge.damage_history[0].amount, 8)
        self.assertEqual(knowledge.status_history[0].before, None)
        self.assertEqual(knowledge.status_history[0].after, "Poison")
        self.assertEqual(knowledge.revealed_opponent_moves, ())
        self.assertEqual(len(knowledge.hp_history), 2)

        full = BattleKnowledge("battle-1", KnowledgePolicy.FULL_EMULATOR).record(first)
        self.assertEqual(full.known_opponent_moves[0][1], "Tackle")

    def test_visibility_values_are_not_collapsed(self):
        observed = KnowledgeFact("Tackle", KnowledgeStatus.OBSERVED, "battle_log")
        known = KnowledgeFact(("Tackle", "Growl"), KnowledgeStatus.KNOWN, "species_learnset")
        inferred = KnowledgeFact("Growl", KnowledgeStatus.INFERRED, "damage_analysis")
        unavailable = KnowledgeFact(None, KnowledgeStatus.UNAVAILABLE, "reader_not_supported")
        unknown = KnowledgeFact(None, KnowledgeStatus.UNKNOWN)
        self.assertEqual(
            {observed.status, known.status, inferred.status, unavailable.status, unknown.status}, set(KnowledgeStatus)
        )

    def test_synchronizer_requires_explicit_authoritative_marker(self):
        sync = BattleActionSynchronizer()
        current = observation(2, ready=False)
        self.assertEqual(sync.observe(current), ())
        self.assertFalse(automatic_action_marker_available())
        self.assertEqual(
            sync.observe(current, AuthoritativeAction("p-1", "player", "Flamethrower")),
            (MoveExecuted(2, "player", "Flamethrower"),),
        )

    def test_synchronizer_deduplicates_frames_and_failed_actions(self):
        sync = BattleActionSynchronizer()
        current = observation(9, ready=False)
        marker = AuthoritativeAction("o-1", "opponent", "Tackle", failed=True)
        expected = (MoveFailed(9, "opponent", "Tackle"),)
        self.assertEqual(sync.observe(current, marker), expected)
        self.assertEqual(sync.observe(current, marker), ())
        self.assertEqual(sync.observe(observation(10, ready=False), marker), ())

    def test_authoritative_actions_accumulate_per_turn_without_predicting_reuse(self):
        sync = BattleActionSynchronizer()
        first = observation(1, ready=True)
        second = observation(2, ready=False)
        third = observation(3, ready=False)
        first = first.__class__(
            **{
                **{field: getattr(first, field) for field in first.__dataclass_fields__},
                "executed_move": ("opponent", "Ember"),
                "executed_move_token": "o-1",
            }
        )
        second = second.__class__(
            **{
                **{field: getattr(second, field) for field in second.__dataclass_fields__},
                "executed_move": ("opponent", "Ember"),
                "executed_move_token": "o-2",
            }
        )
        knowledge = BattleKnowledge("battle-1", KnowledgePolicy.HUMAN_KNOWLEDGE)
        knowledge = knowledge.record(first).record(second).record(third)

        self.assertEqual(knowledge.observed_opponent_moves[0][1], "Ember")
        self.assertEqual([action.move for action in knowledge.action_history], ["Ember", "Ember"])
        self.assertEqual([action.turn_index for action in knowledge.action_history], [0, 1])
        self.assertEqual(knowledge.opponent_move_status("Ember"), KnowledgeStatus.OBSERVED)
        # Seeing Ember does not make an unobserved move appear in the set.
        self.assertEqual(knowledge.opponent_move_status("Flamethrower"), KnowledgeStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
