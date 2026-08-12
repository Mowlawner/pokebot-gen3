import tempfile
import unittest
from enum import Enum
from pathlib import Path

from modules.nuzlocke.events import (
    BattleEnded,
    BattleStarted,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    PokemonFainted,
    WhiteoutOccurred,
)
from modules.nuzlocke.persistence import JsonEventStore
from modules.nuzlocke.projection import (
    CampaignProjection,
    load_campaign_projection,
    reduce_events,
)


class State(Enum):
    OVERWORLD = 1
    BATTLE = 2
    WHITEOUT = 3


class TestCampaignProjection(unittest.TestCase):
    def test_empty_and_basic_reduction(self):
        self.assertEqual(reduce_events(()).last_event_sequence, 0)
        events = (
            MapChanged(1, None, (1, 2)),
            GameStateChanged(2, State.OVERWORLD, State.BATTLE),
            BattleStarted(2, ("WILD",), False, True, False),
            BattleEnded(3, "Won", ("WILD",), False, True, False),
        )
        state = reduce_events(events)
        self.assertEqual(state.current_map, (1, 2))
        self.assertEqual(state.current_game_state, State.BATTLE)
        self.assertIsNone(state.active_battle)
        self.assertEqual(state.last_battle_end.outcome, "Won")

    def test_party_faint_and_whiteout_are_historical(self):
        faint = PokemonFainted(2, 0, "Poochyena", "Pup", 123, "battle")
        events = (
            PartyChanged(1, (0, 1), (), False, ()),
            faint,
            PartyChanged(3, (), (0,), False, ()),
        )
        projection = CampaignProjection()
        for event in events:
            projection.apply(event)
        state = projection.state
        self.assertEqual(state.party_indices, (1,))
        self.assertEqual(state.observed_faints, (faint,))
        self.assertEqual(state.observed_pokemon[0].personality_value, 123)
        projection.apply(WhiteoutOccurred(4))
        self.assertEqual(len(projection.state.observed_whiteouts), 1)

    def test_replay_incremental_duplicate_and_unknown_party(self):
        events = (PartyChanged(1, (), (), True, (0,)), MapChanged(2, (1, 1), (1, 2)))
        incremental = CampaignProjection()
        incremental.apply(events[0])
        incremental.apply(events[1])
        incremental.apply(events[1])
        self.assertEqual(incremental.state, reduce_events(events))
        self.assertIsNone(incremental.state.party_indices)
        with self.assertRaises(ValueError):
            incremental.apply(MapChanged(4, (1, 2), (1, 3)), sequence=9)

    def test_persisted_replay_preserves_sessions_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json", session_id="a")
            store.append(MapChanged(1, None, (1, 1)))
            store.append(WhiteoutOccurred(2), "b")
            state = load_campaign_projection(JsonEventStore(store.path))
        self.assertEqual(state.current_session_id, "b")
        self.assertEqual(state.known_session_ids, ("a", "b"))
        self.assertEqual(state.last_event_sequence, 2)
        self.assertEqual(state.observed_whiteouts[0].session_id, "b")

    def test_events_are_not_mutated(self):
        event = MapChanged(1, (1, 1), (1, 2))
        original = event
        CampaignProjection().apply(event)
        self.assertEqual(event, original)


if __name__ == "__main__":
    unittest.main()
