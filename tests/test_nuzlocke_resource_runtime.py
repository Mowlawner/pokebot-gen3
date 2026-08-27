import unittest
from unittest.mock import patch

from modules.goals import ActivateTrigger
from modules.nuzlocke.resource_policy import PartyResource, ResourceDecision, ResourceSnapshot, RouteRecovery
from modules.nuzlocke.resource_runtime import (
    CampaignCapability,
    HealingSource,
    HealingSourceType,
    discover_healing_source,
    execute_campaign_recovery,
)
from modules.nuzlocke.resource_runtime import _execute_healing_source_interaction
from modules.nuzlocke.resource_runtime import _resolve_recovery_interaction
from modules.nuzlocke.resource_runtime import _navigate_recovery_to_center
from modules.nuzlocke.resource_runtime import _wait_for_center_interior
from modules.agent_control import AgentAction, AgentActionType, ActionDecision
from modules.agent_control import AgentObservation
from modules.interaction_state import InteractionObservation
from modules.memory import GameState
from modules.map_data import MapRSE, PokemonCenter


class CampaignCapabilityTests(unittest.TestCase):
    def test_observed_center_nurse_becomes_healing_affordance(self):
        trigger = type(
            "Trigger",
            (),
            {
                "kind": "object_interaction",
                "affordance_id": "OldaleTown_PokemonCenter_1F_EventScript_Nurse",
                "script_symbol": "OldaleTown_PokemonCenter_1F_EventScript_Nurse",
                "activation_locations": frozenset({((2, 2), (7, 4))}),
                "trigger_id": "object:1:OldaleTown_PokemonCenter_1F_EventScript_Nurse",
            },
        )()
        observation = type(
            "Observation",
            (),
            {
                "overworld": type("World", (), {"triggers": (trigger,)})(),
            },
        )()
        source = discover_healing_source(observation=observation)
        self.assertEqual(source.source_type, HealingSourceType.POKEMON_CENTER_NURSE)
        self.assertEqual(source.interaction_trigger_id, trigger.trigger_id)
        self.assertEqual(source.interaction_target, trigger.trigger_id)

    def test_healing_source_dialogue_uses_common_action_and_fresh_completion(self):
        source = HealingSource("nurse", HealingSourceType.FULL_PARTY_PROVIDER, None, None)
        dialogue = AgentObservation(InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True))
        overworld = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        actions = [
            ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE)),
            ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE)),
        ]
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, False, True]), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[dialogue, overworld]
        ), patch("modules.nuzlocke.resource_runtime.select_action", side_effect=actions), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

        # The common selector owns dialogue; no Center helper is involved.
        self.assertEqual(
            [decision.action.action_type for decision in actions],
            [AgentActionType.ADVANCE_DIALOGUE, AgentActionType.WAIT_REOBSERVE],
        )

    def test_healing_source_does_not_claim_success_after_interaction(self):
        source = HealingSource("nurse", HealingSourceType.FULL_PARTY_PROVIDER, None, None)
        overworld = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False] * 40), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=overworld
        ), patch("modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor):
            with self.assertRaisesRegex(RuntimeError, "HEALING_NOT_CONFIRMED"):
                list(_execute_healing_source_interaction(source))

    def test_healing_source_retains_ownership_during_delayed_interaction_start(self):
        source = HealingSource("nurse", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        overworld = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        dialogue = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True, controllable=False)
        )
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        with patch(
            "modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, False, False, True]
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[overworld, overworld, dialogue]
        ), patch(
            "modules.nuzlocke.resource_runtime.select_action",
            return_value=ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE)),
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

    def test_script_owned_controllable_interaction_remains_owned(self):
        source = HealingSource("source", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        script_owned = AgentObservation(
            InteractionObservation(
                GameState.OVERWORLD,
                controllable=True,
                script_active=True,
                script_function="SomeInteraction",
                native_function="WaitForEffect",
            )
        )
        actions = []
        executor = type("Executor", (), {"execute": lambda self, action, observed: actions.append(action)})()
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, False, True]), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=script_owned
        ), patch("modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor):
            list(_execute_healing_source_interaction(source))
        # The active script preserves ownership without emitting another
        # interaction or falsely reporting HEALING_NOT_CONFIRMED.
        self.assertEqual(actions, [])

    def test_ready_confirmation_precedes_generic_script_ownership(self):
        source = HealingSource("source", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        confirmation = AgentObservation(
            InteractionObservation(
                GameState.OVERWORLD,
                controllable=True,
                script_active=True,
                choice_options=("YES", "NO"),
                choice_menu_active=True,
                choice_menu_input_ready=True,
            )
        )
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        decisions = [ActionDecision(AgentAction(AgentActionType.CHOOSE_DIALOGUE_OPTION, option="YES"))]
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, True]), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=confirmation
        ), patch("modules.nuzlocke.resource_runtime.select_action", side_effect=decisions), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

        self.assertEqual(decisions[0].action.action_type, AgentActionType.CHOOSE_DIALOGUE_OPTION)

    def test_center_recovery_preserves_location_for_selected_center_lookup(self):
        location = ("Route101", (15, 10))
        center = object()

        def heal(_center, **_kwargs):
            yield "healed"

        with patch(
            "modules.nuzlocke.resource_runtime.observe_route_recovery",
            return_value=RouteRecovery(center_available=True, safe_to_reach_center=True),
        ), patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location) as current, patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=center
        ) as find, patch(
            "modules.nuzlocke.resource_runtime._resolve_recovery_interaction", return_value=iter(())
        ), patch(
            "modules.nuzlocke.resource_runtime._navigate_recovery_to_center", return_value=iter(())
        ), patch(
            "modules.nuzlocke.resource_runtime._wait_for_center_interior", return_value=iter(())
        ), patch(
            "modules.nuzlocke.resource_runtime.wait_for_player_avatar_to_be_controllable", return_value=iter(())
        ) as wait, patch(
            "modules.nuzlocke.resource_runtime.heal_in_pokemon_center", side_effect=heal
        ), patch(
            "modules.nuzlocke.resource_runtime.party_is_restored", return_value=True
        ):
            self.assertEqual(list(execute_campaign_recovery()), ["healed"])
        current.assert_called_once_with()
        find.assert_called_once_with(location)
        wait.assert_called_once_with()

    def test_recovery_interaction_waits_for_passive_transition(self):
        observation = AgentObservation(InteractionObservation(GameState.UNKNOWN))
        decision = ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))
        with patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            side_effect=[observation, AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))],
        ), patch("modules.nuzlocke.resource_runtime.select_action", return_value=decision), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor"
        ) as executor:
            self.assertEqual(list(_resolve_recovery_interaction()), [None])
        executor.return_value.execute.assert_not_called()

    def test_recovery_interaction_advances_actionable_dialogue(self):
        observation = AgentObservation(InteractionObservation(GameState.OVERWORLD))
        decision = ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE))
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        controllable = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        with patch("modules.nuzlocke.resource_runtime.observe_agent", side_effect=[observation, controllable]), patch(
            "modules.nuzlocke.resource_runtime.select_action",
            side_effect=[decision, ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))],
        ), patch("modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor), patch(
            "modules.nuzlocke.resource_runtime.wait_for_player_avatar_to_be_controllable", return_value=iter(())
        ):
            self.assertEqual(list(_resolve_recovery_interaction()), [None])

    def test_recovery_navigation_uses_observation_driven_location_goal(self):
        center = type("Center", (), {"value": ("OldaleTown", (6, 16))})()

        class FakeLoop:
            def run(self):
                return self

            def __iter__(self):
                return iter(("navigation",))

        with patch("modules.nuzlocke.resource_runtime.AgentControlLoop", return_value=FakeLoop()) as loop:
            self.assertEqual(list(_navigate_recovery_to_center(center)), ["navigation"])
        goal = loop.call_args.kwargs["goal"]
        self.assertEqual(goal.location, ("OldaleTown", (6, 16)))

    def test_recovery_handoff_waits_for_stable_center_interior(self):
        transient = type(
            "Observation", (), {"overworld": type("World", (), {"map_id": MapRSE.OLDALE_TOWN, "controllable": True})()}
        )()
        stable = type(
            "Observation",
            (),
            {"overworld": type("World", (), {"map_id": MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F, "controllable": True})()},
        )()
        with patch("modules.nuzlocke.resource_runtime.observe_agent", side_effect=[transient, stable]):
            self.assertEqual(list(_wait_for_center_interior(PokemonCenter.OldaleTown)), [None])

    def test_capability_delegates_original_tactical_goal_without_readiness_preflight(self):
        goal = ActivateTrigger("introductory_rival")
        delegated = []

        def delegate(received_goal):
            delegated.append(received_goal)
            yield "tactical"

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "none", False),),
            bag_healing_items=(),
            pc_healing_items=(),
        )
        capability = CampaignCapability("complete_intro_rival", object(), goal, delegate)

        with patch(
            "modules.nuzlocke.resource_runtime.observe_resource_snapshot", return_value=snapshot
        ) as observe, patch("modules.nuzlocke.resource_runtime.assess_campaign_resources") as assess, patch(
            "modules.nuzlocke.resource_runtime.observe_route_recovery"
        ) as find_center:
            self.assertEqual(next(capability()), "tactical")

        self.assertEqual(delegated, [goal])
        observe.assert_not_called()
        assess.assert_not_called()
        find_center.assert_not_called()

    def test_capability_does_not_mount_recovery_from_internal_resource_policy(self):
        goal = ActivateTrigger("introductory_rival")
        delegated = []

        def delegate(received_goal):
            delegated.append(received_goal)
            yield "tactical"

        snapshot = ResourceSnapshot(
            party=(PartyResource(5, 20, "none", False),),
            bag_healing_items=(),
            pc_healing_items=(),
        )
        capability = CampaignCapability("complete_intro_rival", object(), goal, delegate)
        with patch(
            "modules.nuzlocke.resource_runtime.observe_resource_snapshot", return_value=snapshot
        ) as observe, patch("modules.nuzlocke.resource_runtime.assess_campaign_resources") as assess:
            self.assertEqual(list(capability()), ["tactical"])

        self.assertEqual(delegated, [goal])
        observe.assert_not_called()
        assess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
