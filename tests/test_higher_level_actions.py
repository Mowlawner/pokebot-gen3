import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class _ShopEmulator:
    def __init__(self):
        self.fresh_presses = []
        self.presses = []

    def press_button_fresh(self, button):
        self.fresh_presses.append(button)

    def press_button(self, button):
        self.presses.append(button)


class BuyInShopTests(unittest.TestCase):
    def test_shop_wait_advances_post_purchase_dialogue_before_task_condition(self):
        from modules.agent_control import AgentAction, AgentActionType, ActionDecision, AgentObservation
        from modules.interaction_state import InteractionObservation, InteractionPhase, InteractionType, classify_interaction
        from modules.memory import GameState
        from modules.modes.util import higher_level_actions as actions

        dialogue = AgentObservation(
            InteractionObservation(
                GameState.OVERWORLD,
                dialogue_waiting=True,
                interaction_phase=InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
            )
        )
        decision = ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE))
        executor = Mock()
        condition = Mock(return_value=True)

        with (
            patch.object(actions, "observe_interaction", return_value=dialogue.interaction),
            patch.object(actions, "observe_agent", return_value=dialogue),
            patch.object(actions, "select_action", return_value=decision),
        ):
            waiter = actions._wait_for_shop_condition(
                condition,
                "return task",
                dialogue_executor=executor,
            )
            next(waiter)

        self.assertIs(dialogue.interaction_type, InteractionType.DIALOGUE)
        condition.assert_not_called()
        executor.execute.assert_called_once_with(decision.action, dialogue)

    def test_shop_wait_does_not_mistake_stale_message_lifecycle_for_dialogue(self):
        from modules.interaction_state import InteractionObservation, InteractionPhase
        from modules.memory import GameState
        from modules.modes.util import higher_level_actions as actions

        shop_menu = InteractionObservation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        executor = Mock()
        condition = Mock(side_effect=(False, True))

        with patch.object(actions, "observe_interaction", return_value=shop_menu):
            waiter = actions._wait_for_shop_condition(
                condition,
                "return task",
                dialogue_executor=executor,
            )
            next(waiter)

        condition.assert_called_once_with()
        executor.execute.assert_not_called()

    def test_shop_wait_does_not_submit_rendering_confirmation_text(self):
        from modules.interaction_state import (
            InteractionObservation,
            InteractionPhase,
            InteractionType,
            classify_interaction,
        )
        from modules.memory import GameState
        from modules.modes.util import higher_level_actions as actions

        rendering_message = InteractionObservation(
            GameState.OVERWORLD,
            dialogue_waiting=True,
            field_message_lifecycle_active=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        executor = Mock()
        condition = Mock(side_effect=(False, True))

        with patch.object(actions, "observe_interaction", return_value=rendering_message):
            waiter = actions._wait_for_shop_condition(
                condition,
                "return task",
                dialogue_executor=executor,
            )
            next(waiter)

        self.assertIs(classify_interaction(rendering_message), InteractionType.DIALOGUE)
        condition.assert_called_once_with()
        executor.execute.assert_not_called()

    def test_shop_wait_keeps_purchase_task_paused_while_confirmation_renders(self):
        from modules.interaction_state import InteractionObservation, InteractionPhase
        from modules.memory import GameState
        from modules.modes.util import higher_level_actions as actions

        rendering_message = InteractionObservation(
            GameState.OVERWORLD,
            dialogue_waiting=True,
            field_message_lifecycle_active=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
            metadata={"field_message_task_active": True},
        )
        input_message = InteractionObservation(
            GameState.OVERWORLD,
            dialogue_waiting=True,
            field_message_lifecycle_active=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
            metadata={"field_message_task_active": True},
        )
        decision = Mock()
        executor = Mock()
        condition = Mock(return_value=True)

        with (
            patch.object(actions, "observe_interaction", side_effect=(rendering_message, input_message)),
            patch.object(actions, "observe_agent", return_value=Mock()),
            patch.object(actions, "select_action", return_value=decision),
        ):
            waiter = actions._wait_for_shop_condition(
                condition,
                "next purchase task",
                dialogue_executor=executor,
            )
            next(waiter)
            next(waiter)

        condition.assert_not_called()
        executor.execute.assert_called_once()

    def test_shop_interaction_is_not_settled_during_confirmation_message(self):
        from modules.interaction_state import InteractionObservation, InteractionPhase
        from modules.memory import GameState
        from modules.modes.util import higher_level_actions as actions

        rendering_message = InteractionObservation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
            metadata={"field_message_task_active": True},
        )
        input_message = InteractionObservation(
            GameState.OVERWORLD,
            dialogue_waiting=True,
            field_message_lifecycle_active=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
            metadata={"field_message_task_active": True},
        )
        menu = InteractionObservation(GameState.OVERWORLD)

        with patch.object(
            actions,
            "observe_interaction",
            side_effect=(rendering_message, input_message, menu),
        ):
            self.assertFalse(actions._shop_interaction_is_settled())
            self.assertFalse(actions._shop_interaction_is_settled())
            self.assertTrue(actions._shop_interaction_is_settled())

    def test_buy_menu_open_uses_a_fresh_edge_after_clerk_interaction(self):
        from modules.modes.util import higher_level_actions as actions

        ball = SimpleNamespace(name="Poké Ball", price=200)
        emulator = _ShopEmulator()
        return_task_checks = 0

        def task_is_active(task_name):
            nonlocal return_task_checks
            if task_name == "Task_ReturnToShopMenu":
                return_task_checks += 1
                return return_task_checks == 1
            return task_name in {
                "Task_ShopMenu",
                "Task_BuyHowManyDialogueHandleInput",
                "Task_BuyMenu",
            }

        quantity_task = SimpleNamespace(data_value=lambda index: 1)
        fake_context = SimpleNamespace(
            emulator=emulator,
            rom=SimpleNamespace(is_rs=False),
        )
        with (
            patch.object(actions, "context", fake_context),
            patch.object(actions, "task_is_active", side_effect=task_is_active),
            patch.object(actions, "get_mart_buyable_items", return_value=[ball]),
            patch.object(actions, "get_mart_main_menu_scroll_position", return_value=0),
            patch.object(actions, "get_mart_buy_menu_scroll_position", return_value=0),
            patch.object(actions, "get_game_state_symbol", return_value="CB2_BUYMENU"),
            patch.object(actions, "get_task", return_value=quantity_task),
            patch.object(actions, "get_player", return_value=SimpleNamespace(money=200)),
        ):
            list(actions.buy_in_shop([(ball, 1)]))

        self.assertEqual(emulator.fresh_presses, ["A"])
        self.assertNotIn("A", emulator.presses)

    def test_multi_item_purchase_waits_for_confirmation_before_next_item(self):
        from modules.interaction_state import InteractionObservation
        from modules.memory import GameState
        from modules.modes.util import higher_level_actions as actions

        first_item = SimpleNamespace(name="Antidote", price=100)
        second_item = SimpleNamespace(name="Poké Ball", price=200)
        emulator = _ShopEmulator()
        return_task_checks = 0
        menu_position_checks = 0

        def task_is_active(task_name):
            nonlocal return_task_checks
            if task_name == "Task_ReturnToShopMenu":
                return_task_checks += 1
                return return_task_checks == 1
            return task_name in {
                "Task_ShopMenu",
                "Task_BuyHowManyDialogueHandleInput",
                "Task_BuyMenu",
            }

        def buy_menu_position():
            nonlocal menu_position_checks
            menu_position_checks += 1
            return 0 if menu_position_checks == 1 else 1

        fake_context = SimpleNamespace(
            emulator=emulator,
            rom=SimpleNamespace(is_rs=False),
        )
        with (
            patch.object(actions, "context", fake_context),
            patch.object(actions, "task_is_active", side_effect=task_is_active),
            patch.object(actions, "get_mart_buyable_items", return_value=[first_item, second_item]),
            patch.object(actions, "get_mart_main_menu_scroll_position", return_value=0),
            patch.object(actions, "get_mart_buy_menu_scroll_position", side_effect=buy_menu_position),
            patch.object(actions, "get_game_state_symbol", return_value="CB2_BUYMENU"),
            patch.object(actions, "get_task", return_value=SimpleNamespace(data_value=lambda index: 1)),
            patch.object(actions, "get_player", return_value=SimpleNamespace(money=300)),
            patch.object(
                actions,
                "observe_interaction",
                return_value=InteractionObservation(GameState.OVERWORLD),
            ),
            patch.object(actions, "_shop_interaction_is_settled", return_value=True) as settled,
        ):
            list(actions.buy_in_shop([(first_item, 1), (second_item, 1)]))

        settled.assert_called_once_with()

    def test_stalled_buy_callback_is_bounded_and_reports_runtime_state(self):
        from modules.modes.util import higher_level_actions as actions

        ball = SimpleNamespace(name="Poké Ball", price=200)
        emulator = _ShopEmulator()
        fake_context = SimpleNamespace(
            emulator=emulator,
            rom=SimpleNamespace(is_rs=False),
        )
        with (
            patch.object(actions, "context", fake_context),
            patch.object(actions, "task_is_active", return_value=True),
            patch.object(actions, "get_mart_buyable_items", return_value=[ball]),
            patch.object(actions, "get_mart_main_menu_scroll_position", return_value=0),
            patch.object(actions, "get_game_state_symbol", return_value="CB2_OTHER"),
            patch.object(actions, "get_player", return_value=SimpleNamespace(money=200)),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out.*CB2_BUYMENU"):
                list(actions.buy_in_shop([(ball, 1)]))

        self.assertEqual(emulator.fresh_presses, ["A"])


if __name__ == "__main__":
    unittest.main()
