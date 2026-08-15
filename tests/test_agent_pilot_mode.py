import sys
from types import GeneratorType, ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class AgentRivalBattlePilotModeTests(TestCase):
    def test_mode_has_no_argument_constructor_and_configures_generic_loop(self):
        from modules.agent_control import AgentControlLoop
        from modules.goals import ActivateTrigger, INTRODUCTORY_RIVAL_TRIGGER_ID
        from modules.modes.agent_pilot import AgentRivalBattlePilot

        with patch("modules.modes.agent_pilot.context", SimpleNamespace(rom=SimpleNamespace(is_emerald=True))):
            pilot = AgentRivalBattlePilot()

        self.assertIsInstance(pilot.control_loop, AgentControlLoop)
        self.assertEqual(pilot.goal, ActivateTrigger(INTRODUCTORY_RIVAL_TRIGGER_ID))
        self.assertEqual(pilot.name(), "Agent Rival Battle Pilot")

    def test_run_delegates_to_existing_loop_generator(self):
        from modules.agent_control import AgentControlLoop
        from modules.modes.agent_pilot import AgentRivalBattlePilot

        with patch("modules.modes.agent_pilot.context", SimpleNamespace(rom=SimpleNamespace(is_emerald=True))):
            pilot = AgentRivalBattlePilot()
        pilot.control_loop = AgentControlLoop(lambda: None)

        generator = pilot.run()
        self.assertIsInstance(generator, GeneratorType)
        generator.close()

    def test_mode_is_registered_and_resolves_through_existing_factory(self):
        import modules.modes as modes
        from modules.modes.agent_pilot import AgentRivalBattlePilot

        # The registry imports every built-in mode lazily.  Stub unrelated modes
        # here so this focused registration test does not load the optional
        # native mGBA GUI dependency.
        mode_modules = {
            "berry_blend": "BerryBlendMode",
            "bunny_hop": "BunnyHopMode",
            "daycare": "DaycareMode",
            "ev_train": "EVTrainMode",
            "feebas": "FeebasMode",
            "fishing": "FishingMode",
            "game_corner": "GameCornerMode",
            "item_steal": "ItemStealMode",
            "kecleon": "KecleonMode",
            "level_grind": "LevelGrindMode",
            "nugget_bridge": "NuggetBridgeMode",
            "opening": "EmeraldOpeningMode",
            "puzzle_solver": "PuzzleSolverMode",
            "roamer_reencounter": "RoamerReencounterMode",
            "roamer_reset": "RoamerResetMode",
            "rock_smash": "RockSmashMode",
            "safari": "SafariMode",
            "spin": "SpinMode",
            "starters": "StartersMode",
            "sudowoodo": "SudowoodoMode",
            "static_run_away": "StaticRunAway",
            "static_gift_resets": "StaticGiftResetsMode",
            "static_soft_resets": "StaticSoftResetsMode",
            "sweet_scent": "SweetScentMode",
        }
        fake_modules = {}
        for module_name, class_name in mode_modules.items():
            module = ModuleType(f"modules.modes.{module_name}")
            mode_type = type(
                class_name,
                (),
                {"name": staticmethod(lambda mode_name=class_name: mode_name)},
            )
            setattr(module, class_name, mode_type)
            fake_modules[f"modules.modes.{module_name}"] = module

        old_modes = modes._bot_modes
        modes._bot_modes = []
        with patch.dict(sys.modules, fake_modules):
            try:
                names = modes.get_bot_mode_names()
                self.assertIn("Agent Rival Battle Pilot", names)
                mode_type = modes.get_bot_mode_by_name("Agent Rival Battle Pilot")
                self.assertIs(mode_type, AgentRivalBattlePilot)
                with patch("modules.modes.agent_pilot.context", SimpleNamespace(rom=SimpleNamespace(is_emerald=True))):
                    self.assertIsInstance(mode_type(), AgentRivalBattlePilot)
            finally:
                modes._bot_modes = old_modes
