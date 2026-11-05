import os
import shutil
from typing import Generator, Optional, TYPE_CHECKING

from modules.config import Config
from modules.runtime import get_base_path

if TYPE_CHECKING:
    from modules.gui import PokebotGui
    from modules.libmgba import LibmgbaEmulator
    from modules.modes import BotListener, BotMode
    from modules.profiles import Profile
    from modules.roms import ROM
    from modules.nuzlocke.runtime import NuzlockeRuntime
    from modules.stats import StatsDatabase


def _initialise_config() -> None:
    """
    This will check for every file in the `modules/config/templates/` folder
    whether a corresponding file exists in the `profiles/` directory.

    If it finds any that are missing, it will create it by copying over the
    file from the templates directory.

    The idea is that we can keep the template files checked into Git and thus
    overwriting them for each update, while the user can have their own real
    config files that will not be touched.
    """
    config_templates_path = get_base_path() / "modules" / "config" / "templates"
    profiles_path = get_base_path() / "profiles"
    if not profiles_path.exists():
        os.mkdir(profiles_path)
    for dist_file in config_templates_path.iterdir():
        if dist_file.is_file():
            regular_file = profiles_path / dist_file.name
            if not regular_file.exists():
                shutil.copyfile(dist_file, regular_file)


class BotContext:
    def __init__(self, initial_bot_mode: str = "Manual"):
        _initialise_config()
        self.config = Config()

        self.emulator: Optional["LibmgbaEmulator"] = None
        self.gui: Optional["PokebotGui"] = None
        self.nuzlocke_runtime: Optional["NuzlockeRuntime"] = None
        self.profile: Optional["Profile"] = None
        self.stats: Optional["StatsDatabase"] = None
        self.debug: bool = False
        # Normal debug logging is intended to be safe during emulation.  The
        # opening sequence can opt into high-volume diagnostic tracing when a
        # detailed ROM-state investigation is needed.
        self.debug_trace: bool = False
        # Optional performance probe; unlike normal debug tracing this is
        # intentionally off unless an operator explicitly enables it.
        self.debug_profile: bool = False
        # Opt-in, rolling frame trace for correlating visible emulator stalls.
        self.debug_stutter_trace: bool = False
        self.stutter_trace_threshold_ms: float = 50.0
        self.stutter_trace = None
        self.testing: bool = False
        self.battle_decision_source: str | None = None
        self.battle_decision_detail: str | None = None
        self.battle_evaluation_method: str | None = None
        # Temporary live callback used by objective-aware capture search. The
        # capture strategy clears these after each planner call so no battle
        # closure survives into ordinary defeat planning.
        self.battle_capture_success_probability = None
        self.battle_capture_probability_for_state = None
        self.battle_readiness = None
        self.campaign_trainer_checkpoint = None
        self.campaign_trainer_readiness = None
        # Campaign lead rotation owns the party menu while a trainer
        # exposure generator is suspended.  Readiness may temporarily lose
        # the overworld during that menu transaction, but must not replace
        # the generator that will close the menu and resume exposure.
        self.campaign_trainer_lead_rotation = None
        self.campaign_readiness_diagnostic = None
        # ROM-backed facts for all undefeated trainer hazards visible at the
        # current readiness boundary. The singular checkpoint above remains
        # the trainer that owns preparation and battle entry.
        self.campaign_trainer_facts = ()
        self.campaign_defeated_trainer_ids: set[tuple] = set()
        self.campaign_defeated_trainer_session_id = None
        self.campaign_active_trainer_checkpoint = None
        # Immutable AVOID-versus-IGNORE evidence from the encounter-stage
        # owner. Campaign readiness consumes this at the next boundary to
        # promote a known choke-point trainer or preserve an explicit unknown.
        self.campaign_encounter_stage = None
        self.campaign_capability_completion_latch: str | None = None
        self.campaign_preparation_exp_budget = None
        # Voluntary switch requests are matched against the next observed
        # PokemonSwitched event by the battle handler.  This is diagnostic
        # state used to reconcile one dispatched switch with the observed
        # active-slot transition before another planner switch is dispatched.
        self.battle_pending_switches: list[dict] = []
        self.battle_switch_transaction_sequence: int = 0
        self.battle_switch_reconciliation_status: str = "idle"
        self.battle_switch_replan_required: bool = False
        # A battle can reach its first action callback before the ROM-backed
        # battler and inventory snapshots are coherent. Campaign ownership
        # permits one fresh observation/replan at that boundary; repeated
        # empty roots remain a safety stop.
        self.battle_empty_root_replan_used: bool = False
        self.battle_planner_retry_requested: bool = False
        self.battle_planner_retry_count: int = 0
        self.campaign_preparation_target: str | None = None
        # Campaign-owned wild-battle intent consumed by the normalized
        # expectiminimax adapter.  ``None`` means no battle is mounted.
        self.campaign_wild_battle_intent: str | None = None
        # Routine EXP tagging is deliberately disabled.  The campaign
        # training scheduler should lead the intended trainee directly;
        # callers must opt in explicitly for an exceptional free-switch plan.
        self.campaign_allow_exp_tag: bool = False
        # Recovery and navigation handoffs are explicit session state. Keep
        # them initialized so a fresh context cannot inherit a stale marker
        # from a previous controller or recovery attempt.
        self.campaign_recovery_phase = None
        self.campaign_recovery_observed_triggers = None
        self.campaign_recovery_pc_target = None
        self.campaign_recovery_pc_status = None
        self.campaign_recovery_retry_count = 0
        self.campaign_continuation_route = None
        self.battle_authoritative_faints = []
        # A resource capability may defer an unaffordable restock after it has
        # verified that no fundable trainer is available on the current map.
        # Campaign selection clears/ignores this marker when the map changes.
        self.campaign_restock_deferral: tuple[str, tuple[int, int]] | None = None
        # The currently suspended AgentControlLoop, if any. Campaign
        # capabilities often yield through a navigation loop, so the campaign
        # controller cannot inspect that loop directly from its outer
        # generator. AgentControlLoop owns this marker only while its run
        # generator is active and restores the previous value on exit.
        self.active_agent_control_loop = None
        # Optional source-gate evidence recorder.  It is armed externally for
        # a reviewed candidate and remains inert during ordinary navigation.
        self.gate_live_observer = None

        self._current_message: str = ""
        self._campaign_status = None

        self.controller_stack: list[Generator] = []
        self.debug_action_stack: list[str] = []
        self.bot_mode_instance: Optional["BotMode"] = None
        self.bot_listeners: list["BotListener"] = []
        self.frame: int = 0
        self._current_bot_mode: str = initial_bot_mode
        self._previous_bot_mode: str = "Manual"

    def reload_config(self) -> None:
        """
        Triggers a config reload, reload the global config then specific profile config.
        """
        from modules.console import console
        from modules.plugins import plugin_config_reloaded

        try:
            new_config = Config()
            new_config.load(self.config.config_dir, strict=False)
            self.config = new_config
            plugin_config_reloaded()
            console.print("[cyan]Profile settings loaded.[/]")
        except Exception as error:
            if self.debug:
                raise error
            console.print(
                "[bold red]The configuration could not be loaded, no changes have been made.[/]\n"
                "[bold yellow]This is probably due to a malformed file."
                "For more information run the bot with the --debug flag.[/]"
            )
        return

    @property
    def message(self) -> str:
        return self._current_message

    @property
    def campaign_status(self):
        return self._campaign_status

    @campaign_status.setter
    def campaign_status(self, status) -> None:
        if self._campaign_status == status:
            return
        self._campaign_status = status
        self._update_gui()

    @message.setter
    def message(self, new_message: str) -> None:
        self._current_message = new_message
        self._update_gui()

    @property
    def emulation_speed(self) -> float:
        return self.emulator.get_speed_factor() if self.emulator.get_throttle() else 0

    @emulation_speed.setter
    def emulation_speed(self, new_speed: float) -> None:
        if self.emulator:
            if new_speed == 0:
                self.emulator.set_throttle(False)
            else:
                self.emulator.set_throttle(True)
                self.emulator.set_speed_factor(new_speed)
            self._update_gui()

    @property
    def bot_mode(self) -> str:
        return self._current_bot_mode

    @bot_mode.setter
    def bot_mode(self, new_bot_mode: str) -> None:
        if self._current_bot_mode != new_bot_mode:
            self._previous_bot_mode = self._current_bot_mode
            self._current_bot_mode = new_bot_mode
            self._update_gui()

    def toggle_manual_mode(self) -> None:
        if self._current_bot_mode == "Manual":
            self._current_bot_mode = self._previous_bot_mode
            self._previous_bot_mode = "Manual"
        else:
            self._previous_bot_mode = self._current_bot_mode
            self.set_manual_mode(enable_video_and_slow_down=False)
        self._update_gui()

    def set_manual_mode(self, enable_video_and_slow_down: bool = True) -> None:
        if self.bot_mode == "Manual":
            return

        self.bot_mode = "Manual"
        self.emulator.reset_held_buttons()
        if enable_video_and_slow_down:
            from modules.gui.desktop_notification import desktop_notification

            self.emulation_speed = 1
            self.video = True
            desktop_notification(title="Manual Mode", message="The bot has switched to manual mode.")

    def debug_stepping_mode(self) -> None:
        if self.debug and self.gui and self.gui._emulator_screen:
            from modules.gui.desktop_notification import desktop_notification

            self.gui._emulator_screen.toggle_stepping_mode()
            self.video = True
            desktop_notification(title="Manual Mode", message="The bot has switched to stepping mode.")

    @property
    def audio(self) -> bool:
        return self.emulator.get_audio_enabled() if self.emulator else False

    @audio.setter
    def audio(self, audio_on: bool) -> None:
        if self.emulator:
            self.emulator.set_audio_enabled(audio_on)
            self._update_gui()

    def toggle_audio(self) -> None:
        if self.emulator:
            self.emulator.set_audio_enabled(not self.emulator.get_audio_enabled())
            self._update_gui()

    @property
    def video(self) -> bool:
        return self.emulator.get_video_enabled() if self.emulator else False

    @video.setter
    def video(self, video_on: bool) -> None:
        if self.emulator:
            self.emulator.set_video_enabled(video_on)
            self._update_gui()

    def toggle_video(self):
        if self.emulator:
            self.emulator.set_video_enabled(not self.emulator.get_video_enabled())
            self._update_gui()

    @property
    def rom(self) -> Optional["ROM"]:
        return self.profile.rom if self.profile else None

    def _update_gui(self) -> None:
        if self.gui:
            self.gui.on_settings_updated()


context = BotContext()
