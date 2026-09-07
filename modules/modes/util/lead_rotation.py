"""Event-boundary handoff for the campaign's overworld lead policy."""

from __future__ import annotations

from modules.context import context
from modules.console import diagnostic_print
from modules.memory import GameState, get_game_state
from modules.menuing import MenuWrapper, RotatePokemon
from modules.player import player_avatar_is_standing_still
from modules.pokemon_party import get_party
from modules.tasks import get_global_script_context
from modules.battle_state import battle_is_active, get_last_battle_outcome, BattleOutcome
from modules.nuzlocke.field_lead import CampaignFieldLeadContext


_LEAD_HANDOFF_TIMEOUT_FRAMES = 30
_LEAD_MENU_TIMEOUT_FRAMES = 300


def ensure_campaign_field_lead(strategy, field_context: CampaignFieldLeadContext, *, reason: str):
    """Apply the field lead policy once at a stable campaign boundary.

    This generator is intentionally bounded and event-driven. It never runs
    party-menu input on every overworld frame, and a stale battle/script
    boundary simply gives control back to the campaign loop after the short
    wait rather than entering manual mode or retrying forever.
    """

    for wait_frame in range(_LEAD_HANDOFF_TIMEOUT_FRAMES):
        try:
            script_context = get_global_script_context()
            ready = (
                get_game_state() is GameState.OVERWORLD
                and not battle_is_active()
                and get_last_battle_outcome() is not BattleOutcome.InProgress
                and not (script_context is not None and script_context.is_active)
                and player_avatar_is_standing_still()
                and context.bot_mode != "Manual"
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            ready = False
        if not ready:
            yield
            continue

        try:
            old_lead_index = get_party().first_non_fainted.index
            decision = strategy.choose_field_lead(field_context)
            new_lead_index = decision.selected_index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, NotImplementedError) as error:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                    f"rotated=False reason={reason!r} candidate_error={type(error).__name__!r}"
                ),
                trace=True,
            )
            return

        diagnostic_print(
            lambda: (
                "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                f"rotated=False reason={reason!r} old_lead={old_lead_index!r} "
                f"candidate={new_lead_index!r} wait_frame={wait_frame!r} "
                f"policy_reason={getattr(decision, 'reason', None)!r} "
                f"envelope_source={getattr(field_context, 'source', None)!r}"
            ),
            trace=True,
        )
        if new_lead_index is None or new_lead_index == old_lead_index:
            return

        try:
            menu_controller = MenuWrapper(RotatePokemon(new_lead_index, old_lead_index)).step()
            for menu_frame in range(_LEAD_MENU_TIMEOUT_FRAMES):
                try:
                    next(menu_controller)
                except StopIteration:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                            f"rotated=True reason={reason!r} old_lead={old_lead_index!r} "
                            f"new_lead={new_lead_index!r} menu_frames={menu_frame!r}"
                        ),
                        trace=True,
                    )
                    return
                yield
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                    f"rotated=False reason={reason!r} old_lead={old_lead_index!r} "
                    f"new_lead={new_lead_index!r} error={type(error).__name__!r}"
                ),
                trace=True,
            )
            return

        diagnostic_print(
            lambda: (
                "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                f"rotated=False reason={reason!r} old_lead={old_lead_index!r} "
                f"new_lead={new_lead_index!r} menu_frames={_LEAD_MENU_TIMEOUT_FRAMES!r} "
                "outcome='rotation_timeout'"
            ),
            trace=True,
        )
        return

    diagnostic_print(
        lambda: (
            "CAMPAIGN_FIELD_LEAD_HANDOFF: "
            f"rotated=False reason={reason!r} ready=False timeout_frames={_LEAD_HANDOFF_TIMEOUT_FRAMES!r}"
        ),
        trace=True,
    )
