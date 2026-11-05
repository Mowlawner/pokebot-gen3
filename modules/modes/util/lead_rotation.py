"""Event-boundary handoff for the campaign's overworld lead policy."""

from __future__ import annotations

from contextlib import contextmanager

from modules.context import context
from modules.console import diagnostic_print, print_campaign_field_lead
from modules.memory import GameState, get_game_state
from modules.menuing import MenuRotationFailure, MenuWrapper, RotatePokemon
from modules.player import player_avatar_is_standing_still
from modules.pokemon_party import get_party
from modules.tasks import get_global_script_context
from modules.battle_state import battle_is_active, get_last_battle_outcome, BattleOutcome
from modules.nuzlocke.field_lead import CampaignFieldLeadContext


_LEAD_HANDOFF_TIMEOUT_FRAMES = 30
_LEAD_MENU_TIMEOUT_FRAMES = 300


@contextmanager
def _campaign_trainer_lead_rotation_ownership(*, trainer_id: str, target_index: int, old_lead_index: int):
    """Claim the party menu for the duration of a trainer lead handoff.

    Opening the party menu removes the overworld projection for a few frames.
    Campaign readiness must treat that as an owned UI transaction rather than
    as permission to discard the suspended trainer exposure generator.
    """

    previous = getattr(context, "campaign_trainer_lead_rotation", None)
    marker = {
        "trainer_id": trainer_id,
        "target_index": target_index,
        "old_lead_index": old_lead_index,
    }
    context.campaign_trainer_lead_rotation = marker
    try:
        yield marker
    finally:
        # Preserve a nested owner if an embedding temporarily wraps this
        # transaction.  Ordinary campaign execution has only one owner.
        if getattr(context, "campaign_trainer_lead_rotation", None) is marker:
            context.campaign_trainer_lead_rotation = previous


class CampaignTrainerLeadFailure(RuntimeError):
    """A trainer lead handoff failed after claiming the party menu."""

    def __init__(self, trainer_id: str, failure: MenuRotationFailure):
        self.trainer_id = trainer_id
        self.phase = failure.phase
        self.reason = failure.reason
        super().__init__(
            f"trainer lead rotation failed for {trainer_id!r} during {failure.phase}: {failure.reason}"
        )


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
            party = get_party()
            party_slots = _party_slots(party)
            old_lead_index = party.first_non_fainted.index
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
                f"old_lead_pokemon={_party_slot_label(party_slots, old_lead_index)!r} "
                f"candidate_pokemon={_party_slot_label(party_slots, new_lead_index)!r} "
                f"policy_reason={getattr(decision, 'reason', None)!r} "
                f"envelope_source={getattr(field_context, 'source', None)!r}"
            ),
            trace=True,
        )
        print_campaign_field_lead(
            objective_id=getattr(field_context, "objective_id", None),
            current_index=old_lead_index,
            selected_index=new_lead_index,
            reason=getattr(decision, "reason", "field-lead policy returned no reason"),
            route_maps=getattr(field_context, "route_maps", ()),
            envelope_source=getattr(field_context, "source", None),
            party_slots=party_slots,
        )
        if new_lead_index is None or new_lead_index == old_lead_index:
            return

        rotation = None
        try:
            rotation = RotatePokemon(new_lead_index, old_lead_index)
            menu_controller = MenuWrapper(rotation).step()
            for menu_frame in range(_LEAD_MENU_TIMEOUT_FRAMES):
                try:
                    next(menu_controller)
                except StopIteration:
                    if rotation.failure is not None:
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                                f"rotated=False reason={reason!r} old_lead={old_lead_index!r} "
                                f"new_lead={new_lead_index!r} phase={rotation.failure.phase!r} "
                                f"failure={rotation.failure.reason!r}"
                            ),
                            trace=True,
                            prefix="CAMPAIGN_FIELD_LEAD_HANDOFF:",
                        )
                        return
                    if not rotation.completed:
                        return
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                            f"rotated=True reason={reason!r} old_lead={old_lead_index!r} "
                            f"new_lead={new_lead_index!r} "
                            f"old_lead_pokemon={_party_slot_label(party_slots, old_lead_index)!r} "
                            f"new_lead_pokemon={_party_slot_label(party_slots, new_lead_index)!r} "
                            f"menu_frames={menu_frame!r}"
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
                    f"new_lead={new_lead_index!r} "
                    f"old_lead_pokemon={_party_slot_label(party_slots, old_lead_index)!r} "
                    f"new_lead_pokemon={_party_slot_label(party_slots, new_lead_index)!r} "
                    f"error={type(error).__name__!r}"
                ),
                trace=True,
            )
            return

        diagnostic_print(
            lambda: (
                "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                f"rotated=False reason={reason!r} old_lead={old_lead_index!r} "
                f"new_lead={new_lead_index!r} menu_frames={_LEAD_MENU_TIMEOUT_FRAMES!r} "
                f"old_lead_pokemon={_party_slot_label(party_slots, old_lead_index)!r} "
                f"new_lead_pokemon={_party_slot_label(party_slots, new_lead_index)!r} "
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


def ensure_campaign_trainer_lead(target_index: int, *, trainer_id: str, reason: str):
    """Rotate to the readiness-selected lead immediately before exposure.

    Trainer preparation chooses a concrete party slot.  Keep that choice out
    of the general field-lead policy and apply it at the final staging
    boundary, where a later objective handoff cannot replace it before the
    ``EngageTrainer`` route is mounted.
    """

    if not isinstance(target_index, int) or isinstance(target_index, bool) or target_index < 0:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_TRAINER_LEAD: "
                f"rotated=False trainer={trainer_id!r} target={target_index!r} "
                "reason='readiness-selected lead is unavailable'"
            ),
            trace=True,
            prefix="CAMPAIGN_TRAINER_LEAD",
        )
        return False

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
            party = get_party()
            old_lead_index = party.first_non_fainted.index
            target = party[target_index]
            target_alive = not target.is_egg and target.current_hp > 0
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            ready = False
            old_lead_index = None
            target_alive = False

        if not ready or not target_alive:
            yield
            continue
        if old_lead_index == target_index:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_LEAD: "
                    f"rotated=False trainer={trainer_id!r} target={target_index!r} "
                    f"reason={reason!r} verified=True"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_LEAD",
            )
            return True

        rotation = None
        with _campaign_trainer_lead_rotation_ownership(
            trainer_id=trainer_id,
            target_index=target_index,
            old_lead_index=old_lead_index,
        ):
            try:
                rotation = RotatePokemon(target_index, old_lead_index)
                menu_controller = MenuWrapper(rotation).step()
                for menu_frame in range(_LEAD_MENU_TIMEOUT_FRAMES):
                    try:
                        next(menu_controller)
                    except StopIteration:
                        if rotation.failure is not None:
                            raise CampaignTrainerLeadFailure(trainer_id, rotation.failure)
                        if not rotation.completed:
                            raise CampaignTrainerLeadFailure(
                                trainer_id,
                                MenuRotationFailure("unknown", "rotation stopped without completion"),
                            )
                        break
                    yield
                else:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_LEAD: "
                            f"rotated=False trainer={trainer_id!r} old={old_lead_index!r} "
                            f"target={target_index!r} reason={reason!r} outcome='rotation_timeout'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_LEAD",
                    )
                    return False
            except CampaignTrainerLeadFailure:
                raise
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_LEAD: "
                        f"rotated=False trainer={trainer_id!r} old={old_lead_index!r} "
                        f"target={target_index!r} reason={reason!r} error={type(error).__name__!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_LEAD",
                )
                return False

            # Menu completion is not itself proof that the party cache has
            # observed the new order. Re-read it at the next stable boundary.
            for _ in range(3):
                yield
                try:
                    if get_party().first_non_fainted.index == target_index:
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_TRAINER_LEAD: "
                                f"rotated=True trainer={trainer_id!r} old={old_lead_index!r} "
                                f"target={target_index!r} reason={reason!r} verified=True"
                            ),
                            trace=True,
                            prefix="CAMPAIGN_TRAINER_LEAD",
                        )
                        return True
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
                    pass

            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_LEAD: "
                    f"rotated=False trainer={trainer_id!r} old={old_lead_index!r} "
                    f"target={target_index!r} reason={reason!r} verified=False"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_LEAD",
            )
            return False

    diagnostic_print(
        lambda: (
            "CAMPAIGN_TRAINER_LEAD: "
            f"rotated=False trainer={trainer_id!r} target={target_index!r} "
            f"reason={reason!r} ready=False timeout_frames={_LEAD_HANDOFF_TIMEOUT_FRAMES!r}"
        ),
        trace=True,
        prefix="CAMPAIGN_TRAINER_LEAD",
    )
    return False


def _party_slots(party: object) -> tuple[object, ...]:
    """Snapshot party slots without making diagnostic data required for policy."""

    try:
        return tuple(party)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        # Lightweight policy/test doubles may expose only ``first_non_fainted``.
        # They must not make lead selection fail merely because diagnostics are
        # unavailable; a real Party is iterable and takes the normal path.
        slots = getattr(party, "_pokemon", ())
        try:
            return tuple(slots)
        except (TypeError, AttributeError):
            return ()


def _party_slot_label(party_slots: tuple[object, ...], index: int | None) -> str:
    """Return a stable species/nickname/level label for a party slot."""

    if index is None:
        return "unchanged"
    if index < 0 or index >= len(party_slots):
        return "unknown"
    pokemon = party_slots[index]
    species = getattr(pokemon, "species_name_for_stats", None)
    if not isinstance(species, str):
        species_value = getattr(pokemon, "species", None)
        species = getattr(species_value, "name", None) or str(species_value or "unknown")
    nickname = getattr(pokemon, "nickname", None)
    identity = (
        f'{species} "{nickname}"'
        if nickname and str(nickname).casefold() != species.casefold()
        else species
    )
    level = getattr(pokemon, "level", None)
    return f"{identity} Lv{level}" if level is not None else identity
