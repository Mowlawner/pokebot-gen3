"""Conservative synchronization boundary for battle action telemetry.

The Emerald memory reader currently has no validated, unique execution edge.
This module therefore accepts only explicit action markers supplied by a ROM
specific reader (or a test fixture).  It intentionally does not interpret
PP, HP, callbacks, script pointers, animations, or ``gCurrentMove``.

This keeps the boundary useful today: observations can carry an authoritative
marker when one is eventually validated, while all other action telemetry is
explicitly unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.battle_observation import BattleObservation, MoveExecuted, MoveFailed


@dataclass(frozen=True, slots=True)
class AuthoritativeAction:
    """A ROM/test supplied action marker.

    ``token`` must identify one game action, not merely one frame.  A token is
    required because the same move can execute on later turns and a frame
    number is not guaranteed to be present in synthetic observations.
    """

    token: object
    battler: str
    move: str
    failed: bool = False


class BattleActionSynchronizer:
    """Turn explicit action markers into at-most-once battle events."""

    def __init__(self) -> None:
        self._seen: set[object] = set()

    def observe(
        self,
        observation: BattleObservation,
        action: AuthoritativeAction | None = None,
    ) -> tuple[MoveExecuted | MoveFailed, ...]:
        """Return telemetry for ``action`` if it is new.

        ``action=None`` is the normal result for the current Emerald reader:
        no move execution event is emitted.  Replaying an observation or
        marker with the same token is harmless.
        """

        if action is None or action.token in self._seen:
            return ()
        self._seen.add(action.token)
        if action.failed:
            return (MoveFailed(observation.frame, action.battler, action.move),)
        return (MoveExecuted(observation.frame, action.battler, action.move),)

    def reset(self) -> None:
        self._seen.clear()


def automatic_action_marker_available() -> bool:
    """Whether this build currently supplies a validated execution marker."""

    return False
