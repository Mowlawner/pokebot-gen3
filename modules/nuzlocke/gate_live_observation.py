"""Capture explicit, controller-owned evidence for source-derived gate edges.

The recorder is deliberately armed by a caller that has already selected a
source candidate, edge, and guard state.  Generic navigation reports only the
action boundary and its terminal outcome; it never guesses which candidate a
route action represents.  An observation is emitted only after a real input
was issued and the controller observed either a blocked movement or an open
destination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from .gate_impact import EdgeKey
from .gate_runtime_evidence import GateEdgeObservation


LIVE_OBSERVATION_SCHEMA_VERSION = 2
_VALID_OUTCOMES = frozenset({"blocked", "open"})
_VALID_GUARD_STATES = frozenset({"satisfied", "unsatisfied", "unknown", "unavailable"})
_VALID_GUARD_OPERATORS = frozenset({"equals", "not_equals", "greater_than", "greater_or_equal"})


@dataclass(frozen=True, slots=True)
class GateGuardPredicate:
    """A source-derived predicate evaluated against the live save variable."""

    symbol: str
    operator: str
    value: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("guard predicate requires a symbol")
        if self.operator not in _VALID_GUARD_OPERATORS:
            raise ValueError(f"invalid guard operator: {self.operator!r}")

    @property
    def expression(self) -> str:
        operator = {
            "equals": "==",
            "not_equals": "!=",
            "greater_than": ">",
            "greater_or_equal": ">=",
        }[self.operator]
        return f"{self.symbol} {operator} {self.value}"

    def evaluate(self, observed_value: int) -> bool:
        if self.operator == "equals":
            return observed_value == self.value
        if self.operator == "not_equals":
            return observed_value != self.value
        if self.operator == "greater_than":
            return observed_value > self.value
        return observed_value >= self.value


@dataclass(frozen=True, slots=True)
class GateLiveBinding:
    """One externally reviewed candidate/edge association."""

    candidate_id: str
    edge: EdgeKey
    guard: GateGuardPredicate
    source_kind: str | None = None


@dataclass(frozen=True, slots=True)
class GateLiveFixture:
    """Optional disposable emulator setup for a source-gate probe."""

    source_map: tuple[int, int] | None = None
    event_vars: tuple[tuple[str, int], ...] = ()
    player_coordinates: tuple[int, int] | None = None
    object_positions: tuple[tuple[int, tuple[int, int]], ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingNavigation:
    binding: GateLiveBinding
    source_kind: str
    action_kind: str
    action_frame: int | None
    guard_state: str
    guard_value: int
    dynamic_object_positions: tuple[tuple[int, object, object], ...] = ()


def edge_for_navigation(navigation) -> EdgeKey | None:
    """Return a stable edge identity from a navigation action-like object."""

    source = getattr(navigation, "source", None)
    destination = getattr(navigation, "destination", None)
    if source is None or destination is None:
        return None
    try:
        source_map = source[0]
        destination_map = destination[0]
    except (IndexError, TypeError):
        return None
    action_name = getattr(getattr(navigation, "action_type", None), "name", "")
    if action_name == "MOVE":
        kind = "walk"
    else:
        kind = getattr(navigation, "transition_kind", None) or "warp"
    return EdgeKey(_map_identity(source_map), _map_identity(destination_map), kind, None)


def _map_identity(map_id):
    """Normalize enum-backed live map IDs to the graph's tuple identity."""

    value = getattr(map_id, "value", map_id)
    if isinstance(value, list):
        return tuple(_map_identity(item) for item in value)
    return value


def _canonical(value):
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _canonical(value.value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    return value


def _observation_id(
    *,
    session_id: str,
    candidate_id: str,
    edge: EdgeKey,
    source_kind: str,
    outcome: str,
    action_frame: int | None,
    observation_frame: int | None,
) -> str:
    payload = _canonical(
        (
            session_id,
            candidate_id,
            edge,
            source_kind,
            outcome,
            action_frame,
            observation_frame,
        )
    )
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"live:{digest[:20]}"


def _edge_sort_key(edge: EdgeKey) -> str:
    return json.dumps(_canonical(edge), sort_keys=True, separators=(",", ":"))


def _read_event_var(symbol: str) -> int:
    """Read one ROM-backed event variable from the active save."""

    from modules.memory import get_event_var

    normalized = symbol.removeprefix("VAR_")
    value = get_event_var(normalized)
    if value < 0:
        raise ValueError(f"unknown event variable: {symbol!r}")
    return value


def _dynamic_object_positions(objects) -> tuple[tuple[int, object, object], ...]:
    """Normalize live object identity and position into stable evidence."""

    def stable_identity(value):
        canonical = _canonical(value)
        if isinstance(canonical, list):
            return tuple(stable_identity(item) for item in canonical)
        if isinstance(canonical, dict):
            return {key: stable_identity(item) for key, item in canonical.items()}
        return canonical

    positions = []
    for item in objects or ():
        try:
            if isinstance(item, tuple) and len(item) == 3:
                local_id, map_id, coordinates = item
                positions.append((int(local_id), stable_identity(map_id), stable_identity(coordinates)))
                continue
            if isinstance(item, dict):
                local_id = item["local_id"]
                location = item["location"]
            else:
                local_id = getattr(item, "local_id")
                location = getattr(item, "location")
            map_id, coordinates = location
            positions.append((int(local_id), stable_identity(map_id), stable_identity(coordinates)))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    unique = {repr(item): item for item in positions}
    return tuple(sorted(unique.values(), key=repr))


class GateLiveObservationRecorder:
    """Collect explicit gate attempts for one emulator/session provenance."""

    def __init__(
        self,
        session_id: str,
        *,
        output: Path | None = None,
        guard_reader=None,
    ):
        if not session_id:
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.output = output
        self.fixture: GateLiveFixture | None = None
        self._guard_reader = guard_reader or _read_event_var
        self._bindings: dict[tuple[str, EdgeKey, str | None], GateLiveBinding] = {}
        self._pending: dict[tuple[str, EdgeKey, str], _PendingNavigation] = {}
        self._observations: dict[str, GateEdgeObservation] = {}
        self._diagnostics: set[str] = set()

    def arm(
        self,
        candidate_id: str,
        edge: EdgeKey,
        *,
        guard: GateGuardPredicate,
        source_kind: str | None = None,
    ) -> GateLiveBinding:
        """Arm one reviewed candidate/edge association for live capture."""

        if not candidate_id:
            raise ValueError("candidate_id is required")
        binding = GateLiveBinding(candidate_id, edge, guard, source_kind)
        self._bindings[(candidate_id, edge, source_kind)] = binding
        return binding

    @property
    def observations(self) -> tuple[GateEdgeObservation, ...]:
        return tuple(
            sorted(
                self._observations.values(),
                key=lambda item: (item.candidate_id, _edge_sort_key(item.edge), item.source_kind, item.observation_id),
            )
        )

    @property
    def bindings(self) -> tuple[GateLiveBinding, ...]:
        """Return configured candidate/edge associations in stable order."""

        return tuple(
            sorted(
                self._bindings.values(),
                key=lambda item: (
                    item.candidate_id,
                    _edge_sort_key(item.edge),
                    item.source_kind or "",
                ),
            )
        )

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(sorted(self._diagnostics))

    def add_diagnostic(self, message: str) -> None:
        self._diagnostics.add(str(message))

    def _bindings_for(self, edge: EdgeKey, source_kind: str) -> tuple[GateLiveBinding, ...]:
        return tuple(
            binding
            for binding in self._bindings.values()
            if binding.edge == edge and (binding.source_kind is None or binding.source_kind == source_kind)
        )

    def record_navigation_issued(
        self,
        navigation,
        *,
        action_frame: int | None = None,
        source_kind: str | None = None,
        dynamic_objects=(),
    ) -> int:
        """Remember a navigation input without classifying its outcome."""

        edge = edge_for_navigation(navigation)
        if edge is None:
            return 0
        action_name = getattr(getattr(navigation, "action_type", None), "name", "")
        actual_source_kind = source_kind or ("movement" if action_name == "MOVE" else "transition")
        bindings = self._bindings_for(edge, actual_source_kind)
        if not bindings and actual_source_kind == "transition" and self._bindings:
            self._diagnostics.add(
                f"ignored transition action for {_edge_sort_key(edge)}: no matching armed edge"
            )
        recorded = 0
        object_positions = _dynamic_object_positions(dynamic_objects)
        for binding in bindings:
            try:
                guard_value = self._guard_reader(binding.guard.symbol)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._diagnostics.add(
                    f"ignored {binding.candidate_id!r}: could not read {binding.guard.symbol!r}: {exc}"
                )
                continue
            guard_state = "satisfied" if binding.guard.evaluate(guard_value) else "unsatisfied"
            key = (binding.candidate_id, edge, actual_source_kind)
            if key not in self._pending:
                self._pending[key] = _PendingNavigation(
                    binding=binding,
                    source_kind=actual_source_kind,
                    action_kind=action_name.lower() or "navigation",
                    action_frame=action_frame,
                    guard_state=guard_state,
                    guard_value=guard_value,
                    dynamic_object_positions=object_positions,
                )
                recorded += 1
        return recorded

    def record_navigation_terminal_outcome(
        self,
        navigation,
        outcome: str,
        *,
        action_frame: int | None = None,
        observation_frame: int | None = None,
        source_kind: str | None = None,
        dynamic_objects=(),
    ) -> int:
        """Record a terminal result after a controller disposed its pending action.

        ``AgentControlLoop`` deliberately discards pending navigation when it
        raises a terminal error.  A disposable gate probe still has an
        explicit issued input and a terminal blocked boundary, so it needs a
        narrow recovery path that re-reads the reviewed binding rather than
        accepting an unissued synthetic result.
        """

        edge = edge_for_navigation(navigation)
        if edge is None or outcome not in _VALID_OUTCOMES:
            return 0
        action_name = getattr(getattr(navigation, "action_type", None), "name", "")
        actual_source_kind = source_kind or ("movement" if action_name == "MOVE" else "transition")
        bindings = self._bindings_for(edge, actual_source_kind)
        if not bindings:
            return 0
        object_positions = _dynamic_object_positions(dynamic_objects)
        pending_keys = tuple(
            key
            for key in self._pending
            if key[1] == edge and key[2] == actual_source_kind
        )
        for key in pending_keys:
            self._pending.pop(key, None)
        recorded = 0
        for binding in bindings:
            try:
                guard_value = self._guard_reader(binding.guard.symbol)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._diagnostics.add(
                    f"ignored {binding.candidate_id!r}: could not read {binding.guard.symbol!r}: {exc}"
                )
                continue
            recorded += self.record_attempt(
                candidate_id=binding.candidate_id,
                edge=edge,
                source_kind=actual_source_kind,
                action_kind=action_name.lower() or "navigation",
                guard_state="satisfied" if binding.guard.evaluate(guard_value) else "unsatisfied",
                action_issued=True,
                outcome=outcome,
                action_frame=action_frame,
                observation_frame=observation_frame,
                guard_symbol=binding.guard.symbol,
                guard_value=guard_value,
                guard_expression=binding.guard.expression,
                dynamic_object_positions=object_positions,
            )
        return recorded

    def record_navigation_outcome(
        self,
        navigation,
        outcome: str,
        *,
        observation_frame: int | None = None,
        source_kind: str | None = None,
    ) -> int:
        """Classify a previously issued navigation input at an explicit boundary."""

        edge = edge_for_navigation(navigation)
        if edge is None:
            self._diagnostics.add("ignored navigation outcome without a concrete edge")
            return 0
        action_name = getattr(getattr(navigation, "action_type", None), "name", "")
        actual_source_kind = source_kind or ("movement" if action_name == "MOVE" else "transition")
        keys = tuple(
            key
            for key in self._pending
            if key[1] == edge and key[2] == actual_source_kind
        )
        if not keys:
            if not self._bindings_for(edge, actual_source_kind):
                return 0
            self._diagnostics.add(
                f"ignored {actual_source_kind} outcome for {_edge_sort_key(edge)}: no issued action"
            )
            return 0
        if outcome not in _VALID_OUTCOMES:
            self._diagnostics.add(f"ignored navigation outcome {outcome!r}: expected 'blocked' or 'open'")
            return 0
        recorded = 0
        for key in keys:
            pending = self._pending.pop(key)
            recorded += self.record_attempt(
                candidate_id=pending.binding.candidate_id,
                edge=edge,
                source_kind=pending.source_kind,
                action_kind=pending.action_kind,
                guard_state=pending.guard_state,
                action_issued=True,
                outcome=outcome,
                action_frame=pending.action_frame,
                observation_frame=observation_frame,
                guard_symbol=pending.binding.guard.symbol,
                guard_value=pending.guard_value,
                guard_expression=pending.binding.guard.expression,
                dynamic_object_positions=pending.dynamic_object_positions,
            )
        return recorded

    def discard_navigation(
        self,
        navigation,
        *,
        source_kind: str | None = None,
        reason: str = "outcome not observed",
    ) -> int:
        """Forget an attempt when the controller cannot classify its result."""

        edge = edge_for_navigation(navigation)
        if edge is None:
            return 0
        action_name = getattr(getattr(navigation, "action_type", None), "name", "")
        actual_source_kind = source_kind or ("movement" if action_name == "MOVE" else "transition")
        keys = tuple(key for key in self._pending if key[1] == edge and key[2] == actual_source_kind)
        for key in keys:
            self._pending.pop(key, None)
        if keys:
            self._diagnostics.add(
                f"discarded {actual_source_kind} attempt for {_edge_sort_key(edge)}: {reason}"
            )
        return len(keys)

    def record_attempt(
        self,
        *,
        candidate_id: str,
        edge: EdgeKey,
        source_kind: str,
        action_kind: str,
        guard_state: str,
        action_issued: bool,
        outcome: str | None,
        action_frame: int | None = None,
        observation_frame: int | None = None,
        confidence: str = "high",
        guard_symbol: str | None = None,
        guard_value: int | None = None,
        guard_expression: str | None = None,
        dynamic_object_positions=(),
    ) -> int:
        """Record one explicit attempt, refusing incomplete evidence."""

        if not candidate_id:
            self._diagnostics.add("rejected attempt without candidate_id")
            return 0
        if not action_issued:
            self._diagnostics.add(f"rejected {candidate_id!r}: action was not issued")
            return 0
        if outcome not in _VALID_OUTCOMES:
            self._diagnostics.add(f"rejected {candidate_id!r}: terminal outcome is missing or unknown")
            return 0
        if guard_state not in _VALID_GUARD_STATES:
            self._diagnostics.add(f"rejected {candidate_id!r}: invalid guard state {guard_state!r}")
            return 0
        if guard_state in {"unknown", "unavailable"}:
            self._diagnostics.add(f"rejected {candidate_id!r}: guard state is {guard_state!r}")
            return 0
        if guard_state == "satisfied" and outcome != "open":
            self._diagnostics.add(
                f"rejected {candidate_id!r}: a satisfied guard cannot produce blocked edge evidence"
            )
            return 0
        if confidence != "high":
            self._diagnostics.add(f"rejected {candidate_id!r}: explicit live evidence must be high confidence")
            return 0
        blocked = outcome == "blocked"
        observation_id = _observation_id(
            session_id=self.session_id,
            candidate_id=candidate_id,
            edge=edge,
            source_kind=source_kind,
            outcome=outcome,
            action_frame=action_frame,
            observation_frame=observation_frame,
        )
        self._observations[observation_id] = GateEdgeObservation(
            candidate_id=candidate_id,
            edge=edge,
            source_kind=source_kind,
            observation_id=observation_id,
            blocked=blocked,
            confidence=confidence,
            guard_state=guard_state,
            action_kind=action_kind,
            action_frame=action_frame,
            observation_frame=observation_frame,
            session_id=self.session_id,
            guard_symbol=guard_symbol,
            guard_value=guard_value,
            guard_expression=guard_expression,
            dynamic_object_positions=_dynamic_object_positions(dynamic_object_positions),
        )
        self.flush()
        return 1

    def to_dict(self) -> dict[str, object]:
        def observation_dict(observation: GateEdgeObservation) -> dict[str, object]:
            return {
                "candidate_id": observation.candidate_id,
                "edge": _canonical(observation.edge),
                "source_kind": observation.source_kind,
                "observation_id": observation.observation_id,
                "blocked": observation.blocked,
                "confidence": observation.confidence,
                "guard_state": observation.guard_state,
                "action_kind": observation.action_kind,
                "action_frame": observation.action_frame,
                "observation_frame": observation.observation_frame,
                "session_id": observation.session_id,
                "guard_symbol": observation.guard_symbol,
                "guard_value": observation.guard_value,
                "guard_expression": observation.guard_expression,
                "dynamic_object_positions": [
                    {
                        "local_id": local_id,
                        "map_id": map_id,
                        "coordinates": coordinates,
                    }
                    for local_id, map_id, coordinates in observation.dynamic_object_positions
                ],
            }

        pending = [
            {
                "candidate_id": item.binding.candidate_id,
                "edge": _canonical(item.binding.edge),
                "source_kind": item.source_kind,
                "action_kind": item.action_kind,
                "action_frame": item.action_frame,
                "guard_state": item.guard_state,
                "guard_value": item.guard_value,
                "dynamic_object_positions": [
                    {
                        "local_id": local_id,
                        "map_id": map_id,
                        "coordinates": coordinates,
                    }
                    for local_id, map_id, coordinates in item.dynamic_object_positions
                ],
            }
            for item in sorted(
                self._pending.values(),
                key=lambda item: (
                    item.binding.candidate_id,
                    _edge_sort_key(item.binding.edge),
                    item.source_kind,
                ),
            )
        ]
        payload = {
            "schema_version": LIVE_OBSERVATION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "bindings": [
                {
                    "candidate_id": item.candidate_id,
                    "edge": _canonical(item.edge),
                    "guard": {
                        "symbol": item.guard.symbol,
                        "operator": item.guard.operator,
                        "value": item.guard.value,
                    },
                    "source_kind": item.source_kind,
                }
                for item in self.bindings
            ],
            "observations": [observation_dict(item) for item in self.observations],
            "pending": pending,
            "diagnostics": list(self.diagnostics),
        }
        if self.fixture is not None:
            payload["fixture"] = {
                "source_map": list(self.fixture.source_map) if self.fixture.source_map is not None else None,
                "event_vars": {symbol: value for symbol, value in self.fixture.event_vars},
                "player_coordinates": (
                    list(self.fixture.player_coordinates) if self.fixture.player_coordinates is not None else None
                ),
                "object_positions": {
                    str(local_id): list(coordinates)
                    for local_id, coordinates in self.fixture.object_positions
                },
            }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def flush(self) -> None:
        if self.output is None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(self.to_json(), encoding="utf-8")


def _restore_identity(value):
    if isinstance(value, list):
        return tuple(_restore_identity(item) for item in value)
    if isinstance(value, dict):
        return {key: _restore_identity(item) for key, item in value.items()}
    return value


def load_gate_live_observer(path: str | Path) -> GateLiveObservationRecorder:
    """Load an armable recorder configuration used by a live bot process."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("gate live configuration requires a non-empty session_id")
    output_value = payload.get("output")
    output = None
    if output_value:
        output = Path(output_value)
        if not output.is_absolute():
            output = config_path.parent / output
    recorder = GateLiveObservationRecorder(session_id, output=output)
    fixture_payload = payload.get("fixture")
    if fixture_payload is not None:
        if not isinstance(fixture_payload, dict):
            raise ValueError("gate live fixture must be an object")

        source_map_payload = fixture_payload.get("source_map")
        source_map = None
        if source_map_payload is not None:
            if not isinstance(source_map_payload, (list, tuple)) or len(source_map_payload) != 2:
                raise ValueError("gate live fixture source_map must contain two integers")
            source_map = (int(source_map_payload[0]), int(source_map_payload[1]))

        event_vars_payload = fixture_payload.get("event_vars", {})
        if not isinstance(event_vars_payload, dict):
            raise ValueError("gate live fixture event_vars must be an object")
        event_vars = tuple(sorted((str(symbol), int(value)) for symbol, value in event_vars_payload.items()))

        player_payload = fixture_payload.get("player_coordinates")
        player_coordinates = None
        if player_payload is not None:
            if not isinstance(player_payload, (list, tuple)) or len(player_payload) != 2:
                raise ValueError("gate live fixture player_coordinates must contain two integers")
            player_coordinates = (int(player_payload[0]), int(player_payload[1]))

        object_payload = fixture_payload.get("object_positions", {})
        if not isinstance(object_payload, dict):
            raise ValueError("gate live fixture object_positions must be an object")
        object_positions = []
        for local_id, coordinates_payload in object_payload.items():
            if not isinstance(coordinates_payload, (list, tuple)) or len(coordinates_payload) != 2:
                raise ValueError("gate live fixture object position must contain two integers")
            object_positions.append((int(local_id), (int(coordinates_payload[0]), int(coordinates_payload[1]))))
        recorder.fixture = GateLiveFixture(
            source_map=source_map,
            event_vars=event_vars,
            player_coordinates=player_coordinates,
            object_positions=tuple(sorted(object_positions)),
        )
    for item in payload.get("bindings", ()):
        edge_payload = item.get("edge", {})
        edge = EdgeKey(
            _restore_identity(edge_payload.get("source_map")),
            _restore_identity(edge_payload.get("destination_map")),
            edge_payload.get("kind", "walk"),
            edge_payload.get("warp_index"),
        )
        guard_payload = item.get("guard")
        if not isinstance(guard_payload, dict):
            raise ValueError(
                f"gate live binding {item.get('candidate_id', '<unknown>')!r} requires a ROM-backed guard"
            )
        guard = GateGuardPredicate(
            symbol=str(guard_payload["symbol"]),
            operator=str(guard_payload["operator"]),
            value=int(guard_payload["value"]),
        )
        recorder.arm(
            item["candidate_id"],
            edge,
            guard=guard,
            source_kind=item.get("source_kind"),
        )
    return recorder


__all__ = [
    "GateGuardPredicate",
    "GateLiveFixture",
    "GateLiveBinding",
    "GateLiveObservationRecorder",
    "LIVE_OBSERVATION_SCHEMA_VERSION",
    "edge_for_navigation",
    "load_gate_live_observer",
]
