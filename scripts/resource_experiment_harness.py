"""Single-run lifecycle harness for the opening resource experiment.

This script deliberately owns only experiment process/profile lifecycle. It
does not alter bot decisions or campaign state semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


class HarnessError(RuntimeError):
    pass


def read_events(path: Path, offset: int) -> tuple[list[dict], int]:
    if not path.exists():
        return [], offset
    with path.open(encoding="utf-8") as stream:
        stream.seek(offset)
        records = []
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records, stream.tell()


def process_group_gone(process: subprocess.Popen) -> bool:
    if process.poll() is not None:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    return False


def stop_process_group(process: subprocess.Popen, grace: float = 5.0) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process_group_gone(process):
            return
        time.sleep(0.1)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process_group_gone(process):
            return
        time.sleep(0.1)
    raise HarnessError(f"process group {process.pid} did not terminate")


def prepare_profile(source: Path, target: Path) -> None:
    if target.exists():
        raise HarnessError(f"disposable profile already exists: {target}")
    shutil.copytree(source, target)
    for name in ("nuzlocke_events.json", "nuzlocke_events.json.append", "stats.db"):
        candidate = target / name
        if candidate.exists():
            candidate.unlink()


def run_one(args: argparse.Namespace) -> dict:
    run_id = uuid.uuid4().hex
    profile = args.profile_root / f"_resource_experiment_{run_id}"
    telemetry_dir = args.telemetry_root / run_id
    telemetry_dir.mkdir(parents=True, exist_ok=False)
    event_file = profile / "nuzlocke_events.json"
    log_file = telemetry_dir / "bot.log"
    final_file = telemetry_dir / "result.json"
    bot_telemetry_file = telemetry_dir / "bot-telemetry.jsonl"
    state = "CREATED"
    result: dict = {"run_id": run_id, "state": state, "events": []}
    process: subprocess.Popen | None = None
    offset = 0

    try:
        prepare_profile(args.source_profile, profile)
        state = "STARTING"
        result["state"] = state
        command = [
            args.python,
            "pokebot.py",
            profile.name,
            "-m",
            "Campaign Progression",
            "-hl",
            "-nv",
            "-na",
            "-s",
            "0",
            "--no-save-state",
        ]
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=args.repository,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "POKEBOT_EXPERIMENT_RUN_ID": run_id,
                    "POKEBOT_EXPERIMENT_TELEMETRY": str(bot_telemetry_file),
                },
            )
        state = "RUNNING"
        result["state"] = state
        deadline = time.monotonic() + args.timeout
        rival_active = False
        terminal = None
        while True:
            records, offset = read_events(event_file, offset)
            result["events"].extend(records)
            for event in records:
                event_type = event.get("type")
                payload = event.get("payload", {})
                if event_type == "BattleStarted" and payload.get("is_trainer"):
                    rival_active = True
                    state = "RIVAL_ACTIVE"
                    result["state"] = state
                if event_type == "BattleEnded" and rival_active and payload.get("is_trainer"):
                    terminal = payload.get("outcome", "UNKNOWN")
                    break
                if event_type == "WhiteoutOccurred":
                    terminal = "WHITEOUT"
                    break
            if terminal is not None:
                result["outcome"] = terminal
                state = "TERMINAL"
                result["state"] = state
                break
            if process.poll() is not None:
                raise HarnessError(f"bot exited before terminal event: {process.returncode}")
            if time.monotonic() >= deadline:
                if rival_active:
                    deadline = time.monotonic() + args.rival_grace
                    result["timeout_after_rival"] = True
                    # Continue until the active battle produces a terminal event.
                    while time.monotonic() < deadline:
                        records, offset = read_events(event_file, offset)
                        result["events"].extend(records)
                        if any(
                            event.get("type") == "BattleEnded" and event.get("payload", {}).get("is_trainer")
                            for event in records
                        ):
                            result["outcome"] = "RIVAL_TERMINAL"
                            terminal = result["outcome"]
                            state = "TERMINAL"
                            result["state"] = state
                            break
                        time.sleep(0.2)
                    if terminal is not None:
                        break
                    raise HarnessError("rival battle exceeded bounded grace period")
                result["outcome"] = "DIAGNOSTIC_TIMEOUT_PRE_RIVAL"
                state = "TIMED_OUT"
                result["state"] = state
                break
            time.sleep(0.2)
    except Exception as error:
        result["harness_error"] = repr(error)
        result["state"] = "HARNESS_FAILED"
        raise
    finally:
        if process is not None:
            state = "STOPPING"
            result["state"] = state
            stop_process_group(process)
            if not process_group_gone(process):
                raise HarnessError("process group remained after cleanup")
        if profile.exists():
            shutil.rmtree(profile)
        if profile.exists():
            raise HarnessError(f"disposable profile was not removed: {profile}")
        if bot_telemetry_file.exists():
            for line in bot_telemetry_file.read_text(encoding="utf-8").splitlines():
                try:
                    result.setdefault("telemetry", []).append(json.loads(line))
                except json.JSONDecodeError:
                    result.setdefault("telemetry_errors", 0)
                    result["telemetry_errors"] += 1
        result["state"] = "FINALIZED"
        final_file.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rival-grace", type=float, default=60.0)
    args = parser.parse_args()
    args.telemetry_root.mkdir(parents=True, exist_ok=True)
    try:
        result = run_one(args)
    except HarnessError as error:
        print(f"HARNESS_FAILURE: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in result.items() if key != "events"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
