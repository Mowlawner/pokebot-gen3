from collections.abc import Callable
import atexit
import io
from pathlib import Path
import re
from threading import RLock
from time import perf_counter_ns
from typing import TYPE_CHECKING, Iterable

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from modules.map_data import MapRSE
from modules.map_formatting import format_location, format_map_ids_in_text
from modules.battle_readiness import established_search_probability

if TYPE_CHECKING:
    from modules.encounter import EncounterInfo
    from modules.pokemon import Pokemon
    from modules.stats import GlobalStats, EncounterSummary

theme = Theme(
    {
        "normal": "#a8a878",
        "fire": "#f08030",
        "water": "#6890f0",
        "electric": "#f8d030",
        "grass": "#78c850",
        "ice": "#98d8d8",
        "fighting": "#c03028",
        "poison": "#a040a0",
        "ground": "#e0c068",
        "flying": "#a890f0",
        "psychic": "#f85888",
        "bug": "#a8b820",
        "rock": "#b8a038",
        "ghost": "#705898",
        "dragon": "#7038f8",
        "dark": "#705848",
        "steel": "#b8b8d0",
        # This is the '???' type, but because we cannot just use that as a colour name,
        # it has been renamed. This is done by the `Type` class' `safe_name` property.
        "unknown": "#68a090",
    }
)


def iv_value(pokemon: "Pokemon", iv_stat: str):
    iv = pokemon.ivs[iv_stat]
    result = f"[{iv_colour(iv)}]{iv}[/]"
    if iv_stat != "hp" and pokemon.nature.modifiers[iv_stat] > 1:
        result += "[default]↑[/]"
    elif iv_stat != "hp" and pokemon.nature.modifiers[iv_stat] < 1:
        result += "[default]↓[/]"
    return result


def iv_colour(value: "SpeciesRecord | int | None") -> str:
    if value is None:
        return "grey"
    elif int(value) == 31:
        return "yellow"
    elif int(value) == 0:
        return "purple"
    elif int(value) >= 26:
        return "green"
    elif int(value) <= 5:
        return "red"
    else:
        return "white"


def iv_sum_colour(value: "SpeciesRecord | int | None") -> str:
    if value is None:
        return "grey"
    elif int(value) == 186:
        return "yellow"
    elif int(value) == 0:
        return "purple"
    elif int(value) >= 140:
        return "green"
    elif int(value) <= 50:
        return "red"
    else:
        return "white"


def sv_colour(value: "SpeciesRecord | int | None") -> str:
    if value is None:
        return "grey"
    elif int(value) <= 7:
        return "yellow"
    elif int(value) >= 65528:
        return "purple"
    else:
        return "red"


def format_shiny_average(
    encounter_summary: "EncounterSummary | EncounterTotals",
) -> str:
    if encounter_summary.shiny_encounters > 0:
        return f"1/{int(encounter_summary.total_encounters / encounter_summary.shiny_encounters):,}"
    else:
        return "N/A"


def number(value) -> str:
    return f"{int(value):,}" if value is not None else "-"


def percentage(value, total) -> str:
    return f"{100*value/total:0.2f}%" if value is not None and total is not None and total > 0 else "-"


def print_stats(stats: "GlobalStats", encounter: "EncounterInfo") -> None:
    pokemon = encounter.pokemon
    type_colour = pokemon.species.types[0].safe_name.lower()
    if pokemon.gender is not None and not pokemon.species.name.startswith("Nidoran"):
        gender_code = "♂" if pokemon.gender == "male" else "♀"
        gender_label = "[cyan]Male ♂[/]" if pokemon.gender == "male" else "[pink]Female ♀[/]"
    else:
        gender_code = ""
        gender_label = "-"
    rich_name = f"[{type_colour}][bold]{pokemon.species_name_for_stats}{gender_code}[/bold][/{type_colour}]"

    # General Information table
    pokemon_table = Table(show_header=False, border_style="#888888")
    pokemon_table.add_column("Key", justify="right", width=12)
    pokemon_table.add_column("Value", justify="left", width=15)

    pokemon_table.add_row("[bold]PID[/]", f"{pokemon.personality_value:08X}")
    pokemon_table.add_row("[bold]Level[/]", str(pokemon.level))
    pokemon_table.add_row("[bold]Gender[/]", gender_label)
    pokemon_table.add_row("[bold]Item[/]", "-" if pokemon.held_item is None else pokemon.held_item.name)
    pokemon_table.add_row("[bold]Nature[/]", pokemon.nature.name)
    pokemon_table.add_row("[bold]Ability[/]", pokemon.ability.name)
    pokemon_table.add_row(
        "[bold]Hidden Power[/]",
        f"[{pokemon.hidden_power_type.name.lower()}]{pokemon.hidden_power_type.name} ({pokemon.hidden_power_damage})[/]",
    )
    pokemon_table.add_row(
        "[bold]Shiny Value[/]",
        f"[{sv_colour(pokemon.shiny_value)}]{pokemon.shiny_value:,}[/]",
    )
    if pokemon.species.name == "Wurmple":
        pokemon_table.add_row("[bold]Evolution[/]", pokemon.wurmple_evolution.title())

    # IVs table
    iv_table = Table(title="IVs", border_style="#888888")
    iv_table.add_column("HP", justify="center", style=iv_colour(pokemon.ivs.hp), width=5)
    iv_table.add_column("ATK", justify="center", style=iv_colour(pokemon.ivs.attack), width=5)
    iv_table.add_column("DEF", justify="center", style=iv_colour(pokemon.ivs.defence), width=5)
    iv_table.add_column("SPATK", justify="center", style=iv_colour(pokemon.ivs.special_attack), width=5)
    iv_table.add_column("SPDEF", justify="center", style=iv_colour(pokemon.ivs.special_defence), width=5)
    iv_table.add_column("SPD", justify="center", style=iv_colour(pokemon.ivs.speed), width=5)
    iv_table.add_column("Total", justify="right", style=iv_sum_colour(pokemon.ivs.sum()), width=5)

    iv_table.add_row(
        iv_value(pokemon, "hp"),
        iv_value(pokemon, "attack"),
        iv_value(pokemon, "defence"),
        iv_value(pokemon, "special_attack"),
        iv_value(pokemon, "special_defence"),
        iv_value(pokemon, "speed"),
        f"[bold {iv_sum_colour(pokemon.ivs.sum())}]{pokemon.ivs.sum()}[/]",
    )

    # Moves
    move_list = []
    for learned_move in pokemon.moves:
        if learned_move is not None:
            move = learned_move.move
            move_list.append(f"[{move.type.safe_name.lower()}]{move.name}[/]")
    move_list = f"\n[bold]Moves:[/] {', '.join(move_list)}"

    # EV Yield
    ev_yields = []
    for stat in (
        "hp",
        "attack",
        "defence",
        "speed",
        "special_attack",
        "special_defence",
    ):
        if pokemon.species.ev_yield[stat] > 0:
            stat_name = "HP" if stat == "hp" else stat.replace("_", " ").title()
            ev_yields.append(f"{pokemon.species.ev_yield[stat]} {stat_name}")
    ev_yields = f"[bold]EV Yield:[/] {', '.join(ev_yields)}"

    # Statistics table
    stats_table = Table(title="Statistics", border_style="#888888")
    stats_table.add_column("", justify="left", width=10)
    stats_table.add_column("Phase IV Records", justify="center", width=10)
    stats_table.add_column("Phase SV Records", justify="center", width=15)
    stats_table.add_column("Phase Encounters", justify="right", width=10)
    stats_table.add_column("Phase %", justify="right", width=10)
    stats_table.add_column("Shiny Encounters", justify="right", width=10)
    stats_table.add_column("Total Encounters", justify="right", width=10)
    stats_table.add_column("Shiny Average", justify="right", width=10)

    encounter_summaries: list["EncounterSummary"] = list(
        filter(lambda e: e.phase_encounters > 0, stats.encounter_summaries.values())
    )
    encounter_summaries.sort(key=lambda e: e.species_name)
    for summary in encounter_summaries:
        stats_table.add_row(
            summary.species_name,
            f"[red]{number(summary.phase_lowest_iv_sum)}[/] / [green]{number(summary.phase_highest_iv_sum)}",
            f"[green]{number(summary.phase_lowest_sv)}[/] / [{sv_colour(summary.phase_highest_sv)}]{number(summary.phase_highest_sv)}",
            f"{number(summary.phase_encounters)}",
            f"{percentage(summary.phase_encounters, stats.totals.phase_encounters)}",
            f"{number(summary.shiny_encounters)}",
            f"{number(summary.total_encounters)}",
            format_shiny_average(summary),
        )
    stats_table.add_row(
        "[bold yellow]Total",
        f"[red]{number(stats.totals.phase_lowest_iv_sum)}[/] / [green]{number(stats.totals.phase_highest_iv_sum)}",
        f"[green]{number(stats.totals.phase_lowest_sv)}[/] / [{sv_colour(stats.totals.phase_highest_sv)}]{number(stats.totals.phase_highest_sv)}",
        f"[bold yellow]{number(stats.totals.phase_encounters)}",
        "[bold yellow]100%",
        f"[bold yellow]{number(stats.totals.shiny_encounters)}",
        f"[bold yellow]{number(stats.totals.total_encounters)}",
        format_shiny_average(stats.totals),
    )

    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column()
    grid.add_row(pokemon_table, Group(iv_table, move_list, ev_yields))

    map_name = encounter.map.pretty_name if encounter.map is not None else pokemon.location_met
    console.print(
        Panel.fit(
            Group(grid, "\n", stats_table),
            border_style=type_colour,
            title=f"{rich_name} [default]{encounter.type.verb} at [bold]{map_name}[/bold][/default]",
        )
    )


console = Console(theme=theme)


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class _ConsoleLogTee(io.TextIOBase):
    """Write Rich output to the terminal and a plain-text diagnostic file."""

    def __init__(self, terminal, path: Path) -> None:
        super().__init__()
        self.terminal = terminal
        self.path = path
        self._log = path.open("w", encoding="utf-8")
        self._lock = RLock()

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", None) or "utf-8"

    def write(self, text: str) -> int:
        with self._lock:
            written = self.terminal.write(text)
            self._log.write(_ANSI_ESCAPE.sub("", text))
            self.flush()
            return len(text) if written is None else written

    def flush(self) -> None:
        with self._lock:
            self.terminal.flush()
            self._log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.terminal.fileno()

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        self._log.close()


_console_log_tee: _ConsoleLogTee | None = None


def configure_console_log(path: str | Path | None) -> None:
    """Optionally tee all Rich console output into a fresh plain-text file."""

    global _console_log_tee
    close_console_log()
    if path is None:
        return

    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    terminal = console.file
    _console_log_tee = _ConsoleLogTee(terminal, log_path)
    console.file = _console_log_tee


def close_console_log() -> None:
    """Flush and detach the optional console log file."""

    global _console_log_tee
    if _console_log_tee is None:
        return
    tee = _console_log_tee
    _console_log_tee = None
    console.file = tee.terminal
    tee.close()


atexit.register(close_console_log)


def print_battle_action(
    *,
    objective: str,
    target: str,
    intent: str,
    active: str,
    opponent: str,
    action: str,
    source: str,
    reason: str,
    evaluator: str | None = None,
    active_hp: str | None = None,
    active_status: str | None = None,
    opponent_hp: str | None = None,
    opponent_status: str | None = None,
) -> None:
    """Render one selected battle action in a human-readable Rich panel."""

    def projection_rows(value: str):
        """Return readable rows for the packed expectiminimax explanation."""

        fields = [part.strip() for part in value.split(";") if part.strip()]
        structured = any(
            part.startswith((
                "expectiminimax selected",
                "win=",
                "party-wipe risk=",
                "category=",
                "line=",
                "immediate=",
                "switches=",
                "escape=",
            ))
            for part in fields
        )
        if not structured:
            return None

        rows = []
        details = []
        safety = []
        for field in fields:
            if field.startswith("expectiminimax selected"):
                rows.append(("Decision", field))
            elif field.startswith(("win=", "party-wipe risk=", "escape=")):
                safety.append(field)
            elif field.startswith("category="):
                rows.append(("Search", field))
            elif field.startswith("line="):
                rows.append(("Principal variation", field.removeprefix("line=")))
            elif field.startswith("immediate="):
                immediate = field.removeprefix("immediate=")
                player_marker = "player=["
                opponent_marker = "] opponent=["
                order_marker = "] order=["
                if immediate.startswith(player_marker) and opponent_marker in immediate and order_marker in immediate:
                    player, remainder = immediate[len(player_marker):].split(opponent_marker, 1)
                    opponent, order = remainder.split(order_marker, 1)
                    order = order.removesuffix("]")
                    projection = Table.grid(padding=(0, 1))
                    projection.add_column(style="bold white", width=9, no_wrap=True)
                    projection.add_column(ratio=1, overflow="fold")
                    for label, branch_text in (
                        ("Player", player),
                        ("Opponent", opponent),
                        ("Order", order),
                    ):
                        branches = tuple(item.strip() for item in branch_text.split(",") if item.strip())
                        projection.add_row(label, "\n".join(f"• {item}" for item in branches) or "none")
                    rows.append(("Immediate projection", projection))
                else:
                    rows.append(("Immediate projection", immediate))
            elif field.startswith("switches="):
                rows.append(("Switch alternatives", field.removeprefix("switches=")))
            else:
                details.append(field)
        if safety:
            rows.insert(1, ("Safety", "; ".join(safety)))
        if details:
            rows.append(("Details", "; ".join(details)))
        return rows

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=10, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Objective", f"{escape(objective)} → {escape(target)}")
    details.add_row("Intent", escape(intent))
    details.add_row("Matchup", f"{escape(active)}  vs  {escape(opponent)}")
    if active_hp is not None or active_status is not None:
        details.add_row(
            "Active HP",
            f"{escape(str(active_hp or '?'))} ({escape(str(active_status or 'unknown'))})",
        )
    if opponent_hp is not None or opponent_status is not None:
        details.add_row(
            "Opponent HP",
            f"{escape(str(opponent_hp or '?'))} ({escape(str(opponent_status or 'unknown'))})",
        )
    details.add_row("Action", f"[bold white]{escape(action)}[/]")
    details.add_row("Source", f"[magenta]{escape(source)}[/]")
    if evaluator:
        details.add_row("Evaluator", escape(evaluator))
    formatted_reason = projection_rows(reason)
    if formatted_reason is None:
        details.add_row("Reason", escape(reason))
    else:
        reason_grid = Table.grid(padding=(0, 1), expand=True)
        reason_grid.add_column(style="bold white", width=20, no_wrap=True)
        reason_grid.add_column(ratio=1, overflow="fold")
        for label, value in formatted_reason:
            reason_grid.add_row(label, value if isinstance(value, Table) else escape(value))
        details.add_row("Reason", reason_grid)
    console.print(Panel(details, title="[bold cyan]Battle Action[/]", border_style="cyan", expand=True))


def _battle_target_text(name: str | None, party_index: int | None) -> str:
    if not name:
        return "none"
    if party_index is None:
        return escape(str(name))
    return escape(f"{name} [slot {party_index}]")


def print_battle_switch_resolution(
    *,
    status: str,
    requested: str | None = None,
    requested_index: int | None = None,
    final: str | None = None,
    final_index: int | None = None,
    observed: str | None = None,
    observed_index: int | None = None,
    active: str | None = None,
    opponent: str | None = None,
    reason: str = "",
) -> None:
    """Render the executor and observation state for a voluntary switch."""

    status_text = str(status).replace("_", " ").title()
    status_style = {
        "dispatched": "cyan",
        "replanned": "magenta",
        "rejected": "red",
        "observed": "green",
        "mismatch": "red",
        "not observed": "yellow",
    }.get(str(status).casefold(), "yellow")
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style=f"bold {status_style}", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Status", f"[{status_style}]{escape(status_text)}[/]")
    details.add_row("Requested", _battle_target_text(requested, requested_index))
    details.add_row("Final target", _battle_target_text(final, final_index))
    if observed is not None or observed_index is not None:
        details.add_row("Observed", _battle_target_text(observed, observed_index))
    if active or opponent:
        details.add_row("Matchup", f"{escape(active or 'unknown')}  vs  {escape(opponent or 'unknown')}")
    if reason:
        details.add_row("Reason", escape(reason))
    console.print(
        Panel(
            details,
            title="[bold yellow]Battle Switch[/]",
            border_style=status_style,
            expand=True,
        )
    )


def print_battle_safety_stop(
    *,
    action: str,
    target: str,
    repetitions: int,
    active: str,
    opponent: str,
    reason: str,
) -> None:
    """Render the evidence behind an autonomous battle safety stop."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold red", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Action", escape(action))
    details.add_row("Target", escape(target))
    details.add_row("Repetitions", str(repetitions))
    details.add_row("Matchup", f"{escape(active)}  vs  {escape(opponent)}")
    details.add_row("Reason", escape(reason))
    console.print(Panel(details, title="[bold red]Battle Safety Stop[/]", border_style="red", expand=True))


def _battle_roster_text(roster: Iterable[str] | None) -> str:
    if roster is None:
        return "unknown"
    values = tuple(str(value) for value in roster if value)
    return ", ".join(values) if values else "none"


def print_battle_start(
    *,
    battle_type: str,
    encounter_type: str | None = None,
    trainer_name: str | None = None,
    trainer_id: int | None = None,
    opponent_roster: Iterable[str] | None = None,
    player_roster: Iterable[str] | None = None,
    opponent_envelope: Iterable[str] | None = None,
    envelope_source: str | None = None,
) -> None:
    """Render one normal-output record when a battle becomes active."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Type", battle_type or "unknown")
    if encounter_type:
        details.add_row("Encounter", encounter_type)
    if "trainer" in (battle_type or "").casefold():
        details.add_row("Trainer", trainer_name or "unknown")
        if trainer_id is not None:
            details.add_row("Trainer ID", str(trainer_id))
        details.add_row("Opponent party", _battle_roster_text(opponent_roster))
        if opponent_envelope is not None:
            details.add_row("Move envelope", _battle_roster_text(opponent_envelope))
            details.add_row("Envelope source", envelope_source or "unknown")
    else:
        details.add_row("Opponent", _battle_roster_text(opponent_roster))
    if player_roster:
        details.add_row("Player party", _battle_roster_text(player_roster))
    console.print(Panel(details, title="[bold green]Battle Start[/]", border_style="green", expand=True))


def print_battle_event(*, event: str, pokemon: str | None = None, side: str | None = None, detail: str = "") -> None:
    """Render a significant battle transition without per-frame noise."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold yellow", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Event", event)
    if side:
        details.add_row("Side", side)
    if pokemon:
        details.add_row("Pokémon", pokemon)
    if detail:
        details.add_row("Details", detail)
    console.print(Panel(details, title="[bold yellow]Battle Event[/]", border_style="yellow", expand=True))


def print_forced_replacement(
    *,
    fainted: str,
    selected: str | None,
    party_index: int | None,
    reason: str,
    available: bool = True,
) -> None:
    """Render the policy decision made at a mandatory faint prompt."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold red", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Fainted", fainted)
    details.add_row("Decision", selected or "whiteout / no replacement")
    if party_index is not None:
        details.add_row("Party slot", str(party_index))
    details.add_row("Available", str(available))
    details.add_row("Reason", reason)
    console.print(Panel(details, title="[bold red]Forced Replacement[/]", border_style="red", expand=True))


def print_battle_end(
    *,
    outcome: str,
    battle_type: str,
    opponent_roster: Iterable[str] | None = None,
    total_damage_taken: int = 0,
    damage_by_pokemon: Iterable[tuple[str, int]] = (),
    items_used: Iterable[str] = (),
    moves_used: Iterable[str] = (),
    experience_gained: Iterable[tuple[str, int]] = (),
    faints: Iterable[str] = (),
    switches: Iterable[str] = (),
) -> None:
    """Render one aggregate battle result after the ROM leaves battle mode."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=18, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Outcome", outcome or "unknown")
    details.add_row("Type", battle_type or "unknown")
    details.add_row("Opponent party", _battle_roster_text(opponent_roster))
    details.add_row("Damage taken", str(total_damage_taken))
    details.add_row("Damage by Pokémon", _battle_roster_text(f"{name}={amount}" for name, amount in damage_by_pokemon))
    details.add_row("Items used", _battle_roster_text(items_used))
    details.add_row("Moves used", _battle_roster_text(moves_used))
    details.add_row("EXP gained", _battle_roster_text(f"{name}={amount}" for name, amount in experience_gained))
    details.add_row("Faints", _battle_roster_text(faints))
    details.add_row("Switches", _battle_roster_text(switches))
    stopped = str(outcome or "").casefold().replace(" ", "") == "inprogress"
    title = "[bold yellow]Battle Stopped[/]" if stopped else "[bold blue]Battle Finished[/]"
    border_style = "yellow" if stopped else "blue"
    console.print(Panel(details, title=title, border_style=border_style, expand=True))


def print_recovery_request(
    *, location, critical: bool, reason: str, candidate_count: int, destinations: Iterable[str]
) -> None:
    """Render why recovery routing was requested and what it will inspect."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Current location", repr(location))
    details.add_row("Urgency", "critical" if critical else "opportunistic")
    details.add_row("Decision", "pending route calculation")
    details.add_row("Reason", reason)
    details.add_row("Candidates", str(candidate_count))
    details.add_row("Destinations", _battle_roster_text(destinations))
    console.print(Panel(details, title="[bold magenta]Recovery Request[/]", border_style="magenta", expand=True))


def print_recovery_destinations(diagnostics: Iterable[object]) -> None:
    """Render the bounded destination search as a compact table."""

    table = Table(title="Recovery destinations", border_style="magenta")
    for column in ("Rank", "Source", "Destination", "Result", "Distance", "Encounters", "Reason"):
        table.add_column(column, overflow="fold")
    rows = tuple(diagnostics)
    if not rows:
        table.add_row("-", "-", "-", "none", "-", "-", "no candidate diagnostics")
    for item in rows:
        table.add_row(
            str(getattr(item, "rank", "-")),
            str(getattr(item, "source_id", "-")),
            repr(getattr(item, "destination", "-")),
            str(getattr(item, "outcome", "-")),
            str(getattr(item, "distance", "-")),
            str(getattr(item, "encounter_opportunities", "-")),
            str(getattr(item, "reason", "")),
        )
    console.print(Panel(table, title="[bold magenta]Recovery Destination Search[/]", border_style="magenta"))


def print_recovery_attempt(
    *, method: str, destination=None, reason: str = "", execution_kind: str | None = None
) -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold yellow", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", method)
    if execution_kind:
        details.add_row("Execution", execution_kind)
    if destination is not None:
        details.add_row("Destination", repr(destination))
    if reason:
        details.add_row("Reason", reason)
    console.print(Panel(details, title="[bold yellow]Recovery Attempt[/]", border_style="yellow", expand=True))


def print_recovery_success(
    *, method: str, destination=None, detail: str = "", execution_kind: str | None = None
) -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold green", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", method)
    if execution_kind:
        details.add_row("Execution", execution_kind)
    if destination is not None:
        details.add_row("Destination", repr(destination))
    if detail:
        details.add_row("Details", detail)
    console.print(Panel(details, title="[bold green]Recovery Success[/]", border_style="green", expand=True))


def print_recovery_failure(
    *,
    method: str,
    error: str,
    repeated: int = 1,
    escalated: bool = False,
    phase: str | None = None,
    location=None,
    game_state: str | None = None,
    observed_triggers: Iterable[str] | None = None,
    target_pc=None,
    traceback_text: str | None = None,
) -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold red", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", method)
    details.add_row("Failure", error)
    details.add_row("Repeated", str(repeated))
    if phase:
        details.add_row("Phase", phase)
    if location is not None:
        details.add_row("Location", repr(location))
    if game_state:
        details.add_row("Game state", game_state)
    if observed_triggers is not None:
        details.add_row("Observed triggers", ", ".join(str(trigger) for trigger in observed_triggers) or "none")
    if target_pc is not None:
        details.add_row("Target PC", repr(target_pc))
    if traceback_text:
        details.add_row("Traceback", traceback_text)
    details.add_row("Action", "escalated for review" if escalated else "will replan")
    console.print(Panel(details, title="[bold red]Recovery Failure[/]", border_style="red", expand=True))


def print_restock_request(
    *,
    objective: str,
    location=None,
    wallet: int | None = None,
    cash_floor: int = 0,
    required: Iterable[str] = (),
    items: Iterable[str] = (),
    reason: str = "",
) -> None:
    """Render why a resource restock was requested and what it plans to buy."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Objective", objective)
    if location is not None:
        details.add_row("Current location", repr(location))
    details.add_row("Wallet", "unknown" if wallet is None else f"${wallet:,}")
    details.add_row("Cash floor", f"${cash_floor:,}")
    details.add_row("Required", ", ".join(str(value) for value in required) or "none")
    details.add_row("Planned items", ", ".join(str(value) for value in items) or "none")
    if reason:
        details.add_row("Reason", reason)
    console.print(Panel(details, title="[bold cyan]Restock Request[/]", border_style="cyan", expand=True))


def print_restock_attempt(*, method: str, destination=None, items: Iterable[str] = (), total_cost: int = 0) -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold yellow", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", method)
    if destination is not None:
        details.add_row("Destination", repr(destination))
    details.add_row("Items", ", ".join(str(value) for value in items) or "none")
    details.add_row("Total cost", f"${total_cost:,}")
    console.print(Panel(details, title="[bold yellow]Restock Attempt[/]", border_style="yellow", expand=True))


def print_restock_success(*, method: str, destination=None, detail: str = "") -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold green", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", method)
    if destination is not None:
        details.add_row("Destination", repr(destination))
    if detail:
        details.add_row("Details", detail)
    console.print(Panel(details, title="[bold green]Restock Success[/]", border_style="green", expand=True))


def print_restock_failure(*, method: str, error: str, action: str = "will replan") -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold red", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", method)
    details.add_row("Failure", error)
    details.add_row("Action", action)
    console.print(Panel(details, title="[bold red]Restock Failure[/]", border_style="red", expand=True))


def print_money_trainer_request(*, location=None, wallet: int | None = None, reason: str = "") -> None:
    """Render why trainer funding was requested."""

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=16, no_wrap=True)
    details.add_column(ratio=1)
    if location is not None:
        details.add_row("Current location", repr(location))
    details.add_row("Wallet", "unknown" if wallet is None else f"${wallet:,}")
    if reason:
        details.add_row("Reason", reason)
    console.print(Panel(details, title="[bold cyan]Money Funding Request[/]", border_style="cyan", expand=True))


def print_money_trainer_destinations(diagnostics: Iterable[object], *, selected: str | None = None) -> None:
    """Render every reachable trainer considered for funding."""

    table = Table(title="Money funding destinations", border_style="cyan")
    for column in ("Rank", "Map", "Trainer", "Route cost", "Source", "Result"):
        table.add_column(column, overflow="fold")
    rows = tuple(diagnostics)
    if not rows:
        table.add_row("-", "-", "-", "-", "-", "none")
    for rank, item in enumerate(rows):
        trainer_id = str(getattr(item, "trainer_id", "-"))
        table.add_row(
            str(rank),
            repr(getattr(item, "map_id", "-")),
            trainer_id,
            str(getattr(item, "estimated_route_cost", "-")),
            str(getattr(item, "source", "-")),
            (
                "selected"
                if trainer_id == selected
                else (
                    f"rejected: {getattr(item, 'readiness_status', 'unknown')}"
                    if getattr(item, "readiness_status", None) is not None
                    else "candidate"
                )
            ),
        )
    console.print(Panel(table, title="[bold cyan]Money Trainer Search[/]", border_style="cyan"))


def print_money_trainer_attempt(*, trainer: str, destination=None, wallet: int | None = None, reason: str = "") -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold yellow", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", "trainer battle")
    details.add_row("Trainer", trainer)
    if destination is not None:
        details.add_row("Destination", repr(destination))
    details.add_row("Starting wallet", "unknown" if wallet is None else f"${wallet:,}")
    if reason:
        details.add_row("Reason", reason)
    console.print(Panel(details, title="[bold yellow]Money Trainer Attempt[/]", border_style="yellow", expand=True))


def print_money_trainer_success(*, trainer: str, wallet_before: int, wallet_after: int, detail: str = "") -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold green", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Trainer", trainer)
    details.add_row("Wallet before", f"${wallet_before:,}")
    details.add_row("Wallet after", f"${wallet_after:,}")
    details.add_row("Payout", f"${wallet_after - wallet_before:,}")
    if detail:
        details.add_row("Details", detail)
    console.print(Panel(details, title="[bold green]Money Trainer Success[/]", border_style="green", expand=True))


def print_money_trainer_failure(*, trainer: str | None, error: str, action: str = "will replan") -> None:
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold red", width=14, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Method", "trainer battle")
    if trainer:
        details.add_row("Trainer", trainer)
    details.add_row("Failure", error)
    details.add_row("Action", action)
    console.print(Panel(details, title="[bold red]Money Trainer Failure[/]", border_style="red", expand=True))


def format_campaign_objective_id(objective_id: str) -> str:
    """Format a campaign ID for people while preserving its machine ID elsewhere."""

    if not isinstance(objective_id, str) or not objective_id.startswith("obtain_encounter:"):
        return objective_id
    parts = objective_id.split(":")
    if len(parts) != 3:
        return objective_id
    try:
        return f"obtain_encounter:{MapRSE((int(parts[1]), int(parts[2]))).pretty_name}"
    except (TypeError, ValueError):
        return objective_id


def print_campaign_frontier(
    completed: tuple[str, ...],
    available: tuple[str, ...],
    *,
    selected: str | None = None,
    status: str | None = None,
    reason: str | None = None,
    selected_detail: str | None = None,
    decision_details: tuple[dict[str, object], ...] = (),
    encounter_portfolio: tuple[dict[str, object], ...] = (),
) -> None:
    """Render the campaign planner's current objective frontier.

    Objective transitions are useful normal run output, just like the Rich
    encounter reports. The campaign controller calls this only when its
    planning signature changes, so this helper is not part of the per-frame
    hot path.
    """

    def render_ids(objective_ids: tuple[str, ...]) -> str:
        return (
            "\n".join(f"• {format_campaign_objective_id(objective_id)}" for objective_id in objective_ids)
            or "[dim]None[/]"
        )

    frontier = Table(show_header=False, box=None, padding=(0, 1))
    frontier.add_column("Field", style="bold", width=20)
    frontier.add_column("Objectives", overflow="fold")
    frontier.add_row("Completed", render_ids(completed))
    frontier.add_row("Available", render_ids(available))
    frontier.add_row("Selected", format_campaign_objective_id(selected) if selected else "[dim]None[/]")
    if selected_detail or (selected and selected.startswith("restock_")):
        frontier.add_row("Items", selected_detail or "Purchase plan unavailable")
    frontier.add_row("Status", status or "[dim]unknown[/]")
    if reason:
        frontier.add_row("Reason", reason)
    if decision_details:
        tradeoffs = Table.grid(padding=(0, 1), expand=True)
        tradeoffs.add_column(ratio=1, overflow="fold")

        def number(value) -> str:
            if value is None:
                return "?"
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)

        for detail in decision_details:
            task_id = format_campaign_objective_id(str(detail.get("task_id", "unknown")))
            state = "SELECTED" if detail.get("selected") else "SKIPPED"
            route = number(detail.get("route_cost"))
            detour = number(detail.get("detour_cost"))
            score = number(detail.get("strategic_score"))
            trainer_gain = number(detail.get("trainer_preparation_gain"))
            checkpoint = detail.get("trainer_checkpoint_id") or "none"
            policy = detail.get("recommendation") or "unknown"
            classification = detail.get("classification") or "unknown"
            species = ", ".join(str(item) for item in (detail.get("candidate_species") or ())) or "unknown"
            tradeoffs.add_row(
                escape(
                    f"{state} {task_id}: {detail.get('reason', 'no explanation')}; "
                    f"policy={policy}/{classification}; value={score}; route={route}; detour={detour}; "
                    f"trainer_gain={trainer_gain}; gate={checkpoint}; species={species}"
                )
            )
        frontier.add_row("Encounter tradeoffs", tradeoffs)
    if encounter_portfolio:
        portfolio = Table(show_header=True, box=None, padding=(0, 1), expand=True)
        portfolio.add_column("#", justify="right", width=3, no_wrap=True)
        portfolio.add_column("Selection", width=10, no_wrap=True)
        portfolio.add_column("Encounter", ratio=1, overflow="fold")
        portfolio.add_column("Score", justify="right", width=8, no_wrap=True)
        portfolio.add_column("Policy", width=18, no_wrap=True)
        portfolio.add_column("Route", justify="right", width=8, no_wrap=True)
        portfolio.add_column("Detour", justify="right", width=8, no_wrap=True)
        portfolio.add_column("Trainer gain", justify="right", width=12, no_wrap=True)
        portfolio.add_column("Species", ratio=2, overflow="fold")

        def portfolio_number(value) -> str:
            if value is None:
                return "?"
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)

        for detail in encounter_portfolio:
            selection = "SELECTED" if detail.get("selected") else (
                "PORTFOLIO" if detail.get("portfolio_selected") else ""
            )
            objective = format_campaign_objective_id(str(detail.get("task_id", "unknown")))
            method = detail.get("method") or "unknown"
            area = detail.get("area")
            location = format_campaign_objective_id(
                f"obtain_encounter:{area[0]}:{area[1]}"
            ) if isinstance(area, tuple) and len(area) == 2 else str(area or "unknown")
            species = ", ".join(str(item) for item in (detail.get("species") or ())) or "unknown"
            portfolio.add_row(
                str(detail.get("rank", "?")),
                selection,
                escape(f"{objective} ({location}, {method})"),
                portfolio_number(detail.get("score")),
                escape(
                    f"{detail.get('policy', 'unknown')}/{detail.get('classification', 'unknown')}"
                ),
                portfolio_number(detail.get("route_cost")),
                portfolio_number(detail.get("detour_cost")),
                portfolio_number(detail.get("trainer_preparation_gain")),
                escape(species),
            )
        frontier.add_row("Ordered portfolio", portfolio)
    console.print(Panel.fit(frontier, border_style="cyan", title="[bold cyan]Campaign Objective Frontier[/]"))


def print_campaign_encounter_stage(evidence) -> None:
    """Render the bounded exit-stage evidence used by campaign routing."""

    if evidence is None:
        return

    def location_text(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, tuple) and len(value) == 2:
            return format_location(value)
        return format_map_ids_in_text(repr(value))

    details = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    details.add_column("#", justify="right", width=3, no_wrap=True)
    details.add_column("Exit", ratio=1, overflow="fold")
    details.add_column("Safe", width=10, no_wrap=True)
    details.add_column("Permissive", width=12, no_wrap=True)
    details.add_column("Safe cost", justify="right", width=9, no_wrap=True)
    details.add_column("Perm. cost", justify="right", width=9, no_wrap=True)
    details.add_column("Trainer exposure", ratio=1, overflow="fold")
    details.add_column("Grass", justify="right", width=8, no_wrap=True)
    safe_by_index = {
        getattr(candidate, "index", None): candidate
        for candidate in tuple(getattr(evidence, "safe_candidates", ()) or ())
    }
    permissive_by_index = {
        getattr(candidate, "index", None): candidate
        for candidate in tuple(getattr(evidence, "permissive_candidates", ()) or ())
    }
    indexes = sorted(set(safe_by_index) | set(permissive_by_index))
    for index in indexes:
        safe = safe_by_index.get(index)
        permissive = permissive_by_index.get(index)
        candidate = safe or permissive

        def exposure(candidate) -> str:
            if candidate is None or getattr(candidate, "trainer_free", None) is None:
                return "?"
            trainer_ids = tuple(getattr(candidate, "trainer_ids", ()) or ())
            return "trainer free" if not trainer_ids else ", ".join(trainer_ids)

        def grass(candidate) -> str:
            if candidate is None:
                return "—"
            return str(getattr(candidate, "encounter_opportunities", "?"))

        details.add_row(
            str(index),
            escape(
                f"entry={location_text(getattr(candidate, 'entry', None))}; "
                f"approach={location_text(getattr(candidate, 'approach', None))}; "
                f"destination={location_text(getattr(candidate, 'destination', None))}"
            ),
            escape(getattr(safe, "status", "—") if safe is not None else "—"),
            escape(getattr(permissive, "status", "—") if permissive is not None else "—"),
            str(getattr(safe, "route_cost", "—") if safe is not None else "—"),
            str(getattr(permissive, "route_cost", "—") if permissive is not None else "—"),
            escape(
                f"safe={exposure(safe)}; permissive={exposure(permissive)}"
            ),
            escape(f"safe={grass(safe)}; perm={grass(permissive)}"),
        )
    if not indexes:
        details.add_row("—", "no candidate evidence", "—", "—", "—", "—", "—", "—")

    hazards = ", ".join(
        str(item)
        for item in (
            tuple(getattr(evidence, "known_hazard_ids", ()) or ())
            + tuple(getattr(evidence, "unresolved_hazard_ids", ()) or ())
        )
    ) or "none"
    trainer_geometry = "; ".join(
        (
            f"{getattr(item, 'trainer_id', None)!r} "
            f"object={getattr(item, 'object_id', None)!r} "
            f"location={getattr(item, 'location', None)!r} "
            f"type={getattr(item, 'trainer_type', None)!r} "
            f"range={getattr(item, 'trainer_range', None)!r} "
            f"facing={getattr(item, 'facing', None)!r} "
            f"movement={getattr(item, 'movement_type', None)!r} "
            f"movement_range={getattr(item, 'movement_range', None)!r} "
            f"movement_origin={getattr(item, 'movement_origin', None)!r} "
            f"defeated={getattr(item, 'defeated', None)!r} "
            f"hazards={tuple(getattr(item, 'hazard_locations', ()) or ())!r} "
            f"engagement={tuple(getattr(item, 'engagement_locations', ()) or ())!r} "
            f"activation={tuple(getattr(item, 'activation_locations', ()) or ())!r}"
        )
        for item in tuple(getattr(evidence, "trainer_evidence", ()) or ())
    ) or "none"
    status = "BLOCKED" if getattr(evidence, "blocked", False) else "SELECTED"
    summary = Table.grid(padding=(0, 1), expand=True)
    summary.add_column(style="bold cyan", width=18, no_wrap=True)
    summary.add_column(ratio=1, overflow="fold")
    summary.add_row("Stage", escape(status))
    summary.add_row("Current map", escape(location_text(getattr(evidence, "current_map", None))))
    summary.add_row("Next map", escape(location_text(getattr(evidence, "next_map", None))))
    summary.add_row("Selected candidate", escape(str(getattr(evidence, "selected_candidate", None))))
    summary.add_row("Readiness", escape(str(getattr(evidence, "readiness", None))))
    summary.add_row("Trainer hazards", escape(format_map_ids_in_text(hazards)))
    summary.add_row("Trainer geometry", escape(format_map_ids_in_text(trainer_geometry)))
    summary.add_row("Reason", escape(str(getattr(evidence, "reason", "") or "—")))
    summary.add_row("Shared search", escape(str(getattr(evidence, "shared_search", False))))
    summary.add_row("Candidates", details)
    console.print(Panel(summary, title="[bold cyan]Campaign Encounter Stage[/]", border_style="cyan", expand=True))


def print_campaign_snapshot(
    *,
    snapshot,
    status=None,
    level_cap: int | None = None,
    boss=None,
    exp_budget=None,
    trainer_checkpoint=None,
    trainer_facts=(),
    trainer_readiness=None,
    readiness=None,
) -> None:
    """Render the campaign facts used at an execution boundary.

    This is intentionally one boundary report rather than a per-frame panel.
    Detailed evaluator traces remain available through ``debug_trace`` while
    this gives an operator enough context to understand why the next campaign
    action was selected.
    """

    if snapshot is None:
        return
    player = getattr(snapshot, "player", None)
    player_name = getattr(player, "name", None) or "unknown"
    location = getattr(player, "map_name", None) or "unknown"
    objective = getattr(status, "objective", None) or "none"
    intent = getattr(status, "intent", None) or "unknown"
    boss_id = getattr(boss, "boss_id", None) or getattr(boss, "name", None) or "none"
    cap_text = "none" if level_cap is None else str(level_cap)

    party_rows = []
    for pokemon in getattr(snapshot, "party", ()):
        state = "fainted" if getattr(pokemon, "fainted", False) else getattr(pokemon, "status", "healthy")
        species = getattr(pokemon, "species", "unknown")
        nickname = getattr(pokemon, "nickname", None)
        identity = (
            f'{species} "{nickname}"'
            if nickname and str(nickname).casefold() != str(species).casefold()
            else str(species)
        )
        met_location = getattr(pokemon, "met_location", None)
        party_rows.append(
            f"{identity} "
            f"Lv{getattr(pokemon, 'level', '?')} "
            f"HP {getattr(pokemon, 'current_hp', '?')}/{getattr(pokemon, 'max_hp', '?')} "
            f"({state})"
            + (f" [met: {met_location}]" if met_location else "")
        )
    party_text = ", ".join(party_rows) if party_rows else "unknown"

    if exp_budget is None:
        exp_text = "unknown"
    else:
        exp_text = (
            f"headroom={getattr(exp_budget, 'cap_headroom', None)!r}; "
            f"required={getattr(exp_budget, 'required_exp', None)!r}; "
            f"unavoidable={getattr(exp_budget, 'unavoidable_exp', None)!r}; "
            f"deficit={getattr(exp_budget, 'remaining_deficit', None)!r}"
        )

    def probability(value) -> str:
        return "?" if value is None else f"{100 * float(value):.1f}%"

    def enum_value(value):
        return getattr(value, "value", value)

    facts = tuple(trainer_facts or ())
    if not facts and trainer_checkpoint is not None:
        facts = (trainer_checkpoint,)

    def fact_sources(fact) -> str:
        records = tuple(getattr(fact, "source_records", ()) or ())
        if not records:
            records = (
                (
                    getattr(fact, "source", "unknown"),
                    getattr(fact, "checkpoint_id", "unknown"),
                ),
            )
        return ", ".join(f"{identifier} ({source})" for source, identifier in records)

    trainer_facts_text = format_map_ids_in_text("; ".join(
        (
            f"{getattr(fact, 'display_name', 'unknown')} "
            f"[{getattr(fact, 'trainer_affordance_id', 'unknown')}] "
            f"sources={fact_sources(fact)}"
        )
        for fact in facts
    ) or "none (unknown)")

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=16, no_wrap=True)
    details.add_column(ratio=1, overflow="fold")
    details.add_row("Player", escape(str(player_name)))
    details.add_row("Location", escape(str(location)))
    details.add_row("Objective", escape(str(objective)))
    details.add_row("Intent", escape(str(intent)))
    details.add_row("Boss / cap", escape(f"{boss_id} / {cap_text}"))
    details.add_row("Party", escape(party_text))
    details.add_row("EXP", escape(exp_text))
    if readiness is not None:
        decision = enum_value(getattr(readiness, "readiness_decision", None)) or "NOT_EVALUATED"
        decision_reason = enum_value(getattr(readiness, "readiness_reason", None)) or "unknown"
        details.add_row("Readiness gate", escape(f"{decision}: {decision_reason}"))
        if decision_reason == "TRAINER_PREPARATION_CEILING":
            current_level = getattr(readiness, "preparation_current_level", None)
            ceiling_level = getattr(readiness, "preparation_ceiling_level", None)
            level_text = (
                f"Lv{current_level}/{ceiling_level}"
                if current_level is not None and ceiling_level is not None
                else "configured ceiling"
            )
            details.add_row(
                "Prep boundary",
                escape(
                    f"{level_text} reached; preparation complete; no further training; "
                    "battle plan owns engagement risk"
                ),
            )
        overworld_availability = enum_value(getattr(readiness, "overworld_availability", None))
        overworld_reason = getattr(readiness, "overworld_reason", None)
        if overworld_availability not in (None, "known") or overworld_reason:
            details.add_row(
                "Overworld status",
                escape(f"{overworld_availability or 'unknown'}: {overworld_reason or 'unavailable'}"),
            )
    if facts:
        trainer_table = Table.grid(padding=(0, 1), expand=True)
        trainer_table.add_column(style="bold white", no_wrap=True)
        trainer_table.add_column(ratio=1, overflow="fold")
        for fact in facts:
            trainer_table.add_row(
                escape(str(getattr(fact, "display_name", "unknown") or "unknown")),
                escape(
                    format_map_ids_in_text(
                        f"id={getattr(fact, 'trainer_affordance_id', 'unknown')} "
                        f"sources={fact_sources(fact)}"
                    )
                ),
            )
        details.add_row("Trainer facts", trainer_table)
    else:
        details.add_row("Trainer facts", escape(trainer_facts_text))
    if trainer_checkpoint is not None or trainer_readiness is not None:
        trainer_plan = Table.grid(padding=(0, 1), expand=True)
        trainer_plan.add_column(style="bold white", no_wrap=True)
        trainer_plan.add_column(ratio=1, overflow="fold")
        if trainer_checkpoint is not None:
            trainer_plan.add_row(
                "Checkpoint",
                escape(
                    f"{getattr(trainer_checkpoint, 'display_name', 'unknown')} "
                    f"({getattr(trainer_checkpoint, 'checkpoint_id', 'unknown')})"
                ),
            )
            opponent_party = tuple(getattr(trainer_checkpoint, "party", ()) or ())
            if opponent_party:
                opponent_lead = opponent_party[0]
                trainer_plan.add_row(
                    "Opponent lead",
                    escape(
                        f"{getattr(opponent_lead, 'species', 'unknown')} "
                        f"Lv{getattr(opponent_lead, 'level', '?')}"
                    ),
                )
        if trainer_readiness is None:
            trainer_plan.add_row("Battle readiness", "unknown: no complete trainer envelope")
        else:
            lead = getattr(trainer_readiness, "lead_name", None) or "none"
            lead_source = getattr(trainer_readiness, "lead_source", None) or "unknown"
            first_action = getattr(trainer_readiness, "recommended_action", None) or "none"
            trainer_plan.add_row(
                "Battle readiness",
                escape(
                    f"{enum_value(getattr(trainer_readiness, 'status', None))}; "
                    f"lead={lead} ({lead_source}); first_action={first_action}; "
                    f"roster={getattr(trainer_readiness, 'roster_action', None) or 'none'}; "
                    f"coverage={getattr(trainer_readiness, 'roster_covered_opponents', 0)}/"
                    f"{getattr(trainer_readiness, 'roster_opponent_count', 0)}"
                ),
            )
            trainer_plan.add_row(
                "Risk envelope",
                escape(
                    f"win={probability(established_search_probability(getattr(trainer_readiness, 'search', None), 'win_probability'))}; "
                    f"active_faint={probability(established_search_probability(getattr(trainer_readiness, 'search', None), 'active_faint_probability'))}; "
                    f"party_wipe={probability(established_search_probability(getattr(trainer_readiness, 'search', None), 'party_wipe_probability'))}; "
                    f"run_loss={probability(established_search_probability(getattr(trainer_readiness, 'search', None), 'run_loss_probability'))}"
                ),
            )
            trainer_plan.add_row(
                "Why",
                escape(
                    f"{getattr(trainer_readiness, 'reason', '')}; "
                    f"required_hp_ratio={getattr(trainer_readiness, 'required_hp_ratio', None)!r}"
                ),
            )
            assessments = Table.grid(padding=(0, 1), expand=True)
            assessments.add_column(style="bold white", no_wrap=True)
            assessments.add_column(ratio=1, overflow="fold")
            for assessment in tuple(getattr(trainer_readiness, "lead_assessments", ()) or ()):
                lead_state = "safe" if getattr(assessment, "safe", False) else "prepare"
                line_text = []
                for line in tuple(getattr(assessment, "lines", ()) or ()):
                    result = getattr(line, "result", None)
                    variation = " → ".join(tuple(getattr(result, "principal_variation", ()) or ()))
                    line_text.append(
                        f"vs {getattr(line, 'opponent_species', 'unknown')} Lv{getattr(line, 'opponent_level', '?')}: "
                        f"win={probability(established_search_probability(result, 'win_probability'))}, "
                        f"faint={probability(established_search_probability(result, 'active_faint_probability'))}, "
                        f"first={getattr(result, 'action_name', None) or 'unknown'}, "
                        f"category={enum_value(getattr(result, 'category', None)) or 'unknown'}"
                        + (f", line={variation}" if variation else "")
                    )
                assessments.add_row(
                    escape(
                        f"{getattr(assessment, 'name', 'unknown')} ({getattr(assessment, 'source', 'unknown')}, {lead_state})"
                    ),
                    escape("; ".join(line_text) or "no complete modeled line"),
                )
            if tuple(getattr(trainer_readiness, "lead_assessments", ()) or ()):
                trainer_plan.add_row("Candidate lines", assessments)
        details.add_row("Trainer plan", trainer_plan)
    console.print(Panel(details, title="[bold cyan]Campaign Snapshot[/]", border_style="cyan", expand=True))


def print_navigation_pathfinding(
    *,
    phase: str,
    call_id: int,
    algorithm: str,
    start,
    goal: str,
    goal_type: str,
    nodes_expanded: int | None = None,
    nodes_generated: int | None = None,
    duration_ms: float | None = None,
    path_found: bool | None = None,
    action_count: int | None = None,
    route_cost: int | None = None,
    orientation_required: bool | None = None,
    error: str | None = None,
    caller: str | None = None,
) -> None:
    """Print one readable Rich panel for each navigation search boundary.

    This deliberately bypasses ``diagnostic_print``'s background-worker
    suppression. A search running in the route worker is exactly the event
    that explains a frame-loop standstill, so its start record must remain
    visible even while the worker owns the computation.
    """

    from modules.context import context

    frame = getattr(context, "frame", None)

    def compact(value, limit: int = 180) -> str:
        text = repr(value) if not isinstance(value, str) else value
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def location_text(value) -> str:
        if isinstance(value, tuple) and len(value) == 2:
            return format_location(value)
        return format_map_ids_in_text(compact(value))

    # Keep the stable machine-readable marker in the panel body. Existing
    # console/log consumers can still locate a call by ``id=`` and ``frame=``
    # while the surrounding rows make a long ReachWarp repr readable.
    summary = f"NAVIGATION_PATHFINDING phase={phase} id={call_id} frame={frame!r}"
    if duration_ms is not None:
        summary += f" duration_ms={duration_ms!r}"
    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=14, no_wrap=True)
    details.add_column(ratio=1, overflow="fold")
    details.add_row("Event", escape(summary))
    details.add_row("Algorithm", escape(str(algorithm).upper()))
    if orientation_required is not None:
        details.add_row("Facing", escape("required" if orientation_required else "ignored for search"))
    details.add_row("Start", escape(location_text(start)))
    details.add_row("Goal", escape(f"{goal_type}: {format_map_ids_in_text(compact(goal))}"))
    if caller is not None:
        details.add_row("Caller", escape(caller))
    if nodes_expanded is not None or nodes_generated is not None:
        details.add_row(
            "Search",
            escape(
                f"expanded={nodes_expanded if nodes_expanded is not None else '—'}; "
                f"generated={nodes_generated if nodes_generated is not None else '—'}"
            ),
        )
    if path_found is not None or action_count is not None or route_cost is not None:
        details.add_row(
            "Result",
            escape(
                f"{'found' if path_found else 'not found' if path_found is False else 'pending'}; "
                f"actions={action_count if action_count is not None else '—'}; "
                f"cost={route_cost if route_cost is not None else '—'}"
            ),
        )
    if duration_ms is not None:
        details.add_row("Duration", escape(f"{duration_ms:.3f} ms"))
    if error is not None:
        details.add_row("Error", escape(compact(error)))
    console.print(Panel(details, title="[bold cyan]Navigation Pathfinding[/]", border_style="cyan", expand=True))
    file = getattr(console, "file", None)
    if file is not None:
        file.flush()


def print_navigation_cancellation(
    *,
    phase: str,
    reason: str,
    request_key=None,
    replacement_key=None,
    changed=(),
    duration_ms: float | None = None,
    request_label: str | None = None,
    replacement_label: str | None = None,
    worker: str | None = None,
) -> None:
    """Print why a background navigation request was cancelled.

    This is intentionally unconditional like the pathfinding panels. A
    cancellation is the explanation for a visible navigation pause, so it
    must remain available in an ordinary Rich console run without
    ``--debug-trace``.
    """

    def compact(value, limit: int = 180) -> str:
        text = repr(value) if not isinstance(value, str) else value
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def request_start(key) -> str:
        if isinstance(key, tuple) and key:
            return format_location(key[0])
        return compact(key)

    def request_goal(key) -> str:
        if isinstance(key, tuple) and len(key) > 1:
            return format_map_ids_in_text(compact(key[1]))
        return "unknown"

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold yellow", width=16, no_wrap=True)
    details.add_column(ratio=1, overflow="fold")
    details.add_row(
        "Event",
        escape(
            f"NAVIGATION_PATHFINDING_CANCELLED phase={phase} reason={reason!r}"
        ),
    )
    if worker is not None:
        details.add_row("Worker", escape(worker))
    if request_label is not None:
        details.add_row("Request", escape(request_label))
    else:
        details.add_row("Request start", escape(request_start(request_key)))
        details.add_row("Request goal", escape(request_goal(request_key)))
    if replacement_key is not None:
        if replacement_label is not None:
            details.add_row("Replacement", escape(replacement_label))
        else:
            details.add_row("Replacement start", escape(request_start(replacement_key)))
            details.add_row("Replacement goal", escape(request_goal(replacement_key)))
    elif replacement_label is not None:
        details.add_row("Replacement", escape(replacement_label))
    if changed:
        details.add_row("Changed fields", escape(compact(tuple(changed))))
    if duration_ms is not None:
        details.add_row("Elapsed", escape(f"{duration_ms:.3f} ms"))
    console.print(Panel(details, title="[bold yellow]Navigation Cancelled[/]", border_style="yellow", expand=True))
    file = getattr(console, "file", None)
    if file is not None:
        file.flush()


def print_navigation_invalidation(
    *,
    reason: str,
    route_id: int | None = None,
    current=None,
    goal=None,
    action_index: int | None = None,
    remaining_actions: int | None = None,
    total_actions: int | None = None,
    cached_action=None,
    detail: str | None = None,
) -> None:
    """Render why an already-mounted navigation route was discarded.

    A cancelled worker request and an invalidated cached route are different
    boundaries.  The former is covered by ``print_navigation_cancellation``;
    this panel explains the latter even in an ordinary Rich run without
    debug-trace or stutter-trace enabled.
    """

    def compact(value, limit: int = 220) -> str:
        text = repr(value) if not isinstance(value, str) else value
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def location_text(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, tuple) and len(value) == 2:
            return format_location(value)
        return format_map_ids_in_text(compact(value))

    from modules.context import context

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold red", width=18, no_wrap=True)
    details.add_column(ratio=1, overflow="fold")
    details.add_row(
        "Event",
        escape(
            f"NAVIGATION_PATH_INVALIDATED frame={getattr(context, 'frame', None)!r} "
            f"reason={reason!r}"
        ),
    )
    if route_id is not None:
        details.add_row("Route", escape(str(route_id)))
    details.add_row("Current", escape(location_text(current)))
    if goal is not None:
        details.add_row("Goal", escape(format_map_ids_in_text(compact(goal))))
    if action_index is not None or remaining_actions is not None or total_actions is not None:
        details.add_row(
            "Progress",
            escape(
                f"index={action_index if action_index is not None else '—'}; "
                f"remaining={remaining_actions if remaining_actions is not None else '—'}; "
                f"total={total_actions if total_actions is not None else '—'}"
            ),
        )
    if cached_action is not None:
        details.add_row("Cached action", escape(format_map_ids_in_text(compact(cached_action))))
    if detail is not None:
        details.add_row("Detail", escape(format_map_ids_in_text(detail)))
    console.print(Panel(details, title="[bold red]Navigation Invalidated[/]", border_style="red", expand=True))
    file = getattr(console, "file", None)
    if file is not None:
        file.flush()


def print_recovery_handoff(
    *,
    phase: str,
    objective_id: str | None = None,
    execution_phase: str | None = None,
    start=None,
    destination=None,
    source_id: str | None = None,
    route_actions: int | None = None,
    route_cost: int | None = None,
    action: str | None = None,
    action_source=None,
    action_destination=None,
    loop_id: int | None = None,
    detail: str | None = None,
) -> None:
    """Render one visible recovery calculation-to-execution boundary."""

    def location_text(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, tuple) and len(value) == 2:
            return format_location(value)
        return format_map_ids_in_text(repr(value))

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold green", width=18, no_wrap=True)
    details.add_column(ratio=1, overflow="fold")
    details.add_row("Event", escape(f"CAMPAIGN_RECOVERY_HANDOFF phase={phase}"))
    if objective_id is not None:
        details.add_row("Objective", escape(str(objective_id)))
    if execution_phase is not None:
        details.add_row("Controller phase", escape(execution_phase))
    if start is not None:
        details.add_row("Start", escape(location_text(start)))
    if destination is not None:
        details.add_row("Destination", escape(location_text(destination)))
    if source_id is not None:
        details.add_row("Healing source", escape(source_id))
    if route_actions is not None or route_cost is not None:
        details.add_row(
            "Route",
            escape(
                f"actions={route_actions if route_actions is not None else '—'}; "
                f"cost={route_cost if route_cost is not None else '—'}"
            ),
        )
    if action is not None:
        details.add_row("Action", escape(action))
    if action_source is not None:
        details.add_row("Action source", escape(location_text(action_source)))
    if action_destination is not None:
        details.add_row("Action destination", escape(location_text(action_destination)))
    if loop_id is not None:
        details.add_row("Loop", escape(str(loop_id)))
    if detail is not None:
        details.add_row("Detail", escape(detail))
    console.print(Panel(details, title="[bold green]Recovery Handoff[/]", border_style="green", expand=True))
    file = getattr(console, "file", None)
    if file is not None:
        file.flush()


def print_recovery_plan(
    *,
    objective_id: str,
    decision: str,
    reason: str | None,
    route_analysis_candidates: int,
    route_analysis_eligible: int,
    observed_route_available: bool,
    stop_present: bool,
    observed_center_available: bool | None = None,
    observed_healing_source_available: bool | None = None,
    observed_safe_to_reach: bool | None = None,
    observed_destination=None,
    stop_destination=None,
    stop_route_actions: int | None = None,
    stop_route_cost: int | None = None,
    stop_execution_kind: str | None = None,
    detail: str = "",
) -> None:
    """Show how readiness became an executable recovery campaign plan."""

    def location_text(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, tuple) and len(value) == 2:
            return format_location(value)
        return format_map_ids_in_text(repr(value))

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold green", width=22, no_wrap=True)
    details.add_column(ratio=1, overflow="fold")
    details.add_row("Event", escape("CAMPAIGN_RECOVERY_PLAN"))
    details.add_row("Objective", escape(objective_id))
    details.add_row("Decision", escape(decision))
    details.add_row("Reason", escape(reason or "—"))
    details.add_row(
        "Route analysis",
        escape(
            f"candidates={route_analysis_candidates}; eligible={route_analysis_eligible}"
        ),
    )
    details.add_row(
        "Observed route",
        escape(
            f"{'available' if observed_route_available else 'unavailable'}; "
            f"center={observed_center_available if observed_center_available is not None else '—'}; "
            f"healing_source={observed_healing_source_available if observed_healing_source_available is not None else '—'}; "
            f"safe={observed_safe_to_reach if observed_safe_to_reach is not None else '—'}"
        ),
    )
    details.add_row("Observed destination", escape(location_text(observed_destination)))
    details.add_row("Recovery stop", escape("present" if stop_present else "MISSING"))
    if stop_destination is not None:
        details.add_row("Stop destination", escape(location_text(stop_destination)))
    if stop_route_actions is not None or stop_route_cost is not None:
        details.add_row(
            "Stop route",
            escape(
                f"actions={stop_route_actions if stop_route_actions is not None else '—'}; "
                f"cost={stop_route_cost if stop_route_cost is not None else '—'}"
            ),
        )
    if stop_execution_kind:
        details.add_row("Execution", escape(stop_execution_kind))
    if detail:
        details.add_row("Detail", escape(detail))
    console.print(Panel(details, title="[bold green]Recovery Plan[/]", border_style="green", expand=True))
    file = getattr(console, "file", None)
    if file is not None:
        file.flush()


def print_campaign_field_lead(
    *,
    objective_id: str | None,
    current_index: int,
    selected_index: int | None,
    reason: str,
    route_maps: Iterable[object] = (),
    envelope_source: str | None = None,
    party_slots: Iterable[object] | None = None,
) -> None:
    """Render the event-boundary field-lead decision for campaign runs."""

    def format_slot(pokemon: object) -> str:
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

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold cyan", width=16, no_wrap=True)
    details.add_column(ratio=1)
    details.add_row("Objective", escape(format_campaign_objective_id(objective_id) if objective_id else "none"))
    details.add_row("Current slot", str(current_index))
    details.add_row("Selected slot", str(selected_index) if selected_index is not None else "unchanged")
    if party_slots is not None:
        slots = tuple(party_slots)
        slot_text = ", ".join(f"{index}: {format_slot(pokemon)}" for index, pokemon in enumerate(slots))
        details.add_row("Party slots", escape(slot_text or "empty"))
    details.add_row("Status", "rotated" if selected_index is not None else "kept current lead")
    details.add_row("Reason", escape(reason))
    if envelope_source:
        details.add_row("Envelope", escape(envelope_source))
    maps = tuple(route_maps)
    if maps:
        details.add_row("Route maps", escape(repr(maps)))
    console.print(Panel(details, title="[bold magenta]Campaign Field Lead[/]", border_style="magenta", expand=True))

_DIAGNOSTIC_LOG_WHITELIST = (
    # Keep the timing-sensitive recovery experiment small: retain recovery
    # lifecycle records, post-observation coordinates, ROM transitions, and
    # the actual input boundary, while excluding the per-frame planning and
    # readiness dumps that can perturb the frame loop.
    "CAMPAIGN_RECOVERY_EXECUTION: started",
    "CAMPAIGN_RECOVERY_EXECUTION: completed",
    "CAMPAIGN_RECOVERY_ENTRY_WARP",
    "CAMPAIGN_RECOVERY_SOURCE",
    "CAMPAIGN_RECOVERY_INTERACTION_PLAN",
    "CAMPAIGN_RECOVERY_INTERACTION:",
    "CAMPAIGN_RECOVERY_ROUTE_SELECTED",
    "CAMPAIGN_RECOVERY_ROUTE_FAILURE",
    "CAMPAIGN_RECOVERY_CATALOG_SELECTION",
    "CAMPAIGN_RECOVERY_CATALOG_CANDIDATE",
    "CAMPAIGN_RECOVERY_CALCULATION",
    "CAMPAIGN_RECOVERY_PREPARATION_FAILURE",
    "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
    "CAMPAIGN_RECOVERY_PC_WITHDRAWAL",
    "BACKGROUND_ROUTE_JOB",
    "CAMPAIGN_READINESS_CALCULATION_PENDING",
    "READINESS_ROUTE_ANALYSIS",
    "CAMPAIGN_WORLD_ROUTE_DEBUG",
    "recovery_center_candidate_discovery:",
    "recovery_path_start:",
    "recovery_path_complete:",
    "recovery_path_failure:",
    "recovery_center_selected:",
    "ROM_EXECUTION_TRANSITION",
    "CAMPAIGN_ENCOUNTER_STAGE",
    "CAMPAIGN_ENCOUNTER_TILE_SELECTED",
    "CAMPAIGN_ENCOUNTER_ACQUISITION:",
    "CAMPAIGN_ENCOUNTER_PROJECTION",
    "CAMPAIGN_ENCOUNTER_TRAINER_GATE",
    "CAMPAIGN_PREPARATION_CANDIDATE_REJECTED",
    "CAMPAIGN_TRAINER_RISK_DEFERRED",
    "CAMPAIGN_CAPABILITY_COMPLETE",
    "CAMPAIGN_COMPLETED_OWNERSHIP_RELEASE",
    "CAMPAIGN_COMPLETION_WAIT",
    "CAMPAIGN_WILD_CAPTURE_HANDOFF",
    "CAMPAIGN_SHOPPING_PLAN",
    "CAMPAIGN_RESTOCK_OWNERSHIP",
    "CAMPAIGN_RESTOCK_DECISION",
    "CAMPAIGN_POKEBALL_RESTOCK_GATE",
    "CAMPAIGN_RESTOCK_DEFERRED",
    "CAMPAIGN_STATUS_BOUNDARY",
    "CAMPAIGN_FACT_STABILIZED",
    "CAMPAIGN_OBJECTIVE_FRONTIER",
    "CAMPAIGN_FIELD_LEAD_HANDOFF:",
    "CAMPAIGN_TRAINER_STAGING",
    "CAMPAIGN_TRAINER_EXPOSURE",
    "CAMPAIGN_TRAINER_LEAD",
    "CAMPAIGN_TRAINER_LEAD_FAILURE",
    "MENU_ROTATION_PHASE",
    "BATTLE_POST_ROTATION_MENU:",
    "CAMPAIGN_TASK_DISCOVERY",
    "CAMPAIGN_BATTLE_ENDED",
    "BATTLE_START",
    "BATTLE_EVALUATOR",
    "BATTLE_PLANNER",
    "BATTLE_DECISION:",
    "BATTLE_READINESS:",
    "BATTLE_OBJECTIVE_TRANSITION:",
    "BATTLE_PLANNER_FACTS:",
    "BATTLE_EVENT",
    "BATTLE_FORCED_REPLACEMENT",
    "BATTLE_FINISHED",
    "CAMPAIGN_RECOVERY_REQUEST",
    "CAMPAIGN_RECOVERY_DESTINATIONS",
    "CAMPAIGN_RECOVERY_ATTEMPT",
    "CAMPAIGN_RECOVERY_SUCCESS",
    "CAMPAIGN_RECOVERY_FAILURE",
    "CAMPAIGN_RECOVERY_ESCALATED",
    "CAMPAIGN_CAPABILITY_BOUNDARY",
    "CAMPAIGN_WALLY_STATE",
    "WALLY_DIALOGUE_GATE",
    "CAMPAIGN_LIVE_INTERACTION",
    "CAMPAIGN_LIVE_DIALOGUE_INPUT",
    "GATE_LIVE_VALIDATION",
    "GATE_LIVE_FIXTURE",
    "CAMPAIGN_READINESS_RECHECK:",
    "CAMPAIGN_READINESS_RECHECK_DISCARDED",
    "CAMPAIGN_READINESS_RECHECK_FORCED:",
    "CAMPAIGN_READINESS_BOUNDARY",
    "CAMPAIGN_WORLD_ROUTE_DEBUG",
    "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE",
    "READINESS_INPUT:",
    "READINESS_POLICY_BEFORE:",
    "READINESS_POLICY_AFTER:",
    "OPPORTUNISTIC_HEAL_ANALYSIS:",
    "CAMPAIGN_READINESS_RESULT:",
    "CAMPAIGN_READINESS_DECISION:",
    "CAMPAIGN_READINESS_LIFECYCLE:",
    "CAMPAIGN_READINESS_DEFERRED:",
    "CAMPAIGN_READINESS_UNKNOWN_DIAGNOSTIC",
    "TRAINER_CHECKPOINT_RESOLUTION",
    "TRAINER_CATALOG_LIVE",
    "CAMPAIGN_TRAINER_CHECKPOINT",
    "CAMPAIGN_TRAINER_FACTS",
    "CAMPAIGN_TRAINER_READINESS",
    "CAMPAIGN_TRAINER_ROUTE_READINESS",
    "CAMPAIGN_TRAINER_HANDOFF_BLOCKED",
    "CAMPAIGN_TRAINER_PREPARATION_EXCEPTION",
    "CAMPAIGN_TRAINER_PREPARATION_MOUNTED",
    "CAMPAIGN_TRAINER_PREPARATION_COMPLETE",
    "CAMPAIGN_PREPARATION_ROUTE",
    "CAMPAIGN_PREPARATION_MAP_RETRY",
    "CAMPAIGN_PREPARATION_CANDIDATES",
    "CAMPAIGN_PREPARATION_CANDIDATE_SELECTED",
    "CAMPAIGN_PREPARATION_CANDIDATE_FAILURE",
    "CAMPAIGN_PREPARATION_NAVIGATION_SELECTED",
    "CAMPAIGN_PREPARATION_NAVIGATION_RETRY",
    "CAMPAIGN_PREPARATION_TRAINING_LOOP",
    "CAMPAIGN_PREPARATION_TRAINER_SELECTED",
    "CAMPAIGN_PREPARATION_TRAINER_CONTINUATION",
    "CAMPAIGN_MANDATORY_TRAINER_FALLBACK",
    "CAMPAIGN_TRAINER_RISK_PROMOTED",
    "CAMPAIGN_TRAINER_ROUTE_ANALYSIS",
    "OVERWORLD_TRANSITION_STATE",
    "READINESS_SCHEDULER_REFRESH",
    "CAMPAIGN_REFRESH_REQUEST",
    "CAMPAIGN_REFRESH_BOUNDARY",
    "CAMPAIGN_REFRESH_EXCEPTION",
    "CAMPAIGN_STEP_RETURN",
    "CAMPAIGN_PLAN_TRACE",
    "CAMPAIGN_TACTICAL_STEP",
    "AGENT_ROUTE_PLAN",
    "AGENT_AGENT_REPLAN",
    "AGENT_MOVE_BATCH",
    "AGENT_CACHED_ROUTE_DIRECTION_CORRECTION",
    "CAMPAIGN_NAVIGATION_TRACE",
    "CAMPAIGN_NAVIGATION_LOOP",
    "CAMPAIGN_NAVIGATION_LOOP_STOPPED",
    "NAVIGATION_REACH_WARP_DIAGNOSTIC:",
    "AGENT_DIALOGUE_RENDER_RESCUE",
    "AGENT_DIALOGUE_INPUT",
    "EMERALD_CLOCK_INPUT",
    "WARP_INPUT",
    "BATTLE_REPLACEMENT_TRACE:",
    "BATTLE_REPLACEMENT_INPUT:",
    "BATTLE_REPLACEMENT_DISPATCH:",
    "BATTLE_HANDLER_FALLBACK_INPUT:",
    "BATTLE_HANDLER_STRICT_STOP:",
    "BATTLE_HANDLER_TRANSITION_WAIT:",
    "BATTLE_HANDLER_TRANSITION_TIMEOUT:",
    "BATTLE_PROMPT_STATE:",
    "TRAINER_REPLACEMENT_POLICY:",
    "BATTLE_DECISION_DETAIL",
    "BATTLE_ACTION_EVALUATION",
    "BATTLE_ACTION_CHOSEN:",
    "BATTLE_SWITCH_CANDIDATE:",
    "BATTLE_CAPTURE_CANDIDATE:",
    "BATTLE_CAPTURE_DECISION:",
    "NUZLOCKE_SWITCH_EVALUATION:",
    "NUZLOCKE_ACTION_EVALUATION",
)


def diagnostic_print(message: str | Callable[[], str], *, trace: bool = False, prefix: str | None = None) -> None:
    """Print a diagnostic message only when its configured level is enabled.

    ``message`` may be a callable so expensive formatting is skipped entirely
    when the diagnostic level is disabled.  Trace messages are deliberately
    separate from normal debug output because they can be emitted every frame.
    """
    from modules.context import context
    from modules.stutter_trace import background_work_suppressed

    if background_work_suppressed() or not context.debug or (trace and not getattr(context, "debug_trace", False)):
        return
    # Callers with lazy messages should provide the prefix explicitly. This
    # lets a filtered trace return before evaluating expensive state reads and
    # formatting. String messages remain backward compatible and can infer it.
    if trace and prefix is None and isinstance(message, str):
        prefix = message
    if trace and prefix is not None and not prefix.startswith(_DIAGNOSTIC_LOG_WHITELIST):
        return
    rendered = message() if callable(message) else message
    # Keep the raw tuple in diagnostics for machine-readable correlation, but
    # put the ROM map name beside every recognized map ID for operators.
    rendered = format_map_ids_in_text(rendered)
    if trace and prefix is None and not rendered.startswith(_DIAGNOSTIC_LOG_WHITELIST):
        return
    console.print(rendered)
    # Lifecycle/path-boundary records must reach the pipe before a synchronous
    # operation (or KeyboardInterrupt) can prevent the normal frame flush.
    if trace:
        file = getattr(console, "file", None)
        if file is not None:
            file.flush()


def print_always(message: str | Callable[[], str]) -> None:
    """Print a rare, decisive diagnostic regardless of the debug flags.

    ``diagnostic_print`` is silent unless ``--debug`` or ``--debug-trace`` is
    given, even for ``trace=False``.  A few records explain a visible pause or
    a swallowed failure (a failed background search, an exception the
    controller turned into a retry, a controller that has gone idle).  Without
    them an ordinary run shows only "failed; retrying" with no cause, so they
    are written unconditionally, like the pathfinding and cancellation panels.
    Callers must rate limit anything that can repeat.
    """

    rendered = message() if callable(message) else message
    console.print(format_map_ids_in_text(rendered))
    file = getattr(console, "file", None)
    if file is not None:
        file.flush()


def profile_print(message: str | Callable[[], str], *, every: int = 1) -> None:
    """Print performance output under the independent performance gate."""
    from modules.context import context
    from modules.profiler import count, timing, should_emit_output

    if not getattr(context, "debug_profile", False):
        return
    if not should_emit_output(every):
        return
    started_at = perf_counter_ns()
    console.print(message() if callable(message) else message)
    timing("profile_output", started_at)
    count("profile_output_calls")
