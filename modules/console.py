from collections.abc import Callable
from time import perf_counter_ns
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

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
    "CAMPAIGN_RECOVERY_STAGE",
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
    "CAMPAIGN_WILD_CAPTURE_HANDOFF",
    "CAMPAIGN_STATUS_BOUNDARY",
    "CAMPAIGN_TASK_DISCOVERY",
    "CAMPAIGN_BATTLE_ENDED",
    "CAMPAIGN_CAPABILITY_BOUNDARY",
    "CAMPAIGN_WALLY_STATE",
    "WALLY_DIALOGUE_GATE",
    "CAMPAIGN_LIVE_INTERACTION",
    "CAMPAIGN_LIVE_DIALOGUE_INPUT",
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
    "READINESS_SCHEDULER_REFRESH",
    "CAMPAIGN_REFRESH_REQUEST",
    "CAMPAIGN_REFRESH_BOUNDARY",
    "CAMPAIGN_REFRESH_EXCEPTION",
    "CAMPAIGN_PLAN_TRACE",
    "CAMPAIGN_TACTICAL_STEP",
    "AGENT_ROUTE_PLAN",
    "AGENT_AGENT_REPLAN",
    "AGENT_MOVE_BATCH",
    "AGENT_CACHED_ROUTE_DIRECTION_CORRECTION",
    "CAMPAIGN_NAVIGATION_TRACE",
    "AGENT_DIALOGUE_RENDER_RESCUE",
    "AGENT_DIALOGUE_INPUT",
    "WARP_INPUT",
)


def diagnostic_print(message: str | Callable[[], str], *, trace: bool = False, prefix: str | None = None) -> None:
    """Print a diagnostic message only when its configured level is enabled.

    ``message`` may be a callable so expensive formatting is skipped entirely
    when the diagnostic level is disabled.  Trace messages are deliberately
    separate from normal debug output because they can be emitted every frame.
    """
    from modules.context import context

    if not context.debug or (trace and not getattr(context, "debug_trace", False)):
        return
    # Callers with lazy messages should provide the prefix explicitly. This
    # lets a filtered trace return before evaluating expensive state reads and
    # formatting. String messages remain backward compatible and can infer it.
    if trace and prefix is None and isinstance(message, str):
        prefix = message
    if trace and prefix is not None and not prefix.startswith(_DIAGNOSTIC_LOG_WHITELIST):
        return
    rendered = message() if callable(message) else message
    if trace and prefix is None and not rendered.startswith(_DIAGNOSTIC_LOG_WHITELIST):
        return
    console.print(rendered)
    # Lifecycle/path-boundary records must reach the pipe before a synchronous
    # operation (or KeyboardInterrupt) can prevent the normal frame flush.
    if trace:
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
