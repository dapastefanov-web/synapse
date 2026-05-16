"""
ui/terminal.py — Rich terminal dashboard for Synapse.

Responsibilities
─────────────────
1. Live execution tree — updates in real time as graph nodes fire.
2. Syntax-highlighted diff viewer — shows patches before HITL decision.
3. Accept / Decline / Rewrite prompt — the human-in-the-loop gate.
4. Stack and access mode selection menus — used when no inline argument given.
5. General status printing (info, success, warning, error).

Options in menus and inline CLI arguments use identical snake_case strings.
Example: /access no_trust  ←→  menu option "no_trust"
"""

from __future__ import annotations

import sys
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from synapse.state import AccessMode, StackType


console = Console()


# ── HITL response type ─────────────────────────────────────────────────────

class HITLResponse:
    """Returned by present_hitl_prompt() to the main dispatcher."""
    ACCEPT  = "accept"
    DECLINE = "decline"
    REWRITE = "rewrite"

    def __init__(self, action: str, critique: str | None = None) -> None:
        self.action   = action
        self.critique = critique  # populated when action == REWRITE

    @property
    def is_accept(self)  -> bool: return self.action == self.ACCEPT
    @property
    def is_decline(self) -> bool: return self.action == self.DECLINE
    @property
    def is_rewrite(self) -> bool: return self.action == self.REWRITE


# ── Core display helpers ───────────────────────────────────────────────────

class TerminalUI:
    """
    Stateless UI helper. All methods write directly to the console.
    The main dispatcher in main.py constructs one instance per session
    and calls methods as graph nodes complete.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    # ── Status messages ────────────────────────────────────────────────────

    def info(self, message: str) -> None:
        console.print(f"[cyan]ℹ[/cyan]  {message}")

    def success(self, message: str) -> None:
        console.print(f"[green]✓[/green]  {message}")

    def warning(self, message: str) -> None:
        console.print(f"[yellow]⚠[/yellow]  {message}")

    def error(self, message: str) -> None:
        err_console = Console(stderr=True)
        err_console.print(f"[red]✗[/red]  {message}")

    def rule(self, title: str = "") -> None:
        console.rule(f"[bold]{title}[/bold]" if title else "")

    # ── Session header ─────────────────────────────────────────────────────

    def print_banner(self) -> None:
        console.print(Panel(
            Text("Synapse", style="bold cyan", justify="center"),
            subtitle="[dim]multi-agent AI engineering swarm[/dim]",
            border_style="cyan",
            padding=(0, 4),
        ))

    def print_session_info(
        self,
        stack:       StackType,
        access_mode: AccessMode,
        project:     str,
    ) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="bold")
        table.add_row("Stack",   stack.value)
        table.add_row("Access",  access_mode.value)
        table.add_row("Project", project)
        console.print(Panel(table, border_style="dim", padding=(0, 1)))

    # ── Execution tree ─────────────────────────────────────────────────────

    def print_node_start(self, node_name: str) -> None:
        console.print(f"  [cyan]▶[/cyan] [bold]{node_name}[/bold]")

    def print_node_complete(self, node_name: str, detail: str = "") -> None:
        suffix = f"  [dim]{detail}[/dim]" if detail else ""
        console.print(f"  [green]✓[/green] [bold]{node_name}[/bold]{suffix}")

    def print_node_failed(self, node_name: str, reason: str = "") -> None:
        suffix = f"  [dim]{reason}[/dim]" if reason else ""
        console.print(f"  [red]✗[/red] [bold]{node_name}[/bold]{suffix}")

    def print_retry(self, node_name: str, iteration: int, max_iter: int) -> None:
        console.print(
            f"  [yellow]↺[/yellow] [bold]{node_name}[/bold] "
            f"[dim]retry {iteration}/{max_iter}[/dim]"
        )

    def print_parallel_dispatch(self, task_count: int) -> None:
        console.print(
            f"  [cyan]⇉[/cyan]  Dispatching [bold]{task_count}[/bold] parallel branch(es)"
        )

    def print_branch_result(self, task_id: str, status: str) -> None:
        colour = "green" if status == "success" else "red"
        console.print(f"    [{colour}]{'✓' if status == 'success' else '✗'}[/{colour}]  {task_id}")

    # ── Diff viewer ────────────────────────────────────────────────────────

    def print_diff(self, patches: list[Any]) -> None:
        """
        Render each patch's unified diff with syntax highlighting.
        Called by the main dispatcher immediately before present_hitl_prompt().
        """
        if not patches:
            self.warning("No patches to display.")
            return

        self.rule("Proposed Changes")

        for patch in patches:
            console.print(
                f"\n[bold]File:[/bold] [cyan]{patch.file_path}[/cyan]"
                f"  [dim]{patch.description}[/dim]"
            )
            syntax = Syntax(
                patch.unified_diff,
                "diff",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
            console.print(Panel(syntax, border_style="dim", padding=(0, 1)))

        self.rule()

    # ── HITL prompt ────────────────────────────────────────────────────────

    def present_hitl_prompt(self, patches: list[Any]) -> HITLResponse:
        """
        Show the diff and present the Accept / Decline / Rewrite decision.

        Options match their CLI snake_case equivalents exactly:
          accept   — apply patches and continue
          decline  — discard patches and end session
          rewrite  — collect critique and loop back to the Architect

        Returns a HITLResponse the dispatcher acts on.
        """
        self.print_diff(patches)

        console.print(
            "\n[bold]Review complete.[/bold]  "
            "[green]accept[/green]  "
            "[red]decline[/red]  "
            "[yellow]rewrite[/yellow]\n"
        )

        while True:
            choice = Prompt.ask(
                "[bold]Decision[/bold]",
                choices=["accept", "decline", "rewrite"],
                default="accept",
            ).strip().lower()

            if choice == HITLResponse.ACCEPT:
                return HITLResponse(HITLResponse.ACCEPT)

            if choice == HITLResponse.DECLINE:
                self.warning("Patches declined. No files were modified.")
                return HITLResponse(HITLResponse.DECLINE)

            if choice == HITLResponse.REWRITE:
                critique = Prompt.ask("\n[bold]Critique[/bold]").strip()
                if not critique:
                    self.warning("Critique cannot be empty. Try again.")
                    continue
                self.info(f"Routing back to Architect with critique.")
                return HITLResponse(HITLResponse.REWRITE, critique=critique)

    # ── Stack selection menu ───────────────────────────────────────────────

    def prompt_stack_selection(self) -> StackType:
        """
        Interactive menu shown when the user runs /stack without an argument.
        Option strings are identical to StackType values and CLI arguments.
        """
        self.rule("Select Stack")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="dim")
        table.add_row("fast",       "Low-latency, single-file changes")
        table.add_row("balanced",   "Structured planning + static review")
        table.add_row("autonomous", "Parallel multi-file + live test execution")
        console.print(table)
        console.print()

        choice = Prompt.ask(
            "[bold]Stack[/bold]",
            choices=["fast", "balanced", "autonomous"],
            default="balanced",
        )
        return StackType(choice)

    # ── Access mode selection menu ─────────────────────────────────────────

    def prompt_access_selection(self) -> AccessMode:
        """
        Interactive menu shown when the user runs /access without an argument.
        Option strings are identical to AccessMode values and CLI arguments.
        """
        self.rule("Select Access Mode")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="dim")
        table.add_row("trust",    "Auto-apply patches, receive summary")
        table.add_row("no_trust", "Review diff before any file is written")
        console.print(table)
        console.print()

        choice = Prompt.ask(
            "[bold]Access mode[/bold]",
            choices=["trust", "no_trust"],
            default="no_trust",
        )
        return AccessMode(choice)

    # ── Summary display ────────────────────────────────────────────────────

    def print_summary(self, summary: str, files_modified: list[str]) -> None:
        """Render the Summariser's output as the final terminal report."""
        self.rule("Summary")
        console.print(summary)

        if files_modified:
            console.print()
            console.print("[bold]Files modified:[/bold]")
            for f in files_modified:
                console.print(f"  [cyan]{f}[/cyan]")

        self.rule()

    # ── Error display ──────────────────────────────────────────────────────

    def print_invalid_command(self, command: str, argument: str | None) -> None:
        """
        Called by the dispatcher when a command receives an unrecognised option.
        The valid options for known commands are listed so the user can correct.
        """
        _VALID_OPTIONS: dict[str, list[str]] = {
            "stack":  ["fast", "balanced", "autonomous"],
            "access": ["trust", "no_trust"],
        }
        console.print(
            f"[red]✗[/red]  Invalid argument [bold]{argument!r}[/bold] "
            f"for command [bold]/{command}[/bold]."
        )
        if command in _VALID_OPTIONS:
            opts = "  |  ".join(
                f"[cyan]{o}[/cyan]" for o in _VALID_OPTIONS[command]
            )
            console.print(f"   Valid options: {opts}")