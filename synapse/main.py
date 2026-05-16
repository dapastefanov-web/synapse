"""
main.py — Synapse CLI entry point and interactive REPL.

Architecture
─────────────
The Typer `app` object is the shell entry point registered in pyproject.toml.
Invoking `synapse` with no arguments starts the interactive REPL. The REPL
feeds each line through the Lexer, then the dispatcher acts on each token
sequentially (respecting && chaining order).

Command dispatch rules
───────────────────────
- /stack [fast|balanced|autonomous]   — set active stack; menu if no arg
- /access [trust|no_trust]            — set access mode; menu if no arg
- /init-docker                        — bootstrap DevContainer
- /index-project                      — embed project files into vector store
- /resume                             — reload last checkpoint and continue
- Any text not starting with / or @  — forwarded to active graph as a prompt
- @macro text                         — macro-expanded prompt

Argument validation
────────────────────
Commands that accept an argument validate it against a fixed set of options.
Any unrecognised value → print_invalid_command() and abort the token chain.
Any text after a command's option that is NOT `&&` is already prevented by
the lexer (it becomes a separate PROMPT token) — the && chain handles that.

HITL loop
──────────
After every graph run the graph pauses at the apply_patches interrupt node.
In trust mode   → patches applied automatically, graph resumed.
In no_trust mode → diff shown, user chooses accept / decline / rewrite.
  accept  → apply patches, resume graph to END.
  decline → discard patches, end session.
  rewrite → inject critique into state, re-run from architect.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from synapse.config.loader import load_all_configs
from synapse.docker_manager import DockerManager
from synapse.graphs.fast_stack import build_fast_graph
from synapse.graphs.balanced_stack import build_balanced_graph
from synapse.graphs.autonomous_stack import build_autonomous_graph
from synapse.lexer import Lexer, LexedToken, TokenType
from synapse.memory.vector_store import VectorStore
from synapse.state import AccessMode, GlobalState, StackType
from synapse.tools.native import NativeToolkit, ToolContext
from synapse.tools.registry import ToolRegistry
from synapse.ui.terminal import HITLResponse, TerminalUI

logger  = logging.getLogger(__name__)
console = Console()
app     = typer.Typer(
    name="synapse",
    help="Multi-agent AI engineering swarm.",
    invoke_without_command=True,
    add_completion=False,
)

# ── Valid options per command ──────────────────────────────────────────────
# These are the ONLY accepted argument values. Anything else is an error.
_COMMAND_OPTIONS: dict[str, list[str]] = {
    "stack":  ["fast", "balanced", "autonomous"],
    "access": ["trust", "no_trust"],
}

# Commands that take no argument at all
_NO_ARG_COMMANDS = {"init-docker", "index-project", "resume"}


# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    """
    Mutable state that persists across REPL iterations for a single session.
    One instance is created at startup and passed through every dispatch call.
    """
    project_root: str
    stack:        StackType  = StackType.BALANCED
    access_mode:  AccessMode = AccessMode.NO_TRUST
    session_id:   str        = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Infrastructure — populated by /init-docker and /index-project
    docker_manager: DockerManager | None = None
    vector_store:   VectorStore   | None = None
    registry:       ToolRegistry  | None = None

    # Graph instances — built lazily on first prompt execution
    _graphs: dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> str:
        return str(Path(self.project_root) / ".synapse" / "sessions.db")

    def get_graph(self, stack: StackType) -> Any:
        """Return the compiled graph for the given stack, building it if needed."""
        key = stack.value
        if key not in self._graphs:
            reg = self._ensure_registry()
            db  = self.db_path
            if stack == StackType.FAST:
                self._graphs[key] = build_fast_graph(reg, db_path=db)
            elif stack == StackType.BALANCED:
                self._graphs[key] = build_balanced_graph(reg, db_path=db)
            else:
                self._graphs[key] = build_autonomous_graph(reg, db_path=db)
        return self._graphs[key]

    def _ensure_registry(self) -> ToolRegistry:
        if self.registry is None:
            self.registry = _build_registry(self)
        return self.registry


# ═══════════════════════════════════════════════════════════════════════════
# Registry construction
# ═══════════════════════════════════════════════════════════════════════════

def _build_registry(session: SessionState) -> ToolRegistry:
    """
    Build a ToolRegistry for this session, loading:
      1. All six native tools.
      2. Global dynamic tools from ~/.config/synapse/tools/
      3. Project-local dynamic tools from .synapse/tools/
      4. MCP servers defined in agents.yaml (started as subprocesses).
    """
    ctx = ToolContext(
        project_root=session.project_root,
        vector_store=session.vector_store,
        firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY"),
        embedding_model="text-embedding-3-small",
    )
    if session.docker_manager is not None:
        # Inject the Docker client so run_command works
        try:
            import docker
            ctx.docker_client = docker.from_env()
        except Exception:
            pass

    toolkit = NativeToolkit(ctx)
    reg     = ToolRegistry()
    reg.load_native_tools(toolkit)

    # Global user tools
    global_tools = Path.home() / ".config" / "synapse" / "tools"
    reg.scan_dynamic_tools(global_tools)

    # Project-local tools
    local_tools = Path(session.project_root) / ".synapse" / "tools"
    reg.scan_dynamic_tools(local_tools)

    return reg


# ═══════════════════════════════════════════════════════════════════════════
# Graph execution and HITL loop
# ═══════════════════════════════════════════════════════════════════════════

def _build_initial_state(prompt: str, session: SessionState) -> GlobalState:
    return {
        "messages":            [],
        "stack":               session.stack,
        "access_mode":         session.access_mode,
        "user_prompt":         prompt,
        "tasks":               [],
        "patches":             [],
        "project_root":        session.project_root,
        "scratchpad":          "",
        "final_summary":       None,
        "human_critique":      None,
        "session_id":          session.session_id,
        "debugger_verdict":    None,
        "debugger_iterations": 0,
    }


async def _run_graph_until_hitl(
    graph:    Any,
    state:    GlobalState,
    config:   dict,
    ui:       TerminalUI,
) -> GlobalState | None:
    """
    Stream graph events until the apply_patches interrupt fires.
    Prints node start/complete lines as events arrive.
    Returns the final state snapshot, or None on error.
    """
    try:
        final_state = None
        async for event in graph.astream(state, config=config):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    continue
                ui.print_node_complete(node_name)
                final_state = node_output
        return final_state
    except Exception as exc:
        ui.error(f"Graph execution error: {exc}")
        logger.exception("Graph execution failed")
        return None


async def _handle_hitl(
    graph:   Any,
    state:   GlobalState,
    config:  dict,
    session: SessionState,
    ui:      TerminalUI,
) -> None:
    """
    The full Human-in-the-Loop loop.

    In trust mode: apply patches and resume immediately.
    In no_trust mode: show diff, collect decision, act.

    The rewrite path loops — after injecting the critique it re-runs the
    entire graph from scratch (not just the architect) so the new plan
    is built with the full critique in context from the start.
    """
    patches = state.get("patches", [])

    # ── Trust mode: auto-apply ─────────────────────────────────────────────
    if session.access_mode == AccessMode.TRUST:
        if session.docker_manager and patches:
            applied = await session.docker_manager.apply_patches(patches)
            ui.success(f"Applied {len(applied)} patch(es) automatically.")
        else:
            ui.warning("No Docker manager available — patches not written to disk.")

        # Resume past the interrupt to reach END
        async for _ in graph.astream(None, config=config):
            pass

        _print_final_summary(state, ui)
        return

    # ── No-trust mode: HITL prompt ─────────────────────────────────────────
    response = ui.present_hitl_prompt(patches)

    if response.is_decline:
        ui.warning("Session ended. No files were modified.")
        return

    if response.is_accept:
        if session.docker_manager and patches:
            applied = await session.docker_manager.apply_patches(patches)
            ui.success(f"Applied {len(applied)} patch(es).")
        else:
            ui.warning("No Docker manager — patches not written to disk.")
        # Resume past the interrupt
        async for _ in graph.astream(None, config=config):
            pass
        _print_final_summary(state, ui)
        return

    if response.is_rewrite:
        # Build a fresh state with the critique injected, then re-run
        new_state = _build_initial_state(state["user_prompt"], session)
        new_state["human_critique"] = response.critique
        new_state["user_prompt"] = (
            f"{state['user_prompt']}\n\n"
            f"CRITIQUE FROM PREVIOUS ATTEMPT:\n{response.critique}"
        )
        ui.info("Re-running with critique...")
        await _execute_prompt(new_state["user_prompt"], session, ui)


async def _execute_prompt(
    prompt:  str,
    session: SessionState,
    ui:      TerminalUI,
) -> None:
    """
    The core execution path: build state → run graph → handle HITL.
    Called both for new prompts and after a Rewrite decision.
    """
    ui.rule(f"Running [{session.stack.value}] stack")
    ui.print_session_info(session.stack, session.access_mode, session.project_root)

    graph  = session.get_graph(session.stack)
    state  = _build_initial_state(prompt, session)
    config = {"configurable": {"thread_id": session.session_id}}

    final_state = await _run_graph_until_hitl(graph, state, config, ui)
    if final_state is None:
        return

    # Merge the final node output back into our state dict for HITL
    merged = {**state, **final_state}
    await _handle_hitl(graph, merged, config, session, ui)


def _print_final_summary(state: GlobalState, ui: TerminalUI) -> None:
    summary        = state.get("final_summary") or "Run complete."
    files_modified = [p.file_path for p in state.get("patches", [])]
    ui.print_summary(summary, files_modified)


# ═══════════════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════════════

def _validate_command_arg(
    command:  str,
    argument: str | None,
    ui:       TerminalUI,
) -> bool:
    """
    Return True if the argument is valid for this command, False otherwise.
    Prints the error and valid options on failure.

    Commands in _NO_ARG_COMMANDS must have argument=None.
    Commands in _COMMAND_OPTIONS must have argument in their options list.
    """
    if command in _NO_ARG_COMMANDS:
        if argument is not None:
            ui.print_invalid_command(command, argument)
            return False
        return True

    if command in _COMMAND_OPTIONS:
        if argument is None:
            # No argument — caller will show the selection menu
            return True
        if argument not in _COMMAND_OPTIONS[command]:
            ui.print_invalid_command(command, argument)
            return False
        return True

    # Unknown command entirely
    ui.error(f"Unknown command: /{command}")
    return False


async def _dispatch_command(
    token:   LexedToken,
    session: SessionState,
    ui:      TerminalUI,
) -> bool:
    """
    Execute one COMMAND token. Returns False if the token chain should abort
    (e.g. invalid argument), True to continue to the next token.
    """
    command  = token.command
    argument = token.argument

    if not _validate_command_arg(command, argument, ui):
        return False

    # /stack
    if command == "stack":
        if argument is None:
            session.stack = ui.prompt_stack_selection()
        else:
            session.stack = StackType(argument)
        ui.success(f"Stack set to [bold]{session.stack.value}[/bold].")
        # Invalidate cached graphs so the new stack is used
        session._graphs.clear()
        return True

    # /access
    if command == "access":
        if argument is None:
            session.access_mode = ui.prompt_access_selection()
        else:
            session.access_mode = AccessMode(argument)
        ui.success(f"Access mode set to [bold]{session.access_mode.value}[/bold].")
        return True

    # /init-docker
    if command == "init-docker":
        ui.info("Initialising DevContainer...")
        dm = DockerManager(session.project_root)
        session.docker_manager = dm
        try:
            msg = await dm.init_docker()
            await dm.start_container()
            # Rebuild registry so run_command gets the live Docker client
            session.registry = None
            session._graphs.clear()
            ui.success(msg)
        except Exception as exc:
            ui.error(f"Docker init failed: {exc}")
        return True

    # /index-project
    if command == "index-project":
        await _handle_index_project(session, ui)
        return True

    # /resume
    if command == "resume":
        await _handle_resume(session, ui)
        return True

    ui.error(f"Unhandled command: /{command}")
    return False


async def _handle_index_project(session: SessionState, ui: TerminalUI) -> None:
    """
    Walk the project tree, embed each source file, and store in the
    VectorStore at .synapse/memory.db. Only Python, JS, TS, Rust,
    Go, and common config files are indexed to keep the store focused.
    """
    _INDEXABLE_SUFFIXES = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
        ".java", ".c", ".cpp", ".h", ".md", ".yaml", ".yml", ".toml",
    }
    _SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".synapse"}

    db_path = Path(session.project_root) / ".synapse" / "memory.db"
    store   = VectorStore(db_path=str(db_path), embedding_dim=1536)
    store.initialize()
    session.vector_store = store

    # Rebuild registry with the live store
    session.registry = None
    session._graphs.clear()

    root    = Path(session.project_root)
    indexed = 0

    ui.info("Indexing project files...")

    try:
        import litellm
    except ImportError:
        ui.error("litellm not installed — cannot generate embeddings.")
        return

    for path in sorted(root.rglob("*")):
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
        if path.suffix not in _INDEXABLE_SUFFIXES:
            continue
        if not path.is_file():
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue

            resp   = await litellm.aembedding(model="text-embedding-3-small", input=[content])
            vector = resp.data[0]["embedding"]
            rel    = str(path.relative_to(root))

            store.add_document(
                doc_id=rel, content=content,
                embedding=vector, file_path=rel,
                metadata={"suffix": path.suffix},
            )
            indexed += 1
        except Exception as exc:
            logger.debug("Could not index %s: %s", path, exc)

    ui.success(f"Indexed {indexed} file(s) into vector store.")


async def _handle_resume(session: SessionState, ui: TerminalUI) -> None:
    """
    Reload the most recent checkpoint from the session database and resume
    execution. The checkpointer stores state under the session's thread_id.
    """
    ui.info(f"Attempting to resume session '{session.session_id}'...")

    graph  = session.get_graph(session.stack)
    config = {"configurable": {"thread_id": session.session_id}}

    try:
        # Retrieve the checkpoint state from the SQLite checkpointer
        checkpoint = graph.get_state(config)
        if checkpoint is None or not checkpoint.values:
            ui.warning("No checkpoint found for this session.")
            return

        state = checkpoint.values
        ui.success("Checkpoint loaded. Resuming from last interrupt.")
        ui.print_node_start("apply_patches")
        await _handle_hitl(graph, state, config, session, ui)
    except Exception as exc:
        ui.error(f"Resume failed: {exc}")
        logger.exception("Resume error")


# ═══════════════════════════════════════════════════════════════════════════
# REPL
# ═══════════════════════════════════════════════════════════════════════════

async def _repl(session: SessionState, ui: TerminalUI) -> None:
    """
    The main Read-Eval-Print Loop. Reads one line at a time, tokenises it,
    and dispatches each token sequentially. Ctrl-C exits cleanly.
    """
    configs    = load_all_configs(project_root=session.project_root)
    lexer      = Lexer(configs["cli_config"])

    ui.print_banner()
    ui.print_session_info(session.stack, session.access_mode, session.project_root)
    console.print("[dim]Type a prompt, /command, or @macro. Ctrl-C to exit.[/dim]\n")

    while True:
        try:
            raw = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue

        tokens = lexer.tokenize(raw)
        if not tokens:
            continue

        abort = False
        for token in tokens:
            if abort:
                break

            if token.token_type == TokenType.COMMAND:
                ok = await _dispatch_command(token, session, ui)
                if not ok:
                    abort = True

            elif token.token_type in (TokenType.PROMPT, TokenType.MACRO):
                content = token.content or ""
                if content.strip():
                    await _execute_prompt(content.strip(), session, ui)


# ═══════════════════════════════════════════════════════════════════════════
# Typer entry point
# ═══════════════════════════════════════════════════════════════════════════

@app.callback()
def main(
    ctx:     typer.Context = typer.Argument(default=None),
    project: str = typer.Option(
        ".",
        "--project", "-p",
        help="Path to the project root directory.",
    ),
    stack: str = typer.Option(
        "balanced",
        "--stack", "-s",
        help="Initial stack: fast | balanced | autonomous",
    ),
    access: str = typer.Option(
        "no_trust",
        "--access", "-a",
        help="Initial access mode: trust | no_trust",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Start the Synapse interactive shell.

    Examples:
        synapse
        synapse --project ~/my-api --stack autonomous --access trust
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Validate CLI options before entering the REPL
    ui = TerminalUI(verbose=verbose)

    if stack not in _COMMAND_OPTIONS["stack"]:
        ui.print_invalid_command("stack", stack)
        raise typer.Exit(code=1)

    if access not in _COMMAND_OPTIONS["access"]:
        ui.print_invalid_command("access", access)
        raise typer.Exit(code=1)

    project_root = str(Path(project).resolve())

    # Ensure .synapse directory exists for the session DB
    Path(project_root, ".synapse").mkdir(parents=True, exist_ok=True)

    session = SessionState(
        project_root=project_root,
        stack=StackType(stack),
        access_mode=AccessMode(access),
    )

    asyncio.run(_repl(session, ui))