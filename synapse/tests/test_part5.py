"""
tests/test_part5.py — Tests for the terminal UI and CLI dispatch layer.

Two strategies used:
────────────────────
Fast Unit Tests (pytest + pytest-mock):
  _validate_command_arg, _dispatch_command, SessionState, _build_initial_state.
  No real processes, no LLM calls, no Docker.

E2E Interactive Tests (pexpect):
  Spawn isolated Python helper scripts that call TerminalUI methods directly.
  Tests the real Rich prompt rendering and keystroke handling in a PTY.
  Scoped to the interactive UI only — LLM calls and graph execution are
  explicitly excluded because they are non-deterministic and require API keys.
  Those paths are covered by the mocked fast tests in test_part3.py and test_part4.py.

Run all:
    pytest tests/test_part5.py -v

Run only fast tests:
    pytest tests/test_part5.py -v -m "not e2e"

Run only E2E tests:
    pytest tests/test_part5.py -v -m e2e
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# pexpect is only needed for E2E tests — import conditionally so the fast
# tests still run in environments where pexpect is not installed.
try:
    import pexpect
    PEXPECT_AVAILABLE = True
except ImportError:
    PEXPECT_AVAILABLE = False

from synapse.main import (
    SessionState,
    _build_initial_state,
    _COMMAND_OPTIONS,
    _NO_ARG_COMMANDS,
    _validate_command_arg,
    _dispatch_command,
)
from synapse.state import AccessMode, StackType
from synapse.ui.terminal import HITLResponse, TerminalUI
from synapse.tools.registry import ToolRegistry


# ── Pytest marks ───────────────────────────────────────────────────────────
pytestmark_e2e = pytest.mark.e2e


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_session(tmp_path) -> SessionState:
    return SessionState(
        project_root=str(tmp_path),
        stack=StackType.BALANCED,
        access_mode=AccessMode.NO_TRUST,
        session_id="test-session",
    )


def _make_ui() -> TerminalUI:
    return TerminalUI(verbose=False)


def _make_fake_patch(file_path: str = "src/app.py"):
    """Return a minimal object that looks like a FilePatch to the UI."""
    p             = MagicMock()
    p.file_path   = file_path
    p.description = "Test patch"
    p.unified_diff = (
        f"--- a/{file_path}\n+++ b/{file_path}\n"
        "@@ -1,3 +1,3 @@\n"
        " def hello():\n"
        "-    return 'old'\n"
        "+    return 'new'\n"
    )
    return p


def _write_helper_script(tmp_path: Path, body: str) -> Path:
    """
    Write a small Python script to tmp_path/helper.py.
    The script imports TerminalUI and exercises one specific flow.
    Used by pexpect tests to spawn an isolated interactive process.
    """
    script = tmp_path / "helper.py"
    # Prepend sys.path manipulation so the script finds the synapse package
    # regardless of where pytest is run from.
    header = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent.parent)!r})
    """)
    script.write_text(header + textwrap.dedent(body))
    return script


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Fast Unit Tests: _validate_command_arg
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateCommandArg:
    """
    _validate_command_arg is the gatekeeper for all command dispatch.
    Every invalid input must return False and print a helpful message.
    Every valid input must return True silently.
    """

    @pytest.fixture
    def ui(self, capsys):
        return _make_ui()

    # /stack
    def test_stack_valid_arguments_pass(self, ui):
        for opt in ["fast", "balanced", "autonomous"]:
            assert _validate_command_arg("stack", opt, ui) is True

    def test_stack_invalid_argument_returns_false(self, ui):
        assert _validate_command_arg("stack", "turbo", ui) is False

    def test_stack_none_argument_passes(self, ui):
        # No argument means show the selection menu — that is valid.
        assert _validate_command_arg("stack", None, ui) is True

    # /access
    def test_access_valid_trust_passes(self, ui):
        assert _validate_command_arg("access", "trust", ui) is True

    def test_access_valid_no_trust_passes(self, ui):
        assert _validate_command_arg("access", "no_trust", ui) is True

    def test_access_invalid_argument_returns_false(self, ui):
        # The old `*no_trust` syntax must be rejected — only `no_trust` is valid.
        assert _validate_command_arg("access", "*no_trust", ui) is False

    def test_access_none_argument_passes(self, ui):
        assert _validate_command_arg("access", None, ui) is True

    # No-argument commands
    def test_init_docker_with_no_arg_passes(self, ui):
        assert _validate_command_arg("init-docker", None, ui) is True

    def test_init_docker_with_any_arg_returns_false(self, ui):
        assert _validate_command_arg("init-docker", "something", ui) is False

    def test_resume_with_no_arg_passes(self, ui):
        assert _validate_command_arg("resume", None, ui) is True

    def test_resume_with_arg_returns_false(self, ui):
        assert _validate_command_arg("resume", "session_abc", ui) is False

    # Unknown command
    def test_unknown_command_returns_false(self, ui):
        assert _validate_command_arg("nonexistent", None, ui) is False

    def test_unknown_command_with_arg_returns_false(self, ui):
        assert _validate_command_arg("nonexistent", "opt", ui) is False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Fast Unit Tests: _dispatch_command
# ═══════════════════════════════════════════════════════════════════════════

class TestDispatchCommand:
    """
    _dispatch_command returns True to continue the token chain and
    False to abort it. We mock the UI and session to isolate dispatch logic.
    """

    @pytest.fixture
    def ui(self):
        ui = MagicMock(spec=TerminalUI)
        # prompt methods return sensible defaults
        ui.prompt_stack_selection.return_value  = StackType.FAST
        ui.prompt_access_selection.return_value = AccessMode.TRUST
        return ui

    def _make_token(self, command: str, argument: str | None):
        from synapse.lexer import LexedToken, TokenType
        return LexedToken(
            token_type=TokenType.COMMAND,
            raw=f"/{command}" + (f" {argument}" if argument else ""),
            command=command,
            argument=argument,
        )

    async def test_stack_with_valid_arg_updates_session(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("stack", "fast")
        result  = await _dispatch_command(token, session, ui)
        assert result is True
        assert session.stack == StackType.FAST

    async def test_stack_with_invalid_arg_returns_false(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("stack", "invalid_stack")
        result  = await _dispatch_command(token, session, ui)
        assert result is False
        # Stack must be unchanged
        assert session.stack == StackType.BALANCED

    async def test_stack_without_arg_shows_menu(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("stack", None)
        await _dispatch_command(token, session, ui)
        ui.prompt_stack_selection.assert_called_once()

    async def test_access_with_valid_arg_updates_session(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("access", "trust")
        result  = await _dispatch_command(token, session, ui)
        assert result is True
        assert session.access_mode == AccessMode.TRUST

    async def test_access_with_no_trust_updates_session(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("access", "no_trust")
        result  = await _dispatch_command(token, session, ui)
        assert result is True
        assert session.access_mode == AccessMode.NO_TRUST

    async def test_access_with_invalid_arg_returns_false(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("access", "superuser")
        result  = await _dispatch_command(token, session, ui)
        assert result is False

    async def test_access_without_arg_shows_menu(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("access", None)
        await _dispatch_command(token, session, ui)
        ui.prompt_access_selection.assert_called_once()

    async def test_stack_change_clears_graph_cache(self, tmp_path, ui):
        session = _make_session(tmp_path)
        # Pre-populate the graph cache with a fake entry
        session._graphs["balanced"] = MagicMock()
        token  = self._make_token("stack", "fast")
        await _dispatch_command(token, session, ui)
        assert session._graphs == {}

    async def test_init_docker_with_arg_returns_false(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("init-docker", "extra_arg")
        result  = await _dispatch_command(token, session, ui)
        assert result is False

    async def test_resume_with_arg_returns_false(self, tmp_path, ui):
        session = _make_session(tmp_path)
        token   = self._make_token("resume", "session_id_123")
        result  = await _dispatch_command(token, session, ui)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Fast Unit Tests: SessionState
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionState:

    def test_db_path_is_inside_synapse_dir(self, tmp_path):
        session = _make_session(tmp_path)
        assert ".synapse" in session.db_path
        assert "sessions.db" in session.db_path

    def test_get_graph_caches_result(self, tmp_path, mocker):
        mocker.patch("synapse.main.build_balanced_graph", return_value=MagicMock())
        session = _make_session(tmp_path)
        g1 = session.get_graph(StackType.BALANCED)
        g2 = session.get_graph(StackType.BALANCED)
        assert g1 is g2

    def test_get_graph_builds_correct_type_per_stack(self, tmp_path, mocker):
        mock_fast      = mocker.patch("synapse.main.build_fast_graph",       return_value=MagicMock())
        mock_balanced  = mocker.patch("synapse.main.build_balanced_graph",    return_value=MagicMock())
        mock_autonomous = mocker.patch("synapse.main.build_autonomous_graph", return_value=MagicMock())

        session = _make_session(tmp_path)
        session.get_graph(StackType.FAST)
        session.get_graph(StackType.BALANCED)
        session.get_graph(StackType.AUTONOMOUS)

        mock_fast.assert_called_once()
        mock_balanced.assert_called_once()
        mock_autonomous.assert_called_once()

    def test_clearing_graphs_forces_rebuild(self, tmp_path, mocker):
        mock_build = mocker.patch("synapse.main.build_balanced_graph", return_value=MagicMock())
        session    = _make_session(tmp_path)
        session.get_graph(StackType.BALANCED)
        session._graphs.clear()
        session.get_graph(StackType.BALANCED)
        assert mock_build.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Fast Unit Tests: _build_initial_state
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildInitialState:

    def test_all_required_keys_present(self, tmp_path):
        session = _make_session(tmp_path)
        state   = _build_initial_state("Build something", session)
        required = {
            "messages", "stack", "access_mode", "user_prompt", "tasks",
            "patches", "project_root", "scratchpad", "final_summary",
            "human_critique", "session_id", "debugger_verdict",
            "debugger_iterations",
        }
        assert required.issubset(set(state.keys()))

    def test_prompt_stored_correctly(self, tmp_path):
        session = _make_session(tmp_path)
        state   = _build_initial_state("My test prompt", session)
        assert state["user_prompt"] == "My test prompt"

    def test_stack_and_access_mode_from_session(self, tmp_path):
        session             = _make_session(tmp_path)
        session.stack       = StackType.AUTONOMOUS
        session.access_mode = AccessMode.TRUST
        state = _build_initial_state("x", session)
        assert state["stack"]       == StackType.AUTONOMOUS
        assert state["access_mode"] == AccessMode.TRUST

    def test_lists_start_empty(self, tmp_path):
        session = _make_session(tmp_path)
        state   = _build_initial_state("x", session)
        assert state["tasks"]    == []
        assert state["patches"]  == []
        assert state["messages"] == []

    def test_numeric_defaults_are_zero_or_none(self, tmp_path):
        session = _make_session(tmp_path)
        state   = _build_initial_state("x", session)
        assert state["debugger_iterations"] == 0
        assert state["debugger_verdict"]    is None
        assert state["final_summary"]       is None
        assert state["human_critique"]      is None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Fast Unit Tests: HITLResponse
# ═══════════════════════════════════════════════════════════════════════════

class TestHITLResponse:

    def test_accept_properties(self):
        r = HITLResponse(HITLResponse.ACCEPT)
        assert r.is_accept  is True
        assert r.is_decline is False
        assert r.is_rewrite is False
        assert r.critique   is None

    def test_decline_properties(self):
        r = HITLResponse(HITLResponse.DECLINE)
        assert r.is_decline is True
        assert r.is_accept  is False
        assert r.is_rewrite is False

    def test_rewrite_stores_critique(self):
        r = HITLResponse(HITLResponse.REWRITE, critique="Make it async")
        assert r.is_rewrite  is True
        assert r.critique    == "Make it async"

    def test_rewrite_without_critique_is_none(self):
        r = HITLResponse(HITLResponse.REWRITE)
        assert r.critique is None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — E2E Interactive Tests (pexpect)
# ═══════════════════════════════════════════════════════════════════════════
#
# These tests spawn real Python subprocesses in a PTY and interact with
# them via pexpect. They test that:
#   1. Rich actually renders the expected prompt text to a terminal.
#   2. Keyboard input reaches the Prompt.ask() call correctly.
#   3. The accept / decline / rewrite flows complete without errors.
#
# Each test writes a small helper script to tmp_path and spawns it with
# pexpect.spawn(). The helper script does exactly one thing: calls the
# TerminalUI method under test with pre-defined fake data, then prints a
# sentinel line ("SCRIPT_DONE") so the test can confirm clean completion.
#
# IMPORTANT: These tests require pexpect and a working PTY. They will be
# skipped automatically in environments without pexpect or on Windows.
# ═══════════════════════════════════════════════════════════════════════════

_E2E_TIMEOUT = 10  # seconds — generous for slow CI environments


def _spawn(script_path: Path) -> "pexpect.spawn":
    """Spawn a Python script in a PTY with UTF-8 encoding."""
    return pexpect.spawn(
        sys.executable,
        [str(script_path)],
        timeout=_E2E_TIMEOUT,
        encoding="utf-8",
        env={
            # Force Rich to render as a real terminal (not a dumb pipe)
            "TERM":             "xterm-256color",
            "COLUMNS":          "120",
            "LINES":            "40",
            # Pass the Python path so the helper can import synapse
            "PYTHONPATH":       str(Path(__file__).parent.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


@pytest.mark.e2e
@pytest.mark.skipif(not PEXPECT_AVAILABLE, reason="pexpect not installed")
class TestHITLPromptE2E:
    """
    E2E tests for present_hitl_prompt().
    Each test spawns an isolated helper script that calls this method with
    fake patches and interacts with it via pexpect keystrokes.
    """

    def _write_hitl_helper(self, tmp_path: Path) -> Path:
        """
        Write the helper script that presents the HITL prompt and prints
        a sentinel on completion so pexpect knows the flow finished.
        """
        return _write_helper_script(tmp_path, """\
            from unittest.mock import MagicMock
            from synapse.ui.terminal import TerminalUI

            # Build a fake patch object
            patch = MagicMock()
            patch.file_path    = "src/app.py"
            patch.description  = "Replace old with new"
            patch.unified_diff = (
                "--- a/src/app.py\\n+++ b/src/app.py\\n"
                "@@ -1 +1 @@\\n-old\\n+new\\n"
            )

            ui     = TerminalUI()
            result = ui.present_hitl_prompt([patch])
            print(f"ACTION:{result.action}")
            if result.critique:
                print(f"CRITIQUE:{result.critique}")
            print("SCRIPT_DONE")
        """)

    def test_accept_flow(self, tmp_path):
        """User types 'accept' and presses Enter → file should be applied."""
        script = self._write_hitl_helper(tmp_path)
        child  = _spawn(script)

        try:
            # Wait for the decision prompt to appear
            child.expect(r"Decision")
            child.sendline("accept")
            child.expect(r"ACTION:accept")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(
                f"accept flow timed out.\nBuffer: {child.before!r}"
            )
        except pexpect.EOF:
            pytest.fail(
                f"Process exited unexpectedly.\nBuffer: {child.before!r}"
            )
        finally:
            child.close()

    def test_decline_flow(self, tmp_path):
        """User types 'decline' → no files written, session ends cleanly."""
        script = self._write_hitl_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Decision")
            child.sendline("decline")
            child.expect(r"ACTION:decline")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(
                f"decline flow timed out.\nBuffer: {child.before!r}"
            )
        except pexpect.EOF:
            pytest.fail(
                f"Process exited unexpectedly.\nBuffer: {child.before!r}"
            )
        finally:
            child.close()

    def test_rewrite_flow_with_valid_critique(self, tmp_path):
        """
        User types 'rewrite', is prompted for a critique, enters one,
        and the response carries the critique text back correctly.
        """
        script = self._write_hitl_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Decision")
            child.sendline("rewrite")

            # The UI must ask for the critique text
            child.expect(r"Critique")
            child.sendline("Make it fully async with proper error handling")

            child.expect(r"ACTION:rewrite")
            child.expect(r"CRITIQUE:Make it fully async")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(
                f"rewrite flow timed out.\nBuffer: {child.before!r}"
            )
        except pexpect.EOF:
            pytest.fail(
                f"Process exited unexpectedly.\nBuffer: {child.before!r}"
            )
        finally:
            child.close()

    def test_rewrite_empty_critique_rejected(self, tmp_path):
        """
        If the user submits an empty critique, the UI must reject it
        and re-prompt rather than proceeding with an empty string.
        """
        script = self._write_hitl_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Decision")
            child.sendline("rewrite")

            # First critique attempt: empty
            child.expect(r"Critique")
            child.sendline("")  # empty input

            # The UI must warn and show the Decision prompt again
            # because empty critique loops back to the top of the while loop
            # which re-shows the decision prompt (accept/decline/rewrite)
            child.expect(r"Decision")

            # Now provide a valid path to exit cleanly
            child.sendline("decline")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(
                f"empty critique rejection timed out.\nBuffer: {child.before!r}"
            )
        except pexpect.EOF:
            pytest.fail(
                f"Process exited unexpectedly.\nBuffer: {child.before!r}"
            )
        finally:
            child.close()

    def test_diff_is_displayed_before_prompt(self, tmp_path):
        """
        The file path from the patch must appear in the terminal output
        before the Decision prompt — confirms the diff viewer fires first.
        """
        script = self._write_hitl_helper(tmp_path)
        child  = _spawn(script)

        try:
            # The file path should appear in the diff section
            child.expect(r"src/app\.py")
            # Then the decision prompt follows
            child.expect(r"Decision")
            child.sendline("decline")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(
                f"diff display timed out.\nBuffer: {child.before!r}"
            )
        finally:
            child.close()


@pytest.mark.e2e
@pytest.mark.skipif(not PEXPECT_AVAILABLE, reason="pexpect not installed")
class TestStackMenuE2E:
    """E2E tests for the /stack selection menu."""

    def _write_stack_helper(self, tmp_path: Path) -> Path:
        return _write_helper_script(tmp_path, """\
            from synapse.ui.terminal import TerminalUI
            ui     = TerminalUI()
            result = ui.prompt_stack_selection()
            print(f"SELECTED:{result.value}")
            print("SCRIPT_DONE")
        """)

    def test_selecting_fast_returns_fast(self, tmp_path):
        script = self._write_stack_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Stack")
            child.sendline("fast")
            child.expect(r"SELECTED:fast")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(f"stack fast selection timed out.\nBuffer: {child.before!r}")
        finally:
            child.close()

    def test_selecting_autonomous_returns_autonomous(self, tmp_path):
        script = self._write_stack_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Stack")
            child.sendline("autonomous")
            child.expect(r"SELECTED:autonomous")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(f"stack autonomous selection timed out.\nBuffer: {child.before!r}")
        finally:
            child.close()

    def test_menu_displays_all_three_options(self, tmp_path):
        """Confirm all three stack names are visible before the prompt."""
        script = self._write_stack_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"fast")
            child.expect(r"balanced")
            child.expect(r"autonomous")
            child.sendline("balanced")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(f"stack menu display timed out.\nBuffer: {child.before!r}")
        finally:
            child.close()


@pytest.mark.e2e
@pytest.mark.skipif(not PEXPECT_AVAILABLE, reason="pexpect not installed")
class TestAccessMenuE2E:
    """E2E tests for the /access selection menu."""

    def _write_access_helper(self, tmp_path: Path) -> Path:
        return _write_helper_script(tmp_path, """\
            from synapse.ui.terminal import TerminalUI
            ui     = TerminalUI()
            result = ui.prompt_access_selection()
            print(f"SELECTED:{result.value}")
            print("SCRIPT_DONE")
        """)

    def test_selecting_trust_returns_trust(self, tmp_path):
        script = self._write_access_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Access mode")
            child.sendline("trust")
            child.expect(r"SELECTED:trust")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(f"access trust selection timed out.\nBuffer: {child.before!r}")
        finally:
            child.close()

    def test_selecting_no_trust_returns_no_trust(self, tmp_path):
        script = self._write_access_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"Access mode")
            child.sendline("no_trust")
            child.expect(r"SELECTED:no_trust")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(f"access no_trust selection timed out.\nBuffer: {child.before!r}")
        finally:
            child.close()

    def test_menu_displays_both_options(self, tmp_path):
        script = self._write_access_helper(tmp_path)
        child  = _spawn(script)

        try:
            child.expect(r"trust")
            child.expect(r"no_trust")
            child.sendline("trust")
            child.expect(r"SCRIPT_DONE")
        except pexpect.TIMEOUT:
            pytest.fail(f"access menu display timed out.\nBuffer: {child.before!r}")
        finally:
            child.close()