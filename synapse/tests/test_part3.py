"""
tests/test_part3.py — Fast Unit Tests for the Part 3 agent layer.

Strategy: Fast Unit Tests (pytest + pytest-mock + pytest-asyncio)
──────────────────────────────────────────────────────────────────
Every litellm.acompletion call is replaced with a mock that returns a
pre-scripted response. This confirms the agent pipeline logic — conversation
construction, tool dispatch, validation retry, scratchpad updates — without
spending API credits or requiring network access.

Run with:
    pytest tests/test_part3.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from synapse.agents.base import (
    MAX_VALIDATION_RETRIES,
    _strip_markdown_fences,
    call_agent,
)
from synapse.agents.architect import architect_node, _build_architect_messages
from synapse.agents.coder import coder_node, coder_subgraph_node
from synapse.agents.debugger import debugger_node, debugger_subgraph_node
from synapse.agents.summariser import summariser_node
from synapse.state import (
    AccessMode,
    ArchitectOutput,
    CoderOutput,
    DebuggerAnalysis,
    FilePatch,
    GlobalState,
    StackType,
    SubTask,
    SubTaskState,
    SummaryOutput,
    TaskStatus,
)
from synapse.tools.registry import ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_llm_response(content: str) -> MagicMock:
    """Build a minimal mock LiteLLM ModelResponse with the given text content."""
    msg           = MagicMock()
    msg.content   = content
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": content}

    choice         = MagicMock()
    choice.message = msg

    response         = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call_response(
    tool_name: str,
    arguments: dict,
    call_id: str = "call_abc",
) -> MagicMock:
    """Build a mock response where the LLM requests one tool call."""
    tool_call          = MagicMock()
    tool_call.id       = call_id
    tool_call.function = MagicMock()
    tool_call.function.name      = tool_name
    tool_call.function.arguments = json.dumps(arguments)

    msg             = MagicMock()
    msg.content     = None
    msg.tool_calls  = [tool_call]
    msg.model_dump.return_value = {"role": "assistant", "tool_calls": []}

    choice         = MagicMock()
    choice.message = msg

    response         = MagicMock()
    response.choices = [choice]
    return response


def _valid_architect_json() -> str:
    return json.dumps({
        "rationale": "Split into model and route layers.",
        "tasks": [
            {
                "task_id":        "task_1_models",
                "description":    "Create SQLAlchemy models",
                "relevant_files": ["src/models.py"],
                "status":         "pending",
                "iteration_count": 0,
                "result":         None,
                "patch":          None,
                "failure_analysis": None,
            }
        ],
    })


def _valid_coder_json(file_path: str = "src/app.py") -> str:
    return json.dumps({
        "patches": [
            {
                "file_path":    file_path,
                "unified_diff": "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x\n+y",
                "description":  "Replace x with y",
            }
        ],
        "notes": "Simple replacement.",
    })


def _valid_debugger_json(verdict: str = "PASS") -> str:
    return json.dumps({
        "verdict":         verdict,
        "root_cause":      "" if verdict == "PASS" else "Null pointer on line 42",
        "fix_instruction": "" if verdict == "PASS" else "Add None check before call",
    })


def _valid_summary_json() -> str:
    return json.dumps({
        "summary":        "- Added models\n- Added routes",
        "files_modified": ["src/models.py", "src/routes.py"],
    })


def _make_global_state(
    tmp_path,
    stack: StackType = StackType.FAST,
    tasks: list | None = None,
    patches: list | None = None,
) -> GlobalState:
    return {
        "messages":       [],
        "stack":          stack,
        "access_mode":    AccessMode.NO_TRUST,
        "user_prompt":    "Build a FastAPI app",
        "tasks":          tasks or [],
        "patches":        patches or [],
        "project_root":   str(tmp_path),
        "scratchpad":     "",
        "final_summary":  None,
        "human_critique": None,
        "session_id":     "test-session-001",
    }


def _make_subtask_state(tmp_path, iteration_count: int = 0) -> SubTaskState:
    return {
        "task": SubTask(
            task_id="task_1",
            description="Build the auth module",
            relevant_files=["src/auth.py"],
            iteration_count=iteration_count,
        ),
        "project_root": str(tmp_path),
        "temp_dir":     str(tmp_path / "tmp" / "task_1"),
        "messages":     [],
        "scratchpad":   "",
        "session_id":   "test-session-001",
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — agents/base.py
# ═══════════════════════════════════════════════════════════════════════════

class TestStripMarkdownFences:
    def test_plain_json_unchanged(self):
        s = '{"key": "value"}'
        assert _strip_markdown_fences(s) == s

    def test_json_fences_removed(self):
        s = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(s) == '{"key": "value"}'

    def test_plain_fences_removed(self):
        s = '```\n{"key": "value"}\n```'
        assert _strip_markdown_fences(s) == '{"key": "value"}'

    def test_whitespace_stripped(self):
        assert _strip_markdown_fences('  {"a": 1}  ') == '{"a": 1}'


class TestCallAgentSuccess:
    """Happy-path: LLM returns valid JSON on the first attempt."""

    @pytest.fixture
    def agent_config(self):
        return {
            "provider":     "groq",
            "model":        "meta-llama/test-model",
            "system_prompt": "You are a test agent. Return only valid JSON.",
            "tools":        [],
        }

    async def test_returns_validated_model_instance(self, agent_config, mocker):
        mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(return_value=_make_llm_response(_valid_architect_json())),
        )
        result = await call_agent(
            agent_config=agent_config,
            messages=[{"role": "user", "content": "Plan this."}],
            output_schema=ArchitectOutput,
        )
        assert isinstance(result, ArchitectOutput)
        assert len(result.tasks) == 1

    async def test_system_prompt_prepended_to_messages(self, agent_config, mocker):
        mock_call = mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(return_value=_make_llm_response(_valid_architect_json())),
        )
        await call_agent(
            agent_config=agent_config,
            messages=[{"role": "user", "content": "Plan this."}],
            output_schema=ArchitectOutput,
        )
        call_messages = mock_call.call_args.kwargs["messages"]
        assert call_messages[0]["role"]    == "system"
        assert call_messages[0]["content"] == agent_config["system_prompt"]

    async def test_system_prompt_not_duplicated_if_already_present(self, agent_config, mocker):
        mock_call = mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(return_value=_make_llm_response(_valid_architect_json())),
        )
        messages = [
            {"role": "system", "content": "already present"},
            {"role": "user",   "content": "Plan this."},
        ]
        await call_agent(
            agent_config=agent_config,
            messages=messages,
            output_schema=ArchitectOutput,
        )
        call_messages = mock_call.call_args.kwargs["messages"]
        system_messages = [m for m in call_messages if m["role"] == "system"]
        assert len(system_messages) == 1

    async def test_strips_markdown_fences_before_validation(self, agent_config, mocker):
        wrapped = f"```json\n{_valid_architect_json()}\n```"
        mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(return_value=_make_llm_response(wrapped)),
        )
        result = await call_agent(
            agent_config=agent_config,
            messages=[{"role": "user", "content": "x"}],
            output_schema=ArchitectOutput,
        )
        assert isinstance(result, ArchitectOutput)


class TestCallAgentValidationRetry:
    """
    The validation retry loop must inject the Pydantic error back as a user
    message and succeed if a later attempt returns valid JSON.
    """

    @pytest.fixture
    def agent_config(self):
        return {
            "provider":      "groq",
            "model":         "test-model",
            "system_prompt": "Be concise.",
            "tools":         [],
        }

    async def test_retries_after_invalid_json(self, agent_config, mocker):
        responses = [
            _make_llm_response("not json at all"),       # attempt 1: invalid
            _make_llm_response(_valid_architect_json()), # attempt 2: valid
        ]
        mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(side_effect=responses),
        )
        result = await call_agent(
            agent_config=agent_config,
            messages=[{"role": "user", "content": "x"}],
            output_schema=ArchitectOutput,
        )
        assert isinstance(result, ArchitectOutput)

    async def test_error_injected_into_conversation_on_retry(self, agent_config, mocker):
        mock_call = mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(side_effect=[
                _make_llm_response("bad json"),
                _make_llm_response(_valid_architect_json()),
            ]),
        )
        await call_agent(
            agent_config=agent_config,
            messages=[{"role": "user", "content": "x"}],
            output_schema=ArchitectOutput,
        )
        # On the second call, the conversation must include the validation error
        second_call_messages = mock_call.call_args_list[1].kwargs["messages"]
        user_messages = [m for m in second_call_messages if m["role"] == "user"]
        assert any("validation" in m["content"].lower() for m in user_messages)

    async def test_raises_after_max_retries_exhausted(self, agent_config, mocker):
        mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(return_value=_make_llm_response("always bad json")),
        )
        with pytest.raises(ValueError, match="failed to produce valid"):
            await call_agent(
                agent_config=agent_config,
                messages=[{"role": "user", "content": "x"}],
                output_schema=ArchitectOutput,
            )

    async def test_llm_called_exactly_max_retries_times(self, agent_config, mocker):
        mock_call = mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(return_value=_make_llm_response("bad")),
        )
        with pytest.raises(ValueError):
            await call_agent(
                agent_config=agent_config,
                messages=[{"role": "user", "content": "x"}],
                output_schema=ArchitectOutput,
            )
        assert mock_call.call_count == MAX_VALIDATION_RETRIES


class TestCallAgentToolDispatch:
    """
    Tool call loop: LLM returns tool_calls → registry dispatches → LLM gets
    results → LLM returns final text → Pydantic validates.
    """

    @pytest.fixture
    def agent_config(self):
        return {
            "provider":      "groq",
            "model":         "test-model",
            "system_prompt": "Use tools.",
            "tools":         ["file_read"],
        }

    async def test_tool_result_appended_to_conversation(self, agent_config, mocker):
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_schemas_for.return_value = [{"type": "function", "function": {"name": "file_read"}}]
        mock_registry.dispatch = AsyncMock(return_value='{"content": "file contents"}')

        mock_call = mocker.patch(
            "synapse.agents.base._call_llm_with_retry",
            new=AsyncMock(side_effect=[
                _make_tool_call_response("file_read", {"path": "src/app.py"}),
                _make_llm_response(_valid_architect_json()),
            ]),
        )

        result = await call_agent(
            agent_config=agent_config,
            messages=[{"role": "user", "content": "x"}],
            output_schema=ArchitectOutput,
            registry=mock_registry,
        )
        assert isinstance(result, ArchitectOutput)
        mock_registry.dispatch.assert_called_once_with("file_read", {"path": "src/app.py"})

        # Second LLM call must include a tool role message with the result
        second_messages = mock_call.call_args_list[1].kwargs["messages"]
        tool_messages   = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "file contents" in tool_messages[0]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — architect.py
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitectNode:
    async def test_populates_tasks_in_state(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.architect.call_agent",
            new=AsyncMock(return_value=ArchitectOutput(
                rationale="Two independent concerns.",
                tasks=[
                    SubTask(task_id="t1", description="Models", relevant_files=["m.py"]),
                    SubTask(task_id="t2", description="Routes", relevant_files=["r.py"]),
                ],
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.BALANCED)
        result = await architect_node(state)

        assert len(result["tasks"]) == 2
        assert result["tasks"][0].task_id == "t1"

    async def test_uses_autonomous_config_for_autonomous_stack(self, tmp_path, mocker):
        mock_call = mocker.patch(
            "synapse.agents.architect.call_agent",
            new=AsyncMock(return_value=ArchitectOutput(
                rationale="x",
                tasks=[SubTask(task_id="t1", description="x", relevant_files=[])],
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.AUTONOMOUS)
        await architect_node(state)

        agent_cfg = mock_call.call_args.kwargs["agent_config"]
        assert "autonomous" in str(agent_cfg).lower() or agent_cfg["provider"] == "zhipuai"

    async def test_scratchpad_updated_with_rationale(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.architect.call_agent",
            new=AsyncMock(return_value=ArchitectOutput(
                rationale="Because separation of concerns.",
                tasks=[SubTask(task_id="t1", description="x", relevant_files=[])],
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.BALANCED)
        result = await architect_node(state)

        assert "separation of concerns" in result["scratchpad"]
        assert "[Architect]"            in result["scratchpad"]

    def test_build_architect_messages_includes_prompt(self, tmp_path):
        state    = _make_global_state(tmp_path)
        messages = _build_architect_messages(state)
        assert any("Build a FastAPI app" in m["content"] for m in messages)

    def test_build_architect_messages_includes_scratchpad(self, tmp_path):
        state             = _make_global_state(tmp_path)
        state["scratchpad"] = "previous context here"
        messages          = _build_architect_messages(state)
        assert any("previous context here" in m["content"] for m in messages)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — coder.py
# ═══════════════════════════════════════════════════════════════════════════

class TestCoderNode:
    async def test_fast_stack_adds_patches_to_state(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.coder.call_agent",
            new=AsyncMock(return_value=CoderOutput(
                patches=[FilePatch(
                    file_path="src/app.py",
                    unified_diff="--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y",
                    description="Fix",
                )],
                notes="done",
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.FAST)
        result = await coder_node(state)

        assert len(result["patches"]) == 1
        assert result["patches"][0].file_path == "src/app.py"

    async def test_balanced_stack_works_on_pending_task(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.coder.call_agent",
            new=AsyncMock(return_value=CoderOutput(
                patches=[FilePatch(
                    file_path="src/models.py",
                    unified_diff="--- a\n+++ b\n@@ -1 +1 @@",
                    description="Add model",
                )],
                notes="created",
            )),
        )
        tasks  = [SubTask(task_id="t1", description="Build models", relevant_files=["src/models.py"])]
        state  = _make_global_state(tmp_path, stack=StackType.BALANCED, tasks=tasks)
        result = await coder_node(state)

        assert len(result["patches"]) == 1

    async def test_no_pending_tasks_returns_no_new_patches(self, tmp_path, mocker):
        mocker.patch("synapse.agents.coder.call_agent", new=AsyncMock())
        tasks  = [SubTask(
            task_id="t1", description="x", relevant_files=[],
            status=TaskStatus.SUCCESS,
        )]
        state  = _make_global_state(tmp_path, stack=StackType.BALANCED, tasks=tasks)
        result = await coder_node(state)

        assert "patches" not in result or result.get("patches", []) == []

    async def test_existing_patches_preserved(self, tmp_path, mocker):
        existing = [FilePatch(
            file_path="old.py",
            unified_diff="--- a\n+++ b\n@@ -1 +1 @@",
            description="old",
        )]
        mocker.patch(
            "synapse.agents.coder.call_agent",
            new=AsyncMock(return_value=CoderOutput(
                patches=[FilePatch(
                    file_path="new.py",
                    unified_diff="--- a\n+++ b\n@@ -1 +1 @@",
                    description="new",
                )],
                notes="done",
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.FAST, patches=existing)
        result = await coder_node(state)

        file_paths = [p.file_path for p in result["patches"]]
        assert "old.py" in file_paths
        assert "new.py" in file_paths

    async def test_subgraph_increments_iteration_count(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.coder.call_agent",
            new=AsyncMock(return_value=CoderOutput(
                patches=[FilePatch(
                    file_path="src/auth.py",
                    unified_diff="--- a\n+++ b\n@@ -1 +1 @@",
                    description="auth",
                )],
                notes="done",
            )),
        )
        state  = _make_subtask_state(tmp_path, iteration_count=2)
        result = await coder_subgraph_node(state)

        assert result["task"].iteration_count == 3

    async def test_subgraph_sets_status_to_in_progress(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.coder.call_agent",
            new=AsyncMock(return_value=CoderOutput(
                patches=[FilePatch(
                    file_path="src/auth.py",
                    unified_diff="--- a\n+++ b",
                    description="auth",
                )],
                notes="",
            )),
        )
        state  = _make_subtask_state(tmp_path)
        result = await coder_subgraph_node(state)

        assert result["task"].status == TaskStatus.IN_PROGRESS


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — debugger.py
# ═══════════════════════════════════════════════════════════════════════════

class TestDebuggerNode:
    async def test_pass_verdict_returned_in_state(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.debugger.call_agent",
            new=AsyncMock(return_value=DebuggerAnalysis(
                verdict="PASS", root_cause="", fix_instruction=""
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.BALANCED)
        result = await debugger_node(state)

        assert result["debugger_verdict"] == "PASS"

    async def test_fail_verdict_returned_in_state(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.debugger.call_agent",
            new=AsyncMock(return_value=DebuggerAnalysis(
                verdict="FAIL",
                root_cause="Missing import",
                fix_instruction="Add 'from x import y'",
            )),
        )
        state  = _make_global_state(tmp_path, stack=StackType.BALANCED)
        result = await debugger_node(state)

        assert result["debugger_verdict"] == "FAIL"

    async def test_scratchpad_contains_verdict(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.debugger.call_agent",
            new=AsyncMock(return_value=DebuggerAnalysis(
                verdict="PASS", root_cause="", fix_instruction=""
            )),
        )
        state  = _make_global_state(tmp_path)
        result = await debugger_node(state)

        assert "PASS"        in result["scratchpad"]
        assert "[Debugger"   in result["scratchpad"]


class TestDebuggerSubgraphNode:
    async def test_pass_sets_task_status_to_success(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.debugger.call_agent",
            new=AsyncMock(return_value=DebuggerAnalysis(
                verdict="PASS", root_cause="", fix_instruction=""
            )),
        )
        state  = _make_subtask_state(tmp_path)
        result = await debugger_subgraph_node(state, test_output="All tests passed")

        assert result["task"].status == TaskStatus.SUCCESS

    async def test_fail_stores_fix_instruction_on_task(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.debugger.call_agent",
            new=AsyncMock(return_value=DebuggerAnalysis(
                verdict="FAIL",
                root_cause="NullPointerException at line 42",
                fix_instruction="Add None check before calling .profile",
            )),
        )
        state  = _make_subtask_state(tmp_path)
        result = await debugger_subgraph_node(state, test_output="FAILED: NullPointerException")

        assert result["task"].status           == TaskStatus.IN_PROGRESS
        assert "None check"                    in result["task"].failure_analysis

    async def test_fail_keeps_status_in_progress_not_failed(self, tmp_path, mocker):
        # TaskStatus.FAILED is set by the graph's hard-abort edge at iteration > 5,
        # not by the Debugger node itself. The Debugger only sets IN_PROGRESS on fail.
        mocker.patch(
            "synapse.agents.debugger.call_agent",
            new=AsyncMock(return_value=DebuggerAnalysis(
                verdict="FAIL", root_cause="x", fix_instruction="y"
            )),
        )
        state  = _make_subtask_state(tmp_path)
        result = await debugger_subgraph_node(state, test_output="error")

        assert result["task"].status != TaskStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — summariser.py
# ═══════════════════════════════════════════════════════════════════════════

class TestSummariserNode:
    async def test_populates_final_summary(self, tmp_path, mocker):
        mocker.patch(
            "synapse.agents.summariser.call_agent",
            new=AsyncMock(return_value=SummaryOutput(
                summary="- Added models\n- Added routes",
                files_modified=["src/models.py", "src/routes.py"],
            )),
        )
        state  = _make_global_state(tmp_path)
        result = await summariser_node(state)

        assert "Added models" in result["final_summary"]

    async def test_no_registry_passed_to_call_agent(self, tmp_path, mocker):
        """Summariser must never receive a registry — it has no tools."""
        mock_call = mocker.patch(
            "synapse.agents.summariser.call_agent",
            new=AsyncMock(return_value=SummaryOutput(
                summary="done", files_modified=[]
            )),
        )
        state = _make_global_state(tmp_path)
        await summariser_node(state, registry=MagicMock())

        # Even if a registry is passed to the node, call_agent must get registry=None
        assert mock_call.call_args.kwargs.get("registry") is None

    async def test_patches_hint_included_in_user_message(self, tmp_path, mocker):
        mock_call = mocker.patch(
            "synapse.agents.summariser.call_agent",
            new=AsyncMock(return_value=SummaryOutput(
                summary="x", files_modified=["src/auth.py"]
            )),
        )
        patches = [FilePatch(
            file_path="src/auth.py",
            unified_diff="--- a\n+++ b",
            description="auth",
        )]
        state   = _make_global_state(tmp_path, patches=patches)
        await summariser_node(state)

        messages = mock_call.call_args.kwargs["messages"]
        assert any("src/auth.py" in m["content"] for m in messages)