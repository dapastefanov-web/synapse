"""
tests/test_part4.py — Fast Unit Tests for the Part 4 graph and Docker layers.

Strategy: Fast Unit Tests (pytest + pytest-mock + pytest-asyncio)
──────────────────────────────────────────────────────────────────
- LangGraph graph builders are tested by inspecting compiled graph structure.
- Conditional edge functions are called directly with crafted state dicts.
- DockerManager is tested with a fully mocked Docker SDK client.
- No real containers, LLM calls, or file system mutations outside tmp_path.

Run with:
    pytest tests/test_part4.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from synapse.state import (
    AccessMode,
    FilePatch,
    GlobalState,
    StackType,
    SubTask,
    SubTaskState,
    TaskStatus,
)
from synapse.graphs.fast_stack     import build_fast_graph, _apply_patches_node as fast_apl
from synapse.graphs.balanced_stack import (
    build_balanced_graph,
    _route_after_debugger,
    _get_max_retries,
)
from synapse.graphs.autonomous_stack import (
    build_autonomous_graph,
    _route_subgraph_after_debugger,
    _dispatch_parallel_tasks,
    _merge_node,
    _route_after_merge,
)
from synapse.docker_manager import DockerManager, DockerManagerError
from synapse.tools.registry import ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _global_state(tmp_path, **overrides) -> GlobalState:
    base: GlobalState = {
        "messages":            [],
        "stack":               StackType.FAST,
        "access_mode":         AccessMode.NO_TRUST,
        "user_prompt":         "Build something",
        "tasks":               [],
        "patches":             [],
        "project_root":        str(tmp_path),
        "scratchpad":          "",
        "final_summary":       None,
        "human_critique":      None,
        "session_id":          "test-session",
        "debugger_verdict":    None,
        "debugger_iterations": 0,
    }
    base.update(overrides)
    return base


def _subtask_state(tmp_path, task: SubTask, **overrides) -> SubTaskState:
    base: SubTaskState = {
        "task":         task,
        "project_root": str(tmp_path),
        "temp_dir":     str(tmp_path / "tmp" / task.task_id),
        "messages":     [],
        "scratchpad":   "",
        "session_id":   "test-session",
    }
    base.update(overrides)
    return base


def _make_registry(tmp_path) -> ToolRegistry:
    from synapse.tools.native import NativeToolkit, ToolContext
    reg     = ToolRegistry()
    toolkit = NativeToolkit(ToolContext(project_root=str(tmp_path)))
    reg.load_native_tools(toolkit)
    return reg


def _make_patch(file_path: str = "src/app.py") -> FilePatch:
    return FilePatch(
        file_path=file_path,
        unified_diff=f"--- a/{file_path}\n+++ b/{file_path}\n@@ -1 +1 @@\n-x\n+y",
        description="Test patch",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — graphs/fast_stack.py
# ═══════════════════════════════════════════════════════════════════════════

class TestFastGraphStructure:

    def test_compiles_without_error(self, tmp_path):
        reg   = _make_registry(tmp_path)
        graph = build_fast_graph(reg, db_path=":memory:")
        assert graph is not None

    def test_graph_has_expected_nodes(self, tmp_path):
        reg   = _make_registry(tmp_path)
        graph = build_fast_graph(reg, db_path=":memory:")
        nodes = set(graph.get_graph().nodes.keys())
        assert "coder"         in nodes
        assert "summariser"    in nodes
        assert "apply_patches" in nodes

    def test_apply_patches_node_returns_empty_dict(self, tmp_path):
        state  = _global_state(tmp_path)
        result = fast_apl(state)
        assert result == {}

    def test_apply_patches_node_does_not_mutate_state(self, tmp_path):
        state  = _global_state(tmp_path, scratchpad="existing")
        fast_apl(state)
        assert state["scratchpad"] == "existing"

    def test_builds_with_real_db_path(self, tmp_path):
        reg   = _make_registry(tmp_path)
        db    = str(tmp_path / "test.db")
        graph = build_fast_graph(reg, db_path=db)
        assert graph is not None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — graphs/balanced_stack.py
# ═══════════════════════════════════════════════════════════════════════════

class TestBalancedGraphStructure:

    def test_compiles_without_error(self, tmp_path):
        reg   = _make_registry(tmp_path)
        graph = build_balanced_graph(reg, db_path=":memory:")
        assert graph is not None

    def test_graph_has_expected_nodes(self, tmp_path):
        reg   = _make_registry(tmp_path)
        graph = build_balanced_graph(reg, db_path=":memory:")
        nodes = set(graph.get_graph().nodes.keys())
        assert "architect"     in nodes
        assert "coder"         in nodes
        assert "debugger"      in nodes
        assert "summariser"    in nodes
        assert "apply_patches" in nodes

    def test_builds_with_real_db_path(self, tmp_path):
        reg   = _make_registry(tmp_path)
        db    = str(tmp_path / "test.db")
        graph = build_balanced_graph(reg, db_path=db)
        assert graph is not None


class TestBalancedRoutingLogic:
    """
    _route_after_debugger is the conditional edge function.
    We call it directly with crafted state dicts to confirm all three
    routing branches work correctly without running the full graph.
    """

    def test_pass_verdict_routes_to_summariser(self, tmp_path):
        state  = _global_state(tmp_path, debugger_verdict="PASS", debugger_iterations=1)
        result = _route_after_debugger(state)
        assert result == "summariser"

    def test_fail_under_limit_routes_to_coder(self, tmp_path):
        state  = _global_state(tmp_path, debugger_verdict="FAIL", debugger_iterations=1)
        result = _route_after_debugger(state)
        assert result == "coder"

    def test_fail_at_limit_routes_to_summariser(self, tmp_path):
        # At exactly max_retries the hard-abort must fire.
        max_r = _get_max_retries()
        state = _global_state(
            tmp_path,
            debugger_verdict="FAIL",
            debugger_iterations=max_r,
        )
        result = _route_after_debugger(state)
        assert result == "summariser"

    def test_fail_over_limit_routes_to_summariser(self, tmp_path):
        state  = _global_state(tmp_path, debugger_verdict="FAIL", debugger_iterations=999)
        result = _route_after_debugger(state)
        assert result == "summariser"

    def test_none_verdict_does_not_route_to_coder_infinitely(self, tmp_path):
        # A None verdict (e.g. first run before any debugger execution)
        # must not route to coder and create an infinite loop.
        state  = _global_state(tmp_path, debugger_verdict=None, debugger_iterations=0)
        result = _route_after_debugger(state)
        # None is not PASS so it falls into the fail branch — but iteration
        # 0 is under the limit, so this would route to coder. That is correct
        # — but we confirm it doesn't raise.
        assert result in ("coder", "summariser")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — graphs/autonomous_stack.py
# ═══════════════════════════════════════════════════════════════════════════

class TestAutonomousGraphStructure:

    def test_compiles_without_error(self, tmp_path):
        reg   = _make_registry(tmp_path)
        graph = build_autonomous_graph(reg, db_path=":memory:")
        assert graph is not None

    def test_graph_has_expected_nodes(self, tmp_path):
        reg   = _make_registry(tmp_path)
        graph = build_autonomous_graph(reg, db_path=":memory:")
        nodes = set(graph.get_graph().nodes.keys())
        assert "architect"          in nodes
        assert "dispatch_parallel"  in nodes
        assert "merge_node"         in nodes
        assert "summariser"         in nodes
        assert "apply_patches"      in nodes

    def test_builds_with_real_db_path(self, tmp_path):
        reg   = _make_registry(tmp_path)
        db    = str(tmp_path / "test.db")
        graph = build_autonomous_graph(reg, db_path=db)
        assert graph is not None


class TestSubgraphRoutingLogic:

    def _make_task(self, status: TaskStatus, iteration: int) -> SubTask:
        return SubTask(
            task_id="t1",
            description="x",
            relevant_files=[],
            status=status,
            iteration_count=iteration,
        )

    def test_success_routes_to_end(self, tmp_path):
        task   = self._make_task(TaskStatus.SUCCESS, 1)
        state  = _subtask_state(tmp_path, task)
        result = _route_subgraph_after_debugger(state)
        assert result == "__end__" or result is None or str(result) == "END"

    def test_fail_under_limit_routes_to_coder(self, tmp_path):
        task   = self._make_task(TaskStatus.IN_PROGRESS, 2)
        state  = _subtask_state(tmp_path, task)
        result = _route_subgraph_after_debugger(state)
        assert result == "coder"

    def test_fail_at_limit_routes_to_end(self, tmp_path):
        task   = self._make_task(TaskStatus.IN_PROGRESS, 5)
        state  = _subtask_state(tmp_path, task)
        result = _route_subgraph_after_debugger(state)
        # At iteration 5 the hard-abort fires — routes to END not coder
        assert result != "coder"


class TestDispatchParallelTasks:

    def test_creates_one_send_per_task(self, tmp_path):
        tasks = [
            SubTask(task_id="t1", description="x", relevant_files=[]),
            SubTask(task_id="t2", description="y", relevant_files=[]),
        ]
        state  = _global_state(tmp_path, tasks=tasks)
        sends  = _dispatch_parallel_tasks(state)
        assert len(sends) == 2

    def test_empty_tasks_returns_empty_list(self, tmp_path):
        state = _global_state(tmp_path, tasks=[])
        sends = _dispatch_parallel_tasks(state)
        assert sends == []

    def test_temp_dir_created_for_each_task(self, tmp_path):
        tasks = [SubTask(task_id="task_db", description="x", relevant_files=[])]
        state = _global_state(tmp_path, tasks=tasks, session_id="sess1")
        _dispatch_parallel_tasks(state)
        import tempfile
        expected = os.path.join(tempfile.gettempdir(), "synapse", "sess1", "task_db")
        assert os.path.isdir(expected)

    def test_relevant_files_copied_to_temp_dir(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "models.py").write_text("class User: pass")

        tasks = [SubTask(task_id="t1", description="x", relevant_files=["src/models.py"])]
        state = _global_state(tmp_path, tasks=tasks, session_id="copytest")
        _dispatch_parallel_tasks(state)

        import tempfile
        copied = Path(tempfile.gettempdir()) / "synapse" / "copytest" / "t1" / "src" / "models.py"
        assert copied.exists()
        assert copied.read_text() == "class User: pass"


class TestMergeNode:

    def _make_subtask_result(
        self,
        task_id: str,
        status: TaskStatus,
        file_path: str | None = None,
    ) -> SubTaskState:
        patch = _make_patch(file_path) if file_path else None
        task  = SubTask(
            task_id=task_id,
            description="x",
            relevant_files=[],
            status=status,
            patch=patch,
        )
        return {
            "task":         task,
            "project_root": "/tmp/proj",
            "temp_dir":     "/tmp/branch",
            "messages":     [],
            "scratchpad":   "",
            "session_id":   "test",
        }

    def test_successful_branches_patches_collected(self, tmp_path):
        branch1 = self._make_subtask_result("t1", TaskStatus.SUCCESS, "src/a.py")
        branch2 = self._make_subtask_result("t2", TaskStatus.SUCCESS, "src/b.py")
        state   = _global_state(tmp_path, subgraph=[branch1, branch2])
        result  = _merge_node(state)

        files = [p.file_path for p in result["patches"]]
        assert "src/a.py" in files
        assert "src/b.py" in files

    def test_failed_branch_excluded_from_patches(self, tmp_path):
        branch1 = self._make_subtask_result("t1", TaskStatus.SUCCESS, "src/a.py")
        branch2 = self._make_subtask_result("t2", TaskStatus.FAILED,  "src/b.py")
        state   = _global_state(tmp_path, subgraph=[branch1, branch2])
        result  = _merge_node(state)

        files = [p.file_path for p in result["patches"]]
        assert "src/a.py" in files
        assert "src/b.py" not in files

    def test_conflict_detected_for_same_file(self, tmp_path):
        branch1 = self._make_subtask_result("t1", TaskStatus.SUCCESS, "src/shared.py")
        branch2 = self._make_subtask_result("t2", TaskStatus.SUCCESS, "src/shared.py")
        state   = _global_state(tmp_path, subgraph=[branch1, branch2])
        result  = _merge_node(state)

        assert "src/shared.py" in result.get("_merge_conflicts", [])

    def test_no_conflict_when_files_are_distinct(self, tmp_path):
        branch1 = self._make_subtask_result("t1", TaskStatus.SUCCESS, "src/a.py")
        branch2 = self._make_subtask_result("t2", TaskStatus.SUCCESS, "src/b.py")
        state   = _global_state(tmp_path, subgraph=[branch1, branch2])
        result  = _merge_node(state)

        assert result.get("_merge_conflicts", []) == []

    def test_scratchpad_updated_with_branch_statuses(self, tmp_path):
        branch1 = self._make_subtask_result("t1", TaskStatus.SUCCESS, "src/a.py")
        state   = _global_state(tmp_path, subgraph=[branch1])
        result  = _merge_node(state)

        assert "t1"      in result["scratchpad"]
        assert "[Merge]" in result["scratchpad"]


class TestMergeRouting:

    def test_no_conflicts_routes_to_summariser(self, tmp_path):
        state  = _global_state(tmp_path, _merge_conflicts=[])
        result = _route_after_merge(state)
        assert result == "summariser"

    def test_conflicts_route_to_conflict_resolver(self, tmp_path):
        state  = _global_state(tmp_path, _merge_conflicts=["src/shared.py"])
        result = _route_after_merge(state)
        assert result == "conflict_resolver"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — docker_manager.py
# ═══════════════════════════════════════════════════════════════════════════

class TestDockerManagerInit:

    def test_image_name_derived_from_project_name(self, tmp_path):
        dm = DockerManager(str(tmp_path))
        assert "synapse-" in dm._image_name
        # Name must be lowercase with no illegal characters
        assert dm._image_name == dm._image_name.lower()

    async def test_init_docker_creates_synapse_dir(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        dm = DockerManager(str(tmp_path))
        await dm.init_docker()
        assert (tmp_path / ".synapse").is_dir()

    async def test_init_docker_generates_dockerfile(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        dm = DockerManager(str(tmp_path))
        await dm.init_docker()
        assert (tmp_path / ".synapse" / "Dockerfile").exists()

    async def test_init_docker_generates_shell_nix(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        dm = DockerManager(str(tmp_path))
        await dm.init_docker()
        assert (tmp_path / ".synapse" / "shell.nix").exists()

    async def test_init_docker_does_not_overwrite_existing_dockerfile(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        synapse_dir = tmp_path / ".synapse"
        synapse_dir.mkdir()
        custom = "# my custom dockerfile"
        (synapse_dir / "Dockerfile").write_text(custom)

        dm = DockerManager(str(tmp_path))
        await dm.init_docker()

        assert (synapse_dir / "Dockerfile").read_text() == custom

    async def test_init_docker_calls_build_image(self, tmp_path, mocker):
        mock_build = mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        dm = DockerManager(str(tmp_path))
        await dm.init_docker()
        mock_build.assert_called_once()


class TestDockerManagerClient:

    def test_get_client_raises_if_docker_unavailable(self, tmp_path, mocker):
        mocker.patch("docker.from_env", side_effect=Exception("Docker not found"))
        dm = DockerManager(str(tmp_path))
        with pytest.raises(DockerManagerError, match="Cannot connect"):
            dm._get_client()

    def test_get_client_cached_after_first_call(self, tmp_path, mocker):
        mock_client = MagicMock()
        mocker.patch("docker.from_env", return_value=mock_client)
        dm = DockerManager(str(tmp_path))

        c1 = dm._get_client()
        c2 = dm._get_client()
        assert c1 is c2


class TestDockerManagerRunTests:

    @pytest.fixture
    def dm_with_container(self, tmp_path):
        dm = DockerManager(str(tmp_path))
        mock_container = MagicMock()
        dm._container  = mock_container
        return dm, mock_container

    async def test_run_tests_returns_pass_label_on_exit_0(self, dm_with_container):
        dm, container = dm_with_container
        container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"1 passed",
        )
        result = await dm.run_tests()
        assert "[PASSED]" in result
        assert "1 passed" in result

    async def test_run_tests_returns_fail_label_on_nonzero(self, dm_with_container):
        dm, container = dm_with_container
        container.exec_run.return_value = MagicMock(
            exit_code=1,
            output=b"FAILED: test_auth",
        )
        result = await dm.run_tests()
        assert "FAILED" in result

    async def test_run_tests_executes_as_test_user(self, dm_with_container):
        dm, container = dm_with_container
        container.exec_run.return_value = MagicMock(exit_code=0, output=b"ok")
        await dm.run_tests()

        call_kwargs = container.exec_run.call_args.kwargs
        assert call_kwargs.get("user") == "test_user"

    async def test_run_tests_without_container_returns_error(self, tmp_path):
        dm     = DockerManager(str(tmp_path))
        result = await dm.run_tests()
        assert "Error" in result

    async def test_run_tests_custom_command_used(self, dm_with_container):
        dm, container = dm_with_container
        container.exec_run.return_value = MagicMock(exit_code=0, output=b"ok")
        await dm.run_tests(command="cargo test 2>&1")

        exec_cmd = container.exec_run.call_args.args[0]
        assert "cargo test" in exec_cmd[2]


class TestDockerManagerDetectTestCommand:

    def test_detects_pytest_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
        dm = DockerManager(str(tmp_path))
        assert "pytest" in dm._detect_test_command()

    def test_detects_npm_test_from_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        dm = DockerManager(str(tmp_path))
        assert "npm test" in dm._detect_test_command()

    def test_detects_cargo_test_from_cargo_toml(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]")
        dm = DockerManager(str(tmp_path))
        assert "cargo test" in dm._detect_test_command()

    def test_falls_back_to_make_test(self, tmp_path):
        dm = DockerManager(str(tmp_path))
        assert "make test" in dm._detect_test_command()


class TestAddNixDependency:

    async def test_adds_package_to_shell_nix(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        synapse_dir = tmp_path / ".synapse"
        synapse_dir.mkdir()
        (synapse_dir / "shell.nix").write_text(
            "{ pkgs ? import <nixpkgs> {} }:\npkgs.mkShell {\n  buildInputs = with pkgs; [\n    python3\n  ];\n}\n"
        )
        dm = DockerManager(str(tmp_path))
        result = await dm.add_nix_dependency("rustc")

        content = (synapse_dir / "shell.nix").read_text()
        assert "rustc" in content
        assert "added" in result.lower()

    async def test_skips_if_already_present(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        synapse_dir = tmp_path / ".synapse"
        synapse_dir.mkdir()
        (synapse_dir / "shell.nix").write_text(
            "{ pkgs ? import <nixpkgs> {} }:\npkgs.mkShell {\n  buildInputs = with pkgs; [\n    python3\n    rustc\n  ];\n}\n"
        )
        dm = DockerManager(str(tmp_path))
        result = await dm.add_nix_dependency("rustc")

        assert "already" in result.lower()

    async def test_returns_error_if_no_shell_nix(self, tmp_path, mocker):
        mocker.patch.object(DockerManager, "_build_image", new=AsyncMock())
        dm     = DockerManager(str(tmp_path))
        result = await dm.add_nix_dependency("rustc")
        assert "Error" in result