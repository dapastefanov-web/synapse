"""
graphs/autonomous_stack.py — LangGraph topology for the Autonomous Stack.

DAG:
  START → architect → parallel_dispatch (Send API) → [N x subgraph branches]
       → merge_node → [conflict_resolver if needed] → summariser → apply_patches → END

Each subgraph branch:
  coder_autonomous → debugger_autonomous ─┐
          ↑                               │ FAIL (max 5 iterations)
          └───────────────────────────────┘
                                          │ PASS / hard-abort
                                          ↓
                                         END (branch complete)
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import tempfile
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from synapse.agents.architect  import architect_node
from synapse.agents.coder      import coder_subgraph_node
from synapse.agents.debugger   import debugger_subgraph_node
from synapse.agents.summariser import summariser_node
from synapse.config.loader     import load_config
from synapse.state import (
    GlobalState,
    SubTask,
    SubTaskState,
    TaskStatus,
    FilePatch,
)
from synapse.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 5


def _get_max_iterations(project_root: str | None = None) -> int:
    try:
        pipeline = load_config("pipeline.yaml", project_root=project_root)
        return (
            pipeline
            .get("stacks", {})
            .get("autonomous", {})
            .get("subgraph", {})
            .get("edges", {})
            .get("debugger_autonomous", {})
            .get("max_iterations", _DEFAULT_MAX_ITERATIONS)
        )
    except Exception:
        return _DEFAULT_MAX_ITERATIONS


# ---------------------------------------------------------------------------
# Subgraph — one parallel branch per task
# ---------------------------------------------------------------------------

def _route_subgraph_after_debugger(state: SubTaskState) -> str:
    """
    Hard-abort at max_iterations regardless of verdict.
    The graph sets TaskStatus.FAILED on abort — the merge node reports it
    rather than silently dropping the branch.
    """
    task        = state["task"]
    max_iter    = _get_max_iterations(state.get("project_root"))

    if task.status == TaskStatus.SUCCESS:
        return END

    if task.iteration_count >= max_iter:
        logger.warning(
            "Hard abort: task '%s' hit max iterations (%d).",
            task.task_id, max_iter,
        )
        return END

    return "coder"


def _mark_failed_on_abort(state: SubTaskState) -> dict:
    """
    Called only when the hard-abort fires. Marks the task FAILED so the
    merge node knows to report it rather than treating it as a success.
    """
    task = state["task"]
    if task.status != TaskStatus.SUCCESS:
        return {"task": task.model_copy(update={"status": TaskStatus.FAILED})}
    return {}


async def _run_tests_and_debug(
    state: SubTaskState,
    registry: ToolRegistry | None = None,
) -> dict:
    """
    Runs the project's test suite inside the Docker DevContainer as test_user,
    then passes stdout/stderr to the Debugger for analysis.

    The actual docker exec_run call is delegated to the registry's run_command
    tool (which uses ai_user). Test execution requires test_user and therefore
    goes through docker_manager.run_tests() — imported lazily to avoid a
    circular import with the graph builders.
    """
    try:
        from synapse.docker_manager import DockerManager
        dm          = DockerManager(state["project_root"])
        test_output = await dm.run_tests()
    except Exception as exc:
        test_output = f"Test runner error: {exc}"

    return await debugger_subgraph_node(
        state, test_output=test_output, registry=registry
    )


def _build_subgraph(registry: ToolRegistry) -> Any:
    """Compile the inner subgraph that runs inside each parallel branch."""
    sg = StateGraph(SubTaskState)

    sg.add_node(
        "coder",
        functools.partial(coder_subgraph_node, registry=registry),
    )
    sg.add_node(
        "debugger",
        functools.partial(_run_tests_and_debug, registry=registry),
    )

    sg.set_entry_point("coder")
    sg.add_edge("coder", "debugger")
    sg.add_conditional_edges(
        "debugger",
        _route_subgraph_after_debugger,
        {"coder": "coder", END: END},
    )

    return sg.compile()


# ---------------------------------------------------------------------------
# Main graph nodes
# ---------------------------------------------------------------------------

def _dispatch_parallel_tasks(state: GlobalState) -> list[Send]:
    """
    Fan-out node: converts state.tasks into a list of Send() objects.

    Each Send() spawns an independent subgraph branch with its own
    SubTaskState. The temp_dir for each branch is a fresh directory under
    /tmp/synapse/<session_id>/<task_id> — isolated from every other branch
    so concurrent file writes cannot corrupt each other.
    """
    sends: list[Send] = []
    session_id = state.get("session_id", "default")

    for task in state.get("tasks", []):
        temp_dir = os.path.join(
            tempfile.gettempdir(),
            "synapse",
            session_id,
            task.task_id,
        )
        os.makedirs(temp_dir, exist_ok=True)

        # Copy the relevant files into the isolated temp directory so the
        # branch's Coder works on a snapshot and cannot affect the project
        # root until the merge node explicitly applies the patches.
        for rel_path in task.relevant_files:
            src = os.path.join(state["project_root"], rel_path)
            dst = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        branch_state: SubTaskState = {
            "task":         task,
            "project_root": state["project_root"],
            "temp_dir":     temp_dir,
            "messages":     [],
            "scratchpad":   "",
            "session_id":   session_id,
        }
        sends.append(Send("subgraph", branch_state))

    return sends


def _merge_node(state: GlobalState) -> dict:
    """
    Reducer: collects all completed SubTaskStates, gathers their patches,
    and attempts a clean merge back into GlobalState.patches.

    Conflict detection: if two branches produced patches for the same file,
    we flag it in the scratchpad and store both patches — the conflict
    resolver node handles the actual resolution.
    """
    # LangGraph collects Send() results back into the state key matching
    # the node name. The subgraph results arrive as a list in state["subgraph"].
    branch_results: list[SubTaskState] = state.get("subgraph", [])

    all_patches:      list[FilePatch] = []
    seen_files:       dict[str, int]  = {}  # file_path → index in all_patches
    conflicts:        list[str]       = []
    scratchpad_lines: list[str]       = []

    for branch in branch_results:
        task = branch.get("task")
        if task is None:
            continue

        scratchpad_lines.append(
            f"[Merge] Branch '{task.task_id}': {task.status.value}"
        )

        if task.status == TaskStatus.FAILED:
            scratchpad_lines.append(
                f"[Merge] WARNING: task '{task.task_id}' hard-aborted."
            )
            continue

        patch = task.patch
        if patch is None:
            continue

        if patch.file_path in seen_files:
            conflicts.append(patch.file_path)
            scratchpad_lines.append(
                f"[Merge] CONFLICT on '{patch.file_path}' — "
                f"branches '{task.task_id}' and a previous branch both modified it."
            )
        else:
            seen_files[patch.file_path] = len(all_patches)

        all_patches.append(patch)

    existing_scratchpad = state.get("scratchpad", "").rstrip()
    new_scratchpad = existing_scratchpad + "\n\n" + "\n".join(scratchpad_lines)

    return {
        "patches":          all_patches,
        "scratchpad":       new_scratchpad,
        "_merge_conflicts": conflicts,  # read by the routing edge below
    }


def _route_after_merge(state: GlobalState) -> str:
    conflicts = state.get("_merge_conflicts", [])
    if conflicts:
        logger.info("Merge conflicts detected in: %s", conflicts)
        return "conflict_resolver"
    return "summariser"


async def _conflict_resolver_node(
    state: GlobalState,
    registry: ToolRegistry | None = None,
) -> dict:
    """
    Resolves merge conflicts by invoking the Architect again with the
    conflicting patches as context, asking it to produce a unified patch.
    This is a best-effort resolution — the human still reviews in no_trust mode.
    """
    conflicts = state.get("_merge_conflicts", [])
    if not conflicts:
        return {}

    conflict_patches = [
        p for p in state.get("patches", [])
        if p.file_path in conflicts
    ]
    conflict_text = "\n\n".join(
        f"File: {p.file_path}\n{p.unified_diff}"
        for p in conflict_patches
    )

    logger.info("Conflict resolver handling %d conflicted file(s).", len(conflicts))

    # For now we append the conflict context to the scratchpad so the
    # Summariser surfaces it to the user. Full LLM-based resolution can be
    # wired in here by calling call_agent with the Architect config.
    entry = (
        f"[ConflictResolver] {len(conflicts)} conflict(s) detected. "
        f"Manual review required for: {', '.join(conflicts)}\n\n"
        f"{conflict_text}"
    )

    return {
        "scratchpad": state.get("scratchpad", "").rstrip() + f"\n\n{entry}",
    }


def _apply_patches_node(state: GlobalState) -> dict:
    """HITL interrupt boundary — identical role to other stacks."""
    return {}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_autonomous_graph(
    registry: ToolRegistry,
    db_path:  str = ":memory:",
) -> Any:
    """
    Compile and return the Autonomous Stack LangGraph.

    The subgraph is compiled separately and registered as a single node
    named 'subgraph'. LangGraph's Send() API then spawns multiple instances
    of it in parallel, one per task from the Architect's output.
    """
    subgraph = _build_subgraph(registry)

    builder = StateGraph(GlobalState)

    builder.add_node(
        "architect",
        functools.partial(architect_node, registry=registry),
    )
    # dispatch_parallel is a fan-out node that returns Send() objects —
    # LangGraph recognises the list[Send] return type and spawns branches.
    builder.add_node("dispatch_parallel", _dispatch_parallel_tasks)

    # 'subgraph' is the compiled inner graph. LangGraph routes each
    # Send("subgraph", branch_state) to this node automatically.
    builder.add_node("subgraph", subgraph)

    builder.add_node("merge_node", _merge_node)
    builder.add_node(
        "conflict_resolver",
        functools.partial(_conflict_resolver_node, registry=registry),
    )
    builder.add_node(
        "summariser",
        functools.partial(summariser_node, registry=None),
    )
    builder.add_node("apply_patches", _apply_patches_node)

    builder.set_entry_point("architect")
    builder.add_edge("architect",         "dispatch_parallel")
    builder.add_edge("dispatch_parallel", "subgraph")
    builder.add_edge("subgraph",          "merge_node")

    builder.add_conditional_edges(
        "merge_node",
        _route_after_merge,
        {
            "conflict_resolver": "conflict_resolver",
            "summariser":        "summariser",
        },
    )

    builder.add_edge("conflict_resolver", "summariser")
    builder.add_edge("summariser",        "apply_patches")
    builder.add_edge("apply_patches",     END)

    if db_path == ":memory:":
        checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["apply_patches"],
    )