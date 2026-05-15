"""
agents/coder.py — The Coder agent node.

Runs in all three stacks. In the fast stack it operates directly on the
project root. In balanced it works sequentially per task. In autonomous
it operates inside an isolated SubTaskState branch on a temp directory.

The Coder is the only agent that produces FilePatch output. It is explicitly
forbidden in its system prompt from returning raw Markdown — only structured
JSON unified diffs. This contract is what makes programmatic patch application
possible downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

from synapse.agents.base import call_agent
from synapse.config.loader import load_config
from synapse.state import (
    CoderOutput,
    GlobalState,
    StackType,
    SubTask,
    SubTaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def _read_relevant_files(
    relevant_files: list[str],
    root: str,
) -> str:
    """
    Read the contents of every file the Coder needs, returning a single
    formatted string. Missing files are noted rather than raising so the
    Coder is aware it may need to create them from scratch.
    """
    parts: list[str] = []
    for rel_path in relevant_files:
        full = Path(root) / rel_path
        if full.exists() and full.is_file():
            content = full.read_text(encoding="utf-8")
            parts.append(f"=== FILE: {rel_path} ===\n{content}")
        else:
            parts.append(f"=== FILE: {rel_path} === [DOES NOT EXIST — create it]")
    return "\n\n".join(parts)


def _build_fast_messages(state: GlobalState) -> list[dict]:
    """Fast stack: single file, no task decomposition."""
    return [
        {
            "role": "user",
            "content": (
                f"Project Root: {state['project_root']}\n\n"
                f"Request: {state['user_prompt']}"
            ),
        }
    ]


def _build_task_messages(
    task: SubTask,
    root: str,
    failure_context: str | None = None,
) -> list[dict]:
    """
    Balanced and autonomous stacks: task-scoped with file contents injected.
    On retry, the Debugger's failure_analysis is prepended as additional context.
    """
    file_context = _read_relevant_files(task.relevant_files, root)

    content = f"Task ID: {task.task_id}\nTask: {task.description}\n\n{file_context}"

    if failure_context:
        content = (
            f"PREVIOUS ATTEMPT FAILED. Debugger analysis:\n{failure_context}\n\n"
            + content
        )

    return [{"role": "user", "content": content}]


async def coder_node(
    state: GlobalState,
    registry=None,
) -> dict:
    """
    LangGraph node: Coder for the fast and balanced stacks.

    Fast stack  — operates on the user prompt directly.
    Balanced    — operates on state.tasks[0] (the first pending task).
    """
    stack = state.get("stack", StackType.FAST)
    agent_key = "coder_fast" if stack == StackType.FAST else "coder_balanced"

    agents_config = load_config("agents.yaml", project_root=state.get("project_root"))
    agent_config  = agents_config["agents"][agent_key]

    if stack == StackType.FAST:
        messages = _build_fast_messages(state)
    else:
        pending = [t for t in state.get("tasks", []) if t.status == TaskStatus.PENDING]
        if not pending:
            return {"scratchpad": state.get("scratchpad", "") + "\n[Coder] No pending tasks."}
        task     = pending[0]
        messages = _build_task_messages(task, state["project_root"])

    output: CoderOutput = await call_agent(
        agent_config=agent_config,
        messages=messages,
        output_schema=CoderOutput,
        registry=registry,
    )

    logger.info("Coder produced %d patch(es).", len(output.patches))

    existing_patches = list(state.get("patches", []))
    existing_patches.extend(output.patches)

    scratchpad_entry = (
        f"[Coder/{agent_key}] Produced {len(output.patches)} patch(es). Notes: {output.notes[:120]}"
    )
    updated_scratchpad = (
        state.get("scratchpad", "").rstrip() + f"\n\n{scratchpad_entry}"
    )

    return {
        "patches":    existing_patches,
        "scratchpad": updated_scratchpad,
    }


async def coder_subgraph_node(
    state: SubTaskState,
    registry=None,
) -> dict:
    """
    LangGraph node: Coder for autonomous stack parallel branches.
    Operates on the branch-local SubTaskState rather than GlobalState.
    """
    task = state["task"]

    agents_config = load_config("agents.yaml", project_root=state.get("project_root"))
    agent_config  = agents_config["agents"]["coder_autonomous"]

    messages = _build_task_messages(
        task,
        state["project_root"],
        failure_context=task.failure_analysis,
    )

    output: CoderOutput = await call_agent(
        agent_config=agent_config,
        messages=messages,
        output_schema=CoderOutput,
        registry=registry,
    )

    # Store the first patch on the task for the Debugger to reference.
    updated_task = task.model_copy(
        update={
            "status":           TaskStatus.IN_PROGRESS,
            "patch":            output.patches[0] if output.patches else None,
            "iteration_count":  task.iteration_count + 1,
        }
    )

    scratchpad_entry = (
        f"[Coder/autonomous] Task '{task.task_id}' attempt {updated_task.iteration_count}. "
        f"Notes: {output.notes[:100]}"
    )

    return {
        "task":      updated_task,
        "scratchpad": state.get("scratchpad", "").rstrip() + f"\n\n{scratchpad_entry}",
    }