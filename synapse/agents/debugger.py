"""
agents/debugger.py — The Debugger agent node.

Two modes, one function interface
───────────────────────────────────
Static (balanced stack):
  Reads the unified diff and full file content after the Coder's patch.
  Does NOT execute code. Reviews for logical consistency, naming conventions,
  missing error handling, and potential runtime errors.

Execution (autonomous stack):
  Receives real stdout/stderr from pytest or the project's test runner
  running inside the Docker DevContainer as test_user. Identifies the
  precise root cause and formulates a surgical fix instruction.

The conditional edge
─────────────────────
After this node returns, the graph reads verdict from the returned state.
PASS routes forward to the next node; FAIL routes back to the Coder.
The iteration_count hard-abort check (> 5) happens in the graph's
conditional edge function, not here, keeping this node stateless.
"""

from __future__ import annotations

import logging

from synapse.agents.base import call_agent
from synapse.config.loader import load_config
from synapse.state import (
    DebuggerAnalysis,
    GlobalState,
    SubTaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def _build_static_messages(state: GlobalState) -> list[dict]:
    """
    Balanced stack: feed the diff(s) and scratchpad context.
    The Debugger reviews the most recent patches from state.patches.
    """
    patches = state.get("patches", [])
    if not patches:
        return [{"role": "user", "content": "No patches to review."}]

    patches_text = "\n\n".join(
        f"File: {p.file_path}\nDescription: {p.description}\n{p.unified_diff}"
        for p in patches
    )

    return [
        {
            "role": "user",
            "content": (
                f"Review the following patches produced by the Coder.\n\n"
                f"{patches_text}\n\n"
                f"Context from scratchpad:\n{state.get('scratchpad', '')[-800:]}"
            ),
        }
    ]


def _build_execution_messages(
    task_description: str,
    test_output: str,
) -> list[dict]:
    """
    Autonomous stack: feed the real test runner output for root-cause analysis.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Task: {task_description}\n\n"
                f"Test Runner Output:\n{test_output}\n\n"
                "Identify the precise root cause and provide a surgical fix instruction."
            ),
        }
    ]


async def debugger_node(
    state: GlobalState,
    registry=None,
    test_output: str | None = None,
) -> dict:
    """
    LangGraph node: Debugger for the balanced stack (static review).

    Returns a partial state dict with 'scratchpad' updated and a
    'debugger_verdict' key the conditional edge function reads.
    """
    agents_config = load_config("agents.yaml", project_root=state.get("project_root"))
    agent_config  = agents_config["agents"]["debugger_static"]

    messages = _build_static_messages(state)

    output: DebuggerAnalysis = await call_agent(
        agent_config=agent_config,
        messages=messages,
        output_schema=DebuggerAnalysis,
        registry=registry,
    )

    logger.info("Static Debugger verdict: %s", output.verdict)

    entry = (
        f"[Debugger/static] Verdict: {output.verdict}. "
        f"Root cause: {output.root_cause[:120] if output.root_cause else 'None'}"
    )
    updated_scratchpad = state.get("scratchpad", "").rstrip() + f"\n\n{entry}"

    return {
        "scratchpad":       updated_scratchpad,
        "debugger_verdict": output.verdict,
        "debugger_output":  output,
    }


async def debugger_subgraph_node(
    state: SubTaskState,
    test_output: str = "",
    registry=None,
) -> dict:
    """
    LangGraph node: Debugger for the autonomous stack parallel branches.
    Receives real test runner stdout/stderr via test_output.
    """
    task = state["task"]

    agents_config = load_config("agents.yaml", project_root=state.get("project_root"))
    agent_config  = agents_config["agents"]["debugger_autonomous"]

    messages = _build_execution_messages(task.description, test_output)

    output: DebuggerAnalysis = await call_agent(
        agent_config=agent_config,
        messages=messages,
        output_schema=DebuggerAnalysis,
        registry=registry,
    )

    logger.info(
        "Execution Debugger — task '%s' verdict: %s (iteration %d)",
        task.task_id, output.verdict, task.iteration_count,
    )

    if output.verdict == "PASS":
        updated_task = task.model_copy(update={"status": TaskStatus.SUCCESS})
    else:
        updated_task = task.model_copy(
            update={
                "status":           TaskStatus.IN_PROGRESS,
                "failure_analysis": output.fix_instruction,
            }
        )

    entry = (
        f"[Debugger/autonomous] Task '{task.task_id}' — {output.verdict}. "
        f"{output.root_cause[:100] if output.root_cause else ''}"
    )

    return {
        "task":      updated_task,
        "scratchpad": state.get("scratchpad", "").rstrip() + f"\n\n{entry}",
    }