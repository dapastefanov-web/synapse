"""
graphs/balanced_stack.py — LangGraph topology for the Balanced Stack.

DAG:
  START → architect → coder → debugger_static ─┐
                          ↑                    │ FAIL (max 3 retries)
                          └────────────────────┘
                                               │ PASS
                                               ↓
                                          summariser → apply_patches → END

The retry guard is enforced in the conditional edge function. If
debugger_iterations exceeds the pipeline.yaml max_retries value, the
graph routes forward to the summariser regardless of verdict — this
prevents infinite billing loops on pathological inputs.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from synapse.agents.architect  import architect_node
from synapse.agents.coder      import coder_node
from synapse.agents.debugger   import debugger_node
from synapse.agents.summariser import summariser_node
from synapse.config.loader     import load_config
from synapse.state             import GlobalState
from synapse.tools.registry    import ToolRegistry

logger = logging.getLogger(__name__)

# Fallback if pipeline.yaml is unavailable
_DEFAULT_MAX_RETRIES = 3


def _get_max_retries(project_root: str | None = None) -> int:
    """Read the balanced stack's max_retries from pipeline.yaml."""
    try:
        pipeline = load_config("pipeline.yaml", project_root=project_root)
        return (
            pipeline
            .get("stacks", {})
            .get("balanced", {})
            .get("edges", {})
            .get("debugger_static", {})
            .get("max_retries", _DEFAULT_MAX_RETRIES)
        )
    except Exception:
        return _DEFAULT_MAX_RETRIES


def _route_after_debugger(state: GlobalState) -> str:
    """
    Conditional edge function called after every debugger_node execution.

    Routing logic:
      - PASS verdict              → summariser (normal path)
      - FAIL + under retry limit  → coder (self-correction loop)
      - FAIL + at retry limit     → summariser (hard abort, prevents runaway cost)

    The retry limit is read from pipeline.yaml so it can be overridden
    per-project without touching Python code.
    """
    verdict    = state.get("debugger_verdict")
    iterations = state.get("debugger_iterations", 0)
    max_retries = _get_max_retries(state.get("project_root"))

    if verdict == "PASS":
        logger.info("Debugger PASS — routing to summariser.")
        return "summariser"

    if iterations >= max_retries:
        logger.warning(
            "Debugger FAIL but max retries (%d) reached — forcing summariser.",
            max_retries,
        )
        return "summariser"

    logger.info(
        "Debugger FAIL (iteration %d/%d) — routing back to coder.",
        iterations, max_retries,
    )
    return "coder"


def _increment_debugger_iterations(state: GlobalState) -> dict:
    """
    Thin wrapper around debugger_node that increments the iteration counter
    before routing. Keeping the counter update here rather than inside
    debugger.py keeps the agent node stateless and reusable across stacks.
    """
    return {"debugger_iterations": state.get("debugger_iterations", 0) + 1}


def build_balanced_graph(
    registry: ToolRegistry,
    db_path:  str = ":memory:",
) -> Any:
    """
    Compile and return the Balanced Stack LangGraph.

    Args:
        registry: Fully loaded ToolRegistry. Architect, Coder, and static
                  Debugger all receive it. Summariser does not.
        db_path:  SQLite path for checkpoint persistence.

    Returns a compiled LangGraph runnable.
    """
    builder = StateGraph(GlobalState)

    builder.add_node(
        "architect",
        functools.partial(architect_node, registry=registry),
    )
    builder.add_node(
        "coder",
        functools.partial(coder_node, registry=registry),
    )

    # The debugger node is wrapped so the iteration counter is incremented
    # atomically with the node execution — both updates land in the same
    # LangGraph state merge rather than requiring a separate node.
    async def _debugger_with_counter(state: GlobalState) -> dict:
        node_result    = await debugger_node(state, registry=registry)
        counter_update = _increment_debugger_iterations(state)
        return {**node_result, **counter_update}

    builder.add_node("debugger", _debugger_with_counter)
    builder.add_node(
        "summariser",
        functools.partial(summariser_node, registry=None),
    )
    builder.add_node("apply_patches", _apply_patches_node)

    builder.set_entry_point("architect")
    builder.add_edge("architect", "coder")
    builder.add_edge("coder",     "debugger")

    builder.add_conditional_edges(
        "debugger",
        _route_after_debugger,
        {
            "coder":      "coder",
            "summariser": "summariser",
        },
    )

    builder.add_edge("summariser",    "apply_patches")
    builder.add_edge("apply_patches", END)

    if db_path == ":memory:":
        checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver.from_conn_string(db_path)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["apply_patches"],
    )


def _apply_patches_node(state: GlobalState) -> dict:
    """HITL interrupt boundary — identical role to the fast stack version."""
    return {}