"""
graphs/fast_stack.py — LangGraph topology for the Fast Stack.

DAG:
  START → coder_fast → summariser → [HITL if no_trust] → END

No planning phase, no testing phase. The Coder operates directly on the
user prompt and produces patches. The Summariser translates the scratchpad
into a human-readable report. If access_mode is no_trust, the graph pauses
before applying patches so the terminal UI can present the diff.
"""

from __future__ import annotations

import functools
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from synapse.agents.coder      import coder_node
from synapse.agents.summariser import summariser_node
from synapse.state             import GlobalState
from synapse.tools.registry    import ToolRegistry


def build_fast_graph(
    registry:    ToolRegistry,
    db_path:     str = ":memory:",
) -> Any:
    """
    Compile and return the Fast Stack LangGraph.

    Args:
        registry: Fully loaded ToolRegistry. The Coder receives it;
                  the Summariser does not (it has no tools).
        db_path:  SQLite path for checkpoint persistence. Use ':memory:'
                  for tests; use '.synapse/sessions.db' in production so
                  /resume can reload interrupted sessions.

    Returns a compiled LangGraph runnable.
    """
    builder = StateGraph(GlobalState)

    # Bind the registry to the coder node via partial so LangGraph only
    # passes state when invoking the node — the registry is infrastructure,
    # not state, so it must not flow through the graph's state dict.
    builder.add_node(
        "coder",
        functools.partial(coder_node, registry=registry),
    )
    builder.add_node(
        "summariser",
        functools.partial(summariser_node, registry=None),
    )

    # apply_patches is an identity node — it does no work itself.
    # Its sole purpose is to be a named interruption point. When
    # access_mode is no_trust, the graph is compiled with
    # interrupt_before=["apply_patches"], which causes LangGraph to
    # pause here and persist state. The terminal UI in Part 5 reads
    # the checkpoint, presents the diff, and resumes or aborts.
    builder.add_node("apply_patches", _apply_patches_node)

    builder.set_entry_point("coder")
    builder.add_edge("coder",       "summariser")
    builder.add_edge("summariser",  "apply_patches")
    builder.add_edge("apply_patches", END)

    checkpointer = SqliteSaver.from_conn_string(db_path)

    return builder.compile(
        checkpointer=checkpointer,
        # The interrupt fires regardless of access_mode — the terminal
        # dispatcher in main.py checks access_mode and either auto-resumes
        # (trust) or shows the HITL diff prompt (no_trust) before resuming.
        interrupt_before=["apply_patches"],
    )


def _apply_patches_node(state: GlobalState) -> dict:
    """
    Marker node for the HITL interrupt point.

    In trust mode the terminal auto-resumes past this node immediately.
    In no_trust mode the terminal pauses here, shows the diff, and waits
    for Accept / Decline / Rewrite before calling graph.invoke() again.

    The actual patch application (writing files to disk) happens in
    docker_manager.apply_patches(), called by the terminal dispatcher
    after this node completes — keeping file I/O out of the graph keeps
    the state machine pure and resumable.
    """
    # No state mutation — this node is purely an interrupt boundary.
    return {}