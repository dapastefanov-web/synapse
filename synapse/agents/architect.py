"""
agents/architect.py — The Architect agent node for both balanced and autonomous stacks.

Why the Architect never writes implementation code
────────────────────────────────────────────────────
The system prompt in agents.yaml explicitly forbids it. This is the most
important constraint in the entire agent design. Planning requires breadth —
seeing the full problem, identifying dependencies, and mapping work to files.
Implementation requires depth — focusing on one function, one test, one edge
case at a time. Asking a single LLM call to do both produces mediocre plans
and mediocre code. Keeping them separate allows each model to be selected and
prompted for its specific cognitive strength.

LangGraph node contract
────────────────────────
  Input:  GlobalState (full state, passed by LangGraph)
  Output: Partial state dict — only 'tasks' and 'scratchpad' are updated.
  LangGraph merges this partial dict back into the full state automatically.
"""

from __future__ import annotations

import logging

from synapse.agents.base import call_agent
from synapse.config.loader import load_config
from synapse.state import ArchitectOutput, GlobalState, StackType

logger = logging.getLogger(__name__)


def _build_architect_messages(state: GlobalState) -> list[dict]:
    """
    Construct the message list for the Architect's LLM call.

    The user message is structured in three sections so the LLM processes
    them in the right order: first understand the codebase context, then
    read the request, then decompose. Including the scratchpad gives the
    Architect visibility into any context the vector_search tool may have
    populated during a warm-up phase before this node ran.
    """
    user_content = (
        f"Project Root: {state['project_root']}\n\n"
        f"User Request: {state['user_prompt']}\n\n"
    )

    # Include any previously gathered context (e.g. from a warm-up indexing step)
    if state.get("scratchpad"):
        user_content += f"Existing Context:\n{state['scratchpad'].strip()}\n\n"

    # Explicit instruction reminding the Architect of its output contract.
    # Even though the system prompt covers this, repeating the key constraint
    # in the user message reduces the chance of the model drifting into code.
    user_content += (
        "Using your available tools to understand the codebase, decompose the "
        "above request into the minimum number of strictly non-overlapping, "
        "independently parallelisable tasks. "
        "Do not write any implementation code. "
        "Each task must be completable without depending on the output of any other task."
    )

    return [{"role": "user", "content": user_content}]


async def architect_node(
    state: GlobalState,
    registry=None,
) -> dict:
    """
    LangGraph node: run the Architect agent and populate state.tasks.

    This is the entry point for both the balanced and autonomous stacks.
    The two stacks use different agent configurations (different models and
    providers) but the same node function — the stack type determines which
    config key is loaded from agents.yaml.

    Args:
        state:    Full GlobalState provided by LangGraph.
        registry: ToolRegistry instance injected by the graph builder.
                  The Architect uses vector_search, firecrawl_search, and
                  list_directory, so a registry with those tools registered
                  should always be provided.

    Returns a partial state dict with 'tasks' and 'scratchpad' updated.
    """
    # The balanced stack uses a smaller Groq model which is faster and
    # sufficient for simpler decompositions. The autonomous stack uses
    # ZhipuAI's GLM which has stronger instruction-following for the more
    # complex multi-file plans that parallel execution requires.
    stack     = state.get("stack", StackType.BALANCED)
    agent_key = (
        "architect_autonomous"
        if stack == StackType.AUTONOMOUS
        else "architect_balanced"
    )

    agents_config = load_config("agents.yaml", project_root=state.get("project_root"))
    agent_config  = agents_config["agents"][agent_key]

    logger.info("Architect node starting — agent: '%s', stack: '%s'", agent_key, stack)

    messages = _build_architect_messages(state)

    output: ArchitectOutput = await call_agent(
        agent_config=agent_config,
        messages=messages,
        output_schema=ArchitectOutput,
        registry=registry,
    )

    logger.info(
        "Architect produced %d task(s). Rationale preview: %s",
        len(output.tasks),
        output.rationale[:100],
    )

    # Write the decomposition into the scratchpad so the Summariser can
    # report on it at the end of the run and so downstream nodes have
    # the Architect's rationale for reference during debugging.
    scratchpad_entry = (
        f"[Architect] Rationale: {output.rationale}\n"
        f"[Architect] Tasks ({len(output.tasks)} total):\n"
        + "\n".join(
            f"  [{i+1}] {t.task_id}: {t.description[:80]}"
            for i, t in enumerate(output.tasks)
        )
    )

    existing = state.get("scratchpad", "")
    updated_scratchpad = (
        f"{existing.rstrip()}\n\n{scratchpad_entry}" if existing else scratchpad_entry
    )

    # Return only the keys this node is responsible for updating.
    # LangGraph handles the merge — we must not return keys we did not touch
    # or we risk overwriting concurrent updates from other nodes.
    return {
        "tasks":      output.tasks,
        "scratchpad": updated_scratchpad,
    }