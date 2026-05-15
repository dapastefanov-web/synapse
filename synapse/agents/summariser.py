"""
agents/summariser.py — The Summariser agent node.

The Summariser runs last in every stack. It reads the scratchpad — which
accumulates every agent's log entries throughout the run — and translates
that machine-oriented noise into a professional, human-readable terminal
report. It also produces the canonical list of modified files which the
HITL diff viewer in Part 5 uses to know what to display.

The Summariser has no tools and never writes files. It is purely a
translation step from machine state to human communication.
"""

from __future__ import annotations

import logging

from synapse.agents.base import call_agent
from synapse.config.loader import load_config
from synapse.state import GlobalState, SummaryOutput

logger = logging.getLogger(__name__)


async def summariser_node(state: GlobalState, registry=None) -> dict:
    """
    LangGraph node: produce the final human-readable summary.

    Reads state.scratchpad and state.patches for context.
    Returns 'final_summary' which the terminal UI reads for display.
    """
    agents_config = load_config("agents.yaml", project_root=state.get("project_root"))
    agent_config  = agents_config["agents"]["summariser"]

    patches     = state.get("patches", [])
    files_hint  = ", ".join(p.file_path for p in patches) if patches else "none"

    messages = [
        {
            "role": "user",
            "content": (
                f"Original Request: {state.get('user_prompt', '')}\n\n"
                f"Execution Log:\n{state.get('scratchpad', '').strip()}\n\n"
                f"Files touched (hint): {files_hint}\n\n"
                "Produce a professional developer summary of what was accomplished."
            ),
        }
    ]

    # The Summariser has no tools — pass registry=None explicitly so no
    # tool schemas are loaded and the 'tools' key is omitted from the payload.
    output: SummaryOutput = await call_agent(
        agent_config=agent_config,
        messages=messages,
        output_schema=SummaryOutput,
        registry=None,
    )

    logger.info(
        "Summariser complete. Files modified: %s", output.files_modified
    )

    return {"final_summary": output.summary}