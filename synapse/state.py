"""
state.py — The canonical data layer for Synapse.

Every piece of information flowing through the LangGraph DAG is defined here.
The separation between LangGraph state (TypedDict) and LLM output schemas
(Pydantic BaseModel) is intentional:

  - Pydantic BaseModels validate what LLMs *output*. A ValidationError is not
    a crash — it is a self-correction prompt fed back to the LLM automatically.
  - TypedDicts are what LangGraph *manages* as it routes between nodes. Nodes
    receive the full state and return only the keys they want to update.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StackType(str, Enum):
    """The three execution topologies the user can select via /stack."""
    FAST = "fast"
    BALANCED = "balanced"
    AUTONOMOUS = "autonomous"


class AccessMode(str, Enum):
    """
    Controls file-write behaviour at the end of a run.
    TRUST   — patches are applied automatically; user receives a summary.
    NO_TRUST — execution pauses and the user sees a diff with Accept/Decline/Rewrite.
    """
    TRUST = "trust"
    NO_TRUST = "no_trust"


class TaskStatus(str, Enum):
    """Lifecycle of a single SubTask inside the autonomous map-reduce."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Pydantic output schemas — what agents must return, in strict mode.
#
# strict=True means Pydantic will reject coercions. If the LLM outputs
# "1" (string) where an int is expected, it fails loudly rather than
# silently converting, which forces the model to produce correct types.
# ---------------------------------------------------------------------------

class FilePatch(BaseModel):
    """
    The atomic unit of code change. Forcing the Coder to output a structured
    unified diff rather than a raw Markdown code block makes it possible to
    apply changes programmatically and detect conflicts during the merge phase.
    """
    model_config = ConfigDict(strict=True)

    file_path: str = Field(
        description="Repo-relative path to the file being modified, e.g. 'src/auth/router.py'"
    )
    unified_diff: str = Field(
        description="A valid unified diff string in standard --- a/ +++ b/ @@ format"
    )
    description: str = Field(
        description="One sentence explaining what this patch achieves"
    )


class SubTask(BaseModel):
    """
    One isolated unit of work in the autonomous stack's map-reduce.
    The Architect produces a list of these; each becomes an independent
    LangGraph Send() dispatch to a parallel branch.
    """
    model_config = ConfigDict(strict=True)

    task_id: str = Field(
        description="Unique snake_case identifier, e.g. 'task_1_database_layer'"
    )
    description: str = Field(
        description="Full plain-English description of what this branch must implement"
    )
    relevant_files: list[str] = Field(
        description="Every file path this task will read or modify"
    )
    status: TaskStatus = TaskStatus.PENDING
    iteration_count: int = Field(
        default=0,
        description="Coder→Debugger cycles consumed. The graph hard-aborts at 5."
    )
    result: str | None = Field(
        default=None,
        description="Human-readable outcome once the branch reaches a terminal state"
    )
    patch: FilePatch | None = Field(
        default=None,
        description="The final approved patch produced by this branch"
    )
    failure_analysis: str | None = Field(
        default=None,
        description="The Debugger's most recent error analysis, populated on FAIL"
    )


class ArchitectOutput(BaseModel):
    """
    The Architect's complete response. Requiring a 'rationale' field before
    the task list forces chain-of-thought reasoning — the model must articulate
    its decomposition logic before committing to specific tasks, which
    consistently improves plan quality.
    """
    model_config = ConfigDict(strict=True)

    rationale: str = Field(
        description="Step-by-step reasoning used to decompose the prompt into tasks"
    )
    tasks: list[SubTask] = Field(
        description="Ordered list of isolated, non-overlapping tasks for parallel execution"
    )


class CoderOutput(BaseModel):
    """
    What the Coder must return — a list of structured patches and nothing else.
    Raw Markdown blocks are explicitly prohibited by the agent's system prompt.
    """
    model_config = ConfigDict(strict=True)

    patches: list[FilePatch] = Field(
        description="One FilePatch per file modified during this task"
    )
    notes: str = Field(
        description="Brief implementation notes for the Debugger to read as context"
    )


class DebuggerAnalysis(BaseModel):
    """
    The Debugger's verdict after reviewing code (static) or reading test output
    (autonomous). The 'fix_instruction' is injected directly into the Coder's
    context on the next iteration if verdict is FAIL.
    """
    model_config = ConfigDict(strict=True)

    verdict: Literal["PASS", "FAIL"] = Field(
        description="PASS if review/tests succeed, FAIL if any problem is found"
    )
    root_cause: str = Field(
        description="Precise technical identification of the failure. Empty string if PASS."
    )
    fix_instruction: str = Field(
        description="Exact corrective instruction for the Coder. Empty string if PASS."
    )


class SummaryOutput(BaseModel):
    """The human-readable terminal report produced at the end of every stack run."""
    model_config = ConfigDict(strict=True)

    summary: str = Field(
        description="Bulleted list of accomplishments written for a developer audience"
    )
    files_modified: list[str] = Field(
        description="Flat list of every file path that was changed during this run"
    )


# ---------------------------------------------------------------------------
# GlobalState — the shared context that flows through the entire DAG.
#
# LangGraph passes this TypedDict between nodes. Each node receives the full
# state and returns a partial dict of only the keys it wants to update.
# LangGraph merges those updates back in automatically.
#
# The `messages` field uses LangGraph's `add_messages` reducer, which appends
# new messages rather than replacing the list. This is what makes multi-turn
# context accumulation work — without it, every node would wipe the history.
# ---------------------------------------------------------------------------

class GlobalState(TypedDict):
    # Full conversation history. add_messages appends; it does not overwrite.
    messages: Annotated[list[Any], add_messages]

    # Which of the three execution topologies is active for this run
    stack: StackType

    # Whether to auto-apply patches or pause for human review
    access_mode: AccessMode

    # The fully processed user prompt after lexer expansion
    user_prompt: str

    # Task list populated by the Architect in balanced and autonomous stacks
    tasks: list[SubTask]

    # All file patches collected across the entire run
    patches: list[FilePatch]

    # Absolute path to the project directory Synapse is operating on
    project_root: str

    # Running machine-oriented log that agents append to — Summariser reads this
    scratchpad: str

    # Populated once the Summariser node completes
    final_summary: str | None

    # Populated when the user chooses Rewrite in the HITL prompt
    human_critique: str | None

    # Unique identifier for this session, used for SQLite checkpoint persistence
    session_id: str


# ---------------------------------------------------------------------------
# SubTaskState — the isolated state for each parallel branch.
#
# When the autonomous stack fans out via Send(), each branch gets its own
# SubTaskState rather than the full GlobalState. This isolation is crucial:
# it prevents Branch A from accidentally reading Branch B's intermediate
# patches during the map-reduce phase, which would corrupt the merge.
# ---------------------------------------------------------------------------

class SubTaskState(TypedDict):
    # The specific task this branch is responsible for
    task: SubTask

    # Absolute path to the project root, for reading source files
    project_root: str

    # Isolated temp directory for this branch, e.g. /tmp/synapse/task_1_database_layer
    temp_dir: str

    # Branch-local message history — does not share with GlobalState
    messages: Annotated[list[Any], add_messages]

    # Branch-local scratchpad — merged into GlobalState.scratchpad by the Reducer
    scratchpad: str

    # Passed through for checkpoint persistence
    session_id: str