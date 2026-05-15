"""
agents/base.py — The shared async LLM caller for all Synapse agents.

Two-tier retry architecture
────────────────────────────
Tier 1 — Network resilience (tenacity inside _call_llm_with_retry):
  Handles HTTP 429 (Rate Limit) and 502 (Bad Gateway) with exponential
  backoff. These are infrastructure failures — the LLM never even saw the
  request, so resending it unchanged is the correct response.

Tier 2 — Semantic resilience (Python loop inside call_agent):
  Handles Pydantic ValidationError — the LLM responded successfully but
  returned malformed JSON or wrong types. The exact Pydantic error message
  is appended to the conversation before retrying, which gives the LLM
  precise, actionable feedback about what it got wrong.

Tool call handling
──────────────────
Between the two retry tiers lives the tool call loop. When the LLM returns
tool_use content blocks, the loop dispatches each call through the
ToolRegistry, appends the results as 'tool' messages, and asks the LLM to
continue. This repeats until the LLM produces text content, at which point
the Pydantic validation tier takes over.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from synapse.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# TypeVar so call_agent's return type is inferred as the specific
# output_schema subclass rather than the generic BaseModel supertype.
T = TypeVar("T", bound=BaseModel)

# How many times the Pydantic validation loop will ask the LLM to self-correct.
# 3 is the right ceiling here — if the model cannot fix a JSON schema error
# in three attempts, the system prompt needs to be improved, not retried further.
MAX_VALIDATION_RETRIES = 3

# HTTP status codes that represent transient network conditions worth retrying.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 502})


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True if the exception represents a transient network condition.

    We check both LiteLLM's own exception class hierarchy and the generic
    status_code attribute, which covers custom subclasses across different
    LiteLLM versions. Checking by attribute rather than importing a specific
    class makes this future-proof against internal LiteLLM reorganisations.
    """
    # LiteLLM's named exception types for the conditions we care about
    if hasattr(litellm, "exceptions"):
        retryable_types = tuple(
            t for t in (
                getattr(litellm.exceptions, "RateLimitError", None),
                getattr(litellm.exceptions, "ServiceUnavailableError", None),
                getattr(litellm.exceptions, "APIConnectionError", None),
            )
            if t is not None
        )
        if retryable_types and isinstance(exc, retryable_types):
            return True

    # Fallback: any exception carrying a retryable HTTP status code
    return getattr(exc, "status_code", None) in _RETRYABLE_STATUS_CODES


@retry(
    retry=retry_if_exception(_is_retryable),
    # Start at 2 seconds, double on each attempt, cap at 60 seconds.
    # This gives the provider time to recover from a burst without
    # hammering it with rapid retries that make rate limiting worse.
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,  # re-raise the original exception if all attempts fail
)
async def _call_llm_with_retry(
    model_string: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
) -> Any:
    """
    One async LiteLLM call, wrapped in tenacity for network-level retry.

    This function is intentionally kept thin. Its only responsibility is
    making the API call and retrying on transient network errors. Every
    higher-level concern — tool dispatch, schema validation, conversation
    management — lives above this function in call_agent().

    Args:
        model_string: LiteLLM model identifier, e.g. 'groq/meta-llama/llama-4-8b-instruct'.
                      The 'provider/model' format is what LiteLLM uses to route the
                      request to the right API endpoint and authentication credential.
        messages:     Full conversation as a list of OpenAI-format message dicts.
        tools:        Optional LiteLLM tool spec list. Pass None rather than [] so
                      the 'tools' key is omitted from the API payload entirely for
                      agents that don't use tools — some providers reject an empty list.
        temperature:  Sampling temperature forwarded directly to the API.

    Returns the raw LiteLLM ModelResponse.
    """
    kwargs: dict[str, Any] = {
        "model":       model_string,
        "messages":    messages,
        "temperature": temperature,
    }
    # Only include tools in the payload when the agent actually has tools configured
    if tools:
        kwargs["tools"] = tools

    return await litellm.acompletion(**kwargs)


def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output as a defensive measure.

    The system prompts explicitly tell every agent not to wrap JSON in
    markdown fences. Despite this, models occasionally add them anyway.
    Stripping them here means the Pydantic validation loop only sees
    clean JSON, which produces more informative error messages and avoids
    burning a validation retry on a purely cosmetic wrapping issue.
    """
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.split("\n")
    # Remove the opening fence line (```json, ```python, or just ```)
    # and the closing fence line (```), keeping everything between.
    return "\n".join(
        line for line in lines
        if not line.strip().startswith("```")
    ).strip()


async def call_agent(
    agent_config: dict[str, Any],
    messages: list[dict[str, Any]],
    output_schema: Type[T],
    registry: ToolRegistry | None = None,
    temperature: float = 0.2,
) -> T:
    """
    The complete agent calling pipeline: tool dispatch + Pydantic validation.

    This is the single function every agent node in the graph calls. The
    flow is:
      1. Prepend the system prompt from agent_config (unless already present).
      2. Enter the validation retry loop (up to MAX_VALIDATION_RETRIES times).
         a. Inside that loop, enter the tool call loop.
            - Call the LLM via _call_llm_with_retry.
            - If the response contains tool calls, dispatch them, append results,
              call the LLM again. Repeat until the LLM returns text.
         b. Strip markdown fences from the final text response.
         c. Attempt Pydantic validation against output_schema.
         d. On success, return the validated instance.
         e. On ValidationError, append the error to the conversation and
            increment the retry counter.
      3. If all retries are exhausted, raise ValueError.

    Args:
        agent_config:  Agent's config dict from agents.yaml. Must contain
                       'provider', 'model', and 'system_prompt'. 'tools' is optional.
        messages:      The initial conversation history. Typically just one
                       user message from the calling node function.
        output_schema: The Pydantic model class to validate the response against.
                       The TypeVar ensures the return type is inferred correctly.
        registry:      ToolRegistry for dispatching tool calls. Required when
                       the agent has tools listed; can be None for the Summariser.
        temperature:   Forwarded to the LLM. Default 0.2 gives near-deterministic
                       output which is what we want for structured JSON generation.

    Returns a validated instance of output_schema.
    Raises ValueError if validation fails after all retries.
    """
    # Build the LiteLLM model string. The provider and model are stored as
    # separate fields in agents.yaml to make it easy to change either one
    # independently without needing to know LiteLLM's string format rules.
    model_string  = f"{agent_config['provider']}/{agent_config['model']}"
    system_prompt = agent_config.get("system_prompt", "")
    tool_names    = agent_config.get("tools", [])

    # Work on a copy so we do not mutate the caller's list between calls.
    conversation = list(messages)

    # Inject the system prompt as the first message if one is configured
    # and the conversation does not already start with a system message.
    if system_prompt and (not conversation or conversation[0]["role"] != "system"):
        conversation.insert(0, {"role": "system", "content": system_prompt})

    # Resolve this agent's tool schemas from the registry in one call.
    # Passing None (not []) to the API when there are no tools prevents
    # providers from complaining about an empty tools array.
    tool_schemas: list[dict[str, Any]] | None = None
    if registry and tool_names:
        resolved = registry.get_schemas_for(tool_names)
        tool_schemas = resolved if resolved else None

    last_validation_error: str = ""

    for attempt in range(1, MAX_VALIDATION_RETRIES + 1):

        # On the second and subsequent attempts, inject the exact Pydantic
        # error as a user message so the LLM can see precisely what it got wrong.
        if attempt > 1 and last_validation_error:
            logger.debug(
                "Validation retry %d/%d — model: %s",
                attempt, MAX_VALIDATION_RETRIES, model_string,
            )
            conversation.append({
                "role":    "user",
                "content": (
                    "Your previous response failed JSON schema validation. "
                    "Do not repeat explanations or add markdown fences. "
                    "Respond ONLY with the corrected JSON object that fixes this error:\n\n"
                    f"{last_validation_error}"
                ),
            })

        # ── Tool call loop ─────────────────────────────────────────────────
        # The LLM may call tools any number of times before settling on its
        # final text response. We loop until it stops requesting tool calls.
        while True:
            response = await _call_llm_with_retry(
                model_string=model_string,
                messages=conversation,
                tools=tool_schemas,
                temperature=temperature,
            )

            choice  = response.choices[0]
            message = choice.message

            # getattr with None default handles both older and newer LiteLLM
            # versions which represent absent tool_calls differently.
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                # No tool calls — the LLM has produced its final answer.
                # Break out of the tool loop and move to Pydantic validation.
                break

            # The LLM wants to call tools. Append its request to the
            # conversation first, then execute each tool in sequence.
            conversation.append(message.model_dump(exclude_none=True))

            if registry is None:
                # Agent config listed tools but no registry was provided.
                # Log a warning and break to attempt validation on whatever
                # text the LLM may have included alongside the tool call.
                logger.warning(
                    "Agent %s requested tool calls but no ToolRegistry was provided. "
                    "Check that the graph builder injects registry correctly.",
                    model_string,
                )
                break

            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                # Parse arguments defensively — malformed JSON from the LLM
                # should produce an empty dict rather than crashing the loop.
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                    logger.debug(
                        "Could not parse tool call arguments for '%s': %s",
                        tool_name, tool_call.function.arguments,
                    )

                tool_result = await registry.dispatch(tool_name, arguments)
                logger.debug(
                    "Tool '%s' executed. Result preview: %s",
                    tool_name, tool_result[:120],
                )

                # Append the tool result in the exact format LiteLLM expects.
                # The tool_call_id links this result back to the specific
                # tool_call in the previous assistant message.
                conversation.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "name":         tool_name,
                    "content":      tool_result,
                })

            # Continue the while loop: ask the LLM what to do next now
            # that it has all the tool results it requested.

        # ── Pydantic validation ────────────────────────────────────────────
        content = getattr(message, "content", None) or ""
        cleaned = _strip_markdown_fences(content)

        try:
            validated = output_schema.model_validate_json(cleaned)
            # Validation succeeded — return immediately without exhausting
            # the remaining retry budget.
            return validated

        except ValidationError as exc:
            last_validation_error = str(exc)
            # Append the bad assistant response to the conversation so it has
            # full context on the next attempt — the model can see what it
            # produced and why it was wrong.
            conversation.append({"role": "assistant", "content": content})
            logger.debug(
                "Pydantic validation failed on attempt %d/%d: %s",
                attempt, MAX_VALIDATION_RETRIES, last_validation_error[:200],
            )

    # All validation retries exhausted. Raise with enough context to debug.
    raise ValueError(
        f"Agent '{model_string}' failed to produce valid {output_schema.__name__} "
        f"output after {MAX_VALIDATION_RETRIES} attempts. "
        f"Last Pydantic error: {last_validation_error}"
    )