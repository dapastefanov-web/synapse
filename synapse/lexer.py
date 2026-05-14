"""
lexer.py — The Synapse command lexer.

The lexer is the first thing that processes raw terminal input. Its job is to
transform a single line like:

    /stack balanced && /init-docker && @arch Build a FastAPI authentication router

into an ordered list of LexedToken objects that the main loop executes
sequentially. It handles three syntactic constructs:

  1. Sequential chaining with &&
     Segments separated by && are executed left to right. The next segment
     does not begin until the previous one fully completes.

  2. System commands starting with /
     These trigger built-in Synapse operations. The argument is simply the
     text after the first space. There is no special prefix character.
       /stack balanced   →  command="stack",       argument="balanced"
       /access trust     →  command="access",      argument="trust"
       /init-docker      →  command="init-docker", argument=None

  3. Macro expansion starting with @
     The @ token is looked up in cli_config.yaml macros. Its full expansion
     is prepended to the remaining text and the whole thing becomes a PROMPT
     token sent to the active agent.
       @arch Build a router  →  "<full arch directive>\n\nBuild a router"

Aliases (defined in cli_config.yaml) are expanded first, before any of the
above classification steps, so /s and /stack are equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TokenType(str, Enum):
    COMMAND = "command"   # /stack balanced, /access trust, /init-docker
    MACRO   = "macro"     # @arch, @test — expanded to a full prompt prefix
    PROMPT  = "prompt"    # Plain text forwarded directly to the active agent


@dataclass
class LexedToken:
    """A single parsed, fully resolved unit of user input."""
    token_type: TokenType
    raw: str                    # Original text before any transformation

    # Populated for COMMAND tokens
    command: str | None = None  # e.g. "stack" from "/stack balanced"
    argument: str | None = None # e.g. "balanced" from "/stack balanced"

    # Populated for PROMPT and MACRO tokens
    content: str | None = None  # Final text to send to the agent


class LexerError(Exception):
    """Raised when the lexer encounters a construct it cannot parse."""
    pass


class Lexer:
    """
    Stateless lexer. Construct once with the loaded cli_config dict,
    then call tokenize() for every line of input from the terminal.
    """

    def __init__(self, cli_config: dict[str, Any]) -> None:
        self._aliases: dict[str, str] = cli_config.get("aliases", {})
        self._macros:  dict[str, str] = cli_config.get("macros",  {})

    # ── Public API ─────────────────────────────────────────────────────────

    def tokenize(self, raw_input: str) -> list[LexedToken]:
        """
        Transform a raw input string into an ordered list of LexedTokens.

        Empty input returns an empty list. Each && boundary produces a new
        token that will be executed after the previous one completes.
        """
        if not raw_input.strip():
            return []

        # Split on && first so alias expansion and classification happen
        # independently for each segment rather than on the whole line.
        segments = [seg.strip() for seg in raw_input.split("&&")]
        tokens: list[LexedToken] = []

        for segment in segments:
            if segment:  # skip any empty segments produced by trailing &&
                tokens.append(self._parse_segment(segment))

        return tokens

    # ── Internal helpers ───────────────────────────────────────────────────

    def _parse_segment(self, segment: str) -> LexedToken:
        """
        Classify a single segment and produce the appropriate LexedToken.
        Alias expansion is always the first step so that the downstream
        classification logic only ever sees canonical command names.
        """
        segment = self._expand_alias(segment)

        if segment.startswith("/"):
            return self._parse_command(segment)
        elif segment.startswith("@"):
            return self._parse_macro(segment)
        else:
            return LexedToken(
                token_type=TokenType.PROMPT,
                raw=segment,
                content=segment,
            )

    def _expand_alias(self, segment: str) -> str:
        """
        Check the first word of the segment against the alias table.
        If a match is found, replace the command portion and re-attach
        any trailing argument text.

        Example: "/s balanced" → "/stack balanced"
        """
        # split(None, 1) splits on any whitespace, up to a maximum of 2 parts.
        # This gives us the command word and everything after it separately.
        parts = segment.split(None, 1)
        command_word = parts[0]

        if command_word in self._aliases:
            expanded = self._aliases[command_word]
            # Re-attach the trailing argument if one was present
            return f"{expanded} {parts[1]}" if len(parts) > 1 else expanded

        return segment

    def _parse_command(self, segment: str) -> LexedToken:
        """
        Parse a /command [argument] segment.

        The leading slash is stripped, then the remainder is split into at
        most two parts on the first whitespace. This means:
          /stack balanced  →  command="stack",  argument="balanced"
          /init-docker     →  command="init-docker", argument=None
          /access trust    →  command="access", argument="trust"

        Arguments with internal spaces (e.g. file paths) are handled
        correctly because we only split on the *first* space — everything
        after it is kept together as the argument string.
        """
        without_slash = segment[1:]               # strip the leading /
        parts = without_slash.split(None, 1)      # split on first whitespace only

        command_name = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else None

        return LexedToken(
            token_type=TokenType.COMMAND,
            raw=segment,
            command=command_name,
            argument=argument,
        )

    def _parse_macro(self, segment: str) -> LexedToken:
        """
        Parse an @macro [user text] segment.

        The macro key (e.g. @arch) is looked up in the macros table. Its
        full expansion is prepended to the user's remaining instruction text,
        separated by a blank line. The combined result becomes the prompt
        content sent to the active agent.

        If the macro key is not found, the segment passes through unchanged
        as a plain PROMPT token rather than raising an error — the user may
        simply be typing a prompt that happens to start with @.
        """
        parts = segment.split(None, 1)
        macro_key  = parts[0]                                   # e.g. "@arch"
        user_text  = parts[1].strip() if len(parts) > 1 else "" # e.g. "Build a router"

        if macro_key in self._macros:
            prefix = self._macros[macro_key].strip()
            # Blank line between the directive prefix and the user's instruction
            # makes it easier for the LLM to distinguish the two sections.
            full_content = f"{prefix}\n\n{user_text}" if user_text else prefix
        else:
            # Unknown macro — pass through as a prompt without raising
            full_content = segment

        return LexedToken(
            token_type=TokenType.MACRO,
            raw=segment,
            content=full_content,
        )