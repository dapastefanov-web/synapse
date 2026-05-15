"""
tools/mcp_client.py — Async Model Context Protocol client.

Protocol flow on connection
───────────────────────────
1. Spawn server as async subprocess (stdin/stdout/stderr pipes)
2. Send 'initialize' request  →  receive capabilities response
3. Send 'notifications/initialized'  (no response expected)
4. Send 'tools/list'  →  receive available tool schemas
5. Translate MCP schemas to LiteLLM/OpenAI tool format
6. Ready for call_tool() requests

Tool name namespacing
──────────────────────
Tool names from the server are prefixed with server_name__ to prevent
collisions when multiple MCP servers are active simultaneously.
  'query_table' from the 'postgres' server becomes 'postgres__query_table'.
The prefix is stripped before forwarding the call to the actual server.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class MCPClientError(Exception):
    """Raised for protocol-level errors with an MCP server."""
    pass


class MCPClient:
    """
    Manages one MCP server subprocess and its JSON-RPC communication channel.

    Use as an async context manager for clean lifecycle management:
        async with MCPClient("postgres", "npx", [...]) as client:
            result = await client.call_tool("postgres__run_query", {"sql": "SELECT 1"})
    """

    def __init__(self, server_name: str, command: str, args: list[str]) -> None:
        """
        Args:
            server_name: Short identifier used as a namespace prefix for tool names.
            command:     Executable to spawn, e.g. 'npx'.
            args:        Arguments, e.g. ['-y', '@modelcontextprotocol/server-postgres', 'postgresql://...'].
        """
        self._server_name  = server_name
        self._command      = command
        self._args         = args
        self._process:     asyncio.subprocess.Process | None = None
        self._message_id:  int                               = 0
        self._tool_schemas: list[dict[str, Any]]             = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Spawn the server subprocess and complete the full handshake.
        After this returns, tool_schemas is populated and the client is ready.
        """
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 1 — Initialize handshake
        init_id = self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id":      init_id,
            "method":  "initialize",
            "params":  {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "clientInfo":      {"name": "synapse", "version": "1.0.0"},
            },
        })
        await self._receive(init_id)

        # 2 — Confirm initialization (notification, no response expected)
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 3 — Discover available tools
        list_id = self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id":      list_id,
            "method":  "tools/list",
            "params":  {},
        })
        response = await self._receive(list_id)
        mcp_tools = response.get("result", {}).get("tools", [])
        self._tool_schemas = self._translate_tool_schemas(mcp_tools)

    async def stop(self) -> None:
        """Terminate the server subprocess, waiting up to 5 seconds before force-killing."""
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            finally:
                self._process = None

    async def __aenter__(self) -> "MCPClient":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── Tool operations ────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute an MCP tool and return its output as a plain string.

        Args:
            tool_name: The namespaced name (e.g. 'postgres__run_query').
            arguments: Tool parameters as a dict.

        Returns the tool result as a string, ready to inject into the LangGraph
        state as a tool result message content.
        """
        if self._process is None:
            raise MCPClientError(
                f"MCPClient '{self._server_name}' is not started. Call start() first."
            )

        # Strip the server prefix before sending to the actual MCP server
        mcp_name = tool_name.removeprefix(f"{self._server_name}__")

        call_id = self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id":      call_id,
            "method":  "tools/call",
            "params":  {"name": mcp_name, "arguments": arguments},
        })
        response = await self._receive(call_id)

        if "error" in response:
            err = response["error"]
            return f"MCP Error [{err.get('code', '?')}]: {err.get('message', str(err))}"

        return self._extract_text_content(response.get("result", {}))

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """LiteLLM-formatted schemas for all tools this server exposes."""
        return self._tool_schemas

    @property
    def server_name(self) -> str:
        return self._server_name

    # ── Internal helpers ───────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._message_id += 1
        return self._message_id

    async def _send(self, message: dict[str, Any]) -> None:
        """Serialise message to JSON and write it to the subprocess stdin."""
        if self._process is None or self._process.stdin is None:
            raise MCPClientError("Cannot send — subprocess is not running.")
        data = (json.dumps(message) + "\n").encode("utf-8")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _receive(self, expected_id: int, timeout: float = 30.0) -> dict[str, Any]:
        """
        Read stdout lines until a JSON-RPC response matching expected_id arrives.

        Non-JSON lines and JSON messages with a different id are silently skipped —
        many MCP servers write log output or send notifications between responses.
        The timeout prevents hanging indefinitely on a crashed server.
        """
        if self._process is None or self._process.stdout is None:
            raise MCPClientError("Cannot receive — subprocess is not running.")

        try:
            async with asyncio.timeout(timeout):
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        raise MCPClientError(
                            f"MCP server '{self._server_name}' closed stdout unexpectedly."
                        )
                    try:
                        msg = json.loads(line.decode("utf-8").strip())
                        if isinstance(msg, dict) and msg.get("id") == expected_id:
                            return msg
                        # Wrong id or notification — keep reading
                    except json.JSONDecodeError:
                        continue  # non-JSON line — skip silently
        except asyncio.TimeoutError:
            raise MCPClientError(
                f"Timed out waiting for response id={expected_id} "
                f"from MCP server '{self._server_name}'."
            )

    def _translate_tool_schemas(
        self, mcp_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert MCP tool schemas into LiteLLM/OpenAI function calling format.

        MCP:    {"name": "run_query", "description": "...", "inputSchema": {...}}
        LiteLLM: {"type": "function", "function": {"name": "postgres__run_query", ...}}
        """
        return [
            {
                "type": "function",
                "function": {
                    "name":        f"{self._server_name}__{tool['name']}",
                    "description": tool.get("description", ""),
                    "parameters":  tool.get("inputSchema", {
                        "type": "object", "properties": {}
                    }),
                },
            }
            for tool in mcp_tools
        ]

    @staticmethod
    def _extract_text_content(result: dict[str, Any]) -> str:
        """
        Pull plain text from an MCP tool result's content blocks.
        Falls back to raw JSON if no text blocks are present.
        """
        content_blocks = result.get("content", [])
        text_parts = [
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        ]
        return "\n".join(text_parts) if text_parts else json.dumps(result)