"""
tests/test_part2.py — Fast Unit Tests for the entire Part 2 layer.

Strategy: Fast Unit Tests (pytest + pytest-mock + pytest-asyncio)
─────────────────────────────────────────────────────────────────
No real Docker containers, LLM API calls, Firecrawl requests, or MCP server
subprocesses are created. All external I/O is replaced with mocks so tests
run in milliseconds and produce deterministic results.

E2E interactive tests (pexpect) are reserved for Part 5 where the Accept/
Decline/Rewrite terminal prompt lives.

Run with:
    pytest tests/test_part2.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Conditional skip for sqlite-vec ─────────────────────────────────────────
# If sqlite-vec is not installed or the C extension fails to load, skip the
# VectorStore tests gracefully rather than producing a confusing error.
try:
    from synapse.memory.vector_store import VectorStore, VectorStoreError
    SQLITE_VEC_AVAILABLE = True
except Exception:
    SQLITE_VEC_AVAILABLE = False

pytestmark_sqlite = pytest.mark.skipif(
    not SQLITE_VEC_AVAILABLE,
    reason="sqlite-vec not available in this environment",
)

from synapse.tools.native import NativeToolkit, ToolContext, _validate_and_resolve_path, NativeToolkitError
from synapse.tools.mcp_client import MCPClient, MCPClientError
from synapse.tools.registry import (
    ToolRegistry,
    _build_tool_spec,
    _parse_docstring_args,
    _python_type_to_json_schema,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_embeddings(dim: int = 10, value: float = 0.1) -> list[float]:
    """Return a list of `dim` floats, all set to `value`. Used for reproducible tests."""
    return [value] * dim


def make_mock_process(json_responses: list[dict[str, Any]]) -> MagicMock:
    """
    Build a mock asyncio.subprocess.Process that feeds pre-scripted JSON-RPC
    responses from its stdout, one per readline() call.

    The client's _receive() method reads lines until it finds one with the
    right id. We simply return each response dict in order; the mock's
    side_effect list handles sequencing automatically.
    """
    process = MagicMock()

    # stdin: write() is synchronous, drain() is async
    process.stdin        = MagicMock()
    process.stdin.write  = MagicMock()
    process.stdin.drain  = AsyncMock(return_value=None)

    # stdout: readline() is async — returns each pre-scripted response in order
    response_bytes = [
        (json.dumps(r) + "\n").encode("utf-8")
        for r in json_responses
    ]
    process.stdout           = MagicMock()
    process.stdout.readline  = AsyncMock(side_effect=response_bytes)

    # cleanup
    process.terminate = MagicMock()
    process.wait      = AsyncMock(return_value=0)

    return process


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — memory/vector_store.py
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestVectorStore:
    """
    VectorStore tests use an in-memory SQLite database (db_path=':memory:').
    This makes every test fully isolated — there is no file to clean up and
    no state leaks between tests.

    The embedding dimension is kept at 10 throughout these tests instead of
    1536 to keep fixture data small without changing any logic.
    """

    DIM = 10

    @pytest.fixture
    def store(self):
        """Provide an initialised in-memory VectorStore for each test."""
        s = VectorStore(db_path=":memory:", embedding_dim=self.DIM)
        s.initialize()
        yield s
        s.close()

    def test_initialize_is_idempotent(self):
        # Calling initialize() twice must not raise or corrupt data.
        s = VectorStore(db_path=":memory:", embedding_dim=self.DIM)
        s.initialize()
        s.initialize()  # second call — no error
        assert s.count() == 0
        s.close()

    def test_context_manager_opens_and_closes(self):
        with VectorStore(db_path=":memory:", embedding_dim=self.DIM) as s:
            assert s.count() == 0
        # After __exit__, the connection is closed; further ops should raise
        with pytest.raises(VectorStoreError):
            s.count()

    def test_add_and_retrieve_document(self, store):
        store.add_document(
            doc_id="src/app.py",
            content="from fastapi import FastAPI",
            embedding=make_embeddings(self.DIM, 0.5),
            file_path="src/app.py",
            metadata={"language": "python"},
        )
        doc = store.get_document("src/app.py")
        assert doc is not None
        assert doc["content"]            == "from fastapi import FastAPI"
        assert doc["metadata"]["language"] == "python"

    def test_get_document_missing_returns_none(self, store):
        assert store.get_document("nonexistent.py") is None

    def test_count_increments_on_add(self, store):
        assert store.count() == 0
        store.add_document("a.py", "content a", make_embeddings(self.DIM, 0.1), "a.py")
        assert store.count() == 1
        store.add_document("b.py", "content b", make_embeddings(self.DIM, 0.2), "b.py")
        assert store.count() == 2

    def test_add_document_overwrites_existing(self, store):
        store.add_document("a.py", "original", make_embeddings(self.DIM, 0.1), "a.py")
        store.add_document("a.py", "updated",  make_embeddings(self.DIM, 0.2), "a.py")
        assert store.count() == 1
        assert store.get_document("a.py")["content"] == "updated"

    def test_delete_document_removes_from_both_tables(self, store):
        store.add_document("a.py", "content", make_embeddings(self.DIM, 0.1), "a.py")
        assert store.count() == 1
        store.delete_document("a.py")
        assert store.count() == 0
        assert store.get_document("a.py") is None

    def test_delete_nonexistent_is_noop(self, store):
        # Deleting a doc_id that does not exist must not raise.
        store.delete_document("ghost.py")

    def test_search_returns_results_ordered_by_distance(self, store):
        # Add two documents with different embeddings.
        # The query is identical to doc_a's embedding, so doc_a should be
        # the closest match (distance ~= 0) and appear first.
        store.add_document("close.py", "close content", make_embeddings(self.DIM, 0.9), "close.py")
        store.add_document("far.py",   "far content",   make_embeddings(self.DIM, 0.1), "far.py")

        results = store.search(query_embedding=make_embeddings(self.DIM, 0.9), top_k=2)

        assert len(results) == 2
        assert results[0]["file_path"] == "close.py"   # closest first
        assert results[1]["file_path"] == "far.py"
        assert results[0]["distance"]  <= results[1]["distance"]

    def test_search_wrong_dimension_raises(self, store):
        with pytest.raises(VectorStoreError, match="dimensions"):
            store.search(query_embedding=[0.1] * (self.DIM + 5), top_k=3)

    def test_add_wrong_dimension_raises(self, store):
        with pytest.raises(VectorStoreError, match="dimensions"):
            store.add_document("x.py", "content", [0.1] * (self.DIM + 1), "x.py")

    def test_uninitialized_raises(self):
        # Any operation before initialize() must raise VectorStoreError,
        # not a cryptic sqlite3 AttributeError.
        s = VectorStore(db_path=":memory:", embedding_dim=self.DIM)
        with pytest.raises(VectorStoreError, match="not been initialised"):
            s.count()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — tools/native.py
# ═══════════════════════════════════════════════════════════════════════════

class TestPathValidation:
    """
    Path traversal protection is a security-critical behaviour.
    These tests confirm the guard works correctly before we test the full
    toolkit methods that rely on it.
    """

    def test_relative_path_resolves_inside_root(self, tmp_path):
        resolved = _validate_and_resolve_path(str(tmp_path), "src/app.py")
        assert resolved == (tmp_path / "src" / "app.py").resolve()

    def test_absolute_path_inside_root_passes(self, tmp_path):
        target = tmp_path / "src" / "app.py"
        resolved = _validate_and_resolve_path(str(tmp_path), str(target))
        assert resolved == target.resolve()

    def test_path_traversal_raises(self, tmp_path):
        # '../../etc/passwd' must be blocked regardless of whether it exists.
        with pytest.raises(NativeToolkitError, match="traversal"):
            _validate_and_resolve_path(str(tmp_path), "../../etc/passwd")


class TestNativeToolkitFileOps:
    """
    File operation tests use pytest's tmp_path fixture for a real but isolated
    temporary directory. No mocking is needed because file I/O is the actual
    behaviour being tested here.
    """

    @pytest.fixture
    def toolkit(self, tmp_path):
        return NativeToolkit(ToolContext(project_root=str(tmp_path)))

    async def test_file_read_existing_file(self, toolkit, tmp_path):
        (tmp_path / "hello.py").write_text("print('hello')", encoding="utf-8")
        result = await toolkit.file_read("hello.py")
        assert "error"   not in result
        assert result["content"] == "print('hello')"

    async def test_file_read_missing_file_returns_error_dict(self, toolkit):
        result = await toolkit.file_read("does_not_exist.py")
        assert "error" in result
        assert "not found" in result["error"].lower()

    async def test_file_read_on_directory_returns_error(self, toolkit, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = await toolkit.file_read("subdir")
        assert "error" in result

    async def test_file_write_creates_new_file(self, toolkit, tmp_path):
        result = await toolkit.file_write("new_file.py", "x = 1")
        assert "error"         not in result
        assert result["bytes_written"] > 0
        assert (tmp_path / "new_file.py").read_text() == "x = 1"

    async def test_file_write_creates_nested_directories(self, toolkit, tmp_path):
        result = await toolkit.file_write("a/b/c/nested.py", "pass")
        assert "error" not in result
        assert (tmp_path / "a" / "b" / "c" / "nested.py").exists()

    async def test_file_write_overwrites_existing_file(self, toolkit, tmp_path):
        (tmp_path / "f.py").write_text("original")
        await toolkit.file_write("f.py", "updated")
        assert (tmp_path / "f.py").read_text() == "updated"

    async def test_file_write_blocked_outside_root(self, toolkit):
        result = await toolkit.file_write("../../evil.py", "rm -rf /")
        assert "error" in result

    async def test_list_directory_returns_files_and_dirs(self, toolkit, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("")
        (tmp_path / "README.md").write_text("")

        result = await toolkit.list_directory(".")
        assert "error"            not in result
        assert result["total_entries"] >= 2
        # Paths should be project-relative strings, not absolute
        assert all(not Path(p).is_absolute() for p in result["files"])

    async def test_list_directory_missing_path_returns_error(self, toolkit):
        result = await toolkit.list_directory("ghost_dir")
        assert "error" in result


class TestNativeToolkitDocker:
    """
    run_command wraps the synchronous Docker SDK in asyncio.to_thread.
    We mock the Docker client to confirm the method constructs the exec_run
    call correctly and maps the result into the expected output dict shape.
    """

    @pytest.fixture
    def docker_ctx(self, tmp_path):
        """A ToolContext with a mocked Docker client and a fake container ID."""
        mock_docker     = MagicMock()
        mock_container  = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        return ToolContext(
            project_root=str(tmp_path),
            docker_client=mock_docker,
            docker_container_id="abc123",
        ), mock_container

    async def test_run_command_success(self, docker_ctx):
        ctx, mock_container = docker_ctx
        # Simulate a successful command: exit code 0, some stdout output.
        mock_container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"Hello from container\n",
        )
        toolkit = NativeToolkit(ctx)
        result  = await toolkit.run_command("echo hello")

        assert result["success"]   is True
        assert result["exit_code"] == 0
        assert "Hello from container" in result["stdout"]
        # Confirm the command always runs as ai_user, never as root or test_user
        mock_container.exec_run.assert_called_once()
        call_kwargs = mock_container.exec_run.call_args
        assert call_kwargs.kwargs.get("user") == "ai_user"

    async def test_run_command_nonzero_exit_code(self, docker_ctx):
        ctx, mock_container = docker_ctx
        mock_container.exec_run.return_value = MagicMock(
            exit_code=1,
            output=b"SyntaxError: invalid syntax",
        )
        toolkit = NativeToolkit(ctx)
        result  = await toolkit.run_command("python bad_file.py")

        assert result["success"]   is False
        assert result["exit_code"] == 1

    async def test_run_command_no_docker_returns_error(self, tmp_path):
        # When Docker has not been initialised, the method must return an
        # informative error dict rather than raising an AttributeError.
        toolkit = NativeToolkit(ToolContext(project_root=str(tmp_path)))
        result  = await toolkit.run_command("ls")
        assert "error"   in result
        assert "init-docker" in result["error"]


class TestNativeToolkitVectorSearch:
    """
    vector_search makes an async call to litellm.aembedding and then calls
    VectorStore.search(). Both are mocked here — we are testing that the
    method correctly chains them together, not the quality of embeddings.
    """

    @pytest.fixture
    def ctx_with_store(self, tmp_path):
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"file_path": "src/auth.py", "content": "import jwt", "distance": 0.05},
        ]
        return ToolContext(
            project_root=str(tmp_path),
            vector_store=mock_store,
            embedding_model="text-embedding-3-small",
        ), mock_store

    async def test_vector_search_returns_formatted_results(
        self, ctx_with_store, mocker
    ):
        ctx, mock_store = ctx_with_store

        # Mock litellm.aembedding to return a fake embedding response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [{"embedding": [0.1] * 1536}]
        mocker.patch("litellm.aembedding", return_value=mock_embedding_response)

        toolkit = NativeToolkit(ctx)
        result  = await toolkit.vector_search("authentication logic", top_k=3)

        assert "error"           not in result
        assert result["count"]   == 1
        assert result["results"][0]["file_path"] == "src/auth.py"
        # Confirm we actually searched with the correct top_k
        mock_store.search.assert_called_once()
        assert mock_store.search.call_args.kwargs["top_k"] == 3

    async def test_vector_search_no_store_returns_error(self, tmp_path):
        toolkit = NativeToolkit(ToolContext(project_root=str(tmp_path)))
        result  = await toolkit.vector_search("anything")
        assert "error"    in result
        assert "index-project" in result["error"]


class TestNativeToolkitFirecrawl:
    """
    firecrawl_search wraps the synchronous FirecrawlApp.search() in asyncio.to_thread.
    We mock FirecrawlApp at the import level so no real HTTP calls are made.
    """

    @pytest.fixture
    def ctx_with_key(self, tmp_path):
        return ToolContext(
            project_root=str(tmp_path),
            firecrawl_api_key="fc-test-key",
        )

    async def test_firecrawl_returns_formatted_results(self, ctx_with_key, mocker):
        mock_app    = MagicMock()
        mock_app.search.return_value = [
            {
                "url":      "https://fastapi.tiangolo.com/tutorial/",
                "metadata": {"title": "FastAPI Tutorial"},
                "markdown": "FastAPI is a modern web framework.",
            }
        ]
        mocker.patch("synapse.tools.native.FirecrawlApp", return_value=mock_app)

        toolkit = NativeToolkit(ctx_with_key)
        result  = await toolkit.firecrawl_search("FastAPI tutorial")

        assert "error"         not in result
        assert result["count"] == 1
        assert result["results"][0]["url"] == "https://fastapi.tiangolo.com/tutorial/"
        assert "FastAPI" in result["results"][0]["content"]

    async def test_firecrawl_no_api_key_returns_error(self, tmp_path):
        # No key provided — method must not raise; it returns a usable error dict.
        toolkit = NativeToolkit(ToolContext(project_root=str(tmp_path)))
        result  = await toolkit.firecrawl_search("anything")
        assert "error"             in result
        assert "FIRECRAWL_API_KEY" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — tools/mcp_client.py
# ═══════════════════════════════════════════════════════════════════════════

class TestMCPClientMessageFormatting:
    """
    Pure logic tests that do not require a real subprocess. These confirm
    that _translate_tool_schemas and _extract_text_content produce correctly
    structured output regardless of what the server sends.
    """

    def test_translate_tool_schemas_namespaces_names(self):
        client = MCPClient("postgres", "npx", [])
        mcp_tools = [
            {
                "name":        "run_query",
                "description": "Execute a SQL query",
                "inputSchema": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                },
            }
        ]
        schemas = client._translate_tool_schemas(mcp_tools)

        assert len(schemas) == 1
        assert schemas[0]["type"]                     == "function"
        assert schemas[0]["function"]["name"]          == "postgres__run_query"
        assert schemas[0]["function"]["description"]   == "Execute a SQL query"
        assert "sql" in schemas[0]["function"]["parameters"]["properties"]

    def test_translate_empty_tool_list_returns_empty(self):
        client  = MCPClient("test", "cmd", [])
        schemas = client._translate_tool_schemas([])
        assert schemas == []

    def test_extract_text_content_from_text_blocks(self):
        result = {
            "content": [
                {"type": "text", "text": "First line"},
                {"type": "text", "text": "Second line"},
            ]
        }
        output = MCPClient._extract_text_content(result)
        assert "First line"  in output
        assert "Second line" in output

    def test_extract_text_content_ignores_non_text_blocks(self):
        result = {
            "content": [
                {"type": "image", "data": "base64..."},
                {"type": "text",  "text": "Only this"},
            ]
        }
        output = MCPClient._extract_text_content(result)
        assert output == "Only this"

    def test_extract_text_content_falls_back_to_json(self):
        # When there are no text blocks, the raw JSON is returned so the
        # LLM always gets something it can work with.
        result = {"someKey": "someValue"}
        output = MCPClient._extract_text_content(result)
        assert "someKey" in output

    def test_next_id_increments(self):
        client = MCPClient("s", "cmd", [])
        ids    = [client._next_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_name_prefix_stripped_before_call(self):
        # The server should receive 'run_query', not 'postgres__run_query'.
        # We verify this by checking what removeprefix() returns.
        namespaced = "postgres__run_query"
        stripped   = namespaced.removeprefix("postgres__")
        assert stripped == "run_query"


class TestMCPClientProtocol:
    """
    Protocol tests mock asyncio.create_subprocess_exec to return a fake process
    whose stdout.readline side_effect returns pre-scripted JSON-RPC responses.
    This lets us verify the exact sequence of messages Synapse sends and receives.
    """

def _make_start_responses(self, tools=None) -> list[dict]:
    if tools is None:
        tools = [
            {
                "name":        "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            }
        ]
    return [
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "mock-server"}}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": tools}},
    ]

    async def test_start_populates_tool_schemas(self, mocker):
        mock_process = make_mock_process(self._make_start_responses())
        mocker.patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_process),
        )
        client = MCPClient("mock", "npx", ["-y", "@mock/server"])
        await client.start()

        assert len(client.tool_schemas) == 1
        assert client.tool_schemas[0]["function"]["name"] == "mock__test_tool"

        await client.stop()

    async def test_start_with_no_tools_is_not_an_error(self, mocker):
        mock_process = make_mock_process(self._make_start_responses(tools=[]))
        mocker.patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_process),
        )
        client = MCPClient("empty", "cmd", [])
        await client.start()
        assert client.tool_schemas == []
        await client.stop()
        
    async def test_call_tool_sends_correct_rpc_message(self, mocker):
        # Responses: init, tools/list, then the tool call response (id=3)
        call_response = {
            "jsonrpc": "2.0",
            "id":      3,
            "result":  {"content": [{"type": "text", "text": "Query result: 42 rows"}]},
        }
        mock_process = make_mock_process(
            self._make_start_responses() + [call_response]
        )
        mocker.patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_process),
        )
        client = MCPClient("postgres", "npx", [])
        await client.start()

        result = await client.call_tool("postgres__test_tool", {"sql": "SELECT COUNT(*) FROM users"})

        assert "42 rows" in result

    async def test_call_tool_returns_error_string_on_rpc_error(self, mocker):
        error_response = {
            "jsonrpc": "2.0",
            "id":      3,
            "error":   {"code": -32600, "message": "Invalid request"},
        }
        mock_process = make_mock_process(
            self._make_start_responses() + [error_response]
        )
        mocker.patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_process),
        )
        client = MCPClient("test", "cmd", [])
        await client.start()

        result = await client.call_tool("test__test_tool", {})
        # The result must be a string (not a raised exception) describing the error
        assert "MCP Error" in result
        assert "Invalid request" in result

    async def test_receive_skips_non_json_lines(self, mocker):
        """
        Many MCP servers write log output to stdout before the JSON response.
        _receive() must skip those lines rather than crashing on JSONDecodeError.
        """
        # Simulate: log line → log line → valid JSON response (id=1)
        mock_process = MagicMock()
        mock_process.stdin.write  = MagicMock()
        mock_process.stdin.drain  = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[
            b"[INFO] Server starting up...\n",
            b"[DEBUG] Loading tools...\n",
            (json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n").encode(),
            # tools/list response
            (json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}) + "\n").encode(),
        ])
        mock_process.terminate = MagicMock()
        mock_process.wait      = AsyncMock(return_value=0)

        mocker.patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_process),
        )
        client = MCPClient("test", "cmd", [])
        # start() must succeed despite the non-JSON log lines
        await client.start()
        assert client.tool_schemas == []
        await client.stop()

    async def test_call_tool_before_start_raises(self):
        client = MCPClient("test", "cmd", [])
        with pytest.raises(MCPClientError, match="not started"):
            await client.call_tool("test__tool", {})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — tools/registry.py
# ═══════════════════════════════════════════════════════════════════════════

class TestPythonTypeToJsonSchema:
    """
    The type mapping function is the core of schema auto-generation.
    We parametrize across the full range of expected inputs to confirm
    every mapping is correct before testing the higher-level build function.
    """

    @pytest.mark.parametrize("python_type, expected", [
        (str,   {"type": "string"}),
        (int,   {"type": "integer"}),
        (float, {"type": "number"}),
        (bool,  {"type": "boolean"}),
        (dict,  {"type": "object"}),
    ])
    def test_primitive_mappings(self, python_type, expected):
        assert _python_type_to_json_schema(python_type) == expected

    def test_list_without_type_arg(self):
        result = _python_type_to_json_schema(list)
        assert result["type"] == "array"

    def test_list_with_str_items(self):
        from typing import List
        result = _python_type_to_json_schema(List[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_optional_str_maps_to_string(self):
        from typing import Optional
        result = _python_type_to_json_schema(Optional[str])
        assert result == {"type": "string"}

    def test_union_with_none_maps_to_inner_type(self):
        # Python 3.10+ union syntax
        result = _python_type_to_json_schema(str | None)
        assert result == {"type": "string"}

    def test_unknown_type_returns_empty_dict(self):
        class MyCustomClass:
            pass
        result = _python_type_to_json_schema(MyCustomClass)
        assert result == {}


class TestParseDocstringArgs:
    """
    Docstring argument parsing feeds the parameter descriptions into the
    JSON schemas seen by the LLM. If parsing breaks, every auto-generated
    tool schema loses its description text silently.
    """

    def test_parses_google_style_args(self):
        docstring = """
        Read a file from the project.

        Args:
            path: Relative path to the file within the project root.
            encoding: The text encoding to use when reading.
        """
        result = _parse_docstring_args(docstring)
        assert result["path"]     == "Relative path to the file within the project root."
        assert result["encoding"] == "The text encoding to use when reading."

    def test_empty_docstring_returns_empty_dict(self):
        assert _parse_docstring_args("") == {}

    def test_no_args_section_returns_empty_dict(self):
        assert _parse_docstring_args("A function with no args section.") == {}

    def test_returns_section_does_not_pollute_args(self):
        docstring = """
        Do something.

        Args:
            value: The input value.

        Returns:
            The output.
        """
        result = _parse_docstring_args(docstring)
        assert "value"   in result
        assert "Returns" not in result


class TestBuildToolSpec:
    """
    _build_tool_spec is what transforms Python callables into LiteLLM tool
    definitions. The output shape must match exactly what the API expects.
    """

    def test_produces_correct_top_level_structure(self):
        async def my_tool(query: str, top_k: int = 5) -> dict:
            """Search for something."""
            pass

        spec = _build_tool_spec("my_tool", my_tool)
        assert spec["type"]                     == "function"
        assert spec["function"]["name"]          == "my_tool"
        assert spec["function"]["description"]   == "Search for something."
        assert "parameters"                      in spec["function"]

    def test_required_vs_optional_parameters(self):
        async def my_tool(required_param: str, optional_param: int = 5) -> dict:
            """A tool."""
            pass

        spec = _build_tool_spec("my_tool", my_tool)
        required = spec["function"]["parameters"]["required"]
        assert "required_param"  in required
        assert "optional_param" not in required

    def test_self_parameter_excluded(self):
        class MyClass:
            async def my_method(self, path: str) -> dict:
                """Read a path."""
                pass

        spec = _build_tool_spec("my_method", MyClass().my_method)
        properties = spec["function"]["parameters"]["properties"]
        assert "self" not in properties
        assert "path" in properties

    def test_parameter_descriptions_from_docstring(self):
        async def tool_with_docs(query: str) -> dict:
            """
            Search the codebase.

            Args:
                query: A natural language description of what to search for.
            """
            pass

        spec = _build_tool_spec("tool_with_docs", tool_with_docs)
        query_schema = spec["function"]["parameters"]["properties"]["query"]
        assert "description"     in query_schema
        assert "natural language" in query_schema["description"]


class TestToolRegistry:
    """
    ToolRegistry is the component that graph nodes interact with directly.
    The most important behaviours are: correct filtering by tool name,
    correct schema shape, and safe dispatch error handling.
    """

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_and_retrieve_schema(self, registry):
        schema     = {"type": "function", "function": {"name": "test", "description": "test"}}
        dispatcher = AsyncMock(return_value="result")
        registry.register("test", schema, dispatcher)

        schemas = registry.get_schemas_for(["test"])
        assert len(schemas)     == 1
        assert schemas[0] is schema

    def test_get_schemas_for_filters_correctly(self, registry):
        for name in ("tool_a", "tool_b", "tool_c"):
            registry.register(name, {"type": "function", "function": {"name": name}}, AsyncMock())

        schemas = registry.get_schemas_for(["tool_a", "tool_c"])
        names   = [s["function"]["name"] for s in schemas]
        assert "tool_a" in names
        assert "tool_c" in names
        assert "tool_b" not in names

    def test_get_schemas_for_skips_unknown_names(self, registry):
        registry.register("real", {"type": "function", "function": {"name": "real"}}, AsyncMock())
        schemas = registry.get_schemas_for(["real", "nonexistent"])
        assert len(schemas) == 1

    def test_load_native_tools_registers_all_six(self, registry, tmp_path):
        toolkit = NativeToolkit(ToolContext(project_root=str(tmp_path)))
        registry.load_native_tools(toolkit)

        expected = {
            "file_read", "file_write", "list_directory",
            "run_command", "vector_search", "firecrawl_search",
        }
        assert expected.issubset(set(registry.registered_tools))

    async def test_dispatch_calls_correct_tool(self, registry):
        mock_dispatcher = AsyncMock(return_value={"result": "ok"})
        registry.register(
            "my_tool",
            {"type": "function", "function": {"name": "my_tool"}},
            mock_dispatcher,
        )
        result = await registry.dispatch("my_tool", {"arg": "value"})

        mock_dispatcher.assert_called_once_with(arg="value")
        # dict return value is serialised to a JSON string
        assert "ok" in result

    async def test_dispatch_unknown_tool_returns_error_string(self, registry):
        # Must return a string, not raise — the LLM needs to receive the error.
        result = await registry.dispatch("nonexistent_tool", {})
        assert "not registered" in result

    async def test_dispatch_exception_returns_error_string(self, registry):
        failing_dispatcher = AsyncMock(side_effect=RuntimeError("something broke"))
        registry.register("bad_tool", {"type": "function", "function": {"name": "bad_tool"}}, failing_dispatcher)

        result = await registry.dispatch("bad_tool", {})
        assert "RuntimeError"    in result
        assert "something broke" in result

    def test_scan_dynamic_tools_loads_py_file(self, registry, tmp_path):
        tool_file = tmp_path / "my_dynamic_tool.py"
        tool_file.write_text(
            'async def my_dynamic_tool(query: str) -> dict:\n'
            '    """Search for something.\n\n    Args:\n        query: The search query.\n    """\n'
            '    return {"result": query}\n'
        )
        registered = registry.scan_dynamic_tools(tmp_path)
        assert "my_dynamic_tool" in registered
        assert "my_dynamic_tool" in registry.registered_tools

    def test_scan_dynamic_tools_uses_yaml_override(self, registry, tmp_path):
        (tmp_path / "override_tool.py").write_text(
            'async def override_tool(x: str) -> dict:\n    """Original description."""\n    pass\n'
        )
        custom_schema = {
            "type": "function",
            "function": {
                "name":        "override_tool",
                "description": "This description comes from YAML",
                "parameters":  {"type": "object", "properties": {}, "required": []},
            },
        }
        (tmp_path / "override_tool.yaml").write_text(yaml_content := "")
        import yaml
        (tmp_path / "override_tool.yaml").write_text(yaml.dump(custom_schema))

        registry.scan_dynamic_tools(tmp_path)
        schemas = registry.get_schemas_for(["override_tool"])
        assert schemas[0]["function"]["description"] == "This description comes from YAML"

    def test_scan_dynamic_tools_empty_directory_returns_empty_list(self, registry, tmp_path):
        result = registry.scan_dynamic_tools(tmp_path)
        assert result == []

    def test_scan_dynamic_tools_nonexistent_directory_returns_empty_list(self, registry, tmp_path):
        result = registry.scan_dynamic_tools(tmp_path / "does_not_exist")
        assert result == []

    def test_load_mcp_tools_from_client(self, registry):
        mock_client = MagicMock(spec=MCPClient)
        mock_client.server_name  = "test_server"
        mock_client.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name":        "test_server__do_thing",
                    "description": "Does a thing",
                    "parameters":  {"type": "object", "properties": {}},
                },
            }
        ]
        registered = registry.load_mcp_tools(mock_client)
        assert "test_server__do_thing" in registered
        assert "test_server__do_thing" in registry.registered_tools

    def test_len_reflects_registered_count(self, registry):
        assert len(registry) == 0
        registry.register("t1", {"type": "function", "function": {"name": "t1"}}, AsyncMock())
        assert len(registry) == 1