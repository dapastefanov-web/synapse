"""
tools/native.py — Built-in system tools for Synapse agents.

Design: ToolContext + NativeToolkit
─────────────────────────────────────
External dependencies (Docker client, vector store, API keys) are injected once
at construction time through ToolContext. The LLM-facing methods expose only the
parameters that are semantically meaningful to the task, keeping generated JSON
schemas clean and preventing the LLM from hallucinating infrastructure arguments.

All methods are async. Those wrapping synchronous SDKs (Docker, Firecrawl) use
asyncio.to_thread so the event loop is never blocked while waiting for I/O.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolContext:
    """
    Runtime dependencies injected into NativeToolkit at construction time.

    The orchestrator builds one ToolContext per session and passes it to the
    toolkit. This makes the toolkit fully testable — in tests, you provide
    mock objects here rather than setting up real Docker containers or API keys.
    """
    project_root: str

    # Populated after /init-docker completes
    docker_client:       Any | None = None  # docker.DockerClient instance
    docker_container_id: str | None = None  # ID of the running DevContainer

    # Populated after /index-project completes
    vector_store: Any | None = None  # VectorStore instance from memory/vector_store.py

    # Read from environment variable FIRECRAWL_API_KEY at startup
    firecrawl_api_key: str | None = None

    # Which embedding model to use when converting search queries to vectors.
    # Matches the dimension this VectorStore was initialized with.
    embedding_model: str = "text-embedding-3-small"


class NativeToolkitError(Exception):
    """Raised for unrecoverable errors within the native toolkit."""
    pass


def _validate_and_resolve_path(project_root: str, path: str) -> Path:
    """
    Resolve a path and confirm it stays within project_root.

    This is a path traversal guard. An LLM might generate '../../etc/passwd'
    either by mistake or by prompt injection from a malicious source file.
    We resolve both paths to their real absolute forms (normalising all '..'
    segments) and then confirm the result is still a descendant of the root.

    resolve() without strict=True works on non-existent paths (Python 3.6+),
    which is important for file_write creating new files.
    """
    root     = Path(project_root).resolve()
    target   = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise NativeToolkitError(
            f"Path traversal blocked: '{path}' resolves outside the project root '{root}'."
        )

    return target


class NativeToolkit:
    """
    The six built-in tools available to Synapse agents.

    Instantiate with a ToolContext and pass the instance to ToolRegistry.load_native_tools()
    to register all six tools in one call. The registry auto-generates JSON schemas
    from the method signatures and docstrings using Python's inspect module.
    """

    def __init__(self, context: ToolContext) -> None:
        self._ctx = context

    # ── File operations ────────────────────────────────────────────────────

    async def file_read(self, path: str) -> dict[str, Any]:
        """
        Read the full text content of a file in the project.

        Args:
            path: Path to the file, relative to the project root.

        Returns a dict containing 'path' and 'content', or 'error' on failure.
        """
        try:
            resolved = _validate_and_resolve_path(self._ctx.project_root, path)
            if not resolved.exists():
                return {"error": f"File not found: '{path}'"}
            if not resolved.is_file():
                return {"error": f"Path is a directory, not a file: '{path}'"}
            content = resolved.read_text(encoding="utf-8")
            return {"path": path, "content": content}
        except NativeToolkitError as e:
            return {"error": str(e)}
        except OSError as e:
            return {"error": f"Could not read '{path}': {e}"}

    async def file_write(self, path: str, content: str) -> dict[str, Any]:
        """
        Write text content to a file, creating it and any parent directories if needed.

        Args:
            path: Path to the file, relative to the project root.
            content: The complete text content to write. Overwrites any existing content.

        Returns a dict with 'path' and 'bytes_written', or 'error' on failure.
        """
        try:
            resolved = _validate_and_resolve_path(self._ctx.project_root, path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {"path": path, "bytes_written": len(content.encode("utf-8"))}
        except NativeToolkitError as e:
            return {"error": str(e)}
        except OSError as e:
            return {"error": f"Could not write '{path}': {e}"}

    async def list_directory(self, path: str = ".") -> dict[str, Any]:
        """
        List the files and subdirectories inside a project directory.

        Args:
            path: Path to the directory, relative to the project root. Defaults to the project root itself.

        Returns a dict with 'path', 'files', 'directories', and 'total_entries'.
        """
        try:
            resolved = _validate_and_resolve_path(self._ctx.project_root, path)
            if not resolved.exists():
                return {"error": f"Directory not found: '{path}'"}
            if not resolved.is_dir():
                return {"error": f"Path is a file, not a directory: '{path}'"}

            root_resolved = Path(self._ctx.project_root).resolve()
            files:       list[str] = []
            directories: list[str] = []

            for entry in sorted(resolved.iterdir()):
                # Express paths relative to the project root for readability
                display = str(entry.relative_to(root_resolved))
                if entry.is_file():
                    files.append(display)
                elif entry.is_dir():
                    directories.append(display)

            return {
                "path":          path,
                "files":         files,
                "directories":   directories,
                "total_entries": len(files) + len(directories),
            }
        except NativeToolkitError as e:
            return {"error": str(e)}
        except OSError as e:
            return {"error": f"Could not list '{path}': {e}"}

    # ── Docker sandbox ─────────────────────────────────────────────────────

    async def run_command(self, command: str) -> dict[str, Any]:
        """
        Run a shell command inside the project's Docker DevContainer as ai_user.

        ai_user cannot read the .synapse secrets directory, so this is safe
        for general exploration and compilation checks. Test execution that
        requires real API keys uses docker_manager.run_tests() (test_user) instead.

        Args:
            command: The shell command to execute, e.g. 'python -m py_compile src/app.py'.

        Returns a dict with 'stdout', 'exit_code', and 'success'.
        """
        if self._ctx.docker_client is None or self._ctx.docker_container_id is None:
            return {
                "error":     "Docker container not initialised. Run /init-docker first.",
                "stdout":    "",
                "exit_code": -1,
                "success":   False,
            }

        try:
            # exec_run is synchronous — run in a thread to keep the event loop free.
            def _exec() -> Any:
                container = self._ctx.docker_client.containers.get(
                    self._ctx.docker_container_id
                )
                return container.exec_run(
                    cmd=["sh", "-c", command],
                    user="ai_user",
                    stdout=True,
                    stderr=True,
                )

            result = await asyncio.to_thread(_exec)
            output = result.output.decode("utf-8", errors="replace") if result.output else ""

            return {
                "stdout":    output,
                "exit_code": result.exit_code,
                "success":   result.exit_code == 0,
            }
        except Exception as e:
            return {
                "error":     str(e),
                "stdout":    "",
                "exit_code": -1,
                "success":   False,
            }

    # ── Semantic search ────────────────────────────────────────────────────

    async def vector_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """
        Search the project's vector index for files semantically related to a query.

        Args:
            query: A natural language description of what to find, e.g. 'database connection logic'.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns a dict with 'query', 'results' (list of file matches), and 'count'.
        """
        if self._ctx.vector_store is None:
            return {
                "error":   "Vector store not initialised. Run /index-project first.",
                "results": [],
            }

        try:
            import litellm

            embedding_response = await litellm.aembedding(
                model=self._ctx.embedding_model,
                input=[query],
            )
            query_vector = embedding_response.data[0]["embedding"]

            raw_results = self._ctx.vector_store.search(query_vector, top_k=top_k)

            # Truncate content snippets so the results don't flood the context window
            formatted = [
                {
                    "file_path":          r["file_path"],
                    "content_snippet":    r["content"][:500],
                    "similarity_distance": round(r["distance"], 4),
                }
                for r in raw_results
            ]

            return {"query": query, "results": formatted, "count": len(formatted)}

        except Exception as e:
            return {"error": f"Vector search failed: {e}", "results": []}

    # ── Web search ─────────────────────────────────────────────────────────

    async def firecrawl_search(self, query: str) -> dict[str, Any]:
        """
        Search the web for current documentation or reference material using Firecrawl.

        Args:
            query: A specific technical search query, e.g. 'FastAPI lifespan context manager docs'.

        Returns a dict with 'query', 'results' (list with url and content), and 'count'.
        """
        if not self._ctx.firecrawl_api_key:
            return {
                "error":   "FIRECRAWL_API_KEY is not set. Add it to your environment variables.",
                "results": [],
            }

        try:
            from firecrawl import FirecrawlApp

            def _search() -> Any:
                app = FirecrawlApp(api_key=self._ctx.firecrawl_api_key)
                return app.search(query, limit=5)

            raw = await asyncio.to_thread(_search)

            # Normalise across different Firecrawl SDK versions that return
            # either a plain list or a dict with a 'data' key.
            items = raw if isinstance(raw, list) else raw.get("data", [])

            results = [
                {
                    "url":     item.get("url", ""),
                    "title":   item.get("metadata", {}).get("title", item.get("title", "")),
                    "content": item.get("markdown", item.get("content", ""))[:1000],
                }
                for item in items[:5]
            ]

            return {"query": query, "results": results, "count": len(results)}

        except Exception as e:
            return {"error": f"Firecrawl search failed: {e}", "results": []}