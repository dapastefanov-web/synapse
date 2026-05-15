"""
tools/registry.py — The Synapse Tool Registry.

Unification layer that makes all tools — native, dynamic, and MCP — look
identical to agent nodes. Each tool has:
  - A schema dict (LiteLLM/OpenAI tool spec for the API call's 'tools' parameter)
  - A dispatcher callable (what actually runs when the LLM invokes the tool)

The schema generation pipeline (for native and dynamic tools):
  Python function → inspect.signature() → JSON Schema properties
  Python docstring → first line = description, Args: section = parameter descriptions
  Type annotations → _python_type_to_json_schema() → JSON type strings
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import types
import warnings
from pathlib import Path
from typing import Any, Callable, get_args, get_origin, get_type_hints

import yaml

from synapse.tools.native import NativeToolkit
from synapse.tools.mcp_client import MCPClient


# ---------------------------------------------------------------------------
# Type annotation → JSON Schema conversion
# ---------------------------------------------------------------------------

_PRIMITIVE_MAP: dict[type, dict[str, str]] = {
    str:   {"type": "string"},
    int:   {"type": "integer"},
    float: {"type": "number"},
    bool:  {"type": "boolean"},
}


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """
    Recursively convert a Python type annotation to a JSON Schema dict.

    Handled cases:
      str, int, float, bool  →  primitive type schemas
      dict / Dict[...]       →  {"type": "object"}
      list / List[X]         →  {"type": "array", "items": <schema for X>}
      Optional[X] / X | None →  same as schema for X
      Any or unknown         →  {} (permissive — LLM can pass anything)
    """
    import types as builtin_types

    origin = get_origin(python_type)
    args   = get_args(python_type)

    # Python 3.10+ union syntax: X | Y  →  types.UnionType
    if isinstance(python_type, builtin_types.UnionType):
        non_none = [a for a in get_args(python_type) if a is not type(None)]
        return _python_type_to_json_schema(non_none[0]) if len(non_none) == 1 else {}

    # typing.Union (also covers Optional[X] = Union[X, None])
    from typing import Union
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        return _python_type_to_json_schema(non_none[0]) if len(non_none) == 1 else {}

    # list / List[X]
    if origin is list or python_type is list:
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    # if origin is list:
    #     item_schema = _python_type_to_json_schema(args[0]) if args else {}
    #     return {"type": "array", "items": item_schema}

    # dict / Dict[K, V]
    if origin is dict or python_type is dict:
        return {"type": "object"}

    # Primitives
    if python_type in _PRIMITIVE_MAP:
        return _PRIMITIVE_MAP[python_type]

    return {}  # unknown — permissive


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """
    Extract parameter descriptions from a Google-style docstring Args section.

    Given a docstring containing:
        Args:
            path: Path to the file, relative to the project root.
            top_k: Number of results to return.

    Returns: {"path": "Path to the file, relative to the project root.", "top_k": "Number of results to return."}

    Returns an empty dict if there is no Args section or the format differs.
    """
    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    in_args_section                = False
    current_param: str | None      = None

    for line in docstring.splitlines():
        stripped = line.strip()

        if stripped == "Args:":
            in_args_section = True
            continue

        # A new top-level header ends the Args section
        _DOCSTRING_SECTIONS = {"Returns:", "Raises:", "Yields:", "Note:", "Notes:", "Example:", "Examples:"}

        if in_args_section and stripped in _DOCSTRING_SECTIONS:
            in_args_section = False
            continue

        if in_args_section and stripped:
            if ":" in stripped and not stripped.startswith(" "):
                # New parameter line: "    param_name: Description text."
                param, _, desc      = stripped.partition(":")
                current_param       = param.strip()
                descriptions[current_param] = desc.strip()
            elif current_param:
                # Continuation line for the current parameter
                descriptions[current_param] += " " + stripped

    return descriptions


def _build_tool_spec(name: str, func: Callable) -> dict[str, Any]:
    """
    Build a complete LiteLLM tool specification from a Python callable.

    The function's first non-empty docstring line becomes the description.
    Parameters without defaults are marked as required.
    The 'self' and 'cls' parameters are always excluded.
    """
    docstring   = inspect.getdoc(func) or ""
    description = next(
        (line.strip() for line in docstring.splitlines() if line.strip()),
        name,
    )
    arg_descriptions = _parse_docstring_args(docstring)

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    sig         = inspect.signature(func)
    properties: dict[str, Any] = {}
    required:   list[str]      = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        type_schema = _python_type_to_json_schema(hints.get(param_name, Any))
        prop        = {**type_schema}

        if param_name in arg_descriptions:
            prop["description"] = arg_descriptions[param_name]

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name":        name,
            "description": description,
            "parameters":  {
                "type":       "object",
                "properties": properties,
                "required":   required,
            },
        },
    }


def _load_module_from_path(path: Path) -> types.ModuleType:
    """
    Dynamically load a Python module from a file path using importlib.
    Prefixes the module name to avoid shadowing installed packages.
    """
    module_name = f"synapse_dyn_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tool module from '{path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_callable(module: types.ModuleType) -> Callable:
    """
    Find the primary callable in a dynamically loaded tool module.

    Preference order:
      1. A function whose name matches the module file stem.
      2. The first public non-underscore function defined in this module.
    """
    stem = module.__name__.removeprefix("synapse_dyn_")
    if hasattr(module, stem) and callable(getattr(module, stem)):
        return getattr(module, stem)

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_") and obj.__module__ == module.__name__:
            return obj

    raise ImportError(
        f"Tool module '{module.__file__}' must define a public function "
        f"named '{stem}' or at least one non-underscore function."
    )


def _make_mcp_dispatcher(client: MCPClient, tool_name: str) -> Callable:
    """
    Factory that creates an async dispatcher for a single MCP tool.

    Using a factory function rather than a lambda inside a loop is the
    standard Python fix for the loop-closure variable capture bug — each
    call to this factory produces a new scope with its own binding of
    client and tool_name.
    """
    async def dispatcher(**kwargs: Any) -> str:
        return await client.call_tool(tool_name, kwargs)
    return dispatcher


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    The central tool registry for all Synapse agents.

    Three tool sources are registered here:
      1. Native tools (NativeToolkit) — always available at startup
      2. Dynamic tools — scanned from ~/.config/synapse/tools/ and .synapse/tools/
      3. MCP tools — discovered from running MCP server clients

    At runtime, agent nodes use two methods:
      get_schemas_for(tool_names)  — returns the filtered list for the LLM API call
      dispatch(tool_name, args)    — executes the tool and returns a result string
    """

    def __init__(self) -> None:
        self._schemas:     dict[str, dict[str, Any]] = {}
        self._dispatchers: dict[str, Callable]        = {}

    # ── Registration ───────────────────────────────────────────────────────

    def register(
        self,
        tool_name:  str,
        schema:     dict[str, Any],
        dispatcher: Callable,
    ) -> None:
        """Register a tool. Overwrites any existing registration with the same name."""
        self._schemas[tool_name]     = schema
        self._dispatchers[tool_name] = dispatcher

    def load_native_tools(self, toolkit: NativeToolkit) -> None:
        """
        Register all six NativeToolkit methods, auto-generating their schemas
        from type hints and docstrings via _build_tool_spec.
        """
        for method_name in (
            "file_read",
            "file_write",
            "list_directory",
            "run_command",
            "vector_search",
            "firecrawl_search",
        ):
            method = getattr(toolkit, method_name)
            self.register(method_name, _build_tool_spec(method_name, method), method)

    def scan_dynamic_tools(self, tools_dir: str | Path) -> list[str]:
        """
        Scan a directory for .py tool files and register any found.

        For each .py file:
          - If a same-stem .yaml file exists, use it as the manual schema override.
          - Otherwise, auto-generate the schema from the function's type hints.

        Returns the names of successfully registered tools. Failed loads are
        warned rather than raised so a broken tool file doesn't crash Synapse.
        """
        tools_dir  = Path(tools_dir)
        registered: list[str] = []

        if not tools_dir.exists():
            return registered

        for py_file in sorted(tools_dir.glob("*.py")):
            tool_name = py_file.stem
            try:
                module = _load_module_from_path(py_file)
                func   = _find_callable(module)

                yaml_file = py_file.with_suffix(".yaml")
                if yaml_file.exists():
                    with yaml_file.open("r", encoding="utf-8") as fh:
                        schema = yaml.safe_load(fh)
                else:
                    schema = _build_tool_spec(tool_name, func)

                self.register(tool_name, schema, func)
                registered.append(tool_name)

            except Exception as e:
                warnings.warn(
                    f"Skipping dynamic tool '{py_file.name}': {e}",
                    stacklevel=2,
                )

        return registered

    def load_mcp_tools(self, mcp_client: MCPClient) -> list[str]:
        """
        Register all tools exposed by a connected MCP server.
        Must be called after mcp_client.start() so tool_schemas is populated.
        Returns the list of registered tool names.
        """
        registered: list[str] = []
        for schema in mcp_client.tool_schemas:
            tool_name = schema["function"]["name"]
            self.register(tool_name, schema, _make_mcp_dispatcher(mcp_client, tool_name))
            registered.append(tool_name)
        return registered

    # ── Query and dispatch ─────────────────────────────────────────────────

    def get_schemas_for(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """
        Return LiteLLM tool specs for a specific set of tool names.
        Unknown names are silently skipped.
        """
        return [self._schemas[n] for n in tool_names if n in self._schemas]

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call and return its result as a string.

        Returns an informative error string rather than raising on failure,
        so the LLM always receives a response it can reason about.
        """
        if tool_name not in self._dispatchers:
            return (
                f"Error: tool '{tool_name}' is not registered. "
                f"Known tools: {', '.join(sorted(self._schemas))}"
            )
        try:
            result = await self._dispatchers[tool_name](**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {type(e).__name__}: {e}"

    @property
    def registered_tools(self) -> list[str]:
        """All registered tool names, sorted."""
        return sorted(self._schemas.keys())

    def __len__(self) -> int:
        return len(self._schemas)