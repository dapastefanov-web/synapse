"""
config/loader.py — Cascading YAML configuration loader.

Synapse resolves configuration through three layers, applied in order:

  Layer 1 — Defaults:  synapse/config/defaults/  (shipped with the package)
  Layer 2 — Global:    ~/.config/synapse/         (user's machine-wide preferences)
  Layer 3 — Local:     [project]/.synapse/        (project-specific overrides)

Each layer is deep-merged on top of the previous, so a project-local agents.yaml
only needs to contain the agents it wants to change. Everything else falls through
from global or default values automatically.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# The three recognised config filenames
CONFIG_FILES = ("agents.yaml", "pipeline.yaml", "cli_config.yaml")

# Absolute path to the defaults/ directory, resolved relative to this file
# so it works correctly regardless of where the user runs Synapse from.
_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Read a YAML file and return its contents as a dict.
    Returns an empty dict silently if the file does not exist —
    missing optional config layers are not errors.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge `override` into `base`, returning a new dict.

    The recursion handles nested dicts: if both dicts contain the same key
    and both values are dicts, we merge those sub-dicts rather than replacing
    the base value entirely. This allows a project-local config to say
    'change only the model for coder_fast' without having to redefine the
    entire agents block from scratch.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Both sides are dicts — recurse one level deeper
            result[key] = _deep_merge(result[key], value)
        else:
            # Scalar, list, or new key — override directly
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_name: str, project_root: str | None = None) -> dict[str, Any]:
    """
    Load a named config file through the full three-layer cascade.

    Args:
        config_name:  One of 'agents.yaml', 'pipeline.yaml', or 'cli_config.yaml'.
        project_root: Absolute path to the active project directory.
                      When None, the local override layer is skipped entirely.

    Returns:
        A fully merged dict representing the resolved configuration.

    Raises:
        ValueError: If config_name is not one of the three recognised filenames.
    """
    if config_name not in CONFIG_FILES:
        raise ValueError(
            f"Unknown config file '{config_name}'. Must be one of: {CONFIG_FILES}"
        )

    # Layer 1: packaged defaults — always present, the baseline for everything
    config = _load_yaml(_DEFAULTS_DIR / config_name)

    # Layer 2: user's global overrides — applied on top of defaults
    global_path = Path.home() / ".config" / "synapse" / config_name
    config = _deep_merge(config, _load_yaml(global_path))

    # Layer 3: project-specific overrides — highest priority
    if project_root is not None:
        local_path = Path(project_root) / ".synapse" / config_name
        config = _deep_merge(config, _load_yaml(local_path))

    return config


def load_all_configs(project_root: str | None = None) -> dict[str, dict[str, Any]]:
    """
    Load all three config files in a single call.

    Returns a dict with keys 'agents', 'pipeline', and 'cli_config'
    (the .yaml extension is stripped for convenience).
    """
    return {
        "agents":     load_config("agents.yaml",     project_root),
        "pipeline":   load_config("pipeline.yaml",   project_root),
        "cli_config": load_config("cli_config.yaml", project_root),
    }