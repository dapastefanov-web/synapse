# Agents package. Public API is intentionally minimal — graph nodes import
# the node functions directly from their respective modules rather than
# going through this file, keeping import chains explicit and traceable.
from synapse.agents.base import call_agent

__all__ = ["call_agent"]