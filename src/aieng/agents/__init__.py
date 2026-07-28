"""Agents — Lanham, Huyen ch. 6.

An agent is an LLM in a loop with tools and memory. This package implements the
harness — the part that is *your* responsibility, and therefore the part where
every security property lives.
"""

from aieng.agents.loop import AgentResult, LoopGuard, Scratchpad
from aieng.agents.memory import Memory, MemoryStore
from aieng.agents.tools import Tool, ToolRegistry, ToolResult

__all__ = [
    "AgentResult",
    "LoopGuard",
    "Memory",
    "MemoryStore",
    "Scratchpad",
    "Tool",
    "ToolRegistry",
    "ToolResult",
]
