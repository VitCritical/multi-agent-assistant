"""
agents/memory_agent.py — Memory Agent
Responsibility: read, write, forget, and summarise per-user memory.
All operations are strictly scoped to user_id.
Traced via LangSmith.
"""
from typing import Optional
from langsmith import traceable

from src.tools.memory_tools import (
    store_memory,
    get_memories,
    forget_memory,
    forget_memories_by_topic,
    clear_all_memories,
    summarise_memories,
)


class MemoryAgent:
    """
    Distinct agent responsible for ALL memory operations.
    Called by the Orchestrator — never directly from the UI.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

    @traceable(name="MemoryAgent.load_context")
    def load_context(self) -> str:
        """
        Load and return a summary of the user's stored memory.
        Called at the start of every conversation turn.
        """
        summary = summarise_memories(self.user_id)
        return summary

    @traceable(name="MemoryAgent.store")
    def store(self, memory_type: str, content: str, source: str = "user_stated") -> str:
        """
        Persist a new memory item.
        memory_type: preference | correction | investigation | context
        """
        result = store_memory(self.user_id, memory_type, content, source)
        return result

    @traceable(name="MemoryAgent.forget_by_id")
    def forget_by_id(self, memory_id: str) -> str:
        """Soft-delete a specific memory by its UUID."""
        return forget_memory(self.user_id, memory_id)

    @traceable(name="MemoryAgent.forget_topic")
    def forget_topic(self, topic: str) -> str:
        """Soft-delete all memories matching a keyword."""
        return forget_memories_by_topic(self.user_id, topic)

    @traceable(name="MemoryAgent.clear_all")
    def clear_all(self) -> str:
        """Wipe all memories for this user."""
        return clear_all_memories(self.user_id)

    @traceable(name="MemoryAgent.list_memories")
    def list_memories(self) -> list:
        """Return raw list of active memories (for the UI memory panel)."""
        return get_memories(self.user_id)

    @traceable(name="MemoryAgent.detect_memory_intent")
    def detect_memory_intent(self, user_message: str) -> Optional[dict]:
        """
        Lightweight intent classifier for memory-specific commands.
        Returns: {"action": ..., "payload": ...} or None if no memory intent.
        """
        msg_lower = user_message.lower().strip()

        # forget intent
        if any(kw in msg_lower for kw in ["forget", "remove", "delete", "clear"]):
            if "everything" in msg_lower or "all" in msg_lower:
                return {"action": "clear_all", "payload": None}
            # extract topic after "forget that" / "forget about X"
            for kw in ["forget about ", "forget ", "remove ", "delete "]:
                if kw in msg_lower:
                    topic = msg_lower.split(kw, 1)[-1].strip().rstrip(".")
                    if topic:
                        return {"action": "forget_topic", "payload": topic}
            return {"action": "forget_topic", "payload": "last"}

        # memory inspection intent
        if any(kw in msg_lower for kw in [
            "what do you remember", "what have you stored",
            "show my memory", "what do you know about me",
        ]):
            return {"action": "list", "payload": None}

        return None
