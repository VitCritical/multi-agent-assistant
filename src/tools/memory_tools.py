"""
tools/memory_tools.py — all memory read/write/forget/list operations (user-scoped).
Used by the Memory Agent. All DB queries are hard-scoped by user_id.
"""
import json
from datetime import datetime
from typing import List, Optional
from langchain_core.tools import tool

from src.database import get_session
from src.models import Memory, MemoryType


# ── helpers ───────────────────────────────────────────────────────

def _memory_to_dict(m: Memory) -> dict:
    return {
        "id": m.id,
        "type": m.memory_type.value,
        "content": m.content,
        "source": m.source,
        "created_at": m.created_at.isoformat(),
    }


# ── tools (called by Memory Agent) ───────────────────────────────

def store_memory(user_id: str, memory_type: str, content: str, source: str = "user_stated") -> str:
    """Persist a new memory for the given user."""
    try:
        mtype = MemoryType(memory_type)
    except ValueError:
        mtype = MemoryType.context

    with get_session() as session:
        mem = Memory(
            user_id=user_id,
            memory_type=mtype,
            content=content,
            source=source,
        )
        session.add(mem)
        session.commit()
        return f"Stored [{mtype.value}]: {content}"


def get_memories(user_id: str, memory_type: Optional[str] = None) -> List[dict]:
    """Return all active memories for user, optionally filtered by type."""
    with get_session() as session:
        q = session.query(Memory).filter_by(user_id=user_id, is_active=True)
        if memory_type:
            try:
                q = q.filter_by(memory_type=MemoryType(memory_type))
            except ValueError:
                pass
        memories = q.order_by(Memory.created_at.desc()).all()
        return [_memory_to_dict(m) for m in memories]


def forget_memory(user_id: str, memory_id: str) -> str:
    """Soft-delete a specific memory by ID (scoped to user)."""
    with get_session() as session:
        mem = session.query(Memory).filter_by(id=memory_id, user_id=user_id, is_active=True).first()
        if not mem:
            return f"No active memory found with id={memory_id} for this user."
        mem.is_active = False
        mem.updated_at = datetime.utcnow()
        session.commit()
        return f"Forgot memory: {mem.content}"


def forget_memories_by_topic(user_id: str, topic_keyword: str) -> str:
    """Soft-delete all active memories whose content contains a keyword (user-scoped)."""
    with get_session() as session:
        mems = session.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.is_active == True,
            Memory.content.ilike(f"%{topic_keyword}%"),
        ).all()
        if not mems:
            return f"No memories found matching '{topic_keyword}'."
        for m in mems:
            m.is_active = False
            m.updated_at = datetime.utcnow()
        session.commit()
        return f"Forgot {len(mems)} memory/memories related to '{topic_keyword}'."


def clear_all_memories(user_id: str) -> str:
    """Soft-delete ALL active memories for a user."""
    with get_session() as session:
        mems = session.query(Memory).filter_by(user_id=user_id, is_active=True).all()
        for m in mems:
            m.is_active = False
            m.updated_at = datetime.utcnow()
        session.commit()
        return f"Cleared {len(mems)} memories."


def summarise_memories(user_id: str) -> str:
    """
    Return a concise natural-language summary of user memories.
    Used to inject user context into the Orchestrator prompt.
    """
    memories = get_memories(user_id)
    if not memories:
        return "No prior context stored for this user."

    lines = ["User context from memory:"]
    for m in memories:
        lines.append(f"  [{m['type']}] {m['content']}")
    return "\n".join(lines)
