"""
agents/orchestrator.py — Orchestrator Agent (LangGraph StateGraph)
Responsibility:
  1. Load user memory via Memory Agent
  2. Route message to Research Agent (or handle memory-only commands directly)
  3. Decide what new context to store back via Memory Agent
  4. Return final answer to the UI
Traced end-to-end via LangSmith.
"""
import json
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langsmith import traceable

from src.agents.memory_agent import MemoryAgent
from src.agents.research_agent import ResearchAgent
from src.agents.evidence_agent import EvidenceAgent
from src.database import get_session
from src.models import Conversation, Message


# ── State schema ──────────────────────────────────────────────────

class AgentState(TypedDict):
    user_id: str
    conversation_id: str
    user_message: str
    user_memory_summary: str
    memory_intent: Optional[dict]          # detected memory command
    retrieved_chunks: Optional[List[dict]] # from Research Agent retrieval
    evidence_evaluation: Optional[dict]    # from Evidence Agent evaluation
    research_result: Optional[dict]        # synthesis result
    final_answer: str
    citations: List[str]
    conflicts: List[str]
    memory_to_store: Optional[dict]        # {"type": ..., "content": ...}
    error: Optional[str]


# ── Node functions ────────────────────────────────────────────────

def node_load_memory(state: AgentState) -> AgentState:
    """Node 1: Load user memory context."""
    ma = MemoryAgent(state["user_id"])

    # Check for explicit memory commands first
    intent = ma.detect_memory_intent(state["user_message"])
    state["memory_intent"] = intent

    # Always load context summary for RAG context injection
    summary = ma.load_context()
    state["user_memory_summary"] = summary
    return state


def node_handle_memory_command(state: AgentState) -> AgentState:
    """Node 2a: Handle memory-only commands (forget / list)."""
    ma = MemoryAgent(state["user_id"])
    intent = state["memory_intent"]
    action = intent["action"]
    payload = intent["payload"]

    if action == "clear_all":
        result = ma.clear_all()
        state["final_answer"] = f"Done. {result}"
    elif action == "forget_topic":
        result = ma.forget_topic(payload)
        state["final_answer"] = f"Done. {result}"
    elif action == "forget_by_id":
        result = ma.forget_by_id(payload)
        state["final_answer"] = f"Done. {result}"
    elif action == "list":
        memories = ma.list_memories()
        if not memories:
            state["final_answer"] = "I have no stored context about you yet."
        else:
            lines = ["Here's what I remember about you:\n"]
            for m in memories:
                lines.append(f"• [{m['type']}] {m['content']}")
            state["final_answer"] = "\n".join(lines)
    else:
        state["final_answer"] = "Memory command processed."

    state["citations"] = []
    state["conflicts"] = []
    return state


def node_research_retrieve(state: AgentState) -> AgentState:
    """Node 2b: Run Research Agent to retrieve relevant KB chunks."""
    ra = ResearchAgent()
    chunks = ra.retrieve(query=state["user_message"])
    state["retrieved_chunks"] = chunks
    return state


def node_evidence_evaluate(state: AgentState) -> AgentState:
    """Node 2c: Run Evidence Agent to evaluate authority and detect conflicts."""
    ea = EvidenceAgent()
    eval_result = ea.evaluate(chunks=state.get("retrieved_chunks", []), query=state["user_message"])
    state["evidence_evaluation"] = eval_result
    state["conflicts"] = eval_result.get("conflict_notes", [])
    return state


def node_research_synthesize(state: AgentState) -> AgentState:
    """Node 2d: Run Research Agent to generate grounded answer with citations."""
    ra = ResearchAgent()
    eval_result = state.get("evidence_evaluation") or {}
    ranked_chunks = eval_result.get("ranked_chunks") or state.get("retrieved_chunks") or []
    conflict_notes = eval_result.get("conflict_notes") or state.get("conflicts") or []

    result = ra.generate_answer(
        query=state["user_message"],
        chunks=ranked_chunks,
        conflict_notes=conflict_notes,
        user_context=state["user_memory_summary"],
    )
    state["research_result"] = result
    state["final_answer"] = result["answer"]
    state["citations"] = result["cited_chunk_ids"]
    return state


def node_store_memory(state: AgentState) -> AgentState:
    """
    Node 3: Orchestrator decides what (if anything) to store from this turn.
    Looks for correction / preference / investigation signals.
    Memory ≠ KB facts — only user-asserted context is stored.
    """
    msg = state["user_message"].lower()
    ma = MemoryAgent(state["user_id"])

    # Correction pattern: "actually", "I meant", "that's wrong"
    correction_signals = ["actually", "i meant", "that's wrong", "incorrect", "not right", "correction"]
    if any(s in msg for s in correction_signals):
        ma.store("correction", state["user_message"], source="user_stated")

    # Preference pattern: "i prefer", "i like", "always", "please always"
    pref_signals = ["i prefer", "i like", "please always", "i want you to"]
    if any(s in msg for s in pref_signals):
        ma.store("preference", state["user_message"], source="user_stated")

    # Investigation pattern: "investigating", "looking into", "i'm working on"
    invest_signals = ["investigating", "looking into", "i'm working on", "i am working on", "debug"]
    if any(s in msg for s in invest_signals):
        ma.store("investigation", state["user_message"], source="user_stated")

    # User context: plan / team / workspace statements
    context_signals = ["my team", "we are on", "our plan", "my workspace", "i am on", "we use"]
    if any(s in msg for s in context_signals):
        ma.store("context", state["user_message"], source="user_stated")

    return state


def node_persist_message(state: AgentState) -> AgentState:
    """Node 4: Save conversation turn to DB for continuity."""
    with get_session() as session:
        # Ensure conversation exists
        conv = session.query(Conversation).filter_by(id=state["conversation_id"]).first()
        if not conv:
            conv = Conversation(id=state["conversation_id"], user_id=state["user_id"])
            session.add(conv)

        # User message
        session.add(Message(
            conversation_id=state["conversation_id"],
            role="user",
            content=state["user_message"],
            chunk_ids_cited="[]",
        ))

        # Assistant answer
        session.add(Message(
            conversation_id=state["conversation_id"],
            role="assistant",
            content=state["final_answer"],
            chunk_ids_cited=json.dumps(state.get("citations", [])),
        ))
        session.commit()
    return state


# ── Routing ───────────────────────────────────────────────────────

def route_after_memory_load(state: AgentState) -> str:
    """Router: memory command → handle_memory, else → research."""
    if state.get("memory_intent"):
        return "handle_memory_command"
    return "research"


# ── Build the Graph ───────────────────────────────────────────────

def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("load_memory",           node_load_memory)
    graph.add_node("handle_memory_command", node_handle_memory_command)
    graph.add_node("research_retrieve",     node_research_retrieve)
    graph.add_node("evidence_evaluate",     node_evidence_evaluate)
    graph.add_node("research_synthesize",   node_research_synthesize)
    graph.add_node("store_memory",          node_store_memory)
    graph.add_node("persist_message",       node_persist_message)

    graph.set_entry_point("load_memory")

    graph.add_conditional_edges(
        "load_memory",
        route_after_memory_load,
        {
            "handle_memory_command": "handle_memory_command",
            "research": "research_retrieve",
        },
    )

    # memory commands → persist (no store_memory needed)
    graph.add_edge("handle_memory_command", "persist_message")

    # research pipeline: retrieve → evaluate evidence → synthesize → store_memory → persist
    graph.add_edge("research_retrieve",   "evidence_evaluate")
    graph.add_edge("evidence_evaluate",   "research_synthesize")
    graph.add_edge("research_synthesize", "store_memory")
    graph.add_edge("store_memory",        "persist_message")
    graph.add_edge("persist_message",     END)

    return graph.compile()


# ── Public Orchestrator class ─────────────────────────────────────

class Orchestrator:
    def __init__(self):
        self._graph = build_graph()

    @traceable(name="Orchestrator.run")
    def run(self, user_id: str, conversation_id: str, user_message: str) -> dict:
        """
        Entry point for a single conversation turn.
        Returns dict with answer, citations, conflicts.
        """
        initial_state: AgentState = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_message": user_message,
            "user_memory_summary": "",
            "memory_intent": None,
            "retrieved_chunks": None,
            "evidence_evaluation": None,
            "research_result": None,
            "final_answer": "",
            "citations": [],
            "conflicts": [],
            "memory_to_store": None,
            "error": None,
        }

        final_state = self._graph.invoke(initial_state)

        return {
            "answer": final_state["final_answer"],
            "citations": final_state.get("citations", []),
            "conflicts": final_state.get("conflicts", []),
            "research_result": final_state.get("research_result"),
        }
