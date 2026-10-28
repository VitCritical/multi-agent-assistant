"""
app.py — Single-command launch: `streamlit run app.py`
Streamlit UI: chat interface + memory inspection panel + user selector.
"""
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Initialise DB and vector store on first run ────────────────────
from src.database import init_db, get_all_users
from src.corpus_loader import init_vector_store

@st.cache_resource(show_spinner="🔧 Setting up knowledge base (first run only)…")
def startup():
    init_db()
    init_vector_store()
    return True

startup()

# ── Imports after startup ──────────────────────────────────────────
from src.agents.orchestrator import Orchestrator
from src.agents.memory_agent import MemoryAgent

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Asterline Research Assistant",
    page_icon="🔭",
    layout="wide",
)

# ── Sidebar: user selector ─────────────────────────────────────────
with st.sidebar:
    st.title("🔭 Asterline Research Assistant")
    st.caption("Internal knowledge assistant for Asterline Labs employees.")
    st.divider()

    users = get_all_users()
    user_map = {name: uid for uid, name in users}
    selected_name = st.selectbox("👤 Select user", list(user_map.keys()))
    selected_user_id = user_map[selected_name]

    st.divider()
    st.subheader("🧠 Memory Panel")
    ma = MemoryAgent(selected_user_id)
    memories = ma.list_memories()

    st.caption(f"Active User: **{selected_name}** • ({len(memories)} stored memories)")

    # ── Direct memory management form ──────────────────────────────
    with st.expander("➕ Add Memory Manually", expanded=False):
        with st.form("manual_memory_form", clear_on_submit=True):
            new_type = st.selectbox("Category", ["context", "preference", "investigation", "correction"])
            new_content = st.text_area("Content", placeholder="e.g. My workspace is on the Enterprise plan", height=80)
            submitted = st.form_submit_button("Save Memory", use_container_width=True)
            if submitted and new_content.strip():
                ma.store(new_type, new_content.strip(), source="user_stated")
                st.success(f"Saved {new_type}!")
                st.rerun()

    # ── Display active memories ────────────────────────────────────
    if not memories:
        st.caption("No stored context yet for this user. Tell the assistant what to remember (e.g. 'Remember that my team is on Team plan') or add a memory above.")
    else:
        badge_map = {
            "preference": "🟣",
            "correction": "🔴",
            "investigation": "🟠",
            "context": "🔵",
        }
        for mem in memories:
            badge = badge_map.get(mem["type"], "⚪")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"{badge} **`{mem['type'].upper()}`**  \n{mem['content']}")
                st.caption(f"📅 {mem['created_at'][:10]} • {mem.get('source', 'user_stated')}")
            with col2:
                if st.button("🗑", key=f"del_{mem['id']}", help="Forget this item"):
                    ma.forget_by_id(mem["id"])
                    st.rerun()

    st.divider()
    if st.button("🧹 Clear all memories", use_container_width=True, disabled=len(memories) == 0):
        ma.clear_all()
        st.rerun()

# ── Session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = {}

# Per-user chat history and conversation ID
if selected_user_id not in st.session_state.messages:
    st.session_state.messages[selected_user_id] = []
if selected_user_id not in st.session_state.conversation_id:
    st.session_state.conversation_id[selected_user_id] = str(uuid.uuid4())

current_messages = st.session_state.messages[selected_user_id]
current_conv_id = st.session_state.conversation_id[selected_user_id]

# ── Chat history display ───────────────────────────────────────────
st.header(f"Chat — {selected_name}")

for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citations if present
        if msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])} chunks)"):
                for cid in msg["citations"]:
                    st.code(cid)

        # Show conflict warnings if present
        if msg.get("conflicts"):
            with st.expander("⚠️ Conflict notes"):
                for c in msg["conflicts"]:
                    st.warning(c)

        # Show memory saved badge if present
        if msg.get("memory_stored"):
            mem_item = msg["memory_stored"]
            st.info(f"🧠 **Memory saved:** `[{mem_item['type'].upper()}]` {mem_item['content']}")

# ── Chat input ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about Asterline…"):

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    current_messages.append({"role": "user", "content": prompt})

    # Run agents
    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            orchestrator = Orchestrator()
            result = orchestrator.run(
                user_id=selected_user_id,
                conversation_id=current_conv_id,
                user_message=prompt,
            )

        answer = result["answer"]
        citations = result.get("citations", [])
        conflicts = result.get("conflicts", [])
        mem_stored = result.get("memory_stored")

        st.markdown(answer)

        if citations:
            with st.expander(f"📚 Sources ({len(citations)} chunks)"):
                # Show rich chunk data if available
                chunks_data = {}
                if result.get("research_result") and result["research_result"].get("chunks_data"):
                    chunks_data = {c["chunk_id"]: c for c in result["research_result"]["chunks_data"]}

                for cid in citations:
                    if cid in chunks_data:
                        c = chunks_data[cid]
                        st.markdown(f"**[{cid}]** — {c['title']}  \n`{c['source_url']}`")
                    else:
                        st.code(cid)

        if conflicts:
            with st.expander("⚠️ Conflict notes"):
                for c in conflicts:
                    st.warning(c)

        if mem_stored:
            st.info(f"🧠 **Memory saved:** `[{mem_stored['type'].upper()}]` {mem_stored['content']}")

    # Store in session history
    current_messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "conflicts": conflicts,
        "memory_stored": mem_stored,
    })

    # Refresh sidebar memory panel
    st.rerun()
