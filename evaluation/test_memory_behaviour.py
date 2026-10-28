"""
evaluation/test_memory_behaviour.py
Tests memory storage, recall, correction, forget, isolation, and persistence.
Run: python -m evaluation.test_memory_behaviour
"""
import sys, os, uuid, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.database import init_db
from src.agents.memory_agent import MemoryAgent

init_db()

RESULTS = []


def check(test_id: str, desc: str, condition: bool, detail: str = ""):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  [{test_id}] {status} — {desc}")
    if detail:
        print(f"    Detail: {detail}")
    RESULTS.append({"id": test_id, "desc": desc, "passed": condition, "detail": detail})
    return condition


def run():
    print("\n" + "="*60)
    print("MEMORY BEHAVIOUR EVALUATION")
    print("="*60)

    # ── MB-01: Store and recall ────────────────────────────────────
    print("\n[MB-01] Store and recall")
    uid_a = str(uuid.uuid4())
    ma = MemoryAgent(uid_a)
    ma.store("context", "I am on the Enterprise plan", source="user_stated")
    mems = ma.list_memories()
    check("MB-01", "Store and retrieve memory",
          any("Enterprise" in m["content"] for m in mems),
          f"Stored: {[m['content'] for m in mems]}")

    # ── MB-02: Forget by topic ─────────────────────────────────────
    print("\n[MB-02] Forget by topic")
    ma.store("preference", "I prefer concise answers", source="user_stated")
    before = len(ma.list_memories())
    ma.forget_topic("concise")
    after = len(ma.list_memories())
    check("MB-02", "Forget by topic reduces memory count",
          after < before,
          f"Before: {before}, After: {after}")

    # ── MB-03: User isolation ──────────────────────────────────────
    print("\n[MB-03] User isolation")
    uid_b = str(uuid.uuid4())
    ma_a = MemoryAgent(uid_a)
    ma_b = MemoryAgent(uid_b)
    ma_a.store("context", "SECRET_USER_A_FACT", source="user_stated")
    b_mems = ma_b.list_memories()
    check("MB-03", "User B cannot see User A's memory",
          not any("SECRET_USER_A_FACT" in m["content"] for m in b_mems),
          f"User B memories: {[m['content'] for m in b_mems]}")

    # ── MB-04: Persistence across instances ────────────────────────
    print("\n[MB-04] Persistence across agent instances")
    uid_c = str(uuid.uuid4())
    ma_c1 = MemoryAgent(uid_c)
    ma_c1.store("investigation", "Investigating duplicate BigQuery rows", source="user_stated")
    # New instance (simulates restart)
    ma_c2 = MemoryAgent(uid_c)
    mems_c2 = ma_c2.list_memories()
    check("MB-04", "Memory persists across agent instances",
          any("BigQuery" in m["content"] for m in mems_c2),
          f"Retrieved: {[m['content'] for m in mems_c2]}")

    # ── MB-05: Correction storage ──────────────────────────────────
    print("\n[MB-05] Correction storage")
    uid_d = str(uuid.uuid4())
    ma_d = MemoryAgent(uid_d)
    ma_d.store("context", "I am on Team plan", source="user_stated")
    ma_d.store("correction", "Actually I am on Enterprise plan, not Team", source="user_stated")
    mems_d = ma_d.list_memories()
    has_correction = any(m["type"] == "correction" for m in mems_d)
    check("MB-05", "Correction is stored with type=correction",
          has_correction,
          f"Memory types: {[m['type'] for m in mems_d]}")

    # ── MB-06: Memory inspection intent detection ──────────────────
    print("\n[MB-06] Memory inspection intent")
    uid_e = str(uuid.uuid4())
    ma_e = MemoryAgent(uid_e)
    intent = ma_e.detect_memory_intent("What do you remember about me?")
    check("MB-06", "Detect memory inspection intent",
          intent is not None and intent["action"] == "list",
          f"Detected intent: {intent}")

    # ── MB-07: Forget all intent ───────────────────────────────────
    print("\n[MB-07] Forget all intent detection")
    ma_e2 = MemoryAgent(uid_e)
    intent2 = ma_e2.detect_memory_intent("Please forget everything you know about me")
    check("MB-07", "Detect clear_all intent",
          intent2 is not None and intent2["action"] == "clear_all",
          f"Detected intent: {intent2}")

    # ── MB-08: Clear all ──────────────────────────────────────────
    print("\n[MB-08] Clear all memories")
    uid_f = str(uuid.uuid4())
    ma_f = MemoryAgent(uid_f)
    ma_f.store("preference", "I like bullet points", source="user_stated")
    ma_f.store("context", "I work in Dublin office", source="user_stated")
    ma_f.clear_all()
    remaining = ma_f.list_memories()
    check("MB-08", "Clear all leaves no active memories",
          len(remaining) == 0,
          f"Remaining: {remaining}")

    # ── Summary ───────────────────────────────────────────────────
    passed = sum(1 for r in RESULTS if r["passed"])
    total = len(RESULTS)
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} passed ({100*passed//total}%)")
    print("="*60)

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/memory_behaviour.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("Results saved to evaluation/results/memory_behaviour.json")
    return RESULTS


if __name__ == "__main__":
    run()
