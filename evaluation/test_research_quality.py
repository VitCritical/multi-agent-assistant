"""
evaluation/test_research_quality.py
Automated evaluation of Research Agent answer quality.
Tests grounding, citations, conflict resolution, and honest uncertainty.
Run: python -m evaluation.test_research_quality
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.corpus_loader import init_vector_store
from src.agents.research_agent import ResearchAgent

init_vector_store()
agent = ResearchAgent()

# ─────────────────────────────────────────────────────────────────
# Test cases: each has a question, expected keywords in answer,
# forbidden keywords, and expected chunk IDs to be cited.
# ─────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "RQ-01",
        "desc": "Cold storage conflict — should say 20 days (retention:0), not 60 days (security:1)",
        "question": "How long are raw run records kept in cold recovery storage?",
        "expected_keywords": ["20 days", "20-day", "20 d"],
        "forbidden_keywords": ["60 days", "60-day"],
        "expected_chunks": ["retention:0"],
        "expect_conflict": True,
    },
    {
        "id": "RQ-02",
        "desc": "On-call stipend conflict — should say $175 (handbook), not $140 (oncall runbook)",
        "question": "What is the weekly on-call stipend?",
        "expected_keywords": ["175", "$175"],
        "forbidden_keywords": ["$140", "140 per week"],
        "expected_chunks": ["handbook:1"],
        "expect_conflict": True,
    },
    {
        "id": "RQ-03",
        "desc": "AQL timeout — should say 50 seconds (v2.3)",
        "question": "What is the AQL query timeout in the current version?",
        "expected_keywords": ["50 second", "50-second"],
        "forbidden_keywords": ["25 second"],
        "expected_chunks": ["query:0"],
        "expect_conflict": False,
    },
    {
        "id": "RQ-04",
        "desc": "Launch plan EU projects — should say NO",
        "question": "Can a Launch plan workspace create EU projects?",
        "expected_keywords": ["no", "cannot", "only us", "us projects only", "launch cannot"],
        "forbidden_keywords": [],
        "expected_chunks": ["regions:0", "pricing:2"],
        "expect_conflict": False,
    },
    {
        "id": "RQ-05",
        "desc": "HIPAA / healthcare cert — should say not stated / no claims",
        "question": "Is Asterline HIPAA certified or does it have healthcare certifications?",
        "expected_keywords": ["no claims", "not stated", "does not state", "not mentioned", "not found"],
        "forbidden_keywords": ["hipaa certified", "is hipaa"],
        "expected_chunks": ["security:2"],
        "expect_conflict": False,
    },
    {
        "id": "RQ-06",
        "desc": "Root cause of INC-2025-08",
        "question": "What caused the INC-2025-08 ingestion outage?",
        "expected_keywords": ["heartbeat", "consumer", "queuefin", "configuration mismatch"],
        "forbidden_keywords": [],
        "expected_chunks": ["incident-08:1"],
        "expect_conflict": False,
    },
    {
        "id": "RQ-07",
        "desc": "Mobile app — should say no 2026 mobile app planned",
        "question": "Is there a mobile app planned for Asterline?",
        "expected_keywords": ["no mobile", "not planned", "no app"],
        "forbidden_keywords": [],
        "expected_chunks": ["faq:1"],
        "expect_conflict": False,
    },
    {
        "id": "RQ-08",
        "desc": "Team ingestion rate limit",
        "question": "What is the ingestion rate limit for Team plan projects?",
        "expected_keywords": ["1500", "1,500"],
        "forbidden_keywords": [],
        "expected_chunks": ["ingest:1"],
        "expect_conflict": False,
    },
]


def score_result(tc: dict, result: dict) -> dict:
    answer_lower = result["answer"].lower()

    # Keyword checks
    keyword_hit = any(kw.lower() in answer_lower for kw in tc["expected_keywords"])
    forbidden_hit = any(kw.lower() in answer_lower for kw in tc.get("forbidden_keywords", []))

    # Citation check
    cited = result.get("cited_chunk_ids", [])
    citation_hit = any(c in cited for c in tc.get("expected_chunks", []))

    # Conflict check
    conflicts_present = len(result.get("conflicts_detected", [])) > 0
    conflict_ok = (not tc["expect_conflict"]) or conflicts_present

    passed = keyword_hit and not forbidden_hit and citation_hit and conflict_ok

    return {
        "id": tc["id"],
        "desc": tc["desc"],
        "passed": passed,
        "keyword_hit": keyword_hit,
        "forbidden_hit": forbidden_hit,
        "citation_hit": citation_hit,
        "conflict_detected": conflicts_present,
        "conflict_expected": tc["expect_conflict"],
        "answer_snippet": result["answer"][:300],
        "citations": cited,
    }


def run_evaluation():
    results = []
    print("\n" + "="*60)
    print("RESEARCH QUALITY EVALUATION")
    print("="*60)

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['desc']}")
        print(f"  Q: {tc['question']}")
        result = agent.run(tc["question"])
        score = score_result(tc, result)
        results.append(score)
        status = "✅ PASS" if score["passed"] else "❌ FAIL"
        print(f"  {status}")
        if not score["keyword_hit"]:
            print(f"    ⚠ Missing expected keywords: {tc['expected_keywords']}")
        if score["forbidden_hit"]:
            print(f"    ⚠ Forbidden keyword found: {tc['forbidden_keywords']}")
        if not score["citation_hit"]:
            print(f"    ⚠ Expected chunk not cited. Got: {score['citations']}")
        if tc["expect_conflict"] and not score["conflict_detected"]:
            print(f"    ⚠ Conflict expected but not detected")
        print(f"    Answer: {score['answer_snippet']}…")

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} passed ({100*passed//total}%)")
    print("="*60)

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/research_quality.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to evaluation/results/research_quality.json")
    return results


if __name__ == "__main__":
    run_evaluation()
