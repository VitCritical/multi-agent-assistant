"""
agents/evidence_agent.py — Evidence/Conflict Agent
Responsibility: evaluate retrieved chunks for contradictions, scope,
effective dates, versions, and explicit supersession statements.
Flag unresolved conflicts rather than inventing a resolution.
This is a distinct agent from the Research Agent — Research retrieves,
Evidence evaluates trustworthiness and consistency.
Traced via LangSmith.
"""
from typing import List, Dict, Tuple
from langsmith import traceable


# Known supersession pairs extracted from the KB.
# Format: (older_doc_id, newer_doc_id, topic, explanation)
KNOWN_SUPERSESSIONS = [
    ("security", "retention", "cold storage period",
     "retention:0 (v2, effective 2026-01-15) explicitly supersedes the earlier cold-storage period stated in security:1 (v1, published 2025-03-15)"),
    ("oncall", "handbook", "on-call stipend",
     "handbook:1 (v3, effective 2026-01-01) explicitly states $175/wk replaces the older stipend in oncall:0 (v1, published 2025-04-10)"),
]

# Words that indicate a chunk explicitly overrides older information.
# "effective" omitted because it over-matches (appears in benign contexts).
SUPERSESSION_KEYWORDS = [
    "supersedes", "superseding", "replacing", "replaces",
    "this policy supersedes", "no longer",
]


def _parse_date(published) -> int:
    """Convert published field (e.g. '20260310' or 20260310) to int."""
    try:
        return int(str(published).replace("-", "")[:8])
    except (ValueError, TypeError):
        return 0


def _parse_version(version) -> float:
    """Parse a version string to a comparable float. '2.3' → 2.3, '3' → 3.0."""
    try:
        return float(str(version))
    except (ValueError, TypeError):
        return 0.0


class EvidenceAgent:
    """
    Distinct agent responsible for evaluating evidence quality and detecting
    conflicts among retrieved corpus chunks.

    Applies the conflict-resolution protocol from faq:2:
      - Compare specific topic, effective date, scope, and explicit supersession
      - A later general overview does NOT automatically override a specific policy
      - Flag unresolved conflicts rather than silently picking a winner

    Does NOT perform retrieval (that is the Research Agent's job).
    Does NOT touch user memory (that is the Memory Agent's job).
    """

    @traceable(name="EvidenceAgent.evaluate")
    def evaluate(self, chunks: List[Dict], query: str = "") -> Dict:
        """
        Evaluate a set of retrieved chunks for conflicts and authority.

        Returns:
            {
                "conflict_notes": [...],      # human-readable conflict descriptions
                "ranked_chunks": [...],        # chunks re-ordered by authority
                "has_conflicts": bool,
                "supersession_applied": bool,
            }
        """
        conflict_notes = []
        supersession_applied = False

        # Group chunks by doc_id
        by_doc: Dict[str, List[Dict]] = {}
        for c in chunks:
            doc_id = c.get("doc_id", "")
            by_doc.setdefault(doc_id, []).append(c)

        doc_ids = list(by_doc.keys())

        # ── Check known supersession pairs ────────────────────────
        for old_doc, new_doc, topic, explanation in KNOWN_SUPERSESSIONS:
            if old_doc in doc_ids and new_doc in doc_ids:
                conflict_notes.append(
                    f"⚠️ Conflict on '{topic}': {explanation}. "
                    f"The newer, specific policy in '{new_doc}' should be trusted."
                )
                supersession_applied = True

        # ── Check introductory overview vs specific policy (faq:2 / company:2) ──
        if "company" in doc_ids and any(d in doc_ids for d in ["pricing", "policy", "product", "retention", "handbook"]):
            conflict_notes.append(
                "ℹ️ General overview chunk(s) detected. Per Asterline policy (company:2, faq:2), "
                "introductory company overviews do not establish product limits or override specific policies."
            )

        # ── Check for explicit supersession language in chunks ────
        for chunk in chunks:
            text_lower = chunk.get("content", "").lower()
            for kw in SUPERSESSION_KEYWORDS:
                if kw in text_lower:
                    conflict_notes.append(
                        f"ℹ️ [{chunk['chunk_id']}] ({chunk['title']}) "
                        f"contains explicit supersession language ('{kw}'). "
                        f"Treat as authoritative for its specific topic."
                    )
                    break  # one note per chunk is enough

        # ── Check for version/date conflicts between same-topic chunks ─
        conflict_notes.extend(self._check_version_conflicts(chunks, by_doc))

        # ── Re-rank by authority and query relevance ──────────────
        ranked_chunks = self._rank_by_authority(chunks, query=query)

        return {
            "conflict_notes": conflict_notes,
            "ranked_chunks": ranked_chunks,
            "has_conflicts": len(conflict_notes) > 0,
            "supersession_applied": supersession_applied,
        }

    @traceable(name="EvidenceAgent.check_version_conflicts")
    def _check_version_conflicts(self, chunks: List[Dict], by_doc: Dict) -> List[str]:
        """
        Detect cases where release notes describe different states of the
        same feature (e.g., AQL timeout 25s in v2.0 vs 50s in v2.3).
        """
        notes = []
        release_docs = [d for d in by_doc if d.startswith("release-")]
        if len(release_docs) > 1:
            # Sort by version
            release_docs_sorted = sorted(release_docs, key=lambda d: _parse_version(
                by_doc[d][0].get("version", 0)
            ))
            oldest = release_docs_sorted[0]
            newest = release_docs_sorted[-1]
            oldest_ver = by_doc[oldest][0].get("version", "?")
            newest_ver = by_doc[newest][0].get("version", "?")
            if oldest != newest:
                notes.append(
                    f"ℹ️ Multiple release notes retrieved (v{oldest_ver} → v{newest_ver}). "
                    f"For current-state questions, prefer the latest version's information."
                )
        return notes

    @traceable(name="EvidenceAgent.rank_by_authority")
    def _rank_by_authority(self, chunks: List[Dict], query: str = "") -> List[Dict]:
        """
        Re-rank chunks so that more authoritative and query-relevant sources appear first.
        Authority factors (in order):
          1. Chunks with explicit supersession language rank highest
          2. Query topic / keyword relevance (with stemming)
          3. Specific scope (policy > overview, product spec > release notes; penalty for company introductory overview)
          4. Higher version number
          5. More recent published date
        """
        CATEGORY_PRIORITY = {
            "policy": 5,
            "pricing": 4,
            "product": 3,
            "engineering": 3,
            "incident": 2,
            "release-notes": 1,
            "onboarding": 1,
        }

        # Clean query words and basic stems
        raw_words = [
            w.strip("?,.:;\"'()[]") for w in query.lower().split()
            if len(w) > 3 and w not in {"what", "when", "where", "which", "how", "long", "does", "that", "there", "with"}
        ]
        query_stems = set(raw_words + [w.rstrip("ed").rstrip("ing").rstrip("s") for w in raw_words if len(w) > 4])
        if "timeout" in query.lower():
            query_stems.add("time out")

        is_current_query = any(w in query.lower() for w in ["current", "latest", "now", "present"])

        def authority_key(chunk):
            text_lower = chunk.get("content", "").lower()
            title_lower = chunk.get("title", "").lower()

            has_supersession = any(kw in text_lower for kw in SUPERSESSION_KEYWORDS)

            # Query match score
            query_match_score = sum(
                3 if stem in title_lower else 1
                for stem in query_stems
                if stem in text_lower or stem in title_lower
            )

            # Scope penalty: company:0/1/2 introductory overview is not for specific technical/policy limits (faq:2, company:2)
            is_intro_overview = chunk.get("doc_id") == "company"
            scope_penalty = -5 if is_intro_overview else 0

            version = _parse_version(chunk.get("version", 0))
            version_score = version * (5.0 if is_current_query else 1.0)
            date = _parse_date(chunk.get("published", 0))
            category_score = CATEGORY_PRIORITY.get(chunk.get("category", ""), 0)

            return (
                has_supersession,
                version_score if is_current_query else query_match_score + scope_penalty,
                query_match_score + scope_penalty if is_current_query else version_score,
                category_score,
                date,
            )

        return sorted(chunks, key=authority_key, reverse=True)
