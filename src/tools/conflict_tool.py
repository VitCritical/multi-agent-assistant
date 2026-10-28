"""
tools/conflict_tool.py — detect conflicting information among retrieved chunks.
Applies the KB conflict-resolution protocol from faq:2:
  1. Specific > General
  2. Later effective date > Earlier
  3. Explicit supersession statement wins outright
"""
from typing import List, Dict, Tuple


# Known supersession pairs extracted from the KB
# (older_doc_id, newer_doc_id, topic)
KNOWN_SUPERSESSIONS = [
    ("security", "retention", "cold storage period"),
    ("oncall",   "handbook",  "on-call stipend"),
    ("release-20", "release-21", "AQL timeout / ingestion rps"),
    ("release-20", "release-22", "AQL timeout / features"),
    ("release-20", "release-23", "AQL timeout / PagerDuty"),
    ("release-21", "release-23", "AQL timeout"),
]

SUPERSESSION_KEYWORDS = [
    "supersedes", "superseding", "replaced", "replaces",
    "effective", "updated", "overrides", "no longer",
]


def _parse_date(published: str) -> int:
    """Convert published string like '20260310' to int for comparison."""
    try:
        return int(str(published).replace("-", "")[:8])
    except (ValueError, TypeError):
        return 0


def detect_conflicts(chunks: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Given a list of retrieved chunks, detect potential conflicts.

    Returns:
        conflict_notes: List of human-readable conflict descriptions
        ranked_chunks:  Chunks re-ranked with authoritative ones first
    """
    conflict_notes = []
    doc_ids_seen = {}

    # Group chunks by doc_id
    by_doc: Dict[str, List[Dict]] = {}
    for c in chunks:
        doc_id = c.get("doc_id", "")
        by_doc.setdefault(doc_id, []).append(c)

    doc_ids = list(by_doc.keys())

    # Check known supersession pairs
    for old_doc, new_doc, topic in KNOWN_SUPERSESSIONS:
        if old_doc in doc_ids and new_doc in doc_ids:
            old_chunks = by_doc[old_doc]
            new_chunks = by_doc[new_doc]
            # find dates
            old_date = max(_parse_date(c.get("published", 0)) for c in old_chunks)
            new_date = max(_parse_date(c.get("published", 0)) for c in new_chunks)
            if new_date > old_date:
                conflict_notes.append(
                    f"⚠️ Conflict on '{topic}': "
                    f"'{new_doc}' (published {new_date}) supersedes '{old_doc}' (published {old_date}). "
                    f"Trust '{new_doc}'."
                )

    # Check for supersession keywords in chunk content
    for chunk in chunks:
        content_lower = chunk.get("content", "").lower()
        if any(kw in content_lower for kw in SUPERSESSION_KEYWORDS):
            if chunk["chunk_id"] not in doc_ids_seen:
                doc_ids_seen[chunk["chunk_id"]] = True
                conflict_notes.append(
                    f"ℹ️ Chunk '{chunk['chunk_id']}' ({chunk['title']}) "
                    f"contains supersession language — treat as authoritative for its topic."
                )

    # Re-rank: push newer docs to the front
    ranked = sorted(chunks, key=lambda c: _parse_date(c.get("published", 0)), reverse=True)
    return conflict_notes, ranked
