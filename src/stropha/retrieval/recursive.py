"""Recursive retrieval / auto-merging (Phase 3 §6.4).

Post-processing pass that takes the top-K hits from the hybrid retriever
and expands them in two ways:

1. **Parent merging** — when a chunk's parent (via ``parent_chunk_id``)
   would have been merged into the same answer anyway, return the
   parent's snippet instead. Reduces fragmentation when several methods
   of the same class all match.

2. **Adjacent merging** — when two retrieved chunks come from the same
   file and are within ``adjacency_lines`` of each other, merge their
   snippets into a single hit covering the union of their line ranges.
   Recovers context that exact-chunk-boundary retrieval misses.

Local-only, no LLM, no embeddings. Pure SQL + dataclass juggling. Toggle
via ``STROPHA_RECURSIVE_RETRIEVAL=1`` (default off — kept conservative
because merging changes the shape of `SearchHit` lists, which downstream
consumers may not expect).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from ..logging import get_logger
from ..models import SearchHit
from ..storage import Storage

log = get_logger(__name__)

DEFAULT_ADJACENCY_LINES = 5


def _enabled() -> bool:
    return os.environ.get("STROPHA_RECURSIVE_RETRIEVAL", "0") == "1"


def _adjacency_lines() -> int:
    try:
        return int(os.environ.get("STROPHA_RECURSIVE_ADJACENCY", str(DEFAULT_ADJACENCY_LINES)))
    except ValueError:
        return DEFAULT_ADJACENCY_LINES


def merge_hits(
    hits: Sequence[SearchHit],
    storage: Storage,
    *,
    adjacency_lines: int | None = None,
    enabled: bool | None = None,
) -> list[SearchHit]:
    """Return ``hits`` after auto-merge passes.

    Pass-through when disabled or input is empty.

    Order preservation: the first occurrence of any merged group keeps
    its rank. Subsequent merged-in hits are dropped from the result.
    """
    if enabled is None:
        enabled = _enabled()
    if not enabled or not hits:
        return list(hits)
    adjacency_lines = adjacency_lines if adjacency_lines is not None else _adjacency_lines()

    out: list[SearchHit] = []
    used_chunk_ids: set[str] = set()
    # Bucket by rel_path for the adjacency pass.
    by_path: dict[str, list[SearchHit]] = {}
    for h in hits:
        if h.chunk_id in used_chunk_ids:
            continue
        merged = _maybe_merge_parent(h, hits, storage, used_chunk_ids)
        used_chunk_ids.add(merged.chunk_id)
        by_path.setdefault(merged.rel_path, []).append(merged)
        out.append(merged)

    # Adjacency merge — walk each file's hits in start_line order, merge
    # any pair whose gap < adjacency_lines.
    final: list[SearchHit] = []
    for h in out:
        siblings = by_path.get(h.rel_path, [])
        if len(siblings) <= 1:
            final.append(h)
            continue
        # Already-merged check: is there an earlier hit on this path that
        # already covers this one's range?
        absorbed = False
        for i, prev in enumerate(final):
            if (
                prev.rel_path == h.rel_path
                and h.chunk_id != prev.chunk_id
                and _adjacent_or_overlap(prev, h, adjacency_lines)
            ):
                final[i] = _absorb(prev, h, storage)
                absorbed = True
                break
        if not absorbed:
            final.append(h)
    if len(final) != len(hits):
        log.info(
            "recursive.merged",
            input_n=len(hits), output_n=len(final),
            adjacency_lines=adjacency_lines,
        )
    return final


# --------------------------------------------------------------------- helpers


def _maybe_merge_parent(
    hit: SearchHit,
    all_hits: Sequence[SearchHit],
    storage: Storage,
    used_chunk_ids: set[str],
) -> SearchHit:
    """If any sibling in ``all_hits`` shares ``hit``'s parent_chunk_id, promote
    the result to the parent chunk (loaded from storage)."""
    cur = storage._conn.cursor()  # noqa: SLF001
    row = cur.execute(
        "SELECT parent_chunk_id FROM chunks WHERE chunk_id = ?", (hit.chunk_id,),
    ).fetchone()
    if row is None or not row["parent_chunk_id"]:
        return hit
    parent_id = row["parent_chunk_id"]
    # Look for any other hit in the result set sharing this parent.
    has_sibling = any(
        h.chunk_id != hit.chunk_id
        and cur.execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ? AND parent_chunk_id = ? LIMIT 1",
            (h.chunk_id, parent_id),
        ).fetchone() is not None
        for h in all_hits
    )
    if not has_sibling:
        return hit
    parent = cur.execute(
        """SELECT c.*, r.normalized_key, r.remote_url, r.default_branch,
                  r.head_commit
           FROM chunks c LEFT JOIN repos r ON c.repo_id = r.id
           WHERE c.chunk_id = ? LIMIT 1""",
        (parent_id,),
    ).fetchone()
    if parent is None:
        return hit
    return _row_to_hit(parent, score=hit.score, rank=hit.rank)


def _adjacent_or_overlap(a: SearchHit, b: SearchHit, gap: int) -> bool:
    if a.rel_path != b.rel_path:
        return False
    lo_a, hi_a = a.start_line, a.end_line
    lo_b, hi_b = b.start_line, b.end_line
    if hi_a < lo_b:
        return (lo_b - hi_a) <= gap
    if hi_b < lo_a:
        return (lo_a - hi_b) <= gap
    return True  # overlap


def _absorb(host: SearchHit, other: SearchHit, storage: Storage) -> SearchHit:
    """Return a merged hit covering host ∪ other."""
    new_start = min(host.start_line, other.start_line)
    new_end = max(host.end_line, other.end_line)
    # Re-fetch the content spanning the merged range. For simplicity we
    # concatenate the two snippets; a fancier path would read the file
    # from disk, but that introduces I/O on the query hot path.
    new_snippet = host.snippet
    if other.snippet and other.snippet not in host.snippet:
        new_snippet = f"{host.snippet}\n…\n{other.snippet}"
    return host.model_copy(update={
        "start_line": new_start,
        "end_line": new_end,
        "snippet": new_snippet[:1200],
        # Keep the higher of the two scores so RRF semantics aren't lost.
        "score": max(host.score, other.score),
    })


def _row_to_hit(row, *, score: float, rank: int) -> SearchHit:
    from ..models import RepoRef

    repo: RepoRef | None = None
    if row["normalized_key"]:
        repo = RepoRef(
            normalized_key=row["normalized_key"],
            url=row["remote_url"],
            default_branch=row["default_branch"],
            head_commit=row["head_commit"],
        )
    snippet = (row["content"] or "")[:480]
    return SearchHit(
        rank=rank,
        score=score,
        rel_path=row["rel_path"],
        language=row["language"],
        kind=row["kind"],
        symbol=row["symbol"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        snippet=snippet,
        chunk_id=row["chunk_id"],
        repo=repo,
    )
