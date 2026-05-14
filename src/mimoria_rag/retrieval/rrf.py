"""Reciprocal Rank Fusion (Cormack et al., 2009 — spec §6.1).

Formula: score(d) = Σ_i 1 / (k + rank_i(d))   where k=60 by default.

Why RRF beats weighted-sum here:
- BM25 and cosine live in different scales; normalizing across queries is
  brittle. RRF only consumes ranks.
- Robust to outliers in either list.
- No hyperparameters worth tuning (k=60 is the published default).
"""

from __future__ import annotations

from collections.abc import Iterable

from ..models import SearchHit

DEFAULT_K = 60


def rrf_fuse(
    *ranked_lists: Iterable[SearchHit],
    k: int = DEFAULT_K,
    top_k: int = 10,
) -> list[SearchHit]:
    """Merge any number of ranked lists into a single ranked list.

    Identity is by `chunk_id`. The returned `SearchHit` keeps the metadata of
    the first occurrence of each chunk (typically the dense hit, since dense
    is usually passed first), with its `score` replaced by the RRF score.
    """
    scores: dict[str, float] = {}
    seen: dict[str, SearchHit] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            contribution = 1.0 / (k + rank)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + contribution
            if hit.chunk_id not in seen:
                seen[hit.chunk_id] = hit

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[SearchHit] = []
    for new_rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1):
        base = seen[chunk_id]
        out.append(base.model_copy(update={"rank": new_rank, "score": score}))
    return out
