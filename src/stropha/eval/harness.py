"""Golden-dataset evaluation runner.

Schema (JSONL, one case per line):

    {
      "id": "stropha-001",
      "query": "where is the FSRS calculator",
      "expected_paths": ["src/stropha/fsrs.py"],
      "expected_symbols": ["FsrsCalculator"],
      "tags": ["symbol-lookup"]
    }

A hit counts as relevant when any of:
  - ``hit.path`` matches one of ``expected_paths`` (exact or suffix)
  - ``hit.symbol`` matches one of ``expected_symbols`` (case-insensitive)

Recall@K = (cases with ≥1 relevant hit in top-K) / total
MRR     = mean(1 / rank of first relevant hit, 0 when none in top-K)

The runner does NOT depend on any LLM and never re-embeds. It uses
whatever ``SearchEngine`` the caller built (so swapping an adapter and
re-running is a one-line change).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging import get_logger
from ..retrieval.search import SearchEngine

log = get_logger(__name__)


@dataclass(frozen=True)
class EvalCase:
    """One test case loaded from the golden file."""

    id: str
    query: str
    expected_paths: tuple[str, ...] = ()
    expected_symbols: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        return cls(
            id=str(d.get("id") or d.get("query", "")[:40]),
            query=str(d["query"]),
            expected_paths=tuple(d.get("expected_paths") or ()),
            expected_symbols=tuple(d.get("expected_symbols") or ()),
            tags=tuple(d.get("tags") or ()),
        )


@dataclass(frozen=True)
class EvalResult:
    """Outcome of running one :class:`EvalCase`."""

    case: EvalCase
    rank_of_first_hit: int | None  # 1-based, None if not found in top-K
    top_k: int
    matched_path: str | None = None
    matched_symbol: str | None = None

    @property
    def hit(self) -> bool:
        return self.rank_of_first_hit is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank_of_first_hit if self.rank_of_first_hit else 0.0


@dataclass
class EvalReport:
    """Aggregate metrics across an evaluation run."""

    results: list[EvalResult] = field(default_factory=list)
    top_k: int = 10

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def recall_at_k(self) -> float:
        return self.n_hits / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.n

    def by_tag(self) -> dict[str, EvalReport]:
        """Slice results per tag for diagnostic breakdowns."""
        buckets: dict[str, EvalReport] = {}
        for r in self.results:
            for t in r.case.tags or ("(untagged)",):
                buckets.setdefault(t, EvalReport(top_k=self.top_k)).results.append(r)
        return buckets

    def summary(self) -> dict[str, Any]:
        return {
            "cases": self.n,
            "hits": self.n_hits,
            f"recall@{self.top_k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
        }


# --------------------------------------------------------------------- io


def load_golden(path: Path) -> list[EvalCase]:
    """Parse a JSONL file into :class:`EvalCase` instances.

    Empty / comment lines (``#``-prefixed) are skipped. Invalid lines raise
    a ``ValueError`` indicating the line number — fast feedback when
    hand-editing the golden set.
    """
    cases: list[EvalCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict) or "query" not in payload:
            raise ValueError(f"{path}:{lineno}: missing required `query` key")
        cases.append(EvalCase.from_dict(payload))
    return cases


# --------------------------------------------------------------------- runner


def _path_matches(actual: str, expected: Iterable[str]) -> str | None:
    """``actual`` matches when it equals or ends with one of ``expected``."""
    a = actual.strip()
    for e in expected:
        if not e:
            continue
        e = e.strip()
        if a == e or a.endswith("/" + e) or a.endswith(e):
            return e
    return None


def _symbol_matches(actual: str | None, expected: Iterable[str]) -> str | None:
    if not actual:
        return None
    a = actual.lower().strip()
    for e in expected:
        if not e:
            continue
        if a == e.lower().strip() or a.endswith("." + e.lower().strip()):
            return e
    return None


def run_eval(
    cases: Iterable[EvalCase],
    engine: SearchEngine,
    *,
    top_k: int = 10,
) -> EvalReport:
    """Execute every case against ``engine`` and aggregate metrics."""
    report = EvalReport(top_k=top_k)
    for case in cases:
        hits = engine.search(case.query, top_k=top_k)
        first: EvalResult | None = None
        for rank, h in enumerate(hits, 1):
            # Tolerate both real SearchHit (.rel_path) and tests' fake hits (.path).
            actual_path = getattr(h, "rel_path", None) or getattr(h, "path", "")
            mp = _path_matches(actual_path, case.expected_paths)
            ms = _symbol_matches(getattr(h, "symbol", None), case.expected_symbols)
            if mp or ms:
                first = EvalResult(
                    case=case,
                    rank_of_first_hit=rank,
                    top_k=top_k,
                    matched_path=mp,
                    matched_symbol=ms,
                )
                break
        if first is None:
            first = EvalResult(case=case, rank_of_first_hit=None, top_k=top_k)
        report.results.append(first)
        log.info(
            "eval.case.done",
            case=case.id,
            hit=first.hit,
            rank=first.rank_of_first_hit,
            tags=list(case.tags),
        )
    log.info("eval.run.summary", **report.summary())
    return report
