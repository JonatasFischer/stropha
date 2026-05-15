"""Unit tests for ``stropha.eval.harness`` — golden loader, Recall@K, MRR.

We mock the SearchEngine to avoid touching real embedders/storage. The
harness must work with anything that exposes ``search(query, top_k) ->
list[SearchHit]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from stropha.eval import EvalCase, load_golden, run_eval

# --------------------------------------------------------------------------- mock


@dataclass
class _FakeHit:
    path: str
    symbol: str | None
    score: float = 1.0


class _FakeEngine:
    """Returns canned hits per query string. Works as a SearchEngine stub."""

    def __init__(self, by_query: dict[str, list[_FakeHit]]) -> None:
        self._by_query = by_query

    def search(self, query: str, *, top_k: int = 10) -> list[_FakeHit]:
        return self._by_query.get(query, [])[:top_k]


# --------------------------------------------------------------------------- load_golden


def test_load_golden_skips_blank_and_comments(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text(
        "\n"
        "# header comment\n"
        '{"id":"a","query":"foo","expected_paths":["x.py"]}\n'
        "\n"
        '{"id":"b","query":"bar","expected_symbols":["Bar"]}\n',
        encoding="utf-8",
    )
    cases = load_golden(f)
    assert len(cases) == 2
    assert cases[0].id == "a"
    assert cases[0].expected_paths == ("x.py",)
    assert cases[1].expected_symbols == ("Bar",)


def test_load_golden_invalid_json_raises_with_lineno(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text(
        '{"id":"good","query":"x"}\n'
        '{ this is broken }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2:"):
        load_golden(f)


def test_load_golden_missing_query_raises(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text('{"id":"x","expected_paths":["a"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required `query`"):
        load_golden(f)


# --------------------------------------------------------------------------- run_eval


def test_recall_when_path_matches_at_rank_1() -> None:
    cases = [EvalCase(id="c1", query="foo", expected_paths=("a.py",))]
    engine = _FakeEngine({"foo": [_FakeHit("a.py", None)]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.n == 1
    assert report.n_hits == 1
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0


def test_recall_zero_when_no_hit() -> None:
    cases = [EvalCase(id="c1", query="foo", expected_paths=("a.py",))]
    engine = _FakeEngine({"foo": [_FakeHit("z.py", None)]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_path_match_is_suffix_aware() -> None:
    """``expected_paths=['fsrs.py']`` matches ``src/stropha/fsrs.py``."""
    cases = [EvalCase(id="c1", query="q", expected_paths=("fsrs.py",))]
    engine = _FakeEngine({"q": [_FakeHit("src/stropha/fsrs.py", None)]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.n_hits == 1
    assert report.results[0].matched_path == "fsrs.py"


def test_symbol_match_dotted_suffix() -> None:
    """``expected_symbols=['login']`` matches ``AuthService.login``."""
    cases = [EvalCase(id="c1", query="q", expected_symbols=("login",))]
    engine = _FakeEngine({"q": [_FakeHit("a.py", "AuthService.login")]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.n_hits == 1
    assert report.results[0].matched_symbol == "login"


def test_mrr_is_first_relevant_rank() -> None:
    cases = [EvalCase(id="c1", query="q", expected_paths=("target.py",))]
    engine = _FakeEngine({"q": [
        _FakeHit("a.py", None),
        _FakeHit("b.py", None),
        _FakeHit("target.py", None),  # rank 3
    ]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.results[0].rank_of_first_hit == 3
    assert report.mrr == pytest.approx(1.0 / 3)


def test_top_k_cuts_off() -> None:
    cases = [EvalCase(id="c1", query="q", expected_paths=("target.py",))]
    engine = _FakeEngine({"q": [
        _FakeHit("a.py", None), _FakeHit("b.py", None), _FakeHit("target.py", None),
    ]})
    report = run_eval(cases, engine, top_k=2)  # type: ignore[arg-type]
    assert report.recall_at_k == 0.0


def test_aggregate_recall_across_cases() -> None:
    cases = [
        EvalCase(id="c1", query="q1", expected_paths=("a.py",)),
        EvalCase(id="c2", query="q2", expected_paths=("b.py",)),
        EvalCase(id="c3", query="q3", expected_paths=("c.py",)),
    ]
    engine = _FakeEngine({
        "q1": [_FakeHit("a.py", None)],   # hit
        "q2": [_FakeHit("z.py", None)],   # miss
        "q3": [_FakeHit("c.py", None)],   # hit
    })
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.recall_at_k == pytest.approx(2 / 3)


def test_by_tag_buckets() -> None:
    cases = [
        EvalCase(id="c1", query="q1", expected_paths=("a.py",), tags=("retrieval",)),
        EvalCase(id="c2", query="q2", expected_paths=("b.py",), tags=("retrieval",)),
        EvalCase(id="c3", query="q3", expected_paths=("c.py",), tags=("storage",)),
    ]
    engine = _FakeEngine({
        "q1": [_FakeHit("a.py", None)],
        "q2": [_FakeHit("z.py", None)],
        "q3": [_FakeHit("c.py", None)],
    })
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    by_tag = report.by_tag()
    assert by_tag["retrieval"].recall_at_k == 0.5
    assert by_tag["storage"].recall_at_k == 1.0


# --------------------------------------------------------------------------- end-to-end with the shipped golden file


def test_shipped_golden_file_parses() -> None:
    """The default golden set must always parse cleanly."""
    p = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"
    if not p.is_file():
        pytest.skip("golden file not present")
    cases = load_golden(p)
    assert len(cases) >= 20  # we ship ~30
    # Every case has at least one expected_paths or expected_symbols entry
    for c in cases:
        assert c.expected_paths or c.expected_symbols, f"case {c.id} has no expectations"
