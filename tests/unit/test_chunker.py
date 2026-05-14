"""Tests for the chunker dispatcher and per-language chunkers."""

from __future__ import annotations

from pathlib import Path

from stropha.ingest.chunker import Chunker
from stropha.ingest.chunkers.base import MAX_CHARS_PER_CHUNK
from stropha.ingest.chunkers.fallback import FallbackChunker
from stropha.models import SourceFile


def _make_file(tmp_path: Path, name: str, content: str, language: str = "python") -> SourceFile:
    p = tmp_path / name
    p.write_text(content)
    return SourceFile(
        path=p,
        rel_path=name,
        language=language,
        size_bytes=len(content.encode()),
    )


# ---- fallback (file-level) ----

def test_fallback_small_file_yields_single_chunk(tmp_path: Path) -> None:
    sf = _make_file(tmp_path, "x.txt", "tiny content here lorem ipsum\n", language="text")
    chunks = list(FallbackChunker().chunk(sf, sf.path.read_text()))
    assert len(chunks) == 1
    assert chunks[0].kind == "file"
    assert chunks[0].start_line == 1


def test_fallback_oversized_splits(tmp_path: Path) -> None:
    line = "x = 1\n"
    content = line * (MAX_CHARS_PER_CHUNK // len(line) + 100)
    sf = _make_file(tmp_path, "big.txt", content, language="text")
    chunks = list(FallbackChunker().chunk(sf, content))
    assert len(chunks) >= 2
    assert all(c.kind == "file_part" for c in chunks)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.start_line == prev.end_line + 1


def test_fallback_empty_is_skipped(tmp_path: Path) -> None:
    sf = _make_file(tmp_path, "empty.txt", "   \n\n", language="text")
    assert list(FallbackChunker().chunk(sf, "   \n\n")) == []


# ---- dispatcher (Chunker) ----

def test_chunker_python_function(tmp_path: Path) -> None:
    # Needs to exceed SIMPLE_FILE_LINES threshold to trigger AST split.
    src = (
        '"""Module docstring."""\n'
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def alpha(x: int) -> int:\n"
        "    \"\"\"Add one.\"\"\"\n"
        "    return x + 1\n"
        "\n"
        "\n"
        "def beta(x: int) -> int:\n"
        "    \"\"\"Double.\"\"\"\n"
        "    return x * 2\n"
        "\n"
        "\n"
        "class Foo:\n"
        "    def bar(self) -> int:\n"
        "        \"\"\"Return 42.\"\"\"\n"
        "        return 42\n"
        "\n"
        "    def baz(self) -> str:\n"
        "        \"\"\"Return greeting.\"\"\"\n"
        "        return 'hello'\n"
    )
    sf = _make_file(tmp_path, "x.py", src, language="python")
    chunks = list(Chunker().chunk([sf]))
    kinds = {c.kind for c in chunks}
    assert "function" in kinds or "method" in kinds
    symbols = {c.symbol for c in chunks if c.symbol}
    assert any(s in symbols for s in ("alpha", "beta", "Foo.bar", "Foo.baz"))


def test_chunker_unknown_language_falls_back(tmp_path: Path) -> None:
    sf = _make_file(
        tmp_path, "config.toml",
        "[section]\nkey = 'value'\nother = 42\nmore = true\n",
        language="toml",
    )
    chunks = list(Chunker().chunk([sf]))
    assert len(chunks) == 1
    assert chunks[0].kind == "file"


def test_chunker_markdown_splits_by_heading(tmp_path: Path) -> None:
    src = (
        "# Title\n\n"
        "Intro paragraph with enough words to pass the minimum chunk size threshold.\n\n"
        "## First section\n\n"
        "Body of the first section with sufficient content to be indexed.\n\n"
        "## Second section\n\n"
        "Body of the second section also with plenty of content.\n"
    )
    sf = _make_file(tmp_path, "notes.md", src, language="markdown")
    chunks = list(Chunker().chunk([sf]))
    assert len(chunks) >= 2
    symbols = [c.symbol for c in chunks]
    assert "First section" in symbols
    assert "Second section" in symbols
    # Parent link: H2 sections should reference the H1 title.
    h1 = next(c for c in chunks if c.symbol == "Title")
    children = [c for c in chunks if c.parent_chunk_id == h1.chunk_id]
    assert len(children) == 2


def test_chunker_gherkin_emits_scenarios(tmp_path: Path) -> None:
    src = (
        "Feature: Submitting an answer\n"
        "  Background:\n"
        "    Given a student with an active enrollment\n"
        "\n"
        "  Scenario: Correct answer increases streak\n"
        "    When the student submits a correct answer\n"
        "    Then the streak counter increases by 1\n"
        "\n"
        "  Scenario: Wrong answer resets streak\n"
        "    When the student submits a wrong answer\n"
        "    Then the streak counter resets to 0\n"
    )
    sf = _make_file(tmp_path, "submit.feature", src, language="gherkin")
    chunks = list(Chunker().chunk([sf]))
    kinds = [c.kind for c in chunks]
    assert "feature" in kinds
    assert kinds.count("scenario") == 2


def test_chunker_deterministic_ids(tmp_path: Path) -> None:
    src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    sf = _make_file(tmp_path, "x.py", src, language="python")
    first = list(Chunker().chunk([sf]))
    second = list(Chunker().chunk([sf]))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
