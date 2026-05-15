"""Tests for Phase 3 chunker dispatcher and language sub-adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from stropha import adapters  # noqa: F401
from stropha.adapters.chunker.languages.file_level import FileLevelLanguageChunker
from stropha.adapters.chunker.languages.heading_split import (
    HeadingSplitLanguageChunker,
)
from stropha.adapters.chunker.tree_sitter_dispatch import (
    TreeSitterDispatchChunker,
    TreeSitterDispatchConfig,
)
from stropha.errors import ChunkerError
from stropha.ingest.chunker import Chunker as LegacyChunker
from stropha.models import SourceFile
from stropha.pipeline.registry import all_adapters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def py_file(tmp_path: Path) -> SourceFile:
    """Long enough (>15 lines) to bypass AST chunker's SIMPLE_FILE shortcut."""
    p = tmp_path / "main.py"
    body = "\n".join(
        [
            '"""A small module to chunk."""',
            "",
            "class Greeter:",
            "    def __init__(self, name: str) -> None:",
            "        self.name = name",
            "",
            "    def greet(self) -> str:",
            "        return f'hello, {self.name}'",
            "",
            "    def shout(self) -> str:",
            "        return self.greet().upper()",
            "",
            "",
            "def make_greeter(name: str) -> Greeter:",
            "    return Greeter(name)",
            "",
            "",
            "def main() -> None:",
            "    print(make_greeter('world').greet())",
            "",
        ]
    )
    p.write_text(body, encoding="utf-8")
    return SourceFile(path=p, rel_path="main.py", language="python", size_bytes=p.stat().st_size)


@pytest.fixture
def md_file(tmp_path: Path) -> SourceFile:
    p = tmp_path / "README.md"
    p.write_text(
        "# Title\n\nIntro paragraph.\n\n## Section A\n\nA stuff.\n\n## Section B\n\nB stuff.\n",
        encoding="utf-8",
    )
    return SourceFile(path=p, rel_path="README.md", language="markdown", size_bytes=p.stat().st_size)


@pytest.fixture
def txt_file(tmp_path: Path) -> SourceFile:
    p = tmp_path / "notes.txt"
    p.write_text("plain text\nanother line\n", encoding="utf-8")
    return SourceFile(path=p, rel_path="notes.txt", language="text", size_bytes=p.stat().st_size)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_chunker_and_language_chunker_stages_registered() -> None:
    reg = all_adapters()
    assert "tree-sitter-dispatch" in reg["chunker"]
    for name in ("ast-generic", "heading-split", "sfc-split", "regex-feature-scenario", "file-level"):
        assert name in reg["language-chunker"], f"missing {name}"


# ---------------------------------------------------------------------------
# Sub-adapter contracts
# ---------------------------------------------------------------------------


def test_heading_split_sub_adapter(md_file: SourceFile) -> None:
    sub = HeadingSplitLanguageChunker()
    chunks = list(sub.chunk(md_file, md_file.path.read_text()))
    # Two H2 sections + intro/title material → at least 2 chunks.
    assert len(chunks) >= 2
    assert sub.adapter_name == "heading-split"
    assert sub.health().status == "ready"


def test_file_level_sub_adapter(txt_file: SourceFile) -> None:
    sub = FileLevelLanguageChunker()
    chunks = list(sub.chunk(txt_file, txt_file.path.read_text()))
    assert len(chunks) == 1
    assert chunks[0].kind == "file"


# ---------------------------------------------------------------------------
# Dispatcher behavior
# ---------------------------------------------------------------------------


def test_dispatcher_routes_to_python_ast(py_file: SourceFile) -> None:
    chunker = TreeSitterDispatchChunker()
    chunks = list(chunker.chunk([py_file]))
    # File is >15 lines so AST decomposition kicks in; expect class + methods.
    kinds = {c.kind for c in chunks}
    assert "class" in kinds or "method" in kinds or "function" in kinds, (
        f"expected AST kinds, got {kinds}"
    )
    symbols = {c.symbol for c in chunks if c.symbol}
    # At least one of the known symbols must surface.
    assert symbols & {"Greeter", "Greeter.greet", "main", "make_greeter"}


def test_dispatcher_falls_back_to_file_level_for_unknown_language(
    txt_file: SourceFile,
) -> None:
    chunker = TreeSitterDispatchChunker()
    chunks = list(chunker.chunk([txt_file]))
    assert len(chunks) >= 1
    # Plain text -> file-level fallback only.
    assert all(c.kind == "file" for c in chunks)


def test_dispatcher_adapter_id_changes_with_language_override() -> None:
    default = TreeSitterDispatchChunker().adapter_id
    override = TreeSitterDispatchChunker(
        TreeSitterDispatchConfig(
            languages={"markdown": {"adapter": "file-level"}}
        )
    ).adapter_id
    assert default != override


def test_dispatcher_unknown_sub_adapter_raises() -> None:
    with pytest.raises(Exception):
        TreeSitterDispatchChunker(
            TreeSitterDispatchConfig(
                languages={"python": {"adapter": "no-such-sub"}}
            )
        )


def test_dispatcher_health_reports_sub_adapter_count() -> None:
    chunker = TreeSitterDispatchChunker()
    h = chunker.health()
    assert h.status == "ready"
    # 10 default languages + _fallback = 11 entries.
    assert "11" in h.message


# ---------------------------------------------------------------------------
# Output parity vs legacy Chunker
# ---------------------------------------------------------------------------


def test_dispatcher_output_matches_legacy_chunker_for_python(
    py_file: SourceFile,
) -> None:
    """Phase 3 must NOT change chunk_id of unchanged files (no spurious re-embed)."""
    legacy = list(LegacyChunker().chunk([py_file], repo_key="test-repo"))
    new = list(TreeSitterDispatchChunker().chunk([py_file], repo_key="test-repo"))
    assert len(legacy) == len(new)
    legacy_ids = sorted(c.chunk_id for c in legacy)
    new_ids = sorted(c.chunk_id for c in new)
    assert legacy_ids == new_ids


def test_dispatcher_output_matches_legacy_chunker_for_markdown(
    md_file: SourceFile,
) -> None:
    legacy = list(LegacyChunker().chunk([md_file], repo_key="test-repo"))
    new = list(TreeSitterDispatchChunker().chunk([md_file], repo_key="test-repo"))
    legacy_ids = sorted(c.chunk_id for c in legacy)
    new_ids = sorted(c.chunk_id for c in new)
    assert legacy_ids == new_ids


def test_dispatcher_unreadable_file_raises_chunker_error(tmp_path: Path) -> None:
    sf = SourceFile(
        path=tmp_path / "does-not-exist.py",
        rel_path="does-not-exist.py",
        language="python",
        size_bytes=0,
    )
    chunker = TreeSitterDispatchChunker()
    with pytest.raises(ChunkerError):
        list(chunker.chunk([sf]))
