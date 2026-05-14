"""Smoke tests for the Phase 0 walker (filesystem fallback path)."""

from __future__ import annotations

from pathlib import Path

from stropha.ingest.walker import Walker, detect_language


def test_detect_language_known() -> None:
    assert detect_language(Path("Foo.java")) == "java"
    assert detect_language(Path("App.vue")) == "vue"
    assert detect_language(Path("notes.md")) == "markdown"
    assert detect_language(Path("Dockerfile")) == "dockerfile"


def test_detect_language_unknown() -> None:
    assert detect_language(Path("blob.xyz")) is None


def test_walker_skips_binaries_and_oversize(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("print('hi')\n")
    (tmp_path / "big.md").write_text("a" * 1024)
    (tmp_path / "bin.py").write_bytes(b"\x00\x01\x02print")
    (tmp_path / "unknown.xyz").write_text("ignored")

    walker = Walker(tmp_path, max_file_bytes=512)
    rels = sorted(sf.rel_path for sf in walker.discover())
    assert rels == ["good.py"]
