"""Tests for the domain glossary feature."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from stropha.ingest.glossary import (
    GlossaryLoader,
    GlossaryStats,
    GlossaryTerm,
    find_glossary_yaml,
)
from stropha.models import Chunk
from stropha.storage import Storage


# --------------------------------------------------------------------------- #
#                           GlossaryTerm Tests                                #
# --------------------------------------------------------------------------- #


def test_term_content_hash_is_deterministic() -> None:
    """Same term + definition produces same hash."""
    t1 = GlossaryTerm(term="FSRS", definition="Free Spaced Repetition Scheduler")
    t2 = GlossaryTerm(term="FSRS", definition="Free Spaced Repetition Scheduler")
    assert t1.content_hash == t2.content_hash


def test_term_content_hash_changes_with_definition() -> None:
    """Different definitions produce different hashes."""
    t1 = GlossaryTerm(term="FSRS", definition="Free Spaced Repetition Scheduler")
    t2 = GlossaryTerm(term="FSRS", definition="Something else")
    assert t1.content_hash != t2.content_hash


def test_term_to_chunk_has_glossary_kind() -> None:
    """Chunk has kind=glossary."""
    term = GlossaryTerm(term="Test", definition="A test term")
    chunk = term.to_chunk()
    assert chunk.kind == "glossary"
    assert chunk.symbol == "Test"
    assert chunk.rel_path == "<glossary>"


def test_term_to_chunk_includes_aliases_in_content() -> None:
    """Aliases are included in chunk content."""
    term = GlossaryTerm(
        term="FSRS",
        definition="Free Spaced Repetition Scheduler",
        aliases=["fsrs-algorithm", "SRS"],
    )
    chunk = term.to_chunk()
    assert "FSRS" in chunk.content
    assert "fsrs-algorithm" in chunk.content
    assert "SRS" in chunk.content
    assert "Free Spaced Repetition Scheduler" in chunk.content


def test_term_chunk_id_includes_hash() -> None:
    """Chunk ID is prefixed with glossary: and includes hash."""
    term = GlossaryTerm(term="Test", definition="A test term")
    chunk = term.to_chunk()
    assert chunk.chunk_id.startswith("glossary:")


# --------------------------------------------------------------------------- #
#                           GlossaryLoader Tests                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage for testing."""
    storage = MagicMock(spec=Storage)
    storage._conn = MagicMock()
    storage._conn.cursor.return_value = storage._conn
    return storage


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder that returns fixed-size vectors."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1024
    embedder.dim = 1024
    return embedder


def test_loader_load_yaml_parses_terms(tmp_path: Path) -> None:
    """YAML file with terms is parsed correctly."""
    yaml_file = tmp_path / "glossary.yaml"
    yaml_file.write_text("""
terms:
  - term: FSRS
    definition: Free Spaced Repetition Scheduler
    aliases:
      - SRS
  - term: Mastery
    definition: A measure of learning (0-1)
""")
    
    loader = GlossaryLoader(MagicMock())
    terms = loader.load_yaml(yaml_file)
    
    assert len(terms) == 2
    assert terms[0].term == "FSRS"
    assert terms[0].definition == "Free Spaced Repetition Scheduler"
    assert terms[0].aliases == ["SRS"]
    assert terms[1].term == "Mastery"


def test_loader_load_yaml_handles_missing_file(tmp_path: Path) -> None:
    """Missing YAML file returns empty list."""
    loader = GlossaryLoader(MagicMock())
    terms = loader.load_yaml(tmp_path / "nonexistent.yaml")
    assert terms == []


def test_loader_load_yaml_handles_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML returns empty list."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("not: valid: yaml: [[")
    
    loader = GlossaryLoader(MagicMock())
    terms = loader.load_yaml(yaml_file)
    assert terms == []


def test_loader_load_yaml_handles_empty_file(tmp_path: Path) -> None:
    """Empty YAML file returns empty list."""
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    
    loader = GlossaryLoader(MagicMock())
    terms = loader.load_yaml(yaml_file)
    assert terms == []


def test_loader_load_yaml_skips_entries_without_term(tmp_path: Path) -> None:
    """Entries without term field are skipped."""
    yaml_file = tmp_path / "glossary.yaml"
    yaml_file.write_text("""
terms:
  - definition: Missing term field
  - term: Valid
    definition: This one is valid
""")
    
    loader = GlossaryLoader(MagicMock())
    terms = loader.load_yaml(yaml_file)
    
    assert len(terms) == 1
    assert terms[0].term == "Valid"


def test_loader_index_terms_calls_upsert(
    mock_storage: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    """Index terms calls upsert_chunk for each term."""
    mock_storage.chunk_exists.return_value = False
    
    loader = GlossaryLoader(mock_storage)
    terms = [
        GlossaryTerm(term="A", definition="Definition A"),
        GlossaryTerm(term="B", definition="Definition B"),
    ]
    
    indexed = loader.index_terms(terms, mock_embedder)
    
    assert indexed == 2
    assert mock_storage.upsert_chunk.call_count == 2


def test_loader_index_terms_skips_unchanged(
    mock_storage: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    """Unchanged terms are skipped."""
    term = GlossaryTerm(term="A", definition="Definition A")
    
    # Simulate existing chunk with same hash
    mock_storage.chunk_exists.return_value = True
    mock_storage.get_chunk_by_id.return_value = {"content_hash": term.content_hash}
    
    loader = GlossaryLoader(mock_storage)
    indexed = loader.index_terms([term], mock_embedder)
    
    assert indexed == 0
    assert mock_storage.upsert_chunk.call_count == 0


def test_loader_index_terms_updates_changed(
    mock_storage: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    """Changed terms are re-indexed."""
    term = GlossaryTerm(term="A", definition="Definition A")
    
    # Simulate existing chunk with different hash
    mock_storage.chunk_exists.return_value = True
    mock_storage.get_chunk_by_id.return_value = {"content_hash": "different_hash"}
    
    loader = GlossaryLoader(mock_storage)
    indexed = loader.index_terms([term], mock_embedder)
    
    assert indexed == 1
    assert mock_storage.upsert_chunk.call_count == 1


def test_loader_remove_term_deletes_from_storage(mock_storage: MagicMock) -> None:
    """Remove term deletes matching chunks."""
    mock_storage._conn.execute.return_value.fetchall.return_value = [
        {"chunk_id": "glossary:abc123"}
    ]
    
    loader = GlossaryLoader(mock_storage)
    result = loader.remove_term("FSRS")
    
    assert result is True
    mock_storage.delete_chunk.assert_called_once_with("glossary:abc123")


def test_loader_remove_term_returns_false_when_not_found(
    mock_storage: MagicMock,
) -> None:
    """Remove returns False when term not found."""
    mock_storage._conn.execute.return_value.fetchall.return_value = []
    
    loader = GlossaryLoader(mock_storage)
    result = loader.remove_term("NonExistent")
    
    assert result is False
    mock_storage.delete_chunk.assert_not_called()


def test_loader_list_terms_returns_all_glossary_chunks(mock_storage: MagicMock) -> None:
    """List terms queries glossary chunks."""
    mock_storage._conn.execute.return_value.fetchall.return_value = [
        {"symbol": "FSRS", "content": "FSRS\nFree Spaced Repetition Scheduler"},
        {"symbol": "Mastery", "content": "Mastery\nLearning measure"},
    ]
    
    loader = GlossaryLoader(mock_storage)
    terms = loader.list_terms()
    
    assert len(terms) == 2
    assert terms[0]["term"] == "FSRS"
    assert terms[1]["term"] == "Mastery"


def test_loader_export_yaml_writes_file(mock_storage: MagicMock, tmp_path: Path) -> None:
    """Export writes terms to YAML file."""
    mock_storage._conn.execute.return_value.fetchall.return_value = [
        {"symbol": "Test", "content": "Test\nA test term"},
    ]
    
    loader = GlossaryLoader(mock_storage)
    output = tmp_path / "exported.yaml"
    count = loader.export_yaml(output)
    
    assert count == 1
    assert output.exists()
    
    data = yaml.safe_load(output.read_text())
    assert len(data["terms"]) == 1
    assert data["terms"][0]["term"] == "Test"


def test_loader_stats_returns_counts(mock_storage: MagicMock) -> None:
    """Stats returns term counts."""
    mock_storage._conn.execute.return_value.fetchone.return_value = (5,)
    mock_storage.get_meta.return_value = "2026-05-16T12:00:00"
    
    loader = GlossaryLoader(mock_storage)
    stats = loader.stats()
    
    assert stats.total_terms == 5
    assert stats.last_updated == "2026-05-16T12:00:00"


# --------------------------------------------------------------------------- #
#                           find_glossary_yaml Tests                          #
# --------------------------------------------------------------------------- #


def test_find_glossary_yaml_finds_root_file(tmp_path: Path) -> None:
    """Finds stropha-glossary.yaml in repo root."""
    (tmp_path / "stropha-glossary.yaml").touch()
    
    result = find_glossary_yaml(tmp_path)
    
    assert result == tmp_path / "stropha-glossary.yaml"


def test_find_glossary_yaml_finds_dotfolder_file(tmp_path: Path) -> None:
    """Finds .stropha/glossary.yaml."""
    (tmp_path / ".stropha").mkdir()
    (tmp_path / ".stropha" / "glossary.yaml").touch()
    
    result = find_glossary_yaml(tmp_path)
    
    assert result == tmp_path / ".stropha" / "glossary.yaml"


def test_find_glossary_yaml_finds_docs_file(tmp_path: Path) -> None:
    """Finds docs/glossary.yaml."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "glossary.yaml").touch()
    
    result = find_glossary_yaml(tmp_path)
    
    assert result == tmp_path / "docs" / "glossary.yaml"


def test_find_glossary_yaml_returns_none_when_not_found(tmp_path: Path) -> None:
    """Returns None when no glossary file exists."""
    result = find_glossary_yaml(tmp_path)
    
    assert result is None


def test_find_glossary_yaml_prefers_root_over_dotfolder(tmp_path: Path) -> None:
    """Root file takes precedence over .stropha folder."""
    (tmp_path / "stropha-glossary.yaml").touch()
    (tmp_path / ".stropha").mkdir()
    (tmp_path / ".stropha" / "glossary.yaml").touch()
    
    result = find_glossary_yaml(tmp_path)
    
    assert result == tmp_path / "stropha-glossary.yaml"
