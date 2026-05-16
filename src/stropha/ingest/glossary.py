"""Domain glossary ingestion and management.

A glossary is a collection of domain-specific terms with their definitions.
Each term is indexed as a special chunk (kind=glossary) with its own embedding,
enabling conceptual search to surface relevant domain context.

The glossary can be populated from:
1. YAML files (stropha-glossary.yaml in the repo root)
2. Manual additions via CLI (stropha glossary add)
3. Auto-extraction via LLM (future: scan code for domain terms)

Example YAML format:

```yaml
terms:
  - term: FSRS
    definition: "Free Spaced Repetition Scheduler - an algorithm for optimal review timing"
    aliases: ["Free Spaced Repetition Scheduler", "fsrs-algorithm"]
  
  - term: Mastery
    definition: "A measure of how well a user has learned a concept (0-1 scale)"
    aliases: ["mastery level", "mastery score"]
```
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..logging import get_logger
from ..models import Chunk
from ..storage import Storage

log = get_logger(__name__)


@dataclass
class GlossaryTerm:
    """A single glossary entry."""
    
    term: str
    definition: str
    aliases: list[str] | None = None
    category: str | None = None
    
    @property
    def content_hash(self) -> str:
        """Unique hash based on term + definition."""
        data = f"{self.term}:{self.definition}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
    
    def to_chunk(self, repo_id: int | None = None) -> Chunk:
        """Convert to a Chunk for indexing."""
        # Build searchable content from term + aliases + definition
        parts = [self.term]
        if self.aliases:
            parts.extend(self.aliases)
        parts.append(self.definition)
        content = "\n".join(parts)
        
        # Symbol is the term itself for exact lookups
        symbol = self.term
        
        return Chunk(
            chunk_id=f"glossary:{self.content_hash}",
            rel_path="<glossary>",
            language="text",
            kind="glossary",
            symbol=symbol,
            parent_chunk_id=None,
            start_line=0,
            end_line=0,
            content=content,
            content_hash=self.content_hash,
        )


@dataclass
class GlossaryStats:
    """Statistics about the loaded glossary."""
    
    total_terms: int
    by_category: dict[str, int]
    last_updated: str | None


class GlossaryLoader:
    """Load and manage the domain glossary."""
    
    def __init__(self, storage: Storage) -> None:
        self._storage = storage
    
    def load_yaml(self, yaml_path: Path, repo_id: int | None = None) -> list[GlossaryTerm]:
        """Load glossary terms from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file.
            repo_id: Optional repo ID to associate terms with.
            
        Returns:
            List of loaded terms.
        """
        if not yaml_path.exists():
            log.debug("glossary.yaml_not_found", path=str(yaml_path))
            return []
        
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            log.warning("glossary.yaml_parse_error", path=str(yaml_path), error=str(exc))
            return []
        
        if not data or "terms" not in data:
            log.debug("glossary.yaml_empty", path=str(yaml_path))
            return []
        
        terms: list[GlossaryTerm] = []
        for entry in data.get("terms", []):
            if not isinstance(entry, dict):
                continue
            term = entry.get("term")
            definition = entry.get("definition")
            if not term or not definition:
                continue
            
            terms.append(GlossaryTerm(
                term=str(term),
                definition=str(definition),
                aliases=entry.get("aliases"),
                category=entry.get("category"),
            ))
        
        log.info("glossary.loaded", path=str(yaml_path), count=len(terms))
        return terms
    
    def index_terms(
        self,
        terms: list[GlossaryTerm],
        embedder: Any,
        enricher_id: str = "noop",
        repo_id: int | None = None,
    ) -> int:
        """Index glossary terms as chunks.
        
        Args:
            terms: List of terms to index.
            embedder: Embedder instance for generating embeddings.
            enricher_id: ID of the enricher used.
            repo_id: Optional repo ID.
            
        Returns:
            Number of terms indexed.
        """
        if not terms:
            return 0
        
        indexed = 0
        for term in terms:
            chunk = term.to_chunk(repo_id)
            
            # Check if already indexed with same content
            if self._storage.chunk_exists(chunk.chunk_id):
                existing = self._storage.get_chunk_by_id(chunk.chunk_id)
                if existing and existing.get("content_hash") == chunk.content_hash:
                    log.debug("glossary.skip_unchanged", term=term.term)
                    continue
            
            # Generate embedding
            embedding_text = chunk.content
            embedding = embedder.embed_query(embedding_text)
            
            # Store the chunk
            self._storage.upsert_chunk(
                chunk=chunk,
                embedding=embedding,
                embedding_text=embedding_text,
                enricher_id=enricher_id,
                repo_id=repo_id,
            )
            indexed += 1
            log.debug("glossary.indexed", term=term.term)
        
        if indexed > 0:
            self._storage.set_meta("glossary_last_updated", datetime.now(UTC).isoformat())
        
        log.info("glossary.index_complete", indexed=indexed, total=len(terms))
        return indexed
    
    def add_term(
        self,
        term: str,
        definition: str,
        aliases: list[str] | None = None,
        category: str | None = None,
        embedder: Any = None,
        enricher_id: str = "noop",
        repo_id: int | None = None,
    ) -> GlossaryTerm:
        """Add a single term to the glossary.
        
        Args:
            term: The term to add.
            definition: Definition of the term.
            aliases: Optional list of aliases.
            category: Optional category.
            embedder: Embedder for generating embedding (required).
            enricher_id: ID of the enricher.
            repo_id: Optional repo ID.
            
        Returns:
            The created GlossaryTerm.
        """
        entry = GlossaryTerm(
            term=term,
            definition=definition,
            aliases=aliases,
            category=category,
        )
        
        if embedder:
            self.index_terms([entry], embedder, enricher_id, repo_id)
        
        return entry
    
    def remove_term(self, term: str) -> bool:
        """Remove a term from the glossary.
        
        Args:
            term: The term to remove.
            
        Returns:
            True if removed, False if not found.
        """
        # Find chunks with kind=glossary and symbol=term
        cur = self._storage._conn.cursor()
        rows = cur.execute(
            "SELECT chunk_id FROM chunks WHERE kind = 'glossary' AND symbol = ?",
            (term,),
        ).fetchall()
        
        if not rows:
            return False
        
        for row in rows:
            self._storage.delete_chunk(row["chunk_id"])
        
        log.info("glossary.removed", term=term, count=len(rows))
        return True
    
    def list_terms(self) -> list[dict[str, Any]]:
        """List all glossary terms.
        
        Returns:
            List of term dictionaries with term, definition, aliases.
        """
        cur = self._storage._conn.cursor()
        rows = cur.execute(
            "SELECT symbol, content FROM chunks WHERE kind = 'glossary' ORDER BY symbol",
        ).fetchall()
        
        terms: list[dict[str, Any]] = []
        for row in rows:
            # Parse content back to term + definition
            parts = row["content"].split("\n")
            term = row["symbol"]
            definition = parts[-1] if parts else ""
            aliases = parts[1:-1] if len(parts) > 2 else []
            
            terms.append({
                "term": term,
                "definition": definition,
                "aliases": aliases,
            })
        
        return terms
    
    def stats(self) -> GlossaryStats:
        """Get glossary statistics.
        
        Returns:
            GlossaryStats with counts and metadata.
        """
        cur = self._storage._conn.cursor()
        
        total = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE kind = 'glossary'",
        ).fetchone()[0]
        
        # Categories would require schema change, return empty for now
        by_category: dict[str, int] = {}
        
        last_updated = self._storage.get_meta("glossary_last_updated")
        
        return GlossaryStats(
            total_terms=total,
            by_category=by_category,
            last_updated=last_updated,
        )
    
    def export_yaml(self, output_path: Path) -> int:
        """Export glossary to YAML format.
        
        Args:
            output_path: Path to write the YAML file.
            
        Returns:
            Number of terms exported.
        """
        terms = self.list_terms()
        if not terms:
            return 0
        
        data = {"terms": terms}
        
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        log.info("glossary.exported", path=str(output_path), count=len(terms))
        return len(terms)


def find_glossary_yaml(repo_path: Path) -> Path | None:
    """Find glossary YAML file in a repository.
    
    Looks for:
    1. stropha-glossary.yaml
    2. .stropha/glossary.yaml
    3. docs/glossary.yaml
    
    Args:
        repo_path: Root path of the repository.
        
    Returns:
        Path to the glossary file, or None if not found.
    """
    candidates = [
        repo_path / "stropha-glossary.yaml",
        repo_path / ".stropha" / "glossary.yaml",
        repo_path / "docs" / "glossary.yaml",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None
