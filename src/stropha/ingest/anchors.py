"""Architectural anchor detection.

Identifies code elements that serve as architectural entry points or
important structural elements in a codebase:

- Controllers / Resolvers / Handlers (HTTP/GraphQL entry points)
- Aggregate roots / Entities (DDD patterns)
- Configuration classes
- Main entry points
- Service facades

Anchors are detected via:
1. Naming conventions (e.g., *Controller, *Service, *Repository)
2. Annotations/decorators (e.g., @Controller, @Entity, @Configuration)
3. File path patterns (e.g., controllers/, handlers/)

Detection is language-aware and extensible via patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from ..models import Chunk

log = get_logger(__name__)


@dataclass
class AnchorPattern:
    """A pattern for detecting architectural anchors."""
    
    name: str
    category: str  # controller, entity, config, service, etc.
    
    # Match criteria (any match triggers anchor)
    symbol_patterns: list[str] = field(default_factory=list)  # Regex for symbol name
    content_patterns: list[str] = field(default_factory=list)  # Regex for content
    path_patterns: list[str] = field(default_factory=list)  # Regex for file path
    kind_patterns: list[str] = field(default_factory=list)  # Exact match for chunk kind
    
    # Language filter (empty = all languages)
    languages: list[str] = field(default_factory=list)
    
    # Boost factor for search ranking (1.0 = no boost)
    boost: float = 1.5


# Default patterns for common architectural elements
DEFAULT_PATTERNS: list[AnchorPattern] = [
    # Controllers / HTTP handlers
    AnchorPattern(
        name="controller",
        category="entry_point",
        symbol_patterns=[
            r".*Controller$",
            r".*Handler$",
            r".*Endpoint$",
            r".*Resource$",
        ],
        content_patterns=[
            r"@Controller",
            r"@RestController",
            r"@RequestMapping",
            r"@GetMapping|@PostMapping|@PutMapping|@DeleteMapping",
            r"@app\.route|@router\.",
            r"@Get\(|@Post\(|@Put\(|@Delete\(",
        ],
        path_patterns=[
            r"/controllers?/",
            r"/handlers?/",
            r"/endpoints?/",
            r"/api/",
        ],
        kind_patterns=["class", "function", "method"],  # Include functions for Flask/FastAPI
        boost=1.8,
    ),
    
    # GraphQL resolvers
    AnchorPattern(
        name="resolver",
        category="entry_point",
        symbol_patterns=[
            r".*Resolver$",
            r".*Query$",
            r".*Mutation$",
        ],
        content_patterns=[
            r"@Resolver",
            r"@Query\(",
            r"@Mutation\(",
            r"@ResolveField",
        ],
        path_patterns=[
            r"/resolvers?/",
            r"/graphql/",
        ],
        kind_patterns=["class", "function"],
        boost=1.8,
    ),
    
    # Domain entities / Aggregate roots
    AnchorPattern(
        name="entity",
        category="domain",
        symbol_patterns=[
            r".*Entity$",
            r".*Aggregate$",
            r".*Root$",
        ],
        content_patterns=[
            r"@Entity",
            r"@AggregateRoot",
            r"@Document",
            r"@Table\(",
            r"class.*\(.*Model\)",
            r"class.*\(.*Base\)",
        ],
        path_patterns=[
            r"/domain/",
            r"/entities/",
            r"/models/",
            r"/aggregates/",
        ],
        kind_patterns=["class"],
        boost=1.6,
    ),
    
    # Repositories / Data access
    AnchorPattern(
        name="repository",
        category="infrastructure",
        symbol_patterns=[
            r".*Repository$",
            r".*Repo$",
            r".*DAO$",
            r".*Store$",
        ],
        content_patterns=[
            r"@Repository",
            r"extends.*Repository",
            r"implements.*Repository",
        ],
        path_patterns=[
            r"/repositories/",
            r"/repos/",
            r"/data/",
            r"/persistence/",
        ],
        kind_patterns=["class", "interface"],
        boost=1.5,
    ),
    
    # Services / Use cases
    AnchorPattern(
        name="service",
        category="application",
        symbol_patterns=[
            r".*Service$",
            r".*UseCase$",
            r".*Interactor$",
            r".*Manager$",
        ],
        content_patterns=[
            r"@Service",
            r"@Injectable",
            r"@Component",
        ],
        path_patterns=[
            r"/services/",
            r"/usecases/",
            r"/application/",
        ],
        kind_patterns=["class"],
        boost=1.4,
    ),
    
    # Configuration
    AnchorPattern(
        name="config",
        category="infrastructure",
        symbol_patterns=[
            r".*Config$",
            r".*Configuration$",
            r".*Settings$",
            r".*Options$",
        ],
        content_patterns=[
            r"@Configuration",
            r"@ConfigurationProperties",
            r"@Bean",
            r"settings\s*=",
            r"config\s*=",
        ],
        path_patterns=[
            r"/config/",
            r"/configuration/",
            r"/settings/",
        ],
        kind_patterns=["class", "module"],
        boost=1.3,
    ),
    
    # Main entry points
    AnchorPattern(
        name="main",
        category="entry_point",
        symbol_patterns=[
            r"^main$",
            r"^Main$",
            r"^App$",
            r"^Application$",
        ],
        content_patterns=[
            r"if\s+__name__\s*==\s*['\"]__main__['\"]",
            r"public\s+static\s+void\s+main\s*\(",
            r"func\s+main\s*\(",
            r"fn\s+main\s*\(",
            r"@SpringBootApplication",
            r"createApp\(",
            r"bootstrapApplication\(",
        ],
        path_patterns=[
            r"/main\.",
            r"/__main__\.",
            r"/app\.",
            r"/index\.",
        ],
        kind_patterns=["function", "method", "class", "module"],
        boost=2.0,
    ),
    
    # Test fixtures (useful for understanding test structure)
    AnchorPattern(
        name="test_fixture",
        category="test",
        symbol_patterns=[
            r".*Test$",
            r".*Tests$",
            r".*Spec$",
            r"^Test.*",
            r"^test_.*",
        ],
        content_patterns=[
            r"@Test",
            r"@pytest\.fixture",
            r"describe\(",
            r"it\(",
            r"def test_",
        ],
        path_patterns=[
            r"/tests?/",
            r"/__tests__/",
            r"\.test\.",
            r"\.spec\.",
            r"_test\.",
        ],
        kind_patterns=["class", "function"],
        boost=1.2,
    ),
    
    # Middleware / Interceptors
    AnchorPattern(
        name="middleware",
        category="infrastructure",
        symbol_patterns=[
            r".*Middleware$",
            r".*Interceptor$",
            r".*Filter$",
            r".*Guard$",
        ],
        content_patterns=[
            r"@Middleware",
            r"@UseInterceptors",
            r"@UseGuards",
            r"implements.*Interceptor",
            r"implements.*Filter",
        ],
        path_patterns=[
            r"/middleware/",
            r"/interceptors/",
            r"/filters/",
            r"/guards/",
        ],
        kind_patterns=["class"],
        boost=1.4,
    ),
]


@dataclass
class AnchorMatch:
    """Result of anchor detection for a chunk."""
    
    is_anchor: bool
    patterns_matched: list[str]
    categories: list[str]
    boost: float  # Maximum boost from matched patterns


class AnchorDetector:
    """Detects architectural anchors in code chunks."""
    
    def __init__(
        self,
        patterns: list[AnchorPattern] | None = None,
        custom_patterns: list[AnchorPattern] | None = None,
    ) -> None:
        """Initialize detector with patterns.
        
        Args:
            patterns: Override default patterns. If None, uses DEFAULT_PATTERNS.
            custom_patterns: Additional patterns to add to defaults.
        """
        self._patterns = patterns if patterns is not None else list(DEFAULT_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        
        # Pre-compile regex patterns
        self._compiled: dict[str, dict[str, list[re.Pattern]]] = {}
        for p in self._patterns:
            self._compiled[p.name] = {
                "symbol": [re.compile(r, re.IGNORECASE) for r in p.symbol_patterns],
                "content": [re.compile(r, re.MULTILINE) for r in p.content_patterns],
                "path": [re.compile(r, re.IGNORECASE) for r in p.path_patterns],
            }
    
    def detect(self, chunk: Chunk) -> AnchorMatch:
        """Check if a chunk is an architectural anchor.
        
        Args:
            chunk: The chunk to analyze.
            
        Returns:
            AnchorMatch with detection results.
        """
        matched_patterns: list[str] = []
        categories: set[str] = set()
        max_boost = 1.0
        
        for pattern in self._patterns:
            # Language filter
            if pattern.languages and chunk.language not in pattern.languages:
                continue
            
            # Kind filter
            if pattern.kind_patterns and chunk.kind not in pattern.kind_patterns:
                continue
            
            compiled = self._compiled[pattern.name]
            matched = False
            
            # Check symbol patterns
            if chunk.symbol and compiled["symbol"]:
                for regex in compiled["symbol"]:
                    if regex.search(chunk.symbol):
                        matched = True
                        break
            
            # Check content patterns
            if not matched and compiled["content"]:
                for regex in compiled["content"]:
                    if regex.search(chunk.content):
                        matched = True
                        break
            
            # Check path patterns
            if not matched and compiled["path"]:
                for regex in compiled["path"]:
                    if regex.search(chunk.rel_path):
                        matched = True
                        break
            
            if matched:
                matched_patterns.append(pattern.name)
                categories.add(pattern.category)
                max_boost = max(max_boost, pattern.boost)
        
        return AnchorMatch(
            is_anchor=len(matched_patterns) > 0,
            patterns_matched=matched_patterns,
            categories=list(categories),
            boost=max_boost,
        )
    
    def detect_batch(self, chunks: list[Chunk]) -> dict[str, AnchorMatch]:
        """Detect anchors for multiple chunks.
        
        Args:
            chunks: List of chunks to analyze.
            
        Returns:
            Dict mapping chunk_id to AnchorMatch.
        """
        results: dict[str, AnchorMatch] = {}
        for chunk in chunks:
            results[chunk.chunk_id] = self.detect(chunk)
        return results


def get_anchor_categories() -> list[str]:
    """Get list of all anchor categories from default patterns."""
    return sorted(set(p.category for p in DEFAULT_PATTERNS))


def get_anchor_pattern_names() -> list[str]:
    """Get list of all anchor pattern names."""
    return [p.name for p in DEFAULT_PATTERNS]
