"""Tests for architectural anchor detection."""

from __future__ import annotations

import pytest

from stropha.ingest.anchors import (
    AnchorDetector,
    AnchorMatch,
    AnchorPattern,
    DEFAULT_PATTERNS,
    get_anchor_categories,
    get_anchor_pattern_names,
)
from stropha.models import Chunk


def _make_chunk(
    symbol: str = "TestClass",
    content: str = "class TestClass {}",
    rel_path: str = "src/test.py",
    kind: str = "class",
    language: str = "python",
) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        chunk_id=f"test:{symbol}",
        rel_path=rel_path,
        language=language,
        kind=kind,
        symbol=symbol,
        parent_chunk_id=None,
        start_line=1,
        end_line=10,
        content=content,
        content_hash="abc123",
    )


# --------------------------------------------------------------------------- #
#                           AnchorDetector Tests                              #
# --------------------------------------------------------------------------- #


class TestControllerDetection:
    """Test detection of controllers/handlers."""
    
    def test_detects_controller_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="UserController", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "controller" in match.patterns_matched
        assert "entry_point" in match.categories
    
    def test_detects_controller_by_annotation(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="Users",
            content="@RestController\npublic class Users {}",
            kind="class",
            language="java",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "controller" in match.patterns_matched
    
    def test_detects_handler_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="AuthHandler", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "controller" in match.patterns_matched
    
    def test_detects_flask_route(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="get_users",
            content="@app.route('/users')\ndef get_users(): pass",
            kind="function",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
    
    def test_detects_by_path(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="Users",
            rel_path="src/controllers/users.py",
            kind="class",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor


class TestEntityDetection:
    """Test detection of domain entities."""
    
    def test_detects_entity_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="UserEntity", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "entity" in match.patterns_matched
        assert "domain" in match.categories
    
    def test_detects_entity_by_annotation(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="User",
            content="@Entity\npublic class User {}",
            kind="class",
            language="java",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "entity" in match.patterns_matched
    
    def test_detects_aggregate_root(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="OrderAggregate", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor


class TestServiceDetection:
    """Test detection of services."""
    
    def test_detects_service_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="UserService", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "service" in match.patterns_matched
        assert "application" in match.categories
    
    def test_detects_usecase_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="CreateUserUseCase", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "service" in match.patterns_matched


class TestMainDetection:
    """Test detection of main entry points."""
    
    def test_detects_python_main(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="main",
            content='if __name__ == "__main__":\n    main()',
            kind="module",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "main" in match.patterns_matched
        assert match.boost >= 2.0
    
    def test_detects_java_main(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="main",
            content="public static void main(String[] args) {}",
            kind="method",
            language="java",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
    
    def test_detects_spring_boot_app(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="Application",
            content="@SpringBootApplication\npublic class Application {}",
            kind="class",
            language="java",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor


class TestConfigDetection:
    """Test detection of configuration classes."""
    
    def test_detects_config_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="DatabaseConfig", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "config" in match.patterns_matched
        assert "infrastructure" in match.categories
    
    def test_detects_settings_by_name(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="AppSettings", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor


class TestNonAnchorDetection:
    """Test that non-anchors are not detected."""
    
    def test_regular_class_not_anchor(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="Helper",
            content="class Helper:\n    def do_thing(self): pass",
            kind="class",
        )
        
        match = detector.detect(chunk)
        
        assert not match.is_anchor
        assert match.patterns_matched == []
        assert match.boost == 1.0
    
    def test_regular_function_not_anchor(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="process_data",
            content="def process_data(x): return x * 2",
            kind="function",
        )
        
        match = detector.detect(chunk)
        
        assert not match.is_anchor


class TestCustomPatterns:
    """Test custom pattern support."""
    
    def test_custom_pattern_detection(self) -> None:
        custom = AnchorPattern(
            name="custom_worker",
            category="background",
            symbol_patterns=[r".*Worker$"],
            boost=1.5,
        )
        
        detector = AnchorDetector(custom_patterns=[custom])
        chunk = _make_chunk(symbol="EmailWorker", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        assert "custom_worker" in match.patterns_matched
        assert "background" in match.categories
    
    def test_override_default_patterns(self) -> None:
        minimal = AnchorPattern(
            name="only_main",
            category="entry",
            symbol_patterns=[r"^main$"],
            boost=1.0,
        )
        
        detector = AnchorDetector(patterns=[minimal])
        
        # Controller should NOT match with overridden patterns
        chunk1 = _make_chunk(symbol="UserController", kind="class")
        assert not detector.detect(chunk1).is_anchor
        
        # main should still match
        chunk2 = _make_chunk(symbol="main", kind="function")
        assert detector.detect(chunk2).is_anchor


class TestBatchDetection:
    """Test batch detection."""
    
    def test_detect_batch(self) -> None:
        detector = AnchorDetector()
        chunks = [
            _make_chunk(symbol="UserController", kind="class"),
            _make_chunk(symbol="Helper", kind="class"),
            _make_chunk(symbol="UserService", kind="class"),
        ]
        
        results = detector.detect_batch(chunks)
        
        assert len(results) == 3
        assert results["test:UserController"].is_anchor
        assert not results["test:Helper"].is_anchor
        assert results["test:UserService"].is_anchor


class TestBoostValues:
    """Test boost value assignment."""
    
    def test_main_has_highest_boost(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(
            symbol="main",
            content='if __name__ == "__main__": pass',
            kind="module",
        )
        
        match = detector.detect(chunk)
        
        assert match.boost >= 2.0
    
    def test_controller_has_high_boost(self) -> None:
        detector = AnchorDetector()
        chunk = _make_chunk(symbol="UserController", kind="class")
        
        match = detector.detect(chunk)
        
        assert match.boost >= 1.5
    
    def test_multiple_matches_use_max_boost(self) -> None:
        detector = AnchorDetector()
        # This matches both controller (by path) and service (by name)
        chunk = _make_chunk(
            symbol="UserService",
            rel_path="src/controllers/user_service.py",
            kind="class",
        )
        
        match = detector.detect(chunk)
        
        assert match.is_anchor
        # Should have the highest boost from matched patterns
        assert match.boost > 1.0


# --------------------------------------------------------------------------- #
#                           Helper Function Tests                             #
# --------------------------------------------------------------------------- #


def test_get_anchor_categories() -> None:
    """Categories are returned sorted."""
    categories = get_anchor_categories()
    
    assert len(categories) > 0
    assert categories == sorted(categories)
    assert "entry_point" in categories
    assert "domain" in categories


def test_get_anchor_pattern_names() -> None:
    """Pattern names are returned."""
    names = get_anchor_pattern_names()
    
    assert len(names) > 0
    assert "controller" in names
    assert "entity" in names
    assert "service" in names
    assert "main" in names


def test_default_patterns_exist() -> None:
    """Default patterns list is populated."""
    assert len(DEFAULT_PATTERNS) > 0
    
    # Check required fields are set
    for p in DEFAULT_PATTERNS:
        assert p.name
        assert p.category
        assert p.boost >= 1.0
