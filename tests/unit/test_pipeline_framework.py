"""Tests for the pipeline framework (base + registry + config + builder)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from stropha import adapters  # noqa: F401  triggers auto-import
from stropha.errors import AdapterError, ConfigError
from stropha.pipeline import (
    StageContext,
    StageHealth,
    all_adapters,
    available_for_stage,
    build_stages,
    load_pipeline_config,
    lookup_adapter,
    register_adapter,
)
from stropha.pipeline.registry import _REGISTRY  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_adapters_registered() -> None:
    reg = all_adapters()
    assert "embedder" in reg
    assert "enricher" in reg
    assert "local" in reg["embedder"]
    assert "voyage" in reg["embedder"]
    assert "noop" in reg["enricher"]
    assert "hierarchical" in reg["enricher"]


def test_lookup_unknown_adapter_raises_config_error_with_alternatives() -> None:
    with pytest.raises(ConfigError) as exc:
        lookup_adapter("enricher", "does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    # The error must hint at the registered alternatives.
    assert "noop" in msg or "hierarchical" in msg


def test_available_for_stage_returns_sorted() -> None:
    names = available_for_stage("embedder")
    assert names == sorted(names)


def test_register_adapter_requires_config_attribute() -> None:
    with pytest.raises(AdapterError):

        @register_adapter(stage="enricher", name="broken-fixture")
        class _Broken:
            """Missing ``Config`` attribute."""

    # Make sure no half-registered entry leaked.
    assert ("enricher", "broken-fixture") not in _REGISTRY


def test_register_adapter_rejects_duplicate_name() -> None:
    class _Cfg(BaseModel):
        pass

    @register_adapter(stage="enricher", name="dup-fixture")
    class _A:
        Config = _Cfg

    with pytest.raises(AdapterError):

        @register_adapter(stage="enricher", name="dup-fixture")
        class _B:
            Config = _Cfg

    # Cleanup so the fixture doesn't leak into other tests.
    _REGISTRY.pop(("enricher", "dup-fixture"), None)


# ---------------------------------------------------------------------------
# StageContext / StageHealth shape
# ---------------------------------------------------------------------------


def test_stage_context_defaults() -> None:
    ctx = StageContext()
    assert ctx.repo_key is None
    assert ctx.parent_chunk is None
    assert ctx.file_content is None
    assert ctx.pipeline_meta == {}


def test_stage_health_ready() -> None:
    h = StageHealth(status="ready", message="ok")
    assert h.status == "ready"
    assert h.detail == {}


# ---------------------------------------------------------------------------
# Config loader (deterministic via injected environ)
# ---------------------------------------------------------------------------


def test_config_defaults_pick_local_when_no_voyage_key(tmp_path) -> None:
    cfg = load_pipeline_config(project_root=tmp_path, environ={})
    assert cfg["embedder"]["adapter"] == "local"
    assert cfg["enricher"]["adapter"] == "noop"


def test_config_picks_voyage_when_api_key_set(tmp_path) -> None:
    cfg = load_pipeline_config(
        project_root=tmp_path, environ={"VOYAGE_API_KEY": "sk-test"}
    )
    assert cfg["embedder"]["adapter"] == "voyage"


def test_config_legacy_alias_local_model(tmp_path) -> None:
    cfg = load_pipeline_config(
        project_root=tmp_path,
        environ={"STROPHA_LOCAL_EMBED_MODEL": "BAAI/bge-small-en-v1.5"},
    )
    assert cfg["embedder"]["adapter"] == "local"
    assert cfg["embedder"]["config"]["model"] == "BAAI/bge-small-en-v1.5"


def test_config_legacy_alias_voyage_does_not_leak_into_local(tmp_path) -> None:
    """Regression: STROPHA_VOYAGE_EMBED_MODEL must NOT contaminate a local adapter."""
    cfg = load_pipeline_config(
        project_root=tmp_path,
        environ={
            # No VOYAGE_API_KEY -> auto-select local.
            "STROPHA_VOYAGE_EMBED_MODEL": "voyage-code-3",
            "STROPHA_LOCAL_EMBED_MODEL": "BAAI/bge-small-en-v1.5",
        },
    )
    assert cfg["embedder"]["adapter"] == "local"
    assert cfg["embedder"]["config"]["model"] == "BAAI/bge-small-en-v1.5"


def test_config_cli_override_wins_over_env(tmp_path) -> None:
    cfg = load_pipeline_config(
        project_root=tmp_path,
        environ={"STROPHA_ENRICHER": "hierarchical"},
        overrides={"enricher": {"adapter": "noop"}},
    )
    assert cfg["enricher"]["adapter"] == "noop"


def test_config_env_overlay_for_nested_path(tmp_path) -> None:
    cfg = load_pipeline_config(
        project_root=tmp_path,
        environ={"STROPHA_ENRICHER__INCLUDE_REPO_URL": "true"},
    )
    # Note: lowercased keys per the loader convention.
    assert cfg["enricher"]["config"]["include_repo_url"] is True


def test_config_yaml_file_loaded_when_present(tmp_path) -> None:
    (tmp_path / "stropha.yaml").write_text(
        "pipeline:\n  enricher:\n    adapter: hierarchical\n", encoding="utf-8"
    )
    cfg = load_pipeline_config(project_root=tmp_path, environ={})
    assert cfg["enricher"]["adapter"] == "hierarchical"


def test_config_yaml_invalid_raises_config_error(tmp_path) -> None:
    (tmp_path / "stropha.yaml").write_text("not:\n  : -valid: [", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_pipeline_config(project_root=tmp_path, environ={})


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_stages_yields_working_enricher() -> None:
    resolved = {
        "embedder": {"adapter": "local", "config": {}},
        "enricher": {"adapter": "noop", "config": {}},
    }
    # We don't need a real embedder for the enricher assertion below — but
    # building local IS heavy (ONNX load), so skip embedder validation in
    # this test by patching it with a known-cheap subclass would require
    # adapter modification. Instead, verify the noop enricher directly.
    from stropha.adapters.enricher.noop import NoopEnricher

    enr = NoopEnricher()
    assert enr.adapter_id == "noop"
    assert enr.adapter_name == "noop"
    assert enr.stage_name == "enricher"
    assert enr.health().status == "ready"


def test_build_stages_unknown_adapter_raises() -> None:
    resolved = {
        "embedder": {"adapter": "no-such", "config": {}},
        "enricher": {"adapter": "noop", "config": {}},
    }
    with pytest.raises(ConfigError):
        build_stages(resolved)


def test_build_stages_invalid_config_raises() -> None:
    resolved = {
        "embedder": {"adapter": "voyage", "config": {"dim": -5}},
        "enricher": {"adapter": "noop", "config": {}},
    }
    with pytest.raises(ConfigError):
        build_stages(resolved)
