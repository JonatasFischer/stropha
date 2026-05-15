"""Unit tests for the cost dashboard aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.cost import aggregate, default_log_paths, render_json, render_table


def _write_log(p: Path, lines: list[str]) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_aggregate_handles_missing_paths(tmp_path: Path) -> None:
    report = aggregate([tmp_path / "nope.log"])
    assert report.sources_scanned == []
    assert report.repos == {}


def test_aggregate_counts_repo_done(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha.log",
        [
            json.dumps({
                "event": "pipeline.repo_done",
                "repo": "github.com/foo/bar",
                "files": 100, "chunks": 1200, "embedded": 50,
                "reused": 1150, "enricher_cache_hits": 30,
                "timestamp": "2026-05-15T12:00:00Z",
            }),
            json.dumps({
                "event": "pipeline.repo_done",
                "repo": "github.com/foo/bar",
                "files": 100, "chunks": 1200, "embedded": 10,
                "reused": 1190, "enricher_cache_hits": 5,
                "timestamp": "2026-05-15T13:00:00Z",
            }),
        ],
    )
    report = aggregate([log])
    rc = report.repos["github.com/foo/bar"]
    assert rc.runs == 2
    assert rc.files_visited == 200
    assert rc.chunks_seen == 2400
    assert rc.chunks_embedded == 60
    assert rc.last_event == "2026-05-15T13:00:00Z"


def test_aggregate_counts_adapters_built(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha.log",
        [
            json.dumps({"event": "pipeline.adapter.built",
                        "stage": "embedder",
                        "adapter_id": "local:mxbai:1024"}),
            json.dumps({"event": "pipeline.adapter.built",
                        "stage": "embedder",
                        "adapter_id": "local:mxbai:1024"}),
            json.dumps({"event": "pipeline.adapter.built",
                        "stage": "walker",
                        "adapter_id": "git-ls-files:max=524288"}),
        ],
    )
    report = aggregate([log])
    assert report.adapters["local:mxbai:1024"].build_count == 2
    assert report.adapters["git-ls-files:max=524288"].build_count == 1


def test_aggregate_counts_graphify_load_and_embed(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha.log",
        [
            json.dumps({
                "event": "graphify_loader.done",
                "nodes": 1500, "edges_loaded": 8000, "edges_filtered": 2000,
                "timestamp": "2026-05-15T12:00:00Z",
            }),
            json.dumps({
                "event": "graph_vec_loader.done",
                "embedded": 1500,
            }),
        ],
    )
    report = aggregate([log])
    assert report.graph.nodes_loaded == 1500
    assert report.graph.edges_loaded == 8000
    assert report.graph.edges_filtered == 2000
    assert report.graph.embed_runs == 1
    assert report.graph.embedded_nodes == 1500


def test_aggregate_counts_hook_events(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha-hook.log",
        [
            "[2026-05-15T12:00:00Z] commit abc1234 — launching refresh",
            "[stropha-hook] graphify update (commit abc1234)",
            "[stropha-hook] refresh done (commit abc1234)",
            "[2026-05-15T13:00:00Z] commit def5678 — launching refresh",
            "[stropha-hook] graphify update (commit def5678)",
            "[stropha-hook] failed (commit def5678)",
        ],
    )
    report = aggregate([log])
    assert report.hook.refresh_launched == 2
    assert report.hook.refresh_done == 1
    assert report.hook.refresh_failed == 1
    assert report.hook.graphify_updates == 2


def test_aggregate_ignores_unparseable_lines(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha.log",
        [
            "garbage line one",
            "{not valid json}",
            json.dumps({"event": "pipeline.repo_done", "repo": "r",
                        "files": 1, "chunks": 1, "embedded": 0}),
        ],
    )
    report = aggregate([log])
    assert report.repos["r"].runs == 1


def test_render_table_returns_none_when_empty(tmp_path: Path) -> None:
    """Empty report → render_table returns None so caller can show error."""
    report = aggregate([tmp_path / "missing.log"])
    assert render_table(report) is None


def test_render_table_handles_populated_report(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha.log",
        [json.dumps({"event": "pipeline.repo_done", "repo": "r",
                     "files": 1, "chunks": 1, "embedded": 0})],
    )
    report = aggregate([log])
    out = render_table(report)
    assert out is not None  # rich.Group


def test_render_json_is_valid(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "stropha.log",
        [json.dumps({"event": "pipeline.repo_done", "repo": "r",
                     "files": 1, "chunks": 1, "embedded": 0})],
    )
    report = aggregate([log])
    out = render_json(report)
    parsed = json.loads(out)
    assert "repos" in parsed
    assert "r" in parsed["repos"]


def test_default_log_paths_returns_empty_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STROPHA_HOOK_LOG", str(tmp_path / "nope.log"))
    # The fake cache dir is empty too
    paths = default_log_paths()
    assert paths == []


def test_default_log_paths_finds_per_repo_logs(monkeypatch, tmp_path: Path) -> None:
    main = tmp_path / "stropha-hook.log"
    main.write_text("[stropha-hook] refresh done (commit a1)\n")
    sub1 = tmp_path / "stropha-hook-mimoria.log"
    sub1.write_text("[stropha-hook] refresh done (commit m1)\n")
    sub2 = tmp_path / "stropha-hook-foo.log"
    sub2.write_text("[stropha-hook] refresh done (commit f1)\n")
    monkeypatch.setenv("STROPHA_HOOK_LOG", str(main))
    paths = default_log_paths()
    assert main in paths
    assert sub1 in paths
    assert sub2 in paths
