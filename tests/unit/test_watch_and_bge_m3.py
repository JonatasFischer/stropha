"""Tests for `stropha watch` snapshot diffing and bge-m3 adapter registration.

`watch_repo`'s blocking loop is exercised only via `once=True` since we
don't want a real polling loop in unit tests. The bulk of the file is
about the snapshot diff helpers, which are deterministic and small.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stropha.adapters.embedder.bge_m3 import BgeM3Config, BgeM3Embedder
from stropha.pipeline.registry import lookup_adapter
from stropha.watch import _format_changes, _Snapshot, _snapshot, WatchController


# --------------------------------------------------------------------------- snapshot diff


def test_snapshot_empty_diff() -> None:
    a = _Snapshot()
    b = _Snapshot()
    changed, removed = a.diff(b)
    assert changed == set()
    assert removed == set()


def test_snapshot_detects_new_files() -> None:
    a = _Snapshot()
    b = _Snapshot(mtimes={Path("/x/a.py"): 1.0})
    changed, removed = a.diff(b)
    assert changed == {Path("/x/a.py")}


def test_snapshot_detects_changed_mtimes() -> None:
    a = _Snapshot(mtimes={Path("/x/a.py"): 1.0})
    b = _Snapshot(mtimes={Path("/x/a.py"): 2.0})
    changed, removed = a.diff(b)
    assert changed == {Path("/x/a.py")}


def test_snapshot_detects_removed() -> None:
    a = _Snapshot(mtimes={Path("/x/a.py"): 1.0, Path("/x/b.py"): 1.0})
    b = _Snapshot(mtimes={Path("/x/a.py"): 1.0})
    changed, removed = a.diff(b)
    assert changed == set()
    assert removed == {Path("/x/b.py")}


def test_snapshot_walks_real_directory(tmp_path: Path) -> None:
    """End-to-end: snapshot picks up real files via the Walker."""
    # The Walker requires a git repo for git-ls-files but falls back to
    # filesystem walk when .git is absent.
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("# hi", encoding="utf-8")
    snap = _snapshot(tmp_path)
    file_names = {p.name for p in snap.mtimes}
    assert "a.py" in file_names
    assert "b.md" in file_names


# --------------------------------------------------------------------------- formatting


def test_format_changes_empty() -> None:
    out = _format_changes([], [], root=Path("/x"))
    assert out == "no changes"


def test_format_changes_only_changed() -> None:
    out = _format_changes(
        [Path("/x/a.py"), Path("/x/b.py")], [], root=Path("/x"),
    )
    assert "2 changed" in out
    assert "a.py" in out


def test_format_changes_only_removed() -> None:
    out = _format_changes(
        [], [Path("/x/old.py")], root=Path("/x"),
    )
    assert "1 removed" in out


def test_format_changes_caps_long_lists() -> None:
    many = [Path(f"/x/f{i}.py") for i in range(10)]
    out = _format_changes(many, [], root=Path("/x"))
    assert "10 changed" in out
    assert "more" in out  # the "(+N more)" elision


# --------------------------------------------------------------------------- WatchController


def test_watch_controller_starts_and_stops(tmp_path: Path) -> None:
    """WatchController can be started and stopped cleanly."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    ctrl = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=0.1)
    ctrl.start()
    assert ctrl._thread is not None
    assert ctrl._thread.is_alive()
    ctrl.stop(timeout=2.0)
    assert not ctrl._thread.is_alive()


def test_watch_controller_double_start_is_noop(tmp_path: Path) -> None:
    """Starting twice doesn't spawn two threads."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    ctrl = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=0.1)
    ctrl.start()
    first_thread = ctrl._thread
    ctrl.start()  # Should be no-op
    assert ctrl._thread is first_thread
    ctrl.stop(timeout=2.0)


def test_watch_controller_stop_without_start_is_noop(tmp_path: Path) -> None:
    """Stopping without starting doesn't raise."""
    ctrl = WatchController(repo=tmp_path)
    ctrl.stop()  # Should not raise


# --------------------------------------------------------------------------- bge-m3 registry


def test_bge_m3_registered_in_adapter_registry() -> None:
    cls = lookup_adapter("embedder", "bge-m3")
    assert cls is BgeM3Embedder


def test_bge_m3_default_model_is_bge_m3() -> None:
    cfg = BgeM3Config()
    assert cfg.model == "BAAI/bge-m3"


def test_bge_m3_adapter_id_prefix() -> None:
    """We don't load the real model in tests — just check the class
    declares the correct adapter_name without instantiating."""
    assert BgeM3Embedder.__name__ == "BgeM3Embedder"
    # adapter_name is a @property on instances; check via class-level
    # attribute access by reading the source contract.
    import inspect
    src = inspect.getsource(BgeM3Embedder.adapter_name.fget)  # type: ignore[union-attr]
    assert '"bge-m3"' in src
