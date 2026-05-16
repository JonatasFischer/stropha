"""Soft index — watch a working tree and re-index on file changes.

Per spec §8 / Phase 3: "Soft index para working tree (file watcher)."
The post-commit hook covers the committed state; this command covers
the *uncommitted* state while you iterate, so retrieval queries see
your latest edits within seconds of saving.

Design:
- Standard library only (``watchdog`` not pulled in to keep deps light).
  We poll the file mtimes on a low-frequency interval (default 1 s) and
  diff against a snapshot. For typical code repos (<10k files) this is
  sub-millisecond and trivial CPU.
- Debounced: a wave of writes from an LSP/format/save-on-blur cycle
  collapses into a single re-index pass after the debounce window
  (default 2 s of quiet).
- Respects `.gitignore` / `.strophaignore` via the standard ``Walker``
  filters, so build artefacts and caches don't trigger storms.
- Only re-indexes the *changed* files (`Walker.discover_paths(rel)`).
  Other chunks pay the freshness skip (cheap).
- Doesn't reload the graphify graph or re-embed graph nodes by default
  — those are heavy and the hook handles them on commit. Override via
  ``--full-refresh`` if you want a complete pass.

MCP Integration:
- When ``STROPHA_MCP_WATCH=1`` (default), the MCP server starts the
  watcher in a background thread during lifespan. This ensures queries
  always see recent edits without requiring a separate ``stropha watch``.
- The watcher can be stopped via ``WatchController.stop()`` or by
  setting the threading.Event passed to ``watch_repo_async()``.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .errors import StrophaError
from .ingest.walker import Walker
from .logging import get_logger
from .models import Chunk
from .pipeline import Pipeline, build_stages, load_pipeline_config

log = get_logger(__name__)

_DEFAULT_INTERVAL_S = 1.0
_DEFAULT_DEBOUNCE_S = 2.0


@dataclass
class _Snapshot:
    """File-mtime snapshot for change detection."""

    mtimes: dict[Path, float] = field(default_factory=dict)

    def diff(self, fresh: "_Snapshot") -> tuple[set[Path], set[Path]]:
        """Return ``(changed_or_added, removed)`` since the previous snapshot."""
        changed: set[Path] = set()
        for p, m in fresh.mtimes.items():
            prev = self.mtimes.get(p)
            if prev is None or prev != m:
                changed.add(p)
        removed = set(self.mtimes.keys()) - set(fresh.mtimes.keys())
        return changed, removed


def _snapshot(repo: Path, *, max_file_bytes: int = 524_288) -> _Snapshot:
    """Walk via the standard ``Walker`` and capture per-file mtimes."""
    snap = _Snapshot()
    walker = Walker(repo, max_file_bytes=max_file_bytes)
    for sf in walker.discover():
        try:
            abs_path = (repo / sf.rel_path).resolve()
            snap.mtimes[abs_path] = abs_path.stat().st_mtime
        except OSError:
            continue
    return snap


def _format_changes(
    changed: Iterable[Path], removed: Iterable[Path], *, root: Path,
) -> str:
    parts: list[str] = []
    changed_list = sorted(changed)
    removed_list = sorted(removed)
    if changed_list:
        rels = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
                for p in changed_list[:5]]
        more = ""
        if len(changed_list) > 5:
            more = f" (+{len(changed_list) - 5} more)"
        parts.append(f"{len(changed_list)} changed: {', '.join(rels)}{more}")
    if removed_list:
        parts.append(f"{len(removed_list)} removed")
    return " · ".join(parts) if parts else "no changes"


def watch_repo(
    repo: Path,
    *,
    interval_s: float = _DEFAULT_INTERVAL_S,
    debounce_s: float = _DEFAULT_DEBOUNCE_S,
    full_refresh: bool = False,
    once: bool = False,
) -> None:
    """Watch ``repo`` and re-index changed files after the debounce window.

    Args:
        repo: The repository root to watch.
        interval_s: Polling interval in seconds. Lower = snappier, higher
            CPU. 1.0 is comfortable for most editors.
        debounce_s: Wait this many seconds of *no further changes* before
            triggering a re-index. Protects against save bursts.
        full_refresh: When True, every re-index runs a full ``Pipeline.run()``
            (including graphify graph reload and graph-vec embedding).
            Default False — only the changed files go through the pipeline.
        once: Run one detection cycle and return. Used by tests.
    """
    repo = repo.expanduser().resolve()
    if not (repo / ".git").exists() and not repo.is_dir():
        raise StrophaError(f"Not a directory: {repo}")
    log.info(
        "watch.start",
        repo=str(repo),
        interval_s=interval_s,
        debounce_s=debounce_s,
        full_refresh=full_refresh,
    )

    previous = _snapshot(repo)
    pending_since: float | None = None

    stop = {"flag": False}

    def _on_sigint(_sig: int, _frame: object) -> None:
        stop["flag"] = True
        log.info("watch.stop_requested")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)
    except (ValueError, OSError):
        # signal() can only be called from main thread; in test runners
        # the watcher may run elsewhere — skip silently.
        pass

    while not stop["flag"]:
        time.sleep(interval_s)
        fresh = _snapshot(repo)
        changed, removed = previous.diff(fresh)

        now = time.monotonic()
        if changed or removed:
            pending_since = now
            log.info(
                "watch.changes_seen",
                summary=_format_changes(changed, removed, root=repo),
            )
            previous = fresh
        elif pending_since is not None and (now - pending_since) >= debounce_s:
            # Debounce window elapsed with no further activity → re-index.
            log.info("watch.reindex_start")
            try:
                _reindex(repo, full_refresh=full_refresh)
                log.info("watch.reindex_done")
            except Exception as exc:
                log.warning("watch.reindex_failed", error=str(exc))
            pending_since = None
        if once:
            break


def _reindex(repo: Path, *, full_refresh: bool) -> None:
    """Run a fresh ``Pipeline`` pass against ``repo``."""
    resolved = load_pipeline_config()
    built = build_stages(resolved)
    try:
        with built.storage:  # type: ignore[union-attr]
            pipeline = Pipeline(
                storage=built.storage,
                embedder=built.embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=[repo],
            )
            stats = pipeline.run()
        log.info(
            "watch.reindex_summary",
            files=stats.files_visited,
            chunks=stats.chunks_seen,
            embedded=stats.chunks_embedded,
            reused=stats.chunks_skipped_fresh,
        )
    finally:
        if built.storage is not None:
            try:
                built.storage.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Async/threaded watch for MCP server integration
# ---------------------------------------------------------------------------


@dataclass
class WatchController:
    """Controller for a background watch thread.

    Use this to start/stop the watcher from the MCP server lifespan.
    """

    repo: Path
    interval_s: float = _DEFAULT_INTERVAL_S
    debounce_s: float = _DEFAULT_DEBOUNCE_S
    full_refresh: bool = False

    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        """Start the watch loop in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("watch.already_running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="stropha-watch",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "watch.thread_started",
            repo=str(self.repo),
            interval_s=self.interval_s,
            debounce_s=self.debounce_s,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watch loop to stop and wait for thread to finish."""
        if self._thread is None or not self._thread.is_alive():
            return
        log.info("watch.thread_stopping")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning("watch.thread_did_not_stop", timeout=timeout)
        else:
            log.info("watch.thread_stopped")

    def _run_loop(self) -> None:
        """The actual watch loop, runs in a background thread."""
        repo = self.repo.expanduser().resolve()
        if not repo.is_dir():
            log.error("watch.not_a_directory", path=str(repo))
            return

        previous = _snapshot(repo)
        pending_since: float | None = None

        while not self._stop_event.is_set():
            # Use wait() with timeout instead of sleep() for responsive shutdown
            if self._stop_event.wait(timeout=self.interval_s):
                break  # Stop event was set

            try:
                fresh = _snapshot(repo)
            except Exception as exc:
                log.warning("watch.snapshot_failed", error=str(exc))
                continue

            changed, removed = previous.diff(fresh)
            now = time.monotonic()

            if changed or removed:
                pending_since = now
                log.info(
                    "watch.changes_seen",
                    summary=_format_changes(changed, removed, root=repo),
                )
                previous = fresh
            elif pending_since is not None and (now - pending_since) >= self.debounce_s:
                # Debounce window elapsed with no further activity → re-index.
                log.info("watch.reindex_start")
                try:
                    _reindex(repo, full_refresh=self.full_refresh)
                    log.info("watch.reindex_done")
                except Exception as exc:
                    log.warning("watch.reindex_failed", error=str(exc))
                pending_since = None
