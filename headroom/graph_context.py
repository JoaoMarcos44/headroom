"""Thin client for import-graph neighborhood queries.

Graph-scoped narrowing needs one fact: "which files are within N import-hops
of file X". `codebase-memory-mcp` (see `headroom/graph/`) already indexes
that — the proxy's `CodeGraphWatcher` keeps it live via debounced reindexing.
This module asks that same binary instead of re-parsing every supported
language with an in-process tree-sitter graph, so language coverage and
graph correctness stay the indexer's problem, not this repo's.

Failure mode: any uncertainty here (binary missing, query error, timeout)
returns `None`, which callers must treat as "don't narrow" — this is a pure
optimization, so it must degrade to zero-cost pass-through, never to an
incorrectly filtered result.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from headroom._subprocess import run
from headroom.graph.installer import get_cbm_path

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300
DEFAULT_MAX_HOPS = 2
_QUERY_TIMEOUT_SECONDS = 2.0


class GraphContext:
    """Caches import-graph neighborhoods per `(entry_file, max_hops)`."""

    def __init__(
        self,
        project_dir: str | Path,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        cbm_binary: str | None = None,
    ) -> None:
        self.project_dir = str(project_dir)
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple[str, int], tuple[float, frozenset[str]]] = {}
        if cbm_binary:
            self._cbm_binary: str | None = cbm_binary
        else:
            path = get_cbm_path()
            self._cbm_binary = str(path) if path else None

    @property
    def available(self) -> bool:
        return self._cbm_binary is not None

    def related_files(
        self, entry_file: str, max_hops: int = DEFAULT_MAX_HOPS
    ) -> frozenset[str] | None:
        """Files within `max_hops` import-hops of `entry_file`.

        Returns `None` when the codebase-memory-mcp binary isn't installed
        or the query fails for any reason — see module docstring.
        """
        if self._cbm_binary is None:
            return None

        key = (entry_file, max_hops)
        cached = self._cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < self.ttl_seconds:
            return cached[1]

        files = self._query(entry_file, max_hops)
        if files is None:
            return None

        self._cache[key] = (time.monotonic(), files)
        return files

    def _query(self, entry_file: str, max_hops: int) -> frozenset[str] | None:
        try:
            result = run(
                [
                    self._cbm_binary,
                    "cli",
                    "related_files",
                    json.dumps(
                        {
                            "repo_path": self.project_dir,
                            "entry_file": entry_file,
                            "max_hops": max_hops,
                        }
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=_QUERY_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("graph_context: related_files query failed: %s", e)
            return None

        if result.returncode != 0:
            logger.debug(
                "graph_context: related_files exited %d: %s", result.returncode, result.stderr
            )
            return None

        try:
            payload = json.loads(result.stdout)
            return frozenset(payload["files"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("graph_context: related_files returned malformed output: %s", e)
            return None

    def clear_cache(self) -> None:
        self._cache.clear()
