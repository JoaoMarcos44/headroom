"""Tests for graph-scoped narrowing of discovery-tool output (#1925, lean design).

Covers the three pieces added to ContentRouter:
- `_extract_last_read_file`: finds the BFS entry point from message history.
- `_graph_narrow`: the actual line-filtering + CCR-recoverable collapse.
- Wiring: `_get_graph_context` config gate, and the `apply()` pipeline hook.

Design contract under test (see headroom/graph_context.py and
headroom/transforms/content_router.py `_graph_narrow`): any uncertainty
(no entry file, unsupported tool, output already narrow, GraphContext
unavailable, query failure) must degrade to *unchanged* content — this
is a pure optimization, never a correctness risk.
"""

from __future__ import annotations

import pytest

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.parser import CCR_RETRIEVAL_MARKER_RE
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


@pytest.fixture(autouse=True)
def _clean_compression_store():
    reset_compression_store()
    yield
    reset_compression_store()


@pytest.fixture
def tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


@pytest.fixture
def router():
    return ContentRouter(ContentRouterConfig(min_section_tokens=10))


class _FakeGraphContext:
    """Stand-in for GraphContext so tests don't depend on codebase-memory-mcp
    actually being installed."""

    def __init__(self, related: frozenset[str] | None):
        self.available = True
        self._related = related
        self.calls: list[tuple[str, int]] = []

    def related_files(self, entry_file: str, max_hops: int = 2) -> frozenset[str] | None:
        self.calls.append((entry_file, max_hops))
        return self._related


def _grep_content(related: list[str], unrelated: list[str], lines_each: int = 5) -> str:
    lines = []
    for f in related:
        lines.extend(f"{f}:{i}:def handler_{i}():" for i in range(1, lines_each + 1))
    for f in unrelated:
        lines.extend(f"{f}:{i}:def other_{i}():" for i in range(1, lines_each + 1))
    return "\n".join(lines)


def _bare_path_content(related: list[str], unrelated: list[str], repeat: int = 4) -> str:
    return "\n".join((related + unrelated) * repeat)


def _messages_with_read_then_discover(
    read_path: str | None, discover_tool: str, discover_content: str
) -> list[dict]:
    messages: list[dict] = []
    if read_path is not None:
        messages += [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read_1",
                        "name": "Read",
                        "input": {"file_path": read_path},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_1",
                        "content": "line 1\nline 2\n",
                    }
                ],
            },
        ]
    messages += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_discover_1",
                    "name": discover_tool,
                    "input": {},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_discover_1",
                    "content": discover_content,
                }
            ],
        },
    ]
    return messages


# =============================================================================
# _extract_last_read_file
# =============================================================================


class TestExtractLastReadFile:
    def test_no_read_calls_returns_none(self, router):
        messages = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        assert router._extract_last_read_file(messages) is None

    def test_single_read_call(self, router):
        messages = _messages_with_read_then_discover("src/app.py", "Grep", "x" * 10)
        assert router._extract_last_read_file(messages) == "src/app.py"

    def test_keeps_most_recent_read(self, router):
        messages = _messages_with_read_then_discover("src/old.py", "Grep", "x" * 10)
        messages.insert(
            0,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read_0",
                        "name": "Read",
                        "input": {"file_path": "src/older.py"},
                    }
                ],
            },
        )
        # Re-append a later Read so "old.py" is no longer last.
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read_2",
                        "name": "Read",
                        "input": {"file_path": "src/newest.py"},
                    }
                ],
            }
        )
        assert router._extract_last_read_file(messages) == "src/newest.py"

    def test_alternate_tool_name_and_arg_key(self, router):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "lib/util.rs"},
                    }
                ],
            }
        ]
        assert router._extract_last_read_file(messages) == "lib/util.rs"

    def test_ignores_non_read_tool_use(self, router):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            }
        ]
        assert router._extract_last_read_file(messages) is None


# =============================================================================
# _graph_narrow — passthrough / degrade-safe branches
# =============================================================================


class TestGraphNarrowPassthrough:
    def test_passthrough_when_no_entry_file(self, router, monkeypatch):
        monkeypatch.setattr(
            router, "_get_graph_context", lambda: _FakeGraphContext(frozenset({"a.py"}))
        )
        content = _grep_content(["a.py"], ["b.py", "c.py"] * 5)
        assert router._graph_narrow(content, "Grep", None) == content

    def test_passthrough_for_non_discovery_tool(self, router, monkeypatch):
        monkeypatch.setattr(
            router, "_get_graph_context", lambda: _FakeGraphContext(frozenset({"a.py"}))
        )
        content = _grep_content(["a.py"], ["b.py"] * 10)
        assert router._graph_narrow(content, "Read", "a.py") == content

    def test_passthrough_when_output_already_narrow(self, router, monkeypatch):
        monkeypatch.setattr(
            router, "_get_graph_context", lambda: _FakeGraphContext(frozenset({"a.py"}))
        )
        content = "a.py:1:x\nb.py:1:y\n"  # well under graph_narrow_min_lines
        assert router._graph_narrow(content, "Grep", "a.py") == content

    def test_passthrough_when_graph_context_disabled(self, router):
        router.config.enable_graph_narrow = False
        content = _grep_content(["a.py"], ["b.py"] * 10, lines_each=3)
        assert router._graph_narrow(content, "Grep", "a.py") == content

    def test_passthrough_when_graph_context_unavailable_in_env(self, router):
        # No mocking: in this test environment codebase-memory-mcp isn't
        # installed, so GraphContext.available is False and the real
        # _get_graph_context() returns None — proves the zero-cost
        # pass-through contract without faking anything.
        content = _grep_content(["a.py"], ["b.py"] * 10, lines_each=3)
        assert router._graph_narrow(content, "Grep", "a.py") == content

    def test_passthrough_when_related_files_query_fails(self, router, monkeypatch):
        monkeypatch.setattr(router, "_get_graph_context", lambda: _FakeGraphContext(None))
        content = _grep_content(["a.py"], ["b.py"] * 10, lines_each=3)
        assert router._graph_narrow(content, "Grep", "a.py") == content

    def test_passthrough_when_nothing_would_be_dropped(self, router, monkeypatch):
        related = frozenset({"a.py", "b.py"})
        monkeypatch.setattr(router, "_get_graph_context", lambda: _FakeGraphContext(related))
        content = _grep_content(["a.py", "b.py"], [], lines_each=15)
        assert router._graph_narrow(content, "Grep", "a.py") == content


# =============================================================================
# _graph_narrow — actual narrowing behavior
# =============================================================================


class TestGraphNarrowBehavior:
    @pytest.mark.parametrize("tool_name", ["Grep", "Glob", "LS"])
    def test_drops_unrelated_files_keeps_related(self, router, monkeypatch, tool_name):
        related = frozenset({"src/app.py"})
        monkeypatch.setattr(router, "_get_graph_context", lambda: _FakeGraphContext(related))
        if tool_name == "Grep":
            content = _grep_content(
                ["src/app.py"], ["vendor/lib.py", "docs/notes.py"], lines_each=15
            )
        else:
            content = _bare_path_content(
                ["src/app.py"], ["vendor/lib.py", "docs/notes.py"], repeat=10
            )

        result = router._graph_narrow(content, tool_name, "src/app.py")

        assert "vendor/lib.py" not in result
        assert "docs/notes.py" not in result
        assert "src/app.py" in result

    def test_emits_ccr_recoverable_marker_with_correct_dropped_count(self, router, monkeypatch):
        related = frozenset({"src/app.py"})
        monkeypatch.setattr(router, "_get_graph_context", lambda: _FakeGraphContext(related))
        content = _grep_content(["src/app.py"], ["vendor/lib.py"], lines_each=15)

        result = router._graph_narrow(content, "Grep", "src/app.py")

        assert CCR_RETRIEVAL_MARKER_RE.search(result)
        assert result.rstrip().endswith("15_rows_offloaded>>")

    def test_dropped_content_is_recoverable_via_ccr_store(self, router, monkeypatch):
        related = frozenset({"src/app.py"})
        monkeypatch.setattr(router, "_get_graph_context", lambda: _FakeGraphContext(related))
        content = _grep_content(["src/app.py"], ["vendor/lib.py"], lines_each=15)

        result = router._graph_narrow(content, "Grep", "src/app.py")

        import re

        match = re.search(r"<<ccr:([a-f0-9]{12,24})\b", result)
        assert match is not None
        entry = get_compression_store().retrieve(match.group(1))
        assert entry is not None
        assert entry.original_content == content
        assert "vendor/lib.py" in entry.original_content

    def test_respects_configured_max_hops(self, router, monkeypatch):
        captured = {}

        class _RecordingGraphContext(_FakeGraphContext):
            def related_files(self, entry_file, max_hops=2):
                captured["max_hops"] = max_hops
                return super().related_files(entry_file, max_hops)

        router.config.graph_narrow_max_hops = 3
        monkeypatch.setattr(
            router, "_get_graph_context", lambda: _RecordingGraphContext(frozenset({"a.py"}))
        )
        content = _grep_content(["a.py"], ["b.py"] * 10, lines_each=3)
        router._graph_narrow(content, "Grep", "a.py")
        assert captured["max_hops"] == 3


# =============================================================================
# _get_graph_context — config gate
# =============================================================================


def test_get_graph_context_returns_none_when_disabled(router):
    router.config.enable_graph_narrow = False
    assert router._get_graph_context() is None


def test_get_graph_context_returns_none_when_cbm_not_installed(router):
    # Real GraphContext, no mocking: in this test environment
    # codebase-memory-mcp is not installed.
    assert router._get_graph_context() is None


# =============================================================================
# End-to-end: router.apply()
# =============================================================================


class TestApplyIntegration:
    @pytest.mark.parametrize("tool_name", ["Grep", "Glob", "LS"])
    def test_narrowing_flows_through_apply(self, router, tokenizer, monkeypatch, tool_name):
        related = frozenset({"src/app.py"})
        monkeypatch.setattr(router, "_get_graph_context", lambda: _FakeGraphContext(related))

        if tool_name == "Grep":
            content = _grep_content(["src/app.py"], ["vendor/lib.py"], lines_each=15)
        else:
            content = _bare_path_content(["src/app.py"], ["vendor/lib.py"], repeat=12)

        messages = _messages_with_read_then_discover("src/app.py", tool_name, content)

        result = router.apply(
            messages,
            tokenizer,
            # Disable the normal compression pipeline so the final block
            # content is exactly what _graph_narrow produced — proves the
            # wiring (block rebinding) without another compressor's output
            # format obscuring the assertion.
            min_chars_for_block_compression=10**9,
        )

        assert "router:graph_narrow" in result.transforms_applied
        discover_result = next(
            b
            for m in result.messages
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("tool_use_id") == "toolu_discover_1"
        )
        assert "vendor/lib.py" not in discover_result["content"]
        assert "src/app.py" in discover_result["content"]
        assert CCR_RETRIEVAL_MARKER_RE.search(discover_result["content"])

    def test_no_narrowing_without_a_prior_read(self, router, tokenizer, monkeypatch):
        monkeypatch.setattr(
            router, "_get_graph_context", lambda: _FakeGraphContext(frozenset({"a.py"}))
        )
        content = _grep_content(["a.py"], ["b.py"] * 10, lines_each=3)
        messages = _messages_with_read_then_discover(None, "Grep", content)

        result = router.apply(messages, tokenizer, min_chars_for_block_compression=10**9)

        assert "router:graph_narrow" not in result.transforms_applied
        discover_result = next(
            b
            for m in result.messages
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("tool_use_id") == "toolu_discover_1"
        )
        assert discover_result["content"] == content

    def test_disabled_config_skips_narrowing_in_apply(self, router, tokenizer):
        # No mocking: enable_graph_narrow=False must short-circuit
        # _get_graph_context() before it ever looks at GraphContext.
        router.config.enable_graph_narrow = False
        content = _grep_content(["a.py"], ["b.py"] * 10, lines_each=3)
        messages = _messages_with_read_then_discover("a.py", "Grep", content)

        result = router.apply(messages, tokenizer, min_chars_for_block_compression=10**9)

        assert "router:graph_narrow" not in result.transforms_applied
