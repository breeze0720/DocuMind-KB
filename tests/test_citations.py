"""tests/test_citations.py - 引用编号测试：[n] 与 sources 的映射必须严格一一对应。"""
from core.citations import CitationRegistry, format_numbered_context


def make_hit(source: str, chunk_index: int, content: str = "年假十天", **extra):
    return {
        "source": source,
        "source_key": f"key/{source}",
        "chunk_index": chunk_index,
        "content": content,
        "title": source,
        **extra,
    }


def test_register_assigns_sequential_numbers():
    registry = CitationRegistry()
    assert registry.register(make_hit("a.pdf", 0)) == 1
    assert registry.register(make_hit("b.pdf", 0)) == 2
    assert registry.register(make_hit("a.pdf", 1)) == 3
    sources = registry.sources()
    assert [item["n"] for item in sources] == [1, 2, 3]
    assert [item["source"] for item in sources] == ["a.pdf", "b.pdf", "a.pdf"]


def test_register_is_idempotent_for_same_chunk():
    """同一块被两路检索重复命中，不能拿到两个编号（否则 [1] 和 [2] 内容一样）。"""
    registry = CitationRegistry()
    first = registry.register(make_hit("a.pdf", 3))
    second = registry.register(make_hit("a.pdf", 3))
    assert first == second == 1
    assert len(registry) == 1


def test_register_does_not_merge_same_text_from_different_sources():
    """正文相同但来源不同 → 两个编号，前端点击 [1]/[2] 能分别定位。"""
    registry = CitationRegistry()
    assert registry.register(make_hit("a.pdf", 0, content="相同的一段话")) == 1
    assert registry.register(make_hit("b.pdf", 0, content="相同的一段话")) == 2
    assert len(registry) == 2


def test_sources_snapshot_has_preview_and_location():
    registry = CitationRegistry()
    registry.register(make_hit("员工手册.docx", 5, content="年假为每年十天。"))
    source = registry.sources()[0]
    assert source["n"] == 1
    assert source["source"] == "员工手册.docx"
    assert source["chunk_index"] == 5
    assert source["preview"].startswith("年假")


def test_format_numbered_context_maps_numbers_to_sources():
    registry = CitationRegistry()
    hits = [make_hit("a.pdf", 0, content="内容甲"), make_hit("b.pdf", 1, content="内容乙")]
    context = format_numbered_context(hits, registry)
    assert "[1] 来源" in context and "a.pdf" in context
    assert "[2] 来源" in context and "b.pdf" in context
    assert "内容甲" in context and "内容乙" in context
    assert [item["n"] for item in registry.sources()] == [1, 2]


def test_reset_clears_registry():
    registry = CitationRegistry()
    registry.register(make_hit("a.pdf", 0))
    registry.reset()
    assert len(registry) == 0
    assert registry.register(make_hit("b.pdf", 0)) == 1