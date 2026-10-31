"""tests/test_retriever.py - RRF 融合测试：同文本不同来源不合并、双路命中不重复、阈值只作用于单路。"""
from langchain_core.documents import Document

from core.retriever import HybridRetriever, doc_key, fuse_rrf


def make_doc(content: str, source: str, chunk_index: int = 0, **extra):
    metadata = {"source": source, "chunk_index": chunk_index, **extra}
    return Document(page_content=content, metadata=metadata)


class FakeStore:
    """最小存储桩：只实现 retriever 用到的两个检索方法，不碰 Chroma/BM25。"""

    def __init__(self, vector_hits=None, keyword_hits=None):
        self.vector_hits = vector_hits or []
        self.keyword_hits = keyword_hits or []

    def search(self, query, k=5):
        return self.vector_hits[:k]

    def keyword_search(self, query, k=5):
        return self.keyword_hits[:k]

    def count(self):
        return len(self.vector_hits) + len(self.keyword_hits)


def test_doc_key_includes_source():
    """key 是 (来源, 正文)：同一段文字来自不同文件时 key 不同，这是「不合并」的基础。"""
    left = make_doc("相同内容", "a.pdf")
    right = make_doc("相同内容", "b.pdf")
    assert doc_key(left) != doc_key(right)


def test_fuse_rrf_keeps_same_text_from_different_sources():
    vector_hits = [(make_doc("年假十天", "员工手册.docx"), 0.9), (make_doc("年假十天", "FAQ.txt"), 0.8)]
    fused = fuse_rrf(vector_hits, [], rrf_k=60)
    assert len(fused) == 2, "不同来源的同文本被错误合并了"
    assert {item["document"].metadata["source"] for item in fused} == {"员工手册.docx", "FAQ.txt"}
    # 第 0 名贡献 1/60，第 1 名贡献 1/61，排序不能反
    assert fused[0]["score"] > fused[1]["score"]


def test_fuse_rrf_dedupes_same_chunk_across_two_paths():
    """同一块（来源+正文都相同）在向量和 BM25 两路都命中时，只保留一条，分数相加。"""
    doc = make_doc("报销需要发票", "制度.pdf", chunk_index=3)
    fused = fuse_rrf([(doc, 0.8)], [(doc, 12.5)], rrf_k=60)
    assert len(fused) == 1
    assert fused[0]["score"] == 1 / 60 + 1 / 60
    assert fused[0]["score_kind"] == "rrf"


def test_hybrid_search_score_kind_and_fields():
    store = FakeStore(vector_hits=[(make_doc("员工年假十天", "员工手册.docx", 2), 0.77)], keyword_hits=[])
    retriever = HybridRetriever(store, rrf_k=60, top_k=5)
    results = retriever.search("年假多少天", k=5)
    assert len(results) == 1
    result = results[0]
    assert result["score_kind"] == "rrf"
    assert result["source"] == "员工手册.docx"
    assert result["chunk_index"] == 2
    assert result["rank"] == 1
    assert result["content"] == "员工年假十天"


def test_min_score_filters_only_vector_branch():
    """
    min_score 只过滤向量路（0~1 相关性）；BM25 路不受影响。
    这里低分文档 A 在向量路被滤掉，但它也在 BM25 路里 → 仍然出现在结果中；
    这说明阈值没有去过滤融合分（也证明阈值只对单路分数生效）。
    """
    low_doc = make_doc("低相关但关键词命中", "b.txt")
    high_doc = make_doc("高相关文档", "a.txt")
    store = FakeStore(vector_hits=[(high_doc, 0.9), (low_doc, 0.2)], keyword_hits=[(low_doc, 5.0)])
    retriever = HybridRetriever(store, rrf_k=60, top_k=5)
    results = retriever.search("关键词", k=5, min_score=0.5)
    sources = {item["source"] for item in results}
    assert "a.txt" in sources
    assert "b.txt" in sources, "BM25 路不应该被向量阈值过滤"


def test_both_paths_rank_higher():
    """两路都命中的块应排在只单路命中的块前面（RRF 分数相加）。"""
    shared = make_doc("两路都命中", "shared.pdf")
    only_vector = make_doc("只有向量命中", "vec.pdf")
    store = FakeStore(vector_hits=[(only_vector, 0.95), (shared, 0.7)], keyword_hits=[(shared, 8.0)])
    results = HybridRetriever(store, rrf_k=60, top_k=5).search("问题", k=5)
    assert results[0]["source"] == "shared.pdf"


def test_empty_query_returns_empty():
    store = FakeStore(vector_hits=[(make_doc("x", "a.txt"), 0.5)])
    assert HybridRetriever(store).search("   ") == []


def test_k_limits_results():
    hits = [(make_doc(f"内容{i}", f"f{i}.txt", i), 0.5) for i in range(10)]
    results = HybridRetriever(FakeStore(vector_hits=hits), top_k=10).search("内容", k=3)
    assert len(results) == 3
    assert [item["rank"] for item in results] == [1, 2, 3]