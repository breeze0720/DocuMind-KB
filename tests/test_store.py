"""tests/test_store.py - 存储层测试：Chroma + BM25 同增同删、幂等写入、BM25 缓存持久化。

用 4 维「假 embedding」替代真实模型：store 的逻辑与模型无关，
这样测试不用加载 91MB ONNX、不依赖网络，跑起来也快。
"""
import json

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.store import KnowledgeStore


class FakeEmbeddings(Embeddings):
    """确定性假向量（4 维），保证同一文本总是同一向量，测试结果可复现。"""

    def _vec(self, text: str) -> list[float]:
        return [float(len(text) % 5), float(sum(ord(char) for char in text) % 7), 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_docs() -> list[Document]:
    return [
        Document(
            page_content="员工年假为每年十天",
            metadata={"source": "员工手册.md", "chunk_index": 0, "file_hash": "hash-a", "title": "员工手册"},
        ),
        Document(
            page_content="报销需要发票和审批单",
            metadata={"source": "报销制度.md", "chunk_index": 0, "file_hash": "hash-b", "title": "报销制度"},
        ),
    ]


def test_add_search_keyword_and_delete(tmp_path):
    chroma_dir = tmp_path / "chroma"
    store = KnowledgeStore(chroma_dir, FakeEmbeddings(), "test_collection")

    assert store.count() == 0
    assert store.add_documents(make_docs()) == 2
    assert store.count() == 2

    # BM25 缓存落盘（文件名带 collection，避免换后端互相覆盖）
    cache_path = chroma_dir / "bm25_docs_test_collection.json"
    assert cache_path.is_file()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(cached) == 2

    # 关键词检索：jieba 能切出「年假」，命中对应块
    keyword_hits = store.keyword_search("年假", k=3)
    assert keyword_hits
    assert keyword_hits[0][0].page_content == "员工年假为每年十天"

    # 向量检索可用
    assert store.search("年假", k=2)

    # 哈希幂等查询
    existing = store.find_source_by_hash("hash-a")
    assert existing and existing["source"] == "员工手册.md" and existing["chunks"] == 1

    # 文件列表按来源聚合
    sources = {item["source"]: item["chunks"] for item in store.list_sources()}
    assert sources == {"员工手册.md": 1, "报销制度.md": 1}


def test_repeated_add_is_idempotent(tmp_path):
    """重复导入同一文件（id 相同）不能把知识库块数翻倍。"""
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbeddings(), "test_collection")
    store.add_documents(make_docs())
    store.add_documents(make_docs())
    assert store.count() == 2


def test_delete_syncs_both_indexes(tmp_path):
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbeddings(), "test_collection")
    store.add_documents(make_docs())
    assert store.delete_source("员工手册.md") == 1
    assert store.count() == 1
    # 两套索引都删干净：关键词检索不再命中已删除内容
    assert store.keyword_search("年假", k=3) == []
    assert store.find_source_by_hash("hash-a") is None
    # 未匹配的来源返回 0，而不是报错
    assert store.delete_source("不存在.md") == 0


def test_bm25_cache_survives_reopen(tmp_path):
    """模拟进程重启：新 store 从 JSON 缓存 + Chroma 恢复，块数与删除状态一致。"""
    chroma_dir = tmp_path / "chroma"
    store = KnowledgeStore(chroma_dir, FakeEmbeddings(), "test_collection")
    store.add_documents(make_docs())
    store.delete_source("员工手册.md")

    reopened = KnowledgeStore(chroma_dir, FakeEmbeddings(), "test_collection")
    assert reopened.count() == 1
    assert reopened.find_source_by_hash("hash-a") is None
    assert reopened.find_source_by_hash("hash-b") is not None
    assert reopened.keyword_search("报销", k=3)


def test_delete_accepts_legacy_absolute_source(tmp_path):
    """旧 chroma_db 里 source 存的是绝对路径；按文件名删除也要能命中。"""
    store = KnowledgeStore(tmp_path / "chroma", FakeEmbeddings(), "test_collection")
    doc = Document(
        page_content="旧数据内容",
        metadata={"source": r"D:\old\uploads\旧文件.txt", "chunk_index": 0, "file_hash": "hash-old"},
    )
    store.add_documents([doc])
    assert store.delete_source("旧文件.txt") == 1
    assert store.count() == 0