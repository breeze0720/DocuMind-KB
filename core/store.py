"""
core/store.py - 存储层：Chroma 向量索引 + BM25 关键词索引的统一门面

为什么要一个 store 类同时管两套索引：
向量检索擅长语义（「年假」能匹配「带薪休假」），BM25 擅长精确词（工号、金额、条款号）。
两套索引必须「同增同删」，否则会出现「向量能搜到、关键词搜不到」的幽灵块。
原实现把 Chroma 和 BM25 文档列表拆在两个全局变量里，删除时靠调用方手工同步，迟早漏；
现在所有写操作只经过这个类，删除时在同一个锁里同步两边。

BM25 的两个性能/效果坑（原实现都有）：
1) 原来每次查询都 BM25Retriever.from_documents(...) 重建索引，1000 块时每次多 50-200ms。
   现在索引是进程级缓存的，只在 add/delete 后重建，并持久化到 JSON，重启不用重算。
2) 原来 BM25Retriever 默认按空格分词，中文整段被当成一个 token，等于没做关键词检索。
   现在用 jieba.lcut_for_search 分词（搜索引擎模式会把长词再切出子词，召回更高）。

Chroma collection 名称由 embedding 后端决定（带维度后缀），换后端不会维度冲突，见 embeddings.py。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from core.bm25 import bm25_tokenize, build_bm25, load_cached_docs, save_cached_docs

logger = logging.getLogger(__name__)


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Chroma 的 metadata 只接受 str/int/float/bool，None 会被拒绝。
    空值直接丢掉而不是塞空字符串：过滤条件（如 where={"source": x}）不会被空串干扰。
    """
    cleaned: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned


class KnowledgeStore:
    """一个 collection 对应一个 store；换模型/换索引时整体换 collection，互不干扰。"""

    def __init__(self, chroma_dir: str | Path, embeddings: Embeddings, collection_name: str):
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.collection_name = collection_name
        # BM25 缓存文件名带 collection：换 collection/换模型时旧缓存不会被误读
        self.bm25_cache_path = self.chroma_dir / f"bm25_docs_{collection_name}.json"
        self._lock = threading.RLock()
        self._docs: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        # 与 _docs 一一对应的分词结果：关键词命中判定用它，避免「BM25 分数为 0 但其实词命中了」
        # 被误过滤（小语料下 idf 可能恰好为 0，例如 2 篇文档里 1 篇含该词）
        self._tokens: list[list[str]] = []
        self._vs = Chroma(
            persist_directory=str(self.chroma_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
            # 显式用余弦空间：LangChain 的 relevance_score 对 l2 会换算成 1 - distance/√2，
            # 不相关文档可能得到负数；cosine 下分数稳定落在 0~1，min_score 阈值才有可比性。
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._load_indexes()

    # ---------------- 初始化：载入 / 校准 BM25 缓存 ----------------
    def _load_indexes(self) -> None:
        """
        Chroma 是唯一事实来源，BM25 JSON 只是加速缓存：
        缓存缺失（首次启动）或与 Chroma 块数不一致（别人直接写库/缓存损坏）时，
        从 Chroma 全量读回重建，保证两套索引永远一致。
        """
        docs = load_cached_docs(self.bm25_cache_path)
        try:
            chroma_count = self._vs._collection.count()
        except Exception as exc:
            logger.warning("读取 Chroma 块数失败: %s", exc)
            chroma_count = len(docs)
        if (not docs and chroma_count) or len(docs) != chroma_count:
            docs = self._read_all_from_chroma()
        self._docs = docs
        self._rebuild_bm25()

    def _read_all_from_chroma(self) -> list[dict[str, Any]]:
        """全量读取（只用于启动/校准，不在查询路径上）。按 source+chunk_index 排序保证 BM25 结果稳定。"""
        try:
            data = self._vs._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("从 Chroma 读取全部文档失败: %s", exc)
            return []
        docs = []
        for doc_id, text, metadata in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []):
            if text:
                docs.append({"id": doc_id, "content": text, "metadata": metadata or {}})
        docs.sort(key=lambda item: (str(item["metadata"].get("source", "")), int(item["metadata"].get("chunk_index") or 0)))
        return docs

    def _rebuild_bm25(self) -> None:
        """重建内存索引 + 落盘。调用方必须已持有 self._lock（写路径都在锁内）。"""
        self._bm25, self._tokens = build_bm25(self._docs)
        save_cached_docs(self.bm25_cache_path, self._docs)

    # ---------------- 写 ----------------
    def add_documents(self, documents: list[Document]) -> int:
        """
        写入文档块。用 upsert + 确定性 id，而不是 add：
        同一个文件被重复导入时，同 id 覆盖旧块，知识库不会出现两份相同内容（幂等）。
        id 由 file_hash + source + chunk_index 组成，内容不变时 id 不变。
        """
        if not documents:
            return 0
        texts = [doc.page_content for doc in documents]
        metadatas = [_clean_metadata(doc.metadata) for doc in documents]
        ids: list[str] = []
        for metadata in metadatas:
            file_hash = str(metadata.get("file_hash") or "nohash")[:16]
            ids.append(f"{file_hash}:{metadata.get('source', '')}:{metadata.get('chunk_index', 0)}")

        vectors = self.embeddings.embed_documents(texts)  # 先算向量再进锁，减少锁持有时间
        with self._lock:
            try:
                self._vs._collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
            except Exception as exc:
                raise RuntimeError(f"写入 Chroma 失败: {exc}") from exc
            replaced = set(ids)
            self._docs = [item for item in self._docs if item["id"] not in replaced]
            self._docs.extend(
                {"id": doc_id, "content": text, "metadata": metadata}
                for doc_id, text, metadata in zip(ids, texts, metadatas)
            )
            self._rebuild_bm25()
        return len(documents)

    def delete_source(self, source: str) -> int:
        """
        按来源删除整篇文档的块，Chroma 和 BM25 在同一个锁里同步删。
        source 兼容两种写法：文件名（新数据）和旧版本存的全路径（老 chroma_db 里的数据）。
        """
        if not source:
            return 0
        target_name = Path(source).name
        with self._lock:
            ids = [
                item["id"]
                for item in self._docs
                if str(item["metadata"].get("source", "")) == source
                or Path(str(item["metadata"].get("source", ""))).name == target_name
            ]
            if not ids:
                return 0
            try:
                self._vs._collection.delete(ids=ids)
            except Exception as exc:
                raise RuntimeError(f"从 Chroma 删除失败: {exc}") from exc
            removed = set(ids)
            self._docs = [item for item in self._docs if item["id"] not in removed]
            self._rebuild_bm25()
        return len(ids)

    # ---------------- 读 ----------------
    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """向量检索，返回 [(Document, relevance_score)]，分数是 LangChain 归一化到 0~1 的相关性。"""
        if not query.strip() or self.count() == 0:
            return []
        try:
            pairs = self._vs.similarity_search_with_relevance_scores(query, k=max(1, int(k)))
        except Exception as exc:
            logger.warning("向量检索失败: %s", exc)
            return []
        return [(doc, float(score)) for doc, score in pairs]

    def keyword_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        BM25 检索：只返回「至少命中一个查询词」的块。
        不能简单用 score > 0 过滤：小语料下 idf 可能恰好为 0（2 篇文档里 1 篇含该词时
        log((N-df+0.5)/(df+0.5)) = 0），真正命中的块分数也是 0，会被误杀。
        所以先用分词结果的交集判断「词命中」，再按 BM25 分数排序。
        """
        with self._lock:
            bm25, docs, tokens_list = self._bm25, list(self._docs), list(self._tokens)
        if bm25 is None or not tokens_list:
            return []
        tokens = bm25_tokenize(query)
        if not tokens:
            return []
        query_set = set(tokens)
        matched = [index for index, doc_tokens in enumerate(tokens_list) if query_set & set(doc_tokens)]
        if not matched:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(matched, key=lambda index: scores[index], reverse=True)[: max(1, int(k))]
        results: list[tuple[Document, float]] = []
        for index in ranked:
            item = docs[index]
            results.append((Document(page_content=item["content"], metadata=item["metadata"]), float(scores[index])))
        return results

    def count(self) -> int:
        """知识库块数。以内存镜像为准（写路径总是同步维护它），避免每次查询打一次 Chroma。"""
        return len(self._docs)

    def find_source_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """
        上传幂等检查：内容哈希已存在就返回已有来源信息，调用方直接返回「已存在」。
        为什么按内容哈希而不是按文件名：用户把同一份文件改名再传，也不该重复入库。
        """
        if not file_hash:
            return None
        chunks = 0
        source = ""
        title = ""
        for item in self._docs:
            metadata = item["metadata"]
            if str(metadata.get("file_hash", "")) == file_hash:
                chunks += 1
                source = source or str(metadata.get("source", ""))
                title = title or str(metadata.get("title", ""))
        if chunks:
            return {"source": Path(source).name if source else "", "source_key": source, "title": title, "chunks": chunks}
        return None

    def list_sources(self) -> list[dict[str, Any]]:
        """按文件汇总：给 /api/documents 用。source 是展示名（文件名），source_key 是删除用的原始值。"""
        groups: dict[str, dict[str, Any]] = {}
        for item in self._docs:
            metadata = item["metadata"]
            key = str(metadata.get("source", "") or "")
            group = groups.get(key)
            if group is None:
                group = {
                    "source": Path(key).name if key else "",
                    "source_key": key,
                    "title": str(metadata.get("title", "")),
                    "file_hash": str(metadata.get("file_hash", "")),
                    "chunks": 0,
                }
                groups[key] = group
            group["chunks"] += 1
        return sorted(groups.values(), key=lambda item: item["source"])