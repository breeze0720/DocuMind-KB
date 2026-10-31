"""
core/retriever.py - 混合检索（向量 + BM25）与 RRF 融合

RRF（Reciprocal Rank Fusion，倒数排名融合）原理：
每路检索只取「排名」不取「分数」，第 r 名贡献 1/(k+r)，两路相加作为最终分。
为什么要这样：向量相似度（0~1 余弦）和 BM25 分数（无上界）量纲完全不同，
直接加权相加需要调参且换库就失效；RRF 只依赖排名，天然免疫量纲问题，是工业界的默认基线。

原实现的三个问题（本项目已修）：
1) 用正文当 key，不同文件里相同的句子会被合并成一条，来源和 chunk_index 全丢；
   现在 key=(source, 正文)，同文本不同来源各算各的。
2) 结果只返回 content+score，做不了引用溯源；现在带 source/chunk_index/page。
3) 把 RRF 分当成「相关度」喂给模型（模型看到 0.03 会以为很不相关），
   现在 score_kind='rrf' 明确标注量纲，agent 只按排名组织上下文。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

import config
from core.store import KnowledgeStore


def display_source(source: str) -> str:
    """展示用来源名：旧数据可能存的是绝对路径，统一取文件名。"""
    return Path(str(source)).name if source else ""


def doc_key(doc: Document) -> tuple[str, str]:
    """融合去重 key：(来源, 正文)。同一块在两路里出现只算一次，不同来源的同文本保持独立。"""
    return (str((doc.metadata or {}).get("source", "")), doc.page_content)


def fuse_rrf(
    vector_hits: list[tuple[Document, float]],
    keyword_hits: list[tuple[Document, float]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    融合两路检索结果。每个入参是 [(Document, 原始分)]，原始分只用于调试展示，不参与融合。

    返回按融合分降序的 [{"document", "score", "score_kind", "key"}]。
    同一 key 在两路都命中时分数相加——两路都认为相关，本来就该排更前。
    """
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], Document] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, (doc, _raw_score) in enumerate(hits):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        {
            "key": key,
            "document": docs[key],
            "score": scores[key],
            "score_kind": "rrf",
        }
        for key in ordered
    ]


class HybridRetriever:
    """对外只暴露 search()：调用方不需要知道下面是 Chroma、BM25 还是别的库。"""

    def __init__(
        self,
        store: KnowledgeStore,
        rrf_k: int | None = None,
        top_k: int | None = None,
        candidate_floor: int = 20,
    ):
        self.store = store
        self.rrf_k = int(rrf_k or config.settings.rrf_k)
        self.top_k = int(top_k or config.settings.top_k)
        # 融合前每路多取一些候选：RRF 只看排名，候选池太小会让「双路都命中的好块」排不进来。
        # 用固定下限 20 而不是只取 k*4，是为了 k=3 这类小请求也有足够的融合空间。
        self.candidate_floor = candidate_floor

    def search(
        self,
        query: str,
        k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        混合检索入口。

        参数 min_score：可选阈值，**只作用于向量那一路的 0~1 相关性分**。
        为什么不能过滤融合分：RRF 分是「排名的倒数之和」，量纲由 k（默认 60）决定，
        数值大小只表示相对顺序（第一名约 1/60≈0.017），和「相关度」没有可比性；
        拿它当阈值过滤会把所有结果都滤掉或全放行，取决于阈值怎么拍脑袋。
        同理 BM25 分也无上界，不能和向量分共用同一个阈值。
        """
        k = int(k or self.top_k)
        if not query or not query.strip() or k <= 0:
            return []

        pool = max(k * 4, self.candidate_floor)
        vector_hits = self.store.search(query, k=pool)
        if min_score is not None:
            # 只过滤向量路：低于阈值的块不参与融合，BM25 路不受影响（量纲不同）
            vector_hits = [(doc, score) for doc, score in vector_hits if score >= float(min_score)]
        keyword_hits = self.store.keyword_search(query, k=pool)

        fused = fuse_rrf(vector_hits, keyword_hits, rrf_k=self.rrf_k)
        results: list[dict[str, Any]] = []
        for rank, item in enumerate(fused[:k], start=1):
            doc: Document = item["document"]
            metadata = doc.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "content": doc.page_content,
                    "score": round(item["score"], 6),
                    "score_kind": "rrf",
                    "source": display_source(str(metadata.get("source", ""))),
                    "source_key": str(metadata.get("source", "")),
                    "chunk_index": metadata.get("chunk_index"),
                    "page": metadata.get("page"),
                    "sheet": metadata.get("sheet"),
                    "slide": metadata.get("slide"),
                    "section": metadata.get("section"),
                    "title": metadata.get("title", ""),
                }
            )
        return results