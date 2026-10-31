"""
core/runtime.py - 进程级运行时单例

为什么需要它：
本地 bge 模型加载要 1-2 秒，Chroma collection + BM25 缓存也要读盘。如果每个请求都新建一遍，
第一次查询会卡顿、内存里还会出现多份索引。这里用三个模块级变量把
embedding / store / retriever 各缓存一份，进程内复用。

为什么不需要字典/按后端缓存：向量化后端只有一个（本地 bge），
曾经为了做「本地 vs 云端」的 A/B 对比按后端名缓存过一份，对比结束后那条分支已经删掉。
"""
from __future__ import annotations

import logging
import threading

import config
from core.embeddings import EmbeddingError, embedding_status, get_embeddings
from core.retriever import HybridRetriever
from core.store import KnowledgeStore

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_store: KnowledgeStore | None = None
_retriever: HybridRetriever | None = None


def get_store() -> KnowledgeStore:
    """取存储层（Chroma 向量索引 + BM25 缓存）。首次访问时才加载模型/读缓存。"""
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is None:
            embeddings = get_embeddings()
            _store = KnowledgeStore(
                config.settings.chroma_dir,
                embeddings,
                config.settings.collection_name,
            )
        return _store


def get_retriever() -> HybridRetriever:
    """取混合检索器（store + RRF 参数）。"""
    global _retriever
    if _retriever is not None:
        return _retriever
    with _lock:
        if _retriever is None:
            _retriever = HybridRetriever(
                get_store(),
                rrf_k=config.settings.rrf_k,
                top_k=config.settings.top_k,
            )
        return _retriever


def init_runtime() -> None:
    """
    启动时预热。失败只记警告、不阻止服务启动：
    典型场景是模型文件缺失（换机器没跑 download_model.ps1），
    这时用户仍应能打开页面，通过 /api/status 看到明确的中文原因，而不是服务直接起不来。
    """
    try:
        store = get_store()
        # 用 print 而不是 logger.info：默认日志级别是 WARNING，info 不会显示，
        # 但用户启动服务时应该一眼看到「模型/collection/块数」，方便确认索引状态。
        print(
            f"[启动] 向量化就绪：{store.embeddings.model}，collection={store.collection_name}，块数={store.count()}"
        )
    except EmbeddingError as exc:
        logger.warning("向量化初始化失败，服务继续启动（其他接口仍可用）：\n%s", exc)
    except Exception as exc:  # 磁盘/模型文件损坏等
        logger.warning("向量化初始化异常，服务继续启动：%s", exc)


def status_summary() -> dict:
    """
    给 /api/status 用的结构化状态：模型、维度、collection、块数、是否可用。
    只读计数，不触发 embedding 计算。
    """
    info = embedding_status()
    if not info.get("available"):
        info["chunks"] = None
        info["error"] = info.get("detail") or "不可用"
        return info
    try:
        store = get_store()
        info["chunks"] = store.count()
        info["collection"] = store.collection_name
        info["dimension"] = getattr(store.embeddings, "dimension", info.get("dimension"))
        info["error"] = ""
    except Exception as exc:
        info["chunks"] = None
        info["error"] = str(exc)
    return info