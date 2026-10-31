"""
core/bm25.py - BM25 关键词索引的分词与磁盘缓存（store 的辅助模块）

拆出来的理由：分词策略和缓存格式是「BM25 自己的事」，和 Chroma 读写没关系。
独立成文件后 store 只需要关心「两套索引同增同删」，也方便单独测分词/缓存逻辑。

两个效果/性能坑（原实现都有）：
1) 原来每次查询都 BM25Retriever.from_documents(...) 重建索引，1000 块时每次多 50-200ms。
   现在索引是进程级缓存的，只在 add/delete 后重建，并持久化到 JSON，重启不用重算。
2) 原来 BM25Retriever 默认按空格分词，中文整段被当成一个 token，等于没做关键词检索。
   现在用 jieba.lcut_for_search 分词（搜索引擎模式会把长词再切出子词，召回更高）。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def bm25_tokenize(text: str) -> list[str]:
    """
    BM25 分词：jieba 搜索引擎模式。
    为什么用 lcut_for_search 而不是 lcut：它会把「生鲜识别平台」再切出「生鲜」「识别」「平台」，
    用户搜短词时也能命中长标题；代价是索引稍微大一点，个人知识库规模完全可接受。
    """
    return [token for token in jieba.lcut_for_search(text or "") if token.strip()]


def build_bm25(docs: list[dict[str, Any]]) -> tuple[BM25Okapi | None, list[list[str]]]:
    """按当前文档集合重建 BM25 索引，返回 (索引, 与 docs 一一对应的分词结果)。"""
    tokenized = [bm25_tokenize(item["content"]) for item in docs]
    return (BM25Okapi(tokenized) if tokenized else None), tokenized


def load_cached_docs(path: Path) -> list[dict[str, Any]]:
    """读 BM25 JSON 缓存；文件缺失或损坏时返回空列表（调用方会从 Chroma 全量重建）。"""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            {"id": item["id"], "content": item["content"], "metadata": item.get("metadata") or {}}
            for item in raw
        ]
    except Exception as exc:
        logger.warning("BM25 缓存读取失败，将从 Chroma 重建: %s", exc)
        return []


def save_cached_docs(path: Path, docs: list[dict[str, Any]]) -> None:
    """落盘 BM25 文档缓存。写入失败只告警：缓存坏了下次会从 Chroma 重建，不影响本次查询。"""
    try:
        payload = [{"id": item["id"], "content": item["content"], "metadata": item["metadata"]} for item in docs]
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)  # 原子替换，避免进程中途挂掉留下半个 JSON
    except Exception as exc:
        logger.warning("BM25 缓存写入失败（不影响本次查询）: %s", exc)