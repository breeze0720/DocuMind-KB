"""
app/routes_qa.py - 检索与问答接口（/api/search、/api/ask、/api/ask/stream、/api/session/reset）

设计要点：
- /api/search 只做检索、不经过大模型：既是面试时的实时演示入口，也是 eval.py 的评测入口。
- /api/ask 返回 (answer, sources)：sources 的编号与回答里的 [n] 严格对应。
- /api/ask/stream 用 SSE：先发 sources 事件，再逐 token 发，最后 done；
  用 sse_starlette 的 EventSourceResponse，它自带心跳和断线处理。
- 旧路径 /search、/ask、/clear 保留为兼容别名（旧 static 页面仍在用）。
- 参数解析走 read_payload：JSON / form / query 三种调用方式都支持，
  eval.py（JSON）和 curl（form）不用互相迁就。
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

import config
from app.payloads import as_float, as_int, read_payload
from core.agent import ask as agent_ask
from core.agent import ask_stream as agent_ask_stream
from core.agent import clear_history
from core.embeddings import EmbeddingError
from core.runtime import get_retriever

router = APIRouter()


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "message": message, "detail": message})


@router.post("/api/search")
@router.get("/api/search")
@router.post("/search", include_in_schema=False)  # 兼容旧页面
@router.get("/search", include_in_schema=False)
async def search(request: Request):
    """
    混合检索（向量 + BM25 + RRF），不调用大模型。

    响应里回带 embedding 模型名与 collection 名：联调时能一眼确认「这次查的是哪套索引」。
    """
    started = time.perf_counter()
    payload = await read_payload(request)
    query = str(payload.get("query") or "").strip()
    if not query:
        return _error(400, "缺少 query 参数（要检索的问题）")
    k = min(50, max(1, as_int(payload.get("k"), config.settings.top_k)))
    min_score = as_float(payload.get("min_score"))

    try:
        retriever = await run_in_threadpool(get_retriever)
        hits = await run_in_threadpool(retriever.search, query, k, min_score)
    except EmbeddingError as exc:
        return _error(400, str(exc))
    except Exception as exc:
        return _error(500, f"检索失败: {exc}")

    results = []
    for hit in hits:
        content = hit.get("content") or ""
        results.append(
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "score_kind": hit["score_kind"],  # 前端据此提示「RRF 排名分，不是相似度」
                "source": hit["source"],
                "source_key": hit["source_key"],
                "chunk_index": hit["chunk_index"],
                "page": hit.get("page"),
                "title": hit.get("title", ""),
                "preview": content[:120].replace("\n", " "),
                "content": content,  # eval.py 做关键词/证据匹配需要全文
            }
        )
    return {
        "success": True,
        "query": query,
        "k": k,
        "min_score": min_score,
        "embedding": retriever.store.embeddings.model,
        "collection": retriever.store.collection_name,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "results": results,
    }


@router.post("/api/ask")
@router.post("/ask", include_in_schema=False)
async def ask_question(request: Request):
    """Agent 问答（非流式）。返回回答 + 来源列表，sources 的 n 与回答里的 [n] 对应。"""
    started = time.perf_counter()
    payload = await read_payload(request)
    question = str(payload.get("question") or payload.get("query") or "").strip()
    if not question:
        return _error(400, "缺少 question 参数")
    try:
        answer, sources = await run_in_threadpool(agent_ask, question)
    except Exception as exc:
        return _error(500, f"问答失败: {exc}")
    return {
        "success": True,
        "question": question,
        "answer": answer,
        "sources": sources,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@router.get("/api/ask/stream")
@router.post("/api/ask/stream")
async def ask_stream(request: Request):
    """
    SSE 流式问答。事件序列：sources → token(多次) → done（出错时 error）。

    为什么 GET 和 POST 都支持：EventSource 只能 GET（参数走 query），
    而用 fetch 读流的调用方习惯 POST JSON。两种都留着，前端不用为兼容改两套代码。
    """
    payload = await read_payload(request)
    question = str(payload.get("question") or payload.get("query") or "").strip()
    if not question:
        return _error(400, "缺少 question 参数")

    async def event_generator():
        started = time.perf_counter()
        try:
            async for event, data in iterate_in_threadpool(agent_ask_stream(question)):
                if event == "done":
                    data = {**data, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
                # 统一 JSON 编码：event 名区分类型，data 是 JSON 字符串（token 事件格式 {"text": "..."}）
                yield {"event": event, "data": json.dumps(data, ensure_ascii=False)}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)}, ensure_ascii=False)}

    return EventSourceResponse(event_generator(), ping=15000)


@router.post("/api/session/reset")
@router.post("/clear", include_in_schema=False)
async def reset_session():
    """重置会话：清空对话历史（引用编号注册表也会随每轮 ask 自动清空）。"""
    await run_in_threadpool(clear_history)
    return {"success": True, "message": "会话已重置：对话历史已清空"}