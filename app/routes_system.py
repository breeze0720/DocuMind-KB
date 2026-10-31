"""
app/routes_system.py - 状态与评测结果接口

/api/status 是联调时最先被调用的接口：前端靠它显示「用的哪个 embedding 模型、
多少维、索引里有多少块、有哪些文档」。因此这里刻意不触发任何模型加载：
模型文件缺失时也要能返回，只是把 available 置 false 并说明原因。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

import config
from app.ingest import display_name_for_upload
from core.loader import ALLOWED_EXTENSIONS
from core.runtime import get_store, status_summary

router = APIRouter()
_STARTED_AT = time.time()


def _error(status_code: int, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"success": False, "message": message})


def _safe_list_sources() -> list[dict]:
    """取文档列表；模型/索引不可用时返回空列表，不让状态接口 500。"""
    try:
        return get_store().list_sources()
    except Exception:
        return []


def _corpus_files() -> list[str]:
    corpus_dir = Path(config.settings.data_dir) / "corpus"
    if not corpus_dir.is_dir():
        return []
    return sorted(
        path.name for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS and not path.name.startswith(".")
    )


def _upload_files() -> list[dict]:
    upload_dir = Path(config.settings.upload_dir)
    if not upload_dir.is_dir():
        return []
    files = []
    for path in sorted(upload_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        stat = path.stat()
        files.append({"source": display_name_for_upload(path), "stored_as": path.name, "size": stat.st_size})
    return files


def _size_map() -> dict[str, int]:
    """文件名 → 字节数（corpus + uploads）。前端文档列表显示大小用。"""
    sizes: dict[str, int] = {}
    corpus_dir = Path(config.settings.data_dir) / "corpus"
    upload_dir = Path(config.settings.upload_dir)
    for directory in (corpus_dir, upload_dir):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*") if directory == corpus_dir else directory.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            name = display_name_for_upload(path) if directory == upload_dir else path.name
            try:
                sizes.setdefault(name, path.stat().st_size)
            except OSError:
                continue
    return sizes


@router.get("/api/status")
@router.get("/status", include_in_schema=False)
async def status():
    """知识库/模型/目录的整体状态。"""
    summary = await run_in_threadpool(status_summary)
    documents = await run_in_threadpool(_safe_list_sources)
    sizes = await run_in_threadpool(_size_map)
    for item in documents:
        item["size"] = sizes.get(item.get("source"))
    corpus = await run_in_threadpool(_corpus_files)
    uploads = await run_in_threadpool(_upload_files)
    latest_eval = Path(config.settings.eval_dir) / "latest.json"
    chunk_count = summary.get("chunks") or 0
    dimension = summary.get("dimension")
    return {
        "success": True,
        "version": "2.0.0",
        "model_name": config.settings.model_name,
        "embedding": {
            "model": summary.get("model", ""),
            "label": summary.get("label", ""),
            "dimension": dimension,
            "dim": dimension,  # 兼容前端字段 emb.dim
            "collection": summary.get("collection", ""),
            "available": bool(summary.get("available")),
            "detail": summary.get("error", ""),
        },
        "chunk_count": chunk_count,
        "chunks": chunk_count,  # 兼容前端字段 s.chunks
        "documents": documents,
        "uploads": {"count": len(uploads), "files": uploads},
        "corpus": {"dir": str(Path(config.settings.data_dir) / "corpus"), "count": len(corpus), "files": corpus},
        "dirs": {
            "data": str(config.settings.data_dir),
            "chroma": str(config.settings.chroma_dir),
            "uploads": str(config.settings.upload_dir),
            "eval": str(config.settings.eval_dir),
        },
        "config": {
            "max_file_mb": config.settings.max_file_mb,
            "top_k": config.settings.top_k,
            "rrf_k": config.settings.rrf_k,
            "chunk_tokens": config.settings.chunk_tokens,
            "chunk_overlap": config.settings.chunk_overlap,
        },
        "email_configured": bool(config.settings.email_from and config.settings.email_from != "your_email@qq.com"),
        "eval_available": latest_eval.is_file(),
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
    }


@router.get("/api/eval/summary")
async def eval_summary():
    """
    读 eval_results/latest.json 给前端渲染评测卡片。
    文件不存在时返回 200 + available=false，前端可以直接显示「还没有评测数据」，
    不用把 404 当成错误处理（联调阶段少一类异常分支）。
    """
    path = Path(config.settings.eval_dir) / "latest.json"
    if not path.is_file():
        return {
            "success": False,
            "available": False,
            "message": "还没有评测结果：先启动服务，再运行 python eval/eval.py --tag before",
            "path": str(path),
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "available": False, "message": f"评测结果读取失败: {exc}", "path": str(path)}
    # 把嵌套的 metrics 摊平一份：前端直接读 e.hit_rate / e.mrr / e.p95_ms / e.n，
    # 同时保留原始结构（metrics、cases），脚本侧仍可拿到完整数据
    metrics = data.get("metrics") or {}
    return {
        **data,
        "success": True,
        "available": True,
        "path": str(path),
        "hit_rate": metrics.get("hit_rate"),
        "mrr": metrics.get("mrr"),
        "kw_hit_rate": metrics.get("kw_hit_rate"),
        "kw_mrr": metrics.get("kw_mrr"),
        "by_type": data.get("by_type") or {},
        "p50_ms": metrics.get("p50_latency_ms"),
        "p95_ms": metrics.get("p95_latency_ms"),
        "n": metrics.get("total"),
        "k": data.get("k"),
        "tag": data.get("tag"),
        "updated_at": data.get("generated_at"),
    }


@router.get("/api/health")
async def health():
    return {"success": True, "status": "ok", "uptime_seconds": round(time.time() - _STARTED_AT, 1)}