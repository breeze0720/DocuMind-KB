"""
app/routes_documents.py - 文档相关接口：上传 / 列表 / 删除 / 批量重索引

/api/upload 的四道防线（原实现一个都没有）：
1) 扩展名白名单；
2) 大小上限：先看 file.size 快速返回 413，流式落盘时再按累计字节数兜底（防止 size 缺失）；
3) 1MB 分块流式落盘，边写边算 SHA256——不再 await file.read() 一次性读进内存；
4) zip 炸弹检查（docx/xlsx/pptx），并且按 SHA256 做幂等，同一文件传两次不会重复入库。

旧路径 /upload、/delete 保留为兼容别名：用户原来的 static 页面还在用它们。
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import config
from app.ingest import disk_files, display_name_for_upload, index_corpus, index_file
from app.payloads import read_payload
from app.upload_utils import check_extension, check_size, sanitize_filename, stored_filename, zip_bomb_reason
from core.embeddings import EmbeddingError
from core.loader import UnsupportedFormatError
from core.runtime import get_store

router = APIRouter()


def _error(status_code: int, message: str, **extra) -> JSONResponse:
    """统一错误响应：同时给 success/message（前端约定）和 detail（FastAPI 生态习惯）。"""
    return JSONResponse(status_code=status_code, content={"success": False, "message": message, "detail": message, **extra})


@router.post("/api/upload")
@router.post("/upload", include_in_schema=False)  # 兼容旧页面
async def upload_document(file: UploadFile = File(...)):
    """上传并入库。支持 pdf/docx/md/txt/xlsx/pptx/csv。"""
    settings = config.settings
    filename = sanitize_filename(file.filename or "unnamed")
    error = check_extension(filename)
    if error:
        return _error(400, error)
    # 第一道大小检查：file.size 由 multipart 解析时给出，不用读内容就能拒绝 200MB 的包
    size_error = check_size(getattr(file, "size", None), settings.max_file_mb)
    if size_error:
        return _error(413, size_error)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f".upload_{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    total = 0
    oversize = False
    try:
        with open(tmp_path, "wb") as handle:
            while True:
                block = await file.read(1024 * 1024)  # 1MB 分块，内存占用恒定
                if not block:
                    break
                total += len(block)
                if total > settings.max_file_bytes:
                    oversize = True
                    break
                digest.update(block)
                handle.write(block)
    finally:
        await file.close()

    if oversize:
        tmp_path.unlink(missing_ok=True)
        return _error(413, f"文件过大：超过上限 {settings.max_file_mb} MB")

    ext = Path(filename).suffix.lower()
    bomb_reason = await run_in_threadpool(
        zip_bomb_reason, tmp_path, ext, settings.max_zip_ratio, settings.max_decompressed_mb
    )
    if bomb_reason:
        tmp_path.unlink(missing_ok=True)
        return _error(400, bomb_reason)

    file_hash = digest.hexdigest()
    try:
        store = await run_in_threadpool(get_store)
        existing = await run_in_threadpool(store.find_source_by_hash, file_hash)
    except EmbeddingError as exc:
        tmp_path.unlink(missing_ok=True)
        return _error(400, str(exc))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return _error(500, f"向量化初始化失败: {exc}")
    if existing:
        # 内容已存在：删掉临时文件即可，不重复解析、不重复向量化
        tmp_path.unlink(missing_ok=True)
        return {
            "success": True,
            "duplicate": True,
            "message": f"该文件内容已在知识库中（{existing.get('source')}，{existing.get('chunks')} 块），已跳过重复入库",
            "filename": filename,
            "file_hash": file_hash,
            # 兼容前端旧字段 data.file.*（进度条上传完成后读的是 data.file）
            "file": {"name": filename, "duplicate": True, "chunks": 0, "file_hash": file_hash},
            "chunks": 0,
            "total_chunks": store.count(),
        }

    final_path = upload_dir / stored_filename(file_hash, filename)
    os.replace(tmp_path, final_path)  # 同盘原子替换，避免留下半个文件
    try:
        result = await run_in_threadpool(
            index_file, final_path, file_hash=file_hash, source_name=filename
        )
    except UnsupportedFormatError as exc:
        return _error(400, str(exc), file_hash=file_hash)
    except Exception as exc:
        return _error(500, f"文档解析/入库失败: {exc}", file_hash=file_hash, stored_as=final_path.name)

    return {
        "success": True,
        "duplicate": bool(result.get("duplicate")),
        "message": f"文档「{filename}」入库成功，共 {result.get('chunks', 0)} 块",
        "filename": filename,
        "file_hash": file_hash,
        "file": {"name": filename, "duplicate": bool(result.get("duplicate")), "chunks": result.get("chunks", 0), "file_hash": file_hash},
        "chunks": result.get("chunks", 0),
        "total_chunks": result.get("total_chunks", 0),
        "collection": result.get("collection", ""),
    }


@router.get("/api/documents")
async def list_documents():
    """文档列表（按文件聚合块数）+ 磁盘文件大小/上传时间。"""
    try:
        store = await run_in_threadpool(get_store)
    except EmbeddingError as exc:
        return _error(400, str(exc))
    sources = await run_in_threadpool(store.list_sources)
    disk = await run_in_threadpool(disk_files)
    for item in sources:
        item.update(disk.get(item["source"], {"size": None, "stored_as": None, "uploaded_at": None}))
    return {
        "success": True,
        "collection": store.collection_name,
        "total_documents": len(sources),
        "total_chunks": store.count(),
        "documents": sources,
    }


async def _delete_source(source: str):
    """删除实现：向量库 + BM25 同步删除，并清掉 uploads 里的原始文件。"""
    if not source:
        return _error(400, "缺少 source 参数（要删除的文件名）")
    try:
        store = await run_in_threadpool(get_store)
        deleted = await run_in_threadpool(store.delete_source, source)
    except EmbeddingError as exc:
        return _error(400, str(exc))
    except Exception as exc:
        return _error(500, f"删除失败: {exc}")
    if deleted == 0:
        return _error(404, f"未找到来源为「{source}」的文档块")

    removed_files: list[str] = []
    target = Path(source).name
    upload_dir = Path(config.settings.upload_dir)
    if upload_dir.is_dir():
        for path in upload_dir.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.name == target or display_name_for_upload(path) == target:
                try:
                    path.unlink()
                    removed_files.append(path.name)
                except OSError:
                    pass  # 文件被占用时不影响索引删除，只是磁盘上多留一份

    return {
        "success": True,
        "message": f"已删除「{target}」，共 {deleted} 个块",
        "deleted_chunks": deleted,
        "removed_files": removed_files,
        "total_chunks": store.count(),
    }


@router.post("/api/documents/delete")
@router.post("/api/delete", include_in_schema=False)
@router.post("/delete", include_in_schema=False)  # 兼容旧页面（原来是 Form: filename）
async def delete_document(request: Request):
    """删除文档（POST 表单/JSON）。只删 uploads 下的原始文件；data/corpus 是只读语料。"""
    payload = await read_payload(request)
    source = str(payload.get("source") or payload.get("filename") or "").strip()
    return await _delete_source(source)


@router.delete("/api/documents")
async def delete_document_rest(
    request: Request,
    source: str | None = None,
    filename: str | None = None,
):
    """REST 风格删除（前端用 DELETE /api/documents?source=xxx，空间上更语义化）。"""
    payload = await read_payload(request)
    name = source or filename or str(payload.get("source") or payload.get("filename") or "")
    return await _delete_source(name.strip())


@router.post("/api/reindex")
async def reindex(request: Request):
    """
    扫描 data/corpus（可选加 uploads）批量入库。默认跳过已存在内容（按 SHA256），
    传 force=true 可强制重新解析/重新向量化（换了切块参数后需要）。
    """
    payload = await read_payload(request)
    include_uploads = str(payload.get("include_uploads", "false")).lower() in ("1", "true", "yes")
    force = str(payload.get("force", "false")).lower() in ("1", "true", "yes")
    try:
        result = await run_in_threadpool(index_corpus, include_uploads, force)
    except EmbeddingError as exc:
        return _error(400, str(exc))
    except Exception as exc:
        return _error(500, f"批量导入失败: {exc}")
    message = (
        f"扫描 {result['scanned']} 个文件：新入库 {result['indexed']} 个，"
        f"已存在 {result['duplicated']} 个，空文件 {result['empty']} 个，失败 {len(result['errors'])} 个"
    )
    return {"success": True, "message": message, **result}
