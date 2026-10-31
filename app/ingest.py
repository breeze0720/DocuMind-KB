"""
app/ingest.py - 文档入库服务（解析 → 切块 → 写库）

为什么单独一层，而不是把逻辑写在 /api/upload 里：
「上传」和「导入 data/corpus」是两条入口，但入库动作完全一样（算哈希 → 判重 → 解析 → 切块 → 写库）。
抽成一个函数，两条入口共用，行为不会分叉；也方便只读语料目录做批量重索引。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import config
from core.loader import ALLOWED_EXTENSIONS, UnsupportedFormatError, file_sha256, load_documents
from core.runtime import get_store
from core.splitter import split_documents

logger = logging.getLogger(__name__)


def index_file(
    path: str | Path,
    file_hash: str | None = None,
    force: bool = False,
    source_name: str | None = None,
) -> dict[str, Any]:
    """
    把单个文件入库。

    参数：
        file_hash  上传流程已算过就传进来，避免大文件读两遍
        force      强制重新入库（同一文件重新解析/重新向量化）
        source_name 写进 metadata 的展示文件名（上传时是原始文件名，去掉磁盘上的哈希前缀）
    返回结构化结果，供 API 直接组装响应。
    """
    file_path = Path(path)
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(f"不支持的文件格式：{file_path.suffix}")
    store = get_store()
    digest = file_hash or file_sha256(file_path)
    display_name = source_name or file_path.name

    if not force:
        existing = store.find_source_by_hash(digest)
        if existing:
            # 幂等：同一内容不重复解析、不重复向量化
            return {
                "indexed": False,
                "duplicate": True,
                "source": display_name,
                "file_hash": digest,
                "chunks": 0,
                "existing": existing,
                "total_chunks": store.count(),
            }

    documents = load_documents(file_path, file_hash=digest)
    for document in documents:
        # loader 用的是磁盘文件名（可能带哈希前缀），这里统一改写成用户看到的原始名
        document.metadata["source"] = display_name
    chunks = split_documents(documents)
    if not chunks:
        return {
            "indexed": False,
            "empty": True,
            "source": display_name,
            "file_hash": digest,
            "chunks": 0,
            "total_chunks": store.count(),
        }

    added = store.add_documents(chunks)
    logger.info("入库 %s：%d 块", display_name, added)
    return {
        "indexed": True,
        "duplicate": False,
        "source": display_name,
        "title": chunks[0].metadata.get("title", ""),
        "file_hash": digest,
        "chunks": added,
        "collection": store.collection_name,
        "total_chunks": store.count(),
    }


def display_name_for_upload(path: Path) -> str:
    """上传目录里的文件名为 {sha8}_{原始名}，展示/引用时去掉哈希前缀。"""
    prefix, sep, rest = path.name.partition("_")
    if sep and len(prefix) == 8 and all(char in "0123456789abcdef" for char in prefix.lower()):
        return rest or path.name
    return path.name


def disk_files() -> dict[str, dict[str, Any]]:
    """扫描 uploads 与 data/corpus，建立「展示名 → 磁盘信息」索引（文档列表显示大小/来源）。"""
    result: dict[str, dict[str, Any]] = {}
    settings = config.settings
    sources = (
        ("uploads", Path(settings.upload_dir), False),
        ("corpus", Path(settings.data_dir) / "corpus", True),
    )
    for origin, directory, recursive in sources:
        if not directory.is_dir():
            continue
        paths = directory.rglob("*") if recursive else directory.iterdir()
        for path in paths:
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            display = display_name_for_upload(path) if origin == "uploads" else path.name
            # 同名文件以 uploads 优先（用户刚传的文件通常更关心）
            result.setdefault(
                display,
                {"size": stat.st_size, "stored_as": path.name, "uploaded_at": stat.st_mtime, "origin": origin},
            )
    return result


def index_corpus(include_uploads: bool = False, force: bool = False) -> dict[str, Any]:
    """
    扫描 data/corpus（可选加 uploads）批量入库。

    为什么默认关闭启动时自动扫描：语料大时启动会卡住，前端连不上会以为服务挂了。
    默认把导入做成显式动作（/api/reindex），需要开机即用时把 AUTO_INDEX_CORPUS 设为 true。
    """
    settings = config.settings
    corpus_dir = Path(settings.data_dir) / "corpus"
    paths: list[Path] = []
    if corpus_dir.is_dir():
        paths.extend(
            sorted(
                path for path in corpus_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS and not path.name.startswith(".")
            )
        )
    if include_uploads and Path(settings.upload_dir).is_dir():
        paths.extend(
            sorted(
                path for path in Path(settings.upload_dir).iterdir()
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS and not path.name.startswith(".")
            )
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            source_name = display_name_for_upload(path) if path.parent == Path(settings.upload_dir) else path.name
            results.append(index_file(path, force=force, source_name=source_name))
        except Exception as exc:  # 单个文件失败不能中断整批导入
            logger.warning("导入 %s 失败: %s", path.name, exc)
            errors.append({"source": path.name, "error": str(exc)})

    store = get_store()
    return {
        "collection": store.collection_name,
        "corpus_dir": str(corpus_dir),
        "include_uploads": include_uploads,
        "scanned": len(paths),
        "indexed": sum(1 for item in results if item.get("indexed")),
        "duplicated": sum(1 for item in results if item.get("duplicate")),
        "empty": sum(1 for item in results if item.get("empty")),
        "errors": errors,
        "files": results,
        "total_chunks": store.count(),
    }