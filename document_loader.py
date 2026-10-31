"""
document_loader.py - 兼容 shim（旧模块名）

解析（pdf/docx/md/txt/xlsx/pptx/csv）已移到 core/loader.py，按 token 切块在 core/splitter.py。
这里保留旧的 load_and_split 入口，让以前 `from document_loader import load_and_split` 的脚本
继续可用；参数名 chunk_size 为了兼容旧代码保留，语义现在已经是「token 数」。
"""
from langchain_core.documents import Document

from core.loader import ALLOWED_EXTENSIONS, UnsupportedFormatError, load_documents  # noqa: F401
from core.splitter import split_documents


def load_and_split(path, chunk_size: int | None = None, overlap: int | None = None) -> list[Document]:
    """旧接口：解析 + 切块。chunk_size 沿用旧参数名，现在表示 token 预算（默认 500）。"""
    return split_documents(load_documents(path), max_tokens=chunk_size, overlap=overlap)


__all__ = ["ALLOWED_EXTENSIONS", "UnsupportedFormatError", "load_and_split", "load_documents", "split_documents"]