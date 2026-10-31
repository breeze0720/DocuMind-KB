"""
vector_store.py - 兼容 shim（旧模块名）

原文件把 embedding / Chroma / BM25 / RRF 融合 / 删除同步全塞在一起（300+ 行），
现在拆成 core/embeddings.py、core/store.py、core/retriever.py、core/runtime.py。
这里只保留旧函数名做转发，避免用户已有的脚本 import 报错。
新代码请直接用 core.runtime.get_store() / get_retriever()。
"""
from pathlib import Path

from core.runtime import get_retriever, get_store


def init_store(persist_dir=None, embedding_model=None):
    """旧接口：初始化向量库。persist_dir/embedding_model 参数已由 config 统一管理，保留只为兼容。"""
    return get_store()


def add_documents(documents, persist_dir=None):
    """旧接口：写入文档块（Chroma 与 BM25 会在 store 内同步更新）。"""
    return get_store().add_documents(documents)


def get_document_count():
    """旧接口：知识库块数。后端不可用时返回 0，而不是让调用方炸掉。"""
    try:
        return get_store().count()
    except Exception:
        return 0


def delete_documents_by_source(source_path, persist_dir=None):
    """旧接口：按来源删除。旧代码传的是文件绝对路径，这里取文件名交给新 store。"""
    return get_store().delete_source(Path(str(source_path)).name)


def hybrid_search(query, k=5):
    """旧接口：混合检索。返回结构与 core.retriever.HybridRetriever.search 一致。"""
    return get_retriever().search(query, k=k)