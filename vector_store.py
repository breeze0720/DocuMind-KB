"""
vector_store.py - 向量数据库操作与混合检索
使用 Chroma 存储向量，结合 BM25 关键词检索实现混合搜索。
嵌入模型使用 fastembed（轻量级，纯 ONNX 推理，无需 sklearn/scipy）。
"""
import json
import os
from typing import List

# 设置 HuggingFace 镜像（国内访问加速，必须在 import fastembed 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ============================================
# 自定义嵌入类（基于 fastembed，兼容 LangChain 接口）
# ============================================

class FastEmbedWrapper(Embeddings):
    """
    将 fastembed 封装为 LangChain 兼容的 Embeddings 类。
    fastembed 使用 ONNX Runtime 推理，无需 PyTorch/sklearn/scipy，
    避免了这些重型库的版本冲突问题。
    """

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        print(f"[嵌入模型] 正在加载: {model_name}（首次启动会下载模型，约100MB，请稍候）")
        self._model = TextEmbedding(model_name=model_name)
        print(f"[嵌入模型] 加载完成！")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        results = []
        for embedding in self._model.embed(texts):
            results.append(embedding.tolist())
        return results

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()


# ============================================
# 全局变量
# ============================================
_vector_store = None   # Chroma 向量存储实例
_all_docs = []         # 所有文档块，用于 BM25 检索


def _get_bm25_cache_path(persist_dir):
    """获取 BM25 文档缓存文件的路径"""
    return os.path.join(persist_dir, "bm25_docs.json")


def init_store(persist_dir, embedding_model="BAAI/bge-small-zh-v1.5"):
    """
    初始化向量存储。
    如果 persist_dir 中已有数据则加载，否则创建新的。

    参数:
        persist_dir: Chroma 数据持久化目录
        embedding_model: fastembed 嵌入模型名称
    """
    global _vector_store, _all_docs

    # 创建嵌入模型（使用 fastembed，纯 ONNX 推理）
    embeddings = FastEmbedWrapper(model_name=embedding_model)

    # 判断是加载已有数据库还是创建新的
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
        chroma_files = [f for f in os.listdir(persist_dir) if not f.endswith(".json")]
        if chroma_files:
            print(f"[向量存储] 从 {persist_dir} 加载已有向量数据库")
            _vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            print(f"[向量存储] 在 {persist_dir} 创建新的向量数据库")
            os.makedirs(persist_dir, exist_ok=True)
            _vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
    else:
        print(f"[向量存储] 在 {persist_dir} 创建新的向量数据库")
        os.makedirs(persist_dir, exist_ok=True)
        _vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    # 恢复 BM25 文档缓存
    bm25_cache = _get_bm25_cache_path(persist_dir)
    if os.path.exists(bm25_cache):
        try:
            with open(bm25_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
            _all_docs = [
                Document(page_content=item["content"], metadata=item.get("metadata", {}))
                for item in data
            ]
            print(f"[向量存储] 恢复了 {len(_all_docs)} 个文档块用于 BM25 检索")
        except Exception as e:
            print(f"[向量存储] BM25 缓存恢复失败: {e}")
            _all_docs = []

    return _vector_store


def add_documents(documents, persist_dir):
    """
    向向量存储中添加新文档块。

    参数:
        documents: LangChain Document 对象列表
        persist_dir: 持久化目录（用于保存 BM25 缓存）
    """
    global _vector_store, _all_docs

    if _vector_store is None:
        raise RuntimeError("向量存储未初始化！请先调用 init_store()")

    # 添加到 Chroma
    _vector_store.add_documents(documents)
    print(f"[向量存储] 已将 {len(documents)} 个文档块添加到 Chroma")

    # 添加到 BM25 文档列表
    _all_docs.extend(documents)

    # 保存 BM25 文档缓存到 JSON 文件
    bm25_cache = _get_bm25_cache_path(persist_dir)
    try:
        cache_data = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in _all_docs
        ]
        with open(bm25_cache, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[向量存储] BM25 缓存已保存: {bm25_cache}")
    except Exception as e:
        print(f"[向量存储] BM25 缓存保存失败: {e}")


def hybrid_search(query, k=5):
    """
    混合检索：向量相似度 + BM25 关键词匹配。
    使用 RRF（Reciprocal Rank Fusion）算法融合两种检索结果。

    参数:
        query: 查询字符串
        k: 返回结果数量

    返回:
        字典列表，每项包含 content（文本内容）和 score（融合分数）
    """
    global _vector_store, _all_docs

    if _vector_store is None:
        print("[向量存储] 向量存储未初始化，返回空结果")
        return []

    # ---------- 第一步：向量检索 ----------
    vector_results = _vector_store.similarity_search(query, k=k)

    # ---------- 第二步：BM25 关键词检索 ----------
    bm25_results = []
    if _all_docs:
        try:
            bm25_retriever = BM25Retriever.from_documents(_all_docs)
            bm25_retriever.k = k
            bm25_results = bm25_retriever.invoke(query)
        except Exception as e:
            print(f"[向量存储] BM25 检索出错: {e}")

    # ---------- 第三步：RRF 融合 ----------
    # 原理：每种检索方法的结果按排名给分（1/(60+rank)），两种方法的分数相加
    rrf_scores = {}

    for rank, doc in enumerate(vector_results):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank)

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank)

    # 按 RRF 分数降序排序
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ---------- 第四步：取 Top-K（去重）----------
    results = []
    seen = set()
    for content, score in sorted_items:
        if content not in seen:
            seen.add(content)
            results.append({
                "content": content,
                "score": round(score, 4)
            })
        if len(results) >= k:
            break

    return results


def delete_documents_by_source(source_path, persist_dir):
    """
    根据源文件路径删除向量数据库中的文档块。

    参数:
        source_path: 源文件的完整路径
        persist_dir: 持久化目录

    返回:
        删除的文档块数量
    """
    global _vector_store, _all_docs

    if _vector_store is None:
        raise RuntimeError("向量存储未初始化！请先调用 init_store()")

    deleted_count = 0

    # 从 ChromaDB 中删除（通过 metadata 过滤）
    try:
        _vector_store.delete(where={"source": source_path})
        print(f"[向量存储] 已从 Chroma 删除 source={source_path} 的文档块")
    except Exception as e:
        print(f"[向量存储] Chroma 删除失败: {e}")
        # 如果 Chroma 删除失败，也继续尝试清理 BM25 缓存

    # 从 BM25 文档列表中移除
    new_docs = []
    for doc in _all_docs:
        if doc.metadata.get("source") == source_path:
            deleted_count += 1
        else:
            new_docs.append(doc)
    _all_docs = new_docs

    # 保存更新后的 BM25 缓存
    bm25_cache = os.path.join(persist_dir, "bm25_docs.json")
    try:
        cache_data = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in _all_docs
        ]
        with open(bm25_cache, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[向量存储] BM25 缓存已更新: {bm25_cache}")
    except Exception as e:
        print(f"[向量存储] BM25 缓存更新失败: {e}")

    print(f"[向量存储] 共删除 {deleted_count} 个文档块")
    return deleted_count


def get_document_count():
    """获取知识库中文档块的总数"""
    return len(_all_docs)
