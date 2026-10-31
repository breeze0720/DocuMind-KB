"""
core/embeddings.py - 本地向量化（fastembed + bge-small-zh-v1.5，ONNX 推理）

为什么只留本地模型、不留云端后路：
1) 离线可跑：演示与评测都不依赖网络和账号额度，数字可复现；
2) 快：单条 query 向量化约 20ms，是整条检索链路 P95 185ms 的主要构成；云端要 300ms 起；
3) 够用：54 块中文短文本语料上 HitRate@5 100%、MRR 0.9017（见 eval_results/local.md）。
   云端 embedding 做过一轮对比，结论是没有换来更好的排序、延迟还差 6 倍，所以把那条分支删掉了——
   实验记录与取舍写在 docs/design/01-embeddings.md，代码里只留最终方案。

这几个工程点是踩过坑才留下的：
- 模型文件必须提前存在：本机 huggingface.co 不可达，所以**绝不联网兜底**，缺文件直接给中文解决步骤；
- 延迟 import onnxruntime：只有真要向量化时才加载，pytest 的纯函数测试不用等模型；
- 进程级单例 + 锁：加载 91MB ONNX 要 1-2 秒，每个请求重建会让首次查询明显卡顿；
- 维度从 config.json 读、不写死：模型换版本时维度可能变，写死会把索引静默写坏。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from langchain_core.embeddings import Embeddings

import config


class EmbeddingError(RuntimeError):
    """向量化配置/调用错误。上层捕获后转成清晰的中文提示，而不是把栈抛给用户。"""


def _read_dimension(model_dir: Path, default: int = 512) -> int:
    """从 HF config.json 读 hidden_size；读不到就用 bge-small-zh-v1.5 的实测值 512。"""
    try:
        data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return int(data.get("hidden_size") or default)
    except Exception:
        return default


class LocalEmbedding(Embeddings):
    """
    本地 ONNX 推理。继承 LangChain 的 Embeddings，所以能直接塞给 Chroma 当 embedding_function。

    只需要实现两个方法：embed_documents（批量，入库时用）和 embed_query（单条，查询时用）。
    """

    def __init__(self) -> None:
        settings = config.settings
        model_dir = Path(settings.local_model_dir)
        onnx_file = model_dir / "model_optimized.onnx"
        if not onnx_file.is_file():
            # 注意：这里绝不尝试联网下载（huggingface.co 在本机不可达，会白等超时）
            raise EmbeddingError(
                "本地向量模型缺失，无法启动检索。\n"
                f"  期望路径: {onnx_file}\n"
                "  解决办法: 在项目目录执行 .\\download_model.ps1，"
                "或从 ModelScope（https://modelscope.cn）下载 BAAI/bge-small-zh-v1.5 "
                "的 ONNX 文件放到该目录。"
            )
        # 延迟 import：只有用到本地后端时才加载 onnxruntime，测试/纯文字模块不会被拖慢
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=settings.local_embedding_model,
            specific_model_path=str(model_dir),
        )
        self.model = settings.local_embedding_model
        self.dimension = _read_dimension(model_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。fastembed 返回生成器，逐个 tolist 成普通 list 方便 JSON/Chroma。"""
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ============================================
# 进程级单例
# ============================================
_instance: LocalEmbedding | None = None
_lock = threading.Lock()


def get_embeddings() -> LocalEmbedding:
    """
    取向量化实例。为什么必须缓存：加载 ONNX 要 1-2 秒，
    每请求重建会让第一个查询明显卡顿；双检锁保证并发首次调用也只构造一次。
    """
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = LocalEmbedding()
        return _instance


def embedding_status() -> dict:
    """
    供 /api/status 使用：只报告「模型、维度、collection、可用性」，**不加载模型**。
    这样即使模型文件缺失，状态接口也能秒回并给出中文原因，用户知道该跑 download_model.ps1。
    """
    settings = config.settings
    model_dir = Path(settings.local_model_dir)
    available = (model_dir / "model_optimized.onnx").is_file()
    return {
        "model": settings.local_embedding_model,
        "label": "bge-small-zh-v1.5",
        "dimension": _read_dimension(model_dir),
        "collection": settings.collection_name,
        "available": available,
        "detail": "" if available else "本地模型文件缺失，请先执行 .\\download_model.ps1",
    }