"""
core/citations.py - 引用编号注册表（回答里的 [n] 与 sources 一一对应）

核心原则：**编号权在程序手里，不在模型手里**。
如果让模型自己写「[员工手册.pdf]」，它会写错文件名、编造页码，前端也没法做可点击跳转。
正确做法是检索后由程序给每个命中分配 [1][2]…，模型只被允许引用这些编号；
前端拿 sources 数组按编号渲染，答案里的 [2] 必然对应 sources[1]。

为什么注册表是「每次提问一份」而不是全局一份：
同一进程连续问两次，第二次的 [1] 必须是第二次自己的第一条来源，
否则前端会把上一轮的来源显示出来。agent.ask() 每次开始前都会 reset。
"""
from __future__ import annotations

from typing import Any


class CitationRegistry:
    """一次回答的引用注册表。register 幂等：同一块重复命中不会拿到两个编号。"""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._numbers: dict[tuple[str, Any], int] = {}

    @staticmethod
    def _key(hit: dict[str, Any]) -> tuple[str, Any]:
        # 用 (原始来源路径, chunk_index) 做唯一标识：同一篇文档的不同块编号不同；
        # 不同文档里文字相同的块也不会被误合并（来源不同 → key 不同）。
        return (str(hit.get("source_key") or hit.get("source") or ""), hit.get("chunk_index"))

    def register(self, hit: dict[str, Any]) -> int:
        """登记一个命中，返回它的 1 基编号。"""
        key = self._key(hit)
        if key in self._numbers:
            return self._numbers[key]
        number = len(self._items) + 1
        self._numbers[key] = number
        self._items.append(
            {
                "n": number,
                "source": hit.get("source") or "",
                "source_key": hit.get("source_key") or hit.get("source") or "",
                "chunk_index": hit.get("chunk_index"),
                "page": hit.get("page"),
                "sheet": hit.get("sheet"),
                "slide": hit.get("slide"),
                "title": hit.get("title") or "",
                "preview": (hit.get("content") or "")[:120].replace("\n", " "),
                # 分数一并给前端：UI 的来源卡片要显示 score；这是 RRF 融合分，只用于展示排序可信度
                "score": hit.get("score"),
                "score_kind": hit.get("score_kind", "rrf"),
            }
        )
        return number

    def sources(self) -> list[dict[str, Any]]:
        """返回 sources 快照，顺序即编号顺序，供 /api/ask 响应体和前端引用卡片使用。"""
        return [dict(item) for item in self._items]

    def reset(self) -> None:
        """每轮提问开始时清空，避免编号串轮。"""
        self._items.clear()
        self._numbers.clear()

    def __len__(self) -> int:
        return len(self._items)


def format_numbered_context(hits: list[dict[str, Any]], registry: CitationRegistry) -> str:
    """
    把检索结果格式化成给模型的上下文：每条带 [n] 和来源信息。

    注意不把 RRF 分数拼进去：模型会把 0.03 当成「相关度极低」而不敢用这条资料。
    RRF 分只是排序依据，对回答没有信息量，所以只给编号和来源。
    """
    if not hits:
        return ""
    lines = ["以下是从知识库检索到的资料（编号由系统分配，回答时只能引用这些编号）：", ""]
    for hit in hits:
        number = registry.register(hit)
        location = f"《{hit.get('source', '未知文件')}》"
        if hit.get("page"):
            location += f" 第{hit['page']}页"
        if hit.get("sheet"):
            location += f" sheet:{hit['sheet']}"
        if hit.get("slide"):
            location += f" 第{hit['slide']}张幻灯片"
        if hit.get("chunk_index") is not None:
            location += f" 第{hit['chunk_index']}块"
        lines.append(f"[{number}] 来源: {location}")
        lines.append(hit.get("content", ""))
        lines.append("")
    lines.append("引用规则：引用资料里的事实、数字、流程时，必须在句末标注编号，例如 [1]；")
    lines.append("不要编造资料里没有的编号或内容；资料不足以回答时，直接说明知识库中没有找到。")
    return "\n".join(lines)