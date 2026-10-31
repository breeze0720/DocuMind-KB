"""
core/splitter.py - 按 token 预算切块（尽量在句末断开）

为什么按 token 而不是按字符：
本地 bge-small-zh-v1.5 的 config.json 里 max_position_embeddings=512，超过 512 token 的输入
会被 tokenizer 静默截断——多出来的内容不进向量，检索时永远搜不到，且不会报错。
原实现按 500「字符」切，中英混排时 500 字符可能是 300~700 token，超长的部分就这样丢了。
所以这里改成按 token 预算（默认 500 + overlap 50）。

token 怎么数（估算依据）：
- 不引入 tiktoken（它是 GPT 的 BPE，和 bge 的 WordPiece 词表不是一回事，数出来不代表真实占用）。
- 中文：bge 用 bert-base-chinese 词表，中文基本按「单字」切，所以 1 个汉字 ≈ 1 token。
- 英文/数字：WordPiece 会把长单词拆子词，经验上平均 ~4 字符/token，这里按 ceil(len/4) 保守估算。
- 标点/符号：各算 1 token。
这是近似值，但在中文语料上误差很小；真正要「精确不超限」时，换成模型自带 tokenizer 即可，
接口只需要改 count_tokens 一个函数。
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

import config

# CJK 统一表意文字（含扩展 A 与兼容区）：这些字符在 bert 中文词表里通常是单字 token
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# 句子边界：中文标点 + 换行。英文句点故意不切，避免把 "3.5%" / "e.g." / 缩写切断；
# 英文内容靠换行和 token 预算兜底。
_SENTENCE_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")


def count_tokens(text: str) -> int:
    """估算 token 数（中文 1 字≈1 token，英文 ceil(字符/4)），依据见模块头注释。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = _CJK_RE.sub(" ", text)
    ascii_tokens = sum(-(-len(word) // 4) for word in _ASCII_WORD_RE.findall(rest))
    # 其余非空白字符（标点、emoji、公式符号）按 1 个 token 计
    other = sum(1 for ch in rest if not ch.isspace() and not ch.isalnum() and ch and not _CJK_RE.match(ch))
    return cjk + ascii_tokens + other


def _normalize(text: str) -> str:
    """统一换行、压掉连续空行、去掉首尾空白，避免切出「纯空白块」。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """粗粒度分句：保留句末标点，长句不会被拆散（拆散会破坏语义完整性）。"""
    sentences: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        found = [m.group(0).strip() for m in _SENTENCE_RE.finditer(stripped) if m.group(0).strip()]
        sentences.extend(found or [stripped])
    return sentences


def _chars_for_budget(text: str, max_tokens: int) -> int:
    """二分找「最多保留多少字符仍不超 token 预算」，用于单词过长时硬切。"""
    if count_tokens(text) <= max_tokens:
        return len(text)
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo)


def _hard_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    兜底：没有标点的超长文本（表格行、URL、代码）按字符预算硬切。
    硬切会切断语义，所以只在单句超预算时使用，并保留 overlap 字符让相邻块仍有上下文。
    """
    budget = _chars_for_budget(text, max_tokens)
    overlap_chars = min(max(1, int(overlap * budget / max_tokens)), budget // 2)
    step = max(1, budget - overlap_chars)
    pieces: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + budget].strip()
        if not piece:
            continue
        # 极端混排下前缀预算可能估计偏大，最后再校正一次
        if count_tokens(piece) > max_tokens:
            piece = piece[:_chars_for_budget(piece, max_tokens)].strip()
        if piece:
            pieces.append(piece)
        if start + budget >= len(text):
            break
    return pieces


def split_text(text: str, max_tokens: int | None = None, overlap: int | None = None) -> list[str]:
    """
    把一段文本切成若干块，每块 ≤ max_tokens，相邻块在句末重叠约 overlap tokens。

    实现是「句子级贪心装箱」：句子是切分的最小单位，只有单句本身超预算时才硬切。
    这样 normal 文本的块边界永远落在句号/问号/换行处，可读性和引用价值都更好。
    """
    max_tokens = max_tokens or config.settings.chunk_tokens
    overlap = overlap if overlap is not None else config.settings.chunk_overlap
    max_tokens = max(1, int(max_tokens))
    overlap = max(0, int(overlap))
    if overlap >= max_tokens:
        overlap = max_tokens // 2  # overlap 必须小于块预算，否则无法前进

    text = _normalize(text)
    if not text:
        return []

    pairs = [(sent, count_tokens(sent)) for sent in _split_sentences(text)]
    chunks: list[str] = []
    buf: list[tuple[str, int]] = []
    buf_tokens = 0

    def tail_sentences(sentences: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """取尾部若干句作为下一块的开头，总 token 不超过 overlap（句子级重叠）。"""
        tail: list[tuple[str, int]] = []
        total = 0
        for sent, tokens in reversed(sentences):
            if total + tokens > overlap:
                break
            tail.insert(0, (sent, tokens))
            total += tokens
        return tail

    def join_buffer() -> str:
        return "".join(sent for sent, _ in buf).strip()

    for sent, tokens in pairs:
        if tokens > max_tokens:
            # 单句超预算：先把缓冲区吐出去，再硬切这句；硬切后不保留句级 overlap（没有句子边界可用）
            if buf:
                chunks.append(join_buffer())
                buf, buf_tokens = [], 0
            chunks.extend(_hard_split(sent, max_tokens, overlap))
            continue

        if buf and buf_tokens + tokens > max_tokens:
            chunks.append(join_buffer())
            buf = tail_sentences(buf)  # 尾部句子带到下一块开头，实现句子级 overlap
            buf_tokens = sum(token for _, token in buf)

        buf.append((sent, tokens))
        buf_tokens += tokens

    if buf:
        chunks.append(join_buffer())
    return [chunk for chunk in chunks if chunk]


def _infer_title(text: str, source: str = "") -> str:
    """从正文首行猜标题：去掉 Markdown 的 # 和空白，最多 80 字。"""
    for line in (text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return Path(source).stem if source else ""


def split_documents(
    documents: list[Document],
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[Document]:
    """
    对 loader 产出的「页面/sheet 单元」逐一切块，并补齐检索/引用需要的 metadata。

    chunk_index 是整篇文档内全局递增的：引用溯源时要能说「第 12 块」，
    如果每页从 0 开始，同一篇文档里就会出现多个第 0 块，无法唯一定位。
    page/section/sheet 等来源信息从 loader 传下来，切块时原样保留。
    """
    max_tokens = max_tokens or config.settings.chunk_tokens
    overlap = overlap if overlap is not None else config.settings.chunk_overlap
    result: list[Document] = []
    chunk_index = 0

    for doc in documents:
        text = doc.page_content or ""
        metadata = dict(doc.metadata or {})
        title = metadata.get("title") or _infer_title(text, metadata.get("source", ""))
        for piece in split_text(text, max_tokens=max_tokens, overlap=overlap):
            item = dict(metadata)
            item.update(
                {
                    "source": metadata.get("source", ""),
                    "file_hash": metadata.get("file_hash", ""),
                    "title": title,
                    "chunk_index": chunk_index,
                    "char_len": len(piece),
                    "token_count": count_tokens(piece),
                }
            )
            result.append(Document(page_content=piece, metadata=item))
            chunk_index += 1

    return result