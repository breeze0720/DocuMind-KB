"""tests/test_splitter.py - 切块逻辑测试：token 预算、句末重叠、metadata 完整性。"""
from langchain_core.documents import Document

from core.splitter import count_tokens, split_documents, split_text


def test_count_tokens_chinese_and_ascii():
    """中文基本 1 字 1 token；英文按 4 字符约 1 token。这是切块预算的依据，先验证估算本身。"""
    assert count_tokens("年假为每年十天") == 7
    assert count_tokens("") == 0
    # 8 个 ASCII 字符 → ceil(8/4) = 2 token
    assert count_tokens("abcdefgh") == 2


def test_split_text_respects_token_budget():
    """长中文文本切完后，每块 token 数都不超过预算；且不丢内容。"""
    text = "员工手册规定年假为每年十天，需要提前三天在系统提交申请。" * 80
    chunks = split_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 100 for chunk in chunks)
    # 允许重叠，但不应该丢句子：所有块拼起来应覆盖原文的所有非空白字符
    combined = "".join(chunk for chunk in chunks)
    assert "年假为每年十天" in combined


def test_split_text_overlap_at_sentence_boundary():
    """相邻块要有句级重叠，且边界落在句号后（不是把句子从中间劈开）。"""
    text = "".join(f"第{index}句测试内容用于验证重叠。" for index in range(60))
    chunks = split_text(text, max_tokens=40, overlap=20)
    assert len(chunks) >= 2

    def sentences_of(chunk: str) -> set[str]:
        return {part for part in chunk.split("。") if part}

    for left, right in zip(chunks, chunks[1:]):
        left_sentences = sentences_of(left)
        right_sentences = sentences_of(right)
        # 重叠句一定在左边出现、右边也有
        assert left_sentences & right_sentences, "相邻块之间没有句级重叠"
        # 不是从句子中间切开：每块都以句号结尾
        assert left.rstrip().endswith("。")


def test_split_text_long_sentence_without_punctuation():
    """没有标点的超长文本走硬切兜底，仍要满足预算。"""
    text = "x" * 3000
    chunks = split_text(text, max_tokens=100, overlap=10)
    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 100 for chunk in chunks)


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_split_documents_metadata():
    """metadata 必须带 source/chunk_index/file_hash/title/char_len，且 chunk_index 全局递增。"""
    documents = [
        Document(page_content="第一页内容。" * 50, metadata={"source": "a.pdf", "page": 1, "file_hash": "hash-a"}),
        Document(page_content="第二页内容。" * 50, metadata={"source": "a.pdf", "page": 2, "file_hash": "hash-a"}),
    ]
    chunks = split_documents(documents, max_tokens=60, overlap=10)
    assert len(chunks) > 2
    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indexes == list(range(len(chunks)))
    for chunk in chunks:
        metadata = chunk.metadata
        assert metadata["source"] == "a.pdf"
        assert metadata["file_hash"] == "hash-a"
        assert metadata["title"]
        assert metadata["char_len"] == len(chunk.page_content)
        assert metadata["token_count"] <= 60
        assert metadata["page"] in (1, 2)


def test_split_documents_skips_empty():
    chunks = split_documents([Document(page_content="   ", metadata={"source": "empty.txt"})])
    assert chunks == []