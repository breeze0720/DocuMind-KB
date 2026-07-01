"""
document_loader.py - 文档加载与分块模块
支持 PDF、Word、Markdown、TXT 格式的文档，加载后切分成小块。
不依赖 langchain_community / langchain_text_splitters，避免 sklearn/scipy 冲突。
"""
import os
import re
from langchain_core.documents import Document


def _load_pdf(file_path):
    """使用 pypdf 加载 PDF 文件，返回完整文本"""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    print(f"[文档加载] PDF 共 {len(reader.pages)} 页，提取到 {len(pages)} 页有内容的页面")
    return "\n\n".join(pages)


def _load_docx(file_path):
    """使用 docx2txt 加载 Word 文件，返回完整文本"""
    import docx2txt

    text = docx2txt.process(file_path)
    return text if text else ""


def _load_text(file_path):
    """加载纯文本文件（Markdown / TXT）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _simple_split(text, chunk_size=500, overlap=50, source_path=""):
    """
    简单的文本分块器。
    优先按段落分隔，段落太长则按句子分隔，句子还太长则按字符硬切。

    参数:
        text: 要切分的文本
        chunk_size: 每块最大字符数
        overlap: 相邻块之间的重叠字符数
        source_path: 源文件路径，用于标识文档来源

    返回:
        文本块列表，每项为 {"content": str, "metadata": dict}
    """
    chunks = []

    # 第一步：按段落切分
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        if len(para) <= chunk_size:
            # 段落足够短，直接作为一个块
            chunks.append(para)
        else:
            # 段落太长，按句子切分
            sentences = re.split(r'(?<=[。！？；\n])', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    # 如果单个句子还是太长，硬切
                    if len(sent) > chunk_size:
                        for i in range(0, len(sent), chunk_size - overlap):
                            sub = sent[i:i + chunk_size]
                            if sub.strip():
                                chunks.append(sub.strip())
                        current = ""
                    else:
                        current = sent
            if current.strip():
                chunks.append(current.strip())

    # 第二步：构建 LangChain Document 对象列表
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"chunk_index": i, "source": source_path}
        )
        documents.append(doc)

    return documents


def load_and_split(file_path, chunk_size=500, overlap=50):
    """
    根据文件类型加载文档，然后切分成小块。

    参数:
        file_path: 文档文件的路径
        chunk_size: 每块最大字符数
        overlap: 相邻块重叠字符数

    返回:
        LangChain Document 对象列表
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ---- 加载文档 ----
    if ext == ".pdf":
        print(f"[文档加载] 正在加载 PDF 文件: {file_path}")
        text = _load_pdf(file_path)
    elif ext == ".docx":
        print(f"[文档加载] 正在加载 Word 文件: {file_path}")
        text = _load_docx(file_path)
    elif ext in [".md", ".txt"]:
        print(f"[文档加载] 正在加载文本文件: {file_path}")
        text = _load_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF、DOCX、MD、TXT")

    if not text or not text.strip():
        print("[文档加载] 警告：文档内容为空")
        return []

    print(f"[文档加载] 加载完成，文本总长度: {len(text)} 字符")

    # ---- 切分文档 ----
    documents = _simple_split(text, chunk_size=chunk_size, overlap=overlap, source_path=file_path)
    print(f"[文档加载] 切分完成，共 {len(documents)} 个文本块")

    return documents
