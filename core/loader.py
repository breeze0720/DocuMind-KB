"""
core/loader.py - 多格式文档解析

设计要点：loader 只负责「把文件读成一个个内容单元（unit）」，每个 unit 带来源信息
（PDF 页码 / Excel sheet / PPT 幻灯片 / docx 章节），切块交给 splitter。
这样引用能说清「《员工手册》第 3 页第 2 块」，而不是只给一段没头没尾的文字。

格式支持：pdf/docx/md/txt 完整支持；xlsx/pptx/csv 按 sheet/行/幻灯片抽纯文本，
不做样式与合并单元格还原（撑不起复杂表格问答时再换 docling/unstructured）。
依赖库的 import 放在函数内部：用到哪种格式才加载哪个库，启动更快，缺库也不影响其他格式。
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

# 允许上传/导入的扩展名（上传接口和 corpus 导入共用这一份，避免两处白名单不一致）
ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".md", ".txt", ".xlsx", ".pptx", ".csv")

# 这些是 zip 容器格式，上传时要额外做 zip 炸弹检查（见 app/upload_utils.py）
ZIP_CONTAINER_EXTENSIONS: tuple[str, ...] = (".docx", ".xlsx", ".pptx")


class UnsupportedFormatError(ValueError):
    """不支持的扩展名。上层转成 400，而不是 500。"""


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算 SHA256，用作「文件内容唯一标识」（上传幂等、去重）：同名不同内容、不同名同内容都很常见。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(text: str, **metadata) -> tuple[str, dict]:
    """内容单元：正文 + 来源元数据。"""
    return text, metadata


# ============================================
# 各格式解析器：统一返回 [(文本, 来源元数据), ...]
# ============================================
def _parse_pdf(path: Path) -> list[tuple[str, dict]]:
    """PDF 按页解析。pypdf 抽的是文字层，扫描件需要 OCR（本项目不引入，会显式表现为空文本）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    units: list[tuple[str, dict]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            units.append(_unit(text, page=page_no))
    return units


def _parse_docx(path: Path) -> list[tuple[str, dict]]:
    """Word 按「标题→下一个标题」分节，标题行保留在正文里；表格单独成单元（单元格用 | 拼文本）。"""
    import docx

    document = docx.Document(str(path))
    units: list[tuple[str, dict]] = []
    section = ""
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if lines:
            units.append(_unit("\n".join(lines), section=section))
            lines = []

    for paragraph in document.paragraphs:
        text = (paragraph.text or "").strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
        if style.startswith("heading") or style in ("title", "subtitle"):
            flush()
            section = text
            lines = [text]
        else:
            lines.append(text)
    flush()

    for index, table in enumerate(document.tables, start=1):
        rows = [" | ".join((cell.text or "").strip() for cell in row.cells) for row in table.rows]
        rows = [row for row in rows if row.strip(" |")]
        if rows:
            units.append(_unit("\n".join(rows), section=f"表格{index}"))
    return units


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _parse_md(path: Path) -> list[tuple[str, dict]]:
    """Markdown 按 # 标题分节。保留标题文字、去掉 # 号（# 对 embedding 是噪声）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    units: list[tuple[str, dict]] = []
    section = ""
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if lines:
            units.append(_unit("\n".join(lines).strip(), section=section))
            lines = []

    for line in text.splitlines():
        match = _MD_HEADING_RE.match(line)
        if match:
            flush()
            section = match.group(2).strip()
            lines = [section]
        else:
            lines.append(line)
    flush()
    return units


def _parse_txt(path: Path) -> list[tuple[str, dict]]:
    """纯文本整篇一个单元；空行/编码问题在这里兜底（errors=replace，不让一个坏字符毁掉整篇）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [_unit(text)] if text.strip() else []


def _parse_csv(path: Path) -> list[tuple[str, dict]]:
    """CSV 一行一个单元，并给每行补列名（「年假天数: 10」比裸的「10」可检索）；先 utf-8-sig 再 gbk。"""
    rows: list[list[str]] = []
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=encoding, newline="") as fh:
                rows = [row for row in csv.reader(fh)]
            break
        except UnicodeDecodeError:
            continue
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    units = []
    for index, row in enumerate(body, start=2):
        pairs = [f"{header[i]}: {value}" for i, value in enumerate(row) if i < len(header) and value.strip()]
        text = "；".join(pairs) if pairs else " | ".join(value for value in row if value.strip())
        if text.strip():
            units.append(_unit(text, row=index))
    return units


def _parse_xlsx(path: Path) -> list[tuple[str, dict]]:
    """Excel 每个 sheet 一个单元。read_only=True 避免把大表全读进内存。"""
    import openpyxl

    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    units: list[tuple[str, dict]] = []
    try:
        for sheet in workbook.worksheets:
            lines = []
            for row in sheet.iter_rows(values_only=True):
                cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            if lines:
                units.append(_unit("\n".join(lines), sheet=sheet.title))
    finally:
        workbook.close()
    return units


def _parse_pptx(path: Path) -> list[tuple[str, dict]]:
    """PPT 每张幻灯片一个单元，抽取文本框与表格；图片/备注不做 OCR。"""
    from pptx import Presentation

    presentation = Presentation(str(path))
    units: list[tuple[str, dict]] = []
    for slide_no, slide in enumerate(presentation.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs).strip()
                    if text:
                        lines.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [(cell.text or "").strip() for cell in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
        if lines:
            units.append(_unit("\n".join(lines), slide=slide_no))
    return units


_PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".md": _parse_md,
    ".txt": _parse_txt,
    ".csv": _parse_csv,
    ".xlsx": _parse_xlsx,
    ".pptx": _parse_pptx,
}


def _guess_title(units: list[tuple[str, dict]], path: Path) -> str:
    """文档标题：优先章节名（docx/md 的第一个标题），其次首行，最后文件名。"""
    for _, metadata in units:
        section = (metadata.get("section") or "").strip()
        if section and not section.startswith("表格"):
            return section[:80]
    for text, _ in units:
        first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
        if first_line:
            return first_line.strip("# ").strip()[:80]
    return path.stem


def load_documents(path: str | Path, file_hash: str | None = None) -> list[Document]:
    """解析一个文件，返回内容单元列表（带 source/file_hash/title + 页/sheet/章节信息）。
    参数 file_hash：上传流程已算过 SHA256 时传入，避免重复读大文件。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    ext = file_path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFormatError(
            f"不支持的文件格式「{ext}」，仅支持 {', '.join(ALLOWED_EXTENSIONS)}"
        )
    digest = file_hash or file_sha256(file_path)
    units = parser(file_path)
    if not units:
        return []
    title = _guess_title(units, file_path)

    documents: list[Document] = []
    for text, extra_metadata in units:
        if not (text or "").strip():
            continue
        metadata = {"source": file_path.name, "file_hash": digest, "title": title}
        metadata.update({key: value for key, value in extra_metadata.items() if value is not None})
        documents.append(Document(page_content=text.strip(), metadata=metadata))
    return documents