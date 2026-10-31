"""tests/test_upload_utils.py - 上传校验纯函数测试：大小上限、文件名清洗、SHA 幂等、zip 炸弹。"""
import io
import zipfile

from app.upload_utils import (
    check_extension,
    check_size,
    find_duplicate,
    sanitize_filename,
    stored_filename,
    upload_limit_bytes,
    zip_bomb_reason,
)


def make_zip(files: dict[str, bytes]) -> bytes:
    """在内存里构造 zip 字节流，用来模拟 docx/xlsx 等 OOXML 容器。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_check_extension():
    assert check_extension("员工手册.pdf") is None
    assert check_extension("FAQ.MD") is None  # 大小写不敏感
    assert "不支持的文件格式" in check_extension("virus.exe")
    assert "不支持的文件格式" in check_extension("noext")


def test_check_size():
    assert upload_limit_bytes(50) == 50 * 1024 * 1024
    assert check_size(10 * 1024 * 1024, 50) is None
    message = check_size(60 * 1024 * 1024, 50)
    assert message and "超过上限 50 MB" in message
    # size 缺失时不能放行也不能误拒：交给流式写入时的字节计数兜底
    assert check_size(None, 50) is None


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"  # 目录穿越被剥掉
    assert sanitize_filename(r"C:\tmp\a:b?.docx") == "a_b_.docx"
    assert sanitize_filename("") == "unnamed"
    assert len(sanitize_filename("长" * 300)) <= 120


def test_stored_filename_uses_hash_prefix():
    name = stored_filename("abcdef1234567890", "员工手册.pdf")
    assert name == "abcdef12_员工手册.pdf"


def test_find_duplicate():
    assert find_duplicate("abc", ["abc", "def"]) is True
    assert find_duplicate("xyz", ["abc"]) is False
    assert find_duplicate("", [""]) is False  # 空哈希不能算命中


def test_zip_bomb_normal_document_passes():
    data = make_zip({"word/document.xml": b"<w:document>hello</w:document>"})
    assert zip_bomb_reason(data, ".docx") is None


def test_zip_bomb_high_compression_ratio_rejected():
    # 2MB 全零压缩后只有几 KB，压缩比远超 100:1
    data = make_zip({"word/document.xml": b"\x00" * (2 * 1024 * 1024)})
    reason = zip_bomb_reason(data, ".docx")
    assert reason is not None
    assert "压缩比" in reason or "解压后总大小" in reason


def test_zip_bomb_total_size_rejected():
    """解压总量超上限（这里把上限调成 1MB，方便小数据构造）。"""
    data = make_zip({"xl/sharedStrings.xml": b"\x00" * (2 * 1024 * 1024)})
    reason = zip_bomb_reason(data, ".xlsx", max_ratio=10_000, max_total_mb=1)
    assert reason is not None and "解压后总大小" in reason


def test_zip_bomb_too_many_entries_rejected():
    data = make_zip({f"f{index}.xml": b"x" for index in range(5)})
    reason = zip_bomb_reason(data, ".pptx", max_entries=3)
    assert reason is not None and "条目过多" in reason


def test_zip_bomb_ignores_non_zip_and_broken_zip():
    assert zip_bomb_reason(b"hello", ".txt") is None          # txt 不是 zip 容器，不检查
    assert zip_bomb_reason(b"not a real zip", ".docx") is None  # 伪扩展名交给解析器报「文件损坏」