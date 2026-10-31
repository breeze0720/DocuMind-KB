"""
app/upload_utils.py - 上传相关的纯函数校验层

为什么把这些逻辑抽成纯函数（不碰 FastAPI、不碰数据库）：
上传链路的坑（大小超限、幂等、zip 炸弹）最容易出错，而它们本质上是「输入 → 是否拒绝」的
判断，与 Web 框架无关。纯函数可以直接在 pytest 里构造极端输入（构造 1000:1 的压缩比、
伪造扩展名），不需要起服务、不需要真上传 500MB 文件。

对应三道防线：
1) 扩展名白名单：第一道，挡住明显不处理的格式；
2) 大小上限：先看 file.size 快速拒绝（413），流式落盘时再按实际字节数兜底；
3) zip 炸弹：docx/xlsx/pptx 本质是 zip，几百 KB 可解压出几十 GB。
   只读 zip 中央目录里声明的 file_size，不解压，所以检查本身不耗资源。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

from core.loader import ALLOWED_EXTENSIONS, ZIP_CONTAINER_EXTENSIONS, file_sha256

ALLOWED = frozenset(ALLOWED_EXTENSIONS)
_MB = 1024 * 1024
# 一个正常 OOXML 文档的条目数通常 < 1000；超过这个量级基本是构造的压缩包
_MAX_ZIP_ENTRIES = 10000


def upload_limit_bytes(max_mb: int) -> int:
    """把 MB 配置换成字节，避免各调用点重复写乘法、写错单位。"""
    return int(max_mb) * _MB


def check_extension(filename: str) -> str | None:
    """扩展名白名单。返回中文错误文案，None 表示通过。"""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED:
        return f"不支持的文件格式「{ext or '无扩展名'}」，仅支持 {', '.join(ALLOWED_EXTENSIONS)}"
    return None


def check_size(size: int | None, max_mb: int) -> str | None:
    """
    大小预检：优先用 UploadFile.size（Starlette 解析 multipart 时已拿到），
    小于等于上限返回 None。size 为 None（分块传输等拿不到长度）时返回 None，
    由流式写入时的累计字节数兜底——不能因为拿不到长度就放行任意大文件。
    """
    if size is None or size < 0:
        return None
    limit = upload_limit_bytes(max_mb)
    if size > limit:
        return f"文件过大：{size / _MB:.1f} MB，超过上限 {max_mb} MB"
    return None


def sanitize_filename(filename: str) -> str:
    """
    去掉路径和非法字符，防目录穿越（../../.env）与 Windows 保留字符。
    只取最后一段文件名，并限制长度，防止超长文件名在磁盘上报错。
    """
    name = Path((filename or "").replace("\\", "/")).name.strip()
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    name = "".join(ch for ch in name if ord(ch) >= 32).strip(". ")
    return (name or "unnamed")[:120]


def stored_filename(file_hash: str, filename: str) -> str:
    """落盘名 = 内容哈希前 8 位 + 原文件名。哈希前缀让同名文件不互相覆盖，也方便人工核对幂等。"""
    return f"{(file_hash or '')[:8]}_{sanitize_filename(filename)}"


def file_sha256_of(path: str | Path) -> str:
    """薄封装，便于上传模块统一从这里取哈希实现。"""
    return file_sha256(path)


def find_duplicate(file_hash: str, indexed_hashes: Iterable[str]) -> bool:
    """
    幂等判断（纯函数）：内容哈希已存在 → 不重复解析、不重复 embedding、不重复计费。
    为什么按内容而不是文件名：同一份文件改个名再传，不应该在知识库里出现两份。
    """
    return bool(file_hash) and file_hash in set(indexed_hashes)


def zip_bomb_reason(
    source: str | Path | bytes | bytearray,
    ext: str,
    max_ratio: int = 100,
    max_total_mb: int = 500,
    max_entries: int = _MAX_ZIP_ENTRIES,
) -> str | None:
    """
    检查 zip 容器是否像 zip 炸弹，返回中文原因或 None。

    判定依据（对齐 DocsGPT 的做法，但代码自己写）：
    - 解压后总大小 > max_total_mb（默认 500MB）；
    - 压缩比（解压后 / 压缩后）> max_ratio（默认 100:1）；
    - 条目数 > max_entries。
    注意：只读中央目录，不真正解压，因此检查一个 10GB 展开的炸弹也只需要毫秒级。
    非 zip 格式、或根本不是合法 zip（扩展名伪装）直接放行，让真正的解析器报「文件损坏」，
    这样错误信息对用户更有用。
    """
    if ext.lower() not in ZIP_CONTAINER_EXTENSIONS:
        return None
    opened: str | Path | io.BytesIO
    opened = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
    try:
        with zipfile.ZipFile(opened) as archive:
            infos = archive.infolist()
            if len(infos) > max_entries:
                return f"压缩包条目过多（{len(infos)} > {max_entries}），疑似恶意文件，已拒绝"
            total = sum(info.file_size for info in infos)
            compressed = max(1, sum(info.compress_size for info in infos))
    except (zipfile.BadZipFile, OSError):
        return None
    limit = int(max_total_mb) * _MB
    if total > limit:
        return f"解压后总大小约 {total / _MB:.1f} MB，超过上限 {max_total_mb} MB，疑似 zip 炸弹，已拒绝"
    ratio = total / compressed
    if ratio > max_ratio:
        return f"压缩比异常（约 {ratio:.0f}:1 > {max_ratio}:1），疑似 zip 炸弹，已拒绝"
    return None