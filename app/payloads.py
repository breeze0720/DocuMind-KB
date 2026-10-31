"""
app/payloads.py - 请求参数解析兼容层

为什么要手写一个 read_payload：
前端（新同事重写的 static/index.html）会用 JSON 调 /api/*；
而旧的 static 页面、curl 示例、eval.py 习惯用 form 表单；SSE 的 EventSource 只能 GET + query。
如果每个接口都写死一种 Content-Type，换一个调用方就 422，联调时非常浪费时间。
这个函数按 JSON → form → query 的优先级合并参数，同一个接口三种调用方式都能工作。
"""
from __future__ import annotations

from fastapi import Request


async def read_payload(request: Request) -> dict:
    """把 JSON / 表单 / query 参数合并成一个 dict（body 优先，query 只补缺）。"""
    data: dict = {}
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                data.update(body)
        except Exception:
            # body 不是合法 JSON 就退回 query/form，不直接 400，保证老调用方还能用
            pass
    elif "form" in content_type or "multipart" in content_type:
        try:
            form = await request.form()
            data.update({key: value for key, value in form.items()})
        except Exception:
            pass
    for key, value in request.query_params.items():
        data.setdefault(key, value)
    return data


def as_int(value, default: int) -> int:
    """把前端传来的字符串/数字统一转 int，非法值用默认值而不是抛异常（避免 500）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value) -> float | None:
    """可空 float：用于 min_score 这类可选阈值。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None