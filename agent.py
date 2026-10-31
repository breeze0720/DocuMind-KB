"""
agent.py - 兼容 shim（旧入口，保持 import 不坏）

真正的 Agent 实现已拆到 core/agent.py（引用编号、流式输出、bug 修复都在那里）。
保留这个文件是为了让旧的 `from agent import init_agent, ask, clear_history`
（比如用户以前写的脚本/practice 代码）继续能用，不用改 import。
"""
from core.agent import (  # noqa: F401  （兼容导出）
    SYSTEM_PROMPT,
    ask,
    ask_stream,
    clear_history,
    init_agent,
    search_knowledge_base,
    send_email,
)

__all__ = [
    "SYSTEM_PROMPT",
    "ask",
    "ask_stream",
    "clear_history",
    "init_agent",
    "search_knowledge_base",
    "send_email",
]