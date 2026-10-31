"""
core/agent.py - RAG 智能体（LangChain create_agent + DeepSeek）

本文件的三处关键改动（都是「从 demo 到能讲」的分水岭）：
1) 检索工具返回带编号的资料（[1] 来源: 《员工手册》…），系统提示强制模型只许引用这些编号。
   编号由程序分配、模型只负责引用，前端 sources 与 [n] 一一对应。
2) 修掉旧代码里 role 提取那行的运算符优先级 bug：
   旧写法 `a or b if cond else c` 会被解析成 `(a or b) if cond else c`，
   对 dict 消息永远只看 type、对 Message 对象永远只看 type，两个分支逻辑不一致。
   现在显式 if/else，dict 用 role、消息对象用 type。
3) 新增 ask_stream()：SSE 流式回答。先发 sources 事件再逐 token 推进，
   这样前端可以在回答开始时就渲染「参考来源」卡片。

关于引用注册表的传递方式（踩过的坑，值得写下来）：
最初想用 contextvars.ContextVar 把「本次提问的注册表」传进工具，本地试的时候能拿到；
但接进 FastAPI / LangGraph 后工具执行上下文并没有继承 set 进去的值，导致 sources 恒为空
（回答里的 [1] 有、来源列表没有）。正确做法是走 LangGraph 的 config 通道：
invoke/stream 时传 config={"configurable": {"citation_registry": registry}}，
工具声明 RunnableConfig 参数，由框架注入。这样并发请求之间也天然隔离。
"""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from typing import Any, Iterator

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

import config
from core.citations import CitationRegistry, format_numbered_context
from core.embeddings import EmbeddingError
from core.prompts import SYSTEM_PROMPT
from core.runtime import get_retriever


# ============================================
# 工具定义
# ============================================
_fallback_registry = CitationRegistry()


def _registry_from_config(config: RunnableConfig | None) -> CitationRegistry:
    """从框架注入的 config 里取本次提问的引用注册表。"""
    try:
        registry = ((config or {}).get("configurable") or {}).get("citation_registry")
    except Exception:
        registry = None
    return registry if isinstance(registry, CitationRegistry) else _fallback_registry


@tool
def search_knowledge_base(query: str, runnable_config: RunnableConfig) -> str:
    """在知识库中搜索与问题相关的资料，返回带引用编号的片段。
    参数 query 是检索问题或关键词；runnable_config 由框架注入（不暴露给模型），
    用来取本次提问的引用注册表——注意别把参数命名成 config，会遮蔽模块级 import config。"""
    print(f"[Agent] 检索: {query}")
    try:
        hits = get_retriever().search(query, k=config.settings.top_k)
    except EmbeddingError as exc:
        # 配置问题（如模型文件缺失）不应该变成满屏异常栈，让模型用中文告诉用户怎么修
        return f"知识库检索不可用：{exc}"
    except Exception as exc:
        return f"知识库检索出错：{exc}"
    if not hits:
        return "知识库中暂时没有找到相关内容。可以提醒用户先上传文档或调用 /api/reindex 导入语料。"
    return format_numbered_context(hits, _registry_from_config(runnable_config))


@tool
def send_email(to: str, subject: str, content: str) -> str:
    """发送邮件。参数 to：收件人完整邮箱；subject：主题；content：正文。调用前需确认邮箱地址。"""
    settings = config.settings
    if not settings.email_from or settings.email_from == "your_email@qq.com":
        return "邮件发送失败：未配置发件人邮箱，请在 .env 填写 EMAIL_FROM 和 EMAIL_PASSWORD"
    try:
        message = MIMEText(content, "plain", "utf-8")
        message["To"] = to
        message["Subject"] = subject
        message["From"] = settings.email_from
        smtp = smtplib.SMTP_SSL(settings.email_host, port=465)
        smtp.login(settings.email_from, settings.email_password)
        smtp.sendmail(settings.email_from, to, message.as_string())
        smtp.quit()
        return f"邮件已成功发送给 {to}！"
    except smtplib.SMTPAuthenticationError:
        return "邮件发送失败：认证失败，请检查 EMAIL_PASSWORD（QQ 邮箱需要授权码，不是登录密码）"
    except Exception as exc:
        return f"邮件发送失败: {exc}"


# ============================================
# Agent 管理
# ============================================
_agent: Any = None
# 对话历史（进程内）。教学项目够用；多用户/重启不丢需要落库，这里不引入数据库
_history: list[dict[str, str]] = []
_MAX_HISTORY = 40  # 保留最近 20 轮


def init_agent(force: bool = False):
    """初始化 Agent（幂等）。ChatOpenAI 构造不联网，真正的调用发生在 ask 时。"""
    global _agent
    if _agent is not None and not force:
        return _agent
    llm = ChatOpenAI(
        model=config.settings.model_name,
        api_key=config.settings.deepseek_api_key,
        base_url=config.settings.deepseek_base_url,
        temperature=config.settings.temperature,
        max_tokens=config.settings.max_tokens,
        streaming=True,  # 打开后 stream_mode="messages" 才能拿到逐 token 的增量
    )
    _agent = create_agent(model=llm, system_prompt=SYSTEM_PROMPT, tools=[search_knowledge_base, send_email])
    print(f"[Agent] 初始化完成，模型: {config.settings.model_name}")
    return _agent


def _message_role(message: Any) -> str:
    """统一取消息角色。dict 消息（OpenAI 格式）用 role，LangChain 消息对象用 type。"""
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "type", ""))


def _message_text(message: Any) -> str:
    """取消息正文；兼容 content 是 str 或 [{'type':'text','text':...}] 列表两种形态。"""
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _extract_answer(messages: list[Any]) -> str:
    """从消息流里取最后一条有正文的 AI 消息（修复旧代码运算符优先级 bug 的地方）。"""
    for message in reversed(messages):
        role = _message_role(message)
        if role in ("ai", "assistant"):
            text = _message_text(message).strip()
            if text:
                return text
    return "抱歉，没有生成有效回答。"


def _history_messages(question: str) -> list[dict[str, str]]:
    """历史 + 当前问题。只用 role/content 两个字段，避免把工具消息塞回上下文。"""
    messages = [{"role": item["role"], "content": item["content"]} for item in _history]
    messages.append({"role": "user", "content": question})
    return messages


def _remember(question: str, answer: str) -> None:
    global _history
    _history.append({"role": "user", "content": question})
    _history.append({"role": "assistant", "content": answer})
    if len(_history) > _MAX_HISTORY:
        _history = _history[-_MAX_HISTORY:]


def ask(question: str) -> tuple[str, list[dict[str, Any]]]:
    """
    提问，返回 (回答, sources)。

    sources 是本次回答引用的来源列表，顺序与 [n] 编号一致（n 从 1 开始），
    即使模型一次工具都没调用也返回空列表，调用方不用判断 None。
    """
    if _agent is None:
        init_agent()
    registry = CitationRegistry()
    try:
        result = _agent.invoke(
            {"messages": _history_messages(question)},
            config={"configurable": {"citation_registry": registry}},
        )
        answer = _extract_answer(result.get("messages", []))
    except Exception as exc:
        print(f"[Agent] 调用出错: {exc}")
        # 出错不写历史：避免把错误文案当成上一轮回答污染下一轮上下文
        return f"抱歉，智能体处理时出错了: {exc}", registry.sources()
    _remember(question, answer)
    return answer, registry.sources()


def ask_stream(question: str) -> Iterator[tuple[str, Any]]:
    """
    SSE 流式回答的同步生成器，产出 (事件名, 数据)：
    - ("sources", [...]) 先发来源；- ("token", "文本") 逐段发；- ("done", {answer, sources})

    为什么 token 要先缓冲：模型可能先输出一句「我先查一下」再调用工具，
    此时注册表还是空的；缓冲到工具执行完，先发 sources 再一次性把缓冲的 token 发出去，
    保证前端拿到 sources 时它和后续 token 的编号一定对得上。
    模型完全没调工具时，缓冲会在结束时随空的 sources 一起发出（退化为一次性输出，但绝不串号）。
    """
    if _agent is None:
        init_agent()
    registry = CitationRegistry()
    config = {"configurable": {"citation_registry": registry}}
    buffered: list[str] = []
    answer_parts: list[str] = []
    sources_sent = False

    def flush_tokens() -> Iterator[tuple[str, Any]]:
        nonlocal buffered
        for piece in buffered:
            yield "token", piece
        buffered = []

    try:
        for chunk, _metadata in _agent.stream(
            {"messages": _history_messages(question)}, stream_mode="messages", config=config
        ):
            # 只取模型产生的 token。工具结果（ToolMessage）也会出现在消息流里，
            # 它属于「检索结果」而不是「回答」，混进 token 会把资料原文当答案推给前端。
            if not isinstance(chunk, AIMessageChunk):
                continue
            text = _message_text(chunk)
            if not text:
                continue
            answer_parts.append(text)
            if not sources_sent and len(registry) > 0:
                sources_sent = True
                yield "sources", registry.sources()
                yield from flush_tokens()
            if not sources_sent:
                buffered.append(text)  # 工具还没执行完，先不急于发 token
            else:
                yield "token", text
        if not sources_sent:
            yield "sources", registry.sources()
            yield from flush_tokens()
        answer = "".join(answer_parts).strip() or "抱歉，没有生成有效回答。"
        _remember(question, answer)
        yield "done", {"answer": answer, "sources": registry.sources()}
    except Exception as exc:
        print(f"[Agent] 流式调用出错: {exc}")
        yield "error", {"message": f"回答生成失败: {exc}"}


def clear_history() -> None:
    """清空对话历史（/api/session/reset 用）。"""
    global _history
    _history = []
    _fallback_registry.reset()
    print("[Agent] 对话历史已清空")