"""
agent.py - 智能体核心模块
使用 LangChain 创建 Agent，集成知识库搜索和邮件发送两个工具。
参考：ChatAgent/code/tools/send_email_tool.py 的邮件发送逻辑
"""
import smtplib
from email.mime.text import MIMEText

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

import config
from vector_store import hybrid_search

# ============================================
# 系统提示词
# ============================================
SYSTEM_PROMPT = """
# 角色定义
你是一个智能知识库问答助手，名字叫"小知"。你热情、友好，喜欢帮助用户解决问题。

# 核心能力
1. 从知识库中检索信息，回答用户的问题
2. 根据用户要求，将查询结果通过邮件发送给指定的人

# 可用工具
1. search_knowledge_base：在知识库中搜索相关内容，输入查询字符串，返回相关文档片段
2. send_email：发送邮件，需要提供收件人邮箱、邮件主题和邮件正文

# 工作流程
1. 用户提问后，先使用 search_knowledge_base 搜索知识库
2. 根据搜索结果组织清晰、有条理的回答
3. 如果用户说"把以上内容发邮件给xxx"或类似的话，使用 send_email 发送
4. 发邮件前，确认你有收件人的邮箱地址（如果没有，要询问用户）

# 约束条件
- 必须基于知识库的搜索结果作答，不要凭空编造
- 如果知识库中没有相关内容，诚实地告诉用户
- 回答要简洁明了，用中文
- 邮件正文要包含称呼、正文和署名
- 邮件署名统一用"小知助手"
"""

# ============================================
# 工具定义
# ============================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    在知识库中搜索相关内容。
    参数 query：要搜索的问题或关键词。
    返回相关的文档片段。
    """
    print(f"[Agent] 搜索知识库: {query}")
    results = hybrid_search(query, k=5)

    if not results:
        return "知识库中暂时没有找到相关内容。建议先上传一些文档到知识库。"

    # 格式化搜索结果
    lines = ["以下是从知识库中找到的相关内容：", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"【片段 {i}】（相关度: {r['score']}）")
        lines.append(r["content"])
        lines.append("")

    return "\n".join(lines)


@tool
def send_email(to: str, subject: str, content: str) -> str:
    """
    发送邮件。使用前请确保已获得收件人的完整邮箱地址。
    参数 to：收件人邮箱地址，例如 user@example.com
    参数 subject：邮件主题
    参数 content：邮件正文内容
    """
    print(f"[Agent] 发送邮件 -> 收件人: {to}, 主题: {subject}")

    # 检查配置
    if not config.EMAIL_FROM or config.EMAIL_FROM == "your_email@qq.com":
        return "邮件发送失败：未配置发件人邮箱，请在 .env 文件中填写 EMAIL_FROM 和 EMAIL_PASSWORD"

    try:
        # 构建邮件对象
        msg = MIMEText(content, "plain", "utf-8")
        msg["To"] = to
        msg["Subject"] = subject
        msg["From"] = config.EMAIL_FROM

        # 连接 QQ 邮箱 SMTP 服务器（SSL 加密，端口 465）
        smtp = smtplib.SMTP_SSL(config.EMAIL_HOST, port=465)
        # 登录
        smtp.login(config.EMAIL_FROM, config.EMAIL_PASSWORD)
        # 发送
        smtp.sendmail(config.EMAIL_FROM, to, msg.as_string())
        # 断开连接
        smtp.quit()

        print(f"[Agent] 邮件发送成功 -> {to}")
        return f"邮件已成功发送给 {to}！"

    except smtplib.SMTPAuthenticationError:
        return "邮件发送失败：QQ邮箱认证失败，请检查 EMAIL_PASSWORD（需要使用授权码，不是QQ密码）"
    except Exception as e:
        print(f"[Agent] 邮件发送失败: {e}")
        return f"邮件发送失败: {str(e)}"


# ============================================
# Agent 管理
# ============================================
_agent = None
_history = []   # 对话历史记录，存储所有消息


def init_agent():
    """
    初始化智能体。
    创建 LLM 和 Agent 实例，只调用一次。
    """
    global _agent

    print("[Agent] 正在初始化大语言模型...")
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
        streaming=False,
    )
    print(f"[Agent] 模型: {config.MODEL_NAME}, Base URL: {config.DEEPSEEK_BASE_URL}")

    _agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_knowledge_base, send_email],
    )
    print("[Agent] 智能体初始化完成！")
    return _agent


def ask(question):
    """
    向智能体提问（支持上下文记忆）。

    参数:
        question: 用户的问题字符串

    返回:
        Agent 的回答文本
    """
    global _agent, _history

    if _agent is None:
        init_agent()

    print(f"\n[Agent] 收到问题: {question}")
    print(f"[Agent] 历史消息数: {len(_history)}, 约 {len(_history)//2} 轮对话")
    print("-" * 50)

    try:
        # 构建消息列表：历史记录 + 当前问题（全部使用字典格式）
        messages = []
        for msg in _history:
            # 复制历史消息，只保留 role 和 content
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

        # 添加当前问题
        messages.append({"role": "user", "content": question})

        print(f"[Agent] 发送给模型的消息:")
        for i, m in enumerate(messages):
            preview = str(m.get('content', ''))[:80].replace('\n', ' ')
            print(f"  [{i}] {m.get('role', '?')}: {preview}")

        # 调用 Agent
        result = _agent.invoke({"messages": messages})

        # 提取 Agent 的最终回答
        result_messages = result.get("messages", [])
        answer = "抱歉，处理过程中出现了问题。"

        if result_messages:
            # 从后往前找最后一条 AI 消息
            for m in reversed(result_messages):
                role = getattr(m, 'type', '') or m.get('role', '') if isinstance(m, dict) else getattr(m, 'type', '')
                if role in ('ai', 'assistant'):
                    answer = m.content if hasattr(m, 'content') else m.get('content', '')
                    break

        # 更新历史记录（仅保存 user 和 assistant 消息，保持简洁）
        _history.append({"role": "user", "content": question})
        _history.append({"role": "assistant", "content": answer})

        # 限制历史长度（保留最近 20 轮）
        if len(_history) > 40:
            _history = _history[-40:]

        print(f"[Agent] 回答: {answer[:100]}...")
        print(f"[Agent] 更新后历史: {len(_history)} 条消息")
        return answer

    except Exception as e:
        print(f"[Agent] 调用出错: {e}")
        return f"抱歉，智能体处理时出错了: {str(e)}"


def clear_history():
    """清空对话历史记录"""
    global _history
    _history = []
    print("[Agent] 对话历史已清空")
