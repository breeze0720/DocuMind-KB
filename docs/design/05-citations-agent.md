# 05 · Citations 与 Agent：让 `[n]` 一定指得回原文

> 一句话：**编号权在程序手里，不在模型手里**。检索完由程序给每块分配 `[1][2]…`，
> 模型只允许引用这些编号；前端拿 `sources` 数组按编号渲染，`[2]` 必然对应 `sources[1]`。
>
> 术语：**citation**（引用）、**registry**（注册表，本轮编号 → 来源的映射）、
> **tool call**（工具调用）、**stream**（流式输出）、**RunnableConfig**（LangChain 的运行时配置通道）。

## 1. 实现概览

### 1.1 注册表：一次提问一份，register 幂等

`core/citations.py · CitationRegistry`：

```python
@staticmethod
def _key(hit) -> tuple[str, Any]:
    # ① 用 (原始来源, chunk_index) 做唯一标识
    return (str(hit.get("source_key") or hit.get("source") or ""), hit.get("chunk_index"))

def register(self, hit) -> int:
    key = self._key(hit)
    if key in self._numbers:
        return self._numbers[key]          # ② 同一块重复命中，返回同一个编号
    number = len(self._items) + 1          # ③ 编号从 1 开始，按登记顺序递增
    self._numbers[key] = number
    self._items.append({"n": number, "source": hit.get("source") or "", "chunk_index": ..., "preview": ...})
    return number
```

- ① 不用正文做 key：不同文档里文字相同的块也要能分别编号（和 04 篇的去重 key 思路一致）；
- ② 同一块被两路检索重复命中时不能拿两个编号，否则 `[1]` 和 `[2]` 内容一样；
- ③ 注册表**每轮提问新建一份**（`ask()` 里 `registry = CitationRegistry()`），
  所以第二次提问的 `[1]` 是第二次自己的第一条来源，不会串到上一轮。
  `reset()` 只在显式清会话时用。

`format_numbered_context(hits, registry)` 把检索结果拼成给模型的上下文：

```
以下是从知识库检索到的资料（编号由系统分配，回答时只能引用这些编号）：

[1] 来源: 《员工手册.docx》 第3页 第7块
<该块正文>

[2] 来源: 《报销与差旅制度.pdf》 第1页 第2块
<该块正文>

引用规则：引用资料里的事实、数字、流程时，必须在句末标注编号，例如 [1]；
不要编造资料里没有的编号或内容；资料不足以回答时，直接说明知识库中没有找到。
```

注意它**故意不拼 RRF 分数**：模型看到 `score: 0.03` 会认为这条资料「相关度只有 3%」而不敢引用
（04 篇 1.4 讲过原因）。对回答有用的只有编号、来源位置和正文。

### 1.2 工具：模型能调的只有一个检索函数

`core/agent.py` 用 LangChain 的 `@tool` 装饰器定义 `search_knowledge_base`：

```python
@tool
def search_knowledge_base(query: str, runnable_config: RunnableConfig) -> str:
    """在知识库中搜索与问题相关的资料，返回带引用编号的片段。"""
    print(f"[Agent] 检索: {query}")
    try:
        hits = get_retriever().search(query, k=config.settings.top_k)      # ① 默认取 5 条
    except EmbeddingError as exc:
        return f"知识库检索不可用：{exc}"                                    # ② 配置错误变成中文提示
    except Exception as exc:
        return f"知识库检索出错：{exc}"
    if not hits:
        return "知识库中暂时没有找到相关内容。可以提醒用户先上传文档或调用 /api/reindex 导入语料。"
    return format_numbered_context(hits, _registry_from_config(runnable_config))  # ③ 登记编号
```

- ① 工具一次只取 `top_k=5` 条，所以一次回答最多 5 个引用编号；
- ② 工具**永远返回字符串、不抛异常**：抛出会变成满屏栈，返回字符串则模型能用中文告诉用户怎么修
  （比如「本地向量模型文件缺失，先跑 download_model.ps1」）；
- ③ `runnable_config` 由框架注入、不暴露给模型（模型只看到 `query` 一个参数），
  它是本轮注册表的唯一来源。参数故意不叫 `config`，否则会遮蔽模块级 `import config`。

### 1.3 注册表怎么穿过 LangGraph 到工具里（踩过的坑）

```python
result = _agent.invoke(
    {"messages": _history_messages(question)},
    config={"configurable": {"citation_registry": registry}},   # ① 走框架的 config 通道
)
answer = _extract_answer(result.get("messages", []))
return answer, registry.sources()                                 # ② sources 顺序 = 编号顺序
```

- ① 最初用 `contextvars.ContextVar` 传注册表，本地能拿到；接进 FastAPI/LangGraph 后
  工具执行上下文没有继承这个值，导致「回答里有 `[1]`、`sources` 却是空数组」。
  改成 `configurable` 通道后由框架注入，并发请求之间也天然隔离；
- ② `sources()` 返回快照，元素形如
  `{n, source, source_key, chunk_index, page, sheet, slide, title, preview(前120字), score, score_kind}`，
  前端按 `n` 渲染引用卡片、点击跳转。

工具里如果拿不到注册表（比如脱离 Agent 直接调用工具），会退回模块级 `_fallback_registry`，
保证「不会因为少一个 config 就崩」。

### 1.4 提示词：把「只能引用已有编号」写成硬约束

`core/prompts.py · SYSTEM_PROMPT` 的四条引用规则（原文）：

1. 资料前有系统分配的编号，引用事实/数字/流程/责任人时**必须在句末标注编号**，如「年假为每年 10 天 [1]。」；
2. **只能引用检索结果里出现过的编号，禁止编造编号、文件名或页码**；
3. 多个来源支持同一结论可写 `[1][2]`；
4. 检索结果不足以回答时直接说「知识库中没有找到相关内容」，**不要用常识补全**。

配套参数：`temperature=0.7`、`max_tokens=2048`、`streaming=True`；
`create_agent(model=llm, system_prompt=SYSTEM_PROMPT, tools=[search_knowledge_base, send_email])`。
另外系统提示里还有「回答任何公司制度/流程/数据的问题前必须先调用检索工具」——
这条让模型不会跳过检索直接凭参数记忆瞎答。

### 1.5 流式：先发 sources，再发 token

`ask_stream()` 的顺序有讲究：

```python
for chunk, _metadata in _agent.stream(..., stream_mode="messages", config=config):
    if not isinstance(chunk, AIMessageChunk):
        continue                            # ① 工具结果（ToolMessage）不是答案，必须跳过
    text = _message_text(chunk)
    answer_parts.append(text)
    if not sources_sent and len(registry) > 0:
        sources_sent = True
        yield "sources", registry.sources() # ② 检索一完成，先把来源发给前端
        yield from flush_tokens()           # ③ 再把之前缓冲的 token 补发
    if not sources_sent:
        buffered.append(text)               # ④ 工具还没执行，先不发 token
    else:
        yield "token", text
```

- ① 消息流里混着工具返回的资料原文，直接推给前端会把「资料」当成「答案」显示；
- ②③ 模型可能先输出「我先查一下」再调工具，此时注册表还是空的；缓冲到工具执行完，
  先发 sources 再补发缓冲 token，保证前端拿到来源时编号已经确定；
- ④ 如果模型从头到尾没调工具，结束时统一发一个空的 `sources` 再 flush，退化成一次性输出但绝不串号。

> ⚠️ **读代码时发现的一处前后端约定不一致**（不要照抄当作「正确设计」）：
> 后端 `ask_stream` 的 token 事件推的是纯文本（`yield "token", text`），路由层 `json.dumps` 后
> 实际是 JSON 字符串 `"年假"`；而 `routes_qa.py` 注释写的是「token 事件格式 `{"text": "..."}`」，
> 前端 `static/index.html · handleSSE` 也按对象取 `data.text`。三者对不上，
> 流式模式下前端可能显示不出 token。正确修法是后端统一包成 `{"text": ...}`（本项目不许改代码，
> 这里只记录事实）。

### 1.6 对话历史：只回传 role/content

`core/agent.py` 维护进程内 `_history`（列表），每轮问答追加 user + assistant 两条，最多保留 40 条 = 20 轮：

```python
def _history_messages(question: str) -> list[dict[str, str]]:
    messages = [{"role": item["role"], "content": item["content"]} for item in _history]  # ① 只取两个字段
    messages.append({"role": "user", "content": question})
    return messages

# _remember()：追加本轮问答；超过 _MAX_HISTORY=40 就 _history = _history[-40:]（只留最近 20 轮）
```

- ① 工具消息（ToolMessage）和引用元数据不回灌上下文：既省 token，也避免模型把「上次检索到的资料」
  当成这轮的新资料；
- ② 历史是**进程内**的，多用户会串、重启会丢——当前实现不引入数据库，这个局限在已知限制里写明。

注意 `ask()` 出错时**不写历史**（`except` 分支直接 return）：否则「抱歉，处理出错」会被当成
上一轮回答塞进下一轮上下文，模型可能顺着错误文案继续瞎编。

### 1.7 `_extract_answer` 修掉的那个运算符优先级 bug

旧代码想「dict 用 role、消息对象用 type」，写成一行
`a or b if cond else c`——Python 会解析成 `(a or b) if cond else c`，两个分支逻辑不一致：
对 dict 永远先看 `role` 再退回 `type`，对 Message 对象只看 `type`。一旦某个消息对象也有 `role` 属性，行为就漂移。
现在拆成显式 if/else（`_message_role`）。`_message_text` 同理兼容 `content` 是 `str` 或
`[{"type":"text","text":...}]` 列表两种形态，否则取答案会拿到空串。

## 2. 设计取舍与备选方案

### 反例：常见错误写法

```python
# 错版一：让模型自己写来源
prompt = "回答后请注明来源文件名和页码。"
# 错版二：全局单例注册表
REGISTRY = CitationRegistry()          # 所有提问共用
# 错版三：把检索分拼进上下文
context += f"[{i}] score={hit['score']}\n{hit['content']}\n"
```

### 反例的问题

1. **让模型自由写来源，必然编**：它会把「员工手册.pdf」写成「员工手册.docx」、页码凭感觉给，
   前端也无法跳转。程序分配编号后，模型的任务从「记来源」降级为「引编号」，出错空间小得多。
2. **全局注册表会串轮**：第二轮提问的 `[1]` 指向上轮的第一条来源，用户点开是旧文档；
   现在每轮 `ask()`/`ask_stream()` 都新建注册表。
3. **把分数给模型看**：模型看到 0.03 会以为不相关而拒答，明明检索到了却说没找到。
4. **用 contextvars 传注册表**：在 LangGraph 的工具执行上下文里拿不到值，`sources` 恒为空
   ——这个 bug 只有真正跑过端到端才会暴露，是值得记录的排查案例。
5. **不做 `AIMessageChunk` 过滤**：工具返回的资料会混进 token 流，前端把原文当答案逐字打印。
6. **工具抛异常**：用户看到 500 和栈；现在异常转成字符串，由模型用中文解释怎么修配置。

## 3. 常见问题（FAQ）

1. **「回答里的 `[1]` 是谁分配的，怎么保证和 sources 一一对应？」**
   → 程序分配：`format_numbered_context` 按检索结果顺序调 `register()` 拿 1 基编号，
   同时写入 `sources`；模型只被允许引用这些编号，前端按 `n` 渲染。
   编号 key 是 `(source_key, chunk_index)`，重复登记幂等，`tests/test_citations.py` 覆盖了顺序/幂等/不合并。

2. **「注册表怎么传进工具函数？为什么不用全局变量或 contextvars？」**
   → 走 `invoke(..., config={"configurable": {"citation_registry": registry}})`，
   工具声明 `RunnableConfig` 参数由框架注入；全局变量会串轮，contextvars 在 LangGraph 工具上下文里取不到
   （曾经导致 sources 恒空），config 通道还天然隔离并发请求。

3. **「流式为什么先发 sources 再发 token？token 为什么要缓冲？」**
   → 模型可能先说话再调工具，过早发 token 会导致 `[1]` 出现时来源还没定；
   缓冲到注册表非空再「先 sources、后补 token」，保证编号与来源同帧到达；
   同时必须过滤非 `AIMessageChunk` 的消息，避免工具资料混进答案流。