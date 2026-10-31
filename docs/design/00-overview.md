# 00 · 总览与数据流

> 这一篇不讲细节，只解决一个问题：**别人问你「这个项目怎么跑的」，你能从 HTTP 请求一路讲到向量库**。
> 后面每一篇都对应这里的一个环节；读的时候随时回到这张图定位。
>
> 术语约定（第一次出现给中英对照）：**RAG**（Retrieval-Augmented Generation，检索增强生成）、
> **embedding**（向量化 / 嵌入）、**chunk**（文本块）、**BM25**（一种关键词检索打分算法）、
> **RRF**（Reciprocal Rank Fusion，倒数排名融合）、**SSE**（Server-Sent Events，服务端推送）。

## 1. 实现概览

### 1.1 一张图看全系统

```
                        浏览器 static/index.html
                                  │  fetch: 上传 / 检索 / 问答(SSE)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ main.py  （应用装配：FastAPI app + lifespan 预热 + 挂路由和静态目录） │
│   └─ app/api.py  api_router = documents ∪ qa ∪ system                │
└─────────────────────────────────────────────────────────────────────┘
      │                          │                          │
      ▼                          ▼                          ▼
┌───────────────┐        ┌───────────────┐          ┌───────────────┐
│ app/routes_   │        │ app/routes_qa │          │ app/routes_   │
│ documents.py  │        │ .py           │          │ system.py     │
│ 上传/列表/删除 │        │ 检索/问答/SSE │          │ 状态/评测结果 │
└───────┬───────┘        └───────┬───────┘          └───────┬───────┘
        │ app/ingest.py          │                          │
        │ (入库编排+幂等)         │                          │
        ▼                        ▼                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ core/ （纯 RAG 业务，不认识 FastAPI）                                 │
│  loader → splitter → embeddings → store(Chroma+BM25)                 │
│  retriever(RRF) → citations → agent(LangChain create_agent+DeepSeek) │
│  runtime.py = 进程级单例门面（embedding/store/retriever 各一份）     │
└─────────────────────────────────────────────────────────────────────┘
      │                 │                    │
      ▼                 ▼                    ▼
 本地 ONNX 模型     chroma_db/（向量+BM25 缓存 JSON）    外部 API：
 bge-small-zh      kb_bge_512 + bm25_docs_kb_bge_512.json  DeepSeek（LLM）
```

三层各管一件事，**依赖方向永远向下**：`app/` 可以 import `core/`，反过来不行。
`config.py` 在最底层，谁都能读，但只有它自己解析 `.env`。

### 1.2 主线一：上传入库（写路径）

请求 `POST /api/upload`，按顺序经过这些函数：

| 步骤 | 文件 · 函数 | 干了什么 |
| --- | --- | --- |
| 1 | `app/routes_documents.py · upload_document` | 收 `UploadFile`，先洗文件名 |
| 2 | `app/upload_utils.py · check_extension` | 扩展名白名单（7 种），不过就 400 |
| 3 | `app/upload_utils.py · check_size` | 看 `file.size`，超大**不读内容**直接 413 |
| 4 | 同函数内循环 `await file.read(1024*1024)` | 1MB 分块流式落盘，边写边算 SHA256，累计超限立刻 413 |
| 5 | `app/upload_utils.py · zip_bomb_reason` | docx/xlsx/pptx 是 zip，只读中央目录判压缩比/总量/条目数 |
| 6 | `core/store.py · find_source_by_hash` | 内容哈希已存在 → 直接返回「重复」，不解析不重复向量化 |
| 7 | `app/ingest.py · index_file` | 入库编排：查重 → 解析 → 切块 → 写库 |
| 8 | `core/loader.py · load_documents` | 按扩展名分派 parser，产出带 `page/sheet/section` 的单元 |
| 9 | `core/splitter.py · split_documents` | 按 token 预算切块，补 `chunk_index/file_hash/title` |
| 10 | `core/store.py · add_documents` | 算向量 → Chroma `upsert` → 重建 BM25 → 写 JSON 缓存 |

关键点：**同一份文件传两次不会变成两份**。第 6 步按内容哈希判重，第 10 步用
`{file_hash[:16]}:{source}:{chunk_index}` 做确定性 id，双保险。

### 1.3 主线二：检索（读路径，不碰大模型）

`POST /api/search` → `app/routes_qa.py · search` → `core/runtime.py · get_retriever()`
→ `core/retriever.py · HybridRetriever.search`：

1. 每路先取 `pool = max(k*4, 20)` 个候选（默认 k=5 → 20）；
2. 向量路：`core/store.py · search` 走 Chroma，余弦相似度 0~1；
3. 关键词路：`core/store.py · keyword_search` 走内存 BM25（jieba 分词）；
4. `core/retriever.py · fuse_rrf` 用 `1/(60+rank)` 融合两路排名；
5. 取前 k 条，补上 `source/chunk_index/page/score_kind` 返回。

这条链**故意不经过 LLM**：既能做实时演示，也是 `eval/eval.py` 的评测入口，
所以评测数字便宜、可复现（20 题跑完不到 1 秒，检索 P95 21.5ms）。

### 1.4 主线三：问答（Agent 路径）

`POST /api/ask` → `core/agent.py · ask`：

```
问题 → CitationRegistry(本轮新注册表)
     → create_agent.invoke(messages, config={"configurable": {"citation_registry": registry}})
        → LLM 决定调用工具 search_knowledge_base(query)
           → get_retriever().search() → format_numbered_context()  ← 这里给每块分配 [1][2]…
        → LLM 读带编号的资料，写回答并标注 [1]
     → 返回 (answer, registry.sources())
```

流式版 `ask_stream()` 把事件拆成 `sources → token… → done`（见 05、06 两篇），
前端先拿到来源卡片、再逐字出答案。

`main.py` 里的装配代码值得单独看一眼，因为它解释了「启动慢/启动失败」为什么这么处理：

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = config.settings
    for directory in (settings.upload_dir, settings.chroma_dir, settings.data_dir, settings.eval_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)   # ① 四个数据目录先建好，后面代码不用判断存在性
    await run_in_threadpool(init_runtime)                    # ② 预热本地模型/读索引缓存，阻塞操作丢线程池
    if settings.auto_index_corpus:                           # ③ 默认 False：启动时不扫语料，避免大语料把启动拖死
        try:
            result = await run_in_threadpool(index_corpus)
        except Exception as exc:
            print(f"[启动] 语料自动导入失败（服务继续启动）: {exc}")
    try:
        init_agent()                                         # ④ 构造 ChatOpenAI 不联网，失败也不拦启动
    except Exception as exc:
        print(f"[启动] Agent 初始化失败（问答接口会返回错误提示）: {exc}")
    yield
```

逐段解释：
- ① 配置里所有目录都是「可能还不存在」的，统一在建进程时创建，**任何请求路径都不用再 `mkdir`**；
- ② `run_in_threadpool` 是必须的：加载约 90MB 的 ONNX 模型要 1-2 秒，直接 `await` 会卡住事件循环；
- ③ 默认关闭是产品决策——用户打开页面时更关心「页面有没有响应」，而不是「语料导入完没有」；
- ④ 初始化失败**只打印不抛**，因为用户还需要能打开 `/api/status` 看到中文原因（比如模型文件没下载），
  而不是面对一个连不上的端口。

### 1.5 一次问答的请求/响应长什么样

非流式接口 `POST /api/ask` 的响应体由 `app/routes_qa.py · ask_question` 组装，结构是固定的
（下面是**字段示意**，具体文案和数值不是某次真实请求的截图）：

```json
{"success": true, "question": "……", "answer": "…… [1]。",
 "sources": [{"n": 1, "source": "员工手册.docx", "chunk_index": 7, "page": 3,
              "title": "……", "preview": "……", "score": 0.0325, "score_kind": "rrf"}],
 "elapsed_ms": 1832.4}
```

四个字段各有分工：`answer` 是给用户看的文本；`sources` 是给前端渲染引用卡片的数组（顺序 = 编号顺序）；
`elapsed_ms` 是接口耗时（`time.perf_counter()` 差值）；`success` 让前端能统一处理失败分支。
`sources` 里带 `score_kind: "rrf"` 是为了让 UI 知道这个 0.03 不是「3% 相关度」（见 04 篇）。

### 1.6 关键配置速查（`config.py`）

| 配置项 | 默认值 | 影响哪条链 |
| --- | --- | --- |
| `max_file_mb` | 50 | 上传：单文件上限，超了 413 |
| `chunk_tokens` / `chunk_overlap` | 500 / 50 | 入库：切块预算，受 bge 512 token 上限约束 |
| `collection_name` | `kb_bge_512` | 检索：Chroma collection 名（维度写在名字里） |
| `top_k` | 5 | 检索：返回条数，也是引用编号上限 |
| `rrf_k` | 60 | 检索：融合平滑常数 |
| `max_zip_ratio` / `max_decompressed_mb` | 100 / 500 | 上传：zip 炸弹阈值 |
| `auto_index_corpus` | `False` | 启动：是否自动扫描 `data/corpus` |

配置用 pydantic-settings 而不是散落的 `os.getenv`：进程启动时一次性做类型转换和校验
（`"50"` 与 `50` 的区别当场报错，而不是跑一半才炸），并支持 `.env` + 环境变量覆盖，
所以换端口/换语料目录只需要在 `.env` 改一行或启动时设环境变量，不用动代码。同时保留了
`config.MODEL_NAME` 这类模块级常量，兼容第一天写的旧脚本。

## 2. 设计取舍与备选方案

### 反例：常见错误写法

一个几百行的 `main.py`：路由、上传、解析、切块、Chroma、BM25、RRF、Agent 工具全在一个文件里；
再加一个 300+ 行的 `vector_store.py` 把「embedding + 向量库 + 关键词检索 + 融合 + 删除同步」全塞进去。
本项目**改造前就是这个结构**（现在根目录的 `agent.py` / `vector_store.py` / `document_loader.py`
只剩兼容 shim，真正的实现都在 `core/`）。

### 具体问题

1. **改一处要读全文件**：想调 RRF 的 k，得在几百行里找；想换切块参数，混在上传逻辑里。
2. **测试碰不到核心逻辑**：纯函数（token 估算、RRF、命中判定）埋在路由后面，写 pytest 只能起服务、
   传真文件、加载约 90MB 的模型。现在 `tests/` 里 6 个测试文件、43 个用例都不加载真实模型
   （`store` 用 4 维假 embedding），跑一次几秒。
3. **换模型要动全网**：本项目固定本地 bge，模型加载/维度/collection 名都收在
   `core/embeddings.py` + `core/runtime.py` 两个文件里，换模型只改配置和重建索引；
   当年为了做 local vs 云端 A/B 做的那套「按后端缓存 + 双 collection」在对比结束后已删除
   （数据与取舍见 01 篇）。
4. **同步删除容易漏**：向量库和 BM25 是两套数据，写操作分散在多个文件时，「删了 Chroma 忘了删 BM25」
   必然发生。现在所有写操作只经过 `KnowledgeStore`，删除在同一个 `RLock` 里同步两边（见 03 篇）。
5. **上传安全无处安放**：大小/zip 炸弹/幂等本质是「输入 → 是否拒绝」的纯函数，
   抽到 `app/upload_utils.py` 后可以构造极端输入直接测（`tests/test_upload_utils.py` 里
   2MB 全零字节模拟压缩比 100:1 的炸弹），不用真传 500MB。

### 分层判据（一句话）

> **HTTP 的事（参数解析、状态码、SSE 编码）留在 `app/`；RAG 的事（解析、切块、向量、检索、引用）放 `core/`；
> 装配和生命周期放 `main.py`。** 老入口用 shim 转发，保证旧脚本 `import` 不坏。

### 旧代码怎么办：兼容 shim

拆文件最大的风险是「用户以前的脚本 `from vector_store import hybrid_search` 直接报 ImportError」。
所以根目录保留三个只有几十行的转发文件：`agent.py`（转发 `init_agent/ask/ask_stream/clear_history`）、
`vector_store.py`（转发 `get_store/get_retriever` 的旧函数名）、`document_loader.py`（把
`load_documents + split_documents` 包成旧的 `load_and_split`）。

注意 `load_and_split(path, chunk_size)` 的参数名为了兼容没改，但**语义已经从「字符」变成「token」**
——这是拆分层时容易忽略的隐性变更，读旧代码时不要被参数名骗了。

### 怎么跑起来（自测）

```powershell
python main.py                                  # 默认 8000
python -m uvicorn main:app --port 8010          # 联调/自测
python -m pytest tests -q                       # 纯函数测试，不加载模型
curl -X POST "http://127.0.0.1:8010/api/reindex"                        # 导入语料
python eval\eval.py --tag local --base-url http://127.0.0.1:8000
```

## 3. 常见问题（FAQ）

1. **「用户点上传到能问答，中间发生了什么？」**
   → 七步：白名单→大小预检→1MB 流式落盘+算 SHA256→zip 炸弹检查→哈希幂等判重→
   `index_file` 解析切块→`add_documents` 写 Chroma 并重建 BM25；关键数字：50MB 单文件上限、1MB 分块、500 token 切块、54 块语料。

2. **「为什么分 core 和 app 两层，而不是一个文件写完？」**
   → 让「HTTP 关注点」和「RAG 关注点」解耦：`core` 不 import FastAPI 才能被 pytest 直接测
   （6 个测试文件、43 个用例，全部纯函数级）；代价是多一层目录，但换来可测试性。

3. **「服务启动时模型加载失败会怎样？」**
   → 不会起不来：`lifespan` 里 `init_runtime()` 失败只 warning，服务继续可用，
   `/api/status` 会返回 `embedding.available=false` 和中文原因；问答接口返回错误提示而不是异常栈。