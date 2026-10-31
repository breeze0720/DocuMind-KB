# 知识库问答系统（RAG + Agent）

上传公司文档（7 种格式）→ 自动解析切块 → 混合检索（向量 + BM25 + RRF）→ 带**可点击引用**的回答 → 流式输出。
自带一套**检索评测台**：20 题人工标注集，任何检索改动都能量化前后差异。

**技术栈**：FastAPI · LangChain 1.x（`create_agent`）· DeepSeek · Chroma · fastembed（ONNX，CPU 可跑）· jieba + rank_bm25 · 原生 HTML/CSS/JS 前端（无构建链、离线可用）

---

## 1. 30 秒跑起来

```powershell
cd 01-知识库问答-RAG
D:\Anaconda\envs\xhu\python.exe main.py
# 浏览器打开 http://127.0.0.1:8000/
```

- `.env` 里填 `DEEPSEEK_API_KEY`（问答链路需要）；只做检索调试不需要任何 API Key。
- 嵌入模型需已下载到 `models\bge-small-zh-v1.5\`（91MB，ONNX）。首次执行 `.\download_model.ps1` 自动下载（走 `HF_ENDPOINT` 镜像）。
- 索引是运行产物、不入库：首次启动后调一次 `POST /api/reindex`（或把 `.env` 的 `AUTO_INDEX_CORPUS` 设为 true 让启动时自动导入），`data/corpus/` 下 10 份文档 → 54 个块。
- 依赖安装：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

## 2. 能力清单

| 能力 | 实现 |
|---|---|
| 多格式解析 | pdf / docx / md / txt / xlsx / pptx / csv，统一转 `Document`（source、页码、sheet、slide 都进 metadata） |
| 切块 | 按 token 预算装箱（目标 500 token，overlap 50），优先在句末断开 |
| 混合检索 | 向量 Top5 + BM25 关键词 Top5 → **RRF 融合**（k=60），去重 key = `(source, chunk_index)` |
| 引用溯源 | 编号 `[n]` 由**程序分配**并回填来源卡片，模型只能引用已存在的编号 |
| 上传加固 | 50MB 上限（先判 size 再读内容）、1MB 分块流式落盘、SHA256 幂等、zip 炸弹防护 |
| 流式回答 | SSE `/api/ask/stream`：`sources` → `token`* → `done` |
| 评测台 | 20 题（literal 10 / paraphrase 5 / short 5），输出 HitRate@5 / MRR / P50 / P95 |
| 本地嵌入 | bge-small-zh-v1.5（512d，ONNX/CPU，完全离线），进程内单例复用；检索链路（含 query 编码）P50 9.0ms；不依赖任何云端 embedding |

## 3. 实测数字（2026-10-26，20 题 / 54 个块 / 服务预热后）

| 指标 | 本地 bge-small-zh-v1.5 (512d) |
|---|---|
| HitRate@5 | **100%**（20/20） |
| MRR | **0.9017** |
| P50 / P95 检索延迟 | 9.0ms / **21.5ms** |
| 成本 | 0（本地 CPU，离线） |

分题型 MRR：literal 1.00 / short 0.87 / **paraphrase 0.74**（语义改写是短板 → Roadmap 上 rerank）。
报告原文：`eval_results/local.md`，结构化结果 `eval_results/latest.json`（重新生成只需跑一次 `eval/eval.py`）。
（历史对比：曾用百炼 text-embedding-v4 做过一轮 A/B，MRR 0.7833、P50 360ms，结论是本地更好，
那套双后端代码已删除；数据与取舍见 `docs/讲解/01-embeddings.md` §2.3。）

> 注：语料只有 54 个块，HitRate@5 已饱和，不能证明大规模下同样成立——这是主动承认的局限。指标口径是"前 5 条里出现标注来源 **且** 包含答案关键词"（强命中），回复里另有纯关键词命中作为辅助口径；延迟会随机器负载抖动。

## 4. 目录结构

```
01-知识库问答-RAG/
├── main.py                 # 应用入口（FastAPI + lifespan + 托管 static/）
├── config.py               # pydantic-settings 配置（切块/上传/检索参数、模型路径）
├── core/                   # 纯业务层（不依赖 FastAPI，可单测）
│   ├── embeddings.py       # 本地 bge 加载（ONNX）+ 进程级单例 + 状态报告
│   ├── loader.py           # 7 种格式解析 → Document
│   ├── splitter.py         # token 预算切块 + 句级 overlap
│   ├── store.py            # Chroma 封装 + BM25 同增同删 + 缓存落盘
│   ├── bm25.py             # jieba 分词 + rank_bm25 封装
│   ├── retriever.py        # 向量/关键词两路召回 + RRF 融合（fuse_rrf 纯函数）
│   ├── citations.py        # [n] 编号分配与来源回填
│   ├── agent.py            # create_agent + 检索工具 + 提示词约束
│   ├── prompts.py          # 系统提示词
│   └── runtime.py          # 单例装配（store/retriever 的懒加载）
├── app/                    # HTTP 层（路由 + 请求/响应模型 + 上传校验）
├── eval/                   # eval.py 跑评测、report.py 生成 md/json
├── tests/                  # 43 个 pytest 用例（纯函数 + 上传校验 + 评测口径）
├── static/index.html       # 单文件前端：对话 / 检索调试 / 评测 三个页签
├── data/corpus/            # 10 份虚构公司文档（6 种格式），可直接演示
├── data/_generator/        # 语料生成脚本（可复现）
├── docs/讲解/              # 逐模块中文讲解（做了什么 / 为什么这么写 / 面试怎么问）
├── models/bge-small-zh-v1.5/  # 本地嵌入模型（ONNX，91MB，不入 git）
└── eval_results/           # 评测报告 local.md + latest.json（前端读）
```

## 5. API 速查

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 文档数、块数、embedding 模型/维度、模型是否就绪 |
| POST | `/api/upload` | 上传入库（form-data：`file`），返回 `added/duplicate/chunks` |
| GET | `/api/documents` | 文档列表（来源、块数、大小） |
| DELETE | `/api/documents?source=xxx` | 按来源删除（向量库 + BM25 同步删） |
| POST | `/api/documents/delete` | 按来源删除（POST 形式，兼容不支持 DELETE 的客户端） |
| POST | `/api/reindex` | 重建索引（`?force=true` 强制重算全部块） |
| POST / GET | `/api/search` | 只检索不生成：返回 rank/score/score_kind/source/preview（**调试用，不花钱**） |
| POST | `/api/ask` | 一次性问答，返回 answer + sources |
| POST / GET | `/api/ask/stream` | SSE 流式问答（form-data：`question`） |
| POST | `/api/session/reset` | 清空对话记忆 |
| GET | `/api/eval/summary` | 最近一次评测的指标与逐题明细 |
| GET | `/api/health` | 存活探针（进程可用性） |

完整交互式文档：`http://127.0.0.1:8000/docs`。

## 6. 评测怎么做

```powershell
python eval/eval.py --tag local --base-url http://127.0.0.1:8000   # 结果写 eval_results/local.md + latest.json
python eval/eval.py --tag after --base-url http://127.0.0.1:8000   # 改完参数再跑一次，两份报告并排比
```

- 标注集 `eval/eval_set.jsonl` 每行：`{q, kw[], source, type}`，type ∈ `literal|paraphrase|short`；
- 指标：`HitRate@5`（前 5 有无正确来源）、`MRR`（正确来源排名的倒数和均值）、`P50/P95` 延迟百分位；
- 换切块参数 / RRF 的 k，跑一次就能看到涨跌——**这是这个项目最有价值的部分**。

## 7. Roadmap（按优先级）

1. **扩语料**到 200+ 份并加入"同主题干扰文档"，把饱和的 HitRate 压到真实水平；
2. 加 **cross-encoder rerank**，对比 `paraphrase` 题 MRR（当前 0.74）能涨多少；
3. 增量更新：按 chunk diff 而非整文件替换；
4. 引用一致性抽检 → 量化幻觉率；
5. 需要多 worker 时把 Chroma 换服务端模式（当前嵌入式，单进程最简）。

## 8. 已知限制

- 语料规模小，指标饱和（见 §3 注）；
- embedding 固定本地 bge-small-zh-v1.5，换模型需要重建索引（向量空间不同，不能混写）；
- 单进程部署（Chroma 嵌入式 + 进程内单例）；
- BM25 索引随文档变化重建，超大知识库需要换成 ES/OpenSearch；
- 问答质量依赖 DeepSeek API 可用性（检索链完全本地，不受影响）。