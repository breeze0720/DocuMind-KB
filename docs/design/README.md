# 设计文档 · 知识库问答 RAG Agent

记录项目一各模块的实现方式与设计取舍。文中所有代码位置、配置值与实测数字都来自本仓库
（`config.py`、`eval_results/local.md`），可直接对照源码核验。

- 只描述代码中**真实存在**的行为：函数名、表名、阈值、指标均可回溯；
- 每篇固定三段：`1. 实现概览` → `2. 设计取舍与备选方案`（先给反例，再逐条分析） → `3. 常见问题（FAQ）`；
- 术语首次出现给中英对照；代码片段控制在 10–25 行。

## 文档列表

| 文件 | 内容摘要 | 主要代码 |
| --- | --- | --- |
| [00-overview.md](./00-overview.md) | 上传入库 / 检索 / 问答三条调用链、整体架构图、core 与 app 的分层依据 | `main.py`、`app/api.py` |
| [01-embeddings.md](./01-embeddings.md) | 本地 bge-small-zh-v1.5（512 维）的加载与单例复用、为什么只保留本地后端（含当时的 A/B 数据） | `core/embeddings.py`、`core/runtime.py` |
| [02-loader-splitter.md](./02-loader-splitter.md) | 7 种格式解析、按 token 预算切块、句级 overlap、metadata 各字段的作用 | `core/loader.py`、`core/splitter.py` |
| [03-store-bm25.md](./03-store-bm25.md) | Chroma 与 BM25 由同一个 store 管理、jieba 分词、索引缓存与重建时机、同步删除 | `core/store.py`、`core/bm25.py` |
| [04-retriever-rrf.md](./04-retriever-rrf.md) | RRF 公式与手算示例、去重 key 的选取、`score_kind` 的量纲约定、阈值只作用于向量路 | `core/retriever.py` |
| [05-citations-agent.md](./05-citations-agent.md) | `[n]` 编号的分配与回填、工具返回格式、提示词约束、SSE 先发 sources 的顺序保证 | `core/citations.py`、`core/agent.py`、`core/prompts.py` |
| [06-api-upload.md](./06-api-upload.md) | 上传四道防线（体积 → 1MB 流式落盘 → SHA256 幂等 → zip 炸弹）与 SSE 三段实现 | `app/routes_documents.py`、`app/upload_utils.py`、`app/routes_qa.py` |
| [07-eval.md](./07-eval.md) | 标注集设计、HitRate@5 / MRR / P95 公式与手算、双口径、天花板效应 | `eval/eval.py`、`eval/report.py` |

## 阅读顺序

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07
```

- `00` 建立全局调用链，后面每篇都回到这张图定位；
- `01` / `02` 是输入侧：向量化、解析与切块（决定检索原料的质量）；
- `03` / `04` 是存储与检索：两套索引同步、混合召回与融合；
- `05` / `06` 是输出侧与 HTTP 层：引用编号、流式与上传加固；
- `07` 是量化的部分：指标定义、口径选择与局限。

## 文档结构

1. `## 1. 实现概览`：关键代码片段（10–25 行）+ 逐段说明「这段在做什么 / 为什么」；
2. `## 2. 设计取舍与备选方案`：先给常见错误写法，再逐条说明问题所在；
3. `## 3. 常见问题（FAQ）`：每个问题一句话结论 + 关键数字或代码依据。

## 关键指标

| 数字 | 含义 |
| --- | --- |
| 512 | bge-small-zh-v1.5 的向量维度（也是模型的 `max_position_embeddings`） |
| 500 + 50 | 切块 token 预算 + 句级 overlap（模型上限 512，留余量） |
| 10 份 / 54 块 / 20 题 | 语料文档数、切块后块数、标注集题数 |
| 60 | RRF 平滑常数 k；第 1 名贡献 1/60≈0.0167，两路都第 1 最多 0.0333 |
| 50 MB / 1 MB | 单文件上传上限 / 流式落盘分块大小 |
| 100:1 / 500 MB / 10000 | zip 炸弹的压缩比 / 解压总量 / 条目数阈值 |
| 18.033 / 20 | MRR 的分子和分母（得 0.9017） |
| 0.9017 | MRR（主口径）；HitRate@5 100%（20/20） |
| 9.0ms / 21.5ms | 检索 P50 / P95（服务预热后；机器忙时会更高） |
| `[n]` ≤ 5 | 一次回答最多 5 条引用（`top_k=5`） |

## 已知局限

- **HitRate@5 100% 不代表大规模下可用**：54 块、20 题导致指标饱和（天花板效应），
  信息量更大的是 MRR（0.9017）与分题型 MRR（paraphrase 0.7400 最弱）；真实场景需扩语料后重测（见 07 篇）。
- **P95 取不到最大值**：n=20 时 P95 取第 19 个，逐题里最慢的一条（21.7ms）不进统计，长尾被低估。
- **流式 token 事件的前后端约定不一致**：后端推纯文本、路由层 `json.dumps` 成 JSON 字符串，
  而 `app/routes_qa.py` 注释与前端 `handleSSE` 都按 `{"text": ...}` 取值（详见 05 篇 1.5 的 ⚠️）。
  当前以文档形式记录，是后续需要修掉的契约问题。
- 评测只覆盖**检索**，不覆盖**生成质量**（答案正确性、引用是否支撑结论），属于下一阶段工作。

## 如何验证文档中的说法

```powershell
# 1) 纯函数测试（不需要起服务、不加载模型）
python -m pytest tests -q

# 2) 评测报告原文（所有指标数字都在这里）
type eval_results\local.md

# 3) 起服务后直接调检索接口（只读，不产生 LLM 费用）
curl "http://127.0.0.1:8000/api/search?query=年假&k=5"
```

> 数字取自 `eval_results/local.md`（2026-10-26 重跑，54 块语料、20 题），与 `eval/eval.py` 的输出格式一致；
> 需要复现时按根目录 `README.md` 启动服务后重跑 `eval/eval.py`。

## 相关文件

- 评测报告原文：[../../eval_results/local.md](../../eval_results/local.md)
- 语料与出题说明：[../../data/README.md](../../data/README.md)
- 测试用例（可直接对照断言）：[../../tests/](../../tests/)
