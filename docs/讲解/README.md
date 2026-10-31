# 项目一 · 教学讲解（目录索引）

> 读者：基础薄弱、英文阅读吃力、要在 1-2 周内把项目**完全讲懂并能应对面试追问**的本科生。
> 全部中文，术语第一次出现给中英对照；不贴大段代码，只贴 10-25 行关键片段并逐段解释。
>
> 所有数字都来自项目里的真实代码和评测报告（`eval_results/local.md`），
> 没有编造；引用代码一律写「相对路径 · 函数名」。

## 目录

| 文件 | 一句话摘要 | 主要代码 |
| --- | --- | --- |
| [00-总览与数据流.md](./00-总览与数据流.md) | 上传入库 / 检索 / 问答三条调用链 + ASCII 架构图 + 为什么分 core 和 app 两层 | `main.py`、`app/api.py` |
| [01-embeddings.md](./01-embeddings.md) | 本地 bge-small-zh-v1.5（512 维）怎么加载与缓存、为什么最后把云端 embedding 分支删掉（含当时的对比数据） | `core/embeddings.py` |
| [02-loader-splitter.md](./02-loader-splitter.md) | 7 种格式怎么解析、为什么按 token 切、句级 overlap、metadata 每一项为什么必须有 | `core/loader.py`、`core/splitter.py` |
| [03-store-bm25.md](./03-store-bm25.md) | Chroma + BM25 为什么由一个 store 管、jieba 分词、索引缓存与重建时机、同增同删的坑 | `core/store.py`、`core/bm25.py` |
| [04-retriever-rrf.md](./04-retriever-rrf.md) | RRF 公式与手算例子、去重 key 为什么不能只用 content、score_kind、为什么融合分不能当阈值 | `core/retriever.py` |
| [05-citations-agent.md](./05-citations-agent.md) | `[n]` 编号怎么分配和回填、create_agent 工具返回格式、提示词怎么约束「只能引已有编号」 | `core/citations.py`、`core/agent.py`、`core/prompts.py` |
| [06-api-upload.md](./06-api-upload.md) | 上传加固 4 点（size→413、1MB 流式、边写边算 SHA256 幂等、zip 炸弹）+ SSE 三段实现 | `app/routes_documents.py`、`app/upload_utils.py`、`app/routes_qa.py` |
| [07-eval.md](./07-eval.md) | 标注集怎么设计、HitRate@5 / MRR / P95 公式与手算、双口径、天花板效应与 Roadmap | `eval/eval.py`、`eval/report.py` |

## 建议阅读顺序（约 7 天，每天 1-2 小时）

```
第 1 天  00-总览与数据流        ← 先建立全局图，后面每篇都回到这张图定位
第 2 天  01-embeddings          ← 知道「向量是什么、本地模型怎么加载、为什么不用云端」
第 3 天  02-loader-splitter     ← 检索的原料是怎么来的（重点：token 预算、metadata）
第 4 天  03-store-bm25          ← 两套索引为什么必须同步（高频追问点）
第 5 天  04-retriever-rrf       ← 必须能自己手算一遍 RRF 例子
第 6 天  05-citations-agent     ← 引用编号 + Agent 工具 + 流式，面试最容易被追问
第 7 天  06-api-upload + 07-eval ← 上传安全与评测数字，最后整体复述一遍三条链路
```

如果只有 2-3 天：`00 → 04 → 05 → 07`，先保证「能讲清检索和引用、能解释评测数字」。
时间够再补 02/03/06 的实现细节。

## 每篇的固定结构

1. **`## 1. 这段代码在做什么`**：关键代码片段（10-25 行）+ 逐段注释「这行在做什么 / 为什么」；
2. **`## 2. 为什么这样写（对比错写法）`**：先给一个新手会写的错版本，再逐条说明问题；
3. **`## 3. 面试官可能怎么问`**：3 个问题，每题一句话答案 + 关键数字/依据。

建议用法：先看第 1 节自己复述代码在干嘛；遮住第 2 节，先自己想「为什么不能那样写」；
最后用第 3 节自测，答不上来说明这篇没过。

## 必须记住的数字（面试自测）

| 数字 | 含义 |
| --- | --- |
| 512 | bge-small-zh-v1.5 的向量维度（也是模型的 max_position_embeddings） |
| 500 + 50 | 切块 token 预算 + 句级 overlap（bge 上限 512，留余量） |
| 10 份 / 54 块 / 20 题 | 语料文档数、切块后块数、标注集题数 |
| 60 | RRF 平滑常数 k；第 1 名贡献 1/60≈0.0167，两路都第 1 最多 0.0333 |
| 50 MB / 1 MB | 单文件上传上限 / 流式落盘分块大小 |
| 100:1 / 500 MB / 10000 | zip 炸弹的压缩比 / 解压总量 / 条目数阈值 |
| 18.033 / 20 | MRR 的分子和分母（得 0.9017） |
| 0.9017 | MRR（主口径）；HitRate@5 100%（20/20） |
| 9.0ms / 21.5ms | 检索 P50 / P95（服务预热后；机器忙时会到 100ms+） |
| `[n]` ≤ 5 | 一次回答最多 5 条引用（top_k=5） |

## 已知局限与文档不一致（诚实清单）

- **HitRate@5 100% 不可当成卖点**：54 块、20 题导致指标饱和（天花板效应），
  真正有信息量的是 MRR（0.9017）和分题型 MRR（paraphrase 0.7400 最弱）；真实场景需扩语料后再测（见 07 篇）。
- **P95 取不到最大值**：n=20 时 P95 取第 19 个，逐题里最慢的一条（21.7ms）不会进统计，长尾被低估。
- **流式 token 事件的前后端约定不一致**：后端推纯文本、路由层 `json.dumps` 成 JSON 字符串，
  而 `app/routes_qa.py` 注释和前端 `handleSSE` 都按 `{"text": ...}` 对象取值（详见 05 篇 1.5 的 ⚠️）。
  本项目只写文档不改代码，这里记录事实，面试时可作为「读代码发现契约问题」的加分点。
- 评测只覆盖**检索**，不覆盖**生成质量**（答案对不对、引用是否支持结论），这是下一阶段的 Roadmap。

## 怎么自己验证这些说法

```powershell
# 1) 跑纯函数测试（不用起服务、不加载模型，几秒）
D:\Anaconda\envs\xhu\python.exe -m pytest tests -q

# 2) 看评测报告原文（数字都在这里）
type eval_results\local.md

# 3) 起服务后自己调检索接口（只读）
curl "http://127.0.0.1:8000/api/search?query=年假&k=5"
```

> 注：数字取自 `eval_results/local.md`（2026-10-26 重跑，54 块语料、20 题），
> 与 `eval/eval.py` 的输出格式完全一致；需要现场演示时按 `README.md` 启动服务再重跑。

## 相关文件

- 原始评测报告：[../../eval_results/local.md](../../eval_results/local.md)
- 语料与出题说明：[../../data/README.md](../../data/README.md)
- 测试用例（讲不清时直接看断言）：[../../tests/](../../tests/)