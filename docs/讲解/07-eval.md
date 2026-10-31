# 07 · 评测：标注集、指标公式与天花板效应

> 一句话：**没有基线数字，一切「优化」都无法证明有效**。这一篇讲清标注集怎么设计、
> HitRate@5 / MRR / P95 怎么算，以及为什么本项目 100% 的 HitRate 要主动承认「不可全信」。
>
> 术语：**benchmark**（评测基准）、**HitRate@K**（前 K 条里至少命中一条的题目占比）、
> **MRR**（Mean Reciprocal Rank，平均倒数排名）、**P95**（95 分位延迟）、**ground truth**（标注答案）、
> **ceiling effect**（天花板效应，指标已接近上限，无法区分优劣）。

## 1. 这段代码在做什么

### 1.1 标注集：20 条题，三种题型，覆盖 10 份文档

`eval/eval_set.jsonl` 每行一个 JSON，字段是 `q`（问题）、`kw`（必须出现的证据词）、
`source`（期望来源文件）、`type`（题型）：

| 题型 | 条数 | 设计目的 | 例子 |
| --- | --- | --- | --- |
| `literal` 原词 | 10 | 问题用词直接出现在原文，测「能不能检索到」 | 「P1故障要求多长时间内响应？」→ kw `15分钟` |
| `paraphrase` 同义改写 | 5 | 换一种说法，测语义检索（向量路的价值） | 「辞职时需要交接哪些内容？」→ 原文「离职交接 5 项」 |
| `short` 短问题 | 5 | 2~6 个字，测关键词检索（BM25 的价值） | 「报销时限」「试用期多久」 |

- 覆盖全部 10 份文档，每份都至少被一道题指向；
- 其中 2 条 paraphrase 是**跨文档**的：「员工手册要求辞职提前一个月，IT 部门几个工作日内回收账号？」
  答案在《IT运维手册.md》而不是《员工手册.docx》，用来测多跳；
- `kw` 都是在对应文档里用 python-docx / pypdf / openpyxl / python-pptx 抽出的原文里**逐字校验过**的
  （`data/README.md` 有说明），不是拍脑袋写的。

判定逻辑在 `eval/report.py`：

```python
def _source_match(case, result) -> bool:
    expected_sources = [s.lower() for s in case.get("sources") or []]
    source = str(result.get("source") or "").lower()
    return bool(expected_sources) and any(
        source == expected or expected in source or source.endswith(expected)   # ① 兼容全路径/文件名
        for expected in expected_sources)

def _keyword_match(case, result) -> bool:
    keywords = [k.lower() for k in case.get("keywords") or []]
    content = str(result.get("content") or "").lower()
    return bool(keywords) and any(k in content for k in keywords)               # ② 子串匹配

def match_case(case, results):
    has_keywords = bool(case.get("keywords"))
    for result in results:
        if not _source_match(case, result):
            continue
        if not has_keywords or _keyword_match(case, result):
            return True, int(result.get("rank") or 0)                           # ③ 强命中：来源 + 关键词
    return False, None
```

- ① 标注写文件名，但旧数据 source 可能是全路径，三种比较方式兜底；
- ② 关键词用子串匹配，英文统一小写，避免大小写导致漏判；
- ③ **主指标 = 强命中**：来源对 **且** 该块正文里有关键词，两个条件同时满足才算检索到答案。

### 1.2 三个指标的公式

```
HitRate@K = 命中的题目数 / 总题数
MRR       = (1/N) · Σ (1 / 该题第一次命中的排名)        # 没命中记 0
P95       = 排序后第 ceil(0.95 × N) 个延迟值            # 最近秩法（nearest-rank）
```

注意 `k=5` 与系统 `TOP_K=5` 对齐：命中排名必须 ≤5，第 6 名算没命中。

### 1.3 手算一遍（面试很可能让你算）

**MRR**：20 题里 20 题命中，逐题 RR 为
10 道 literal 全 1.0（共 10.0）、paraphrase 是 `1 + 0.5 + 1 + 1 + 0.2 = 3.7`、
short 是 `1 + 0.333 + 1 + 1 + 1 = 4.333`：

```
MRR = (10.0 + 3.7 + 4.333) / 20 = 18.033 / 20 = 0.9017
```

**P95**：20 个延迟排序后，`ceil(0.95 × 20) = 19`，取第 19 个 = **21.5ms**
（`eval/report.py · percentile` 里是 `index = ceil(p*n) - 1`）；
P50 取第 10 个 = **9.0ms**（逐题延迟从 6.6ms 到 21.7ms）。

⚠️ 有一条**没进 P95 的离群值**：`case-3` 21.7ms 是最大值，n=20 时 P95 取第 19 个，
最大值永远进不去，长尾被低估——这是小样本评测的固有缺陷，面试要主动讲。
（本机重跑过两次：MRR 两次都是 0.9017，P50/P95 在 9ms / 21ms 附近抖动，说明排序指标稳定、延迟随机器负载抖动。
历史记录里还出现过 P50 104ms / P95 185ms 的一次运行，那是机器被别的进程占满时的数字。）

### 1.4 双口径：主指标（强命中）vs 辅指标（仅关键词）

| 口径 | 判定 | 用途 |
| --- | --- | --- |
| 主（`hit`/`mrr`） | 来源匹配 **且** 正文含关键词 | 对外报告、给结论 |
| 辅（`kw_hit`/`kw_mrr`） | 正文含关键词，不管来源 | debug：区分「没检索到」和「检索到但来源排错」 |

为什么要分两套（`eval/report.py` 模块头注释写得很直白）：如果主指标取「来源 OR 关键词」的**并集**，
会出现「Top1 是别的文档、但关键词碰巧命中」也被算命中，**指标虚高、口径说不清**，
这正是改之前踩过的坑。`tests/test_eval_report.py::test_match_case_ignores_keyword_only_hit`
专门锁死这条：关键词命中但来源不同 → 主指标**不算**命中。

辅口径还能定位问题：`case-12`（一线城市住宿标准是否下调）local 主指标排名 2、kw 排名 1，
说明「关键词在第 1 块就出现了，但来源是会议纪要、不是报销制度」——属于多跳题的正常表现。

### 1.5 天花板效应：必须主动承认

报告里每个 Markdown 都会自动插入这段（`write_markdown`）：

> **天花板效应**：本套语料仅 54 个块、20 道题，检索指标已接近上限；
> 真实场景需扩大语料、加入同主题干扰文档后再复测。

事实是：HitRate@5 **就是 100%**，这个指标在 54 块的小语料上已经饱和，**它区分不出好坏**。
还有信息量的是 MRR（0.9017）和分题型 MRR：

| 题型 | 题数 | HitRate@5 | MRR |
| --- | --- | --- | --- |
| literal | 10 | 100% | 1.0000 |
| paraphrase | 5 | 100% | 0.7400 |
| short | 5 | 100% | 0.8667 |

paraphrase 是最弱的一档（0.7400），这是语义检索真正的短板，
也是下一步最该优化的地方——比「HitRate 100%」这句话有信息量得多。

**Roadmap（写进报告结论时要能说出来）**：
1. 扩大语料到 100+ 篇、加入同主题干扰文档，让 HitRate 不再是 100%；
2. 增加 paraphrase / 多跳题比例（现在 5/20 太少），并补充「答案在多个文档」的题；
3. 加生成侧指标（答案正确率、引用是否支持结论），现在只测检索不测回答；
4. `k=5` 固定值做一次 k=3/5/10 的敏感性分析；
5. 把评测跑进 CI，防止改切块/换模型时静默退化（现在依赖人工跑 `python eval/eval.py`）。

### 1.6 评测脚本的执行流程（6 步）

`eval/eval.py` 的结构很清楚，面试问「评测怎么做的」可以按这六步讲：

1. **解析参数**：`--tag`（结果文件名）、`--k 5`、`--base-url`、`--eval-set`；
2. **读标注集**：`load_eval_set()` 兼容 `question/query/q`、`source/sources`、`keywords/kw` 多种字段名，
   并清洗成统一结构；坏行跳过、不中断；
3. **探活 + 防空测**：先 `GET /api/status`，如果 `chunk_count == 0` 直接退出并提示
   `POST /api/reindex`——避免「对着空库跑出一堆 0」；
4. **逐题调用 `/api/search`**：`run_case()` 记耗时、原始命中判定、Top3 来源；单题异常只记 `error` 不中断整轮；
5. **聚合**：`_aggregate()` 算主/辅 HitRate 与 MRR；再按 `type` 分组各算一遍（`by_type`），
   `percentile()` 算 P50/P95；
6. **落盘两份**：`eval_results/<tag>.md`（人看）和 `eval_results/latest.json`（前端 `/api/eval/summary` 读）。

`--tag` 是刻意设计的：`--tag before` / `--tag after` 可以并排比较同一套索引在改动前后的变化；
文件名带 tag，历史结果不会互相覆盖（`latest.json` 永远是最新一次，`<tag>.md` 留档）。

### 1.7 报告长什么样

`eval/report.py · write_markdown` 固定输出五块：天花板说明 → 总览表（主/辅口径 + P50/P95 + 命中数）
→ 分题型表 → 逐题明细（含 `主命中/排名/kw命中/kw排名/RR/延迟/Top1来源`）→ 失败题目（有 error 才输出）。
逐题明细是最有用的部分：`case-12` 主排名 2、kw 排名 1 这种组合，一眼能看出「关键词先命中、但来源是另一份文档」，
这就是 paraphrase 多跳题的典型表现，也是调 prompt / 切块 / 融合参数的依据。

`latest.json` 存的是同一份数据的结构化版本（多存了 `cases` 全量），
`/api/eval/summary` 会把它摊平成 `hit_rate/mrr/p95_ms/n` 等字段给前端，前端不用理解嵌套结构。

## 2. 为什么这样写（对比错写法）

### 新手会写成什么样

```python
# 错版一：只看关键词就算命中
hit = any(kw in r["content"] for r in results for kw in case["keywords"])

# 错版二：只报平均延迟
print("平均延迟", sum(latencies) / len(latencies))

# 错版三：出题、调参、评测用同一批题，还只跑一次
```

### 问题在哪

1. **口径太松 → 指标虚高**：Top1 是别的文档、关键词碰巧命中就算「检索到了」，面试官追问
   「你怎么定义命中」时答不清；现在主口径要求「来源 + 证据词」双命中，辅口径只用于 debug。
2. **只报平均值 → 长尾被藏**：本项目当前 P50 9ms，但逐题里最慢一条 21.7ms；
   换成真实规模的语料，慢请求可能到几秒，平均值会把它们抹平。所以报告同时给 P50 和 P95，
   并且**承认 P95 也取不到最大值**（n=20 时取第 19 个）。
3. **不记录环境 → 数字不可复现**：报告里必须带 `embedding/collection/chunk_count/k/标注集路径/时间`，
   否则「昨天 100%、今天 80%」根本不知道为什么变了。`eval.py` 会先查 `/api/status` 记下模型与 collection，
   再做空库检查。
4. **只跑一次 → 无法证明优化有效**：要有 before/after 两次同环境对比（`--tag before` / `--tag after`），
   否则分不清是改动生效还是机器抖动（延迟尤其明显）。
5. **占位题当真数据**：语料没生成时 `load_eval_set` 会写入 5 条 placeholder 题，报告里必须打
   `placeholder=True` 警告（`write_markdown` 会输出 ⚠️），否则会拿「无意义数字」当结论。
6. **自己出题自己调参**：标注集要独立于调参过程，否则就是过拟合自己的 20 道题；
   本项目标注集的 `kw` 是从原文抽出来校验过的，不是「跑完看结果再改」。

## 3. 面试官可能怎么问

1. **「HitRate@5 和 MRR 分别是什么，怎么算？」**
   → HitRate@5 = 前 5 条里至少一条强命中的题目占比；MRR = 每题「第一次命中排名的倒数」的平均值
   （第 1 名 1.0、第 2 名 0.5…）。本项目：HitRate@5 100%（20/20）、MRR 0.9017（18.033/20）。
   HitRate 饱和之后，MRR 才能区分排序质量。

2. **「为什么主指标要求来源和关键词同时命中？只用关键词会怎样？」**
   → 只用关键词会出现「Top1 是别的文档但碰巧含关键词」也算命中，指标虚高；
   主口径（强命中）保证「来源对、证据也对」，辅口径（仅关键词）只用于 debug 定位是召回问题还是排序问题；
   `tests/test_eval_report.py` 有专门用例锁住这个口径。

3. **「HitRate 100% 能说明系统很好吗？」**
   → 不能：语料只有 54 块、20 道题，HitRate 已饱和（100%），这是天花板效应；
   能区分的是 MRR（0.9017）和分题型 MRR（paraphrase 0.7400 最弱）。
   下一步要扩语料+加干扰文档、增加同义改写题、补生成侧指标，再谈真实场景效果。