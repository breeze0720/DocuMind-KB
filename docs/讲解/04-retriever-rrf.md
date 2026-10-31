# 04 · Retriever 与 RRF 融合

> 一句话：**两路检索各自给出排名，用 `1/(60+rank)` 把排名换成可加的分数再排序**。
> 只依赖排名、不依赖原始分数，所以向量相似度和 BM25 分数量纲不同也能融合。
>
> 术语：**hybrid search**（混合检索）、**RRF**（Reciprocal Rank Fusion，倒数排名融合）、
> **candidate pool**（候选池，融合前每路多取一些）、**score_kind**（分数类型标记，说明这个分是什么量纲）、
> **threshold**（阈值过滤）。

## 1. 这段代码在做什么

### 1.1 入口：两路各取 20 个候选，再融合取前 k

`core/retriever.py · HybridRetriever.search`：

```python
k = int(k or self.top_k)                     # 默认 5
pool = max(k * 4, self.candidate_floor)      # ① k=5 → 20；k=3 也至少 20
vector_hits = self.store.search(query, k=pool)
if min_score is not None:
    vector_hits = [(doc, s) for doc, s in vector_hits if s >= float(min_score)]  # ② 只过滤向量路
keyword_hits = self.store.keyword_search(query, k=pool)
fused = fuse_rrf(vector_hits, keyword_hits, rrf_k=self.rrf_k)   # ③ 融合
for rank, item in enumerate(fused[:k], start=1):                # ④ 输出前 k 条 + 元数据
    ...
```

- ① 候选池要比 k 大：RRF 只看排名，如果每路只取 5 条，「向量第 6 名但在 BM25 第 1 名」的块根本没机会加总；
  固定下限 20 是为了让 `k=3` 之类的小请求也有足够融合空间；
- ② `min_score` **只作用于向量路的 0~1 相关性分**，BM25 路不受影响（原因见 1.4）；
- ③ 融合是纯函数，可以单独测试；
- ④ 输出补 `source/chunk_index/page/sheet/slide/section/title` 和 `rank`，让引用能定位。

### 1.2 RRF 公式与实现

数学形式（k 是平滑常数，`rank` 从 1 开始）：

```
score(d) = Σ_over_paths  1 / (k + rank_path(d))          # k = 60（Cormack et al. 2009 的经验值）
```

代码里 `rank` 从 0 开始，所以第 1 名是 `1/60`，第 2 名 `1/61`，以此类推：

```python
def fuse_rrf(vector_hits, keyword_hits, rrf_k=60):
    scores: dict[tuple[str, str], float] = {}
    docs: dict[tuple[str, str], Document] = {}
    for hits in (vector_hits, keyword_hits):             # ① 两路依次累加
        for rank, (doc, _raw_score) in enumerate(hits):  # ② 只要下标，原始分丢掉
            key = doc_key(doc)                           # ③ 去重 key=(来源, 正文)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)                    # ④ 保留先出现的那份文档对象
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)  # ⑤ 按融合分降序
    return [{"key": key, "document": docs[key], "score": scores[key], "score_kind": "rrf"}
            for key in ordered]
```

- ① 两路都命中的块会**加两次分**，这是有意的：两套方法都认为相关，本来就该排更前；
- ② `_raw_score` 完全不参与计算。这是 RRF 的核心——向量分是 0~1 余弦，BM25 分无上界（可能 12.5），
  直接加权相加必须调参、换库就失效；
- ③ key 见 1.3；
- ④ `setdefault` 保证同一 key 只保留一份 Document，避免重复返回；
- ⑤ 稳定排序，分数相同按插入顺序（向量路在前）。

### 1.3 手算一个例子（面试会让你算）

设 `k=60`，向量路排名（从 1 开始）：A、B、C；BM25 路排名：B、C、E。每条只出现一次除 B、C 外：

| 块 | 向量路贡献 | BM25 贡献 | 融合分 |
| --- | --- | --- | --- |
| A | 1/60 = 0.016667 | — | **0.016667** |
| B | 1/61 = 0.016393 | 1/60 = 0.016667 | **0.033060** |
| C | 1/62 = 0.016129 | 1/61 = 0.016393 | **0.032522** |
| E | — | 1/62 = 0.016129 | **0.016129** |

最终排序：**B > C > A > E**。

两个结论：
- B 在向量路只排第 2，但两路都命中，融合后反超 A 排第 1——这就是混合检索的价值；
- E 在 BM25 排第 3、向量路没有，融合后垫底——单路命中的块要靠前排名才能留下。
- 还能看出分数的绝对大小：**单路第 1 名约 0.0167，两路都第 1 名最多 0.0333**。这个数量级记住，1.4 会用到。

### 1.4 去重 key、score_kind 与「为什么不能拿融合分做阈值」

**key = `(source, 正文)`**，不是正文本身：

```python
def doc_key(doc: Document) -> tuple[str, str]:
    return (str((doc.metadata or {}).get("source", "")), doc.page_content)
```

原实现用正文做 key，结果「员工手册」和「FAQ」里都有的同一句话被合并成一条，
来源和 `chunk_index` 直接丢掉，引用无法溯源。现在同文本不同来源各算各的
（`tests/test_retriever.py::test_fuse_rrf_keeps_same_text_from_different_sources`）；
同一块在两路都命中时 `(source, 正文)` 相同，正常相加去重。

**`score_kind="rrf"`** 是给前端和模型的「量纲标签」：

- `store.search()` 返回的是 0~1 的 cosine 相关性分；
- `keyword_search()` 返回 BM25 分，无上界；
- `fuse_rrf()` 返回的是 RRF 分，**只表示相对排名**，最大值约 0.0333。

同一份结果里三种分数混在一起，如果前端拿 `0.03` 显示成「3% 相关」，用户会以为检索效果很差。
所以每条结果都带 `score_kind`，`/api/search` 原样返回，前端表格里显示「rrf」而不是百分比。

**为什么不能用融合分做阈值过滤**（这是原实现的第三个问题）：

1. RRF 分由 `k` 决定：`k=60` 时第 1 名 0.0167，改成 `k=10` 就变成 0.1，**同一个阈值换参数就失效**；
2. 它和「相关度」没有可比性：两路都排第 1 也只有 0.0333，拍一个 `min_score=0.5` 会把所有结果滤光；
3. 反过来，`0.01` 这种阈值又几乎全放行，等于没过滤。阈值必须落在有明确语义的量纲上——
   只有向量路的 cosine 分（0~1）满足这个条件。

所以 `min_score` 只过滤向量路，`tests/test_retriever.py::test_min_score_filters_only_vector_branch`
专门断言「低分块在向量路被滤掉、但 BM25 路仍然命中，最终仍出现在结果里」——
这既证明阈值没有作用在融合分上，也证明两路是独立的。

### 1.5 返回结构：每条结果带哪些字段

`HybridRetriever.search` 最后组装的 dict 是 `/api/search` 和引用注册表的共同输入：

| 字段 | 用途 |
| --- | --- |
| `rank` | 1 基排名，评测的 MRR 和引用编号都依赖它 |
| `score` + `score_kind="rrf"` | 融合分只用于排序展示，`round(..., 6)`；`score_kind` 说明量纲 |
| `source` / `source_key` | 展示用文件名 / 删除和注册表用的原始值 |
| `chunk_index` | 块号，引用定位 `(source_key, chunk_index)` |
| `page/sheet/slide/section` | 更可读的出处（第几页 / 哪个 sheet / 第几张幻灯片） |
| `title` | 文档标题，引用卡片标题 |
| `content` | 全文（`/api/search` 给 eval 做关键词匹配必须用全文，不能只给预览） |

`k` 参数只影响返回条数，不影响候选池下限（`pool = max(k*4, 20)`）。
`k=5` 时两路各取 20 个候选、融合后取前 5；如果 `k=10`，候选池变成 40，融合空间更大。

### 1.6 候选池为什么不能等于 k（一个具体例子）

假设 `k=5`，两路都只取 5 条。某块在向量路排第 7、在 BM25 路排第 1：
如果 pool=5，它根本没进向量候选，融合时只有 BM25 的 `1/60≈0.0167`；
而另一块在向量路排第 1（`1/60`）+ BM25 路排第 5（`1/64`）能到约 0.0323，直接压过它。
把 pool 提到 20 后，前者拿到 `1/66 + 1/60 ≈ 0.0318`，两路都命中的证据才被算进来。
这就是 `candidate_floor=20` 的存在理由——**RRF 的收益来自「同一块在两路都被看到」**。

### 1.7 min_score 怎么用（只在需要时加）

```powershell
# 不加阈值：纯 RRF 排序，默认行为
curl "http://127.0.0.1:8010/api/search?query=年假&k=5"

# 加阈值：只过滤向量路（cosine 0~1），BM25 命中的块仍然保留
curl "http://127.0.0.1:8010/api/search?query=年假&k=5&min_score=0.35"
```

`min_score` 在 `HybridRetriever.search` 里只对 `vector_hits` 生效，理由见 1.4。
实际调参建议：先看 `/api/search` 返回结果里向量分的分布（不加阈值跑几次），
再决定阈值；小语料上通常不需要加，加了容易把「语义弱但关键词强」的块误杀。

## 2. 为什么这样写（对比错写法）

### 新手会写成什么样

```python
# 错版一：加权求和（还得归一化）
score = 0.7 * cosine_sim + 0.3 * bm25_score / max_bm25_score

# 错版二：用正文当 key 去重
seen = set()
for doc in vector_hits + keyword_hits:
    if doc.page_content in seen: continue
    seen.add(doc.page_content)
    results.append({"content": doc.page_content, "score": ...})
```

### 问题在哪

1. **加权求和要调参且不稳定**：余弦分 0~1、BM25 无上界，`0.7/0.3` 是拍脑袋；
   换 embedding 模型或语料，最优权重全变；还要处理 max_bm25 为 0 的除零。
   RRF 只需要一个 `k=60`，论文经验值，换库不用重调。
2. **用正文 key 会丢来源**：两篇文档里都有「年假为每年十天」，合并后只剩一条，
   用户点引用看到的是另一份文件；也无法定位 `chunk_index`。
3. **只返回 `content + score` 做不了引用**：没有 source/chunk_index，前端无法渲染来源卡片，
   更谈不上「点击 [2] 跳到原文」。
4. **把 RRF 分当相关度喂给模型**：模型看到「0.03」会认为这条资料极不相关而不敢引用，
   于是明明检索到了却答「知识库中没有」。正确做法是只把编号和正文给模型（见 05 篇）。
5. **阈值过滤融合分**：上面 1.4 已说明，会把结果全滤掉或全放行，指标看起来「玄学」。

## 3. 面试官可能怎么问

1. **「RRF 的公式是什么，k=60 有什么用？」**
   → `score = Σ 1/(k+rank)`，`k=60` 来自 Cormack 2009 的经验值；它的作用是压平「第 1 名 vs 第 2 名」的差距
   （1/60 vs 1/61 只差 0.0003），让两路排名平滑融合，不会一路的第 1 名独大。
   手算：向量路 A、B、C，BM25 路 B、C、E → B=1/61+1/60=0.0331 排第 1。

2. **「为什么不直接把两路分数加权相加？」**
   → 量纲不同：余弦 0~1、BM25 无上界，加权要归一化和调参，换模型/库就失效；
   RRF 只用排名，天然免疫量纲问题，是工业界默认基线。

3. **「融合后的分数能不能当阈值过滤？」**
   → 不能：RRF 分只表示相对排名（第 1 名约 0.0167，两路都第 1 约 0.0333），且随 `k` 变化；
   所以 `min_score` 只作用于向量路的 0~1 相关性分，BM25 路不过滤。