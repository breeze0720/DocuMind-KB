# 01 · Embeddings：本地 bge-small-zh-v1.5（512 维）

> 一句话：**「文字 → 向量」固定用本地 ONNX 推理（fastembed + bge-small-zh-v1.5）**，
> 进程内单例复用，离线、单条约 10ms；云端 embedding 做过一轮对比后删掉了，
> 结论写在 [§2.3](#23-为什么最后把云端分支删掉)，留一段历史比留一套用不上的代码更有用。
>
> 术语：**dimension**（向量维度）、**collection**（Chroma 里的集合，可理解成一张表）、
> **HNSW**（Chroma 底层用的近似最近邻索引）、**cosine**（余弦相似度）、**ONNX**（跨框架的模型推理格式）。

## 1. 实现概览

`core/embeddings.py` 只有 100 行左右：一个类、一个单例、一个状态函数。

### 1.1 一个类、两个方法

```python
class LocalEmbedding(Embeddings):
    """本地 ONNX 推理。继承 LangChain 的 Embeddings，所以能直接塞给 Chroma 当 embedding_function。"""

    def __init__(self) -> None:
        settings = config.settings
        model_dir = Path(settings.local_model_dir)
        onnx_file = model_dir / "model_optimized.onnx"
        if not onnx_file.is_file():
            # 绝不联网下载（huggingface.co 在本机不可达，会白等超时），直接给解决办法
            raise EmbeddingError(f"本地向量模型缺失…期望路径: {onnx_file}…先执行 .\\download_model.ps1")
        from fastembed import TextEmbedding          # ① 延迟 import
        self._model = TextEmbedding(model_name=settings.local_embedding_model,
                                    specific_model_path=str(model_dir))
        self.model = settings.local_embedding_model
        self.dimension = _read_dimension(model_dir)  # ② 维度从 config.json 读，不写死

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
```

- **① 延迟 import**：`onnxruntime` + `fastembed` 加起来是几百 MB 的依赖，import 就要几百毫秒。
  放在函数里，只有真的要向量化时才加载；`pytest` 里那些纯函数测试（切块、RRF、命中判定）
  根本不会触发它，测试跑 7 秒就能全绿。
- **② 维度从 `config.json` 读**：bge-small-zh-v1.5 是 `hidden_size=512`。写死 512 也能跑，
  但哪天换成 bge-base（768 维），旧代码会把 768 维向量往 512 维的 collection 里写，
  Chroma 直接报 `Dimensionality of (768) does not match collection dimensionality (512)`。
  从模型文件读，维度永远是「模型真正输出的维度」。

继承 `langchain_core.embeddings.Embeddings` 而不是自定义接口，是因为 Chroma 的
`embedding_function` 参数就认这个基类；换成自己的协议还得包一层适配器，没有收益。

### 1.2 进程级单例：加载一次，全进程复用

```python
_instance: LocalEmbedding | None = None
_lock = threading.Lock()

def get_embeddings() -> LocalEmbedding:
    global _instance
    if _instance is not None:          # 快路径：无锁
        return _instance
    with _lock:                        # 双检锁：并发首次调用也只构造一次
        if _instance is None:
            _instance = LocalEmbedding()
        return _instance
```

为什么必须缓存：ONNX 模型 91MB，加载 + 初始化推理会话要 1-2 秒。
如果每个请求都 `LocalEmbedding()`，第一个查询会卡 2 秒、内存里还会堆出多份模型。
`core/runtime.py` 的 `get_store()` 里再套一层同样的双检锁，保证
「embedding + Chroma + BM25」这套东西整个进程只有一份。

> 顺带一提：`get_embeddings()` 是唯一入口，`get_store()` 也只从这里拿模型，
> 所以「模型加载失败」只会在第一次访问时抛 `EmbeddingError`，
> 上层（路由/Agent）捕获后给中文提示，不会把用户吓一跳。

### 1.3 状态接口：只报告，不加载模型

`embedding_status()` 是 `/api/status` 的数据源，它**不实例化 `LocalEmbedding`**，
只读配置 + 判断文件在不在：

```python
def embedding_status() -> dict:
    settings = config.settings
    model_dir = Path(settings.local_model_dir)
    available = (model_dir / "model_optimized.onnx").is_file()
    return {
        "model": settings.local_embedding_model,
        "label": "bge-small-zh-v1.5",
        "dimension": _read_dimension(model_dir),
        "collection": settings.collection_name,
        "available": available,
        "detail": "" if available else "本地模型文件缺失，请先执行 .\\download_model.ps1",
    }
```

为什么这么设计：状态接口是前端每次刷新都会调的，如果它也去加载模型，
打开页面就要等 2 秒；更糟的是模型缺失时它会抛异常，前端只能显示「连接失败」，
用户根本不知道是模型没下载。现在最差情况是页面正常打开、嵌入 chip 显示
`bge-small-zh-v1.5 · 512d`、`/api/status` 的 `embedding.detail` 给出中文原因。

## 2. 为什么固定本地模型

### 2.1 四条理由

| 维度 | 本地 bge-small-zh-v1.5 | 云端 embedding（百炼 text-embedding-v4） |
| --- | --- | --- |
| 网络 | 完全离线，断网可跑 | 每次请求都要联网 |
| 延迟（query 单条） | 约 5-10ms（含在检索 P50 9ms 里） | 300-500ms 起，长尾更差 |
| 成本 | 0 | 按 token 计费 |
| 数据 | 文档不出本机 | 文档正文要发到外部服务 |

个人知识库这个场景（本机、几十到几千块文档、单用户），本地小模型是明显更划算的工程选择：
**一次加载、离线、零延迟抖动、隐私不出机器**。

### 2.2 切块为什么是 500 tokens

`config.py` 里 `chunk_tokens=500` 不是拍脑袋定的：bge-small-zh-v1.5 的 `config.json`
写着 `max_position_embeddings=512`，减去 `[CLS]/[SEP]` 两个特殊 token，正文最多约 510。
取 500 留安全余量——超了不会报错，tokenizer 会**静默截断**，表现是「检索时后半段内容永远搜不到」，
这种坑不写注释根本没人体会。这也是切块按 token 而不是按字符切的原因（见 02 篇）。

### 2.3 为什么最后把云端分支删掉

**第一阶段**做的是「接口 + 两个实现 + 工厂」：`EmbeddingProvider` 抽象类，
`LocalFastEmbed` 和 `DashScopeEmbedding` 两个实现，collection 名带后端标签和维度
（`kb_bge_512` / `kb_dashscope_1024`），`/api/search?embedding=...`、`/api/reindex?embedding=...`、
`eval.py --embedding ...` 全程可切换，同一套 20 道题跑两遍出两份报告。

当时（2026-10，同一份 54 块语料、20 道题）的实测：

| 指标 | 本地 bge-small-zh-v1.5（512 维） | 百炼 text-embedding-v4（1024 维） |
| --- | --- | --- |
| HitRate@5（主口径） | 100% | 100% |
| MRR（主口径） | **0.9017** | 0.7833 |
| paraphrase 题型 MRR | **0.7400** | 0.5667 |
| P50 检索延迟 | 104ms | 359.7ms |
| P95 检索延迟 | 185ms | 1209ms（还有一次 21.5s 的离群值没进 P95） |

> 延迟那两行是当时那台机器、那次运行的记录（机器负载偏高）；同一份代码后来在空闲机器上重跑，
> local 的 P50/P95 是 9ms/21ms（见 §4）。MRR/HitRate 每次运行都一样，延迟会随负载漂移。

结论很直接：**云端 1024 维没有换来更好的排序**，同义改写的题反而更差（0.57 vs 0.74），
延迟差一个数量级，还得配 key、联网、按 token 付费。于是第二阶段把这套 A/B 机制整个删掉：

- `EmbeddingProvider` 抽象类 / 两个实现 / 工厂字典 → 一个 `LocalEmbedding`；
- `EMBEDDING_PROVIDER`、`DASHSCOPE_*` 六个配置项 → 全删（`.env` 里也不再需要 key）；
- `collection_name` 从「按后端+维度算出来」→ 配置里一个固定的 `kb_bge_512`；
- `/api/search`、`/api/reindex`、`/api/upload`、`/api/documents` 的 `embedding` 参数 → 全删；
- `eval.py --embedding` → 删，只保留一套索引、一个 tag；
- `eval_results/dashscope.md`、`kb_dashscope_1024` collection、对应的 BM25 缓存 → 删；
- `core/runtime.py` 里「按后端名缓存三件套」的字典 → 三个模块级变量。

**为什么值得记录**：做一次对比、用数据确认本地方案更优、然后主动删掉不需要的抽象，
比「我实现了一个可插拔的多后端架构」更能说明工程判断——**抽象层是有成本的**，
只有一个实现时，接口+工厂+双份配置+双份评测就是把 100 行代码变成 240 行的负债。
现在 `core/embeddings.py` 102 行、`core/runtime.py` 86 行，读一遍就能全懂。

## 3. 几个容易踩的坑（现在的代码怎么防）

1. **维度必须跟 collection 绑定**：Chroma 建 collection 时固定维度，换模型不换 collection 会直接报错。
   本项目 collection 固定叫 `kb_bge_512`，名字里带维度，换模型时一眼能看出要重建索引（`/api/reindex?force=true`）。
2. **换模型 = 重建索引**：向量空间不同，旧向量不能复用。代码里没有「自动迁移」，
   宁可让用户显式重建，也不偷偷混写两套向量。
3. **模型文件必须提前准备**：本机 huggingface.co 不可达，所以 `download_model.ps1`
   从 ModelScope 下载 ONNX 到 `models/bge-small-zh-v1.5/`；代码里缺文件直接报错并给出这一步，
   绝不做「联网重试 3 次再超时 30 秒」这种让用户以为服务挂了的兜底。
4. **`.tolist()` 不是多余**：fastembed 返回 numpy 数组，直接塞给 Chroma/JSON 会序列化失败，
   统一转成 Python list，入库和接口返回共用一套数据形状。

## 4. 实测数据（`eval_results/local.md`，54 块语料 / 20 道题）

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| HitRate@5 | 100.00% | 来源命中且该块含标注关键词（主口径） |
| MRR | 0.9017 | 同上，按强命中排名计算 |
| P50 检索延迟 | 9.0 ms | 检索接口耗时（服务预热后） |
| P95 检索延迟 | 21.5 ms | 长尾耗时 |
| literal MRR | 1.0000 | 原词题，基本都能排第 1 |
| paraphrase MRR | 0.7400 | 同义改写题，短板所在 |
| short MRR | 0.8667 | 两个字的短问题，靠 BM25 补语义 |

⚠️ 天花板效应：HitRate 已经 100% 饱和，**它区分不出好坏**，真正有信息量的是 MRR 和分题型。
paraphrase 的 0.74 说明纯向量检索在「语义相近但用词完全不同」的题上还会排错，
这也是混合检索（BM25 + RRF）存在的理由（见 04 篇）。

> 延迟数字与机器负载有关：同一份代码在机器空闲时 P50 约 9ms，跑着别的服务时会到 100ms+。
> 需要解释「为什么这么快」时，如实说明「54 块的小索引 + 本地小模型，上限本来就低」比堆数字可靠。

## 5. 常见问题（FAQ）

1. **「为什么用 bge-small-zh-v1.5，不用更大的 bge-large 或者 OpenAI embedding？」**
   → 中文短文本 + 小语料，small 的 512 维已经够用（HitRate 100%）；large 是 1024 维、模型 300MB+，
   推理慢 3-4 倍，在这个场景属于过度设计。OpenAI embedding 要联网+付费，且中文语料上没优势。
   用数据说话：本地方案 MRR 0.9017、P95 21ms。

2. **「换 embedding 模型时最容易出什么事？」**
   → 维度不一致导致写入/查询报错（所以 collection 名带维度）；其次是旧向量不能复用，
   必须重建索引；还有 tokenizer 的 max_position_embeddings 会静默截断长文本，
   所以切块要按 token 且留余量。

3. **「你把云端 embedding 的代码删了，那你怎么证明本地更好？」**
   → 当时同一套标注集、同一份语料跑过双后端：云端 MRR 0.7833、P50 359.7ms；
   本地 MRR 0.9017、P50 104ms（那台机器较忙），同一份代码后来在空闲机器上是 9ms。
   排序指标稳定，延迟随负载漂移——延迟口径需要注明机器与预热状态。
   删代码删的是「用不上的分支」，不是删实验记录。

4. **「可插拔架构不是更灵活吗？删掉不可惜？」**
   → 灵活是有成本的：两个实现要两套配置、两套 collection、两套评测、两倍的状态接口分支，
   还要维护「切换后端时旧索引怎么办」这种边界。真实项目里只有一个后端时，
   这层抽象就是纯负债。留着实验结论、删掉实验脚手架，是更划算的工程选择。