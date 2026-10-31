# 03 · Store：Chroma 封装 + BM25 关键词索引

> 一句话：**一个 `KnowledgeStore` 同时管向量库（Chroma）和关键词索引（BM25），所有写操作只走它，
> 保证两套数据同增同删**。BM25 索引是进程级缓存 + 磁盘 JSON，只在写操作后重建。
>
> 术语：**facade**（门面，用一个类包住多个子系统）、**idempotent**（幂等，重复执行结果不变）、
> **upsert**（存在就更新、不存在就插入）、**jieba**（中文分词库）、**idf**（逆文档频率，
> BM25 里的权重项，词越常见权重越低）、**锁（RLock）**（可重入互斥锁，防并发写坏数据）。

## 1. 这段代码在做什么

### 1.1 初始化：Chroma 压阵，BM25 只是缓存

```python
self.bm25_cache_path = self.chroma_dir / f"bm25_docs_{collection_name}.json"  # ① 按 collection 分开存
self._lock = threading.RLock()
self._docs: list[dict] = []          # 内存镜像：BM25 用，也是「块数」的来源
self._bm25: BM25Okapi | None = None
self._tokens: list[list[str]] = []   # 与 _docs 一一对应的分词结果（关键词命中判定用）
self._vs = Chroma(
    persist_directory=str(self.chroma_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
    collection_metadata={"hnsw:space": "cosine"},   # ② 显式余弦空间
)
self._load_indexes()
```

- ① 缓存文件名带 collection：当前是 `bm25_docs_kb_bge_512.json`。
  带名字而不是固定 `bm25_docs.json`，是为了「换模型/换 collection 时旧缓存不会被误读」
  ——这条规则来自一次真实事故（见本节末尾第 6 条）；
- ② 为什么必须显式 cosine：LangChain 对 `l2` 会把距离换算成 `1 - distance/√2`，不相关文档可能得到**负数**，
  `min_score` 阈值就没法统一理解；cosine 下分数稳定落在 0~1。

`_load_indexes()` 是一段「对账」逻辑，值得背：

```python
docs = load_cached_docs(self.bm25_cache_path)          # 先试磁盘缓存
try:
    chroma_count = self._vs._collection.count()
except Exception as exc:
    chroma_count = len(docs)
if (not docs and chroma_count) or len(docs) != chroma_count:
    docs = self._read_all_from_chroma()                # 缓存缺失/条数对不上 → 从 Chroma 全量重建
self._docs = docs
self._rebuild_bm25()
```

**Chroma 是唯一事实来源（single source of truth），BM25 JSON 只是加速缓存。**
条数对不上就重建，所以「别人直接往 Chroma 写数据」「缓存文件损坏」「删库只删了一半」这类情况，
重启后都能自愈。`_read_all_from_chroma` 还会按 `(source, chunk_index)` 排序，保证 BM25 结果稳定。

### 1.2 写：确定性 id + 两套索引在同一个锁里

```python
texts = [doc.page_content for doc in documents]
metadatas = [_clean_metadata(doc.metadata) for doc in documents]
ids = []
for metadata in metadatas:
    file_hash = str(metadata.get("file_hash") or "nohash")[:16]
    ids.append(f"{file_hash}:{metadata.get('source', '')}:{metadata.get('chunk_index', 0)}")  # ①

vectors = self.embeddings.embed_documents(texts)   # ② 先算向量再进锁，减少锁持有时间
with self._lock:
    self._vs._collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)  # ③
    replaced = set(ids)
    self._docs = [item for item in self._docs if item["id"] not in replaced]   # ④ 内存镜像同步替换
    self._docs.extend({"id": i, "content": t, "metadata": m} for i, t, m in zip(ids, texts, metadatas))
    self._rebuild_bm25()                                                        # ⑤ 重建 + 落盘
```

- ① id 由「内容哈希前 16 位 + 来源 + 块号」组成：内容不变 id 不变，重复导入会覆盖同一条而不是翻倍；
- ② 向量化是批量推理（54 块约几十毫秒，语料大时会到几百毫秒），放在锁外，避免阻塞其他请求的查询；
- ③ 用 `upsert` 而不是 `add`，这是幂等的关键；
- ④⑤ Chroma 写完立刻同步内存镜像和 BM25，**同一个锁内完成**，中间不会出现「向量能搜到、关键词搜不到」的窗口。

删除同理，`delete_source()` 在一个锁里做三件事：Chroma 删 id、`_docs` 过滤、重建 BM25。
它还兼容两种 source 写法：新数据存文件名，旧 chroma_db 里存的是绝对路径——
`Path(...).name` 比较一次就能都命中（`tests/test_store.py` 里有专门用例）。

### 1.3 读：BM25 用分词交集判断命中，不用分数

```python
def keyword_search(self, query: str, k: int = 5):
    with self._lock:
        bm25, docs, tokens_list = self._bm25, list(self._docs), list(self._tokens)
    tokens = bm25_tokenize(query)                       # ① jieba 搜索引擎模式
    query_set = set(tokens)
    matched = [i for i, doc_tokens in enumerate(tokens_list) if query_set & set(doc_tokens)]  # ② 词命中
    if not matched:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(matched, key=lambda i: scores[i], reverse=True)[: max(1, int(k))]
    ...
```

- ① `jieba.lcut_for_search`（搜索引擎模式）会把「生鲜识别平台」再切出「生鲜/识别/平台」，
  用户搜短词也能命中长标题；用默认的按空格分词，中文整段会变成一个 token，等于没做关键词检索；
- ② **不能用 `score > 0` 过滤**：小语料下 idf 可能恰好为 0。BM25 的 idf 项是
  `log((N-df+0.5)/(df+0.5))`，当 N=2、df=1 时结果就是 0，真正命中的块分数也是 0，会被误杀。
  这里先用「查询词与块分词结果有交集」判断命中，再按分数排序，逻辑和量纲解耦。

向量侧 `search()` 直接调 `similarity_search_with_relevance_scores`，异常时返回 `[]` 并 warning，
不让一次检索失败把整个问答打挂。

### 1.4 BM25 缓存的三个时机

| 时机 | 动作 | 目的 |
| --- | --- | --- |
| 启动/首次访问 | `load_cached_docs` → 与 Chroma 对账 → 必要时全量重建 | 重启不用把整个语料重新分词一遍 |
| `add_documents` | 重建 + `save_cached_docs`（临时文件 + `os.replace` 原子替换） | 新增内容立刻可被关键词搜到 |
| `delete_source` | 重建 + 落盘 | 删除后不残留「幽灵块」 |

`save_cached_docs` 用 `tmp_path.write_text` + `os.replace`，避免进程写到一半挂掉留下半个 JSON；
写失败只 warning——缓存坏了下次从 Chroma 重建，不影响本次查询。

### 1.5 删除：为什么必须在一个锁里删两次

```python
def delete_source(self, source: str) -> int:
    target_name = Path(source).name                        # ① 兼容旧数据里的绝对路径
    with self._lock:
        ids = [item["id"] for item in self._docs
               if str(item["metadata"].get("source", "")) == source
               or Path(str(item["metadata"].get("source", ""))).name == target_name]
        if not ids:
            return 0                                        # ② 没找到返回 0，而不是抛异常
        self._vs._collection.delete(ids=ids)                # ③ 先删向量库
        removed = set(ids)
        self._docs = [item for item in self._docs if item["id"] not in removed]  # ④ 再同步内存镜像
        self._rebuild_bm25()                                # ⑤ 最后重建关键词索引
    return len(ids)
```

- ① 新数据的 `source` 是文件名，旧 `chroma_db` 里可能是 `D:\...\旧文件.txt`，两种都要能删；
- ② 删除不存在的来源是正常情况（前端重复点删除），返回 `deleted_chunks: 0`，路由层再转 404；
- ③④⑤ 三步必须原子：如果只删 Chroma，BM25 里那块还能被关键词检索到，点进去来源已不存在；
  反过来只删 BM25 也一样。`tests/test_store.py::test_delete_syncs_both_indexes` 断言删除后
  `keyword_search` 也搜不到、`find_source_by_hash` 也查不到。

删除完成后路由层还会顺手删掉 `uploads/` 里的原始文件（`app/routes_documents.py · _delete_source`），
但**数据结构和文件是两件事**：文件被占用删不掉时只 warning，索引删除不受影响——宁可磁盘多留一份，
也不要出现「索引删了一半」的状态。

### 1.6 两个「顺带」的查询接口

`find_source_by_hash(file_hash)` 是上传幂等的判据：遍历内存镜像 `_docs`，找 `metadata.file_hash` 相同的块，
返回 `{source, source_key, title, chunks}`。用内存镜像而不是查 Chroma，是因为上传判重要快，
而且这个字段本来就随块存了。

`list_sources()` 给 `/api/documents` 用：按 `source` 分组统计块数，返回
`{source, source_key, title, file_hash, chunks}`。注意 `source` 是展示名（文件名），
`source_key` 是删除用的原始值——老数据存的是全路径，两个字段分开才既好看又能删。

### 1.7 metadata 为什么要先洗一遍

Chroma 的 metadata 只接受 `str/int/float/bool`，`None` 会直接抛异常；
而 loader 会带 `page=None`、`sheet=None` 这类字段进来。`_clean_metadata` 的做法是：
`None` 直接丢掉（不是塞空串），其他类型转 `str`。丢掉的原因是 `where={"source": x}` 这类过滤条件
会被空字符串干扰——「没这一页」和「第 0 页」是两回事。

## 2. 为什么这样写（对比错写法）

### 新手会写成什么样

```python
# 错版：两个全局变量 + 调用方手工同步
chroma_db = Chroma(...)
all_documents = []                       # BM25 用的文档列表，另一个全局
bm25 = None

def add(docs):
    chroma_db.add_documents(docs)
    all_documents.extend(docs)           # 忘了这行 → 关键词搜不到

def delete(source):
    chroma_db.delete(...)                # 忘了重建 BM25 → 幽灵块

def search(query):
    global bm25
    bm25 = BM25Retriever.from_documents(all_documents)   # 每次查询都重建
    ...                                  # 默认空格分词，中文失效
```

### 问题在哪

1. **同步靠自觉，迟早漏**：删除只删 Chroma，BM25 里那块还在 → 用户删了文件还能被检索到，
   引用卡片指向一个「已删除」的来源。这是最容易在答辩时被抓的 bug。
2. **每次查询重建 BM25**：原实现在 1000 块时每次查询多 50-200ms（代码注释里的经验估计），
   而现在只在写操作后重建，查询只做 `get_scores`。
3. **默认分词器对中文无效**：`BM25Retriever` 默认按空格切，一整段中文是 1 个 token，
   「报销时限」这种短查询基本命中不了；改成 `lcut_for_search` 才有真正的关键词召回。
4. **`add` 不幂等**：同一文件传两次，块数翻倍、引用出现重复条目；现在是确定性 id + `upsert`，
   `tests/test_store.py::test_repeated_add_is_idempotent` 断言重复导入后仍是 2 块。
5. **metadata 里塞 None**：Chroma 只接受 `str/int/float/bool`，`None` 会抛异常；
   `_clean_metadata` 直接丢掉空值，避免 `where={"source": x}` 过滤被空串干扰。
6. **缓存不按 collection 分**：早期所有 collection 共用一个 `bm25_docs.json`，
   换模型/换 collection 时 BM25 和向量库对不上，检索结果会「串台」；
   现在文件名带 collection 名，换 collection 自动换缓存文件（旧文件已在重构中清理掉）。

## 3. 面试官可能怎么问

1. **「向量库和 BM25 是两套数据，怎么保证一致？」**
   → 所有写操作只经过 `KnowledgeStore`，在同一个 `RLock` 里先写 Chroma 再同步内存镜像和 BM25；
   启动时用「缓存条数 vs Chroma 块数」对账，不一致就从 Chroma 全量重建，所以缓存丢了也能自愈。

2. **「BM25 索引什么时候重建？为什么不是每次查询都建？」**
   → 只在 `add_documents` / `delete_source` 之后重建，并落盘 JSON；启动时读缓存。
   原实现每次查询重建，1000 块时每次多 50-200ms，而检索本身只要十几毫秒（P95 21.5ms）。

3. **「为什么关键词检索不直接用 BM25 分数 > 0 过滤？」**
   → 小语料下 idf 会等于 0：N=2、df=1 时 `log((2-1+0.5)/(1+0.5))=0`，
   真正命中的块分数也是 0，会被误杀；所以先用「查询词 ∩ 块分词」判命中，再按分数排序。