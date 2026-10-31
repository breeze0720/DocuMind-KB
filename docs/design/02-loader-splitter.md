# 02 · Loader 与 Splitter：从文件到可检索的块

> 一句话：**loader 负责「把文件拆成带出处的内容单元」，splitter 负责「按 token 预算把单元切成块」**。
> 这两步决定了检索的原料质量，metadata 决定了引用能不能说清「哪份文件第几页第几块」。
>
> 术语：**unit**（内容单元，如 PDF 的一页、Excel 的一个 sheet）、**chunk**（切块后的检索单位）、
> **token**（模型输入的最小单位，中文约等于一个字）、**metadata**（随块一起存的来源信息）、
> **overlap**（相邻块的重叠内容）。

## 1. 实现概览

### 1.1 loader：一个文件 → 多个带出处的单元

`core/loader.py` 用一张 `_PARSERS` 分派表支持 7 种扩展名：
`.pdf / .docx / .md / .txt / .xlsx / .pptx / .csv`（`ALLOWED_EXTENSIONS`，上传白名单和语料导入共用这一份）。
每个 parser 返回 `[(文本, 来源元数据), ...]`：

| 格式 | 切单元的方式 | 额外 metadata |
| --- | --- | --- |
| pdf | 一页一个单元（`pypdf`，抽文字层，扫描件会是空） | `page` |
| docx | 按「标题→下一个标题」分节；表格单独成单元（单元格用 ` \| ` 拼） | `section` |
| md | 按 `#` 标题分节，保留标题文字、去掉 `#`（`#` 对 embedding 是噪声） | `section` |
| txt | 整篇一个单元 | — |
| csv | 一行一个单元，并补列名（`年假天数: 10` 比裸的 `10` 可检索） | `row` |
| xlsx | 一个 sheet 一个单元；`read_only=True` 避免大表全读进内存 | `sheet` |
| pptx | 一张幻灯片一个单元，抽文本框 + 表格 | `slide` |

解析完统一在 `load_documents()` 里补三个**必须有**的字段：

```python
def load_documents(path, file_hash=None) -> list[Document]:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFormatError(f"不支持的文件格式「{ext}」，……")   # ① 报 400 而不是 500
    digest = file_hash or file_sha256(file_path)                    # ② 上传流程算过就复用，大文件不读两遍
    units = parser(file_path)
    if not units:
        return []                                                    # ③ 扫描件/空文档：返回空，由上层提示
    title = _guess_title(units, file_path)                           # ④ 第一章标题 > 首行 > 文件名

    documents = []
    for text, extra_metadata in units:
        if not (text or "").strip():
            continue                                                 # ⑤ 纯空白单元直接丢，别让它进库
        metadata = {"source": file_path.name, "file_hash": digest, "title": title}
        metadata.update({k: v for k, v in extra_metadata.items() if v is not None})
        documents.append(Document(page_content=text.strip(), metadata=metadata))
    return documents
```

- ① 不支持格式抛自定义 `UnsupportedFormatError`，路由层转 400，用户看到「不支持 .exe」而不是 500；
- ② `file_sha256` 本身是 1MB 分块流式计算，重复调用不会爆内存；传参只是省一次磁盘 IO；
- ③ 空单元列表必须返回 `[]` 而不是抛异常——扫描版 PDF 是常见输入，要能提示「没抽到文字」；
- ④ 标题用于引用卡片和文档列表，猜不到就退回文件名；
- ⑤ 空块一旦进库，会成为「永远检索不到但占位置」的垃圾数据。

### 1.2 splitter：按 token 预算切，尽量在句末断开

`core/splitter.py` 的核心决策是**按 token 而不是按字符**，依据写在模块头注释里：
本地 bge-small-zh-v1.5 的 `config.json` 里 `max_position_embeddings=512`，**超过 512 token 会被
tokenizer 静默截断**——不报错、日志也没有，多出来的内容永远检索不到。所以配置成
`chunk_tokens=500` + `chunk_overlap=50`（留出 `[CLS]`/`[SEP]` 的余量）。

token 怎么数（不引入 tiktoken：它是 GPT 的 BPE，和 bge 的 WordPiece 词表不是一回事）：

```python
def count_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))                      # ① 中文 1 字 ≈ 1 token
    rest = _CJK_RE.sub(" ", text)
    ascii_tokens = sum(-(-len(w) // 4) for w in _ASCII_WORD_RE.findall(rest))  # ② 英文按 4 字符 ≈ 1 token
    other = sum(1 for ch in rest if not ch.isspace() and not ch.isalnum())     # ③ 标点各算 1
    return cjk + ascii_tokens + other
```

切块算法是「**句子级贪心装箱**」：句子是最小单位，只有单句本身超预算才硬切。

```python
for sent, tokens in pairs:
    if tokens > max_tokens:                    # ① 单句超预算：先吐缓冲区，再按字符硬切
        if buf:
            chunks.append(join_buffer())
            buf, buf_tokens = [], 0
        chunks.extend(_hard_split(sent, max_tokens, overlap))
        continue

    if buf and buf_tokens + tokens > max_tokens:
        chunks.append(join_buffer())           # ② 当前块装不下了，先落盘
        buf = tail_sentences(buf)              # ③ 取尾部若干整句带到下一块开头 = 句级 overlap
        buf_tokens = sum(t for _, t in buf)

    buf.append((sent, tokens))                 # ④ 继续装
    buf_tokens += tokens
```

- ① `_hard_split` 用二分找「最多保留多少字符仍不超预算」，硬切只用于无标点的超长文本（表格行、URL）；
- ②③ 最关键：overlap 不是按字符尾截（那会从句子中间劈开），而是**取上一块末尾的完整句子**，
  总长不超过 50 token；`tests/test_splitter.py` 里专门断言「相邻块都有重叠句、且每块以句号结尾」；
- ④ 分句正则只认中文标点 `。！？；`、英文 `!?;` 和换行，**故意不切英文句点**——
  否则 `3.5%`、`e.g.`、`No.1` 会被切碎。

最后 `split_documents()` 给每块补齐 metadata，其中 `chunk_index` 是**整篇文档全局递增**：

```python
item.update({
    "source": metadata.get("source", ""),
    "file_hash": metadata.get("file_hash", ""),
    "title": title,
    "chunk_index": chunk_index,        # 整篇文档内递增，不是每页从 0 开始
    "char_len": len(piece),
    "token_count": count_tokens(piece),
})
chunk_index += 1
```

### 1.3 metadata 每一项都是必须的

| 字段 | 谁在用 | 为什么不能省 |
| --- | --- | --- |
| `source` | 引用卡片、删除接口、文档列表 | 用户要看「《员工手册.docx》第 3 块」，不能给内部路径 |
| `file_hash` | 上传幂等、确定性 id、重导入 | 唯一能识别「同一内容改个名再传」的标识 |
| `chunk_index` | 引用编号 key、稳定排序、删除 | 同一篇里定位到块；每页从 0 开始就会出现多个「第 0 块」 |
| `page/sheet/slide/section` | 引用可读性 | 「第 3 页」「sheet:权限矩阵」比「第 12 块」有用得多 |
| `title` | 引用卡片、文档列表 | 章节名/首行比文件名更能表达内容 |
| `char_len/token_count` | 调参、排查切块质量 | 能一眼看出有没有超 500 token 的块 |

实测规模：`data/corpus` 10 份文档（约 1 万字）切出 **54 块**，这是评测报告里「知识库块数：54」的来源。

## 2. 设计取舍与备选方案

### 反例：常见错误写法

```python
# 错版一：按字符切（原实现）
def split_text(text, chunk_size=500, overlap=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

# 错版二：整篇一个 Document，不切
documents = [Document(page_content=整篇文本, metadata={"source": 文件绝对路径})]
```

### 反例的问题

1. **按字符切 → 内容被静默截断**：500 个中文字符 ≈ 500 token 已经贴上限，中英混排时同样 500 字符可能是
   300~700 token；超过 512 的部分进不了向量，检索永远搜不到，而且**不会报任何错**。
2. **字符切会把句子劈开**：「年假为每年 / 十天」这种块，embedding 语义已经残缺；引用给用户看也很怪。
   现在边界落在句末，`overlap` 也是整句重叠。
3. **不切 → 直接不可用**：一篇 3000 字的文档只产生 1 个向量，语义被平均成一个点，
   问什么都只能命中「整篇」，引用精确到「第 0 块」毫无意义。
4. **metadata 只存全路径**：引用里出现 `<工作目录>\员工手册.docx`；而且换机器/改目录后
   删除和展示都失效——现在 `source` 存文件名，`source_key` 才存原始值（见 03 篇的兼容处理）。
5. **空块/空白块入库**：白白增加块数，还会让「块数」这个指标失真；现在 loader 和 splitter 双层过滤空文本。
6. **每页 chunk_index 从 0 开始**：`(source, chunk_index)` 无法唯一定位，删除/引用都会错位。

## 3. 常见问题（FAQ）

1. **「为什么按 token 切，不按字符切？」**
   → bge-small-zh-v1.5 的 `max_position_embeddings=512`，超长输入被 tokenizer 静默截断；
   配置 500 token + 50 overlap 留余量；用近似估算（中文 1 字≈1 token、英文 4 字符≈1 token），
   10 份约 1 万字语料切出 54 块，全部 ≤500 token。

2. **「overlap 怎么做才不会切断句子？」**
   → 先分句（中文 `。！？；` + 换行），切块以整句为最小单位；块满时把上一块**尾部的完整句子**
   带到下一块开头，总长不超过 50 token；只有单句本身超预算才走二分硬切。

3. **「metadata 里为什么必须有 file_hash 和 chunk_index？」**
   → `file_hash` 是内容级去重依据（改名不重复入库）并参与确定性 id；
   `chunk_index` 配合 `source` 提供块级唯一坐标，引用编号与删除都靠它，
   且必须整篇递增，否则同一文档会出现多个「第 0 块」。