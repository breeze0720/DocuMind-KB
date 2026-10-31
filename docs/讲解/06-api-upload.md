# 06 · API 上传加固与 SSE 流式回答

> 一句话：上传接口做了 **4 道防线 + 1 个幂等**，SSE 接口按 **sources → token… → done** 三段流式返回。
> 这一篇是两个「面试高频追问区」：上传安全和流式实现。
>
> 术语：**multipart**（浏览器上传文件用的表单编码）、**streaming**（流式，边读边写不整块进内存）、
> **SHA256**（内容哈希，用来识别「是不是同一份文件」）、**zip bomb**（压缩炸弹，几 KB 解压出几十 GB）、
> **SSE**（Server-Sent Events，服务端单向推送的文本协议）、**event**（SSE 里的一条消息类型）。

## 1. 这段代码在做什么

### 1.1 上传四道防线（按执行顺序）

| # | 防线 | 代码位置 | 失败返回 |
| --- | --- | --- | --- |
| 0 | 文件名清洗（防目录穿越） | `app/upload_utils.py · sanitize_filename` | 清洗而非拒绝 |
| 1 | 扩展名白名单（7 种） | `upload_utils.check_extension` | 400 |
| 2 | 大小预检（读 `file.size`，**不读内容**） | `upload_utils.check_size` | **413** |
| 3 | 1MB 分块流式落盘 + 边写边算 SHA256 + 累计字节兜底 | `routes_documents.upload_document` | **413** |
| 4 | zip 炸弹判定（docx/xlsx/pptx） | `upload_utils.zip_bomb_reason` | 400 |
| 5 | 内容哈希幂等（已存在就不解析、不计费） | `store.find_source_by_hash` | 200 + `duplicate: true` |

核心循环（节选 `app/routes_documents.py · upload_document`）：

```python
size_error = check_size(getattr(file, "size", None), settings.max_file_mb)   # ① 白拿 size，不读内容
if size_error:
    return _error(413, size_error)

tmp_path = upload_dir / f".upload_{uuid.uuid4().hex}.part"                   # ② 先写临时文件，成功再改名
digest = hashlib.sha256()
total = 0
oversize = False
try:
    with open(tmp_path, "wb") as handle:
        while True:
            block = await file.read(1024 * 1024)                            # ③ 1MB 一块，内存占用恒定
            if not block:
                break
            total += len(block)
            if total > settings.max_file_bytes:                             # ④ 兜底：size 拿不到时也拦
                oversize = True
                break
            digest.update(block)                                            # ⑤ 边写边算哈希，不多读一遍盘
            handle.write(block)
finally:
    await file.close()
```

- ① `UploadFile.size` 是 Starlette 解析 multipart 时就算好的，**一个 200MB 的包不用读完就能拒绝**；
- ② 临时文件名 `.upload_<uuid>.part` 以点开头，`disk_files()` 和文档列表都会跳过它；
- ③ 如果写成 `content = await file.read()`，整个文件进内存，内存占用随并发上传线性上涨；
- ④ `check_size` 在 size 为 `None`（分块传输拿不到长度）时返回 None，所以这里必须再按实际字节兜底；
- ⑤ 哈希只算一次：后面 `index_file(..., file_hash=...)` 直接复用，不重复读大文件。

### 1.2 幂等：按内容哈希，不按文件名

```python
file_hash = digest.hexdigest()
store = await run_in_threadpool(get_store, embedding)
existing = await run_in_threadpool(store.find_source_by_hash, file_hash)
if existing:
    tmp_path.unlink(missing_ok=True)          # 临时文件删掉，不落盘
    return {"success": True, "duplicate": True,
            "message": f"该文件内容已在知识库中（{existing.get('source')}，{existing.get('chunks')} 块），已跳过重复入库",
            "file": {"name": filename, "duplicate": True, "chunks": 0, "file_hash": file_hash}, ...}

final_path = upload_dir / stored_filename(file_hash, filename)   # ① 落盘名 = sha8_原文件名
os.replace(tmp_path, final_path)                                 # ② 同盘原子替换
result = await run_in_threadpool(index_file, final_path, file_hash=file_hash, source_name=filename)
```

- ① `stored_filename` = `文件哈希前 8 位 + 原文件名`：同名不同内容不会互相覆盖，人工也能核对幂等；
- ② `os.replace` 在同一个盘内是原子的，不会出现「写了一半的文件被当成完整文件解析」；
- 判重放在**解析和向量化之前**：重复入库等于白算一遍 embedding（还要重写 BM25）；
  同时 `store.add_documents` 用确定性 id + `upsert` 做了第二层保险（删了库重传也不会翻倍）。

文件名清洗的两个坑（`sanitize_filename`）：`../../etc/passwd` → `passwd`（只取最后一段，防目录穿越）；
`C:\tmp\a:b?.docx` → `a_b_.docx`（Windows 保留字符替换），并把长度截到 120。

### 1.3 zip 炸弹：只读中央目录，不解压

docx/xlsx/pptx 本质是 zip，构造一个「解压后 50GB」的包只要几百 KB。判定三条（`zip_bomb_reason`）：

```python
with zipfile.ZipFile(opened) as archive:
    infos = archive.infolist()                                  # ① 只读中央目录，不解压
    if len(infos) > max_entries:                                # ② 默认 10000 条
        return f"压缩包条目过多（{len(infos)} > {max_entries}），疑似恶意文件，已拒绝"
    total = sum(info.file_size for info in infos)               # ③ 解压后总大小
    compressed = max(1, sum(info.compress_size for info in infos))
...
if total > max_total_mb * _MB:                                  # ④ 默认 500MB
    return f"解压后总大小约 {total / _MB:.1f} MB，超过上限 {max_total_mb} MB，疑似 zip 炸弹，已拒绝"
if total / compressed > max_ratio:                              # ⑤ 默认 100:1
    return f"压缩比异常（约 {total / compressed:.0f}:1 > {max_ratio}:1），疑似 zip 炸弹，已拒绝"
```

- ① `infolist()` 读的是 zip 中央目录里**声明的** `file_size`，不解压，所以检查一个 10GB 展开的炸弹也是毫秒级；
- ②③④⑤ 三个阈值对应 `config.py` 的 `max_zip_ratio=100`、`max_decompressed_mb=500`，外加硬编码的条目数 10000；
- 非 zip 格式、或扩展名伪装成 docx 但根本不是合法 zip，直接放行，让真正的解析器报「文件损坏」——
  错误信息对用户更有用（`tests/test_upload_utils.py` 覆盖了这几种情况，包括 2MB 全零字节构造的 100:1 炸弹）。

### 1.4 SSE 三段：sources → token → done

`app/routes_qa.py · ask_stream` 负责把同步生成器桥接成 SSE：

```python
async def event_generator():
    started = time.perf_counter()
    try:
        async for event, data in iterate_in_threadpool(agent_ask_stream(question)):   # ① 同步生成器 → 异步
            if event == "done":
                data = {**data, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
            yield {"event": event, "data": json.dumps(data, ensure_ascii=False)}      # ② 统一 JSON 编码
    except Exception as exc:
        yield {"event": "error", "data": json.dumps({"message": str(exc)}, ensure_ascii=False)}

return EventSourceResponse(event_generator(), ping=15000)                             # ③ 心跳 15 秒
```

- ① `agent_ask_stream` 是同步生成器（内部调 LangChain 的 `stream`），用 `iterate_in_threadpool` 包成异步，
  否则会阻塞事件循环；`run_in_threadpool` 同理用于 upload/store 等阻塞调用；
- ② 所有 data 统一 `json.dumps`：来源列表、token 文本、done 对象都走一种编码，
  前端只需 `JSON.parse`；`ensure_ascii=False` 让中文原样传输；
- ③ `ping=15000` 是 SSE 心跳，防止中间代理把空闲连接掐断；`EventSourceResponse` 还自带断线清理。

事件语义（与 `core/agent.py · ask_stream` 对应）：

| event | data | 前端动作 |
| --- | --- | --- |
| `sources` | 来源数组（`n/source/chunk_index/page/preview/score_kind`） | 立刻渲染「参考来源」卡片 |
| `token` | 一段答案文本 | 追加到气泡（逐字输出） |
| `done` | `{answer, sources, elapsed_ms}` | 收尾、落历史、显示总耗时 |
| `error` | `{message}` | 显示失败原因 |

前端用 `fetch` 读 `res.body` 流、按 `\n\n` 切事件（`static/index.html · handleSSE`），
所以 `GET /api/ask/stream`（EventSource 只能 GET）和 `POST`（fetch 习惯）两个都支持，
参数解析统一走 `app/payloads.py · read_payload`（JSON → form → query 三种都认）。

## 2. 为什么这样写（对比错写法）

### 新手会写成什么样

```python
# 错版上传
@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()                        # ① 整文件进内存
    path = UPLOAD_DIR / file.filename                  # ② 直接用原始文件名（路径穿越/覆盖）
    open(path, "wb").write(content)
    chunks = load_and_split(path)                      # ③ 不判重、不限大小、不查 zip 炸弹
    vector_store.add_documents(chunks)

# 错版流式
@app.get("/ask/stream")
async def ask_stream(q: str):
    answer, sources = agent.ask(q)                     # ④ 先算完整答案
    return StreamingResponse(iter(answer), media_type="text/event-stream")   # ⑤ 事后切片，不是真流式
```

### 问题在哪

1. **① 内存**：不限制大小 + 整块读，50MB 文件占 50MB 内存，并发几个就 OOM；
   现在是 1MB 分块，内存恒定，且累计超限立刻中断落盘。
2. **② 文件名**：`file.filename` 可以是 `../../.env`，直接拼路径会把文件写到项目外；
   现在 `sanitize_filename` 只取 basename + 替换非法字符 + 截断，落盘名再加哈希前缀。
3. **③ 没有幂等/安全检查**：同一文件传两次块数翻倍；zip 炸弹在 `extractall` 时才爆，磁盘直接满。
   现在先 `infolist()` 判三条阈值，再按内容哈希判重。
4. **④⑤ 假流式**：先等完整回答再切片返回，首字延迟等于总耗时（检索 P95 只有 21ms，但 LLM 首 token 要 1 秒以上），
   用户体感是「卡住很久然后刷一下全出来」；真流式是模型每出一个 token 就推一条。
5. **SSE 不做 JSON 编码**：token 里有换行时会把一条事件拆成两行，前端解析全乱；统一 `json.dumps` 后换行被转义。
6. **在 async 函数里直接调阻塞代码**：`agent.ask()`、`store.search()` 都是同步阻塞，
   直接在 `async def` 里调用会卡住整个事件循环，所有并发请求一起变慢——所以处处 `run_in_threadpool`。

## 3. 面试官可能怎么问

1. **「上传大文件怎么防内存爆掉，大小限制怎么做？」**
   → 两层：先看 multipart 解析给出的 `file.size`，超 50MB 不读内容直接 413；
   再 1MB 分块 `await file.read(1MB)` 流式落盘，累计字节超限也 413（覆盖 size 缺失的情况），内存占用恒定 1MB。

2. **「上传幂等怎么实现的，为什么按哈希不按文件名？」**
   → 流式写盘时顺便算 SHA256，入库前 `find_source_by_hash` 命中就直接返回 `duplicate`（不解析、不 embedding、不计费）；
   写库用 `{file_hash[:16]}:{source}:{chunk_index}` 确定性 id + `upsert` 兜底。
   改名再传同一份内容也算重复，这是按文件名做不到的。

3. **「SSE 的三段事件怎么实现，done 里为什么还要回 sources？」**
   → `EventSourceResponse` 包一个异步生成器，逐个 `yield {"event", "data"}`；
   `sources` 先发让前端立刻渲染来源卡片，`token` 增量追加，`done` 带完整 answer + sources + elapsed_ms
   （便于前端替换/校正、落历史、展示耗时）；`ping=15000` 心跳保活，异常走 `error` 事件而不是断连。