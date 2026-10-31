"""
main.py - FastAPI 主入口（v2：分层后的 RAG 知识库问答服务）

启动方式：
    D:\\Anaconda\\envs\\xhu\\python.exe main.py                 # 默认 8000（用户 PyCharm 的用法）
    D:\\Anaconda\\envs\\xhu\\python.exe -m uvicorn main:app --port 8010   # 自测/联调用 8010
接口文档：http://127.0.0.1:<port>/docs

分层：main.py（应用装配）→ app/（HTTP 路由、上传、入库）→ core/（RAG 链路：loader/splitter/
embeddings/store/retriever/citations/agent）。旧模块 agent.py / vector_store.py / document_loader.py
保留为兼容 shim，老脚本 import 不会坏。

为什么要 lifespan 里延迟/容错初始化：
- 本地 embedding 模型加载要 1-2 秒，放在启动阶段预热，第一个查询不用等；
- 但初始化失败（模型文件缺失）不能直接让服务起不来——
  用户还需要打开 /api/status 看到清晰的中文原因，而不是一个连不上的端口。
"""
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# 抑制 PyCharm 调试模式下 NumPy 重复加载的警告（原项目就有这个问题）
warnings.filterwarnings("ignore", message=".*The NumPy module was reloaded.*")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import config
from app.api import api_router
from app.ingest import index_corpus
from core.agent import init_agent
from core.runtime import init_runtime

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用启动/关闭。阻塞操作（加载模型、扫描语料）都丢到线程池，不卡事件循环。"""
    print("=" * 60)
    print("  个人知识库智能问答助手 v2 启动中...")
    print("=" * 60)
    settings = config.settings
    for directory in (settings.upload_dir, settings.chroma_dir, settings.data_dir, settings.eval_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"[启动] 上传目录: {settings.upload_dir}")
    print(f"[启动] 向量库目录: {settings.chroma_dir}")

    await run_in_threadpool(init_runtime)

    if settings.auto_index_corpus:
        # 默认关闭；需要开机导入 data/corpus 时在 .env 设 AUTO_INDEX_CORPUS=true
        try:
            result = await run_in_threadpool(index_corpus)
            print(f"[启动] 语料自动导入: {result['indexed']} 个新文件，共 {result['total_chunks']} 块")
        except Exception as exc:
            print(f"[启动] 语料自动导入失败（服务继续启动）: {exc}")

    try:
        init_agent()
    except Exception as exc:
        print(f"[启动] Agent 初始化失败（问答接口会返回错误提示）: {exc}")

    print("=" * 60)
    print("  服务已就绪！")
    print("  接口文档: http://127.0.0.1:8000/docs（自测用 8010 端口）")
    print("=" * 60)
    yield


app = FastAPI(
    title="个人知识库智能问答助手",
    description="RAG + Agent：本地 bge 向量化、混合检索（向量+BM25+RRF）、引用溯源、检索评测",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """首页：直接返回 static/index.html（同事重写前端时只改这个文件，不用改后端）。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.is_file():
        return html_path.read_text(encoding="utf-8")
    return "<h1>页面未找到</h1><p>static/index.html 不存在。</p>"


if __name__ == "__main__":
    import uvicorn

    # 默认仍是 8000（保持用户 PyCharm 习惯）；自测用 PORT=8010 或 --port 8010，不要动别人的进程
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))