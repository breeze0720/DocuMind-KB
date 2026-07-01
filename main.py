"""
main.py - FastAPI 主入口
个人知识库智能问答助手 Web 服务

启动方式：D:/Anaconda/envs/xhu/python.exe main.py
访问地址：http://127.0.0.1:8000
接口文档：http://127.0.0.1:8000/docs
"""
import os
import uuid
import warnings
from pathlib import Path

# 抑制 PyCharm 调试模式下 NumPy 重复加载的警告
warnings.filterwarnings("ignore", message=".*The NumPy module was reloaded.*")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import config
from document_loader import load_and_split
from vector_store import init_store, add_documents, get_document_count, delete_documents_by_source
from agent import init_agent, ask as agent_ask, clear_history

# ============================================
# 配置
# ============================================
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

# ============================================
# 创建 FastAPI 应用
# ============================================
app = FastAPI(
    title="个人知识库智能问答助手",
    description="基于 RAG + Agent 的智能问答系统，支持文档上传、知识库搜索、邮件发送",
    version="1.0.0"
)


# ============================================
# 启动事件
# ============================================
@app.on_event("startup")
async def startup():
    """应用启动时自动初始化"""
    print("=" * 60)
    print("  个人知识库智能问答助手 启动中...")
    print("=" * 60)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    print(f"[启动] 上传目录: {UPLOAD_DIR}")
    print(f"[启动] 向量数据库目录: {CHROMA_DIR}")

    init_store(str(CHROMA_DIR))
    print(f"[启动] 知识库当前文档块数: {get_document_count()}")

    init_agent()

    print("=" * 60)
    print("  服务已就绪！")
    print("  接口文档: http://127.0.0.1:8000/docs")
    print("=" * 60)


# ============================================
# API 接口
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """首页 - 聊天界面"""
    html_path = BASE_DIR / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>页面未找到</h1>"


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档到知识库。支持 PDF、Word(.docx)、Markdown(.md)、TXT 格式。
    上传后自动解析、分块、存入向量数据库。
    """
    # 检查文件类型
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": f"不支持的文件格式「{ext}」，仅支持 {list(ALLOWED_EXTENSIONS)}"
            }
        )

    # 保存文件
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = UPLOAD_DIR / unique_name

    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        print(f"[上传] 已保存: {save_path} ({len(content)} 字节)")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"文件保存失败: {str(e)}"}
        )

    # 加载并切分文档
    try:
        chunks = load_and_split(str(save_path))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"文档解析失败: {str(e)}"}
        )

    if not chunks:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "文档内容为空，无法处理"}
        )

    # 存入向量数据库
    try:
        add_documents(chunks, str(CHROMA_DIR))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"向量存储失败: {str(e)}"}
        )

    return {
        "success": True,
        "message": f"文档「{file.filename}」上传成功！",
        "文件名": file.filename,
        "本次添加块数": len(chunks),
        "知识库总块数": get_document_count()
    }


@app.post("/delete")
async def delete_document(filename: str = Form(...)):
    """
    从知识库中删除指定文件。
    会同时删除磁盘上的文件和向量数据库中的对应文档块。
    """
    # 在 uploads 目录中查找匹配的文件
    found = None
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("*"):
            if f.name == ".gitkeep":
                continue
            # 文件名格式: {uuid}_{原始文件名}
            parts = f.name.split("_", 1)
            if len(parts) == 2 and parts[1] == filename:
                found = f
                break
            elif f.name == filename:
                found = f
                break

    if found is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"未找到文件「{filename}」"}
        )

    try:
        # 1. 删除向量数据库中的文档块
        deleted_chunks = delete_documents_by_source(str(found), str(CHROMA_DIR))
        print(f"[删除] 从向量库删除 {deleted_chunks} 个文档块")

        # 2. 删除磁盘上的文件
        found.unlink()
        print(f"[删除] 已删除文件: {found}")

        return {
            "success": True,
            "message": f"文件「{filename}」已删除",
            "删除文档块数": deleted_chunks,
            "知识库总块数": get_document_count()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"删除失败: {str(e)}"}
        )


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """
    向智能助手提问。Agent 会自动搜索知识库并组织回答。
    如果需要发送邮件，Agent 会自行调用邮件工具。
    """
    if not question.strip():
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "问题不能为空"}
        )

    try:
        answer = agent_ask(question.strip())
        return {
            "success": True,
            "问题": question,
            "回答": answer
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"处理失败: {str(e)}"}
        )


@app.get("/status")
async def status():
    """查看知识库和系统状态"""
    # 获取已上传文件列表（排除 .gitkeep）
    uploaded_files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("*"):
            if f.name != ".gitkeep":
                # 从文件名中提取原始文件名（去掉前缀的 uuid_）
                original_name = f.name.split("_", 1)[-1] if "_" in f.name else f.name
                uploaded_files.append({
                    "文件名": original_name,
                    "大小": f.stat().st_size,
                    "上传时间": f.stat().st_mtime
                })

    return {
        "知识库文档块数": get_document_count(),
        "向量数据库路径": str(CHROMA_DIR),
        "上传目录": str(UPLOAD_DIR),
        "已上传文件数": len(uploaded_files),
        "已上传文件列表": uploaded_files,
        "当前模型": config.MODEL_NAME,
        "邮件配置": "已配置" if (config.EMAIL_FROM and config.EMAIL_FROM != "your_email@qq.com") else "未配置"
    }


@app.post("/clear")
async def clear_chat():
    """清空对话历史"""
    clear_history()
    return {"success": True, "message": "对话历史已清空"}


# ============================================
# 直接运行入口
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
