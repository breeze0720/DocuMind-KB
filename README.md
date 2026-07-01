# 个人知识库智能问答助手

基于 **RAG（检索增强生成）+ AI Agent** 的智能问答系统。上传你的文档（PDF/Word/Markdown/TXT），即可向 AI 提问文档中的任何内容。支持对话记忆和邮件发送。

**技术栈**: FastAPI + LangChain + DeepSeek + ChromaDB + FastEmbed

## 功能特性

- **文档上传**：支持 PDF、Word (.docx)、Markdown、TXT 四种格式
- **混合检索**：向量相似度 + BM25 关键词匹配，RRF 融合算法确保检索准确性
- **智能对话**：基于 LangChain Agent，支持上下文记忆（最近 20 轮）
- **邮件发送**：Agent 可自动将回答通过 QQ 邮箱发送给指定收件人
- **Web 界面**：内置美观的聊天 UI，支持文件管理（上传/删除）
- **轻量嵌入**：使用 FastEmbed（ONNX Runtime），无需 PyTorch/sklearn 等重型依赖

## 项目结构

```
Agent Project1 test/
├── main.py              # FastAPI 主入口，定义所有 API 接口
├── agent.py             # LangChain Agent 核心（工具定义 + LLM + 对话管理）
├── vector_store.py      # ChromaDB 向量存储 + 混合检索（RRF）
├── document_loader.py   # 文档加载与分块（PDF/Word/Markdown/TXT）
├── config.py            # 从 .env 加载配置
├── requirements.txt     # Python 依赖
├── static/
│   └── index.html       # 前端聊天界面
├── uploads/             # 用户上传的文件（不上传 Git）
└── chroma_db/           # 向量数据库持久化目录（不上传 Git）
```

## 本地运行

### 前置条件

- **Python 3.10+**
- **DeepSeek API Key**（注册地址：[platform.deepseek.com](https://platform.deepseek.com)）
- QQ 邮箱授权码（可选，用于邮件发送功能）

### 安装步骤

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd "Agent Project1 test"

# 2. 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
# 复制下面的内容，创建 .env 文件并填入你的信息
```

### 配置 .env 文件

在项目根目录创建 `.env` 文件，内容如下：

```ini
# DeepSeek API 配置（必填）
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat

# HuggingFace 镜像（国内访问加速，可选）
HF_ENDPOINT=https://hf-mirror.com

# 邮件发送配置（QQ邮箱，可选，不填则无法发送邮件）
EMAIL_HOST=smtp.qq.com
EMAIL_FROM=your_email@qq.com
EMAIL_PASSWORD=your_qq_authorization_code
```

> 安全提示：`.env` 文件已在 `.gitignore` 中排除，不会被上传到 GitHub。请勿将 API Key 硬编码在代码中。

### 启动服务

```bash
python main.py
```

启动后访问：
- 聊天界面：http://127.0.0.1:8000
- API 文档（Swagger）：http://127.0.0.1:8000/docs

### 使用指南

1. 打开浏览器访问 http://127.0.0.1:8000
2. 在左侧边栏上传你的文档（PDF/Word/Markdown/TXT）
3. 上传成功后，在下方的输入框中向 AI 提问
4. AI 会从你上传的文档中检索相关内容并回答
5. 如需发送邮件，告诉 AI "把以上内容发给 xxx@example.com"

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天界面 |
| POST | `/upload` | 上传文档到知识库 |
| POST | `/delete` | 从知识库删除文档 |
| POST | `/ask` | 向 AI 提问 |
| GET | `/status` | 查看知识库状态 |
| POST | `/clear` | 清空对话历史 |
| GET | `/docs` | Swagger API 文档 |

## 注意事项

- 首次启动会自动下载嵌入模型（约 100MB），请耐心等待
- 嵌入模型使用 HuggingFace 镜像（国内加速），如需取消可在 `.env` 中删除 `HF_ENDPOINT`
- ChromaDB 数据存储在 `chroma_db/` 目录，删除该目录即清除所有知识库数据
- QQ 邮箱需要使用**授权码**而非 QQ 密码，获取方式：QQ 邮箱 -> 设置 -> 账户 -> POP3/SMTP 服务 -> 生成授权码
