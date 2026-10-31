"""
config.py - 全局配置（pydantic-settings）

为什么用 pydantic-settings：
1) 配置项不少（上传上限、切块预算、检索参数、目录），os.getenv 散落在各个文件里，
   漏写默认值/类型转错（"50" vs 50）只能在运行时才炸；
   pydantic-settings 在进程启动时一次性做类型转换和校验，配置写错立刻报错。
2) 它原生支持 .env + 环境变量覆盖（环境变量优先），改配置不用改代码。
3) 仍然保留模块级常量（MODEL_NAME / DEEPSEEK_API_KEY / EMAIL_* 等），
   让旧代码 `import config; config.MODEL_NAME` 不用改，兼容第一天的写法。
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：所有相对路径都以它为准，避免「从哪个目录启动 uvicorn」导致写进不同地方
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """所有配置项的唯一定义处。字段名自动对应同名环境变量（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",          # .env 里可能还有给别的项目用的变量，忽略即可
        case_sensitive=False,    # MODEL_NAME / model_name 都认
    )

    # ---------- DeepSeek 大模型 ----------
    model_name: str = "deepseek-chat"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 2048

    # ---------- 邮件（旧功能保留） ----------
    email_host: str = "smtp.qq.com"
    email_from: str = ""
    email_password: str = ""

    # ---------- 上传与切块 ----------
    # 50MB 是「单个文档」上限：再大就不是个人知识库场景，且解析/embedding 会占满内存
    max_file_mb: int = 50
    # 500 tokens：本地 bge-small-zh-v1.5 的 config.json 里 max_position_embeddings=512，
    # 减去 [CLS]/[SEP] 两个特殊 token，正文最多约 510；取 500 留安全余量，
    # 超过会被 tokenizer 静默截断——这是「按 token 而不是按字符切」的直接原因。
    chunk_tokens: int = 500
    chunk_overlap: int = 50

    # ---------- 向量化（固定本地 bge-small-zh-v1.5，不接云端 embedding） ----------
    # 为什么只有本地一种：
    #   ① 离线可跑，评测/演示不依赖网络和额度；② 单条 query 向量化约 20ms，比云端快一个量级；
    #   ③ 中文短文本语料上，本地 bge 的排序质量已经够用（评测见 eval_results/local.md）。
    #   云端 embedding 做过对比后没有采用，理由写在 docs/讲解/01-embeddings.md。
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    local_model_dir: Path = BASE_DIR / "models" / "bge-small-zh-v1.5"
    # collection 名：Chroma 建库时维度就固定了，名字里带维度可以避免「换模型后写进旧库」直接报错。
    # 保持 kb_bge_512 是为了现有索引（54 块）不用重建。
    collection_name: str = "kb_bge_512"

    # ---------- 检索 ----------
    top_k: int = 5
    # RRF 的平滑常数：60 是论文（Cormack et al. 2009）里的经验值，
    # 作用是压平「第 1 名 vs 第 2 名」的差距，让两路检索的排名都更平滑地融合
    rrf_k: int = 60

    # ---------- 目录 ----------
    data_dir: Path = BASE_DIR / "data"            # data/corpus 放待导入语料
    chroma_dir: Path = BASE_DIR / "chroma_db"     # Chroma 持久化 + BM25 JSON 缓存
    upload_dir: Path = BASE_DIR / "uploads"       # 用户上传的原始文件
    eval_dir: Path = BASE_DIR / "eval_results"    # 评测结果（latest.json 给前端读）

    # ---------- 上传安全 ----------
    # zip 系容器（docx/xlsx/pptx）解压总量和压缩比上限，防 zip 炸弹
    max_decompressed_mb: int = 500
    max_zip_ratio: int = 100

    # 启动时是否自动扫描 data/corpus 导入；默认关（避免启动被大语料拖慢，需要时调 /api/reindex）
    auto_index_corpus: bool = False

    @property
    def max_file_bytes(self) -> int:
        """上传大小上限（字节）。单独提供，避免每个调用点都写 * 1024 * 1024。"""
        return self.max_file_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例：整个进程只解析一次 .env。lru_cache 保证各模块拿到的是同一个对象。"""
    return Settings()


settings = get_settings()

# ============================================
# 旧代码兼容层：第一天的代码写的是 config.MODEL_NAME 这种常量
# 保留它们，`import config` 的老脚本不用改；新代码建议直接用 config.settings.xxx
# ============================================
MODEL_NAME = settings.model_name
DEEPSEEK_API_KEY = settings.deepseek_api_key
DEEPSEEK_BASE_URL = settings.deepseek_base_url
EMAIL_HOST = settings.email_host
EMAIL_FROM = settings.email_from
EMAIL_PASSWORD = settings.email_password
MAX_FILE_MB = settings.max_file_mb
TOP_K = settings.top_k
RRF_K = settings.rrf_k
DATA_DIR = settings.data_dir
CHROMA_DIR = settings.chroma_dir
EVAL_DIR = settings.eval_dir
UPLOAD_DIR = settings.upload_dir