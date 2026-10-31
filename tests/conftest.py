"""
tests/conftest.py - pytest 公共配置

把项目根目录放进 sys.path：`python -m pytest` 从项目根运行时本来就在 sys.path 里，
但从 IDE/其他目录跑 pytest 时不一定。显式加一次，保证 import config / core / app 都稳定。
"""
import os
import sys
from pathlib import Path

# Chroma 默认会发匿名遥测；测试环境不需要，也避免离线机器上白等网络超时
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))