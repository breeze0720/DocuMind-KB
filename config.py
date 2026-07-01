"""
config.py - 配置加载模块
从 .env 文件读取所有配置项，简单直白。
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# 邮件配置
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
