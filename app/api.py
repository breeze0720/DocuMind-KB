"""
app/api.py - 路由汇总

按职责把路由拆成三个文件（documents / qa / system），这里只负责拼装：
- 单个文件控制在 250 行以内，一个文件只负责一类接口；
- 以后加接口只改对应文件，不用在 500 行的 main.py 里翻半天。
"""
from fastapi import APIRouter

from app.routes_documents import router as documents_router
from app.routes_qa import router as qa_router
from app.routes_system import router as system_router

api_router = APIRouter()
api_router.include_router(documents_router)
api_router.include_router(qa_router)
api_router.include_router(system_router)