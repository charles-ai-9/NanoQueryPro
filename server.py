# -*- coding: utf-8 -*-
"""
server.py —— FastAPI 后端服务

将 AgentEngine 的流式能力通过 HTTP 接口暴露给前端：
  POST /api/chat    → SSE 流式推送文本 / 工具播报 / HITL 拦截信号
  POST /api/approve → 接收审批决策，继续 SSE 流式输出
"""

import sys
import json
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── 路径与环境变量 ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(dotenv_path=ROOT_DIR / ".env")
warnings.filterwarnings("ignore")

# ── 消音冗余日志 ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
for _lib in ["httpx", "sentence_transformers", "httpcore", "huggingface_hub"]:
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from src.agent.engine import AgentEngine

# ── 全局运行时容器（在 lifespan 中填充）─────────────────────────────────────
_runtime: dict[str, Any] = {}

DB_PATH = str(ROOT_DIR / "data" / "memory" / "nanoquery.db")


# ── lifespan：应用启动/关闭时管理资源 ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在服务启动时初始化 Store、Saver 和 AgentEngine；
    服务关闭时自动释放数据库连接。
    """
    logger.info("🚀 服务启动中，正在初始化 AgentEngine...")
    async with AsyncSqliteStore.from_conn_string(DB_PATH) as store:
        await store.setup()
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
            _runtime["engine"] = AgentEngine(memory=memory, store=store)
            logger.info("✅ AgentEngine 初始化完毕，服务就绪。")
            yield  # ← 服务正常运行期间阻塞在此处
    logger.info("🛑 服务已关闭，资源已释放。")


# ── FastAPI 应用实例 ───────────────────────────────────────────────────────────
app = FastAPI(
    title="NanoQueryPro API",
    description="基于 LangGraph AgentEngine 的 Text-to-SQL 智能风控后端",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS 中间件：允许所有跨域请求 ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求体模型 ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """POST /api/chat 请求体"""
    question: str       # 用户提问
    thread_id: str      # 会话 ID，用于 checkpoint 隔离


class ApproveRequest(BaseModel):
    """POST /api/approve 请求体"""
    thread_id: str           # 与 /api/chat 相同的会话 ID
    action: str              # "y" | "n" | "f"
    tool_calls: list         # 从 HITL_REQUIRED 信号中透传的 tool_calls 列表
    target_node: str         # 从 HITL_REQUIRED 信号中透传的 target_node
    feedback: str = ""       # action=="f" 时的口谕文本


# ── 工具函数：将 chunk dict 序列化为 SSE 的 data 字段 ─────────────────────────
def _to_sse_data(chunk: dict) -> str:
    return json.dumps(chunk, ensure_ascii=False)


# ── POST /api/chat ────────────────────────────────────────────────────────────
@app.post("/api/chat", summary="发起提问，SSE 流式接收 Agent 输出")
async def chat(req: ChatRequest):
    """
    接收用户提问，通过 Server-Sent Events 流式推送三种事件：

    - `{"type": "text",         "content": "..."}` —— 大模型流式文本片段
    - `{"type": "tool_start",   "name": "..."}` —— 工具启动播报
    - `{"type": "HITL_REQUIRED","tool_calls": [...], "target_node": "..."}` —— HITL 拦截信号
    - `{"type": "done"}` —— 流结束标记
    """
    engine: AgentEngine = _runtime["engine"]
    config = {
        "configurable": {"thread_id": req.thread_id},
        "recursion_limit": 50,
    }

    ##### Python闭包的魔力：在 event_generator 内部访问外部的 engine 和 config 变量，无需参数传递 #####
    async def event_generator():
        try:
            ## Interact with AgentEngine and yield SSE events
            async for chunk in engine.stream_run(req.question, config):
                yield {"data": _to_sse_data(chunk)}
        except Exception as e:
            logger.error(f"stream_run 异常: {e}", exc_info=True)
            yield {"data": _to_sse_data({"type": "error", "message": str(e)})}
        finally:
            # 流结束时发送关闭标记，前端可据此关闭 EventSource
            yield {"data": _to_sse_data({"type": "done"})}

    return EventSourceResponse(event_generator())


# ── POST /api/approve ─────────────────────────────────────────────────────────
@app.post("/api/approve", summary="提交 HITL 审批决策，SSE 流式接收后续 Agent 输出")
async def approve(req: ApproveRequest):
    """
    接收人工审批决策，驱动 Agent 从中断点继续运行，SSE 推送后续输出。

    action 取值：
      - `"y"` —— 放行，直接执行工具
      - `"n"` —— 驳回，注入"已驳回" ToolMessage 后重新规划
      - `"f"` —— 口谕反馈，追加 HumanMessage 后重新规划
    """
    engine: AgentEngine = _runtime["engine"]
    config = {
        "configurable": {"thread_id": req.thread_id},
        "recursion_limit": 50,
    }

    async def event_generator():
        try:
            async for chunk in engine.handle_approval(
                config,
                req.action,
                req.tool_calls,
                req.target_node,
                req.feedback,
            ):
                yield {"data": _to_sse_data(chunk)}
        except Exception as e:
            logger.error(f"handle_approval 异常: {e}", exc_info=True)
            yield {"data": _to_sse_data({"type": "error", "message": str(e)})}
        finally:
            yield {"data": _to_sse_data({"type": "done"})}

    return EventSourceResponse(event_generator())


# ── 健康检查 ──────────────────────────────────────────────────────────────────
@app.get("/health", summary="健康检查")
async def health():
    return {"status": "ok", "engine_ready": "engine" in _runtime}


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

