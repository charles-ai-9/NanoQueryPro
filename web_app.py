# -*- coding: utf-8 -*-
"""
web_app.py —— Streamlit 前端对话页面

对接 FastAPI 后端（server.py），提供：
  - 侧边栏 Session ID 管理
  - 流式打字机效果对话
  - HITL 审批交互（放行 / 驳回 / 口谕）

启动方式：
  nano_query_env/bin/streamlit run web_app.py
"""

import json
import uuid

import requests
import streamlit as st

# ── 常量 ──────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
CHAT_URL = f"{API_BASE}/api/chat"
APPROVE_URL = f"{API_BASE}/api/approve"

# ── 页面基础配置 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 星际金融风控系统",
    page_icon="🕵️",
    layout="wide",
)


# ── Session State 初始化 ───────────────────────────────────────────────────────
def _init_state():
    """初始化所有 session_state 字段，防止 KeyError。"""
    defaults = {
        "thread_id": str(uuid.uuid4())[:8],   # 当前会话 ID
        "messages": [],                         # 历史消息列表 [{"role": "user"/"assistant", "content": "..."}]
        "hitl_pending": None,                   # 待审批的 HITL 信号 dict | None
        "feedback_input": "",                   # 口谕输入框暂存值
        "show_feedback_box": False,             # 是否显示口谕输入框
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _parse_sse_stream(response: requests.Response) -> list[dict]:
    """
    解析 SSE 流式响应，返回所有 data 行解析后的 dict 列表。
    每行格式为：  data: {...json...}\n
    """
    chunks = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                try:
                    chunks.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return chunks


def _stream_sse(url: str, payload: dict) -> str:
    """
    通用 SSE 流式渲染函数。
    手动逐块渲染文本，确保错误信息和 HITL 信号都能被正确处理。
    返回完整的文本内容。
    """
    full_text = ""
    placeholder = st.empty()

    debug_chunks = []  # 收集所有原始 chunk，调试用
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if not payload_str:
                    continue
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                debug_chunks.append(chunk)
                ctype = chunk.get("type")

                if ctype == "text":
                    full_text += chunk.get("content", "")
                    placeholder.markdown(full_text + "▌")

                elif ctype == "tool_start":
                    full_text += f"\n\n`⚙️ 工具启动：{chunk.get('name', '')}`\n\n"
                    placeholder.markdown(full_text + "▌")

                elif ctype == "HITL_REQUIRED":
                    st.session_state.hitl_pending = chunk

                elif ctype == "error":
                    msg = chunk.get("message", "未知错误")
                    full_text += f"\n\n⚠️ **后端错误**：{msg}\n\n"
                    placeholder.markdown(full_text)

                elif ctype == "done":
                    break

    except requests.exceptions.ConnectionError:
        full_text = "⚠️ **无法连接后端服务**，请确认 `server.py` 已启动（端口 8000）。"
    except requests.exceptions.Timeout:
        full_text += "\n\n⚠️ **请求超时**，请重试。"
    except Exception as e:
        full_text += f"\n\n⚠️ **请求异常**：{type(e).__name__}: {e}"

    # 最终渲染（去掉光标）
    if full_text:
        placeholder.markdown(full_text)
    else:
        # 显示调试信息，帮助排查收到了什么
        debug_info = f"_（未收到文本内容）_\n\n**调试：收到 {len(debug_chunks)} 个 SSE 事件：**\n```\n"
        for c in debug_chunks:
            debug_info += f"{c}\n"
        debug_info += "```"
        placeholder.markdown(debug_info)

    return full_text


def _call_chat_stream(question: str, thread_id: str) -> str:
    """调用 /api/chat，流式渲染并返回完整文本。"""
    payload = {"question": question, "thread_id": thread_id}
    return _stream_sse(CHAT_URL, payload)


def _call_approve_stream(action: str, feedback: str = "") -> str:
    """调用 /api/approve，流式渲染并返回完整文本。"""
    pending = st.session_state.hitl_pending
    payload = {
        "thread_id": st.session_state.thread_id,
        "action": action,
        "tool_calls": pending.get("tool_calls", []) if pending else [],
        "target_node": pending.get("target_node", "") if pending else "",
        "feedback": feedback,
    }
    return _stream_sse(APPROVE_URL, payload)


# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🕵️ 星际风控系统")
    st.markdown("---")

    st.subheader("🔌 会话管理")
    # Session ID 输入框（实时同步到 session_state）
    new_id = st.text_input(
        "Session ID",
        value=st.session_state.thread_id,
        help="相同 ID 可恢复历史对话，留空后点击【新建会话】自动生成",
    )
    if new_id != st.session_state.thread_id:
        st.session_state.thread_id = new_id
        st.session_state.messages = []
        st.session_state.hitl_pending = None
        st.session_state.show_feedback_box = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 随机生成", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.hitl_pending = None
            st.session_state.show_feedback_box = False
            st.rerun()
    with col2:
        if st.button("🗑️ 清空记录", use_container_width=True):
            st.session_state.messages = []
            st.session_state.hitl_pending = None
            st.session_state.show_feedback_box = False
            st.rerun()

    st.markdown("---")
    st.caption(f"**当前 Thread ID**\n`{st.session_state.thread_id}`")

    # 健康检查
    st.markdown("---")
    st.subheader("🩺 服务状态")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        if health.get("engine_ready"):
            st.success("后端已就绪 ✅")
        else:
            st.warning("后端启动中... ⏳")
    except Exception:
        st.error("无法连接后端 ❌\n请先运行 `server.py`")

    st.markdown("---")
    st.markdown(
        """
        **使用说明**
        - 在下方输入框提问
        - 遇到 HITL 审批时选择操作
        - `q` 或 `quit` 退出当前话题
        """
    )


# ── 主页面标题 ─────────────────────────────────────────────────────────────────
st.title("🤖 星际金融风控系统")
st.caption("Text-to-SQL · 知识库检索 · Human-in-the-Loop 审批")
st.markdown("---")

# ── 渲染历史消息 ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── HITL 审批区域 ──────────────────────────────────────────────────────────────
if st.session_state.hitl_pending:
    pending = st.session_state.hitl_pending

    st.markdown("---")
    st.warning("⏸️ **[HITL 审批]** Agent 正在等待您的授权！")

    # 展示待审批的工具调用详情
    with st.expander("🔍 查看待执行的工具调用", expanded=True):
        for tc in pending.get("tool_calls", []):
            st.markdown(f"**工具名称：** `{tc.get('name', '')}`")
            st.json(tc.get("args", {}))

    # 三个审批按钮
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("✅ 放行", use_container_width=True, type="primary"):
            with st.chat_message("assistant"):
                full_text = _call_approve_stream("y")
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            st.session_state.hitl_pending = None
            st.session_state.show_feedback_box = False
            st.rerun()

    with btn_col2:
        if st.button("❌ 驳回", use_container_width=True):
            with st.chat_message("assistant"):
                full_text = _call_approve_stream("n")
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            st.session_state.hitl_pending = None
            st.session_state.show_feedback_box = False
            st.rerun()

    with btn_col3:
        if st.button("💬 下达口谕", use_container_width=True):
            st.session_state.show_feedback_box = not st.session_state.show_feedback_box
            st.rerun()

    # 口谕输入框（点击"下达口谕"后展开）
    if st.session_state.show_feedback_box:
        with st.form("feedback_form", clear_on_submit=True):
            feedback_text = st.text_area(
                "✍️ 输入口谕（探长指示）",
                placeholder="例如：请只查询最近 30 天的数据...",
                height=100,
            )
            submitted = st.form_submit_button("📤 提交口谕", type="primary", use_container_width=True)
            if submitted and feedback_text.strip():
                with st.chat_message("user"):
                    st.markdown(f"💬 **口谕**：{feedback_text}")
                st.session_state.messages.append(
                    {"role": "user", "content": f"💬 **口谕**：{feedback_text}"}
                )
                with st.chat_message("assistant"):
                    full_text = _call_approve_stream("f", feedback=feedback_text)
                st.session_state.messages.append({"role": "assistant", "content": full_text})
                st.session_state.hitl_pending = None
                st.session_state.show_feedback_box = False
                st.rerun()

    st.markdown("---")

# ── 提问输入框 ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("💬 请输入您的问题（例如：查询星际重工的逾期天数）"):
    # 渲染用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 重置上一轮 HITL 状态，开始新一轮对话
    st.session_state.hitl_pending = None
    st.session_state.show_feedback_box = False

    # 调用后端，流式渲染 Assistant 消息
    with st.chat_message("assistant"):
        full_text = _call_chat_stream(prompt, st.session_state.thread_id)

    st.session_state.messages.append({"role": "assistant", "content": full_text or ""})

    # 如果有 HITL 信号，触发重渲染以显示审批按钮
    if st.session_state.hitl_pending:
        st.rerun()

