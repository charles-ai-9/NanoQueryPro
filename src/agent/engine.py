# -*- coding: utf-8 -*-
"""
AgentEngine：封装 LangGraph 图的运行时逻辑，解耦 main.py 中的底层流控代码。

核心职责：
- 持有编译好的 graph 实例
- stream_run: 异步生成器，推送流式文本、工具播报、HITL 拦截信号
- handle_approval: 封装人工审批后的状态更新与后续流转

【学习笔记】
LangGraph 的图（Graph）编译后，有两种主要的运行方式：
  1. graph.ainvoke()      - 一次性运行到结束，返回最终状态
  2. graph.astream_events() - 流式运行，每产生一个事件就 yield 出来

本文件选择第 2 种，原因是可以实现"打字机效果"和"中途拦截审批"。
"""

import logging
from typing import AsyncGenerator

# HumanMessage: 代表用户说的话
# ToolMessage:  代表工具调用的返回结果（必须跟在带 tool_calls 的 AIMessage 后面）
from langchain_core.messages import HumanMessage, ToolMessage

# 导入图的工厂函数，build_graph_with_deps 会把 checkpointer 和 store 注入图中
from src.agent.graph import build_graph_with_deps

# 获取当前模块的日志记录器，日志名称为 "src.agent.engine"
logger = logging.getLogger(__name__)

# ── 消音第三方冗余日志 ────────────────────────────────────────────────────────
# 这些库启动时会打印大量无用日志，设置为 WARNING 级别只显示警告和错误
for _lib in ["httpx", "sentence_transformers", "httpcore", "huggingface_hub"]:
    logging.getLogger(_lib).setLevel(logging.WARNING)


class AgentEngine:
    """
    Agent 运行引擎。

    【设计思路】
    把 LangGraph 图的运行、流式输出、HITL 审批三件事封装在一个类里，
    让 main.py（CLI）和 server.py（API）都能复用，不需要各自重复写逻辑。

    使用方式：
        async with AsyncSqliteSaver.from_conn_string(db) as memory:
            engine = AgentEngine(memory, store)
            async for chunk in engine.stream_run(question, config):
                ...
    """

    def __init__(self, memory=None, store=None):
        """
        初始化引擎，编译 LangGraph 图。

        :param memory: AsyncSqliteSaver 实例
                       作用：把每一步的图状态（State）持久化到 SQLite，
                       这样同一个 thread_id 的对话可以跨请求恢复历史。
                       也是实现 interrupt_before（HITL 中断）的必要条件。
        :param store:  AsyncSqliteStore 实例
                       作用：跨线程的长期记忆存储，类似"全局笔记本"，
                       不同于 memory 只存当前对话，store 可以跨对话共享数据。
        """
        # build_graph_with_deps 返回一个编译好的 CompiledGraph 对象
        # checkpointer=memory 开启状态持久化
        # interrupt_before=["sql_tools","knowledge_tools"] 在工具执行前暂停，等待人工审批
        self.graph = build_graph_with_deps(memory=memory, store=store)
        logger.info("AgentEngine 初始化完成，图已编译 ✅")

    async def stream_run(
        self, question: str | None, config: dict
    ) -> AsyncGenerator[dict, None]:
        """
        【核心方法】异步生成器：驱动图运行并产出结构化信号。

        【什么是异步生成器？】
        普通函数用 return 返回一个值就结束了。
        生成器函数用 yield 可以"暂停并产出"多个值，调用方用 async for 逐个接收。
        这里每产出一个 chunk（字典），前端就能立刻显示，实现"打字机效果"。

        产出类型（dict）：
          - {"type": "text",  "content": str}
              → 大模型流式输出的文本片段，供调用方直接打印
          - {"type": "tool_start", "name": str}
              → 工具即将启动的播报信号（此时工具还没执行）
          - {"type": "HITL_REQUIRED", "tool_calls": list, "target_node": str}
              → 遇到 interrupt_before 拦截，图已暂停，需要人工审批后才能继续

        :param question: 用户输入的问题；为 None 时表示从 HITL 中断点继续运行
        :param config:   LangGraph 运行配置，最重要的字段是 thread_id，
                         相同 thread_id 的请求会共享同一条对话历史
        """
        # 如果有新问题，包装成 HumanMessage 作为图的输入
        # 如果 question 为 None（HITL 放行后继续），传 None 让图从上次中断点恢复
        state_input = {"messages": [HumanMessage(content=question)]} if question else None

        # ── 阶段 1：流式推送大模型输出 ────────────────────────────────────────

        # 这两个节点只做内部路由/系统工作，它们的输出不是给用户看的
        # intent 节点输出 "CHAT"/"PLAN"/"BUSINESS" 这样的路由词
        # check_freshness 节点输出数据时间戳
        SKIP_NODES = {"intent", "check_freshness"}

        # 即使节点不在 SKIP_NODES 里，这些前缀的内容也属于系统内部标记，不推给用户
        SKIP_CONTENT_PREFIX = ("📝 确认意图:", "👔 【局长指令】", "摸底计划", "🎯 【当前阶段任务】")

        # 标记本轮是否产出过任何文本，用于触发兜底逻辑
        has_text = False

        # astream_events 是 LangGraph 的流式事件 API
        # version="v2" 是当前推荐版本，事件格式更丰富
        # 每个 event 是一个字典，包含 event 类型、节点名、数据等信息
        async for event in self.graph.astream_events(state_input, config, version="v2"):
            kind = event["event"]  # 事件类型，如 "on_chat_model_stream"、"on_tool_start"
            # 从元数据中取出当前事件所属的图节点名称
            node_name = event.get("metadata", {}).get("langgraph_node", "")

            # 工具开始执行时触发，此时工具还没有返回结果
            # 可以用来给用户展示"正在查询数据库..."这样的提示
            if kind == "on_tool_start":
                yield {"type": "tool_start", "name": event["name"]}

            # 大模型每产生一个 token（文字片段）就触发一次
            # 这是实现"打字机效果"的关键事件
            elif kind == "on_chat_model_stream":
                if node_name not in SKIP_NODES:
                    # event["data"]["chunk"] 是 AIMessageChunk 对象
                    # .content 是这个片段的文字内容
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content and not any(chunk_content.startswith(p) for p in SKIP_CONTENT_PREFIX):
                        has_text = True
                        yield {"type": "text", "content": chunk_content}

        # ── 阶段 1.5：兜底逻辑 ───────────────────────────────────────────────
        # 某些节点（如 intent 的 chat 分支）直接构造 AIMessage 返回，
        # 不经过 LLM 调用，所以不会触发 on_chat_model_stream 事件。
        # 此时 has_text 仍为 False，需要主动从最终 state 里读取结果。
        if not has_text:
            try:
                # aget_state 读取当前 thread_id 对应的最新图状态
                final_state = await self.graph.aget_state(config)
                msgs = final_state.values.get("messages", [])

                # 延迟导入，避免循环依赖
                from langchain_core.messages import AIMessage as _AIMsg

                # 这些前缀是系统内部用的标记消息，不应该展示给用户
                SYSTEM_PREFIXES = ("📝 确认意图:", "👔 【局长指令】", "摸底计划", "🎯 【当前阶段任务】", "【系统")

                # 从消息列表末尾往前找，找到第一条"用户可见的"AIMessage
                # 条件：是 AIMessage、有内容、没有 tool_calls（带 tool_calls 的是工具调用请求）、不是系统标记
                last_ai = next(
                    (m for m in reversed(msgs)
                     if isinstance(m, _AIMsg)
                     and m.content
                     and not m.tool_calls
                     and not any(str(m.content).startswith(p) for p in SYSTEM_PREFIXES)),
                    None
                )
                if last_ai:
                    yield {"type": "text", "content": str(last_ai.content)}
                else:
                    # 兜底的兜底：实在找不到内容，说明任务完成但没有文字总结
                    yield {"type": "text", "content": "✅ 任务已完成。"}
            except Exception as e:
                logger.warning(f"兜底读取 state 失败: {e}")
                yield {"type": "text", "content": f"⚠️ 读取结果失败: {e}"}

        # ── 阶段 2：检测是否命中 HITL 拦截点 ─────────────────────────────────
        # interrupt_before=["sql_tools","knowledge_tools"] 配置会让图在执行这些节点前暂停
        # 暂停后，current_state.next 会包含即将执行的节点名（如 "sql_tools"）
        # 最后一条消息是 AIMessage，里面的 tool_calls 就是即将执行的工具调用参数
        try:
            current_state = await self.graph.aget_state(config)
            msgs = current_state.values.get("messages", [])
            # 检查图是否处于中断等待状态，且等待的是工具节点
            if current_state.next and any("tools" in str(n) for n in current_state.next) and msgs:
                last_msg = msgs[-1]
                # 确认最后一条消息确实携带了 tool_calls（工具调用请求）
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    yield {
                        "type": "HITL_REQUIRED",
                        # tool_calls 包含工具名称和参数，前端用来展示"即将执行什么操作"
                        "tool_calls": last_msg.tool_calls,
                        # target_node 是被拦截的节点名，审批后需要告诉图从哪里继续
                        "target_node": current_state.next[0],
                    }
        except Exception as e:
            logger.warning(f"HITL 检测失败: {e}")

    async def handle_approval(
        self,
        config: dict,
        action: str,
        tool_calls: list,
        target_node: str,
        feedback: str = "",
    ) -> AsyncGenerator[dict, None]:
        """
        【HITL 审批处理方法】处理人工审批决策，更新状态并继续流式运行。

        【HITL 工作原理】
        当图在 interrupt_before 节点暂停时，图的状态被保存在 checkpointer（SQLite）里。
        人工做出决策后，我们通过 aupdate_state 把决策结果写入状态，
        然后用 stream_run(None, config) 让图从中断点继续往下跑。

        :param config:      LangGraph 运行配置（含 thread_id，用于找到暂停的图状态）
        :param action:      审批动作："y" 放行 | "n" 驳回 | "f" 指示反馈
        :param tool_calls:  被拦截的工具调用列表（来自 HITL_REQUIRED 信号）
        :param target_node: 被拦截的节点名（来自 HITL_REQUIRED 信号）
        :param feedback:    action=="f" 时的口谕文本
        """
        if action == "y":
            # ✅ 放行：不修改任何状态，直接让图从中断点继续执行
            # stream_run(None, config) 中 None 表示不注入新消息，从暂存状态恢复
            async for chunk in self.stream_run(None, config):
                yield chunk

        elif action == "n":
            # ❌ 驳回：
            # 图暂停时，最后一条消息是带 tool_calls 的 AIMessage（工具调用请求）。
            # Qwen 的协议要求：有 tool_calls 的 AIMessage 后面必须跟对应的 ToolMessage。
            # 所以驳回时，我们要为每个 tool_call 注入一条"已驳回"的 ToolMessage，
            # 关闭协议链，让图可以继续往下走（通常会重新规划）。
            rejects = [
                ToolMessage(
                    tool_call_id=tc["id"],   # 必须与 AIMessage 里的 tool_call id 对应
                    name=tc["name"],
                    content="已驳回",
                )
                for tc in tool_calls
            ]
            # aupdate_state 把这些 ToolMessage 注入图的状态
            # as_node=target_node 表示以"工具节点"的身份写入，保持状态机的正确性
            await self.graph.aupdate_state(config, {"messages": rejects}, as_node=target_node)
            # 注入后让图继续运行
            async for chunk in self.stream_run(None, config):
                yield chunk

        elif action == "f":
            # 💬 口谕反馈：两步操作
            # 第一步：和驳回一样，先注入 ToolMessage 关闭协议链
            # 第二步：再追加一条 HumanMessage，内容是探长的指导意见
            # 这样 AI 在继续运行时会读到这条指导，重新规划执行方向
            fb_tool_msgs = [
                ToolMessage(
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    content="【已收到探长口谕，暂停执行原计划】",
                )
                for tc in tool_calls
            ]
            # 把探长的口谕包装成 HumanMessage，AI 会把它当作新的用户指令
            fb_human_msg = HumanMessage(
                content=f"【探长口谕】：{feedback} 请根据此指示重新规划并执行。"
            )
            # 一次性把 ToolMessage 列表 + HumanMessage 全部写入状态
            await self.graph.aupdate_state(
                config,
                {"messages": fb_tool_msgs + [fb_human_msg]},
                as_node=target_node,
            )
            # 注入后让图继续运行
            async for chunk in self.stream_run(None, config):
                yield chunk

        else:
            # 未知动作，只记录日志，不做任何操作
            logger.warning(f"未知审批动作: {action}，已忽略")
