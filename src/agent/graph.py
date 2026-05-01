"""
graph.py - NanoQuery 拓扑升级：支持工具执行循环与物理装甲
"""
import sys
import os
from src.agent.nodes import (
    check_data_freshness_node,
    generate_sql_node,
    intent_node,
    parallel_detective_node,  # 🌟 确保引入
    distribute_tasks,
    aggregate_reports_node    # 🌟 确保引入了汇总节点
)


# langgraph dev 加载此模块时 stdout 可能为 ASCII 编码
# 在此处强制重绑定为 UTF-8，防止任何中文/emoji 输出触发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from src.agent.nodes import check_data_freshness_node, generate_sql_node, intent_node
from src.agent.state import MessagesState
from src.tools.sql_tools import execute_sql, search_knowledge_base
from src.agent.subgraphs.rca_graph import rca_graph


# 意图路由器，根据当前会话状态决定流程走向
# state: 当前的消息状态对象，包含上下文信息，封装好了后续用到
def intent_router(state: MessagesState):
    """【交通警察】：根据 route 字段决定流程分支"""
    route = state.route

    # 🌟 集团篇核心逻辑：如果是并行模式，直接调用分发函数
    # distribute_tasks 会返回一个包含多个 Send 对象的列表 [Send(...), Send(...)]
    # LangGraph 看到列表里是 Send 对象，就会自动开启“影分身”并发模式
    if route == "parallel":
        return distribute_tasks(state)

    # --- 以下是原有的单兵路由逻辑 ---
    if route == "chat":
        return END
    if route == "meta":
        return "generate_sql"
    if route == "analysis":
        return "rca_subgraph"
    # 其他情况（默认 business），进入数据新鲜度检查节点
    return "check_freshness"


def _build_builder() -> StateGraph:
    builder = StateGraph(MessagesState)

    network_armor = RetryPolicy(
        initial_interval=2.0,
        backoff_factor=2.0,
        max_attempts=3
    )

    # ============== 步骤 A：注册节点 (关键修正) ==============
    builder.add_node("intent", intent_node, retry_policy=network_armor)
    builder.add_node("check_freshness", check_data_freshness_node)
    builder.add_node("generate_sql", generate_sql_node, retry_policy=network_armor)
    builder.add_node("tools", ToolNode([execute_sql, search_knowledge_base]))
    builder.add_node("rca_subgraph", rca_graph)

    # 🌟 必须注册分身探员和主编节点，否则报错 NodeNotFound
    builder.add_node("parallel_detective", parallel_detective_node)
    builder.add_node("aggregate_reports", aggregate_reports_node)

    # ============== 步骤 B：布置连线 (关键修正) ==============
    builder.add_edge(START, "intent")

    # 意图路由
    builder.add_conditional_edges(
        "intent",
        intent_router,
        {
            END: END,
            "generate_sql": "generate_sql",
            "check_freshness": "check_freshness",
            "rca_subgraph": "rca_subgraph",
            "parallel_detective": "parallel_detective"
        }
    )

    # 常规业务流
    builder.add_edge("check_freshness", "generate_sql")
    builder.add_edge("rca_subgraph", END)

    # 🌟 并行协作流：Map-Reduce 的精髓
    # 1. 所有分身干完活，必须去汇总节点集合，不能连向 END
    builder.add_edge("parallel_detective", "aggregate_reports")
    # 2. 汇总节点写完综述后，正式结案
    builder.add_edge("aggregate_reports", END)

    # --- 步骤 C：ReAct 循环 ---
    builder.add_conditional_edges(
        "generate_sql",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "generate_sql")

    return builder


def build_graph():
    """供 LangGraph Studio/Server 框架调用。
    新版 langgraph-api >= 0.7.95 要求工厂函数必须无参数，
    checkpointer 和 store 由框架在运行时自动注入。"""
    return _build_builder().compile(interrupt_before=["tools"])


def build_graph_with_deps(memory=None, store=None):
    """供 main.py 本地运行调用。
    手动传入 checkpointer (memory) 和 store，实现持久化与长期记忆。"""
    return _build_builder().compile(
        checkpointer=memory,
        store=store,
        interrupt_before=["tools"]
    )
