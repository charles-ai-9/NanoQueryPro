import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from src.agent.nodes import (
    check_data_freshness_node,
    generate_sql_node,
    intent_node,
    parallel_detective_node,
    distribute_tasks,
    aggregate_reports_node,
    supervisor_node,
    supervisor_router,
    knowledge_node
)
from src.agent.state import MessagesState
from src.tools.sql_tools import execute_sql, search_knowledge_base
from src.agent.subgraphs.rca_graph import rca_graph


def intent_router(state: MessagesState):
    """【交通警察】：根据 route 字段决定流程分支"""
    route = state.route
    if route == "parallel":
        return distribute_tasks(state)
    if route == "chat":
        return END
    if route == "analysis":
        return "rca_subgraph"
    # 🌟 集团篇：复杂业务查询和查表结构，全权交给局长调度！
    return "supervisor"


def _build_builder() -> StateGraph:
    builder = StateGraph(MessagesState)
    network_armor = RetryPolicy(initial_interval=2.0, backoff_factor=2.0, max_attempts=3)

    # ============== 步骤 A：注册节点 ==============
    builder.add_node("intent", intent_node, retry_policy=network_armor)
    builder.add_node("check_freshness", check_data_freshness_node)

    # 🌟 新增局长和探员
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sql_detective", generate_sql_node, retry_policy=network_armor)
    builder.add_node("knowledge_agent", knowledge_node, retry_policy=network_armor)

    # 🚨 核心修复：为两名特工分配各自专属的工具箱，绝不串岗
    builder.add_node("sql_tools", ToolNode([execute_sql]))
    builder.add_node("knowledge_tools", ToolNode([search_knowledge_base]))

    builder.add_node("rca_subgraph", rca_graph)
    builder.add_node("parallel_detective", parallel_detective_node)
    builder.add_node("aggregate_reports", aggregate_reports_node)

    # ============== 步骤 B：布置连线 ==============
    builder.add_edge(START, "intent")

    # 1. 意图路口
    builder.add_conditional_edges(
        "intent",
        intent_router,
        {
            END: END,
            "supervisor": "supervisor",
            "rca_subgraph": "rca_subgraph",
            "parallel_detective": "parallel_detective"
        }
    )

    # 2. 局长传送门
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "sql_detective": "sql_detective",
            "knowledge_agent": "knowledge_agent",
            END: END
        }
    )

    # 3. 🚨 核心闭环：干完活必须回局长办公室！

    # --- SQL 探员专线 ---
    builder.add_conditional_edges(
        "sql_detective",
        tools_condition,
        {
            "tools": "sql_tools",  # 如果触发工具，强制定向到 sql_tools
            END: "supervisor"  # 如果没触发工具（纯文本结论），回见局长
        }
    )
    builder.add_edge("sql_tools", "sql_detective")  # 用完工具必须滚回 SQL 办公室

    # --- 知识探员专线 ---
    builder.add_conditional_edges(
        "knowledge_agent",
        tools_condition,
        {
            "tools": "knowledge_tools",  # 如果触发工具，强制定向到 knowledge_tools
            END: "supervisor"  # 汇报给局长
        }
    )
    builder.add_edge("knowledge_tools", "knowledge_agent")  # 用完工具必须滚回知识办公室

    # 其他分支
    builder.add_edge("rca_subgraph", END)
    builder.add_edge("parallel_detective", "aggregate_reports")
    builder.add_edge("aggregate_reports", END)

    return builder


def build_graph():
    # 🚨 拦截点更新：监控两个独立的工具箱
    return _build_builder().compile(interrupt_before=["sql_tools", "knowledge_tools"])


def build_graph_with_deps(memory=None, store=None):
    # 🚨 拦截点更新：监控两个独立的工具箱
    return _build_builder().compile(
        checkpointer=memory,
        store=store,
        interrupt_before=["sql_tools", "knowledge_tools"]
    )