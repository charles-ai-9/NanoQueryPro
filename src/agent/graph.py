from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.agent.state import MessagesState
from src.tools.sql_tools import execute_sql, search_knowledge_base
from src.agent.nodes import (
    intent_node, supervisor_node, generate_sql_node, knowledge_node,
    planner_node, task_dispatcher_node, safe_get
)


def intent_router(state: MessagesState):
    """【交通警察】：精准分发路由"""
    route = safe_get(state, "route", "chat")
    if route == "plan":
        return "planner"
    if route == "chat":
        return END
    # 根据注册的节点名，这里走向局长
    return "supervisor"


def supervisor_router(state: MessagesState):
    """【局长传送门】：拦截早退"""
    next_action = safe_get(state, "route", "FINISH")
    if next_action == "FINISH":
        if safe_get(state, "plan", []):
            return "task_dispatcher"
        return END
    return next_action


def _build_builder() -> StateGraph:
    builder = StateGraph(MessagesState)

    # 注册节点
    builder.add_node("intent", intent_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sql_detective", generate_sql_node)
    builder.add_node("knowledge_agent", knowledge_node)
    builder.add_node("planner", planner_node)
    builder.add_node("task_dispatcher", task_dispatcher_node)
    builder.add_node("sql_tools", ToolNode([execute_sql]))
    builder.add_node("knowledge_tools", ToolNode([search_knowledge_base]))

    # 连线
    builder.add_edge(START, "intent")

    # 🚨 修正这里的连线，使用标准路由器，支持正确的 END
    builder.add_conditional_edges(
        "intent",
        intent_router,
        {
            "planner": "planner",
            "supervisor": "supervisor",
            END: END
        }
    )

    builder.add_edge("planner", "task_dispatcher")
    builder.add_edge("task_dispatcher", "supervisor")

    builder.add_conditional_edges("supervisor", supervisor_router, {
        "sql_detective": "sql_detective",
        "knowledge_agent": "knowledge_agent",
        "task_dispatcher": "task_dispatcher",
        END: END
    })

    # 特工与工具闭环
    builder.add_conditional_edges("sql_detective", tools_condition, {"tools": "sql_tools", END: "supervisor"})
    builder.add_edge("sql_tools", "sql_detective")
    builder.add_conditional_edges("knowledge_agent", tools_condition, {"tools": "knowledge_tools", END: "supervisor"})
    builder.add_edge("knowledge_tools", "knowledge_agent")

    return builder


def build_graph_with_deps(memory=None, store=None):
    return _build_builder().compile(checkpointer=memory, store=store, interrupt_before=["sql_tools", "knowledge_tools"])