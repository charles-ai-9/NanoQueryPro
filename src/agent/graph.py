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

    # ════════════════════════════════════════════════════════════════════════════
    # 🔄 特工与工具的 ReAct 闭环 (Reason + Act)
    # 核心逻辑：大脑思考 -> 决定是否用工具 -> (如果是) 去用工具 -> 强制带着结果回到大脑继续思考
    # ════════════════════════════════════════════════════════════════════════════

    # ─── 🕵️‍♂️ 1. SQL 探员（数据库数据调取专家）的行动路线 ───

    # 【十字路口】：探员大脑运转完毕后，根据 tools_condition（安检员）的判断决定去向
    builder.add_conditional_edges(
        "sql_detective",  # 当前所在节点：SQL 探员的大脑
        tools_condition,  # 裁判函数：检查大模型输出中是否包含了 tool_calls（工具调用指令）
        {
            "tools": "sql_tools",  # 岔路 A：如果要用工具，就引导去 "sql_tools" 节点（去机房执行 SQL）
            END: "supervisor"  # 岔路 B：如果不用工具（查完了或放弃了），就顺着 END 路由直接向局长复命
        }
    )

    # 【强制单行道】：只要在 "sql_tools" 节点干完了活，不管成功还是报错，都必须回大脑
    builder.add_edge(
        "sql_tools",  # 起点：刚刚执行完 SQL 查询的工具节点
        "sql_detective"  # 终点：把查询结果（Observation）强制塞回探员大脑，让他结合结果进行下一轮思考
    )

    # ─── 📚 2. 知识库探员（风控手册检索专家）的行动路线 ───

    # 【十字路口】：逻辑与 SQL 探员完全一致
    builder.add_conditional_edges(
        "knowledge_agent",  # 当前所在节点：知识库探员的大脑
        tools_condition,  # 裁判函数：判断是否需要查阅资料
        {
            "tools": "knowledge_tools",  # 岔路 A：去 "knowledge_tools" 节点（调用 FAISS/BM25 混合检索）
            END: "supervisor"  # 岔路 B：资料查够了，去向局长汇报
        }
    )

    # 【强制单行道】：把检索到的文档片段强制带回给大脑
    builder.add_edge(
        "knowledge_tools",  # 起点：刚刚检索完向量数据库的工具节点
        "knowledge_agent"  # 终点：把大段参考文本喂给探员，让他总结出人类能看懂的结论
    )
    return builder


def build_graph_with_deps(memory=None, store=None):
    return _build_builder().compile(checkpointer=memory, store=store, interrupt_before=["sql_tools", "knowledge_tools"])


def build_graph():
    """供 LangGraph Studio / langgraph dev 框架调用的无参工厂函数。
    checkpointer 和 store 由框架在运行时自动注入，此处不传入。
    """
    return _build_builder().compile(interrupt_before=["sql_tools", "knowledge_tools"])
