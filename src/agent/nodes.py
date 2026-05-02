import logging
import traceback
from functools import lru_cache
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from .state import MessagesState, IntentOutput, SupervisorDecision, Plan
from src.core.llm_client import get_llm
from src.tools.sql_tools import execute_sql, search_knowledge_base

logger = logging.getLogger(__name__)


# ================== 🛡️ 全局防弹组件 ==================

def safe_get(state, key, default=None):
    if isinstance(state, dict): return state.get(key, default)
    return getattr(state, key, default)


def sanitize_history(messages):
    """
    🚨 绝对无菌协议洗白器 (Rebuild Everything)
    彻底销毁原有的 AIMessage 和 ToolMessage 对象，提取纯文本重新实例化。
    绝不保留任何 additional_kwargs 隐藏属性，让 Qwen 只看到纯粹的文字对话历史。
    """
    clean_msgs = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            # 🔪 提取纯文本，如果为空则给占位符。创建一个崭新的 AIMessage。
            content = msg.content if msg.content else "【系统日志】：发起了一次数据检索请求。"
            # 确保内容是字符串格式
            if not isinstance(content, str):
                content = str(content)
            clean_msgs.append(AIMessage(content=content))

        elif isinstance(msg, ToolMessage):
            # 🔪 彻底抛弃 ToolMessage，转生为 HumanMessage
            content = msg.content if msg.content else "无结果"
            clean_msgs.append(HumanMessage(content=f"📊 [工具检索结果]:\n{content}"))

        else:
            # HumanMessage 和 SystemMessage 直接保留
            clean_msgs.append(msg)

    return clean_msgs


@lru_cache(maxsize=1)
def get_llm_with_tools():
    _llm = get_llm()
    return _llm.bind_tools([execute_sql, search_knowledge_base])


# ================== 核心节点逻辑 ==================

async def intent_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """【意图中心】：返璞归真，使用最稳健的文本匹配"""
    _llm = get_llm()
    messages = safe_get(state, "messages", [])
    if not messages: return {"route": "chat"}

    system_prompt = """你是指挥中心的意图路由器。请根据探长的输入，判断意图并【严格只输出以下英文单词之一】，绝不要有任何标点符号或其他废话：
    - PLAN: 包含先后顺序的复合任务（例如“先查...再查...最后写...”）
    - CHAT: 闲聊或打招呼
    - BUSINESS: 常规的业务查询、查数或查手册
    """

    try:
        # 直接拿纯文本，彻底抛弃 with_structured_output 的不稳定解析
        res = await _llm.ainvoke([SystemMessage(content=system_prompt)] + messages[-1:])
        res_text = res.content.strip().upper()

        # 稳如老狗的文本匹配
        if "PLAN" in res_text:
            route = "plan"
        elif "CHAT" in res_text:
            route = "chat"
        else:
            route = "business"

        if route == "chat":
            return {"route": "chat", "messages": [AIMessage(content="收到，探长！")]}
        return {"route": route, "messages": [AIMessage(content=f"📝 确认意图: {route.upper()}")]}
    except Exception as e:
        logger.error(f"意图识别故障: {e}")
        return {"route": "business"}  # 兜底扔给业务线


async def planner_node(state: MessagesState):
    _llm = get_llm()
    planner_brain = _llm.with_structured_output(Plan)
    messages = sanitize_history(safe_get(state, "messages", []))

    system_prompt = "你是军师。请将任务拆解为步骤列表 JSON。例如: ['查询逾期', '查手册', '写报告']。"
    try:
        plan_result = await planner_brain.ainvoke([SystemMessage(content=system_prompt)] + messages[-5:])
        return {"plan": plan_result.steps, "messages": [AIMessage(content=f"摸底计划已制定。")]}
    except:
        return {"plan": ["查询数据", "查阅政策", "输出建议"]}


def task_dispatcher_node(state: MessagesState):
    plan = safe_get(state, "plan", [])
    if not plan: return {"messages": [AIMessage(content="✅ 计划执行完毕。")]}
    current_task = plan[0]
    return {
        "plan": plan[1:],
        "messages": [HumanMessage(content=f"🎯 【当前阶段任务】：{current_task}\n请局长立刻派人执行。")]
    }


async def supervisor_node(state: MessagesState):
    """【铁血局长】：抛弃脆弱的 JSON，使用硬核文本匹配"""

    # 🚨 防伪标记：只要新代码跑了，终端一定会打印这句话！
    print("\n\033[91m🚨 [DEBUG] 探长，我是新的纯文本局长！如果您没看到这句话，说明旧代码还在跑！\033[0m")

    _llm = get_llm()
    # 清洗历史，提取状态
    messages = sanitize_history(safe_get(state, "messages", []))
    plan = safe_get(state, "plan", [])

    # 终极降维提示词：不做简答题，只做单选题
    system_prompt = """你是风控局指挥官。请根据最新的【当前阶段任务】指示，决定下一步由谁来干活。
    你【必须且只能】回复以下三个英文单词之一，绝不允许包含任何标点符号或额外解释：
    - sql_detective : 如果任务需要查询数据库、底层业务数据
    - knowledge_agent : 如果任务需要查阅政策、操作手册、处罚标准
    - FINISH : 如果当前阶段任务已经完成，或者无事可做
    """

    try:
        # 彻底抛弃 with_structured_output，直接请求纯文本
        res = await _llm.ainvoke([SystemMessage(content=system_prompt)] + messages[-6:])
        res_text = res.content.strip().lower()

        # 稳如老狗的字符串匹配
        if "sql" in res_text:
            next_action = "sql_detective"
            instruction = "请调取数据库执行核查。"
        elif "knowledge" in res_text:
            next_action = "knowledge_agent"
            instruction = "请翻阅风控操作手册。"
        else:
            next_action = "FINISH"
            instruction = "当前阶段任务结束，准备流转。"

        logger.info(f"👔 [局长决策]：去向 -> {next_action}")
        return {
            "route": next_action,
            "messages": [AIMessage(content=f"👔 【局长指令】：{instruction}", name="supervisor")]
        }

    except Exception as e:
        # 如果是纯文本版本报错，提示语也会变，不会再有 NoneType 报错了
        logger.warning(f"局长发生不可抗力网络故障 ({type(e).__name__}: {e})，启动容错降级...")
        if plan:
            return {"route": "task_dispatcher", "messages": [AIMessage(content="【系统容错】强行推入下一项计划。")]}
        return {"route": "FINISH", "messages": [AIMessage(content="【系统容错】收队。")]}


async def generate_sql_node(state: MessagesState):
    _llm_with_tools = get_llm_with_tools()
    messages = sanitize_history(safe_get(state, "messages", []))
    sys = SystemMessage(content="你是 SQL 探员。查完后直接文本汇报结果。")
    response = await _llm_with_tools.ainvoke([sys] + messages[-8:])
    return {"messages": [response]}


async def knowledge_node(state: MessagesState):
    _llm_with_tools = get_llm_with_tools()
    messages = sanitize_history(safe_get(state, "messages", []))
    sys = SystemMessage(content="你是知识库专家。请通过工具查阅事实，严禁脑补。")
    response = await _llm_with_tools.ainvoke([sys] + messages[-5:])
    return {"messages": [response]}


async def check_data_freshness_node(state: MessagesState):
    return {"messages": [SystemMessage(content="水位：2024-12-23")], "data_freshness": "2024-12-23"}