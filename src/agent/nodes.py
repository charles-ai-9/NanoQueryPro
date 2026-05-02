import logging
import asyncio
from typing import List, Optional, Literal
from functools import lru_cache
import random

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command, Send
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from .state import MessagesState, WorkerState, IntentOutput, SupervisorDecision
from src.core.llm_client import get_llm
from src.tools.sql_tools import execute_sql, search_knowledge_base


logger = logging.getLogger(__name__)


# ================== 1. 结构化协议定义 (集团化核心) ==================

class UserMemory(BaseModel):
    has_preference: bool = Field(description="用户是否表达了个人喜好或特征？")
    preference_content: str = Field(description="提取的特征内容")




# ================== 2. 基础单例加载 ==================

@lru_cache(maxsize=1)
def get_kb_instance():
    from src.core.vector_store import KnowledgeBase
    kb = KnowledgeBase()
    if not kb.load_index(): kb.build_index()
    return kb


@lru_cache(maxsize=1)
def get_llm_with_tools():
    _llm = get_llm()
    if _llm is None: raise ValueError("❌ LLM 初始化失败")
    return _llm.bind_tools([execute_sql, search_knowledge_base])


# ================== 3. 核心节点逻辑 ==================

async def intent_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_name = config.get("configurable", {}).get("user_name", "Jack")
    _llm = get_llm()
    last_msg_content = state.messages[-1].content.strip()
    # --- 🧠 结构化意图大脑 ---
    # 绑定协议：告诉 LLM，你必须返回 IntentOutput 定义的 JSON 格式
    intent_analyzer = _llm.with_structured_output(IntentOutput)

    system_prompt = """你现在是星际金融风控局的指挥中心。
    你的任务是分析用户的输入，并决定由哪个部门接手。

    1. 【PARALLEL】：当用户要求'对比'、'排查这几家'、'看看他们三个'等涉及多个实体的指令时。
       - 关键：你必须精准提取出名单列表存入 targets。
    2. 【ANALYSIS】：询问'为什么'、'原因'、'归因排查'。
    3. 【BUSINESS】：单一的业务咨询或数据查询。
    4. 【META】：问表结构、有哪些表。
    5. 【CHAT】：闲聊、打招呼。
    """

    # 让 LLM 开始思考并提取
    try:
        decision = await intent_analyzer.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_msg_content)
        ])

        logger.info(f"🧠 [局长大脑决策]：意图={decision.route}, 目标={decision.targets}")

        # 如果是闲聊，直接把 LLM 写好的回复包在消息里
        if decision.route == "chat":
            return {
                "route": "chat",
                "messages": [AIMessage(content=decision.chat_reply or "敬礼！探长！")]
            }

        # 核心返回：route 决定去哪，targets 决定分身分裂出多少个
        return {
            "route": decision.route,
            "targets": decision.targets,
            "messages": [
                AIMessage(content=f"📝 指挥部指令：转交【{decision.route.upper()}】部门，涉及目标：{decision.targets}")]
        }

    except Exception as e:
        logger.error(f"决策大脑故障: {e}")
        return {"route": "chat", "messages": [AIMessage(content="报告探长，我刚才走神了，能再说一遍吗？")]}


async def generate_sql_node(state: MessagesState, config: RunnableConfig):
    """【SQL 侦探】：负责精准查数"""
    _llm_with_tools = get_llm_with_tools()
    user_name = config.get("configurable", {}).get("user_name", "探员")
    messages = state.messages

    # 构建纠错指令
    correction = ""
    if messages and "ERROR" in messages[-1].content:
        correction = (
            f"\n[🚩 紧急纠错]：前次执行报错 {messages[-1].content}。\n"
            "请分析原因修正 SQL。如果经过查字典发现数据库里根本没有相关表，请【直接用纯文本回复】向局长说明情况，绝对不要再强行调用工具！"
        )

    sys_instruction = SystemMessage(content=(
        f"你是严谨的金融 SQL 侦探。操作员是 {user_name}。\n"
        "💡 核心侦查指南（绝不盲从）：\n"
        "1. 【先查字典】：绝对不要相信历史对话中别人（甚至是局长）凭空捏造的表名！如果在之前的对话里没有查过表结构，你第一步必须且只能先执行 `SELECT name, sql FROM sqlite_master WHERE type='table';` 来确认真实的表结构。\n"
        "2. 【打卡下班】：一旦你查到了真实数据，或者发现数据库里根本没有能查的业务表，请【直接输出一段自然语言文本】向局长汇报结论（千万不要带有任何 tool_calls）。只要你用纯文本汇报，你的任务就结束了，流程会自动交回给局长。\n"
        "🚨 铁律：严禁反复调用同一条会报错的 SQL 陷入死循环！\n"
        f"{correction}"
    ))

    response = await _llm_with_tools.ainvoke([sys_instruction] + messages[-10:])
    return {"messages": [response]}


# ================== 4. 集团篇：分身术专属节点 ==================

# src/agent/nodes.py
# 🌟 记得在顶部导入 create_react_agent
from langgraph.prebuilt import create_react_agent

## 这是通过 Send API 并发触发的节点!!!!
async def parallel_detective_node(state: dict):
    target = state.get("target", "未知目标")
    logger.info(f"🕵️‍♂️ [分身出勤]: 正在数据库中搜寻 {target} 的真实记录...")

    # =================================================================
    # 🛡️ 架构师级限流：随机错峰 0.5 到 2.5 秒，防止击穿 DashScope QPS，否则会报错的！云端大模型的保护机制
    # =================================================================
    delay = random.uniform(0.5, 2.5)
    logger.info(f"⏳ [错峰限流] 探员 {target} 正在通道排队，等待 {delay:.1f} 秒...")
    await asyncio.sleep(delay)

    _llm = get_llm()
    ## 函数在底层帮你写好了一个 while 循环（也就是一个小型的 LangGraph），让他具备了 ReAct (Reason 推理 + Act 行动) 的能力
    mini_agent = create_react_agent(_llm, tools=[execute_sql])

    # =================================================================
    # 🚨 探长高亮修改区：为探员配发精准的“数据库地图” (Prompt Engineering)
    # =================================================================
    # 🌟 进阶版：赋予探员自主探索能力的通用 Prompt。 但是可能会浪费token,反复推敲试探。
    mission = f"""你是一个高级金融数据侦探，当前的专项排查目标是实体：【{target}】。
        你的任务是利用 execute_sql 工具，在数据库中搜寻关于该目标的所有高风险线索（如信用情况、负债、逾期或交易异常等）。

        🕵️‍♂️ 侦查行动指南：
        1. 【摸底】：如果你不知道当前数据库有哪些表，请先用 SQL 查询系统表（例如 `SELECT name, sql FROM sqlite_master WHERE type='table';`）来了解表结构。
        2. 【搜证】：根据分析得出的表结构，尝试编写 SQL 查找与【{target}】相关的数据。请灵活应对，目标名称的列名可能是 target_name, name, company_name 等。如果一张表查不到，可以尝试其他相关表。
        3. 【结案】：综合你查到的所有真实数据，为探长输出一段专业、客观的风险审计结论。

        🚨 铁律：
        - 绝不允许凭空捏造数据或常识性编造！
        - 如果经过多次 SQL 查询（尝试了不同表和字段）后，确无该目标任何记录，请如实回复：“经全面排查，数据库中未见【{target}】的相关记录。”"""
    # =================================================================
    ## 方便快速测试的简化版 Prompt（直接给表结构，省去摸底环节）。暂时不用的
    mission_for_test = f"""你是一个审计探员，目标是：【{target}】。
    必须使用 execute_sql 工具，查询 `risk_indicators` 表。
    ⚠️ 核心机密（表结构绝对约束）：
    - 目标名称所在的列名叫作 `target_name` （请务必使用 WHERE target_name = '{target}'）
    - 严禁盲目猜测列名（如 name, user_name 等）！
    请查出该目标的 credit_score (信用分), dpd (逾期天数) 和 recent_status (当前状态)。
    如果查不到，如实报告；如果查到了，请根据数据给出一句风险研判。"""
    try:
        # 捕捉微型智能体的内部异常
        result = await mini_agent.ainvoke({"messages": [SystemMessage(content=mission)]})
        final_answer = result["messages"][-1].content
    except Exception as e:
        logger.error(f"❌· 探员 {target} 呼叫总台失败: {e}")
        final_answer = f"由于星际通讯干扰 (API Error)，暂未获取到 {target} 的详细情报。"

    final_report = f"📊 【{target} 真实审计结果】：\n{final_answer}"

    return {
        "parallel_reports": [final_report],
        "messages": [AIMessage(content=f"✅ [{target}] 专线实地考察已结束。")]
    }

def distribute_tasks(state: MessagesState):
    """【分发器】：LangGraph 的分身术发动器"""
    targets = state.targets
    if not targets:
        logger.warning("未检测到具体对比清单，回退至闲聊模式")
        return "chat"

    logger.info(f"🌀 [发动分身术]：目标清单 {targets}")

    # 🌟 灵魂：返回 Send 对象的列表，框架会自动并行执行 parallel_detective_node
    # 分发（Fan-out）：intent 节点通过 Send 把状态分裂成多个小包裹。跟Fan-in/Reduce成对的。 graph.py 文件里面叫parallel_detective
    return [Send("parallel_detective", {"target": t}) for t in targets]


## 因为 MessagesState.parallel_reports 使用了 operator.add，
# 所以当多个分身节点指向同一个 aggregate_reports 节点时，LangGraph 会自动等待所有分身执行完毕，并把他们的报告全部塞进那个列表里，再交给汇总官。
async def aggregate_reports_node(state: MessagesState):
    """
    【集团主编】：负责把所有分身探员交上来的零散报告，整合成一份最终研报。
    """
    reports = state.parallel_reports
    _llm = get_llm()

    if not reports:
        return {"messages": [AIMessage(content="报告探长，分身探员们空手而归，未找到有效信息。")]}

    # 把散落的报告拼起来作为上下文
    combined_context = "\n\n".join(reports)

    system_prompt = """你是一个星际风控局的首席审计官。
    请将以下几份来自不同分身探员的审计片段，整合成一份结构清晰、语气专业的【集团对比审计研报】。
    要求：
    1. 使用 Markdown 表格或清晰的分段。
    2. 突出各目标之间的风险差异。
    3. 最后给出一个总体的风控建议。"""

    response = await _llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"这是各探员汇总回来的原始素材：\n{combined_context}")
    ])

    return {"messages": [response]}


# 2. 添加局长节点
async def supervisor_node(state: MessagesState):
    """【铁血局长】：总览全局，分配任务，审核结果"""
    _llm = get_llm()
    # 强制结构化输出
    supervisor_brain = _llm.with_structured_output(SupervisorDecision)

    system_prompt = """你是星际金融风控局的指挥官。请审阅卷宗并决策：
    1. 需要查底层业务数据（逾期、流水等） -> 指派 sql_detective。
    2. 需要查风控术语、政策、操作手册 -> 指派 knowledge_agent。
    3. 如果情报已足或报错无法解决 -> 选择 FINISH 结案。

    🚨 局长准则：如果探员报错，你必须在 instruction 里给出修正指导，打回重做！"""

    decision = await supervisor_brain.ainvoke([SystemMessage(content=system_prompt)] + state.messages)

    logger.info(f"👔 [局长决策]：去向 -> {decision.next_action}")

    return {
        "route": decision.next_action,
        "messages": [AIMessage(content=f"👔 【局长指令】：{decision.instruction}", name="supervisor")]
    }


# 3. 添加局长传送门
def supervisor_router(state: MessagesState):
    route = getattr(state, "route", "FINISH")
    if route == "FINISH":
        from langgraph.graph import END
        return END
    return route


# 4. 添加知识库探员节点（为了配合局长的点将模式）
async def knowledge_node(state: MessagesState):
    """【知识特工】：负责查阅内部手册，自带防幻觉光环"""
    _llm_with_tools = get_llm_with_tools()
    messages = state.messages

    # =====================================================================
    # 🚨 探长高亮修改区：下达防幻觉死命令
    # =====================================================================
    sys_msg = SystemMessage(content=(
        "你是内部知识库专家。请调用 `search_knowledge_base` 回答局长的疑问。\n"
        "🚨 知识局铁律（违者开除）：\n"
        "1. 你的所有回答必须 100% 来源于检索到的知识库内容。\n"
        "2. 如果知识库中没有查到相关信息，你必须直白地汇报：『报告局长，知识库中未找到相关规则』。\n"
        "3. 绝不允许使用“假设”、“推测”、“大概”等词汇，绝不允许凭空捏造或脑补任何表结构和业务逻辑！"
    ))

    response = await _llm_with_tools.ainvoke([sys_msg] + messages[-5:])
    return {"messages": [response]}



async def supervisor_node(state: MessagesState):
    """【铁血局长】：总览全局，分配任务，审核结果"""
    _llm = get_llm()
    supervisor_brain = _llm.with_structured_output(SupervisorDecision)

    system_prompt = """你是星际金融风控局的指挥官。请审阅卷宗并决策：
    1. 需要查底层业务数据（逾期、流水等） -> 指派 sql_detective。
    2. 需要查风控术语、政策、操作手册 -> 指派 knowledge_agent。
    3. 如果情报已足或报错无法解决 -> 选择 FINISH 结案。

    🚨 局长铁律：
    - 绝不亲自编造数据！如果探员上报查不到，直接向用户说明。
    - 如果探员报错，必须在 instruction 里给出明确的修改意见，打回重做！"""

    try:
        decision = await supervisor_brain.ainvoke([SystemMessage(content=system_prompt)] + state.messages)
        logger.info(f"👔 [局长决策]：去向 -> {decision.next_action} | 指示 -> {decision.instruction}")
        return {
            "route": decision.next_action,
            "messages": [AIMessage(content=f"👔 【局长指令】：{decision.instruction}", name="supervisor")]
        }
    except Exception as e:
        logger.error(f"局长脑部宕机: {e}")
        return {"route": "FINISH", "messages": [AIMessage(content="指挥中心通讯故障，强制结案。")]}

def supervisor_router(state: MessagesState):
    route = getattr(state, "route", "FINISH")
    if route == "FINISH":
        from langgraph.graph import END
        return END
    return route

async def knowledge_node(state: MessagesState):
    """【知识特工】：负责查阅内部手册"""
    _llm_with_tools = get_llm_with_tools()
    messages = state.messages
    sys_msg = SystemMessage(content="你是内部知识库专家。请调用 search_knowledge_base 回答局长的疑问。如果没有工具调用需求，直接回复结论。")
    response = await _llm_with_tools.ainvoke([sys_msg] + messages[-5:])
    return {"messages": [response]}


async def check_data_freshness_node(state: MessagesState):
    """【哨兵】：水位检查"""
    date = "2024-12-23"
    return {"messages": [SystemMessage(content=f"当前数据水位：{date}")], "data_freshness": date}